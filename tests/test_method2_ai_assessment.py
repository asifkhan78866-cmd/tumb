"""Method 2 AI assessment — labelled honestly, never silently guessing, never calling out in tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import requires_api

pytestmark = requires_api


def _assessment(**overrides):
    from backend.methods.method2.ai_assessment import AIAssessment

    data = {
        "image_is_brain_scan": True,
        "observed_modality": "MRI",
        "image_quality": "good",
        "predicted_class": "meningioma",
        "likelihoods": {"normal": 5, "glioma": 10, "meningioma": 80, "pituitary": 5},
        "key_findings": ["extra-axial dural-based mass"],
        "rationale": "Broad dural attachment.",
    }
    data.update(overrides)
    return AIAssessment(**data)


class FakeClient:
    """Stands in for anthropic.Anthropic; records calls, returns a canned response."""

    def __init__(self, parsed=None, stop_reason="end_turn"):
        self.calls = []
        self._response = SimpleNamespace(
            parsed_output=parsed,
            stop_reason=stop_reason,
            stop_details=SimpleNamespace(category="bio") if stop_reason == "refusal" else None,
            model="claude-opus-5",
            usage=SimpleNamespace(iterations=None),
            _request_id="req_test",
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_suite_never_enables_the_paid_api():
    from backend import config

    assert config.method2_ai_available() is False


def test_request_sends_png_and_structured_schema(png_bytes):
    from backend.methods.method2.ai_assessment import AIAssessment, assess_image

    client = FakeClient(parsed=_assessment())
    result = assess_image(png_bytes, client=client, provider="anthropic")
    call = client.calls[0]
    image = call["messages"][0]["content"][0]
    assert image["source"]["media_type"] == "image/png"
    assert call["output_format"] is AIAssessment
    assert call["model"] == "claude-opus-5"
    assert result.assessment.predicted_class == "meningioma"


def test_refusal_raises_instead_of_returning_a_class(png_bytes):
    from backend.methods.method2.ai_assessment import AIAssessmentError, assess_image

    with pytest.raises(AIAssessmentError, match="declined"):
        assess_image(png_bytes, client=FakeClient(parsed=None, stop_reason="refusal"), provider="anthropic")


# --- OpenRouter ------------------------------------------------------------- #
def _openrouter_client(handler):
    import httpx

    return httpx.Client(transport=httpx.MockTransport(handler))


def _or_payload(content, model="google/gemma-4-31b-it:free"):
    return {"id": "gen-1", "model": model,
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


def test_openrouter_request_uses_free_model_with_fallbacks_and_schema(monkeypatch, png_bytes):
    import json

    import httpx

    from backend.methods.method2.ai_assessment import assess_image

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_or_payload(_assessment().model_dump_json(),
                                                    model=seen["body"]["model"]))

    result = assess_image(png_bytes, client=_openrouter_client(handler), provider="openrouter")
    body = seen["body"]
    assert seen["auth"] == "Bearer sk-or-test"
    assert body["model"].endswith(":free")
    assert body["models"][0] == body["model"] and len(body["models"]) > 1
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.assessment.predicted_class == "meningioma"
    assert result.provider == "openrouter" and result.served_by_fallback is False


def test_openrouter_accepts_fenced_json_and_flags_fallback_model(monkeypatch, png_bytes):
    import httpx

    from backend.methods.method2.ai_assessment import assess_image

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fenced = "```json\n" + _assessment().model_dump_json() + "\n```"
    client = _openrouter_client(
        lambda r: httpx.Response(200, json=_or_payload(fenced, model="google/gemini-2.5-flash-lite"))
    )
    result = assess_image(png_bytes, client=client, provider="openrouter")
    assert result.served_by_fallback is True


@pytest.mark.parametrize("status,needle", [(401, "rejected"), (402, "credits"), (429, "rate limited"), (503, "503")])
def test_openrouter_http_errors_are_readable_and_leak_no_key(monkeypatch, png_bytes, status, needle):
    import httpx

    from backend.methods.method2.ai_assessment import AIAssessmentError, assess_image

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
    client = _openrouter_client(lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(AIAssessmentError) as err:
        assess_image(png_bytes, client=client, provider="openrouter")
    assert needle in str(err.value) and "sk-or-secret-value" not in str(err.value)


def test_openrouter_malformed_answer_is_an_error_not_a_guess(monkeypatch, png_bytes):
    import httpx

    from backend.methods.method2.ai_assessment import AIAssessmentError, assess_image

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = _openrouter_client(lambda r: httpx.Response(200, json=_or_payload("probably glioma")))
    with pytest.raises(AIAssessmentError, match="format"):
        assess_image(png_bytes, client=client, provider="openrouter")


def test_likelihoods_are_normalised_to_100():
    from backend.methods.method2.ai_assessment import normalised_likelihoods

    out = normalised_likelihoods(_assessment(likelihoods={"normal": 1, "glioma": 1, "meningioma": 2, "pituitary": 0}))
    assert out == {"normal": 25.0, "glioma": 25.0, "meningioma": 50.0, "pituitary": 0.0}


def _predict_with(monkeypatch, png_bytes, assessment=None, error=None):
    from backend import config
    from backend.methods.method2 import ai_assessment
    from backend.methods.method2.inference import Method2Engine

    monkeypatch.setattr(config, "method2_ai_available", lambda: True)
    # The AI path only runs while the DCN is untrained; simulate that regardless of disk state.
    from backend.methods.method2 import config as m2

    monkeypatch.setattr(m2, "DCN_WEIGHTS_PATH", config.LOGS_DIR / "__no_such_checkpoint__.pth")

    def fake_assess(_bytes):
        if error:
            raise ai_assessment.AIAssessmentError(error)
        return ai_assessment.AIAssessmentResult(assessment=assessment, model="claude-opus-5", elapsed_s=0.1)

    monkeypatch.setattr(ai_assessment, "assess_image", fake_assess)
    return Method2Engine().predict(png_bytes)


def test_ai_result_is_labelled_and_not_passed_off_as_the_dcn(monkeypatch, png_bytes):
    r = _predict_with(monkeypatch, png_bytes, assessment=_assessment())
    assert r.prediction == "Meningioma"
    assert r.confidence == 80.0
    assert r.model_version == "ai:claude-opus-5"
    assert r.details["result_source"] == "ai_assessment"
    assert r.details["ai_assessment"]["likelihoods_are_calibrated"] is False
    assert any("NOT from this method's trained MRI–SPECT fusion network" in w for w in r.warnings)
    assert any("No MRI–SPECT fusion was performed" in w for w in r.warnings)


def test_indeterminate_ai_answer_reports_no_class(monkeypatch, png_bytes):
    r = _predict_with(monkeypatch, png_bytes, assessment=_assessment(predicted_class="indeterminate"))
    assert r.prediction is None and r.confidence is None and r.probabilities == {}
    assert r.details["result_source"] == "ai_assessment"


def test_api_failure_becomes_a_warning_not_a_guess(monkeypatch, png_bytes):
    r = _predict_with(monkeypatch, png_bytes, error="AI assessment failed: could not reach the API.")
    assert r.prediction is None
    assert r.details["result_source"] == "none"
    assert "AI assessment failed: could not reach the API." in r.warnings


def test_ai_assessment_never_writes_metrics(monkeypatch, png_bytes, tmp_path):
    from backend import config

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    _predict_with(monkeypatch, png_bytes, assessment=_assessment())
    assert not any(tmp_path.iterdir()), "the AI assessment must not write any metrics file"
