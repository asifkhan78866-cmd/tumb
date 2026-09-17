"""Method 2 AI assessment — a vision-language model's reading of an uploaded scan.

Used only while Method 2's DCN has no trained checkpoint, and only when
``METHOD2_AI_ENABLED=true`` with a key for ``METHOD2_AI_PROVIDER``:

* ``openrouter`` (default) — ``OPENROUTER_API_KEY``; a free vision model with
  OpenRouter-side fallbacks (``METHOD2_AI_FALLBACK_MODELS``).
* ``anthropic`` — ``ANTHROPIC_API_KEY``; Claude via the official SDK.

What this is and is not:

* It **is** a general-purpose AI model's assessment of one image, returned in a
  fixed schema over Method 2's class list.
* It is **not** the Dense Convolutional Network, was not trained on this
  project's data, has no measured accuracy, and its per-class likelihoods are
  the model's own uncalibrated estimates — not softmax probabilities. Callers
  attach warnings saying so, and nothing here writes to the metrics store.

API keys are read from the environment and never appear in a response, a log
line, an error message or a report.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from backend import config

__all__ = ["AIAssessment", "AIAssessmentResult", "AIAssessmentError", "assess_image"]

# Longest edge sent to the API; larger images are downscaled before upload.
_MAX_EDGE = 1568
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLASS_KEYS = ("normal", "glioma", "meningioma", "pituitary")


class ClassLikelihoods(BaseModel):
    normal: int
    glioma: int
    meningioma: int
    pituitary: int


class AIAssessment(BaseModel):
    """The structured answer the model must return."""

    image_is_brain_scan: bool
    observed_modality: Literal["MRI", "CT", "SPECT", "PET", "other", "unclear"]
    image_quality: Literal["good", "limited", "poor"]
    predicted_class: Literal["normal", "glioma", "meningioma", "pituitary", "indeterminate"]
    likelihoods: ClassLikelihoods
    key_findings: list[str]
    rationale: str


# Hand-written (no $ref/$defs) so providers with partial JSON-schema support accept it.
ASSESSMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "image_is_brain_scan": {"type": "boolean"},
        "observed_modality": {"type": "string", "enum": ["MRI", "CT", "SPECT", "PET", "other", "unclear"]},
        "image_quality": {"type": "string", "enum": ["good", "limited", "poor"]},
        "predicted_class": {"type": "string", "enum": [*CLASS_KEYS, "indeterminate"]},
        "likelihoods": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in CLASS_KEYS},
            "required": list(CLASS_KEYS),
            "additionalProperties": False,
        },
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "image_is_brain_scan", "observed_modality", "image_quality",
        "predicted_class", "likelihoods", "key_findings", "rationale",
    ],
    "additionalProperties": False,
}


@dataclass
class AIAssessmentResult:
    assessment: Optional[AIAssessment]
    model: str
    elapsed_s: float
    provider: str = ""
    request_id: Optional[str] = None
    served_by_fallback: bool = False
    warnings: list[str] = field(default_factory=list)


class AIAssessmentError(RuntimeError):
    """The assessment could not be obtained (network, auth, refusal, bad output)."""


SYSTEM_PROMPT = """You review a single brain scan image for a research demo of a \
brain-tumour classification pipeline. Your output is shown to the user as an AI \
model's assessment, clearly marked as unvalidated and not a diagnosis.

Classify the image into exactly one of: normal (no visible tumour), glioma, \
meningioma, pituitary (pituitary-region tumour). Use "indeterminate" when the image \
is not a brain scan, is too poor to read, or the evidence does not support one class.

Give an integer likelihood from 0 to 100 for each of the four classes, summing to \
100, reflecting your genuine uncertainty. Report the imaging modality you actually \
observe, even if it is not SPECT. List the concrete visual findings you relied on \
(location, margins, enhancement pattern, mass effect, anatomical relationships) and \
explain briefly how they support your choice. Do not overstate certainty."""

JSON_INSTRUCTION = (
    "Respond with only a JSON object with exactly these keys: image_is_brain_scan "
    "(boolean), observed_modality (MRI|CT|SPECT|PET|other|unclear), image_quality "
    "(good|limited|poor), predicted_class (normal|glioma|meningioma|pituitary|"
    "indeterminate), likelihoods (object with integer normal, glioma, meningioma, "
    "pituitary), key_findings (array of strings), rationale (string)."
)


def _encode_png(image_bytes: bytes) -> str:
    """Decode any supported upload, downscale, and return base64 PNG."""
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = _MAX_EDGE / max(h, w)
    if scale < 1.0:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Could not re-encode the image as PNG.")
    return base64.standard_b64encode(png.tobytes()).decode("ascii")


def assess_image(image_bytes: bytes, client=None, provider: Optional[str] = None) -> AIAssessmentResult:
    """Get a structured assessment of one image from the configured provider.

    Raises :class:`ValueError` for an undecodable image and
    :class:`AIAssessmentError` for every API-side failure, with a message safe
    to show a user (no credential material).
    """
    data = _encode_png(image_bytes)
    provider = (provider or config.METHOD2_AI_PROVIDER).lower()
    if provider == "openrouter":
        return _assess_openrouter(data, client)
    if provider == "anthropic":
        return _assess_anthropic(data, client)
    raise AIAssessmentError(f"Unknown METHOD2_AI_PROVIDER {provider!r}; use openrouter or anthropic.")


# --------------------------------------------------------------------------- #
# OpenRouter (OpenAI-compatible chat completions over HTTP)
# --------------------------------------------------------------------------- #
def _parse_json_text(text: str) -> AIAssessment:
    """Validate the model's JSON, tolerating a surrounding markdown code fence."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    elif not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return AIAssessment.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise AIAssessmentError(
            "AI assessment returned an answer that did not match the required format."
        ) from exc


def _assess_openrouter(data: str, client=None) -> AIAssessmentResult:
    import httpx

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise AIAssessmentError("AI assessment unavailable: OPENROUTER_API_KEY is not set.")
    primary = config.METHOD2_AI_MODEL
    models = [primary] + [m for m in config.METHOD2_AI_FALLBACK_MODELS if m != primary]
    body = {
        "model": primary,
        "models": models,  # OpenRouter falls through these in order on errors / rate limits
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Assess this scan. " + JSON_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "brain_scan_assessment", "strict": True, "schema": ASSESSMENT_JSON_SCHEMA},
        },
        "temperature": 0,
        # Room for reasoning models to think before the JSON; 2000 truncated in practice.
        "max_tokens": 8000,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "NeuroSeg AI",
    }

    own_client = client is None
    client = client or httpx.Client(timeout=config.METHOD2_AI_TIMEOUT_S)
    started = time.perf_counter()
    try:
        response = client.post(OPENROUTER_URL, json=body, headers=headers)
    except httpx.TimeoutException as exc:
        raise AIAssessmentError("AI assessment failed: the request timed out.") from exc
    except httpx.HTTPError as exc:
        raise AIAssessmentError("AI assessment failed: could not reach OpenRouter.") from exc
    finally:
        if own_client:
            client.close()
    elapsed = time.perf_counter() - started

    status = response.status_code
    if status == 401:
        raise AIAssessmentError("AI assessment failed: the OPENROUTER_API_KEY was rejected.")
    if status == 402:
        raise AIAssessmentError(
            "AI assessment failed: OpenRouter reports insufficient credits (a paid model is "
            "in METHOD2_AI_FALLBACK_MODELS, or every free model was unavailable)."
        )
    if status == 429:
        raise AIAssessmentError(
            "AI assessment failed: rate limited (free models have daily limits). Try again later."
        )
    if status >= 400:
        raise AIAssessmentError(f"AI assessment failed: OpenRouter error {status}.")

    try:
        payload = response.json()
        choice = payload["choices"][0]
        message = choice.get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AIAssessmentError("AI assessment failed: unexpected response from OpenRouter.") from exc
    if payload.get("error"):
        raise AIAssessmentError("AI assessment failed: the model provider returned an error.")
    if message.get("refusal"):
        raise AIAssessmentError("AI assessment declined by the model.")
    if choice.get("finish_reason") == "length":
        raise AIAssessmentError("AI assessment returned no usable structured answer (truncated).")
    content = message.get("content")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not content:
        raise AIAssessmentError("AI assessment returned an empty answer.")

    served_model = str(payload.get("model") or primary)
    return AIAssessmentResult(
        assessment=_parse_json_text(content),
        model=served_model,
        elapsed_s=round(elapsed, 3),
        provider="openrouter",
        request_id=payload.get("id"),
        served_by_fallback=served_model.split(":")[0] != primary.split(":")[0],
    )


# --------------------------------------------------------------------------- #
# Anthropic (official SDK)
# --------------------------------------------------------------------------- #
def _assess_anthropic(data: str, client=None) -> AIAssessmentResult:
    import anthropic

    client = client or anthropic.Anthropic(timeout=config.METHOD2_AI_TIMEOUT_S, max_retries=2)
    model = config.METHOD2_AI_MODEL if config.METHOD2_AI_PROVIDER == "anthropic" else "claude-opus-5"
    started = time.perf_counter()
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": "Assess this scan."},
                ],
            }],
            output_format=AIAssessment,
        )
    except anthropic.AuthenticationError as exc:
        raise AIAssessmentError("AI assessment failed: the ANTHROPIC_API_KEY was rejected.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise AIAssessmentError("AI assessment failed: the API key lacks permission for this model.") from exc
    except anthropic.NotFoundError as exc:
        raise AIAssessmentError(f"AI assessment failed: model {model!r} was not found.") from exc
    except anthropic.RateLimitError as exc:
        raise AIAssessmentError("AI assessment failed: rate limited by the API. Try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise AIAssessmentError(f"AI assessment failed: API error {exc.status_code}.") from exc
    except anthropic.APIConnectionError as exc:
        raise AIAssessmentError("AI assessment failed: could not reach the API.") from exc
    elapsed = time.perf_counter() - started

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise AIAssessmentError(
            f"AI assessment declined by the model (category: {category or 'unspecified'})."
        )
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise AIAssessmentError("AI assessment returned no usable structured answer.")
    served_by_fallback = any(
        getattr(entry, "type", None) == "fallback_message"
        for entry in (getattr(response.usage, "iterations", None) or [])
    )
    return AIAssessmentResult(
        assessment=response.parsed_output,
        model=getattr(response, "model", model),
        elapsed_s=round(elapsed, 3),
        provider="anthropic",
        request_id=getattr(response, "_request_id", None),
        served_by_fallback=served_by_fallback,
    )


def run_ai_assessment(
    image_bytes: bytes,
    *,
    not_from: str,
    reason: str,
    allowed_modalities: tuple[str, ...] = ("MRI",),
) -> tuple[Optional[dict], list[str]]:
    """Assess an image and describe the result honestly for a method's response.

    ``not_from`` names the trained model this result is *not* from; ``reason``
    says why the AI is answering instead. Failures become warnings, never a
    guessed class. The disclaimer is always the first warning.
    """
    try:
        # Looked up on the module at call time so tests can substitute it.
        res = globals()["assess_image"](image_bytes)
    except AIAssessmentError as exc:
        return None, [str(exc)]
    a = res.assessment
    details = {
        "provider": res.provider,
        "model": res.model,
        "served_by_fallback_model": res.served_by_fallback,
        "request_id": res.request_id,
        "elapsed_s": res.elapsed_s,
        "reason": reason,
        "predicted_class": a.predicted_class,
        "likelihoods": normalised_likelihoods(a),
        "likelihoods_are_calibrated": False,
        "image_is_brain_scan": a.image_is_brain_scan,
        "observed_modality": a.observed_modality,
        "image_quality": a.image_quality,
        "key_findings": list(a.key_findings),
        "rationale": a.rationale,
    }
    warnings = [
        f"AI ASSESSMENT: this result comes from a general-purpose AI model ({res.model}) "
        f"reading the image, NOT from {not_from}. It has no measured accuracy on this "
        f"project's data, its percentages are the model's own uncalibrated estimates, and "
        f"it is not a diagnosis.",
    ]
    if not a.image_is_brain_scan:
        warnings.append("The AI model judged that this image is not a brain scan.")
    if a.observed_modality not in allowed_modalities:
        warnings.append(
            f"The AI model observed {a.observed_modality} imaging; this method is specified "
            f"for {' and '.join(allowed_modalities)}."
        )
    if a.image_quality != "good":
        warnings.append(f"The AI model rated image quality as {a.image_quality}.")
    return details, warnings


def normalised_likelihoods(assessment: AIAssessment) -> dict[str, float]:
    """Model-reported likelihoods rescaled to sum to 100 (they are still uncalibrated)."""
    raw = {k: max(0, int(v)) for k, v in assessment.likelihoods.model_dump().items()}
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: round(100.0 * v / total, 2) for k, v in raw.items()}
