"""Two independent brain-tumor methods, described by a shared registry.

``backend.methods.registry`` is the only place that knows which methods exist.
Importing this package pulls in nothing heavy — the registry is static data and
each method's engine is imported on demand.
"""
from backend.methods.registry import (  # noqa: F401
    METHOD_IDS,
    METHODS,
    MethodSpec,
    get_method,
    list_methods,
    load_engine,
    runtime_status,
)

__all__ = [
    "METHODS",
    "METHOD_IDS",
    "MethodSpec",
    "get_method",
    "list_methods",
    "load_engine",
    "runtime_status",
]
