from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint", "status"])
ERROR_COUNT = Counter("error_count", "Total error responses", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
TOKENS_GENERATED = Counter("tokens_generated", "Approx tokens generated", ["endpoint"])
