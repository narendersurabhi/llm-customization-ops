from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from prometheus_client import generate_latest

from llm_customization_ops.ops.health import ready_status
from llm_customization_ops.ops.logging import get_logger
from llm_customization_ops.serving.metrics import (
    ERROR_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOKENS_GENERATED,
)
from llm_customization_ops.serving.model_loader import LoadedModel
from llm_customization_ops.serving.schemas import (
    ExtractRequest,
    ExtractResponse,
    GenerateRequest,
    GenerateResponse,
)
from llm_customization_ops.training.registry import get_template

router = APIRouter()
logger = get_logger(service="api")


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    loaded: LoadedModel = request.app.state.loaded_model
    status = ready_status(loaded.model is not None)
    return {"status": status.message}


@router.get("/metrics")
async def metrics() -> Any:
    return Response(content=generate_latest(), media_type="text/plain")


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, payload: GenerateRequest) -> GenerateResponse:
    start = time.time()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    log = logger.bind(request_id=request_id)
    loaded: LoadedModel = request.app.state.loaded_model
    template = get_template(request.app.state.registry_path, payload.template_id)
    prompt = template.template.format(text=payload.prompt)
    try:
        if loaded.model is None:
            text = f"[fake] {prompt}"
        else:
            inputs = loaded.tokenizer(prompt, return_tensors="pt")
            outputs = loaded.model.generate(
                **inputs,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
            )
            text = loaded.tokenizer.decode(outputs[0], skip_special_tokens=True)
        TOKENS_GENERATED.labels(endpoint="generate").inc(len(text.split()))
        REQUEST_COUNT.labels(endpoint="generate", status="200").inc()
        return GenerateResponse(text=text, model=str(request.app.state.base_model))
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="generate").inc()
        REQUEST_COUNT.labels(endpoint="generate", status="500").inc()
        log.error("generate_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="generation failed") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint="generate").observe(time.time() - start)


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(request: Request, payload: ExtractRequest) -> ExtractResponse:
    start = time.time()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    log = logger.bind(request_id=request_id)
    loaded: LoadedModel = request.app.state.loaded_model
    template = get_template(request.app.state.registry_path, payload.template_id)
    prompt = template.template.format(text=payload.text)
    try:
        if loaded.model is None:
            output = "{}"
        else:
            inputs = loaded.tokenizer(prompt, return_tensors="pt")
            outputs = loaded.model.generate(
                **inputs,
                max_new_tokens=payload.max_new_tokens,
                temperature=0.1,
            )
            output = loaded.tokenizer.decode(outputs[0], skip_special_tokens=True)
        REQUEST_COUNT.labels(endpoint="extract", status="200").inc()
        data = json.loads(output) if output.strip().startswith("{") else {"raw": output}
        return ExtractResponse(data=data)
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="extract").inc()
        REQUEST_COUNT.labels(endpoint="extract", status="500").inc()
        log.error("extract_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="extract failed") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint="extract").observe(time.time() - start)
