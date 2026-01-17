# Observability

The FastAPI service emits structured JSON logs, OpenTelemetry traces, and Prometheus metrics.

## Metrics
- `request_count`
- `request_latency_seconds`
- `tokens_generated`
- `error_count`

Run with Docker Compose to scrape metrics with Prometheus:
```bash
docker compose -f docker/compose.yml up --build
```
