# Roadmap (Now / Next / Later)

## NOW (Sprint)

### 1) Docker + CI/CD
**Goal:** Reproducible builds and tests on every PR.
**Tasks:**
- Add `Dockerfile` and `docker-compose.yml` (API + optional UI).
- GitHub Actions: lint (ruff/black), pytest, build image (GHCR).
- `Makefile` targets: `make test | build | run`.
**DoD:**
- CI badge is visible and green in README.
- `docker compose up` starts the app locally.

### 2) FastAPI inference service
**Goal:** Standalone prediction service decoupled from UI.
**Tasks:**
- Endpoints: `GET /health`, `POST /predict` (multipart image or URL).
- Pydantic validation: content type/MIME, size ≤ 3 MB; reject unsupported inputs.
- Structured logging (request-id) and request/response timing.
**DoD:**
- README includes a working `curl` example.
- End-to-end test (pytest) passes locally and in CI.

### 3) MLflow experiments (local tracking minimum)
**Goal:** Sustainable, comparable experiments.
**Tasks:**
- Log params/metrics/artifacts during training.
- Run at least 3 distinct experiments and persist metrics.
**DoD:**
- Screenshot of MLflow UI added to README.
- Best configuration summarized in the Model Card.

---

## NEXT

### 4) Data validation (Great Expectations)
**DoD:** GE suite with ≥6 expectations (schema, ranges, nulls, image dimensions). Failing inputs break the pipeline in tests; validation report committed.

### 5) Quality monitoring + drift
**DoD:** Rolling quality metrics (e.g., Accuracy@threshold) plus drift detection (PSI/KL on basic descriptors/embeddings) with thresholds and an alert (cron/log). Report (PNG/CSV) committed.

### 6) Load testing
**DoD:** Locust/k6 scenario committed; p95/p99 latency reported in README; SLO defined (e.g., p95 < 200 ms) and a Runbook describing actions on breach.

### 7) Model Registry (MLflow) + promotion
**DoD:** Model promoted `Staging → Production` based on clear criteria; rollback procedure documented and tested.

---

## LATER

### 8) Cloud (S3 + 1× remote train)
**DoD:** Artifacts stored in a versioned S3 bucket; at least one remote training run (EC2/ECS or SageMaker); **Cost table** ($/run, storage) added to README.

### 9) Orchestration (Prefect/Airflow)
**DoD:** Scheduled flow `ingest → train → evaluate → register` (e.g., weekly). Screenshot of the orchestration UI committed.

### 10) Security & safety polish
**DoD:** Reject unsupported MIME types; enforce file-size limits and timeouts; add simple rate-limit if public; run `bandit` in CI.

---

## Issues & Labels
Create a separate **Issue** for each item (include the DoD checklist). Use labels:
- `now`, `next`, `later`, `good-first-issue`, `infra`, `mlops`, `api`, `docs`
