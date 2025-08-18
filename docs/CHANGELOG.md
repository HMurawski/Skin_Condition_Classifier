# Changelog
All notable changes to this project will be documented in this file.
Format follows Keep a Changelog, versioning via SemVer.

## v0.1.0 - 2025-08-18
### Added
- **Dockerization (first step):** `Dockerfile` and `docker-compose.yml` to run the app locally.
- **CI/CD (GitHub Actions):** `lint-and-test` workflow (Ruff + pytest) and a Docker image build workflow.
- **Makefile:** targets `fmt`, `lint`, `test`, `ci`, `build`, `run` for smoother local development.


### Changed
- Lint fixes aligned with Ruff (split imports, remove unused imports, clearer variable names).
- `evaluate_with_threshold`: explicitly pass `labels=range(len(classes))` to `classification_report` and `confusion_matrix` so reports stay stable regardless of threshold.
- Tests: `tests/test_evaluate_threshold.py` now imports the function from `src.evaluate` (no redefinition) and uses `pytest.approx` for float comparisons.

### Documentation
- Updated README with container run instructions (`docker compose up`) and CI status badge.

## [0.0.0] - 2025-08-16
### Added
- **Research MVP** of Skin Condition Classifier (not a medical device).
- Streamlit UI for single-image inference (live demo).
- Trained transfer-learning model + saved artifacts (model weights + class index mapping).
- Preprocessing & inference pipeline, basic evaluation metrics and README with usage.
- MIT License, `.env.example`, and project structure under `src/`.

### Known limitations
- No standalone inference API (UI-only).
- No container image / CI pipeline yet.


