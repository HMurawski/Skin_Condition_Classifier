# Makefile
.PHONY: install lint test run build up down

install:
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

test:
	python -m pytest -q

run:
	streamlit run app.py

build:
	docker build -t skinclf:dev .

up:
	docker compose up --build

down:
	docker compose down
