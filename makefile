black:
	black training_pipeline/ app/

.PHONY: app
app:
	(cd app && uvicorn main:app --host 0.0.0.0 --port 8080)

build:
	docker build -t mlops-app:0.1.0 .

run:
	docker compose up
