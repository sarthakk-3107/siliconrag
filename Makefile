.PHONY: install ingest embed retrieve agent evaluate api app docker clean

install:
	pip install -r requirements.txt

ingest:
	python src/ingest.py

embed:
	python src/embed.py

retrieve:
	python src/retrieve.py

agent:
	python src/agent.py

evaluate:
	python src/evaluate.py

api:
	cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app.py

docker-build:
	docker build -t siliconrag .

docker-run:
	docker run -p 8000:8000 --env-file .env siliconrag

clean:
	rm -rf data/chroma_db data/processed/*.json eval/results/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
