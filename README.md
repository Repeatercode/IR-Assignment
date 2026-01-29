# Information Retrieval Coursework

This project implements a vertical search engine for Coventry University ICS Research Centre publications and a document classification component for news topics.

## Requirements
- Python 3.13 (venv provided in `./venv`)
- Dependencies in `requirements.txt`

Install dependencies:

```sh
./venv/bin/pip install -r requirements.txt
```

## Search Engine (Task 1)

### Crawl and build the index

```sh
./venv/bin/python -m search_engine.crawler \
  --seed "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo/" \
  --max-pages 0 \
  --delay 1.2
```

Output files:
- `data/publications.jsonl` (raw publications)
- `data/index.json` (inverted index + metadata)

### CLI search

```sh
./venv/bin/python -m search_engine.cli_search --q "machine learning" --top 10
```

### Web UI

```sh
./venv/bin/python manage.py runserver 8000
```

Open `http://127.0.0.1:8000`.

## Classification (Task 2)

### Collect dataset (RSS)

```sh
./venv/bin/python -m classifier.rss_collect --per-class 40
```

### Train classification model (TF-IDF + ComplementNB, balanced)

```sh
./venv/bin/python -m classifier.train
```

### Predict class (CLI)

```sh
./venv/bin/python -m classifier.predict --text "Healthcare costs are rising in many countries"
```

The web UI has a Classification page that assigns the class as well.

## Scheduling

Weekly crawl scripts:
- `scripts/run_weekly.sh` (bash)
- `scripts/run_weekly.bat` (Windows)

## Configuration

Crawler defaults are in `search_engine/config.py` (user-agent, delay, max pages).

