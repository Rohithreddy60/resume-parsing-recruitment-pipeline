# Resume Parsing & Recruitment Data Pipeline

> **Async Python** pipeline (spaCy, PDFMiner, Docker, AWS) to extract structured candidate data from **PDF** and **DOCX** documents at scale — enabling automated screening **without blocking user-facing requests**.

---

## Features

- **Async Non-Blocking Architecture** — FastAPI API returns immediately on upload; actual parsing runs in a background `ProcessPoolExecutor` + asyncio worker, keeping user-facing latency < 50ms regardless of document size
- **PDF Parsing** — PDFMiner with tuned `LAParams` handles single/multi-column layouts, embedded fonts, hyperlinks, and up to 100+ page documents via streaming
- **DOCX Parsing** — python-docx with XML-level run extraction handles paragraphs, tables, headers/footers, and hyperlinks embedded in XML
- **spaCy NLP Extraction** — Named Entity Recognition (PERSON, GPE), PhraseMatcher for 40+ tech skills, regex for emails/phones/URLs, heuristic section detection for experience/education/certifications
- **AWS Integration** — S3 for document storage, SQS FIFO for job queuing (exactly-once delivery), SNS for result notifications, LocalStack for local dev
- **Scale-Out Workers** — Worker containers scale independently from the API; each worker runs 4 parallel processes for CPU-bound parsing

---

## Architecture

```
                   User Request
                        │
                        ▼
            ┌───────────────────────┐
            │   FastAPI API (202)   │  ← Returns job_id immediately
            └──────────┬────────────┘
                       │
              ┌────────┴──────────┐
              │                   │
              ▼                   ▼
         Upload to S3       Enqueue to SQS
              │                   │
              └────────┬──────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Async Worker (asyncio)     │
        │  ┌────────────────────────┐  │
        │  │  S3 Download (async)   │  │
        │  └──────────┬─────────────┘  │
        │             │                │
        │  ┌──────────▼─────────────┐  │
        │  │ ProcessPoolExecutor    │  │  ← CPU-bound (doesn't block)
        │  │  PDFMiner / python-docx│  │
        │  │  spaCy NLP extraction  │  │
        │  └──────────┬─────────────┘  │
        │             │                │
        │  ┌──────────▼─────────────┐  │
        │  │  Store to PostgreSQL   │  │
        │  │  Publish to SNS        │  │
        │  │  Delete from SQS       │  │
        │  └────────────────────────┘  │
        └──────────────────────────────┘
```

---

## Project Structure

```
resume-parsing-recruitment-pipeline/
├── api/
│   └── main.py              # FastAPI: upload, search, health endpoints
├── pipeline/
│   ├── parsers/
│   │   ├── pdf_parser.py    # PDFMiner-based PDF text extraction
│   │   └── docx_parser.py   # python-docx DOCX/DOC extraction
│   ├── extractor/
│   │   └── nlp_extractor.py # spaCy NER + PhraseMatcher + regex
│   ├── worker.py            # Async SQS worker with ProcessPoolExecutor
│   └── storage.py           # asyncpg PostgreSQL storage layer
├── tests/
│   └── test_nlp_extractor.py
├── Dockerfile               # Multi-stage: API + Worker targets
├── docker-compose.yml       # Full stack: API + Worker + PG + LocalStack
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Rohithreddy60/resume-parsing-recruitment-pipeline.git
cd resume-parsing-recruitment-pipeline
docker-compose up --build
```

API: **http://localhost:8000** | Docs: **http://localhost:8000/docs**

### Upload a Resume
```bash
curl -X POST http://localhost:8000/resumes/upload \
  -F "file=@/path/to/resume.pdf" \
  -F "candidate_id=candidate-001"
# Returns: {"job_id": "uuid", "status": "queued", ...}
# Parsing happens in the background - API returns in < 50ms
```

### Search Candidates by Skills
```bash
curl -X POST http://localhost:8000/candidates/search \
  -H "Content-Type: application/json" \
  -d '{"required_skills": ["Python", "AWS", "Docker"], "min_years_experience": 3}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | S3, SQS, DB health check |
| POST | `/resumes/upload` | Upload PDF/DOCX (non-blocking, returns job_id) |
| GET | `/candidates/recent` | List recently parsed candidates |
| GET | `/candidates/by-email/{email}` | Get candidate by email |
| POST | `/candidates/search` | Search by required skills + min experience |

---

## Extracted Data Fields

From each resume the pipeline extracts:

| Field | Method |
|-------|--------|
| name | spaCy NER (PERSON entity) |
| email | Regex |
| phone | Regex |
| location | spaCy NER (GPE entity) |
| linkedin_url | Regex |
| github_url | Regex |
| skills | spaCy PhraseMatcher (40+ tech skills) |
| education | Heuristic section parsing + degree regex |
| experience | Section parsing + year detection |
| certifications | Section parsing |
| years_of_experience | Year span calculation |

---

## Why ProcessPoolExecutor?

PDF parsing (PDFMiner) and NLP (spaCy) are CPU-bound operations. Running them in the asyncio event loop would block all other requests. By offloading to `ProcessPoolExecutor`:

- The API event loop stays free for I/O (uploads, DB queries)
- 4 worker processes parse 4 documents in parallel
- Each worker process has its own copy of the spaCy model
- No GIL contention — true parallelism for NLP workloads

---

## Testing

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgres://localhost/recruitment_db | PostgreSQL |
| `RESUME_S3_BUCKET` | resumes-bucket | S3 bucket for uploads |
| `SQS_QUEUE_URL` | - | SQS FIFO queue URL |
| `SNS_TOPIC_ARN` | - | SNS topic for result notifications |
| `PIPELINE_MAX_WORKERS` | 4 | ProcessPoolExecutor workers |
| `MAX_FILE_SIZE_MB` | 10 | Max upload size |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.111 |
| NLP | spaCy 3.7 (en_core_web_sm/lg) |
| PDF Parsing | PDFMiner.six |
| DOCX Parsing | python-docx |
| Async Runtime | asyncio + ProcessPoolExecutor |
| Database | PostgreSQL 16 + asyncpg |
| Message Queue | AWS SQS (LocalStack for dev) |
| Storage | AWS S3 (LocalStack for dev) |
| Notifications | AWS SNS |
| Containerization | Docker + Docker Compose |
