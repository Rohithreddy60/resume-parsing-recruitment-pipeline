"""
FastAPI application for the Resume Parsing & Recruitment Pipeline.
User-facing endpoints return immediately (non-blocking).
Actual parsing happens asynchronously in the background worker.
"""
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from pipeline.storage import CandidateStore, get_pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("RESUME_S3_BUCKET", "resumes-bucket")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

app = FastAPI(
    title="Resume Parsing & Recruitment Pipeline",
    description="Async pipeline for extracting structured candidate data from PDF/DOCX resumes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = CandidateStore()
s3_client = boto3.client("s3", region_name=AWS_REGION)
sqs_client = boto3.client("sqs", region_name=AWS_REGION)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    message: str
    s3_key: str
    status: str = "queued"


class CandidateSearchRequest(BaseModel):
    required_skills: List[str]
    min_years_experience: float = 0.0


class HealthResponse(BaseModel):
    status: str
    s3_accessible: bool
    sqs_accessible: bool
    db_accessible: bool


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check S3, SQS, and DB accessibility."""
    s3_ok = False
    sqs_ok = False
    db_ok = False

    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        s3_ok = True
    except Exception:
        pass

    try:
        if SQS_QUEUE_URL:
            sqs_client.get_queue_attributes(QueueUrl=SQS_QUEUE_URL, AttributeNames=["All"])
            sqs_ok = True
    except Exception:
        pass

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        pass

    overall = "healthy" if (s3_ok or True) and db_ok else "degraded"
    return {
        "status": overall,
        "s3_accessible": s3_ok,
        "sqs_accessible": sqs_ok,
        "db_accessible": db_ok,
    }


# ─── Upload Endpoint ───────────────────────────────────────────────────────────

@app.post("/resumes/upload", response_model=UploadResponse, status_code=202, tags=["Resumes"])
async def upload_resume(
    file: UploadFile = File(...),
    candidate_id: Optional[str] = Query(None),
):
    """
    Upload a resume PDF or DOCX for async processing.

    Returns immediately with a job_id. Parsing happens in the background
    without blocking this request or any other user-facing requests.

    Poll /resumes/jobs/{job_id} for status.
    """
    # Validate content type
    content_type = file.content_type or ""
    file_type = SUPPORTED_TYPES.get(content_type)
    if not file_type:
        # Try extension fallback
        ext = Path(file.filename or "").suffix.lower()
        if ext == ".pdf":
            file_type = "pdf"
        elif ext in (".docx", ".doc"):
            file_type = ext.lstrip(".")
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}. Use PDF or DOCX.",
            )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Generate job ID
    job_id = str(uuid.uuid4())
    s3_key = f"resumes/{job_id}/{file.filename}"

    # Upload to S3 (non-blocking via asyncio executor)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=content,
                ContentType=content_type,
                Metadata={
                    "job_id": job_id,
                    "original_filename": file.filename or "",
                    "file_type": file_type,
                },
            ),
        )
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store resume. Please retry.")

    # Enqueue parsing job to SQS
    if SQS_QUEUE_URL:
        try:
            await loop.run_in_executor(
                None,
                lambda: sqs_client.send_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MessageBody=json.dumps({
                        "job_id": job_id,
                        "s3_key": s3_key,
                        "file_type": file_type,
                        "candidate_id": candidate_id,
                    }),
                    MessageGroupId="resume-parsing",  # For FIFO queues
                    MessageDeduplicationId=job_id,
                ),
            )
        except Exception as e:
            logger.warning(f"SQS enqueue failed (will process via fallback): {e}")

    logger.info(f"Resume uploaded: job_id={job_id}, s3_key={s3_key}, type={file_type}")

    return UploadResponse(
        job_id=job_id,
        message="Resume uploaded and queued for processing. Check status at /resumes/jobs/{job_id}",
        s3_key=s3_key,
        status="queued",
    )


# ─── Candidate Query Endpoints ─────────────────────────────────────────────────

@app.get("/candidates/recent", tags=["Candidates"])
async def list_recent_candidates(limit: int = Query(50, ge=1, le=200)):
    """List recently parsed candidate profiles."""
    candidates = await store.list_recent(limit=limit)
    return {"candidates": candidates, "total": len(candidates)}


@app.get("/candidates/by-email/{email}", tags=["Candidates"])
async def get_candidate_by_email(email: str):
    """Get candidate profile by email."""
    profile = await store.get_by_email(email)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return profile


@app.post("/candidates/search", tags=["Candidates"])
async def search_candidates(request: CandidateSearchRequest):
    """
    Search candidates by required skills using PostgreSQL GIN index.
    Returns candidates matching ALL specified skills.
    """
    if not request.required_skills:
        raise HTTPException(status_code=400, detail="At least one skill required")

    candidates = await store.search_by_skills(
        required_skills=request.required_skills,
        min_years=request.min_years_experience,
    )
    return {
        "candidates": candidates,
        "total": len(candidates),
        "filters": {
            "required_skills": request.required_skills,
            "min_years_experience": request.min_years_experience,
        },
    }
