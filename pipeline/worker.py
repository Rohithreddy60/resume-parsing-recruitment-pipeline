"""
Async pipeline worker for resume processing.
Uses asyncio + ProcessPoolExecutor to parse documents without
blocking user-facing requests.

Flow:
  1. API receives upload -> stores to S3 -> enqueues job to SQS
  2. Worker polls SQS -> downloads from S3 -> parses in process pool
  3. Stores structured profile to DB -> publishes result to SNS
"""
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from pipeline.parsers.pdf_parser import PDFResumeParser, PDFParserError
from pipeline.parsers.docx_parser import DOCXResumeParser, DOCXParserError
from pipeline.extractor.nlp_extractor import NLPExtractor, CandidateProfile
from pipeline.storage import CandidateStore

logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")
S3_BUCKET = os.getenv("RESUME_S3_BUCKET", "resumes-bucket")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")

# Worker Configuration
MAX_WORKERS = int(os.getenv("PIPELINE_MAX_WORKERS", "4"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
MAX_MESSAGES_PER_POLL = int(os.getenv("MAX_MESSAGES_PER_POLL", "10"))
VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "120"))


@dataclass
class ParseJob:
    """Represents a resume parsing job from SQS."""
    job_id: str
    s3_key: str
    file_type: str  # "pdf" or "docx"
    candidate_id: Optional[str] = None
    receipt_handle: Optional[str] = None
    retries: int = 0


def _parse_document_sync(file_content: bytes, file_type: str, job_id: str) -> dict:
    """
    CPU-bound parsing function - runs in ProcessPoolExecutor.
    Isolated from the async event loop for non-blocking execution.
    """
    try:
        if file_type == "pdf":
            parser = PDFResumeParser()
            text = parser.extract_text_from_bytes(file_content, filename=job_id)
        elif file_type in ("docx", "doc"):
            parser = DOCXResumeParser()
            text = parser.extract_text_from_bytes(file_content, filename=job_id)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        extractor = NLPExtractor()
        profile = extractor.extract(text)
        return {"success": True, "profile": profile.to_dict(), "job_id": job_id}

    except (PDFParserError, DOCXParserError) as e:
        return {"success": False, "error": str(e), "error_type": "parse_error", "job_id": job_id}
    except Exception as e:
        logger.exception(f"Unexpected error in _parse_document_sync for job {job_id}")
        return {"success": False, "error": str(e), "error_type": "unknown", "job_id": job_id}


class ResumeProcessingWorker:
    """
    Async worker that processes resume parsing jobs from SQS.

    Key design decisions:
    - asyncio event loop handles I/O (S3 downloads, SQS polling, DB writes)
    - ProcessPoolExecutor handles CPU-bound parsing (PDF/DOCX/NLP)
    - Non-blocking: API requests are never delayed by parsing work
    - Graceful shutdown: finishes in-flight jobs on SIGTERM
    """

    def __init__(self):
        self.sqs = boto3.client("sqs", region_name=AWS_REGION)
        self.s3 = boto3.client("s3", region_name=AWS_REGION)
        self.sns = boto3.client("sns", region_name=AWS_REGION)
        self.store = CandidateStore()
        self.executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        self._running = False
        self._active_jobs = 0

    async def start(self):
        """Start the worker loop."""
        self._running = True
        logger.info(f"Resume pipeline worker started (max_workers={MAX_WORKERS})")
        await self._poll_loop()

    async def stop(self):
        """Graceful shutdown: stop polling, wait for active jobs."""
        logger.info("Shutting down worker...")
        self._running = False
        # Wait for active jobs to complete (up to 60s)
        for _ in range(60):
            if self._active_jobs == 0:
                break
            await asyncio.sleep(1)
        self.executor.shutdown(wait=True)
        logger.info("Worker shutdown complete.")

    async def _poll_loop(self):
        """Main polling loop for SQS messages."""
        while self._running:
            try:
                messages = await self._receive_messages()
                if messages:
                    tasks = [self._process_message(msg) for msg in messages]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _receive_messages(self) -> list:
        """Poll SQS for new parsing jobs (non-blocking via executor)."""
        if not SQS_QUEUE_URL:
            return []
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs.receive_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MaxNumberOfMessages=MAX_MESSAGES_PER_POLL,
                    WaitTimeSeconds=20,  # Long polling
                    VisibilityTimeout=VISIBILITY_TIMEOUT,
                    AttributeNames=["ApproximateReceiveCount"],
                ),
            )
            return response.get("Messages", [])
        except (BotoCoreError, ClientError) as e:
            logger.error(f"SQS receive error: {e}")
            return []

    async def _process_message(self, message: dict):
        """Process a single SQS message: download -> parse -> store -> ack."""
        self._active_jobs += 1
        receipt_handle = message["ReceiptHandle"]
        start_time = time.perf_counter()

        try:
            body = json.loads(message["Body"])
            job = ParseJob(
                job_id=body["job_id"],
                s3_key=body["s3_key"],
                file_type=body["file_type"],
                candidate_id=body.get("candidate_id"),
                receipt_handle=receipt_handle,
            )

            logger.info(f"Processing job {job.job_id} (type={job.file_type}, s3={job.s3_key})")

            # Step 1: Download from S3 (async I/O)
            file_content = await self._download_from_s3(job.s3_key)

            # Step 2: Parse document (CPU-bound, offloaded to process pool)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                _parse_document_sync,
                file_content,
                job.file_type,
                job.job_id,
            )

            if result["success"]:
                # Step 3: Store structured profile to DB
                profile_dict = result["profile"]
                profile_dict["job_id"] = job.job_id
                profile_dict["s3_key"] = job.s3_key
                await self.store.save_candidate(job.candidate_id, profile_dict)

                # Step 4: Publish completion to SNS
                await self._publish_result(job.job_id, profile_dict, success=True)

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"Job {job.job_id} completed in {elapsed:.0f}ms")
            else:
                logger.error(f"Job {job.job_id} failed: {result['error']}")
                await self._publish_result(job.job_id, result, success=False)

            # Step 5: Delete from SQS (acknowledge)
            await self._delete_message(receipt_handle)

        except Exception as e:
            logger.exception(f"Failed to process message: {e}")
            # Leave in queue for retry (visibility timeout expires)
        finally:
            self._active_jobs -= 1

    async def _download_from_s3(self, s3_key: str) -> bytes:
        """Download resume file from S3 asynchronously."""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.s3.get_object(Bucket=S3_BUCKET, Key=s3_key),
            )
            content = await loop.run_in_executor(
                None,
                lambda: response["Body"].read(),
            )
            logger.debug(f"Downloaded {len(content)} bytes from S3: {s3_key}")
            return content
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 download failed for {s3_key}: {e}") from e

    async def _publish_result(self, job_id: str, data: dict, success: bool):
        """Publish parsing result to SNS for downstream consumers."""
        if not SNS_TOPIC_ARN:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.sns.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Message=json.dumps({
                        "job_id": job_id,
                        "success": success,
                        "data": data,
                    }),
                    Subject=f"resume_parsed_{'ok' if success else 'error'}",
                ),
            )
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"SNS publish failed for job {job_id}: {e}")

    async def _delete_message(self, receipt_handle: str):
        """Delete processed message from SQS."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.sqs.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle,
                ),
            )
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"SQS delete failed: {e}")


async def main():
    """Entrypoint for running the worker directly."""
    import signal
    worker = ResumeProcessingWorker()

    def handle_signal(sig):
        asyncio.create_task(worker.stop())

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: handle_signal(signal.SIGTERM))
    loop.add_signal_handler(signal.SIGINT, lambda: handle_signal(signal.SIGINT))

    await worker.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
