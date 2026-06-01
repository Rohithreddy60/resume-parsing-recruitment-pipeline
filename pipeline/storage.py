"""
Storage layer for candidate profiles.
Supports PostgreSQL (via asyncpg) for structured candidate data.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/recruitment_db"
)

# Async connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=30,
            statement_cache_size=100,
        )
        await _ensure_tables(_pool)
    return _pool


async def _ensure_tables(pool: asyncpg.Pool):
    """Create candidate_profiles table if it doesn't exist."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_profiles (
                id SERIAL PRIMARY KEY,
                job_id TEXT UNIQUE NOT NULL,
                candidate_id TEXT,
                s3_key TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                location TEXT,
                linkedin_url TEXT,
                github_url TEXT,
                summary TEXT,
                skills JSONB DEFAULT '[]',
                education JSONB DEFAULT '[]',
                experience JSONB DEFAULT '[]',
                certifications JSONB DEFAULT '[]',
                years_of_experience FLOAT,
                raw_text_length INTEGER,
                parsed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        # Indexes for fast screening queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_candidate_email ON candidate_profiles(email);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_candidate_skills ON candidate_profiles USING GIN(skills);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_candidate_job_id ON candidate_profiles(job_id);
        """)
        logger.info("Candidate profiles table and indexes ensured.")


class CandidateStore:
    """Async interface for storing and querying candidate profiles."""

    async def save_candidate(self, candidate_id: Optional[str], profile: dict) -> int:
        """Insert or update a candidate profile."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row_id = await conn.fetchval("""
                INSERT INTO candidate_profiles (
                    job_id, candidate_id, s3_key, name, email, phone,
                    location, linkedin_url, github_url, summary,
                    skills, education, experience, certifications,
                    years_of_experience, raw_text_length, parsed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                          $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb,
                          $15, $16, $17)
                ON CONFLICT (job_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    skills = EXCLUDED.skills,
                    experience = EXCLUDED.experience,
                    parsed_at = NOW()
                RETURNING id
            """,
                profile.get("job_id"),
                candidate_id,
                profile.get("s3_key"),
                profile.get("name"),
                profile.get("email"),
                profile.get("phone"),
                profile.get("location"),
                profile.get("linkedin_url"),
                profile.get("github_url"),
                profile.get("summary"),
                json.dumps(profile.get("skills", [])),
                json.dumps(profile.get("education", [])),
                json.dumps(profile.get("experience", [])),
                json.dumps(profile.get("certifications", [])),
                profile.get("years_of_experience"),
                profile.get("raw_text_length"),
                datetime.utcnow(),
            )
            logger.info(f"Saved candidate profile id={row_id}")
            return row_id

    async def get_by_email(self, email: str) -> Optional[dict]:
        """Retrieve candidate profile by email."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM candidate_profiles WHERE email = $1 ORDER BY parsed_at DESC LIMIT 1",
                email.lower(),
            )
            return dict(row) if row else None

    async def search_by_skills(self, required_skills: list[str], min_years: float = 0) -> list[dict]:
        """
        Find candidates with matching skills using PostgreSQL GIN index.
        skills column is JSONB, enabling fast array containment queries.
        """
        skills_json = json.dumps([s.lower() for s in required_skills])
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, job_id, name, email, skills, years_of_experience, parsed_at
                FROM candidate_profiles
                WHERE skills @> $1::jsonb
                  AND (years_of_experience IS NULL OR years_of_experience >= $2)
                ORDER BY years_of_experience DESC NULLS LAST
                LIMIT 100
            """, skills_json, min_years)
            return [dict(r) for r in rows]

    async def list_recent(self, limit: int = 50) -> list[dict]:
        """List most recently parsed candidate profiles."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, job_id, name, email, skills, years_of_experience, parsed_at
                FROM candidate_profiles
                ORDER BY parsed_at DESC
                LIMIT $1
            """, limit)
            return [dict(r) for r in rows]
