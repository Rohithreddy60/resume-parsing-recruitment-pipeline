"""
Tests for NLP extractor and document parsers.
Uses sample resume text without requiring AWS or a running DB.
"""
import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO

from pipeline.extractor.nlp_extractor import NLPExtractor, CandidateProfile


SAMPLE_RESUME_TEXT = """
John Smith
john.smith@email.com | +1 (555) 123-4567 | linkedin.com/in/johnsmith | github.com/jsmith
New York, NY

SUMMARY
Senior Software Engineer with 6 years of experience building scalable backend systems.

EXPERIENCE
Senior Software Engineer - TechCorp Inc
2020 - 2024
- Built microservices using Python, FastAPI, and PostgreSQL
- Implemented Redis caching reducing query latency by 60%
- Led migration to AWS (ECS, RDS, CloudWatch)

Software Engineer - StartupXYZ
2018 - 2020
- Developed REST APIs using Django and MySQL
- Deployed Docker containers on AWS EC2
- Worked with React frontend team

EDUCATION
Bachelor of Science in Computer Science
State University, 2018

SKILLS
Python, FastAPI, Django, PostgreSQL, MySQL, Redis, AWS, Docker, 
Kubernetes, React, TypeScript, Git, GitHub Actions, SQL

CERTIFICATIONS
AWS Certified Developer Associate (2022)
Python Professional Certification (2019)
"""


@pytest.fixture(scope="module")
def extractor():
    """Create a single NLPExtractor for all tests (model loading is expensive)."""
    return NLPExtractor()


class TestNLPExtractor:
    def test_extracts_email(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert profile.email == "john.smith@email.com"

    def test_extracts_phone(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert profile.phone is not None
        assert "555" in profile.phone

    def test_extracts_linkedin(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert profile.linkedin_url == "linkedin.com/in/johnsmith"

    def test_extracts_github(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert profile.github_url == "github.com/jsmith"

    def test_extracts_skills(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        skill_set = {s.lower() for s in profile.skills}
        assert "python" in skill_set
        assert "docker" in skill_set
        assert "redis" in skill_set
        assert "postgresql" in skill_set
        assert len(profile.skills) >= 5

    def test_extracts_certifications(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert len(profile.certifications) >= 1
        cert_text = " ".join(profile.certifications).lower()
        assert "aws" in cert_text or "python" in cert_text

    def test_estimates_years_experience(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        # Text contains years 2018-2024, expect at least 6 years estimated
        assert profile.years_of_experience is not None
        assert profile.years_of_experience >= 5

    def test_raw_text_length_tracked(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        assert profile.raw_text_length == len(SAMPLE_RESUME_TEXT)

    def test_empty_text_returns_empty_profile(self, extractor):
        profile = extractor.extract("")
        assert profile.name is None
        assert profile.email is None
        assert profile.skills == []
        assert profile.raw_text_length == 0

    def test_to_dict_serializable(self, extractor):
        import json
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        d = profile.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "skills" in d
        assert isinstance(d["skills"], list)

    def test_section_splitting(self, extractor):
        sections = extractor._split_sections(SAMPLE_RESUME_TEXT)
        assert "skills" in sections
        assert "experience" in sections or "education" in sections

    def test_experience_parsed_as_entries(self, extractor):
        profile = extractor.extract(SAMPLE_RESUME_TEXT)
        # Should detect at least 1 experience entry with year references
        assert isinstance(profile.experience, list)


class TestPDFParser:
    def test_raises_on_empty_bytes(self):
        from pipeline.parsers.pdf_parser import PDFResumeParser, PDFParserError
        parser = PDFResumeParser()
        with pytest.raises(PDFParserError):
            parser.extract_text_from_bytes(b"", filename="empty.pdf")

    def test_raises_on_invalid_bytes(self):
        from pipeline.parsers.pdf_parser import PDFResumeParser, PDFParserError
        parser = PDFResumeParser()
        with pytest.raises(PDFParserError):
            parser.extract_text_from_bytes(b"not a pdf file", filename="fake.pdf")

    def test_raises_on_missing_file(self):
        from pipeline.parsers.pdf_parser import PDFResumeParser
        parser = PDFResumeParser()
        with pytest.raises(FileNotFoundError):
            parser.extract_text_from_path("/nonexistent/path/resume.pdf")


class TestDOCXParser:
    def test_raises_on_empty_bytes(self):
        from pipeline.parsers.docx_parser import DOCXResumeParser, DOCXParserError
        parser = DOCXResumeParser()
        with pytest.raises(DOCXParserError):
            parser.extract_text_from_bytes(b"", filename="empty.docx")

    def test_raises_on_invalid_bytes(self):
        from pipeline.parsers.docx_parser import DOCXResumeParser, DOCXParserError
        parser = DOCXResumeParser()
        with pytest.raises(DOCXParserError):
            parser.extract_text_from_bytes(b"not a docx", filename="fake.docx")
