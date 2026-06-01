"""
NLP-based structured data extractor using spaCy.
Extracts: name, email, phone, skills, education, experience,
          years_of_experience, and certifications from raw resume text.
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass, field

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language

logger = logging.getLogger(__name__)

# Load spaCy English model (will use en_core_web_sm or lg based on availability)
_nlp: Optional[Language] = None


def get_nlp() -> Language:
    """Lazy-load spaCy model (expensive, loaded once per worker)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy model: en_core_web_lg")
        except OSError:
            try:
                _nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm (lg not available)")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Run: python -m spacy download en_core_web_sm"
                )
    return _nlp


@dataclass
class CandidateProfile:
    """Structured candidate data extracted from resume text."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    years_of_experience: Optional[float] = None
    raw_text_length: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization/DB storage."""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "linkedin_url": self.linkedin_url,
            "github_url": self.github_url,
            "summary": self.summary,
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,
            "certifications": self.certifications,
            "years_of_experience": self.years_of_experience,
            "raw_text_length": self.raw_text_length,
        }


class NLPExtractor:
    """
    Extracts structured candidate information from raw resume text using spaCy.

    Pipeline:
    1. Named Entity Recognition (NER) for names, locations, organizations
    2. Regex patterns for emails, phones, URLs
    3. Keyword/phrase matching for skills
    4. Heuristic section detection for experience/education parsing
    """

    # Common tech skills for phrase matching
    TECH_SKILLS = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "React", "Vue", "Angular", "Node.js", "Django", "FastAPI", "Flask",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
        "Git", "GitHub", "GitLab", "CI/CD", "Jenkins", "GitHub Actions",
        "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch",
        "pandas", "NumPy", "scikit-learn", "SQL", "NoSQL", "GraphQL", "REST",
        "gRPC", "Kafka", "RabbitMQ", "Celery", "Linux", "Bash", "Shell",
        "spaCy", "NLTK", "transformers", "OpenAI", "LangChain",
    ]

    # Section header patterns
    SECTION_PATTERNS = {
        "experience": re.compile(
            r"(?i)^(work experience|professional experience|employment|experience|work history)",
            re.MULTILINE,
        ),
        "education": re.compile(
            r"(?i)^(education|academic|qualifications|degrees?)",
            re.MULTILINE,
        ),
        "skills": re.compile(
            r"(?i)^(skills|technical skills|core competencies|technologies|stack)",
            re.MULTILINE,
        ),
        "certifications": re.compile(
            r"(?i)^(certifications?|licenses?|credentials?|courses?)",
            re.MULTILINE,
        ),
        "summary": re.compile(
            r"(?i)^(summary|objective|profile|about me|overview)",
            re.MULTILINE,
        ),
    }

    # Regex for contact info
    EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    PHONE_RE = re.compile(
        r"(\+?1?[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    )
    LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
    GITHUB_RE = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

    def __init__(self):
        self.nlp = get_nlp()
        self._setup_matchers()

    def _setup_matchers(self):
        """Initialize spaCy PhraseMatcher for skill detection."""
        self.skill_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        skill_docs = [self.nlp.make_doc(skill) for skill in self.TECH_SKILLS]
        self.skill_matcher.add("TECH_SKILL", skill_docs)

    def extract(self, text: str) -> CandidateProfile:
        """
        Main extraction entry point.
        Returns a fully populated CandidateProfile.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to NLPExtractor")
            return CandidateProfile(raw_text_length=0)

        profile = CandidateProfile(raw_text_length=len(text))

        # Run spaCy NLP pipeline
        doc = self.nlp(text[:100000])  # Limit to 100k chars for performance

        # Extract contact info via regex (more reliable than NER for these)
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.linkedin_url = self._extract_linkedin(text)
        profile.github_url = self._extract_github(text)

        # Extract name via NER (PERSON entities near top of document)
        profile.name = self._extract_name(doc)

        # Extract location via NER
        profile.location = self._extract_location(doc)

        # Extract skills via PhraseMatcher
        profile.skills = self._extract_skills(doc)

        # Extract sections for experience/education/certifications
        sections = self._split_sections(text)
        profile.experience = self._parse_experience_section(sections.get("experience", ""))
        profile.education = self._parse_education_section(sections.get("education", ""))
        profile.certifications = self._parse_certifications(sections.get("certifications", ""))
        profile.summary = sections.get("summary", "")[:500] if sections.get("summary") else None

        # Estimate years of experience
        profile.years_of_experience = self._estimate_years_experience(text)

        logger.info(
            f"Extracted profile: name={profile.name}, "
            f"skills={len(profile.skills)}, "
            f"experience_entries={len(profile.experience)}"
        )
        return profile

    def _extract_email(self, text: str) -> Optional[str]:
        match = self.EMAIL_RE.search(text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        match = self.PHONE_RE.search(text)
        return match.group(0).strip() if match else None

    def _extract_linkedin(self, text: str) -> Optional[str]:
        match = self.LINKEDIN_RE.search(text)
        return match.group(0) if match else None

    def _extract_github(self, text: str) -> Optional[str]:
        match = self.GITHUB_RE.search(text)
        return match.group(0) if match else None

    def _extract_name(self, doc) -> Optional[str]:
        """Extract candidate name: first PERSON entity in top 500 chars."""
        first_500 = doc.text[:500]
        short_doc = self.nlp(first_500)
        for ent in short_doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                return ent.text.strip()
        return None

    def _extract_location(self, doc) -> Optional[str]:
        """Extract first GPE (geo-political entity) as location."""
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                return ent.text.strip()
        return None

    def _extract_skills(self, doc) -> list[str]:
        """Extract skills using PhraseMatcher."""
        matches = self.skill_matcher(doc)
        seen = set()
        skills = []
        for _, start, end in matches:
            skill_text = doc[start:end].text
            normalized = skill_text.lower()
            if normalized not in seen:
                seen.add(normalized)
                skills.append(skill_text)
        return skills

    def _split_sections(self, text: str) -> dict:
        """Split resume text into named sections."""
        sections = {}
        positions = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            for match in pattern.finditer(text):
                positions.append((match.start(), section_name, match.end()))

        positions.sort()
        for i, (start, name, end) in enumerate(positions):
            next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections[name] = text[end:next_start].strip()
        return sections

    def _parse_experience_section(self, text: str) -> list[dict]:
        """Parse experience section into list of job entries."""
        if not text:
            return []
        entries = []
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        current = {}
        for line in lines:
            years = self.YEAR_RE.findall(line)
            if len(years) >= 1 and len(line) < 150:
                if current:
                    entries.append(current)
                current = {"title_line": line, "years_mentioned": years, "description": []}
            elif current:
                current["description"].append(line)
        if current:
            entries.append(current)
        return entries[:20]  # Cap at 20 entries

    def _parse_education_section(self, text: str) -> list[dict]:
        """Parse education section into list of degree entries."""
        if not text:
            return []
        entries = []
        degree_pattern = re.compile(
            r"(?i)(bachelor|master|phd|doctorate|b\.s|m\.s|b\.a|m\.a|mba|associate)",
        )
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        current = {}
        for line in lines:
            if degree_pattern.search(line):
                if current:
                    entries.append(current)
                current = {"degree_line": line, "details": []}
            elif current:
                current["details"].append(line)
        if current:
            entries.append(current)
        return entries

    def _parse_certifications(self, text: str) -> list[str]:
        """Parse certifications as a simple list of non-empty lines."""
        if not text:
            return []
        return [l.strip() for l in text.split("\n") if l.strip()][:20]

    def _estimate_years_experience(self, text: str) -> Optional[float]:
        """
        Rough estimate: count unique year mentions in experience section.
        E.g., years 2018-2024 = 6 years.
        """
        years = [int(y) for y in self.YEAR_RE.findall(text)]
        years = [y for y in years if 1990 <= y <= 2026]
        if len(years) >= 2:
            return float(max(years) - min(years))
        return None
