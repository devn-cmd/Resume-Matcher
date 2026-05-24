
import re
from dataclasses import dataclass, field

# Canonical section -> header aliases that introduce it
_SECTION_HEADERS = {
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "career history",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "qualifications", "educational background",
    ],
    "certifications": [
        "certifications", "certification", "certificates", "licenses",
        "licenses and certifications", "courses and certifications",
    ],
    "skills": ["skills", "technical skills", "core competencies"],
    "projects": ["projects", "personal projects", "key projects"],
    "summary": ["summary", "profile", "objective", "about"],
}
_ALIAS_TO_SECTION = {
    alias: canon
    for canon, aliases in _SECTION_HEADERS.items()
    for alias in aliases
}

# A standalone header line: short, only letters/spaces, optional bullet/colon
_HEADER_RE = re.compile(r"^[\s\-•*]*([A-Za-z &/]+?)[\s:]*$")

_DATE = (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|"
         r"March|April|June|July|August|September|October|November|December)?"
         r"\.?\s*\d{4}")
_DATE_RANGE = re.compile(rf"({_DATE})\s*(?:-|–|—|to)\s*({_DATE}|present|current)",
                         re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_TITLE_ORG = re.compile(r"^(.*?)\s*(?:—|–|-|,|\bat\b|@)\s*(.*)$")
_DEGREE = re.compile(
    r"\b(ph\.?d|doctor(?:ate)?|m\.?s\.?c?|m\.?b\.?a|m\.?tech|master(?:'?s)?|"
    r"b\.?s\.?c?|b\.?e|b\.?tech|bachelor(?:'?s)?|associate|diploma)\b",
    re.IGNORECASE,
)
_INSTITUTION = re.compile(r"\b(university|college|institute|school|academy)\b",
                          re.IGNORECASE)


@dataclass
class ExperienceEntry:
    title: str = ""
    organization: str = ""
    start: str = ""
    end: str = ""
    raw: str = ""


@dataclass
class EducationEntry:
    degree: str = ""
    institution: str = ""
    year: str = ""
    raw: str = ""


@dataclass
class CertificationEntry:
    name: str = ""
    issuer: str = ""
    year: str = ""
    raw: str = ""


@dataclass
class StructuredResume:
    experience: list = field(default_factory=list)
    education: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)


def _match_header(line: str):
    m = _HEADER_RE.match(line.strip())
    if not m:
        return None
    return _ALIAS_TO_SECTION.get(m.group(1).strip().lower())


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into {canonical_section: block}. Lines before the
    first recognized header land under 'header' (name/contact area)."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        sec = _match_header(line)
        if sec:
            current = sec
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()
            if "\n".join(v).strip()}


def _split_entries(block: str) -> list[str]:
    """Split a section block into entries on blank lines."""
    chunks, cur = [], []
    for line in block.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            chunks.append("\n".join(cur)); cur = []
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def parse_experience(block: str) -> list[ExperienceEntry]:
    entries = []
    for chunk in _split_entries(block):
        lines = chunk.splitlines()
        first = lines[0].strip(" -•*\t") if lines else ""
        title, org = first, ""
        m = _TITLE_ORG.match(first)
        if m:
            title, org = m.group(1).strip(), m.group(2).strip()
        start = end = ""
        dr = _DATE_RANGE.search(chunk)
        if dr:
            start, end = dr.group(1).strip(), dr.group(2).strip()
        entries.append(ExperienceEntry(title=title, organization=org,
                                       start=start, end=end, raw=chunk.strip()))
    return entries


def parse_education(block: str) -> list[EducationEntry]:
    entries = []
    for chunk in _split_entries(block):
        dm = _DEGREE.search(chunk)
        ym = _YEAR.search(chunk)
        inst = ""
        for line in chunk.splitlines():
            if _INSTITUTION.search(line):
                inst = line.strip(" -•*\t"); break
        entries.append(EducationEntry(
            degree=dm.group(0) if dm else "",
            institution=inst,
            year=ym.group(0) if ym else "",
            raw=chunk.strip(),
        ))
    return entries


def parse_certifications(block: str) -> list[CertificationEntry]:
    entries = []
    for line in block.splitlines():
        line = line.strip(" -•*\t")
        if not line:
            continue
        ym = _YEAR.search(line)
        im = re.search(r"(?:by|from|–|—|-|,)\s*([A-Z][\w &.]+)", line)
        name = _YEAR.sub("", line).strip(" -–—,()")
        entries.append(CertificationEntry(
            name=name,
            issuer=im.group(1).strip() if im else "",
            year=ym.group(0) if ym else "",
            raw=line,
        ))
    return entries


def extract_entities(text: str) -> StructuredResume:
    """Parse raw resume text into structured experience/education/cert records."""
    secs = split_sections(text)
    return StructuredResume(
        experience=parse_experience(secs.get("experience", "")),
        education=parse_education(secs.get("education", "")),
        certifications=parse_certifications(secs.get("certifications", "")),
        sections=secs,
    )