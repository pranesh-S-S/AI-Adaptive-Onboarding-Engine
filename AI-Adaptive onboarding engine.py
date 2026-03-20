import fitz
from docx import Document
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import re
import logging
import os
import json
from groq import Groq
from collections import defaultdict

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── GROQ CLIENT SETUP ─────────────────────────────────────────────────────────

client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"   # ✅ FREE + FAST


# ── GROQ CALLER ───────────────────────────────────────────────────────────────

def call_groq(prompt: str, max_tokens: int = 1024) -> dict:
    try:
        log.info("Calling Groq API...")
        response = client.chat.completions.create(
            model      = GROQ_MODEL,
            messages   = [
                {
                    "role"   : "system",
                    "content": "You are an expert resume analyst. Always respond in valid JSON only. Never add explanation or markdown."
                },
                {
                    "role"   : "user",
                    "content": prompt
                }
            ],
            temperature = 0.1,
            max_tokens  = max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return {}
    except Exception as e:
        log.error(f"Groq API call failed: {e}")
        return {}


# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}', '', text)
    text = re.sub(r'[^\w\s.,+-]', '', text)
    return text.strip()


# ── SKILL NORMALIZATION ───────────────────────────────────────────────────────

SKILL_MAP = {
    "ml"                          : "Machine Learning",
    "machine learning"            : "Machine Learning",
    "ai"                          : "Artificial Intelligence",
    "dl"                          : "Deep Learning",
    "nlp"                         : "Natural Language Processing",
    "cv"                          : "Computer Vision",
    "torch"                       : "PyTorch",
    "tf"                          : "TensorFlow",
    "sklearn"                     : "Scikit-learn",
    "sk-learn"                    : "Scikit-learn",
    "scikit learn"                : "Scikit-learn",
    "postgres"                    : "PostgreSQL",
    "mongo"                       : "MongoDB",
    "js"                          : "JavaScript",
    "ts"                          : "TypeScript",
    "k8s"                         : "Kubernetes",
    "aws"                         : "Amazon Web Services",
    "gcp"                         : "Google Cloud Platform",
    "oop"                         : "Object Oriented Programming",
    "rest"                        : "REST APIs",
    "restful"                     : "REST APIs",
    "dsa"                         : "Data Structures and Algorithms",
    "data analysis"               : "Data Analysis",
    "deep learning"               : "Deep Learning",
    "natural language processing" : "Natural Language Processing",
    "computer vision"             : "Computer Vision",
}

def normalize_skill(skill: str) -> str:
    return SKILL_MAP.get(skill.lower().strip(), skill.strip())


# ── STOPWORDS ─────────────────────────────────────────────────────────────────

COMMON_STOPWORDS = {
    "and", "the", "with", "using", "good", "team", "also",
    "have", "has", "for", "are", "was", "been", "from",
    "this", "that", "use", "used", "will", "can", "able",
    "strong", "well", "such", "both", "like", "work",
    "worked", "experience", "knowledge", "skills", "skill",
    "during", "while", "various", "projects", "across"
}


# ── SECTION KEYWORDS & WEIGHTS ────────────────────────────────────────────────

SECTION_KEYWORDS = {
    "skills"    : ["skill", "technical", "tools", "competencies",
                   "expertise", "technologies", "proficiencies", "stack"],
    "experience": ["experience", "work", "employment", "career",
                   "history", "internship", "position", "role"],
    "education" : ["education", "academic", "qualification",
                   "degree", "university", "college", "school"],
    "projects"  : ["project", "portfolio", "built", "developed",
                   "work sample", "personal work", "open source"]
}

SECTION_WEIGHTS = {
    "skills"    : 1.0,
    "experience": 0.8,
    "projects"  : 0.7,
    "education" : 0.5,
    "others"    : 0.3
}


# ── SECTION DETECTION ─────────────────────────────────────────────────────────

def detect_sections(text: str) -> tuple:
    sections        = {k: [] for k in [*SECTION_KEYWORDS.keys(), "others"]}
    section_hits    = {k: 0  for k in sections}
    current_section = "others"

    for line in text.split("\n"):
        l = line.lower().strip()
        for section, keywords in SECTION_KEYWORDS.items():
            if l.startswith(tuple(keywords)):
                current_section = section
                section_hits[current_section] += 1
                break
        sections[current_section].append(line)

    for key in sections:
        sections[key] = "\n".join(sections[key]).strip()

    total_hits = max(sum(section_hits.values()), 1)
    confidence = {
        f"{k}_confidence": round(section_hits[k] / total_hits, 2)
        for k in section_hits
    }

    return sections, confidence


# ── SKILL EXTRACTION ─────────────────────────────────────────────────────────

def extract_raw_skills(sections: dict) -> list:
    skills_text = sections["skills"]
    candidates  = re.split(r',|\n', skills_text)
    cleaned     = []

    for skill in candidates:
        skill = skill.strip()
        if not skill:
            continue
        words = [
            w for w in skill.split()
            if w.lower() not in COMMON_STOPWORDS
        ]
        if not words:
            continue
        phrase     = " ".join(words)
        normalized = normalize_skill(phrase)
        cleaned.append(normalized)

    return list(set(cleaned))


# ── SKILL FREQUENCY ───────────────────────────────────────────────────────────

def detect_skill_frequency(text: str, skills: list) -> dict:
    text_lower = text.lower()
    frequency  = {}

    for skill in skills:
        skill_lower = skill.lower()
        count = text_lower.count(skill_lower)
        for alias, normalized in SKILL_MAP.items():
            if normalized.lower() == skill_lower:
                count += text_lower.count(alias.lower())
        frequency[skill] = count

    return frequency


# ── PRIORITY SCORING ─────────────────────────────────────────────────────────

def compute_skill_scores(sections: dict, skills: list) -> dict:
    skill_scores = defaultdict(float)
    for section, weight in SECTION_WEIGHTS.items():
        section_text = sections.get(section, "").lower()
        for skill in skills:
            if skill.lower() in section_text:
                skill_scores[skill] = max(skill_scores[skill], weight)
    return dict(skill_scores)


# ── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 1500) -> list:
    return [text[i:i+size] for i in range(0, len(text), size)]


def merge_llm_results(results: list) -> dict:
    merged = {}
    for result in results:
        if not result:
            continue
        for skill, data in result.get("skills", {}).items():
            if skill not in merged:
                merged[skill] = data
            else:
                if len(data.get("reason", "")) > len(merged[skill].get("reason", "")):
                    merged[skill] = data
    return {"skills": merged}


# ── FILE HANDLER ─────────────────────────────────────────────────────────────

def extract_text_from_file(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(path)
    elif ext == ".docx":
        return extract_from_docx(path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX supported.")


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────

def extract_from_pdf(path: str) -> str:
    try:
        doc      = fitz.open(path)
        all_text = []
        for page in doc:
            page_text = []
            blocks    = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip()
                if not block_text:
                    continue
                for line in block_text.split("\n"):
                    if line.strip():
                        page_text.append(line.strip())
            all_text.append("\n".join(page_text))
        doc.close()
        full_text = "\n\n".join(all_text).strip()
        if len(full_text) < 100:
            log.warning("Scanned PDF detected — switching to OCR...")
            full_text = extract_from_scanned_pdf(path)
        return clean_text(full_text)
    except Exception as e:
        log.error(f"PDF extraction failed: {e}")
        raise


# ── OCR ───────────────────────────────────────────────────────────────────────

def preprocess_image(image):
    image     = np.array(image)
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    return thresh


def extract_from_scanned_pdf(path: str) -> str:
    try:
        images   = convert_from_path(path, dpi=200, thread_count=4)
        all_text = []
        for i, image in enumerate(images):
            log.info(f"OCR processing page {i+1}...")
            processed = preprocess_image(image)
            text      = pytesseract.image_to_string(processed)
            lines     = [l.strip() for l in text.split("\n") if l.strip()]
            all_text.append("\n".join(lines))
        return clean_text("\n\n".join(all_text))
    except Exception as e:
        log.error(f"OCR extraction failed: {e}")
        raise


# ── DOCX EXTRACTION ───────────────────────────────────────────────────────────

def extract_from_docx(path: str) -> str:
    try:
        doc      = Document(path)
        all_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n".join(all_text))
    except Exception as e:
        log.error(f"DOCX extraction failed: {e}")
        raise


# ═════════════════════════════════════════════════════════════════════════════
# JOB DESCRIPTION PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

JD_SECTION_KEYWORDS = {
    "required_skills" : ["required", "must have", "mandatory",
                         "qualifications", "requirements"],
    "preferred_skills": ["preferred", "nice to have", "good to have",
                         "bonus", "optional", "plus"],
    "responsibilities": ["responsibilities", "duties", "you will",
                         "what you'll do", "role", "job description"],
    "about_role"      : ["about", "overview", "summary",
                         "who we are", "the role"]
}


def detect_jd_sections(text: str) -> dict:
    sections        = {k: [] for k in [*JD_SECTION_KEYWORDS.keys(), "others"]}
    current_section = "others"

    for line in text.split("\n"):
        l = line.lower().strip()
        for section, keywords in JD_SECTION_KEYWORDS.items():
            if l.startswith(tuple(keywords)):
                current_section = section
                break
        sections[current_section].append(line)

    for key in sections:
        sections[key] = "\n".join(sections[key]).strip()

    return sections


def extract_jd_skills(jd_text: str) -> list:
    """
    Extract clean technical skills only from JD
    """

    candidates = re.split(r',|\n|•|-|\*', jd_text)

    skills = []

    for candidate in candidates:
        candidate = candidate.strip().lower()

        if not candidate:
            continue

        # ❌ REMOVE garbage phrases
        if any(x in candidate for x in [
            "responsibilities", "required", "preferred",
            "press enter", "build", "collaborate",
            "work with", "develop", "you will"
        ]):
            continue

        # ❌ REMOVE long sentences
        if len(candidate.split()) > 3:
            continue

        # ❌ REMOVE stopwords
        words = [
            w for w in candidate.split()
            if w not in COMMON_STOPWORDS
        ]

        if not words:
            continue

        phrase = " ".join(words)
        normalized = normalize_skill(phrase)

        skills.append(normalized)

    return list(set(skills))

def process_job_description(jd_input: str) -> dict:
    """
    Handles:
    - File input (.pdf / .docx)
    - Multi-line plain text
    - Single-line pasted JD (auto formatted)
    """
    jd_text      = ""
    cleaned_path = jd_input.strip().strip('"').strip("'")

    # ── case 1: file input ────────────────────────────────────────
    if os.path.exists(cleaned_path):
        log.info("JD input detected as file — extracting text...")
        jd_text = extract_text_from_file(cleaned_path)

    # ── case 2: plain text input ──────────────────────────────────
    else:
        log.info("JD input detected as plain text...")
        raw_text = jd_input.strip()

        # auto-format if no line breaks detected
        if "\n" not in raw_text:
            raw_text = re.sub(r"(Required Skills:)",
                              r"\n\1\n", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"(Preferred Skills:)",
                              r"\n\1\n", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"(Responsibilities:)",
                              r"\n\1\n", raw_text, flags=re.IGNORECASE)

        jd_text = clean_text(raw_text)

    if not jd_text:
        return {"error": "Job description is empty"}

    # ── section detection ─────────────────────────────────────────
    jd_sections    = detect_jd_sections(jd_text)
    required_text  = (jd_sections["required_skills"] + "\n" +
                      jd_sections["responsibilities"])
    preferred_text = jd_sections["preferred_skills"]

    required_skills  = extract_jd_skills(required_text)
    preferred_skills = extract_jd_skills(preferred_text)

    # ── role title extraction ─────────────────────────────────────
    role_title = ""
    for line in jd_text.split("\n"):
        line = line.strip()
        if line and len(line.split()) <= 8:
            role_title = line
            break

    if not role_title:
        role_title = "Unknown Role"

    return {
        "role_title"       : role_title,
        "raw_jd_text"      : jd_text,
        "jd_sections"      : jd_sections,
        "required_skills"  : required_skills,
        "preferred_skills" : preferred_skills,
    }


# ═════════════════════════════════════════════════════════════════════════════
# GAP ANALYSIS  ← THIS WAS THE MISSING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def perform_gap_analysis(resume_skills: list,
                          required_skills: list,
                          preferred_skills: list) -> dict:
    """
    Compares resume skills vs JD required/preferred skills.
    Returns overlapping skills, gap skills, match score, gap score.
    """
    resume_lower = [s.lower() for s in resume_skills]

    overlapping   = [
        s for s in required_skills
        if s.lower() in resume_lower
    ]
    gap_skills    = [
        s for s in required_skills
        if s.lower() not in resume_lower
    ]
    preferred_gap = [
        s for s in preferred_skills
        if s.lower() not in resume_lower
    ]

    total_required = max(len(required_skills), 1)
    match_score    = round((len(overlapping) / total_required) * 100, 1)
    gap_score      = round((len(gap_skills)  / total_required) * 100, 1)

    return {
        "overlapping_skills": overlapping,
        "gap_skills"        : gap_skills,
        "preferred_gap"     : preferred_gap,
        "matched_count"     : len(overlapping),
        "total_required"    : len(required_skills),
        "match_score"       : match_score,
        "gap_score"         : gap_score,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

def prepare_chunk_prompt(chunk: str, chunk_index: int,
                          total_chunks: int) -> str:
    return f"""
You are an expert resume analyst. This is part {chunk_index + 1} of {total_chunks} of a resume.

Extract ALL technical skills from this section.
For each skill, classify the level and provide reasoning.

Resume Text (chunk {chunk_index + 1}/{total_chunks}):
{chunk}

Instructions:
- Extract only real technical skills (no soft skills, no generic words).
- Classify each as: Beginner, Intermediate, or Advanced.
- Base classification on context clues: years, project depth, job titles.
- Normalize aliases (e.g. "torch" -> "PyTorch", "ml" -> "Machine Learning").

Return STRICT JSON only, no explanation, no markdown:
{{
  "skills": {{
    "Python": {{
      "level": "Intermediate",
      "reason": "Used in 2 projects, 1 year experience mentioned"
    }}
  }}
}}
"""


def prepare_llm_input(sections: dict, skill_frequency: dict,
                      skill_scores: dict) -> str:
    freq_hint = ""
    if skill_frequency:
        top        = sorted(skill_frequency.items(),
                            key=lambda x: x[1], reverse=True)[:10]
        freq_hint  = "Skill mention frequency (higher = more experienced):\n"
        freq_hint += "\n".join(f"  - {s}: {c} time(s)" for s, c in top)

    score_hint = ""
    if skill_scores:
        sorted_scores = sorted(skill_scores.items(),
                               key=lambda x: x[1], reverse=True)
        score_hint  = "Skill priority scores:\n"
        score_hint += "\n".join(
            f"  - {s}: {round(sc, 2)}" for s, sc in sorted_scores[:10]
        )

    return f"""
You are an expert resume analyst.

Extract ALL technical skills from the resume sections below.
For each skill, classify the proficiency level and explain your reasoning.

{freq_hint}

{score_hint}

Resume Sections:

[SKILLS]
{sections['skills']}

[EXPERIENCE]
{sections['experience']}

[PROJECTS]
{sections['projects']}

[EDUCATION]
{sections['education']}

Instructions:
- Extract only real technical skills, no generic soft skills.
- Classify each skill as: Beginner, Intermediate, or Advanced.
- Base classification on: years mentioned, project depth, job titles.
- Normalize aliases (e.g. "torch" -> "PyTorch", "ml" -> "Machine Learning").

Return STRICT JSON only, no explanation, no markdown:
{{
  "skills": {{
    "Python": {{
      "level": "Intermediate",
      "reason": "Used in 2 projects, 1 year experience mentioned"
    }},
    "SQL": {{
      "level": "Beginner",
      "reason": "Only basic SELECT queries mentioned"
    }}
  }}
}}
"""


def prepare_gap_analysis_prompt(resume_skills: list,
                                 gap_analysis: dict,
                                 role_title: str,
                                 resume_text: str) -> str:
    return f"""
You are an expert career coach and skill assessor.

Candidate is targeting the role: {role_title}

Resume text:
\"\"\"{resume_text[:2000]}\"\"\"

Overlapping skills (candidate HAS these, role REQUIRES them):
{gap_analysis['overlapping_skills']}

Gap skills (candidate is MISSING these for the role):
{gap_analysis['gap_skills']}

Preferred skills candidate is also missing:
{gap_analysis['preferred_gap']}

Task:
1. For each OVERLAPPING skill, diagnose the candidate's current level
   based on how they used it in their resume.
2. For each GAP skill, provide a short learning recommendation.

Return STRICT JSON only, no explanation, no markdown:
{{
  "overlapping_skills": {{
    "Python": {{
      "diagnosed_level"  : "Beginner",
      "required_level"   : "Advanced",
      "needs_upskilling" : true,
      "evidence"         : "1 year basic scripting mentioned",
      "focus"            : "Learn ML libraries, OOP, data pipelines"
    }}
  }},
  "gap_skills": {{
    "Docker": {{
      "priority"        : "High",
      "reason"          : "Required for deployment in ML engineer role",
      "recommended_path": "Learn Docker basics, containers, docker-compose"
    }}
  }}
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# GROQ LLM CALLS
# ═════════════════════════════════════════════════════════════════════════════

def call_groq_resume_analysis(sections: dict, skill_frequency: dict,
                               skill_scores: dict) -> dict:
    log.info("Calling Groq for resume skill analysis...")
    prompt = prepare_llm_input(sections, skill_frequency, skill_scores)
    result = call_groq(prompt, max_tokens=1500)
    log.info(f"Resume analysis returned {len(result.get('skills', {}))} skills.")
    return result


def call_groq_chunk_analysis(chunks: list) -> dict:
    log.info(f"Calling Groq for {len(chunks)} chunk(s)...")
    results = []
    for i, chunk in enumerate(chunks):
        log.info(f"Processing chunk {i+1}/{len(chunks)}...")
        prompt = prepare_chunk_prompt(chunk, i, len(chunks))
        result = call_groq(prompt, max_tokens=1024)
        results.append(result)
    merged = merge_llm_results(results)
    log.info(f"Merged chunk results: {len(merged.get('skills', {}))} skills.")
    return merged


def call_groq_gap_analysis(resume_skills: list, gap_analysis: dict,
                            role_title: str, resume_text: str) -> dict:
    log.info("Calling Groq for gap analysis and level diagnosis...")
    prompt = prepare_gap_analysis_prompt(
        resume_skills, gap_analysis, role_title, resume_text
    )
    result = call_groq(prompt, max_tokens=2000)
    log.info("Gap analysis complete.")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# RESUME PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def process_resume(path: str) -> dict:
    log.info(f"Processing resume: {path}")

    try:
        raw_text = extract_text_from_file(path)
    except Exception as e:
        log.error(f"Extraction error: {e}")
        return {"error": str(e)}

    lines      = [l for l in raw_text.split("\n") if l.strip()]
    paragraphs = [p for p in raw_text.split("\n\n") if p.strip()]

    sections, confidence = detect_sections(raw_text)
    normalized_skills    = extract_raw_skills(sections)
    skill_frequency      = detect_skill_frequency(raw_text, normalized_skills)
    skill_scores         = compute_skill_scores(sections, normalized_skills)
    chunks               = chunk_text(raw_text, size=1500)

    log.info(f"Resume split into {len(chunks)} chunk(s) for LLM.")

    return {
        "raw_text"         : raw_text,
        "sections"         : sections,
        "confidence"       : confidence,
        "normalized_skills": normalized_skills,
        "skill_frequency"  : skill_frequency,
        "skill_scores"     : skill_scores,
        "chunks"           : chunks,
        "stats"            : {
            "lines"        : len(lines),
            "paragraphs"   : len(paragraphs),
            "chars"        : len(raw_text),
            "chunks"       : len(chunks)
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(resume_path: str, jd_input: str) -> dict:
    log.info("=" * 60)
    log.info("Starting full pipeline...")
    log.info("=" * 60)

    # step 1 — resume
    log.info("Step 1: Processing resume...")
    resume_result = process_resume(resume_path)
    if "error" in resume_result:
        return {"error": f"Resume error: {resume_result['error']}"}

    # step 2 — job description
    log.info("Step 2: Processing job description...")
    jd_result = process_job_description(jd_input)
    if "error" in jd_result:
        return {"error": f"JD error: {jd_result['error']}"}

    # step 3 — gap analysis
    log.info("Step 3: Performing gap analysis...")
    gap_analysis = perform_gap_analysis(
        resume_skills    = resume_result["normalized_skills"],
        required_skills  = jd_result["required_skills"],
        preferred_skills = jd_result["preferred_skills"]
    )

    # step 4 — groq resume skill analysis
    log.info("Step 4: Groq resume skill level analysis...")
    if resume_result["stats"]["chunks"] > 1:
        groq_resume_result = call_groq_chunk_analysis(
            resume_result["chunks"]
        )
    else:
        groq_resume_result = call_groq_resume_analysis(
            sections        = resume_result["sections"],
            skill_frequency = resume_result["skill_frequency"],
            skill_scores    = resume_result["skill_scores"]
        )

    # step 5 — groq gap analysis + level diagnosis
    log.info("Step 5: Groq gap analysis + level diagnosis...")
    groq_gap_result = call_groq_gap_analysis(
        resume_skills = resume_result["normalized_skills"],
        gap_analysis  = gap_analysis,
        role_title    = jd_result["role_title"],
        resume_text   = resume_result["raw_text"]
    )

    return {
        "resume"            : resume_result,
        "jd"                : jd_result,
        "gap_analysis"      : gap_analysis,
        "groq_skill_levels" : groq_resume_result,
        "groq_gap_diagnosis": groq_gap_result,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("   AI ADAPTIVE ONBOARDING ENGINE — GROQ + LLAMA 3")
    print("="*60)

    resume_path = input("\nEnter resume path (.pdf or .docx): ").strip()
    print("\nEnter job description:")
    print("(Type/paste text OR enter a .pdf/.docx file path)")
    jd_input    = input("JD input: ").strip()

    result = run_full_pipeline(resume_path, jd_input)

    if "error" in result:
        print(f"\nError: {result['error']}")

    else:
        print("\n" + "="*60)
        print("RESUME — EXTRACTED SKILLS")
        print("="*60)
        for skill in result["resume"]["normalized_skills"]:
            print(f"  - {skill}")

        print("\n" + "="*60)
        print("RESUME — SECTION CONFIDENCE")
        print("="*60)
        for k, v in result["resume"]["confidence"].items():
            print(f"  {k}: {v}")

        print("\n" + "="*60)
        print("JOB DESCRIPTION")
        print("="*60)
        print(f"  Role       : {result['jd']['role_title']}")
        print(f"  Required   : {result['jd']['required_skills']}")
        print(f"  Preferred  : {result['jd']['preferred_skills']}")

        print("\n" + "="*60)
        print("GAP ANALYSIS")
        print("="*60)
        g = result["gap_analysis"]
        print(f"  Match Score : {g['match_score']}%")
        print(f"  Gap Score   : {g['gap_score']}%")
        print(f"  Matched     : {g['matched_count']} / {g['total_required']}")
        print(f"  Overlapping : {g['overlapping_skills']}")
        print(f"  Missing     : {g['gap_skills']}")
        print(f"  Preferred   : {g['preferred_gap']}")

        print("\n" + "="*60)
        print("GROQ — SKILL LEVEL ANALYSIS")
        print("="*60)
        skills = result["groq_skill_levels"].get("skills", {})
        for skill, data in skills.items():
            print(f"\n  {skill}")
            print(f"    Level  : {data.get('level',  'N/A')}")
            print(f"    Reason : {data.get('reason', 'N/A')}")

        print("\n" + "="*60)
        print("GROQ — GAP DIAGNOSIS + LEARNING PATH")
        print("="*60)

        overlapping = result["groq_gap_diagnosis"].get("overlapping_skills", {})
        print("\n  --- Overlapping Skills (level diagnosis) ---")
        for skill, data in overlapping.items():
            print(f"\n  {skill}")
            print(f"    Diagnosed Level  : {data.get('diagnosed_level',  'N/A')}")
            print(f"    Required Level   : {data.get('required_level',   'N/A')}")
            print(f"    Needs Upskilling : {data.get('needs_upskilling', 'N/A')}")
            print(f"    Evidence         : {data.get('evidence',         'N/A')}")
            print(f"    Focus            : {data.get('focus',            'N/A')}")

        gap_skills = result["groq_gap_diagnosis"].get("gap_skills", {})
        print("\n  --- Gap Skills (learning recommendations) ---")
        for skill, data in gap_skills.items():
            print(f"\n  {skill}")
            print(f"    Priority         : {data.get('priority',         'N/A')}")
            print(f"    Reason           : {data.get('reason',           'N/A')}")
            print(f"    Recommended Path : {data.get('recommended_path', 'N/A')}")

        print("\n" + "="*60)
        print("STATS")
        print("="*60)
        for k, v in result["resume"]["stats"].items():
            print(f"  {k}: {v}")