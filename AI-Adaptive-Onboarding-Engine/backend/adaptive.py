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
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from functools import lru_cache

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── GROQ CLIENT SETUP ─────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Set it with: set GROQ_API_KEY=your_key_here (Windows) "
        "or export GROQ_API_KEY=your_key_here (Mac/Linux)"
    )

client     = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"


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
        if not raw:
            return {"skills": {}}
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return {"skills": {}}
    except Exception as e:
        log.error(f"Groq API call failed: {e}")
        return {"skills": {}}


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
    "react.js"                    : "React",
    "reactjs"                     : "React",
    "node.js"                     : "Node.js",
    "nodejs"                      : "Node.js",
    "vue.js"                      : "Vue.js",
    "vuejs"                       : "Vue.js",
    "next.js"                     : "Next.js",
    "nextjs"                      : "Next.js",
}

def normalize_skill(skill: str) -> str:
    return SKILL_MAP.get(skill.lower().strip(), skill.strip())


def normalize_skill_list(skills) -> list:
    result = []
    for s in skills:
        if isinstance(s, dict):
            name = s.get("skill", "")
        else:
            name = str(s)
        if name.strip():
            result.append(normalize_skill(name.strip()))
    return list(set(result))


# ── STOPWORDS ─────────────────────────────────────────────────────────────────

COMMON_STOPWORDS = {
    "and", "the", "with", "using", "good", "team", "also",
    "have", "has", "for", "are", "was", "been", "from",
    "this", "that", "use", "used", "will", "can", "able",
    "strong", "well", "such", "both", "like", "work",
    "worked", "experience", "knowledge", "skills", "skill",
    "during", "while", "various", "projects", "across"
}


# ── IMPORTANCE & LEVEL WEIGHTS ────────────────────────────────────────────────

IMPORTANCE_WEIGHTS = {
    "High"   : 1.0,
    "Medium" : 0.6,
    "Low"    : 0.3
}

LEVEL_WEIGHTS = {
    "Beginner"     : 0.3,
    "Intermediate" : 0.6,
    "Advanced"     : 1.0
}


# ── SKILL CLUSTERS ────────────────────────────────────────────────────────────

SKILL_CLUSTERS = {
    "Frontend"          : ["React", "HTML", "CSS", "JavaScript", "TypeScript",
                           "Vue.js", "Next.js", "Angular", "Tailwind CSS",
                           "Bootstrap", "Redux", "Webpack"],
    "Backend"           : ["Node.js", "Django", "Flask", "FastAPI", "Express",
                           "Spring Boot", "REST APIs", "GraphQL", "PHP",
                           "Ruby on Rails", "Java", "Go", "Rust"],
    "Database"          : ["SQL", "PostgreSQL", "MongoDB", "MySQL", "Redis",
                           "SQLite", "Cassandra", "Firebase", "Elasticsearch"],
    "Machine Learning"  : ["Machine Learning", "Deep Learning", "TensorFlow",
                           "PyTorch", "Scikit-learn", "Keras", "NLP",
                           "Computer Vision", "Pandas", "NumPy",
                           "Artificial Intelligence", "Data Analysis"],
    "DevOps & Cloud"    : ["Docker", "Kubernetes", "AWS", "GCP", "Azure",
                           "CI/CD", "Jenkins", "Terraform", "Linux",
                           "Amazon Web Services", "Google Cloud Platform"],
    "Programming"       : ["Python", "Java", "C++", "C#", "JavaScript",
                           "TypeScript", "Go", "Rust", "Kotlin", "Swift",
                           "R", "Scala", "MATLAB"],
    "Data Engineering"  : ["Spark", "Hadoop", "Kafka", "Airflow", "ETL",
                           "Data Pipelines", "BigQuery", "Snowflake",
                           "Data Warehousing"],
    "Mobile"            : ["Flutter", "React Native", "iOS", "Android",
                           "Swift", "Kotlin", "Xamarin"],
    "Tools & Practices" : ["Git", "GitHub", "Agile", "Scrum", "JIRA",
                           "Unit Testing", "TDD", "REST APIs",
                           "Object Oriented Programming",
                           "Data Structures and Algorithms"]
}


def cluster_skills(skills: list) -> dict:
    clustered   = defaultdict(list)
    unclustered = []
    for skill in skills:
        skill_lower = skill.lower()
        placed      = False
        for cluster, cluster_skills_list in SKILL_CLUSTERS.items():
            for cs in cluster_skills_list:
                if skill_lower == cs.lower() or skill_lower in cs.lower():
                    clustered[cluster].append(skill)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            unclustered.append(skill)
    if unclustered:
        clustered["Other"] = unclustered
    return dict(clustered)


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — DYNAMIC SKILL GRAPH VIA LLM
# ═════════════════════════════════════════════════════════════════════════════

# Static fallback graph (used if LLM fails)
STATIC_SKILL_GRAPH = {
    "React"            : ["JavaScript"],
    "Next.js"          : ["React"],
    "Redux"            : ["React"],
    "Tailwind CSS"     : ["CSS"],
    "CSS"              : ["HTML"],
    "JavaScript"       : [],
    "HTML"             : [],
    "TypeScript"       : ["JavaScript"],
    "Vue.js"           : ["JavaScript"],
    "Angular"          : ["JavaScript", "TypeScript"],
    "Django"           : ["Python"],
    "Flask"            : ["Python"],
    "FastAPI"          : ["Python"],
    "Node.js"          : ["JavaScript"],
    "Express"          : ["Node.js"],
    "Machine Learning" : ["Python"],
    "Deep Learning"    : ["Machine Learning"],
    "TensorFlow"       : ["Deep Learning", "Python"],
    "PyTorch"          : ["Deep Learning", "Python"],
    "Scikit-learn"     : ["Machine Learning", "Python"],
    "Pandas"           : ["Python"],
    "NumPy"            : ["Python"],
    "NLP"              : ["Machine Learning"],
    "Computer Vision"  : ["Deep Learning"],
    "Kubernetes"       : ["Docker"],
    "Docker"           : [],
    "CI/CD"            : ["Git"],
    "Git"              : [],
    "PostgreSQL"       : ["SQL"],
    "MySQL"            : ["SQL"],
    "SQL"              : [],
    "MongoDB"          : [],
    "Amazon Web Services"   : [],
    "Google Cloud Platform" : [],
    "Azure"            : [],
}


def build_dynamic_skill_graph(gap_skills: list, role_title: str) -> dict:
    """
    Uses Llama 3 via Groq to infer skill dependencies dynamically
    based on the actual role and gap skills.
    Falls back to static graph for any skill not returned by LLM.
    """
    if not gap_skills:
        return STATIC_SKILL_GRAPH

    log.info("Building dynamic skill graph via Groq LLM...")

    prompt = f"""
You are an expert technical curriculum designer.

For the role: {role_title}
The candidate needs to learn these skills: {gap_skills}

For each skill, list what prerequisite skills must be learned FIRST.
Only include prerequisites from the same skill list above.
If a skill has no prerequisites, return an empty list.

Return STRICT JSON only:
{{
  "skill_dependencies": {{
    "React": ["JavaScript"],
    "Machine Learning": ["Python"],
    "Docker": [],
    "TensorFlow": ["Python", "Machine Learning"]
  }}
}}
"""

    result = call_groq(prompt, max_tokens=800)
    dynamic_graph = result.get("skill_dependencies", {})

    # merge dynamic graph with static fallback
    merged = dict(STATIC_SKILL_GRAPH)
    for skill, deps in dynamic_graph.items():
        if isinstance(deps, list):
            merged[normalize_skill(skill)] = [
                normalize_skill(d) for d in deps
            ]

    # repair dependency graph
    merged = repair_skill_graph(merged)
    log.info(f"Dynamic skill graph built: {len(dynamic_graph)} skills mapped.")

    return merged



def repair_skill_graph(graph: dict) -> dict:
    """
    Fix circular dependencies and invalid prerequisites
    generated by LLM.
    """

    repaired = {}

    for skill, deps in graph.items():

        # remove self dependency
        deps = [d for d in deps if d != skill]

        valid = []

        for d in deps:
            # remove circular dependency
            if skill not in graph.get(d, []):
                valid.append(d)

        repaired[skill] = valid

    # ensure at least one root node exists
    roots = [s for s, d in repaired.items() if not d]

    if not roots and repaired:
        first = list(repaired.keys())[0]
        repaired[first] = []

    return repaired


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — LLM-BASED PROGRESS TRACKING
# ═════════════════════════════════════════════════════════════════════════════

def llm_progress_tracking(gap_skills: list, resume_text: str) -> dict:
    """
    Uses Llama 3 to estimate exposure level for each gap skill
    based on project descriptions, years of experience, and context.
    Much more accurate than simple keyword counting.
    """
    if not gap_skills:
        return {}

    log.info("Running LLM-based progress tracking...")

    prompt = f"""
You are an expert resume analyst.

Analyze this resume and estimate the candidate's exposure level
for each of the following skills they need to learn:
{gap_skills}

Resume text:
\"\"\"{resume_text[:2000]}\"\"\"

For each skill, estimate exposure from 0.0 to 1.0:
- 0.0  = never mentioned, complete beginner
- 0.3  = lightly mentioned or related context found
- 0.5  = moderate exposure, some related work detected
- 0.7  = strong indirect exposure, almost qualified
- 1.0  = fully covered (should not be in gap list)

Return STRICT JSON only:
{{
  "progress": {{
    "Docker":           {{"score": 0.3, "reason": "Used in deployment context"}},
    "Machine Learning": {{"score": 0.0, "reason": "Not mentioned anywhere"}},
    "React":            {{"score": 0.5, "reason": "Built frontend project"}}
  }}
}}
"""

    result   = call_groq(prompt, max_tokens=1000)
    raw_prog = result.get("progress", {})

    progress = {}
    for skill in gap_skills:
        data            = raw_prog.get(skill, {})
        progress[skill] = float(data.get("score", 0.0))

    log.info(f"LLM progress tracking complete for {len(progress)} skills.")
    return progress


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — TIME-BASED ADAPTATION
# ═════════════════════════════════════════════════════════════════════════════

# Base learning time estimates in days (for average learner)
BASE_LEARNING_DAYS = {
    "HTML"              : 3,
    "CSS"               : 5,
    "JavaScript"        : 14,
    "TypeScript"        : 7,
    "React"             : 10,
    "Next.js"           : 7,
    "Vue.js"            : 10,
    "Angular"           : 14,
    "Node.js"           : 10,
    "Express"           : 5,
    "Django"            : 10,
    "Flask"             : 7,
    "FastAPI"           : 5,
    "Python"            : 14,
    "SQL"               : 7,
    "PostgreSQL"        : 5,
    "MongoDB"           : 5,
    "Git"               : 3,
    "Docker"            : 5,
    "Kubernetes"        : 14,
    "Machine Learning"  : 21,
    "Deep Learning"     : 21,
    "TensorFlow"        : 14,
    "PyTorch"           : 14,
    "Scikit-learn"      : 10,
    "Pandas"            : 7,
    "NumPy"             : 5,
    "NLP"               : 14,
    "Computer Vision"   : 14,
    "AWS"               : 14,
    "Amazon Web Services": 14,
    "GCP"               : 14,
    "Google Cloud Platform": 14,
    "Azure"             : 14,
    "CI/CD"             : 7,
    "REST APIs"         : 5,
    "Data Analysis"     : 10,
}

DIFFICULTY_LEVELS = {
    "Beginner"     : 1.0,   # no adjustment
    "Intermediate" : 1.5,   # 50% longer
    "Advanced"     : 2.0,   # double the time
}

SPEED_MULTIPLIERS = {
    "slow"   : 1.5,
    "average": 1.0,
    "fast"   : 0.7,
}


def estimate_learning_time(skill: str, current_level: str,
                            required_level: str,
                            learning_speed: str = "average") -> dict:
    """
    Estimates how many days it takes to learn a skill
    based on current level, required level, and learning speed.
    """
    base_days  = BASE_LEARNING_DAYS.get(skill, 7)     # default 7 days
    diff_mult  = DIFFICULTY_LEVELS.get(required_level, 1.0)
    speed_mult = SPEED_MULTIPLIERS.get(learning_speed, 1.0)

    # reduce time if candidate has partial knowledge
    level_reduction = LEVEL_WEIGHTS.get(current_level, 0.0)
    adjusted_days   = base_days * diff_mult * speed_mult * (1 - level_reduction * 0.5)
    adjusted_days   = max(1, round(adjusted_days))

    return {
        "skill"         : skill,
        "estimated_days": adjusted_days,
        "current_level" : current_level,
        "target_level"  : required_level,
    }


def build_weekly_roadmap(roadmap: list, learning_speed: str = "average") -> list:
    """
    Takes the adaptive roadmap and organizes skills into weekly chunks.
    Respects dependency order — locked skills moved to later weeks.
    Returns a list of weeks, each containing skills to learn that week.
    """
    roadmap = sorted(roadmap, key=lambda x: x["priority"], reverse=True)

    weeks      = []
    week       = []
    week_days  = 0
    max_days   = 5     # working days per week

    for node in roadmap:
        if node["status"] == "Locked":
            continue

        skill         = node["skill"]
        current_level = "Beginner"
        target_level  = "Intermediate" if node["importance"] != "High" else "Advanced"

        time_est  = estimate_learning_time(
            skill, current_level, target_level, learning_speed
        )
        days      = time_est["estimated_days"]

        if week_days + days > max_days and week:
            weeks.append(week)
            week      = []
            week_days = 0

        week.append({
            "skill"         : skill,
            "days"          : days,
            "importance"    : node["importance"],
            "task"          : node["task"],
            "estimated_days": days,
        })
        week_days += days

    if week:
        weeks.append(week)

    return weeks


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4 — FEEDBACK LOOP (RE-EVALUATION)
# ═════════════════════════════════════════════════════════════════════════════

def re_evaluate(resume_path: str, jd_input: str,
                previous_result: dict = None) -> dict:
    """
    Feedback loop — re-runs the full pipeline on an updated resume.
    Compares new results with previous to show improvement.
    Use this when: candidate updates their resume after learning skills.
    """
    log.info("Running feedback loop re-evaluation...")

    new_result = run_full_pipeline(resume_path, jd_input)

    if "error" in new_result:
        return new_result

    if previous_result and "error" not in previous_result:
        prev_score = previous_result["advanced_score"]["final_score"]
        new_score  = new_result["advanced_score"]["final_score"]
        improvement = round(new_score - prev_score, 1)

        prev_gaps  = set(previous_result["gap_analysis"]["gap_skills"])
        new_gaps   = set(new_result["gap_analysis"]["gap_skills"])
        closed_gaps = list(prev_gaps - new_gaps)

        new_result["feedback"] = {
            "previous_score" : prev_score,
            "new_score"      : new_score,
            "improvement"    : improvement,
            "closed_gaps"    : closed_gaps,
            "remaining_gaps" : list(new_gaps),
            "message"        : (
                f"Score improved by {improvement}% — "
                f"{len(closed_gaps)} skill(s) mastered!"
                if improvement > 0
                else "Keep learning — re-upload your resume after improving skills."
            )
        }

    return new_result


# ═════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5 — PERSONALIZATION LAYER
# ═════════════════════════════════════════════════════════════════════════════

def collect_user_profile() -> dict:
    """
    Collects user preferences to personalize the learning roadmap.
    Called once at startup before the pipeline runs.
    """
    print("\n" + "="*60)
    print("PERSONALIZATION — Tell us about yourself")
    print("="*60)

    print("\nLearning speed:")
    print("  1 = Slow (more time per topic)")
    print("  2 = Average")
    print("  3 = Fast (quick learner)")
    speed_input = input("Your learning speed (1/2/3): ").strip()
    speed_map   = {"1": "slow", "2": "average", "3": "fast"}
    learning_speed = speed_map.get(speed_input, "average")

    print("\nAvailable hours per day for learning:")
    hours_input = input("Hours per day (e.g. 2): ").strip()
    try:
        hours_per_day = float(hours_input)
    except ValueError:
        hours_per_day = 2.0

    print("\nCareer goal (optional — press Enter to skip):")
    career_goal = input("Your career goal: ").strip() or "Get hired for target role"

    print("\nPreferred learning style:")
    print("  1 = Videos / tutorials")
    print("  2 = Documentation / reading")
    print("  3 = Projects / hands-on")
    style_input = input("Your preference (1/2/3): ").strip()
    style_map   = {
        "1": "video",
        "2": "reading",
        "3": "hands-on"
    }
    learning_style = style_map.get(style_input, "hands-on")

    return {
        "learning_speed" : learning_speed,
        "hours_per_day"  : hours_per_day,
        "career_goal"    : career_goal,
        "learning_style" : learning_style,
    }


def personalize_roadmap(roadmap: list, user_profile: dict) -> list:
    """
    Adjusts roadmap tasks and time estimates based on user profile.
    - Adds style-specific learning resources per skill
    - Adjusts estimated days by hours_per_day
    - Re-prioritizes based on career goal
    """
    learning_speed = user_profile.get("learning_speed", "average")
    hours_per_day  = user_profile.get("hours_per_day", 2.0)
    learning_style = user_profile.get("learning_style", "hands-on")

    style_hints = {
        "video"    : "Watch YouTube tutorials and Udemy courses",
        "reading"  : "Read official documentation and books",
        "hands-on" : "Build a mini project using this skill",
    }

    for node in roadmap:
        if node["status"] == "Locked":
            continue

        skill  = node["skill"]
        target = "Advanced" if node["importance"] == "High" else "Intermediate"

        time_est = estimate_learning_time(
            skill, "Beginner", target, learning_speed
        )

        # adjust days by available hours (standard = 2 hrs/day)
        adjusted_days = round(time_est["estimated_days"] * (2.0 / max(hours_per_day, 0.5)))
        adjusted_days = max(1, adjusted_days)

        node["estimated_days"]  = adjusted_days
        node["learning_tip"]    = style_hints.get(learning_style, "Practice daily")
        node["task"]            = (
            f"Learn {skill} in ~{adjusted_days} day(s) — "
            f"{style_hints.get(learning_style, 'Practice daily')}"
        )

    return roadmap


def generate_adaptive_roadmap(gap_skills: list, resume_skills: list,
                               jd_required: list   = None,
                               importance_map: dict = None,
                               skill_graph: dict   = None) -> list:
    """
    Generates a prioritized adaptive learning roadmap.
    Uses dynamic skill graph if provided, else static fallback.
    """
    if importance_map is None:
        importance_map = {}
    if skill_graph is None:
        skill_graph = STATIC_SKILL_GRAPH

    roadmap = []

    for skill in gap_skills:
        deps       = skill_graph.get(skill, [])
        importance = importance_map.get(skill, "Medium")
        depth      = len(deps)
        gap_score  = 1.0

        priority = (
            IMPORTANCE_WEIGHTS.get(importance, 0.6) * 0.5 +
            depth * 0.2 +
            gap_score * 0.3
        )

        roadmap.append({
            "skill"         : skill,
            "depends_on"    : deps,
            "importance"    : importance,
            "priority"      : round(priority, 2),
            "status"        : "Start",
            "progress"      : 0.0,
            "estimated_days": BASE_LEARNING_DAYS.get(skill, 7),
            "task"          : f"Learn {skill} fundamentals and build a project",
            "resources"     : [],    # populated by attach_resources() after update
        })

    return sorted(roadmap, key=lambda x: x["priority"], reverse=True)


def adaptive_update(roadmap: list, progress: dict) -> list:
    completed_skills = []

    for node in roadmap:
        skill = node["skill"]
        deps  = node["depends_on"]

        unmet = [d for d in deps if d not in completed_skills]
        if unmet:
            node["status"]    = "Locked"
            node["locked_by"] = unmet
            continue

        prog             = progress.get(skill, 0.0)
        node["progress"] = prog

        if prog >= 0.9:
            node["status"] = "Completed"
        elif prog >= 0.4:
            node["status"] = "In Progress"
        else:
            node["status"] = "Start"

    # ensure at least one skill is available to start
# FIXED
        if roadmap and not any(n["status"] == "Start" for n in roadmap):
            candidate = min(
                roadmap,
                key=lambda x: len(x["depends_on"])
            )
            candidate["status"] = "Start"    

        return roadmap

def adaptive_priority_update(roadmap: list) -> list:
    """
    Original adaptive logic:
    Increase priority of unlocked skills
    and penalize blocked skills.
    """

    for node in roadmap:

        deps = node.get("depends_on", [])

        if not deps:
            node["priority"] += 0.5

        if node["status"] == "Locked":
            node["priority"] -= 0.3

    return sorted(roadmap, key=lambda x: x["priority"], reverse=True)


def get_next_best_skill(roadmap: list) -> dict:
    candidates = [
        n for n in roadmap
        if n["status"] not in ["Completed", "Locked"]
    ]
    if not candidates:
        # fallback: pick lowest dependency skill
        unlocked = [n for n in roadmap if n["status"] != "Completed"]

        if unlocked:
           return min(unlocked, key=lambda x: len(x["depends_on"]))
        
        return None
    return max(candidates, key=lambda x: x["priority"])


def attach_resources(roadmap: list) -> list:
    """
    Fetches Groq-powered learning resources for every non-locked skill.
    Attaches a resources list to each roadmap node.
    Skips Locked nodes — no point fetching resources for blocked skills.
    """
    log.info("Fetching Groq learning resources for roadmap skills...")
    for node in roadmap:
        if node["status"] != "Locked":
            node["resources"] = generate_resources(node["skill"])
        else:
            node["resources"] = []
    return roadmap


def compute_learning_score(progress: dict) -> float:

    if not progress:
        return 0.0

    total = sum(progress.values())

    score = (total / len(progress)) * 100

    return round(score, 1)


# ── SEMANTIC SKILL MATCHING ───────────────────────────────────────────────────

_encoder = SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=512)
def semantic_skill_match(skill_a: str, skill_b: str) -> float:
    vec_a = _encoder.encode(skill_a, convert_to_tensor=True)
    vec_b = _encoder.encode(skill_b, convert_to_tensor=True)
    return float(st_util.cos_sim(vec_a, vec_b))


def precompute_embeddings(skills: list) -> dict:
    return {
        skill: _encoder.encode(skill, convert_to_tensor=True)
        for skill in skills
    }


def is_skill_covered_fast(required: str,
                           resume_embeddings: dict,
                           threshold: float = 0.75) -> bool:
    req_vec = _encoder.encode(required, convert_to_tensor=True)
    for owned_vec in resume_embeddings.values():
        if float(st_util.cos_sim(req_vec, owned_vec)) >= threshold:
            return True
    return False


def skill_match_trace(required: str, resume_embeddings: dict,
                      threshold: float = 0.75) -> dict:
    """
    Returns the best semantic match with full reasoning trace.
    Shows WHY each skill was matched or not — used for the
    Reasoning Trace section (10% of hackathon evaluation score).
    Example output:
      required_skill : "ML"
      best_match     : "Machine Learning"
      score          : 0.94
      decision       : "covered"
    """
    req_vec    = _encoder.encode(required, convert_to_tensor=True)
    best_score = 0.0
    best_match = None

    for skill, vec in resume_embeddings.items():
        score = float(st_util.cos_sim(req_vec, vec))
        if score > best_score:
            best_score = score
            best_match = skill

    covered = best_score >= threshold

    return {
        "covered": covered,
        "trace"  : {
            "required_skill": required,
            "best_match"    : best_match,
            "score"         : round(best_score, 2),
            "threshold"     : threshold,
            "decision"      : "covered" if covered else "not covered"
        }
    }


# ── RESOURCE GENERATOR (GROQ-POWERED, CACHED) ────────────────────────────────

resource_cache = {}    # never fetch the same skill twice


def generate_resources(skill: str) -> list:
    """
    Uses Groq LLM to fetch the 2 best learning resources for a skill.
    Cached in resource_cache — same skill is never fetched twice.
    Returns list of {type, name, link} dicts.
    """
    if skill in resource_cache:
        return resource_cache[skill]

    prompt = f"""
Give 2 best learning resources for the skill: {skill}

Rules:
- 1 course (YouTube, Udemy, or Coursera)
- 1 official documentation or tutorial
- Return STRICT JSON only, no explanation, no markdown

Format:
{{
  "resources": [
    {{
      "type": "course",
      "name": "Resource name here",
      "link": "https://..."
    }},
    {{
      "type": "documentation",
      "name": "Resource name here",
      "link": "https://..."
    }}
  ]
}}
"""

    result    = call_groq(prompt, max_tokens=400)
    resources = result.get("resources", [])

    # fallback if Groq returns nothing
    if not resources:
        resources = [
            {
                "type": "search",
                "name": f"{skill} tutorial",
                "link": f"https://www.google.com/search?q={skill.replace(' ', '+')}+tutorial"
            }
        ]

    resource_cache[skill] = resources
    return resources


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
# JOB DESCRIPTION — LLM BASED WITH IMPORTANCE SCORING
# ═════════════════════════════════════════════════════════════════════════════

def extract_jd_with_groq(jd_text: str) -> dict:
    prompt = f"""
You are a highly accurate ATS (Applicant Tracking System).

Extract ONLY technical skills from the job description below.
For each skill assign an importance level.

STRICT RULES:
- Do NOT include sentences or responsibilities
- Do NOT include soft skills
- Only return actual SKILLS (Python, React, SQL, Docker, etc.)
- Remove duplicates
- Normalize names (JS -> JavaScript, React.js -> React, ML -> Machine Learning)
- Importance: High = core skill, Medium = regularly used, Low = nice to know

Job Description:
{jd_text[:3000]}

Return STRICT JSON ONLY:
{{
  "role": "short job title here",
  "required_skills": [
    {{"skill": "Python",     "importance": "High"}},
    {{"skill": "SQL",        "importance": "Medium"}}
  ],
  "preferred_skills": [
    {{"skill": "TypeScript", "importance": "Medium"}}
  ]
}}
"""
    result = call_groq(prompt, max_tokens=1200)
    if not result or "required_skills" not in result:
        log.warning("Groq JD extraction returned empty — using fallback")
        return {"role": "Unknown Role", "required_skills": [], "preferred_skills": []}
    return result
def detect_input_type(jd_input: str) -> str:
    """
    Detects whether the user input is:
    - structured job description
    - simple role / career intent
    """

    text = jd_input.strip().lower()

    # short inputs are likely job titles
    if len(text.split()) <= 5:
        return "role_intent"

    # JD keywords
    jd_keywords = [
        "responsibilities",
        "requirements",
        "required skills",
        "preferred skills",
        "qualifications",
        "job description",
        "experience"
    ]

    for k in jd_keywords:
        if k in text:
            return "structured_jd"

    return "structured_jd"

def expand_role_to_jd(role_title: str) -> str:
    """
    Uses Groq LLM to expand a simple job title into
    a structured job description.
    """

    log.info(f"Expanding role intent '{role_title}' into full skill graph via LLM...")

    prompt = f"""
You are a technical hiring expert.

Convert the job title into a structured job description.

Role: {role_title}

Return STRICT JSON only:

{{
  "role": "{role_title}",
  "required_skills": [
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Git"
  ],
  "preferred_skills": [
    "TypeScript",
    "Tailwind CSS",
    "Next.js"
  ],
  "responsibilities": [
    "Build responsive web applications",
    "Integrate REST APIs",
    "Collaborate with backend teams"
  ]
}}
"""

    result = call_groq(prompt, max_tokens=800)

    if not result:
        return role_title

    skills = []

    for s in result.get("required_skills", []):
        skills.append(s)

    for s in result.get("preferred_skills", []):
        skills.append(s)

    synthetic_jd = f"""
Role: {result.get("role", role_title)}

Required Skills:
{", ".join(result.get("required_skills", []))}

Preferred Skills:
{", ".join(result.get("preferred_skills", []))}

Responsibilities:
{", ".join(result.get("responsibilities", []))}
"""

    return synthetic_jd



def process_job_description(jd_input: str) -> dict:

    jd_text      = ""
    cleaned_path = jd_input.strip().strip('"').strip("'")

    # CASE 1 — JD file
    if os.path.exists(cleaned_path):

        log.info("JD input detected as file — extracting text...")
        jd_text = extract_text_from_file(cleaned_path)

    else:

        text = jd_input.strip()

        # CASE 2 — Detect if this is only a job title / career intent
        if len(text.split()) <= 5:

            log.info("Detected career intent — expanding role using LLM...")
            jd_text = expand_role_to_jd(text)

        else:

            log.info("JD input detected as structured job description...")
            jd_text = clean_text(text)

    # Safety check
    if not jd_text:
        return {"error": "Job description is empty"}

    # ── Extract skills using Groq ─────────────────────────────
    log.info("Extracting JD skills via Groq LLM...")
    jd_llm = extract_jd_with_groq(jd_text)

    raw_required  = jd_llm.get("required_skills", [])
    raw_preferred = jd_llm.get("preferred_skills", [])

    # ── Importance mapping ───────────────────────────────────
    importance_map = {}

    for item in raw_required + raw_preferred:

        if isinstance(item, dict):

            skill      = normalize_skill(item.get("skill", ""))
            importance = item.get("importance", "Medium")

            if skill:
                importance_map[skill] = importance

    # ── Normalize skills ─────────────────────────────────────
    required_skills  = normalize_skill_list(raw_required)
    preferred_skills = normalize_skill_list(raw_preferred)

    role_title = jd_llm.get("role", "Unknown Role").strip() or "Unknown Role"

    return {
        "role_title"      : role_title,
        "raw_jd_text"     : jd_text,
        "required_skills" : required_skills,
        "preferred_skills": preferred_skills,
        "importance_map"  : importance_map,
    }

# ═════════════════════════════════════════════════════════════════════════════
# GAP ANALYSIS WITH IMPORTANCE + LEVEL GAP
# ═════════════════════════════════════════════════════════════════════════════

def perform_gap_analysis(resume_skills: list,
                          required_skills: list,
                          preferred_skills: list,
                          importance_map: dict     = None,
                          groq_resume_result: dict = None) -> dict:
    if importance_map is None:
        importance_map = {}

    resume_embeddings = precompute_embeddings(resume_skills)

    # use skill_match_trace for every required skill so we get full reasoning
    overlapping     = []
    gap_skills      = []
    reasoning_trace = {}

    for skill in required_skills:
        trace_result              = skill_match_trace(skill, resume_embeddings)
        reasoning_trace[skill]    = trace_result["trace"]
        if trace_result["covered"]:
            overlapping.append(skill)
        else:
            gap_skills.append(skill)

    preferred_gap = [s for s in preferred_skills
                     if not is_skill_covered_fast(s, resume_embeddings)]

    total_weight   = sum(
        IMPORTANCE_WEIGHTS.get(importance_map.get(s, "Medium"), 0.6)
        for s in required_skills
    )
    matched_weight = sum(
        IMPORTANCE_WEIGHTS.get(importance_map.get(s, "Medium"), 0.6)
        for s in overlapping
    )

    level_gaps    = []
    level_penalty = 0.0

    if groq_resume_result:
        resume_levels = groq_resume_result.get("skills", {})
        for skill in overlapping:
            resume_level   = resume_levels.get(skill, {}).get("level", "Intermediate")
            importance     = importance_map.get(skill, "Medium")
            required_level = (
                "Advanced"     if importance == "High"   else
                "Intermediate" if importance == "Medium" else
                "Beginner"
            )
            resume_val   = LEVEL_WEIGHTS.get(resume_level,   0.6)
            required_val = LEVEL_WEIGHTS.get(required_level, 0.6)
            if resume_val < required_val:
                level_gaps.append({
                    "skill"         : skill,
                    "current_level" : resume_level,
                    "required_level": required_level,
                    "gap"           : required_level
                })
                level_penalty += ((required_val - resume_val) *
                                   IMPORTANCE_WEIGHTS.get(importance, 0.6))

    total_required  = len(required_skills)
    total_preferred = len(preferred_skills)
    matched_req     = len(overlapping)
    matched_pref    = total_preferred - len(preferred_gap)

    required_score  = matched_weight / max(total_weight, 1)
    preferred_score = matched_pref   / max(total_preferred, 1)

    level_penalty_normalized = min(level_penalty / max(total_weight, 1), 0.3)

    final_score = round(
        ((required_score * 0.8 + preferred_score * 0.2) - level_penalty_normalized)
        * 100, 1
    )
    final_score = max(0.0, final_score)
    gap_score   = round(100 - final_score, 1)

    return {
        "overlapping_skills": overlapping,
        "gap_skills"        : gap_skills,
        "preferred_gap"     : preferred_gap,
        "level_gaps"        : level_gaps,
        "gap_clusters"      : cluster_skills(gap_skills),
        "pref_clusters"     : cluster_skills(preferred_gap),
        "matched_required"  : matched_req,
        "total_required"    : total_required,
        "matched_preferred" : matched_pref,
        "total_preferred"   : total_preferred,
        "match_score"       : final_score,
        "gap_score"         : gap_score,
        "reasoning_trace"   : reasoning_trace,   # ← full match logic per skill
    }


# ═════════════════════════════════════════════════════════════════════════════
# ADVANCED ATS SCORING
# ═════════════════════════════════════════════════════════════════════════════

def compute_advanced_score(resume_skills: list,
                            required_skills: list,
                            preferred_skills: list,
                            groq_result: dict,
                            importance_map: dict = None) -> dict:
    if importance_map is None:
        importance_map = {}

    resume_embeddings = precompute_embeddings(resume_skills)
    required_match    = [s for s in required_skills
                         if is_skill_covered_fast(s, resume_embeddings)]
    preferred_match   = [s for s in preferred_skills
                         if is_skill_covered_fast(s, resume_embeddings)]

    total_weight   = sum(
        IMPORTANCE_WEIGHTS.get(importance_map.get(s, "Medium"), 0.6)
        for s in required_skills
    )
    matched_weight = sum(
        IMPORTANCE_WEIGHTS.get(importance_map.get(s, "Medium"), 0.6)
        for s in required_match
    )
    required_score  = matched_weight / max(total_weight, 1)
    preferred_score = len(preferred_match) / max(len(preferred_skills), 1)

    strength_scores = []
    skills_data     = groq_result.get("skills", {})
    for skill in required_match:
        data       = skills_data.get(skill, {})
        level      = data.get("level", "Beginner")
        importance = importance_map.get(skill, "Medium")
        strength   = (LEVEL_WEIGHTS.get(level, 0.3) *
                      IMPORTANCE_WEIGHTS.get(importance, 0.6))
        strength_scores.append(strength)

    strength_score = (sum(strength_scores) / len(strength_scores)
                      if strength_scores else 0.0)

    final_pct = round((required_score * 0.5 +
                       preferred_score * 0.2 +
                       strength_score  * 0.3) * 100, 1)

    if final_pct >= 75:   readiness = "Job Ready"
    elif final_pct >= 50: readiness = "Almost Ready"
    elif final_pct >= 25: readiness = "Needs Improvement"
    else:                 readiness = "Significant Gaps Found"

    return {
        "required_score" : round(required_score  * 100, 1),
        "preferred_score": round(preferred_score * 100, 1),
        "strength_score" : round(strength_score  * 100, 1),
        "final_score"    : final_pct,
        "readiness"      : readiness,
    }


# ── PROGRESS BAR ─────────────────────────────────────────────────────────────

def print_progress_bar(label: str, score: float):
    bars  = int(score // 10)
    empty = 10 - bars
    print(f"  {label}: {'█' * bars}{'░' * empty} {score}%")


# ═════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

def prepare_chunk_prompt(chunk: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
You are an expert resume analyst. This is part {chunk_index + 1} of {total_chunks} of a resume.

Extract ALL technical skills from this section.
For each skill, classify the level and provide reasoning.

Resume Text (chunk {chunk_index + 1}/{total_chunks}):
{chunk}

Return STRICT JSON only:
{{
  "skills": {{
    "Python": {{"level": "Intermediate", "reason": "Used in 2 projects"}}
  }}
}}
"""


def prepare_llm_input(sections: dict, skill_frequency: dict,
                      skill_scores: dict) -> str:
    freq_hint = ""
    if skill_frequency:
        top        = sorted(skill_frequency.items(),
                            key=lambda x: x[1], reverse=True)[:10]
        freq_hint  = "Skill mention frequency:\n"
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
- Extract only real technical skills.
- Classify each as: Beginner, Intermediate, or Advanced.
- Base classification on: years, project depth, job titles.

Return STRICT JSON only:
{{
  "skills": {{
    "Python": {{"level": "Intermediate", "reason": "1 year, used in projects"}},
    "SQL":    {{"level": "Beginner",     "reason": "Basic SELECT queries only"}}
  }}
}}
"""


def prepare_gap_analysis_prompt(resume_skills: list, gap_analysis: dict,
                                 role_title: str, resume_text: str,
                                 importance_map: dict = None) -> str:
    if importance_map is None:
        importance_map = {}

    gap_with_importance = [
        f"{s} (importance: {importance_map.get(s, 'Medium')})"
        for s in gap_analysis['gap_skills']
    ]

    return f"""
You are an expert career coach and skill assessor.

Candidate is targeting the role: {role_title}
Resume: \"\"\"{resume_text[:2000]}\"\"\"

Overlapping skills: {gap_analysis['overlapping_skills']}
Level gaps: {gap_analysis.get('level_gaps', [])}
Missing required skills: {gap_with_importance}
Preferred skills missing: {gap_analysis['preferred_gap']}

Task:
1. Diagnose level for overlapping skills.
2. Explain level improvement needed for level gaps.
3. Provide learning recommendation for each gap skill.
4. Explain competitive advantage for each preferred skill.

Return STRICT JSON only:
{{
  "overlapping_skills": {{
    "Python": {{
      "diagnosed_level": "Beginner", "required_level": "Advanced",
      "needs_upskilling": true, "evidence": "1 year basic use",
      "focus": "Learn ML libraries and OOP"
    }}
  }},
  "level_gap_skills": {{
    "SQL": {{
      "current_level": "Beginner", "required_level": "Intermediate",
      "recommended_path": "Learn JOINs, window functions"
    }}
  }},
  "gap_skills": {{
    "Docker": {{
      "priority": "High", "reason": "Required for deployment",
      "recommended_path": "Learn Docker basics and docker-compose"
    }}
  }},
  "preferred_gap_skills": {{
    "TypeScript": {{
      "priority": "Medium", "reason": "Adds type safety",
      "recommended_path": "Learn TS types and interfaces"
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
    log.info(f"Resume analysis: {len(result.get('skills', {}))} skills.")
    return result


def call_groq_chunk_analysis(chunks: list) -> dict:
    log.info(f"Calling Groq for {len(chunks)} chunk(s)...")
    results = []
    for i, chunk in enumerate(chunks):
        log.info(f"Processing chunk {i+1}/{len(chunks)}...")
        results.append(call_groq(
            prepare_chunk_prompt(chunk, i, len(chunks)), max_tokens=1024
        ))
    return merge_llm_results(results)


def call_groq_gap_analysis(resume_skills: list, gap_analysis: dict,
                            role_title: str, resume_text: str,
                            importance_map: dict = None) -> dict:
    log.info("Calling Groq for gap analysis...")
    prompt = prepare_gap_analysis_prompt(
        resume_skills, gap_analysis, role_title,
        resume_text, importance_map
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
            "lines"     : len(lines),
            "paragraphs": len(paragraphs),
            "chars"     : len(raw_text),
            "chunks"    : len(chunks)
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(resume_path: str, jd_input: str,
                      user_profile: dict = None) -> dict:
    log.info("=" * 60)
    log.info("Starting full pipeline...")
    log.info("=" * 60)

    if user_profile is None:
        user_profile = {
            "learning_speed": "average",
            "hours_per_day" : 2.0,
            "career_goal"   : "Get hired for target role",
            "learning_style": "hands-on",
        }

    # step 1 — resume
    log.info("Step 1: Processing resume...")
    resume_result = process_resume(resume_path)
    if "error" in resume_result:
        return {"error": f"Resume error: {resume_result['error']}"}

    # step 2 — JD via Groq LLM
    log.info("Step 2: Processing job description via Groq LLM...")
    jd_result = process_job_description(jd_input)
    if "error" in jd_result:
        return {"error": f"JD error: {jd_result['error']}"}

    importance_map = jd_result.get("importance_map", {})

    # step 3 — groq resume skill level analysis
    log.info("Step 3: Groq resume skill level analysis...")
    if resume_result["stats"]["chunks"] > 1:
        groq_resume_result = call_groq_chunk_analysis(resume_result["chunks"])
    else:
        groq_resume_result = call_groq_resume_analysis(
            sections        = resume_result["sections"],
            skill_frequency = resume_result["skill_frequency"],
            skill_scores    = resume_result["skill_scores"]
        )

    # step 4 — gap analysis
    log.info("Step 4: Performing gap analysis...")
    gap_analysis = perform_gap_analysis(
        resume_skills      = resume_result["normalized_skills"],
        required_skills    = jd_result["required_skills"],
        preferred_skills   = jd_result["preferred_skills"],
        importance_map     = importance_map,
        groq_resume_result = groq_resume_result
    )

    # step 5 — groq gap analysis + level diagnosis
    log.info("Step 5: Groq gap analysis + level diagnosis...")
    groq_gap_result = call_groq_gap_analysis(
        resume_skills  = resume_result["normalized_skills"],
        gap_analysis   = gap_analysis,
        role_title     = jd_result["role_title"],
        resume_text    = resume_result["raw_text"],
        importance_map = importance_map
    )

    # step 6 — advanced ATS scoring
    log.info("Step 6: Computing advanced ATS score...")
    advanced_score = compute_advanced_score(
        resume_skills    = resume_result["normalized_skills"],
        required_skills  = jd_result["required_skills"],
        preferred_skills = jd_result["preferred_skills"],
        groq_result      = groq_resume_result,
        importance_map   = importance_map
    )

    # step 7 — cluster all skills
    log.info("Step 7: Clustering skills...")
    resume_clusters   = cluster_skills(resume_result["normalized_skills"])
    required_clusters = cluster_skills(jd_result["required_skills"])

    gap_skills = gap_analysis["gap_skills"]

    # step 8 — dynamic skill graph via LLM
    log.info("Step 8: Building dynamic skill graph...")
    skill_graph = build_dynamic_skill_graph(
        gap_skills  = gap_skills,
        role_title  = jd_result["role_title"]
    )

    # step 9 — LLM-based progress tracking
    log.info("Step 9: LLM-based progress tracking...")
    progress = llm_progress_tracking(
        gap_skills  = gap_skills,
        resume_text = resume_result["raw_text"]
    )

    # step 10 — generate + personalize adaptive roadmap
    log.info("Step 10: Generating personalized adaptive roadmap...")
    roadmap = generate_adaptive_roadmap(
        gap_skills     = gap_skills,
        resume_skills  = resume_result["normalized_skills"],
        jd_required    = jd_result["required_skills"],
        importance_map = importance_map,
        skill_graph    = skill_graph
    )
    roadmap         = adaptive_update(roadmap, progress)
    roadmap         = adaptive_priority_update(roadmap)
    roadmap         = personalize_roadmap(roadmap, user_profile)
    roadmap         = attach_resources(roadmap)     # ← Groq resources per skill
    weekly_roadmap  = build_weekly_roadmap(
        roadmap, user_profile.get("learning_speed", "average")
    )
    next_skill      = get_next_best_skill(roadmap)
    learning_score  = compute_learning_score(progress)

    return {
        "resume"            : resume_result,
        "jd"                : jd_result,
        "gap_analysis"      : gap_analysis,
        "groq_skill_levels" : groq_resume_result,
        "groq_gap_diagnosis": groq_gap_result,
        "advanced_score"    : advanced_score,
        "resume_clusters"   : resume_clusters,
        "required_clusters" : required_clusters,
        "skill_graph"       : skill_graph,
        "roadmap"           : roadmap,
        "weekly_roadmap"    : weekly_roadmap,
        "next_skill"        : next_skill,
        "learning_score"    : learning_score,
        "progress"          : progress,
        "user_profile"      : user_profile,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("   AI ADAPTIVE ONBOARDING ENGINE — GROQ + LLAMA 3")
    print("="*60)

    # collect personalization profile first
    user_profile = collect_user_profile()

    resume_path = input("\nEnter resume path (.pdf or .docx): ").strip()
    print("\nEnter job description:")
    print("(Type/paste text OR enter a .pdf/.docx file path)")
    jd_input    = input("JD input: ").strip()

    result = run_full_pipeline(resume_path, jd_input, user_profile)

    if "error" in result:
        print(f"\nError: {result['error']}")

    else:
        imp = result["jd"].get("importance_map", {})

        # ── resume skills ─────────────────────────────────────────
        print("\n" + "="*60)
        print("RESUME — EXTRACTED SKILLS")
        print("="*60)
        for skill in result["resume"]["normalized_skills"]:
            print(f"  - {skill}")

        # ── resume clusters ───────────────────────────────────────
        print("\n" + "="*60)
        print("RESUME — SKILL CLUSTERS")
        print("="*60)
        for cluster, skills in result["resume_clusters"].items():
            print(f"\n  [{cluster}]")
            for s in skills:
                print(f"    - {s}")

        # ── section confidence ────────────────────────────────────
        print("\n" + "="*60)
        print("RESUME — SECTION CONFIDENCE")
        print("="*60)
        for k, v in result["resume"]["confidence"].items():
            print(f"  {k}: {v}")

        # ── job description ───────────────────────────────────────
        print("\n" + "="*60)
        print("JOB DESCRIPTION — LLM EXTRACTED")
        print("="*60)
        print(f"  Role : {result['jd']['role_title']}")
        print(f"\n  Required Skills ({len(result['jd']['required_skills'])}):")
        for s in result["jd"]["required_skills"]:
            print(f"    - {s:30} [{imp.get(s, 'Medium')}]")
        print(f"\n  Preferred Skills ({len(result['jd']['preferred_skills'])}):")
        for s in result["jd"]["preferred_skills"]:
            print(f"    - {s:30} [{imp.get(s, 'Medium')}]")

        # ── required clusters ─────────────────────────────────────
        print("\n" + "="*60)
        print("JD — REQUIRED SKILL CLUSTERS")
        print("="*60)
        for cluster, skills in result["required_clusters"].items():
            print(f"\n  [{cluster}]")
            for s in skills:
                print(f"    - {s:30} [{imp.get(s, 'Medium')}]")

        # ── gap analysis ──────────────────────────────────────────
        print("\n" + "="*60)
        print("GAP ANALYSIS")
        print("="*60)
        g = result["gap_analysis"]
        print(f"\n  Required Skills Matched  : "
              f"{g['matched_required']} / {g['total_required']}")
        print(f"  Preferred Skills Matched : "
              f"{g['matched_preferred']} / {g['total_preferred']}")
        print(f"\n  Match Score : {g['match_score']}%")
        print(f"  Gap Score   : {g['gap_score']}%")
        print(f"\n  Overlapping : {g['overlapping_skills']}")
        print(f"  Missing     : {g['gap_skills']}")
        print(f"  Preferred   : {g['preferred_gap']}")

        if g.get("level_gaps"):
            print(f"\n  --- Level Gaps ---")
            for lg in g["level_gaps"]:
                print(f"    {lg['skill']:25} "
                      f"{lg['current_level']:15} -> {lg['required_level']}")

        if g.get("gap_clusters"):
            print(f"\n  --- Missing Skills by Category ---")
            for cluster, skills in g["gap_clusters"].items():
                print(f"    [{cluster}]: {', '.join(skills)}")

        # ── reasoning trace ───────────────────────────────────────
        print("\n" + "="*60)
        print("REASONING TRACE — SKILL MATCHING LOGIC")
        print("="*60)
        print("  (Shows WHY each required skill was matched or flagged as missing)")
        trace = result["gap_analysis"].get("reasoning_trace", {})
        if trace:
            for skill, data in trace.items():
                icon = "✅" if data["decision"] == "covered" else "❌"
                print(f"\n  {icon} {skill}")
                print(f"     Best Match : {data['best_match']}")
                print(f"     Score      : {data['score']}  "
                      f"(threshold: {data['threshold']})")
                print(f"     Decision   : {data['decision']}")
        else:
            print("  No reasoning trace available.")

        # ── advanced scoring ──────────────────────────────────────
        print("\n" + "="*60)
        print("ADVANCED SCORING SYSTEM (ATS-LEVEL)")
        print("="*60)
        score = result["advanced_score"]
        print()
        print_progress_bar("Core Match Score  ", score['required_score'])
        print_progress_bar("Competitive Score ", score['preferred_score'])
        print_progress_bar("Skill Strength    ", score['strength_score'])
        print()
        print_progress_bar("FINAL READINESS   ", score['final_score'])
        print(f"\n  Readiness Level : {score['readiness']}")

        # ── groq skill levels ─────────────────────────────────────
        print("\n" + "="*60)
        print("GROQ — SKILL LEVEL ANALYSIS")
        print("="*60)
        skills = result["groq_skill_levels"].get("skills", {})
        if skills:
            for skill, data in skills.items():
                print(f"\n  {skill}")
                print(f"    Level  : {data.get('level',  'N/A')}")
                print(f"    Reason : {data.get('reason', 'N/A')}")
        else:
            print("  No skill levels returned from Groq.")

        # ── groq gap diagnosis ────────────────────────────────────
        print("\n" + "="*60)
        print("GROQ — GAP DIAGNOSIS + LEARNING PATH")
        print("="*60)

        overlapping = result["groq_gap_diagnosis"].get("overlapping_skills", {})
        print("\n  --- Overlapping Skills ---")
        if overlapping:
            for skill, data in overlapping.items():
                print(f"\n  {skill}  [{imp.get(skill, 'Medium')}]")
                print(f"    Diagnosed Level  : {data.get('diagnosed_level',  'N/A')}")
                print(f"    Required Level   : {data.get('required_level',   'N/A')}")
                print(f"    Needs Upskilling : {data.get('needs_upskilling', 'N/A')}")
                print(f"    Evidence         : {data.get('evidence',         'N/A')}")
                print(f"    Focus            : {data.get('focus',            'N/A')}")
        else:
            print("  No overlapping skills.")

        level_gap_skills = result["groq_gap_diagnosis"].get("level_gap_skills", {})
        print("\n  --- Level Gap Skills ---")
        if level_gap_skills:
            for skill, data in level_gap_skills.items():
                print(f"\n  {skill}")
                print(f"    Current Level    : {data.get('current_level',    'N/A')}")
                print(f"    Required Level   : {data.get('required_level',   'N/A')}")
                print(f"    Recommended Path : {data.get('recommended_path', 'N/A')}")
        else:
            print("  No level gaps detected.")

        gap_skills_diag = result["groq_gap_diagnosis"].get("gap_skills", {})
        print("\n  --- Gap Skills ---")
        if gap_skills_diag:
            for skill, data in gap_skills_diag.items():
                print(f"\n  {skill}  [{imp.get(skill, 'Medium')}]")
                print(f"    Priority         : {data.get('priority',         'N/A')}")
                print(f"    Reason           : {data.get('reason',           'N/A')}")
                print(f"    Recommended Path : {data.get('recommended_path', 'N/A')}")
        else:
            print("  No gap skills.")

        preferred_gap_skills = result["groq_gap_diagnosis"].get("preferred_gap_skills", {})
        print("\n  --- Preferred Gap Skills ---")
        if preferred_gap_skills:
            for skill, data in preferred_gap_skills.items():
                print(f"\n  {skill}")
                print(f"    Priority         : {data.get('priority',         'N/A')}")
                print(f"    Reason           : {data.get('reason',           'N/A')}")
                print(f"    Recommended Path : {data.get('recommended_path', 'N/A')}")
        else:
            print("  No preferred gap skills.")

        # ── adaptive learning roadmap ─────────────────────────────
        print("\n" + "="*60)
        print("ADAPTIVE LEARNING ROADMAP")
        print("="*60)
        print(f"  Profile: {user_profile['learning_speed']} learner | "
              f"{user_profile['hours_per_day']}h/day | "
              f"{user_profile['learning_style']} style")

        status_icons = {
            "Completed"  : "✅",
            "In Progress": "🔄",
            "Start"      : "🔵",
            "Locked"     : "🔒"
        }

        for node in result["roadmap"]:
            icon = status_icons.get(node["status"], "•")
            print(f"\n  {icon} {node['skill']:30} "
                  f"[{node['importance']}] "
                  f"Priority: {node['priority']} | "
                  f"~{node.get('estimated_days', '?')} day(s)")
            print(f"     Status   : {node['status']}")
            print(f"     Progress : {round(node['progress']*100, 1)}%")
            if node.get("depends_on"):
                print(f"     Requires : {', '.join(node['depends_on'])}")
            if node.get("locked_by"):
                print(f"     Locked by: {', '.join(node['locked_by'])}")
            if node.get("learning_tip"):
                print(f"     Tip      : {node['learning_tip']}")
            print(f"     Task     : {node['task']}")
            if node.get("resources"):
                print(f"     Resources:")
                for r in node["resources"]:
                    rtype = r.get("type", "link")
                    rname = r.get("name", "Resource")
                    rlink = r.get("link", "#")
                    print(f"       [{rtype:13}] {rname}")
                    print(f"                     {rlink}")

        # ── weekly roadmap ────────────────────────────────────────
        print("\n" + "="*60)
        print("WEEKLY LEARNING PLAN")
        print("="*60)
        for week_num, week in enumerate(result["weekly_roadmap"], 1):
            total_days = sum(s["days"] for s in week)
            print(f"\n  Week {week_num} ({total_days} day(s)):")
            for item in week:
                print(f"    - {item['skill']:30} "
                      f"~{item['days']} day(s)  [{item['importance']}]")
                if item.get("resources"):
                    top = item["resources"][0]
                    print(f"        -> {top.get('name', '')} : {top.get('link', '')}")

        # ── next best skill ───────────────────────────────────────
        print("\n" + "-"*60)
        print("NEXT BEST SKILL TO LEARN:")
        next_s = result["next_skill"]
        if next_s:
            print(f"  -> {next_s['skill']}  "
                  f"(Priority: {next_s['priority']}, "
                  f"Importance: {next_s['importance']}, "
                  f"~{next_s.get('estimated_days', '?')} day(s))")
            print(f"     {next_s['task']}")
            if next_s.get("resources"):
                print(f"     Resources:")
                for r in next_s["resources"]:
                    print(f"       {r.get('name', '')} -> {r.get('link', '')}")
        else:
            print("  All skills completed or roadmap is empty!")

        # ── learning score ────────────────────────────────────────
        print()
        print_progress_bar("Learning Progress", result["learning_score"])

        # ── feedback loop hint ────────────────────────────────────
        print("\n" + "="*60)
        print("FEEDBACK LOOP")
        print("="*60)
        print("  After learning new skills, update your resume and re-run:")
        print("  result = re_evaluate('new_resume.pdf', jd_input, result)")
        print("  This will show your score improvement and closed gaps.")

        # ── stats ─────────────────────────────────────────────────
        print("\n" + "="*60)
        print("STATS")
        print("="*60)
        for k, v in result["resume"]["stats"].items():
            print(f"  {k}: {v}")

