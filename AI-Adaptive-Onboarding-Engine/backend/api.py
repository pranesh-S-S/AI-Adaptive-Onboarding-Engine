from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from adaptive import run_full_pipeline
from typing import Optional
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None),       # raw text (optional)
    jd_file: Optional[UploadFile] = File(None)
):
    print("🔥 REQUEST RECEIVED")

    file_path = os.path.join(UPLOAD_FOLDER, resume.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    print("📄 FILE SAVED")

    if jd_file and jd_file.filename:
        jd_path = os.path.join(UPLOAD_FOLDER, jd_file.filename)
        with open(jd_path, "wb") as buffer:
            shutil.copyfileobj(jd_file.file, buffer)
        jd_input = jd_path          # pass file path → backend handles extraction
        print(f"📄 JD FILE SAVED: {jd_file.filename}")
    elif job_description and job_description.strip():
        jd_input = job_description  # pass raw text directly
        print("📝 JD TEXT RECEIVED")
    else:
        return {"error": "Please provide a job description — either as text or as a file."}
    
    result = run_full_pipeline(file_path, jd_input)

    print("✅ PIPELINE DONE")

    return result

