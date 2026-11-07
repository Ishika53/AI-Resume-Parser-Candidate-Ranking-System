
import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document


# --------------------------------------------------
# 1. INITIAL SETUP
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------
# 2. JOB DESCRIPTION MODEL
# --------------------------------------------------

class JobD(BaseModel):
    role: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


jobd_schema = JobD.model_json_schema()


# --------------------------------------------------
# 3. JOB DESCRIPTION EXTRACTION
# --------------------------------------------------

system_prompt = f"""
You are an expert job description analyzer.

Extract the important information from the given job description.

Return the result as valid JSON matching this schema:

{json.dumps(jobd_schema)}

Instructions:
- Extract the job role.
- Identify required and preferred skills.
- Extract minimum experience if mentioned.
- Extract education requirements.
- Extract job responsibilities.
- If information is not available, use an empty list or null.
- Return only valid JSON.
"""


def extract_job_description(job_description):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": job_description
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    data = json.loads(response_text)

    job = JobD(**data)

    return job


# --------------------------------------------------
# 4. RESUME DATA MODELS
# --------------------------------------------------

class Experience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    score: float
    details: dict


# --------------------------------------------------
# 5. RESUME PARSING
# --------------------------------------------------

resume_schema = Resume.model_json_schema()

resume_system_prompt = f"""
You are an expert resume parser.

Extract information from the given resume.

Return the result as valid JSON matching this schema:

{json.dumps(resume_schema)}

Instructions:
- Extract the candidate's name, email and phone number.
- Extract all technical and professional skills.
- Extract education details.
- Extract work experience, including company, role, duration and responsibilities.
- Extract projects and certifications.
- Do not invent information.
- If information is missing, use an empty string or empty list.
- Return only valid JSON.
"""


def parse_resume(resume_text):

    messages = [
        {
            "role": "system",
            "content": resume_system_prompt
        },
        {
            "role": "user",
            "content": resume_text
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    data = json.loads(response_text)

    resume = Resume(**data)

    return resume


# --------------------------------------------------
# 6. FINAL RESUME SCORING
# --------------------------------------------------

def final_score(job, resume):

    system_prompt = """
    You are an expert Applicant Tracking System (ATS).

    Compare the candidate's resume with the job description.

    Evaluate:
    1. Matching skills.
    2. Missing skills.
    3. Relevant work experience.
    4. Education suitability.
    5. Project relevance.

    Give a score between 0 and 100.

    Return only valid JSON in this format:

    {
        "score": 85,
        "details": {
            "matching_skills": [],
            "missing_skills": [],
            "experience_analysis": "",
            "education_analysis": "",
            "project_analysis": "",
            "verdict": ""
        }
    }
    """

    user_prompt = f"""
    JOB DESCRIPTION:

    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:

    {resume.model_dump_json(indent=2)}

    Compare the resume with the job description and provide the score.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    data = json.loads(response_text)

    result = MatchResult(**data)

    return result


# --------------------------------------------------
# 7. READ PDF FILE
# --------------------------------------------------

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text = text + page_text + "\n"

    return text


# --------------------------------------------------
# 8. READ DOCX FILE
# --------------------------------------------------

def read_docx(file_path):

    document = Document(file_path)

    text = ""

    # Read paragraphs
    for paragraph in document.paragraphs:
        text = text + paragraph.text + "\n"

    # Read tables
    for table in document.tables:

        for row in table.rows:

            for cell in row.cells:
                text = text + cell.text + " "

            text = text + "\n"

    return text


# --------------------------------------------------
# 9. READ RESUME BASED ON FILE TYPE
# --------------------------------------------------

def read_resume(file_path):

    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)

    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)

    else:
        raise ValueError("Unsupported resume format: " + file_path.suffix)


# --------------------------------------------------
# 10. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    # Read job description
    job_description = """
    We are looking for a Junior Software Engineer.

    Required skills:
    Python, Java, SQL, HTML, CSS and JavaScript.

    Preferred skills:
    React, Git and REST APIs.

    Minimum experience:
    0 years.

    Education:
    Bachelor's degree in Computer Science or related field.

    Responsibilities:
    Develop software applications, write clean code,
    debug programs and work with development teams.
    """

    print("\nExtracting job description...")

    job = extract_job_description(job_description)

    print("\nJOB DESCRIPTION DETAILS")
    print(job.model_dump_json(indent=2))


    # Locate resumes folder
    resume_folder = Path(__file__).parent / "resumes"

    if not resume_folder.exists():
        raise FileNotFoundError("Resumes folder not found.")


    all_results = []

    # Process every resume
    for file_path in resume_folder.iterdir():

        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        print("\n-----------------------------------")
        print("Processing:", file_path.name)
        print("-----------------------------------")

        try:

            # Read resume
            resume_text = read_resume(file_path)

            if not resume_text.strip():
                print("No readable text found in this resume.")
                continue

            # Extract resume details
            resume = parse_resume(resume_text)

            print("\nRESUME DETAILS")
            print(resume.model_dump_json(indent=2))

            # Score resume
            result = final_score(job, resume)

            print("\nMATCH SCORE:", result.score)

            print("\nMATCH DETAILS")
            print(json.dumps(result.details, indent=2))

            all_results.append(
                {
                    "file_name": file_path.name,
                    "name": resume.name,
                    "score": result.score,
                    "details": result.details
                }
            )

            # Pause between API requests
            time.sleep(2)

        except Exception as error:
            print("Error processing", file_path.name)
            print(error)


    # --------------------------------------------------
    # 11. DISPLAY FINAL RANKING
    # --------------------------------------------------

    all_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    print("\n===================================")
    print("FINAL RESUME RANKING")
    print("===================================")

    for index, result in enumerate(all_results, start=1):

        print("\nRank:", index)
        print("Candidate:", result["name"])
        print("File:", result["file_name"])
        print("Score:", result["score"])


    # Display top 2 resumes
    print("\n===================================")
    print("TOP 2 RESUMES")
    print("===================================")

    top_2 = all_results[:2]

    for result in top_2:

        print("\nCandidate:", result["name"])
        print("Score:", result["score"])


    # Display bottom 2 resumes
    print("\n===================================")
    print("WORST 2 RESUMES")
    print("===================================")

    worst_2 = all_results[-2:]

    for result in worst_2:

        print("\nCandidate:", result["name"])
        print("Score:", result["score"])
