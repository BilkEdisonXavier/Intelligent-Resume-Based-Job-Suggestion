# End-to-End AI Resume–Job Matching System

An end-to-end AI system that takes a candidate’s resume, extracts the text with AWS Lambda, stores it in DynamoDB, fetches live job postings from the Adzuna API, and matches the resume to jobs using semantic similarity and keyword analysis. The UI is built with Streamlit.

---

## 🧩 Features

- 📄 **Resume upload** (PDF / DOCX / TXT) via Streamlit.
- ☁️ **S3-backed storage** of original resumes.
- ⚙️ **AWS Lambda** auto-triggered on S3 upload to extract resume text.
- 🗄️ **DynamoDB** tables for both resume metadata and job postings.
- 🔍 **Dynamic job fetch** from Adzuna API based on resume-derived queries.
- 🤖 **AI matching engine**:
  - Sentence-transformers (MiniLM) or TF-IDF similarity.
  - Keyword extraction and overlap scoring.
  - Job recency & popularity weighting.
- 📊 **Interactive Streamlit UI** for:
  - Viewing ranked job matches.
  - Inspecting missing skills and suggested learning paths.
  - Debugging end-to-end data flow.

---

## 🏗️ Architecture

**High-level flow:**

```text
[User Browser]
      |
      v
[Streamlit App: app.py]
  - Upload resume (PDF/DOCX/TXT)
  - Fetch jobs from Adzuna
  - Run matching & show results
      |
      v
[Amazon S3 Bucket: edisonresume/resumes/{user_id}/...]
      |
      v (S3 put event)
[AWS Lambda: lamda.py]
  - Download file from S3
  - Extract text (PDF/DOCX/TXT)
  - Write item to DynamoDB (resumes_meta)
      |
      +------------------> [DynamoDB: resumes_meta]
      |
      +------------------> [DynamoDB: jobs_meta]  (via Streamlit + Adzuna)

Core AWS Resources

S3 bucket: edisonresume

DynamoDB tables:

resumes_meta

Partition key: user_id (String)

Sort key: resume_id (String, timestamp-based)

jobs_meta

Lambda function: lamda.py
Trigger: S3 PutObject on prefix resumes/.

🛠️ Tech Stack

Frontend: Streamlit

Backend / AI: Python, sentence-transformers, scikit-learn (TF-IDF, cosine similarity), NumPy, Pandas

Cloud: AWS S3, AWS Lambda, AWS DynamoDB

External API: Adzuna Jobs API
📂 Project Structure
.
├── app.py          # Streamlit web application
├── lamda.py        # AWS Lambda for resume text extraction
├── requirements.txt
└── README.md
Note: In AWS, lamda.py is packaged and deployed as the Lambda function.
app.py is executed locally (or on a server) with Streamlit.
☁️ AWS Configuration

IAM user / role with permissions for:

s3:GetObject, s3:PutObject on your bucket (edisonresume).

dynamodb:PutItem, dynamodb:Query, dynamodb:Scan on resumes_meta and jobs_meta.

S3 bucket: edisonresume

Add an event trigger for PUT on prefix resumes/ → invokes lamda.py Lambda.

DynamoDB tables:

resumes_meta with keys:

PK: user_id (String)

SK: resume_id (String)

jobs_meta (no strict key schema required for this demo; typical PK: job_id).

Lambda environment variables (in Lambda console):

DDB_TABLE = resumes_meta

AWS_REGION = ap-south-1 (or your chosen region)

MAX_TEXT_CHARS = 300000 (optional)
🚀 Usage Flow

Upload Resume

Enter a user_id (e.g. user_123).

Upload your resume file (PDF/DOCX/TXT).

The app uploads it to s3://edisonresume/resumes/{user_id}/....

S3 triggers Lambda, which extracts text and writes to resumes_meta.

Ingest Jobs from Adzuna

In the “Fetch related Jobs” section, either type a query or let the app auto-generate one from the resume text.

The app calls Adzuna, normalizes job fields, and writes them to jobs_meta.

Run Matching

Go to “Agentic Resume ↔ Jobs analysis”.

Make sure the user_id matches the one used during upload (e.g. user_123).

Click Run Agentic Compare.

The app:

Gets the latest resume for that user from resumes_meta.

Reads job items from jobs_meta.

Computes semantic similarity, keyword overlap, recency, and popularity.

Produces a final_score and shows the top-K jobs in a table and expandable panels.

Inspect Results

See matching scores per job.

See missing skills (keywords present in the job but not in the resume).

Get quick learning suggestions (e.g. “search beginner {skill} course”).

📌 Roadmap / Extensions

Add user authentication and per-user dashboards.

Integrate OCR (Textract) for scanned resumes.

Multi-language support.

Feedback loop: allow users to mark matches as good/bad and learn from it.

Deploy Streamlit app to a cloud host (e.g., EC2, ECS, Streamlit Community Cloud).
