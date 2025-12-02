import streamlit as st
import requests
import os
import json
import time
import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError
from math import exp
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: sentence-transformers for better semantic similarity
USE_SENTENCE_TRANSFORMER = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_SENTENCE_TRANSFORMER = True
except Exception:
    USE_SENTENCE_TRANSFORMER = False

from boto3.dynamodb.conditions import Key

# ───────────────────────── AWS / DYNAMODB CONFIG ─────────────────────────

AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
RESUMES_TABLE = os.environ.get("RESUME_TABLE", "resumes_meta")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "jobs_meta")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
resumes_table = dynamodb.Table(RESUMES_TABLE)
jobs_table = dynamodb.Table(JOBS_TABLE)

# ───────────────────────── ADZUNA CONFIG ─────────────────────────

ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "40d1b3ac")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "6641443dc7818370be7112767085a036")
ADZUNA_COUNTRY = os.environ.get("ADZUNA_COUNTRY", "in")  # e.g. 'in', 'gb', 'us'

# ───────────────────────── UTILITIES ─────────────────────────


def _ensure_text(x):
    return x if isinstance(x, str) and x.strip() else ""


def _get_latest_resume_for_user(user_id: str):
    """
    Fetch latest resume item for user_id from resumes_meta table.

    Optimized to use Query on (user_id, resume_id) instead of full Scan.
    Falls back to Scan only if Query fails (for safety).
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return None

    # Try efficient Query first
    try:
        resp = resumes_table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,  # newest resume_id first (assuming timestamp-based SK)
            Limit=1,
        )
        items = resp.get("Items", [])
        if items:
            return items[0]
    except Exception as e:
        # If table doesn't have user_id as PK or resume_id as SK, fall back to scan.
        st.warning(
            f"DynamoDB Query on {RESUMES_TABLE} failed, falling back to Scan. Error: {e}"
        )

    # Fallback: Scan (slower but safe)
    try:
        resp = resumes_table.scan()
    except NoCredentialsError:
        st.error(
            "AWS credentials not found while reading resumes_meta. "
            "Run `aws configure` or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )
        return None
    except ClientError as e:
        st.error(f"DynamoDB error when reading resumes_meta: {e}")
        return None

    items = resp.get("Items", [])
    user_items = [it for it in items if it.get("user_id") == user_id]
    if not user_items:
        return None

    def _ts(item):
        t = item.get("updated_ts") or item.get("created_ts") or item.get("resume_id")
        try:
            return datetime.fromisoformat(t)
        except Exception:
            try:
                return datetime.fromtimestamp(float(t) / 1000)
            except Exception:
                return datetime.fromtimestamp(0)

    user_items_sorted = sorted(user_items, key=_ts, reverse=True)
    return user_items_sorted[0]


def _scan_jobs(filter_location=None):
    """Scan jobs_meta table; optional filter by location substring."""
    try:
        resp = jobs_table.scan()
    except NoCredentialsError:
        st.error(
            "AWS credentials not found while reading jobs_meta. "
            "Configure credentials locally (`aws configure`) or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )
        return []
    except ClientError as e:
        st.error(f"DynamoDB error when reading jobs_meta: {e}")
        return []

    items = resp.get("Items", [])
    if filter_location:
        fl = filter_location.lower()
        items = [
            it
            for it in items
            if fl in str(it.get("location", "")).lower()
            or fl in str(it.get("job_city", "")).lower()
            or fl in str(it.get("job_country", "")).lower()
        ]
    return items


def _extract_keywords_top_n(corpus_texts, top_n=20):
    """Return top_n keywords per document using TF-IDF feature ranking."""
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=1000, ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(corpus_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords_per_doc = []
    for row in X:
        row_arr = row.toarray().ravel()
        if row_arr.sum() == 0:
            keywords_per_doc.append([])
            continue
        top_idx = row_arr.argsort()[::-1][:top_n]
        kws = [feature_names[i] for i in top_idx if row_arr[i] > 0]
        keywords_per_doc.append(kws)
    return keywords_per_doc


def compute_semantic_similarity_pairs(resume_text, job_texts):
    """
    Return list of semantic similarities in [0,1] between resume and each job text.
    Uses sentence-transformers if available, else TF-IDF cosine fallback.
    """
    resume_text = _ensure_text(resume_text)
    job_texts = [_ensure_text(t) for t in job_texts]

    if not resume_text or not any(job_texts):
        return [0.0] * len(job_texts)

    # Better semantic model if available
    if USE_SENTENCE_TRANSFORMER:
        try:
            emb_resume = model.encode(resume_text, convert_to_tensor=True)
            emb_jobs = model.encode(job_texts, convert_to_tensor=True)
            sims = st_util.cos_sim(emb_resume, emb_jobs).cpu().numpy().ravel()
            sims = ((sims + 1) / 2).clip(0, 1)
            return sims.tolist()
        except Exception:
            pass

    # Fallback TF-IDF cosine
    texts = [resume_text] + job_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < 2:
        return [0.0] * len(job_texts)
    resume_vec = X[0]
    job_vecs = X[1:]
    sims = cosine_similarity(resume_vec, job_vecs).ravel()
    sims = np.nan_to_num(sims).clip(0, 1)
    return sims.tolist()


def compute_keyword_overlap_score(resume_kws, job_kws):
    rs = set([k.lower() for k in resume_kws])
    js = set([k.lower() for k in job_kws])
    if not rs:
        return 0.0
    inter = rs & js
    union = rs | js
    return len(inter) / len(union) if union else 0.0


def compute_recency_weight(created_ts):
    """Simple linear decay: score = max(0, 1 - days/90)."""
    if not created_ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(created_ts)
    except Exception:
        try:
            t = float(created_ts)
            if t > 1e12:
                dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(t, tz=timezone.utc)
        except Exception:
            return 0.0
    now = datetime.now(dt.tzinfo or timezone.utc)
    delta_days = (now - dt).days
    score = max(0.0, 1.0 - (delta_days / 90.0))
    return float(score)


def compute_popularity_score(job_item):
    for k in ("popularity", "views", "apply_count", "num_applicants"):
        v = job_item.get(k)
        if v is None:
            continue
        try:
            v = float(v)
            return float(max(0.0, min(1.0, v / (v + 100.0))))
        except Exception:
            continue
    if job_item.get("employer_logo") or job_item.get("job_apply_link") or job_item.get("url"):
        return 0.5
    return 0.1


def compare_resume_with_jobs(user_id, top_k=10, location_filter=None):
    """
    Returns dict:
      {error, message, results(DataFrame), resume_kws}
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return {
            "error": "no_user_id",
            "message": "User ID is empty. Please enter the same User ID you used to upload the resume.",
            "results": [],
        }

    resume_item = _get_latest_resume_for_user(user_id)
    if not resume_item:
        return {
            "error": "no_resume",
            "message": f"No resume found for user '{user_id}'. Check the User ID and S3 upload.",
            "results": [],
        }

    resume_text = _ensure_text(resume_item.get("extracted_text", ""))
    if not resume_text:
        return {
            "error": "empty_resume",
            "message": f"Resume text empty for user '{user_id}'. The file may be scanned (image-only) or unsupported.",
            "results": [],
        }

    job_items = _scan_jobs(filter_location=location_filter)
    if not job_items:
        return {
            "error": "no_jobs",
            "message": "No jobs found in jobs_meta (for given filter). Try fetching Adzuna jobs first.",
            "results": [],
        }

    job_texts = []
    job_ids = []
    for it in job_items:
        desc = (
            it.get("description")
            or it.get("job_description")
            or it.get("job_text")
            or it.get("snippet")
            or ""
        )
        job_texts.append(_ensure_text(desc))
        job_ids.append(it.get("job_id") or it.get("id") or it.get("url") or str(time.time()))

    semantic_sims = compute_semantic_similarity_pairs(resume_text, job_texts)

    corpus = [resume_text] + job_texts
    kws_list = _extract_keywords_top_n(corpus, top_n=30)
    resume_kws = kws_list[0]
    job_kws_list = kws_list[1:]

    rows = []
    for idx, it in enumerate(job_items):
        semantic = float(semantic_sims[idx]) if idx < len(semantic_sims) else 0.0
        job_kws = job_kws_list[idx] if idx < len(job_kws_list) else []
        kw_overlap = compute_keyword_overlap_score(resume_kws, job_kws)
        recency = compute_recency_weight(
            it.get("created_ts") or it.get("created") or it.get("date_posted") or ""
        )
        popularity = compute_popularity_score(it)

        final_score = 0.55 * semantic + 0.25 * kw_overlap + 0.10 * recency + 0.10 * popularity
        missing_skills = sorted(
            list(set([k.lower() for k in job_kws]) - set([k.lower() for k in resume_kws]))
        )
        rows.append(
            {
                "job_id": job_ids[idx],
                "title": it.get("title") or it.get("job_title") or "",
                "company": it.get("company") or it.get("employer_name") or "",
                "semantic": round(semantic, 4),
                "keyword_overlap": round(kw_overlap, 4),
                "recency": round(recency, 4),
                "popularity": round(popularity, 4),
                "final_score": round(float(final_score), 4),
                "missing_skills": missing_skills,
                "raw_item": it,
            }
        )

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    top_df = df.head(top_k)
    return {"error": None, "message": "ok", "results": top_df, "resume_kws": resume_kws}


# ─────────────────────── ADZUNA JOB FETCH + WRITE ───────────────────────


def fetch_jobs_5(query, results_per_page=5):
    """
    Fetch up to `results_per_page` jobs from Adzuna API for the given query.
    Docs: https://developer.adzuna.com/
    Uses env vars: ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY
    """
    app_id = ADZUNA_APP_ID
    app_key = ADZUNA_APP_KEY
    country = ADZUNA_COUNTRY

    if not app_id or not app_key:
        st.error(
            "Adzuna APP_ID / APP_KEY not set. Please set ADZUNA_APP_ID and ADZUNA_APP_KEY environment variables."
        )
        return []

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": query,
        "content-type": "application/json",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"Adzuna API request failed: {e}")
        return []

    if r.status_code != 200:
        st.error(f"Adzuna Error {r.status_code}: {r.text[:300]}")
        return []

    data = r.json()
    jobs = data.get("results", [])
    return jobs[:results_per_page]


def _normalize_job(raw, user_id=None):
    """
    Normalize job fields from Adzuna API into a jobs_meta record.
    """
    job_id = raw.get("job_id") or raw.get("id") or raw.get("uuid") or raw.get("link")
    if not job_id:
        job_id = f"job_{int(time.time() * 1000)}"

    title = raw.get("job_title") or raw.get("title") or raw.get("position") or ""

    company = ""
    if isinstance(raw.get("company"), dict):
        company = raw["company"].get("display_name") or ""
    else:
        company = raw.get("employer_name") or raw.get("company_name") or raw.get("company") or ""

    location = ""
    if isinstance(raw.get("location"), dict):
        location = raw["location"].get("display_name") or ""
    else:
        location = raw.get("job_city") or raw.get("location") or raw.get("job_country") or ""

    description = raw.get("job_description") or raw.get("description") or raw.get("snippet") or ""

    url = (
        raw.get("redirect_url")
        or raw.get("job_apply_link")
        or raw.get("apply_link")
        or raw.get("url")
        or raw.get("link")
        or ""
    )

    created = raw.get("created") or raw.get("date_posted") or raw.get("post_date") or ""

    item = {
        "job_id": str(job_id),
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "url": url,
        "created_ts": created,
        "source": "adzuna",
        "raw": json.dumps(raw)[:20000],
    }

    if user_id:
        item["user_id"] = user_id

    item["written_ts"] = datetime.now(timezone.utc).astimezone().isoformat()
    return item


def write_jobs_to_dynamo(jobs, user_id=None):
    if not isinstance(jobs, (list, tuple)):
        return 0, [{"error": "jobs is not a list"}]

    written = 0
    failed = []
    for raw in jobs:
        try:
            item = _normalize_job(raw, user_id=user_id)
            jobs_table.put_item(Item=item)
            written += 1
        except NoCredentialsError as e:
            failed.append({"raw": raw, "error": "No AWS credentials: " + str(e)})
            st.error(
                "AWS credentials not found while writing jobs. "
                "Configure them via `aws configure` or environment variables."
            )
            break
        except Exception as e:
            failed.append({"raw": raw, "error": str(e)})

    return written, failed


# ─────────────────────── S3 RESUME UPLOAD ───────────────────────

S3_BUCKET = "edisonresume"
s3_client = boto3.client("s3", region_name=AWS_REGION)

st.title("End-to-End AI Resume–Job Matching System")

st.header("1️⃣ Upload resume (PDF/DOCX/TXT) to S3")

user_id = st.text_input(
    "User ID (used for S3 path & DynamoDB)",
    value=st.session_state.get("canonical_user_id", ""),
    key="canonical_user_id",
    help="Use the SAME user id every time (e.g., user_123 or your email).",
)
st.write("Current user id:", f"`{user_id.strip() or '<empty>'}`")

uploaded_file = st.file_uploader("Choose a resume file", type=["pdf", "docx", "txt"])

if st.button("Upload to S3"):
    uid = user_id.strip()
    if uploaded_file is None:
        st.error("Please choose a file first.")
    elif not uid:
        st.error("Please enter a user id.")
    else:
        original_filename = uploaded_file.name
        ts = int(time.time() * 1000)
        safe_filename = f"{ts}_{original_filename.replace(' ', '_')}"
        s3_key = f"resumes/{uid}/{safe_filename}"

        ext = original_filename.lower().split(".")[-1]
        if ext == "pdf":
            content_type = "application/pdf"
        elif ext == "docx":
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext == "txt":
            content_type = "text/plain"
        else:
            content_type = "application/octet-stream"

        try:
            uploaded_file.seek(0)
            s3_client.upload_fileobj(
                Fileobj=uploaded_file,
                Bucket=S3_BUCKET,
                Key=s3_key,
                ExtraArgs={"ContentType": content_type, "ACL": "private"},
            )

            st.success(f"Upload successful: `s3://{S3_BUCKET}/{s3_key}`")
            st.write("Lambda will now extract text and store it into DynamoDB.")
        except NoCredentialsError:
            st.error(
                "AWS credentials not found. Run `aws configure` or set AWS_ACCESS_KEY_ID / "
                "AWS_SECRET_ACCESS_KEY in your environment."
            )
        except ClientError as e:
            st.error(f"Upload failed: {e.response.get('Error', {}).get('Message', str(e))}")
        except Exception as e:
            st.error(f"Unexpected error during upload: {e}")


# ─────────────────────── DYNAMIC JOB FETCH FROM RESUME ───────────────────────


def build_dynamic_query_from_resume(resume_text, top_n=6):
    resume_text = (resume_text or "").strip()
    if not resume_text:
        return "software developer"

    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    headline = lines[0] if lines else ""
    headline_tokens = []
    if headline and len(headline.split()) <= 6:
        headline_tokens = re.findall(r"[A-Za-z0-9\+\-#\.]{2,}", headline)
        headline_tokens = [t for t in headline_tokens if len(t) > 1]

    try:
        vect = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
        X = vect.fit_transform([resume_text])
        feat = np.array(vect.get_feature_names_out())
        scores = X.toarray().ravel()
        top_idx = scores.argsort()[::-1]
        tfidf_keywords = [feat[i] for i in top_idx if scores[i] > 0][: top_n * 3]
    except Exception:
        tfidf_keywords = []

    candidates = []
    for t in headline_tokens:
        tl = t.lower()
        if tl not in candidates:
            candidates.append(tl)
    for t in tfidf_keywords:
        tok = t.lower()
        if len(tok) > 60:
            continue
        if tok in ("experience", "years", "year", "worked", "responsible", "profile", "summary"):
            continue
        if tok not in candidates:
            candidates.append(tok)
    if not candidates:
        found = re.findall(
            r"(python|java|c\+\+|c#|javascript|react|node|sql|aws|docker|kubernetes|spark|etl)",
            resume_text,
            flags=re.I,
        )
        candidates = [f.lower() for f in found]
    final = []
    for c in candidates:
        if c not in final:
            final.append(c)
        if len(final) >= top_n:
            break

    query = ", ".join(final) if final else "software developer"
    return query


st.header("2️⃣ Fetch related Jobs from Adzuna (Dynamic Query)")

latest_resume_item = _get_latest_resume_for_user(user_id)
resume_text_for_query = latest_resume_item.get("extracted_text", "") if latest_resume_item else ""

query = st.text_input("Dynamic query (you can type or leave blank to auto-generate)", value="")
if not query and resume_text_for_query:
    query = build_dynamic_query_from_resume(resume_text_for_query, top_n=6)
    st.info(f"Auto-generated dynamic query from latest resume: `{query}`")

if st.button("Fetch related Jobs (dynamic)"):
    if not query:
        st.error("Please enter a query or upload a resume so we can generate one.")
    else:
        st.info(f"Using dynamic query: {query!r} — calling Adzuna...")
        try:
            jobs = fetch_jobs_5(query)
        except Exception as e:
            st.error("fetch_jobs_5 call failed: " + str(e))
            jobs = []

        if not jobs:
            st.warning(
                "No jobs returned. Possible causes: Adzuna keys missing/invalid, quota issue, or query produced no results."
            )
        else:
            st.success(f"Fetched {len(jobs)} jobs from Adzuna.")
            st.write("Sample titles returned:")
            for j in jobs[:10]:
                st.write("-", j.get("title") or j.get("job_title") or j.get("position") or "<no title>")

            try:
                written, failed = write_jobs_to_dynamo(jobs, user_id=user_id.strip())
                st.success(f"Wrote {written} jobs to {JOBS_TABLE}")
                if failed:
                    st.error(f"{len(failed)} jobs failed to write — showing up to 5 failed items")
                    st.json(failed[:5])
            except Exception as e:
                st.error("Failed to write fetched jobs to DynamoDB: " + str(e))


# ─────────────────────── AGENTIC UI ───────────────────────


def run_agentic_ui():
    st.header("3️⃣ Agentic Resume ↔ Jobs analysis")
    top_k = st.slider("Top K jobs to show", 1, 50, 10)

    if st.button("Run Agentic Compare"):
        effective_user_id = user_id.strip() or st.session_state.get("canonical_user_id", "").strip()
        if not effective_user_id:
            st.error("User ID is empty. Enter the SAME user ID you used during resume upload.")
            return

        with st.spinner("Comparing resume with jobs..."):
            out = compare_resume_with_jobs(user_id=effective_user_id, top_k=top_k)

        st.subheader("DEBUG: Agent view (resume → query → jobs → scores)")

        resume_item_dbg = _get_latest_resume_for_user(effective_user_id)
        if not resume_item_dbg:
            st.error(
                f"No resume found for user '{effective_user_id}' (DEBUG). "
                "Check the user id or upload a resume first."
            )
        else:
            resume_text_dbg = resume_item_dbg.get("extracted_text", "")
            st.markdown("**Resume text (first 800 chars):**")
            st.text(resume_text_dbg[:800])

            try:
                dyn_q = build_dynamic_query_from_resume(resume_text_dbg, top_n=8)
            except Exception as e:
                dyn_q = "software developer"
                st.write("build_dynamic_query_from_resume error:", e)
            st.write("Generated query:", dyn_q)

            job_items_dbg = _scan_jobs()
            st.write("Jobs in DB count:", len(job_items_dbg))

            job_texts_dbg = []
            job_titles_dbg = []
            for it in job_items_dbg[:50]:
                desc = (
                    it.get("description")
                    or it.get("job_description")
                    or it.get("job_text")
                    or it.get("snippet")
                    or ""
                )
                job_texts_dbg.append(_ensure_text(desc))
                job_titles_dbg.append(it.get("title") or it.get("job_title") or it.get("position") or "<no title>")

            st.write("Sample job titles (first 10):", job_titles_dbg[:10])

            try:
                sims_dbg = (
                    compute_semantic_similarity_pairs(resume_text_dbg, job_texts_dbg)
                    if job_texts_dbg
                    else []
                )
            except Exception as e:
                sims_dbg = []
                st.write("compute_semantic_similarity_pairs error:", e)

            if sims_dbg:
                sims_with_titles = list(zip(job_titles_dbg, sims_dbg))
                sims_with_titles_sorted = sorted(sims_with_titles, key=lambda x: x[1], reverse=True)[:10]
                st.write("Top 10 jobs by semantic similarity (title, score):")
                for t, s in sims_with_titles_sorted:
                    st.write(f"- {t}  →  {round(s, 4)}")
            else:
                st.write("No semantic similarities computed (empty job_texts or error).")

            try:
                c = [resume_text_dbg] + job_texts_dbg[:10]
                kwlists = _extract_keywords_top_n(c, top_n=20)
                st.write("Resume top keywords:", kwlists[0][:30])
                st.write("First jobs' top keywords (first 3 jobs):")
                for i, k in enumerate(kwlists[1:4], start=1):
                    st.write(f"Job {i} keywords:", k[:15])
            except Exception as e:
                st.write("Keyword extraction error:", e)

            st.info("Run the full agentic compare to produce ranked results in the table below.")

        if out.get("error"):
            st.error(out.get("message"))
            return

        df = out["results"]
        resume_kws = out.get("resume_kws", [])
        st.subheader("Resume top keywords")
        st.write(resume_kws[:40])

        st.subheader(f"Top {len(df)} matching jobs (by final_score)")
        display_df = df[
            [
                "job_id",
                "title",
                "company",
                "final_score",
                "semantic",
                "keyword_overlap",
                "recency",
                "popularity",
                "missing_skills",
            ]
        ]
        st.dataframe(display_df)

        for i, row in df.iterrows():
            with st.expander(
                f"{i+1}. {row['title']} — {row['company']} (score {row['final_score']})",
                expanded=False,
            ):
                st.write("Missing skills (suggested to learn):", row["missing_skills"])
                if row["missing_skills"]:
                    st.markdown("**Suggested learning searches (quick):**")
                    for skill in row["missing_skills"][:6]:
                        st.write(f"- Search: `beginner {skill} course` on Coursera/Udemy/YouTube")
                st.write("Raw job item:")
                st.json(row["raw_item"])

        # Optional visual comparison if matplotlib/seaborn available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            comp_df = df[["semantic", "keyword_overlap", "recency", "popularity"]].astype(float)
            if not comp_df.empty:
                fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(comp_df))))
                sns.heatmap(
                    comp_df,
                    annot=True,
                    fmt=".2f",
                    yticklabels=df["job_id"].values,
                    cbar=True,
                    ax=ax,
                )
                st.pyplot(fig)
        except Exception:
            pass


# Run main UI
run_agentic_ui()

# Optional tiny debug helper – shows which user_ids exist in resumes_meta
with st.expander("🔧 DEBUG: Show user_ids in resumes_meta"):
    try:
        resp = resumes_table.scan(Limit=1000)
        user_ids = sorted({it.get("user_id", "<missing>") for it in resp.get("Items", [])})
        st.write(user_ids)
    except Exception as e:
        st.write("DynamoDB scan error:", e)
