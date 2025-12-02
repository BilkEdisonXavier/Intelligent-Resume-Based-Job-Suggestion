import os
import json
import boto3
import tempfile
import traceback
import zipfile
import re
import urllib.parse
import time
from datetime import datetime, timezone

from PyPDF2 import PdfReader  # simple, no extra layer required

# Config
DDB_TABLE = os.environ.get("DDB_TABLE", "resumes_meta")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "300000"))

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DDB_TABLE)


def extract_text_from_pdf(path):
    """Extract text from a text-based PDF with PyPDF2."""
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        print("PDF extraction error:", e, traceback.format_exc())
    return "\n".join(text_parts)


def extract_text_from_docx(path):
    """
    Lightweight DOCX text extraction without external dependencies.
    Not perfect but good enough for most resumes.
    """
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml").decode("utf-8", errors="ignore")
            xml_content = xml_content.replace("</w:p>", "\n")
            cleaned = re.sub(r"<[^>]+>", "", xml_content)
            return cleaned
    except Exception as e:
        print("DOCX extraction error:", e, traceback.format_exc())
        return ""


def extract_text_local(local_path, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(local_path)
    elif fname.endswith(".docx"):
        return extract_text_from_docx(local_path)
    elif fname.endswith(".txt"):
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"unsupported_filetype: {fname}")


def update_dynamo(user_id, resume_id, bucket, key, filename, status, extracted_text=None, error_msg=None):
    """
    Write an item ensuring it contains the required table keys:
      - user_id (Partition Key)
      - resume_id (Sort Key)
    """
    item = {
        "user_id": user_id,
        "resume_id": resume_id,
        "s3_bucket": bucket,
        "s3_key": key,
        "filename": filename,
        "status": status,
        "updated_ts": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    if extracted_text is not None:
        item["extracted_text"] = extracted_text[:MAX_TEXT_CHARS]
    if error_msg is not None:
        item["error"] = error_msg[:1000]

    print("Writing item to DynamoDB:", json.dumps({k: v for k, v in item.items() if k != "extracted_text"}))
    table.put_item(Item=item)


def handler(event, context):
    """
    Lambda handler for S3 Put events.
    Trigger whenever a resume is uploaded to S3.
    """
    print("Event:", json.dumps(event)[:2000])

    for rec in event.get("Records", []):
        try:
            s3_bucket = rec["s3"]["bucket"]["name"]
            raw_key = rec["s3"]["object"]["key"]
            s3_key = urllib.parse.unquote(raw_key)
            print("RAW_KEY:", raw_key, "DECODED:", s3_key)

            filename = os.path.basename(s3_key)

            # Infer user_id from key path like resumes/{user_id}/...
            parts = s3_key.split("/")
            user_id = "unknown_user"
            if len(parts) >= 2 and parts[0].lower() == "resumes":
                user_id = parts[1]
            elif len(parts) >= 1:
                user_id = parts[0]
            user_id = str(user_id)

            # Create a resume_id that matches your table's sort key (string)
            # Use ms timestamp to avoid collisions
            resume_id = str(int(time.time() * 1000))

            # Download file to /tmp
            with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                local_path = tmpf.name

            print(f"Downloading s3://{s3_bucket}/{s3_key} -> {local_path}")
            try:
                s3.download_file(s3_bucket, s3_key, local_path)
            except Exception as e:
                print("Download failed:", e, traceback.format_exc())
                update_dynamo(
                    user_id,
                    resume_id,
                    s3_bucket,
                    s3_key,
                    filename,
                    "error_download",
                    error_msg=str(e),
                )
                continue

            # Extract text
            try:
                extracted_text = extract_text_local(local_path, filename)
                if extracted_text:
                    status = "processed"
                else:
                    status = "processed_empty_text"
                print(f"Extraction OK, {len(extracted_text)} chars, status={status}")
                # Log a small sample for debugging
                print("EXTRACTED TEXT SAMPLE:", extracted_text[:400].replace("\n", " ") + "...")
                update_dynamo(
                    user_id,
                    resume_id,
                    s3_bucket,
                    s3_key,
                    filename,
                    status,
                    extracted_text=extracted_text,
                )
                print("DynamoDB updated with extracted text.")
            except ValueError as ve:
                print("Unsupported file type:", ve)
                update_dynamo(
                    user_id,
                    resume_id,
                    s3_bucket,
                    s3_key,
                    filename,
                    "unsupported_filetype",
                    error_msg=str(ve),
                )
            except Exception as e:
                tb = traceback.format_exc()
                print("Extraction failed:", str(e), tb)
                update_dynamo(
                    user_id,
                    resume_id,
                    s3_bucket,
                    s3_key,
                    filename,
                    "error_extraction",
                    error_msg=str(e) + " " + tb,
                )
            finally:
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        except Exception as e:
            print("Top-level processing error:", e, traceback.format_exc())

    return {"status": "done"}
