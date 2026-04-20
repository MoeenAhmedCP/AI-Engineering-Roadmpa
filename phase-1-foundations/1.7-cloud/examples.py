"""
1.7 Cloud Fundamentals — Examples
Demonstrates: S3 document storage, Lambda handler, IAM policy generation,
and monthly cost estimation for AI apps.

Set SIMULATE = True to run without real AWS credentials (all calls are mocked).
"""

import json
import io
import time
import random
import string
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Simulation mode — flip to False when using real AWS credentials
# ---------------------------------------------------------------------------

SIMULATE = True


# ---------------------------------------------------------------------------
# 1. S3DocumentStorage — upload, download, presigned URL, list
# ---------------------------------------------------------------------------

class SimulatedS3:
    """In-memory S3 mock that mirrors the boto3 client API we use."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._metadata: dict[str, dict] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = "", **kwargs):
        self._store[Key] = Body
        self._metadata[Key] = {
            "ContentType": ContentType,
            "LastModified": datetime.utcnow(),
            "ContentLength": len(Body),
        }
        print(f"  [SimS3] PUT s3://{Bucket}/{Key} ({len(Body)} bytes)")

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self._store:
            raise _SimClientError("NoSuchKey", f"The specified key does not exist: {Key}")
        data = self._store[Key]
        print(f"  [SimS3] GET s3://{Bucket}/{Key} ({len(data)} bytes)")
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    def generate_presigned_url(self, operation: str, Params: dict, ExpiresIn: int = 3600) -> str:
        key = Params["Key"]
        bucket = Params["Bucket"]
        token = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
        expiry = datetime.utcnow() + timedelta(seconds=ExpiresIn)
        url = (
            f"https://{bucket}.s3.amazonaws.com/{key}"
            f"?X-Amz-Signature={token}&X-Amz-Expires={ExpiresIn}"
        )
        print(f"  [SimS3] Presigned URL generated (expires: {expiry.strftime('%H:%M:%S UTC')})")
        return url

    def list_objects_v2(self, Bucket: str, Prefix: str = "") -> dict:
        contents = []
        for key, data in self._store.items():
            if key.startswith(Prefix):
                contents.append({
                    "Key": key,
                    "Size": len(data),
                    "LastModified": self._metadata[key]["LastModified"],
                })
        print(f"  [SimS3] LIST s3://{Bucket}/{Prefix}* → {len(contents)} objects")
        return {"Contents": contents, "KeyCount": len(contents)}


class _SimClientError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


def _make_s3_client():
    """Return a real or simulated S3 client depending on SIMULATE flag."""
    if SIMULATE:
        return SimulatedS3()
    import boto3
    return boto3.client("s3", region_name="us-east-1")


class S3DocumentStorage:
    """
    Wrapper around S3 for document storage operations.
    All methods work with both the real boto3 client and the simulated client.
    """

    def __init__(self, bucket: str = "docsense-documents"):
        self.bucket = bucket
        self._s3 = _make_s3_client()

    def upload_document(self, file_bytes: bytes, filename: str, prefix: str = "uploads") -> str:
        """
        Upload file bytes to S3. Returns the S3 key.
        Key format: uploads/2026/04/19/<filename>
        """
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        s3_key = f"{prefix}/{date_path}/{filename}"

        content_type = self._infer_content_type(filename)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
        return s3_key

    def download_document(self, key: str) -> bytes:
        """Download a document by S3 key. Raises FileNotFoundError if missing."""
        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except (_SimClientError, Exception) as e:
            # Handle both simulated and real ClientError
            err = getattr(e, "response", {}).get("Error", {})
            if err.get("Code") == "NoSuchKey":
                raise FileNotFoundError(f"Document not found: {key}") from e
            raise

    def generate_presigned_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """
        Generate a time-limited URL for direct download.
        Useful for email attachments, frontend file downloads.
        Default expiry: 1 hour.
        """
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )

    def list_documents(self, prefix: str = "uploads") -> list[dict]:
        """List all documents under a given prefix. Returns list of S3 object metadata."""
        response = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return response.get("Contents", [])

    @staticmethod
    def _infer_content_type(filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "json": "application/json",
        }.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# 2. Lambda handler: SQS trigger → process document
# ---------------------------------------------------------------------------

def _simulate_document_processing(document_id: str) -> dict:
    """Simulate downloading + analyzing a document (stand-in for real Claude call)."""
    time.sleep(0.01)  # simulate processing time
    return {
        "document_id": document_id,
        "status": "done",
        "summary": f"Simulated analysis for document {document_id}",
        "confidence": round(random.uniform(0.75, 0.99), 2),
        "processed_at": datetime.utcnow().isoformat(),
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda function triggered by SQS.
    Each SQS message contains a JSON body with 'document_id' and 's3_key'.

    Real usage:
      - SQS receives message when a document is uploaded via POST /upload
      - Lambda downloads from S3, extracts text, calls Claude, stores result
      - On success: update DB row to status='done'
      - On failure: Lambda automatically retries (SQS visibility timeout)

    Args:
        event: Lambda event dict. Contains 'Records' list of SQS messages.
        context: Lambda context (request_id, remaining time, etc.)

    Returns:
        Dict with statusCode and processing summary.
    """
    records = event.get("Records", [])
    print(f"[Lambda] Received {len(records)} SQS message(s)")

    results = []
    errors = []

    for i, record in enumerate(records):
        try:
            # Parse the SQS message body
            body = json.loads(record.get("body", "{}"))
            document_id = body.get("document_id")
            s3_key = body.get("s3_key", "")

            if not document_id:
                raise ValueError("Missing 'document_id' in SQS message body")

            print(f"[Lambda] Processing record {i+1}/{len(records)}: document_id={document_id}")

            # Step 1: Download document from S3
            if SIMULATE:
                file_bytes = b"Simulated PDF content for " + document_id.encode()
            else:
                storage = S3DocumentStorage()
                file_bytes = storage.download_document(s3_key)

            print(f"[Lambda]   Downloaded {len(file_bytes)} bytes from S3")

            # Step 2: Extract text (would use services/parser.py in real app)
            extracted_text = file_bytes.decode("utf-8", errors="ignore")[:50_000]
            print(f"[Lambda]   Extracted {len(extracted_text)} chars of text")

            # Step 3: Analyze with Claude (simulated)
            result = _simulate_document_processing(document_id)
            print(f"[Lambda]   Analysis complete — confidence: {result['confidence']}")

            # Step 4: Store result in database (simulated)
            print(f"[Lambda]   Stored result to DB — status: {result['status']}")

            results.append(result)

        except Exception as e:
            error_msg = f"Failed to process record {i+1}: {e}"
            print(f"[Lambda] ERROR: {error_msg}")
            errors.append(error_msg)
            # In production: mark document as status='error' in DB
            # Do NOT re-raise — let the other records process
            # SQS will retry failed messages up to the max receive count

    response = {
        "statusCode": 200 if not errors else 207,
        "processed": len(results),
        "errors": len(errors),
        "results": results,
    }
    if errors:
        response["error_details"] = errors

    print(f"[Lambda] Done. Processed: {len(results)}, Errors: {len(errors)}")
    return response


# ---------------------------------------------------------------------------
# 3. IAM policy for least-privilege S3 access
# ---------------------------------------------------------------------------

def generate_s3_iam_policy(bucket_name: str, prefix: str = "uploads") -> dict:
    """
    Generate a least-privilege IAM policy for a service that needs to
    read and write documents in a specific S3 bucket prefix.

    Grants ONLY:
    - PutObject: upload new documents
    - GetObject: download existing documents
    - DeleteObject: clean up old documents

    Does NOT grant:
    - ListBucket (not needed for direct key access)
    - CreateBucket, DeleteBucket
    - s3:* (never use wildcards in production policies)
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowDocumentReadWrite",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                ],
                "Resource": f"arn:aws:s3:::{bucket_name}/{prefix}/*",
                "Condition": {
                    "StringEquals": {
                        "s3:x-amz-server-side-encryption": "AES256"
                    }
                }
            },
            {
                "Sid": "AllowListBucketForPrefix",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket_name}",
                "Condition": {
                    "StringLike": {
                        "s3:prefix": f"{prefix}/*"
                    }
                }
            },
            {
                "Sid": "DenyUnencryptedObjectUploads",
                "Effect": "Deny",
                "Action": "s3:PutObject",
                "Resource": f"arn:aws:s3:::{bucket_name}/*",
                "Condition": {
                    "StringNotEquals": {
                        "s3:x-amz-server-side-encryption": "AES256"
                    }
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# 4. Monthly cost estimator for AI apps
# ---------------------------------------------------------------------------

# Model pricing per million tokens (as of early 2026, approximate)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-3-5":   {"input": 0.80, "output": 4.00},
    "gpt-4o":            {"input": 5.00, "output": 15.00},
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo":     {"input": 0.50, "output": 1.50},
}

# ECS Fargate pricing (us-east-1)
FARGATE_VCPU_PER_HOUR = 0.04048   # per vCPU-hour
FARGATE_GB_PER_HOUR   = 0.004445  # per GB-hour

# S3 pricing
S3_STORAGE_PER_GB_MONTH = 0.023
S3_PUT_PER_1000         = 0.005
S3_GET_PER_1000         = 0.0004


def estimate_monthly_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "claude-sonnet-4-6",
    ecs_vcpu: float = 0.5,
    ecs_memory_gb: float = 1.0,
    ecs_task_count: int = 2,
    avg_file_size_kb: float = 500.0,
) -> dict:
    """
    Estimate monthly costs for an AI document processing app.

    Args:
        requests_per_day:    Number of documents analyzed per day
        avg_input_tokens:    Average tokens sent to LLM per request
        avg_output_tokens:   Average tokens received from LLM per request
        model:               LLM model name (see MODEL_PRICING)
        ecs_vcpu:            vCPU per Fargate task
        ecs_memory_gb:       Memory (GB) per Fargate task
        ecs_task_count:      Number of always-running Fargate tasks
        avg_file_size_kb:    Average uploaded file size in KB

    Returns:
        Dict with cost breakdown and total.
    """
    if model not in MODEL_PRICING:
        available = list(MODEL_PRICING.keys())
        raise ValueError(f"Unknown model: {model!r}. Choose from: {available}")

    pricing = MODEL_PRICING[model]
    requests_per_month = requests_per_day * 30

    # LLM costs
    input_cost  = (requests_per_month * avg_input_tokens  / 1_000_000) * pricing["input"]
    output_cost = (requests_per_month * avg_output_tokens / 1_000_000) * pricing["output"]
    llm_cost = input_cost + output_cost

    # ECS Fargate costs (always-on tasks)
    hours_per_month = 24 * 30
    ecs_cost = ecs_task_count * (
        ecs_vcpu * FARGATE_VCPU_PER_HOUR * hours_per_month
        + ecs_memory_gb * FARGATE_GB_PER_HOUR * hours_per_month
    )

    # S3 costs
    s3_storage_gb = (requests_per_month * avg_file_size_kb) / (1024 * 1024)
    s3_storage_cost = s3_storage_gb * S3_STORAGE_PER_GB_MONTH
    s3_put_cost = (requests_per_month / 1000) * S3_PUT_PER_1000
    s3_get_cost = (requests_per_month / 1000) * S3_GET_PER_1000
    s3_cost = s3_storage_cost + s3_put_cost + s3_get_cost

    # Rough estimates for other costs
    alb_cost = 16.00          # base ALB cost (us-east-1)
    nat_cost = 32.00          # NAT Gateway base
    rds_cost = 25.00          # t3.micro RDS PostgreSQL estimate

    total = llm_cost + ecs_cost + s3_cost + alb_cost + nat_cost + rds_cost

    return {
        "summary": {
            "model": model,
            "requests_per_day": requests_per_day,
            "requests_per_month": requests_per_month,
        },
        "costs": {
            "llm_input":  round(input_cost, 2),
            "llm_output": round(output_cost, 2),
            "llm_total":  round(llm_cost, 2),
            "ecs_fargate": round(ecs_cost, 2),
            "s3":          round(s3_cost, 4),
            "alb":         round(alb_cost, 2),
            "nat_gateway": round(nat_cost, 2),
            "rds":         round(rds_cost, 2),
            "total_monthly_usd": round(total, 2),
        },
        "notes": [
            "LLM costs dominate at scale — consider caching repeated analyses",
            "NAT Gateway can be eliminated by using S3/Secrets Manager VPC endpoints",
            "RDS estimate assumes t3.micro with 20GB storage in us-east-1",
            "Data transfer costs not included — add ~$0.09/GB for outbound traffic",
        ]
    }


# ---------------------------------------------------------------------------
# Demo block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("1.7 Cloud Fundamentals — Demo")
    print(f"SIMULATE = {SIMULATE}")
    print("=" * 65)

    # --- S3 Storage Demo ---
    print("\n[Demo 1] S3DocumentStorage")
    storage = S3DocumentStorage(bucket="docsense-documents")

    # Upload a fake PDF
    fake_pdf = b"%PDF-1.4 fake document content for demonstration purposes"
    s3_key = storage.upload_document(fake_pdf, "contract_v2.pdf")
    print(f"  Uploaded → key: {s3_key}")

    # Upload a text file
    txt_key = storage.upload_document(b"Meeting notes content here.", "notes.txt")

    # Download it back
    downloaded = storage.download_document(s3_key)
    assert downloaded == fake_pdf, "Download mismatch!"
    print(f"  Downloaded → {len(downloaded)} bytes (matches original)")

    # Generate presigned URL
    url = storage.generate_presigned_url(s3_key, expiry_seconds=1800)
    print(f"  Presigned URL: {url[:80]}...")

    # List all uploads
    objects = storage.list_documents(prefix="uploads")
    print(f"  Listed {len(objects)} object(s) in uploads/")

    # Test FileNotFoundError
    try:
        storage.download_document("uploads/nonexistent/file.pdf")
    except FileNotFoundError as e:
        print(f"  FileNotFoundError caught correctly: {e}")

    # --- Lambda Handler Demo ---
    print("\n[Demo 2] Lambda Handler (SQS trigger)")
    fake_sqs_event = {
        "Records": [
            {
                "body": json.dumps({
                    "document_id": "doc-abc-123",
                    "s3_key": s3_key,
                })
            },
            {
                "body": json.dumps({
                    "document_id": "doc-def-456",
                    "s3_key": txt_key,
                })
            },
        ]
    }
    result = lambda_handler(fake_sqs_event, context=None)
    print(f"  Lambda response: {json.dumps(result, indent=2)}")

    # --- IAM Policy Demo ---
    print("\n[Demo 3] IAM Least-Privilege Policy")
    policy = generate_s3_iam_policy("docsense-documents", prefix="uploads")
    print(json.dumps(policy, indent=2))

    # --- Cost Estimator Demo ---
    print("\n[Demo 4] Monthly Cost Estimator")

    scenarios = [
        {
            "label": "Small app (100 docs/day, Claude Sonnet)",
            "kwargs": {
                "requests_per_day": 100,
                "avg_input_tokens": 8000,
                "avg_output_tokens": 1000,
                "model": "claude-sonnet-4-6",
            }
        },
        {
            "label": "Medium app (1000 docs/day, Claude Haiku)",
            "kwargs": {
                "requests_per_day": 1000,
                "avg_input_tokens": 8000,
                "avg_output_tokens": 1000,
                "model": "claude-haiku-3-5",
                "ecs_task_count": 3,
            }
        },
        {
            "label": "Scale test (10000 docs/day, GPT-4o Mini)",
            "kwargs": {
                "requests_per_day": 10000,
                "avg_input_tokens": 6000,
                "avg_output_tokens": 800,
                "model": "gpt-4o-mini",
                "ecs_vcpu": 1.0,
                "ecs_memory_gb": 2.0,
                "ecs_task_count": 4,
            }
        }
    ]

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['label']}")
        estimate = estimate_monthly_cost(**scenario["kwargs"])
        costs = estimate["costs"]
        print(f"    LLM:         ${costs['llm_total']:>8.2f}")
        print(f"    ECS Fargate: ${costs['ecs_fargate']:>8.2f}")
        print(f"    S3:          ${costs['s3']:>8.4f}")
        print(f"    ALB + NAT:   ${costs['alb'] + costs['nat_gateway']:>8.2f}")
        print(f"    RDS:         ${costs['rds']:>8.2f}")
        print(f"    ─────────────────────────")
        print(f"    TOTAL/month: ${costs['total_monthly_usd']:>8.2f}")

    # Test invalid model
    print("\n[Demo 5] Invalid model error handling")
    try:
        estimate_monthly_cost(100, 5000, 500, model="gpt-99-ultra")
    except ValueError as e:
        print(f"  ValueError caught: {e}")

    print("\nAll demos complete.")
