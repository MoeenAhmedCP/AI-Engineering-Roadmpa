# 1.7 Cloud Fundamentals for AI Apps

## Regions, Availability Zones, and VPCs

### The City Analogy

Think of AWS like a global city grid:
- A **Region** is a country (us-east-1 = Northern Virginia, eu-west-1 = Ireland). Each region is physically isolated with its own power and networking. Data doesn't leave a region unless you explicitly move it — important for GDPR compliance.
- An **Availability Zone (AZ)** is a neighborhood within that country — a separate data center complex. Each region has 2–6 AZs. Deploying across multiple AZs means your app survives a single data center failure.
- A **VPC (Virtual Private Cloud)** is your private gated community inside that country. You control who gets in (Security Groups, NACLs) and which internal streets connect where (subnets, route tables).

### Key Networking Concepts

| Concept | What It Is | Why It Matters |
|---|---|---|
| VPC | Your private network in AWS | Isolates your resources from other AWS customers |
| Public Subnet | Subnet with route to internet gateway | Where your load balancer lives |
| Private Subnet | No direct internet route | Where your ECS tasks and RDS live — not directly reachable |
| Security Group | Stateful firewall per resource | Allow only port 443 inbound, all outbound |
| NAT Gateway | Lets private subnet reach internet | ECS tasks need it to pull Docker images / call APIs |
| Internet Gateway | Connects VPC to public internet | Required for public-facing load balancers |

### Typical 3-Tier Layout for a FastAPI AI App

```
Internet
    │
    ▼
[Application Load Balancer]  ← Public Subnet (AZ-a, AZ-b)
    │
    ▼
[ECS Fargate Tasks]          ← Private Subnet (AZ-a, AZ-b)
    │                    │
    ▼                    ▼
[RDS PostgreSQL]    [S3 Bucket]  ← Accessed via VPC endpoint (no internet)
```

---

## S3 with boto3: The Core Operations

S3 (Simple Storage Service) is the standard for storing user-uploaded files, documents, ML model artifacts, and anything that doesn't belong in a relational database.

### Setup

```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    # In production on ECS, boto3 picks up the task role automatically.
    # For local dev:
    # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BUCKET = "docsense-documents"
```

### Upload a File

```python
def upload_file(file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type,
        # Server-side encryption (always enable in production)
        ServerSideEncryption="AES256",
    )
    return s3_key
```

### Download a File

```python
def download_file(s3_key: str) -> bytes:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"S3 key not found: {s3_key}") from e
        raise
```

### Presigned URLs (Give Temporary Access Without Exposing Credentials)

```python
def generate_presigned_url(s3_key: str, expiry_seconds: int = 3600) -> str:
    """
    Generate a time-limited URL that lets anyone download the file
    without needing AWS credentials. Used for sharing results,
    email attachments, frontend downloads.
    """
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": s3_key},
        ExpiresIn=expiry_seconds,
    )
```

### List Objects

```python
def list_objects(prefix: str = "") -> list[dict]:
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return response.get("Contents", [])
```

---

## IAM: Identity and Access Management

### Roles vs Users

| | IAM User | IAM Role |
|---|---|---|
| Who uses it | A human or long-lived key | A service (EC2, ECS, Lambda) |
| Credentials | Access key + secret key | Temporary credentials (auto-rotated) |
| Best for | Local development, CI/CD pipelines | Production workloads — NO keys to leak |
| Risk | Keys can be committed, stolen | No keys to steal — credentials expire |

**The golden rule**: Never use IAM users (long-lived access keys) in production. Attach an IAM role to your ECS task and boto3 picks it up automatically. Zero keys to manage, zero keys to leak.

### Principle of Least Privilege

Give each service only the exact permissions it needs and nothing more.

**Bad** (too broad):
```json
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*"
}
```

**Good** (least privilege):
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject"],
  "Resource": "arn:aws:s3:::docsense-documents/uploads/*"
}
```

The second policy lets the service read and write only within the `uploads/` prefix of a specific bucket — nothing else in AWS.

### Never Use Root or Admin Keys in Code

The root account has unlimited power with no audit trail. An admin key is almost as dangerous. If either leaks:
- Attacker can spin up GPU instances for crypto mining (thousands of dollars/hour)
- Attacker can exfiltrate all your S3 data
- Attacker can delete everything

Always create purpose-specific roles with the minimum permissions required.

---

## Lambda: When to Use for AI Workloads

AWS Lambda runs code in response to events without managing servers. Billing is per invocation + per millisecond of execution.

### Good Fits for Lambda in AI Apps

| Use Case | Why Lambda Works |
|---|---|
| Async document processing | Triggered by S3 upload event or SQS message |
| Webhook receivers (Stripe, GitHub) | Short-lived, burst traffic, event-driven |
| Scheduled jobs (nightly reports) | EventBridge cron trigger |
| Image/file preprocessing | Fan-out processing for multiple file types |

### Lambda Handler Pattern for Document Processing

```python
import json
import boto3

def handler(event: dict, context) -> dict:
    """
    Triggered by SQS. Each record contains a document_id.
    Downloads from S3, processes, stores result.
    """
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        document_id = body["document_id"]
        # process_document(document_id)  # your logic here
        print(f"Processed document: {document_id}")

    return {"statusCode": 200, "processed": len(event.get("Records", []))}
```

### The Cold Start Problem

Lambda functions that haven't been called recently are "cold" — the container needs to spin up. This adds 200ms–2s of latency.

**Mitigation strategies:**
- Keep Lambda packages small (zip only what you need)
- Use Lambda Layers for large dependencies (numpy, pandas)
- Enable Provisioned Concurrency for latency-sensitive functions
- For AI inference with heavy models (PyTorch), use ECS instead of Lambda — Lambda has a 250MB package limit and 15-minute timeout

---

## ECS Fargate: The Standard for Deploying FastAPI AI Apps

ECS (Elastic Container Service) with Fargate runs Docker containers without managing EC2 instances. Fargate handles the underlying VM — you just define CPU, memory, and your container image.

### Key Concepts

| Component | What It Is |
|---|---|
| Task Definition | Blueprint: Docker image, CPU, memory, env vars, ports |
| Task | A running instance of a task definition |
| Service | Keeps N tasks running, replaces failed tasks, integrates with ALB |
| Cluster | Logical grouping of services |
| ALB | Application Load Balancer — routes HTTP traffic to tasks |

### Sample Task Definition (simplified)

```json
{
  "family": "docsense-backend",
  "cpu": "512",
  "memory": "1024",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "executionRoleArn": "arn:aws:iam::123:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123:role/docsense-task-role",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/docsense-backend:latest",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"}
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123:secret:docsense/prod"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/docsense",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      }
    }
  ]
}
```

### Deployment Flow

```
1. git push → GitHub Actions triggers
2. docker build → push to ECR (Elastic Container Registry)
3. ecs update-service --force-new-deployment
4. ECS pulls new image, starts new task, health checks ALB target
5. ALB drains old task connections → terminates old task
6. Zero-downtime deploy complete
```

---

## Cost Awareness

Understanding what costs money prevents surprise AWS bills.

### Cost Table

| Resource | Pricing Model | Rough Cost |
|---|---|---|
| ECS Fargate (0.5 vCPU, 1GB) | Per vCPU-hour + per GB-hour | ~$15–20/month always-on |
| EC2 t3.small | Per instance-hour | ~$15/month |
| S3 Storage | Per GB/month | ~$0.023/GB |
| S3 PUT requests | Per 1000 requests | $0.005/1000 |
| S3 GET requests | Per 1000 requests | $0.0004/1000 |
| Data Transfer Out | Per GB | $0.09/GB (first 10TB) |
| RDS t3.micro (PostgreSQL) | Per instance-hour + storage | ~$25/month |
| ALB | Per hour + per LCU | ~$16/month base |
| NAT Gateway | Per hour + per GB | ~$32/month + data |
| Claude Sonnet (input) | Per million tokens | ~$3/million |
| Claude Sonnet (output) | Per million tokens | ~$15/million |
| Lambda | Per million invocations | $0.20/million |

### Biggest Cost Surprises for AI Apps

1. **LLM tokens**: Analyzing a 50-page PDF can cost $0.10–0.50 per document. At 10,000 documents/month, that's $1,000–5,000 just in API costs.
2. **Data transfer**: Downloading large documents from S3 to your ECS task is charged. Keep your ECS tasks in the same region as your S3 bucket.
3. **NAT Gateway**: A commonly forgotten cost. Every GB of outbound traffic from private subnets costs $0.045. An app pulling lots of data can generate hundreds in NAT costs.
4. **Idle Fargate tasks**: If you're running 2 tasks 24/7 for redundancy, you're paying even at 3 AM with zero traffic. Use ECS auto-scaling to scale down to 1 task overnight.

---

## Railway/Render vs ECS: Decision Guide

| Factor | Railway / Render | ECS Fargate |
|---|---|---|
| Setup time | 5 minutes | 2–4 hours |
| Monthly cost (small app) | $5–20 | $60–120 (ALB + NAT + tasks) |
| Scaling | Automatic | Manual config or auto-scaling policy |
| Custom VPC/networking | No | Full control |
| Compliance (HIPAA, SOC2) | Limited | Supported |
| Multi-region | No | Yes |
| CI/CD | Built-in | Custom (GitHub Actions + ECR) |
| Best for | Side projects, MVPs, <10k users | Production apps, regulated industries |

**Rule of thumb**: Start on Railway/Render. Migrate to ECS when you need compliance requirements, > 50k users, cost optimization at scale, or multi-region redundancy.
