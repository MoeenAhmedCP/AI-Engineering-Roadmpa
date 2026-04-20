# 4.8 Data Privacy in AI Systems

## Why Privacy Matters More for AI

Traditional software might log a request path and a status code. AI systems log full conversation
transcripts, which can include names, diagnoses, financial details, and intimate personal concerns.
A single data breach or misuse incident can destroy user trust and result in massive regulatory fines.
Understanding the legal and technical landscape is non-negotiable for production AI systems.

---

## GDPR Core Principles

The General Data Protection Regulation (EU 2016/679) establishes principles that apply whenever you
process data about EU residents, regardless of where your company is based:

1. **Lawfulness, fairness, transparency** — Users must know you are processing their data and why.
2. **Purpose limitation** — Data collected for customer support cannot be repurposed to train marketing
   models without separate consent.
3. **Data minimisation** — Only collect what is strictly necessary. If a user's zip code is not needed
   to answer their question, don't log it.
4. **Accuracy** — Keep stored data correct and current.
5. **Storage limitation** — Delete data when it's no longer needed. Don't keep chat logs forever "just
   in case."
6. **Integrity and confidentiality** — Encrypt data at rest and in transit.
7. **Accountability** — You must be able to demonstrate compliance with all of the above.

Violations can result in fines up to 4% of annual global turnover or €20 million, whichever is higher.

---

## What Counts as PII in AI Systems

PII (Personally Identifiable Information) is broader than most engineers assume:

**Direct identifiers:**
- Full name, email address, phone number
- Social Security / National Insurance / Tax ID numbers
- Passport or driver's license numbers
- Credit card / bank account numbers

**Indirect identifiers (can identify someone when combined):**
- IP address (considered PII under GDPR)
- Device ID, cookie ID, advertising ID
- Location data (home address, GPS coordinates)
- Behavioral data (browsing history, purchase patterns)
- Biometric data (voice recordings, face images)

**Sensitive special categories (extra protections):**
- Health and medical data
- Racial or ethnic origin
- Political opinions, religious beliefs
- Sexual orientation

In AI systems, users often volunteer sensitive information naturally in conversation: "I'm diabetic and
taking metformin, will this medication interact?" That's health data, even though you didn't explicitly
ask for it.

---

## Handling PII in Practice

### Scrub Before Sending to the LLM API

Never send raw user input directly to a third-party API if it contains PII you don't need the model
to see. Run a PII scrubber on the input first:

```python
scrubbed, found = scrub_pii(user_message)
response = call_llm(scrubbed)  # LLM never sees the original PII
```

This applies to external APIs (OpenAI, Anthropic). If you use an on-premises model, you have more
control, but scrubbing is still good practice for logging purposes.

### Pseudonymize in Logs

Never log raw user IDs or email addresses. Hash them with a salt:

```python
user_log_key = hash_user_id(user_email, salt=SECRET_SALT)
logger.info({"user": user_log_key, "action": "query", "model": "..."})
```

This allows you to correlate logs for a specific user (using the hash) without exposing the original
identifier. It also enables right-to-erasure compliance — delete by hash, no need to scan for email.

### Encrypt at Rest

All databases and object storage containing user data must use encryption at rest. For cloud databases
(RDS, Supabase, Cloud SQL), this is typically enabled by default. Verify it. For logs shipped to
S3 or GCS, use server-side encryption.

---

## Data Residency

GDPR generally requires that EU personal data stays within the EU or countries with "adequate protection"
(UK, Canada, Japan, and a few others). Sending EU user data to a US-based LLM API creates compliance risk.

**Options:**
- **Azure OpenAI Service** — Run GPT-4 inside your Azure tenant in the EU region. Data stays in your
  subscription, does not cross to OpenAI's shared infrastructure.
- **Anthropic API (EU)** — Check current data processing agreements; Anthropic provides DPAs (Data
  Processing Agreements) for enterprise customers.
- **On-premises open-source models** — Run Llama, Mistral, or Qwen on your own GPU infrastructure.
  Data never leaves your datacenter. Required in healthcare (HIPAA), defense, and some banking contexts.

---

## On-Premises Open-Source: When Cloud APIs Are Prohibited

Certain sectors cannot use cloud LLM APIs regardless of data residency agreements:

- **Healthcare (HIPAA):** Patient health information (PHI) has strict rules. Many healthcare orgs run
  Llama 3 or Mistral on local NVIDIA DGX systems rather than use any cloud API.
- **Defense / classified:** Classified information cannot touch public cloud infrastructure. Period.
- **Banking / financial:** Some jurisdictions require data to stay within country-specific infrastructure
  with specific audit requirements that cloud providers cannot satisfy.

In these contexts, look at vLLM, Ollama, or TGI (Text Generation Inference) as serving frameworks for
open-source models.

---

## Audit Logging

Every AI interaction should be logged for compliance and debugging, but with care:

**Log:**
- User identifier (hashed/pseudonymized)
- Timestamp
- Model used
- Prompt template ID (not full prompt if it contains PII)
- Response length (bytes/tokens)
- Latency
- Session ID (hashed)

**Do not log:**
- Full prompt content (may contain PII)
- Full response content (may contain PII or sensitive inferences)
- Raw user identifiers

Keeping audit logs separate from application logs is good practice. Audit logs may need to be retained
for 3–7 years for compliance, while application logs can be rotated more aggressively.

---

## Right to Erasure (Article 17 GDPR)

Users have the right to request deletion of all their data. In AI systems, this is more complex than
deleting a database row:

1. **Chat history / database rows:** Easy — delete by user_id.
2. **Application logs:** Harder — requires log scanning or using hashed IDs that can be deleted by key.
3. **Vector embeddings:** Hard — if you embedded user queries for semantic search or personalization,
   those vectors may contain information. You must track which vectors belong to which user and delete them.
4. **Backups:** Very hard — restoring a backup may re-introduce deleted data. Your backup retention
   policy must account for erasure obligations.
5. **Model weights (training data):** Extremely hard — if user data was used in fine-tuning, the model
   itself encodes that information. This is an active area of ML research (machine unlearning).

---

## OpenAI / Anthropic Terms of Service

Key commercial use points:
- **Training opt-out:** Enterprise/API tiers (not the default ChatGPT interface) generally do not use
  your data for training. Verify with your specific plan.
- **Data retention:** OpenAI retains API inputs/outputs for up to 30 days for safety monitoring, then
  deletes. Check the current policy.
- **Content restrictions:** You cannot use the API to generate CSAM, facilitate violence, or undermine
  AI oversight — these are hard blocks, not soft guidelines.
- **Prohibited use cases:** Certain industries (payday lending, certain financial products) have
  restrictions. Check the usage policies before building.

---

## EU AI Act Overview

The EU AI Act (effective 2024–2026, phased enforcement) categorizes AI systems by risk:

**Prohibited (Article 5):**
- Real-time biometric identification in public spaces
- Social scoring by governments
- Subliminal manipulation of behavior
- Emotion recognition in workplaces/schools

**High-risk (Annex III):**
- AI in hiring decisions, credit scoring, medical diagnosis, law enforcement
- Must: register in EU database, conduct conformity assessment, maintain technical documentation,
  implement human oversight, ensure accuracy and robustness

**General-purpose AI (Foundation models like GPT-4, Claude):**
- Must maintain training data documentation, respect copyright, publish capability evaluations
- Systemic-risk models (> 10^25 FLOPs training compute) face additional obligations

**Minimal risk:** Chatbots, spam filters — mostly transparency obligations (users must know they're
talking to an AI).

---

## Summary Checklist

- [ ] PII scrubbing before sending to external LLM APIs
- [ ] Pseudonymized user IDs in all logs
- [ ] Encryption at rest and in transit
- [ ] Data residency verified (EU data stays in EU)
- [ ] Audit logs without full prompt/response content
- [ ] Right-to-erasure procedure documented and tested
- [ ] DPA (Data Processing Agreement) signed with any third-party AI provider
- [ ] Privacy policy updated to disclose AI usage
- [ ] EU AI Act risk classification assessed for your system
