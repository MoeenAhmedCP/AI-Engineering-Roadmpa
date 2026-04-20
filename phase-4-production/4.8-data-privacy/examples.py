"""
4.8 Data Privacy in AI Systems — Examples
Implements PII scrubbing, audit logging, pseudonymization, and compliance checks.
No external dependencies required.
Run: python examples.py
"""

import hashlib
import json
import re
import time
from typing import Optional


# ---------------------------------------------------------------------------
# 1. PII Pattern Definitions
# ---------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone":       re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def scrub_pii(text: str, replacement: str = "[REDACTED]") -> tuple[str, list[str]]:
    """
    Scrub PII from text using regex patterns.

    Returns:
        (scrubbed_text, found_types) where found_types is a list of PII
        type names that were detected and replaced.
    """
    found_types: list[str] = []
    scrubbed = text

    for pii_type, pattern in PII_PATTERNS.items():
        new_text, count = pattern.subn(replacement, scrubbed)
        if count > 0:
            found_types.append(pii_type)
            scrubbed = new_text

    return scrubbed, found_types


# ---------------------------------------------------------------------------
# 2. Pseudonymization
# ---------------------------------------------------------------------------

def hash_user_id(user_id: str, salt: str) -> str:
    """
    Pseudonymize a user identifier using SHA-256 with a salt.

    The salt prevents rainbow table attacks. Store the salt securely
    (e.g., in environment variables or a secrets manager) — never in the DB.

    This is reversible only if you know the salt and the original ID,
    which makes it pseudonymization rather than full anonymization.
    """
    payload = f"{salt}:{user_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]  # 16 hex chars for readability


# ---------------------------------------------------------------------------
# 3. Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    GDPR-compliant audit log for AI interactions.
    Logs metadata only — never full prompt content (may contain PII).
    Supports right-to-erasure via delete_user_data().
    In production, back this with a database or append-only log store.
    """

    def __init__(self):
        # {user_id_hash: [log_entry, ...]}
        self._log: dict[str, list[dict]] = {}
        self._all_entries: list[dict] = []

    def log_interaction(
        self,
        user_id_hash: str,
        model: str,
        prompt_template_id: str,
        response_length: int,
        latency_ms: float,
    ) -> None:
        """
        Log an AI interaction. Note what is NOT logged:
        - Full prompt content (may contain PII)
        - Full response content (may contain sensitive inferences)
        - Raw user email or identifier
        """
        entry = {
            "user_id_hash": user_id_hash,
            "timestamp": time.time(),
            "model": model,
            "prompt_template_id": prompt_template_id,
            "response_length_bytes": response_length,
            "latency_ms": round(latency_ms, 1),
        }

        if user_id_hash not in self._log:
            self._log[user_id_hash] = []
        self._log[user_id_hash].append(entry)
        self._all_entries.append(entry)

    def get_user_log(self, user_id_hash: str) -> list[dict]:
        """Retrieve all log entries for a given user (by hashed ID)."""
        return self._log.get(user_id_hash, [])

    def delete_user_data(self, user_id_hash: str) -> int:
        """
        GDPR Article 17 — Right to Erasure.
        Deletes all log entries for the given user.
        Returns the number of entries deleted.
        """
        if user_id_hash not in self._log:
            return 0
        count = len(self._log[user_id_hash])
        # Remove from per-user index
        del self._log[user_id_hash]
        # Remove from global log
        self._all_entries = [
            e for e in self._all_entries
            if e["user_id_hash"] != user_id_hash
        ]
        return count

    def total_entries(self) -> int:
        return len(self._all_entries)

    def summary(self) -> dict:
        return {
            "total_entries": self.total_entries(),
            "unique_users": len(self._log),
        }


# ---------------------------------------------------------------------------
# 4. Terms of Service / Compliance Check
# ---------------------------------------------------------------------------

def check_terms_compliance(text: str) -> dict:
    """
    Basic heuristic compliance check before sending text to an LLM API.
    Returns {compliant: bool, issues: list[str]}.

    This is NOT a substitute for legal review. It catches obvious violations:
    - SSNs and credit card numbers should never be sent to external APIs.
    - HIPAA-sensitive terms signal potential PHI.
    - Profanity / hate speech violates most ToS.
    """
    issues: list[str] = []

    # Check for financial PII
    if PII_PATTERNS["ssn"].search(text):
        issues.append("SSN detected — do not send to external LLM API (GDPR, HIPAA, CCPA)")
    if PII_PATTERNS["credit_card"].search(text):
        issues.append("Credit card number detected — do not send to external LLM API (PCI-DSS)")

    # Check for HIPAA-sensitive medical terms (simplified signal)
    hipaa_terms = re.compile(
        r"\b(diagnosis|prescription|medical record|patient id|dob|date of birth|"
        r"social security|medicare|medicaid|insurance number)\b",
        re.IGNORECASE,
    )
    if hipaa_terms.search(text):
        issues.append("Possible PHI/HIPAA-sensitive content detected — review before sending")

    # Check for prohibited content signals
    if re.search(r"\b(bomb|explosives|weapon|assassination)\b", text, re.IGNORECASE):
        issues.append("Potentially prohibited content detected — review against API ToS")

    return {
        "compliant": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# 5. Privacy Demo
# ---------------------------------------------------------------------------

def privacy_demo():
    print("=" * 65)
    print("  DATA PRIVACY IN AI SYSTEMS — DEMO")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 5a. PII Scrubbing
    # -----------------------------------------------------------------------
    print("\n--- PII SCRUBBING ---\n")

    test_texts = [
        "Hi, my name is Jane and my email is jane.doe@example.com. Please help.",
        "My phone is (555) 867-5309 and my SSN is 123-45-6789.",
        "Charge my card 4111111111111111 for the order.",
        "I'm connecting from 192.168.1.105. Is that blocked?",
        "I'd like to return the shoes I bought. Order was placed last Tuesday.",
    ]

    for i, text in enumerate(test_texts, 1):
        scrubbed, found = scrub_pii(text)
        print(f"  [{i}] Original:  {text}")
        print(f"       Scrubbed:  {scrubbed}")
        if found:
            print(f"       PII found: {', '.join(found)}")
        else:
            print(f"       PII found: none")
        print()

    # -----------------------------------------------------------------------
    # 5b. Pseudonymization
    # -----------------------------------------------------------------------
    print("--- PSEUDONYMIZATION ---\n")
    SECRET_SALT = "my-app-salt-2024"  # In production: load from env var

    users = ["alice@example.com", "bob@company.org", "charlie@email.net"]
    for user in users:
        hashed = hash_user_id(user, SECRET_SALT)
        print(f"  {user:<30} → {hashed}")
    print()

    # Show that same user always gets same hash
    user = "alice@example.com"
    h1 = hash_user_id(user, SECRET_SALT)
    h2 = hash_user_id(user, SECRET_SALT)
    print(f"  Deterministic: hash(alice, salt1) called twice → {h1} == {h2}: {h1 == h2}")

    # Show that different salts produce different hashes
    h3 = hash_user_id(user, "different-salt")
    print(f"  Salt isolation: hash(alice, salt2) → {h3}  (different from {h1})")
    print()

    # -----------------------------------------------------------------------
    # 5c. Audit Logging
    # -----------------------------------------------------------------------
    print("--- AUDIT LOGGING ---\n")

    logger = AuditLogger()
    salt = SECRET_SALT

    # Simulate interactions from 3 users
    interactions = [
        ("alice@example.com",   "claude-sonnet-4-6", "support_v2", 312,  450.2),
        ("alice@example.com",   "claude-sonnet-4-6", "support_v2", 198,  380.5),
        ("bob@company.org",     "claude-haiku-3",    "faq_v1",     87,   95.1),
        ("charlie@email.net",   "claude-sonnet-4-6", "support_v2", 445,  510.8),
        ("bob@company.org",     "claude-haiku-3",    "faq_v1",     102,  88.3),
        ("alice@example.com",   "claude-sonnet-4-6", "support_v3", 289,  430.0),
    ]

    for raw_id, model, template, resp_len, latency in interactions:
        uid_hash = hash_user_id(raw_id, salt)
        logger.log_interaction(uid_hash, model, template, resp_len, latency)

    print(f"  Logged {logger.total_entries()} interactions from {logger.summary()['unique_users']} users")
    print()

    # Show Alice's log
    alice_hash = hash_user_id("alice@example.com", salt)
    alice_log = logger.get_user_log(alice_hash)
    print(f"  Alice's log ({len(alice_log)} entries):")
    for entry in alice_log:
        print(f"    model={entry['model']}  template={entry['prompt_template_id']}  "
              f"latency={entry['latency_ms']}ms  resp_len={entry['response_length_bytes']}B")
    print()

    # Exercise right to erasure for Alice
    deleted = logger.delete_user_data(alice_hash)
    print(f"  Right to erasure: deleted {deleted} entries for Alice")
    print(f"  Remaining log entries: {logger.total_entries()}")
    print()

    # -----------------------------------------------------------------------
    # 5d. Terms Compliance Check
    # -----------------------------------------------------------------------
    print("--- TERMS COMPLIANCE CHECK ---\n")

    compliance_tests = [
        "What is the weather like in Paris?",
        "My SSN is 123-45-6789, can you help me file taxes?",
        "Patient diagnosis: Type 2 diabetes. Prescription: Metformin 500mg.",
        "My credit card 4111111111111111 was charged twice.",
        "How do I implement a binary search tree in Python?",
    ]

    for text in compliance_tests:
        result = check_terms_compliance(text)
        status = "PASS" if result["compliant"] else "FAIL"
        print(f"  [{status}] {text[:50]}{'...' if len(text) > 50 else ''}")
        for issue in result["issues"]:
            print(f"         Issue: {issue}")
    print()

    print("=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print("""
  1. Scrub PII before sending to external LLM APIs
  2. Pseudonymize user IDs in logs using HMAC/SHA-256 with a salt
  3. Log metadata only — not full prompt/response content
  4. Implement right-to-erasure: delete by hashed user ID
  5. Check compliance before sending sensitive data to APIs
  6. EU data must stay in EU — use Azure OpenAI or on-premises models
  """)


if __name__ == "__main__":
    privacy_demo()
