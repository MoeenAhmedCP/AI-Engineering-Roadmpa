"""
3.8 Structured Output — Examples
=================================
Demonstrates Pydantic schemas, validation/retry logic, batch extraction,
and an Instructor-style wrapper — all without real API calls (SIMULATE=True).

Run: python examples.py
"""

import json
import re
import random
from typing import Optional

SIMULATE = True

# ---------------------------------------------------------------------------
# Optional dependency: pydantic
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("[WARN] pydantic not installed — using dict-based stubs. `pip install pydantic`")

# ---------------------------------------------------------------------------
# Optional dependency: instructor
# ---------------------------------------------------------------------------
try:
    import instructor  # noqa: F401
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

# ===========================================================================
# 1. Pydantic Models
# ===========================================================================

if PYDANTIC_AVAILABLE:
    class ContactInfo(BaseModel):
        name: str = Field(description="Full name of the person")
        email: str = Field(description="Email address")
        company: str = Field(description="Company or organisation name")
        phone: Optional[str] = Field(None, description="Phone number, if present")

        @field_validator("email")
        @classmethod
        def email_must_contain_at(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("email must contain '@'")
            return v.lower().strip()

    class ActionItem(BaseModel):
        owner: str = Field(description="Person responsible for this action")
        task: str = Field(description="Description of the task to complete")
        deadline: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format or None")

    class MeetingNotes(BaseModel):
        title: str = Field(description="Meeting title or subject")
        date: str = Field(description="Date of the meeting YYYY-MM-DD")
        attendees: list[str] = Field(description="List of attendee names")
        action_items: list[ActionItem] = Field(default_factory=list)
        summary: Optional[str] = Field(None, description="Brief summary of decisions made")

else:
    # Minimal dict-based stubs when pydantic is absent
    class _DictModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def model_dump(self):
            return self.__dict__

    class ContactInfo(_DictModel):
        pass

    class ActionItem(_DictModel):
        pass

    class MeetingNotes(_DictModel):
        pass

    class ValidationError(Exception):
        pass


# ===========================================================================
# 2. Simulated LLM responses (stand-ins for real API calls)
# ===========================================================================

_CONTACT_RESPONSES = [
    {"name": "Alice Nguyen", "email": "alice@acmecorp.com", "company": "Acme Corp", "phone": "+1-555-0101"},
    {"name": "Bob Patel",    "email": "bob.patel@initech.io", "company": "Initech", "phone": None},
    {"name": "Carol Smith",  "email": "carol@globaldata.net", "company": "Global Data Inc.", "phone": "+44 20 7946 0958"},
    {"name": "David Lee",    "email": "d.lee@startup.xyz",   "company": "Startup XYZ", "phone": None},
    {"name": "Eve Torres",   "email": "eve@consultco.com",   "company": "ConsultCo", "phone": "+1-555-0202"},
]

_MEETING_RESPONSE = {
    "title": "Q2 Product Roadmap Sync",
    "date": "2026-04-15",
    "attendees": ["Alice Nguyen", "Bob Patel", "Carol Smith"],
    "action_items": [
        {"owner": "Bob Patel",   "task": "Draft technical spec for caching layer", "deadline": "2026-04-22"},
        {"owner": "Carol Smith", "task": "Schedule user interview sessions",        "deadline": "2026-04-19"},
        {"owner": "Alice Nguyen","task": "Update roadmap slide deck",               "deadline": None},
    ],
    "summary": "Team agreed to prioritise the caching layer in Q2 sprint 1. User research to start next week.",
}

# A deliberately broken response to demonstrate validation/retry
_BAD_CONTACT_RESPONSE = {
    "name": "Mallory",
    "email": "not-an-email",   # fails validator
    "company": 12345,           # wrong type (int instead of str)
    # "phone" intentionally missing — that's fine (Optional)
}


# ===========================================================================
# 3. extract_with_schema
# ===========================================================================

def extract_with_schema(text: str, schema_cls, simulate: bool = True) -> dict:
    """
    Extract structured data from `text` conforming to `schema_cls`.

    In simulate mode returns a canned response; in real mode you would call
    an LLM with the schema embedded in the prompt.

    Returns a plain dict (schema-conformant).
    """
    if simulate:
        # Pick a canned response based on schema type
        if schema_cls.__name__ == "ContactInfo":
            # Pick deterministically by hashing the text length
            idx = len(text) % len(_CONTACT_RESPONSES)
            raw = _CONTACT_RESPONSES[idx]
        elif schema_cls.__name__ == "MeetingNotes":
            raw = _MEETING_RESPONSE
        else:
            raw = {}

        # Validate through Pydantic if available
        if PYDANTIC_AVAILABLE:
            instance = schema_cls.model_validate(raw)
            return instance.model_dump()
        return raw

    # --- real path (not executed in SIMULATE=True mode) ---
    raise NotImplementedError("Set simulate=True or supply a real LLM client.")


# ===========================================================================
# 4. validate_and_retry
# ===========================================================================

def validate_and_retry(raw_json: str, schema_cls, max_retries: int = 3) -> dict:
    """
    Parse `raw_json`, validate against `schema_cls`, and show the retry
    pattern when validation fails.

    Returns the validated dict on success, or the last raw parsed dict on
    exhausting retries.
    """
    attempt_json = raw_json

    for attempt in range(1, max_retries + 1):
        print(f"\n  [validate_and_retry] Attempt {attempt}/{max_retries}")
        try:
            data = json.loads(attempt_json)
        except json.JSONDecodeError as exc:
            print(f"    JSON parse error: {exc}")
            # In a real system you would ask the LLM to fix the JSON
            print("    -> Would resend: 'Your response was not valid JSON. Please return only valid JSON.'")
            attempt_json = json.dumps({"name": "Fixed Name", "email": "fixed@example.com", "company": "Fixed Co"})
            continue

        if PYDANTIC_AVAILABLE:
            try:
                instance = schema_cls.model_validate(data)
                print(f"    Validation passed!")
                return instance.model_dump()
            except ValidationError as exc:
                print(f"    Validation error: {exc.error_count()} issue(s)")
                for err in exc.errors():
                    print(f"      Field '{'.'.join(str(l) for l in err['loc'])}': {err['msg']}")
                print("    -> Would resend error to LLM as context for self-correction.")
                # Simulate the model self-correcting on retry
                attempt_json = json.dumps(_CONTACT_RESPONSES[0])
        else:
            print("    Pydantic not available — accepting raw dict.")
            return data

    print("  [validate_and_retry] Exhausted retries — returning last parsed dict.")
    return json.loads(attempt_json)


# ===========================================================================
# 5. batch_extract
# ===========================================================================

def batch_extract(texts: list[str], schema_cls, simulate: bool = True) -> list[dict]:
    """
    Extract structured data from a list of texts.

    Returns a list of dicts (one per text). Failed extractions are stored as
    {"_error": "<message>", "_index": i} so the caller can re-queue them.
    """
    results = []
    for i, text in enumerate(texts):
        try:
            result = extract_with_schema(text, schema_cls, simulate=simulate)
            result["_index"] = i
            results.append(result)
        except Exception as exc:
            results.append({"_error": str(exc), "_index": i})
    return results


# ===========================================================================
# 6. InstructorWrapper
# ===========================================================================

class InstructorWrapper:
    """
    A minimal stub that mimics the Instructor library's interface.

    If instructor is installed, this could wrap a real client:
        client = instructor.from_anthropic(Anthropic())
        result = client.messages.create(..., response_model=ContactInfo)

    Here we simulate that behaviour using canned responses.
    """

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self._call_count = 0
        if INSTRUCTOR_AVAILABLE:
            print("  [InstructorWrapper] instructor is installed — could use real client.")
        else:
            print("  [InstructorWrapper] instructor not installed — running in stub mode.")

    def create(self, prompt: str, response_model, **kwargs) -> dict:
        """
        Mimics instructor client.messages.create(..., response_model=...).
        Returns a validated dict (or Pydantic instance if available).
        """
        self._call_count += 1
        print(f"  [InstructorWrapper] call #{self._call_count} for model '{response_model.__name__}'")

        raw = extract_with_schema(prompt, response_model, simulate=self.simulate)
        print(f"  [InstructorWrapper] extracted: {raw}")
        return raw

    def stats(self) -> dict:
        return {"total_calls": self._call_count}


# ===========================================================================
# Main demo
# ===========================================================================

def demo_contact_extraction():
    print("=" * 60)
    print("DEMO 1: Extract ContactInfo from 5 email snippets")
    print("=" * 60)

    email_snippets = [
        "Hi, I'm Alice Nguyen from Acme Corp. Reach me at alice@acmecorp.com or +1-555-0101.",
        "Bob Patel here — bob.patel@initech.io — happy to connect anytime.",
        "Carol Smith, Global Data Inc. carol@globaldata.net | +44 20 7946 0958",
        "David Lee, d.lee@startup.xyz — no phone, email only.",
        "Best, Eve Torres | ConsultCo | eve@consultco.com | +1-555-0202",
    ]

    results = batch_extract(email_snippets, ContactInfo, simulate=True)

    for r in results:
        idx = r.pop("_index", "?")
        print(f"\n  [{idx}] {r.get('name', 'N/A')} <{r.get('email', 'N/A')}>"
              f"  company={r.get('company', 'N/A')}  phone={r.get('phone', 'N/A')}")


def demo_meeting_extraction():
    print("\n" + "=" * 60)
    print("DEMO 2: Extract MeetingNotes from a transcript")
    print("=" * 60)

    transcript = """
    Q2 Product Roadmap Sync — April 15 2026
    Attendees: Alice Nguyen, Bob Patel, Carol Smith

    Alice: Let's lock in the caching layer for sprint 1.
    Bob: I'll have the technical spec ready by the 22nd.
    Carol: I'll schedule the user interviews before Thursday.
    Alice: I'll update the slide deck too — no hard deadline.
    """

    result = extract_with_schema(transcript, MeetingNotes, simulate=True)
    print(f"\n  Title    : {result['title']}")
    print(f"  Date     : {result['date']}")
    print(f"  Attendees: {', '.join(result['attendees'])}")
    print(f"  Summary  : {result.get('summary', 'N/A')}")
    print(f"  Action items ({len(result['action_items'])}):")
    for item in result["action_items"]:
        dl = item.get("deadline") or "no deadline"
        print(f"    - [{item['owner']}] {item['task']} (due: {dl})")


def demo_validation_retry():
    print("\n" + "=" * 60)
    print("DEMO 3: Validation error + retry pattern")
    print("=" * 60)

    bad_json = json.dumps(_BAD_CONTACT_RESPONSE)
    print(f"  Input JSON: {bad_json}")

    result = validate_and_retry(bad_json, ContactInfo, max_retries=3)
    print(f"\n  Final result: {result}")


def demo_instructor_wrapper():
    print("\n" + "=" * 60)
    print("DEMO 4: InstructorWrapper stub")
    print("=" * 60)

    wrapper = InstructorWrapper(simulate=True)
    result = wrapper.create("Extract contact from this email…", ContactInfo)
    print(f"  Stats: {wrapper.stats()}")


if __name__ == "__main__":
    demo_contact_extraction()
    demo_meeting_extraction()
    demo_validation_retry()
    demo_instructor_wrapper()

    print("\n" + "=" * 60)
    print("All structured-output demos complete.")
    print("=" * 60)
