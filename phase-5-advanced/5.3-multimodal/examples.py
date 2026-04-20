"""
5.3 Multimodal AI — Examples
Demonstrates vision message construction, invoice extraction, image search, and audio.
SIMULATE=True by default — no API keys or image files needed.
Run: python examples.py
"""

import base64
import hashlib
import json
import math
import os
from typing import Optional

SIMULATE = os.getenv("SIMULATE", "true").lower() != "false"


# ---------------------------------------------------------------------------
# 1. Image Encoding
# ---------------------------------------------------------------------------

def encode_image_base64(image_path: str) -> str:
    """
    Read an image file and return its base64-encoded string.
    Used to embed images directly in API requests.
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            f"Tip: Place a JPEG or PNG image at this path to test real encoding."
        )


def detect_media_type(image_path: str) -> str:
    """Infer media type from file extension."""
    ext = image_path.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# 2. Vision Message Builder
# ---------------------------------------------------------------------------

def build_vision_message(
    prompt: str,
    image_b64: Optional[str] = None,
    image_url: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Build an API message dict that includes an image and a text prompt.
    Supports both base64 and URL image sources.

    Returns the message dict in Anthropic format (easily adapted for OpenAI).
    """
    content = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        })
    elif image_url:
        # Anthropic uses "url" source type; OpenAI uses "image_url" type
        content.append({
            "type": "image",
            "source": {
                "type": "url",
                "url": image_url,
            },
        })

    content.append({"type": "text", "text": prompt})

    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# 3. Invoice Field Extraction
# ---------------------------------------------------------------------------

def extract_invoice_fields(image_data: str, simulate: bool = True) -> dict:
    """
    Extract structured fields from an invoice image.

    In production: pass image_data (base64) to a VLM with a structured
    extraction prompt. Here we return mock data.

    Returns:
        {vendor, invoice_number, date, due_date, subtotal, tax, total, line_items}
    """
    if simulate:
        return {
            "vendor": "Acme Supplies Co.",
            "invoice_number": "INV-2024-00847",
            "date": "2024-11-15",
            "due_date": "2024-12-15",
            "subtotal": 1250.00,
            "tax": 112.50,
            "total": 1362.50,
            "currency": "USD",
            "line_items": [
                {"description": "Widget Type A", "qty": 10, "unit_price": 75.00, "amount": 750.00},
                {"description": "Widget Type B", "qty": 5,  "unit_price": 80.00, "amount": 400.00},
                {"description": "Shipping & Handling", "qty": 1, "unit_price": 100.00, "amount": 100.00},
            ],
        }

    # Production path (requires ANTHROPIC_API_KEY)
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = build_vision_message(
            prompt=(
                "Extract all fields from this invoice as JSON. Include: "
                "vendor, invoice_number, date, due_date, subtotal, tax, total, currency, "
                "and line_items (each with description, qty, unit_price, amount)."
            ),
            image_b64=image_data,
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[message],
        )
        # Parse JSON from response
        text = response.content[0].text
        json_match = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse JSON from response", "raw": text}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 4. CLIP Simulation — Image Search Engine
# ---------------------------------------------------------------------------

def fake_clip_embed(text: str) -> list[float]:
    """
    Simulate a 64-dimensional CLIP text embedding.
    In production, use the real CLIP model or OpenAI's text-embedding-3-small.

    This is deterministic: same text → same vector, enabling reproducible search tests.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Use 32 bytes → 32 floats, then repeat/extend to 64 dims
    floats = [(b - 127.5) / 127.5 for b in digest]
    # Double to 64 dims with slight variation
    extended = floats + [(f * 0.9 + 0.1) for f in floats]
    # Normalize
    magnitude = sum(x * x for x in extended) ** 0.5
    if magnitude == 0:
        return extended
    return [x / magnitude for x in extended]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = sum(a * a for a in v1) ** 0.5
    m2 = sum(b * b for b in v2) ** 0.5
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


class ImageSearchEngine:
    """
    Image search engine that uses text descriptions + CLIP embeddings.
    In production: store real CLIP image embeddings in a vector database (Pinecone, pgvector).
    Here we simulate with description embeddings.
    """

    def __init__(self):
        self._images: list[dict] = []

    def add_image(self, path: str, description: str) -> None:
        """
        Index an image by embedding its description.
        In production, also compute the actual image embedding using CLIP.
        """
        embedding = fake_clip_embed(description)
        self._images.append({
            "path": path,
            "description": description,
            "embedding": embedding,
        })

    def search(self, text_query: str, k: int = 3) -> list[dict]:
        """
        Search for images by text query. Returns top-k matches with similarity scores.
        """
        query_vec = fake_clip_embed(text_query)
        scored = []
        for img in self._images:
            score = cosine_similarity(query_vec, img["embedding"])
            scored.append({
                "path": img["path"],
                "description": img["description"],
                "similarity": round(score, 4),
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# 5. Audio Demo
# ---------------------------------------------------------------------------

def transcribe_audio_demo(simulate: bool = True) -> None:
    """
    Demonstrates the Whisper API call pattern.
    In simulation, shows the code structure without making a real API call.
    """
    print("--- WHISPER AUDIO TRANSCRIPTION ---\n")

    code_example = '''
# Production Whisper API call (requires OPENAI_API_KEY):
import openai

client = openai.OpenAI()

with open("meeting_recording.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",   # includes timestamps
        timestamp_granularities=["word"], # word-level timestamps
        language="en",                    # optional: hint for language
    )

print(transcript.text)
# "Good morning everyone, let's get started with the Q3 review..."

# With diarization (using pyannote.audio):
# 1. Run speaker diarization to get speaker segments
# 2. Run Whisper to get word-level timestamps
# 3. Align: assign each word to a speaker based on timestamp overlap
# Result: "Speaker 1 (0:00–0:15): Good morning everyone..."
    '''

    print(code_example)

    if simulate:
        print("  [SIMULATED] Transcription result:")
        print("  'Good morning team. Today we will review Q3 results.")
        print("   Revenue was up 12% year over year. Marketing spend...")
        print()


# ---------------------------------------------------------------------------
# 6. Full Multimodal Demo
# ---------------------------------------------------------------------------

def multimodal_demo():
    print("=" * 65)
    print("  5.3 MULTIMODAL AI — EXAMPLES DEMO")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 6a. Vision Message Format
    # -----------------------------------------------------------------------
    print("\n--- VISION MESSAGE FORMAT ---\n")

    # Show URL-based message
    url_msg = build_vision_message(
        prompt="What does this chart show? Summarize the key trend.",
        image_url="https://example.com/q3-revenue-chart.png",
    )
    print("  URL-based message structure:")
    # Print without the full base64 data
    print(json.dumps(url_msg, indent=4))

    # Show base64-based message structure (with placeholder data)
    b64_msg = build_vision_message(
        prompt="Extract all text from this invoice.",
        image_b64="[BASE64_DATA_HERE]",
        media_type="image/jpeg",
    )
    print("\n  Base64-based message structure:")
    print(json.dumps(b64_msg, indent=4))

    # -----------------------------------------------------------------------
    # 6b. Invoice Extraction
    # -----------------------------------------------------------------------
    print("\n--- INVOICE FIELD EXTRACTION ---\n")

    fields = extract_invoice_fields(image_data="[simulated]", simulate=True)
    print("  Extracted invoice fields:")
    print(f"    Vendor:          {fields['vendor']}")
    print(f"    Invoice #:       {fields['invoice_number']}")
    print(f"    Date:            {fields['date']}")
    print(f"    Due Date:        {fields['due_date']}")
    print(f"    Subtotal:       ${fields['subtotal']:,.2f}")
    print(f"    Tax:            ${fields['tax']:,.2f}")
    print(f"    Total:          ${fields['total']:,.2f}")
    print(f"\n    Line Items ({len(fields['line_items'])}):")
    for item in fields["line_items"]:
        print(f"      {item['description']:<30} qty={item['qty']}  "
              f"@ ${item['unit_price']:.2f}  = ${item['amount']:.2f}")

    # -----------------------------------------------------------------------
    # 6c. Image Search with CLIP-like Embeddings
    # -----------------------------------------------------------------------
    print("\n--- IMAGE SEARCH ENGINE (CLIP Simulation) ---\n")

    engine = ImageSearchEngine()

    # Index a collection of images with descriptions
    image_catalog = [
        ("./photos/invoice_2024_001.jpg",  "business invoice with line items and totals"),
        ("./photos/dog_park.jpg",          "golden retriever dog playing fetch in a park"),
        ("./photos/revenue_chart.jpg",     "bar chart showing quarterly revenue growth"),
        ("./photos/office_desk.jpg",       "modern office desk with laptop and coffee"),
        ("./photos/cat_sleeping.jpg",      "orange tabby cat sleeping on a sofa"),
        ("./photos/pie_chart.jpg",         "pie chart showing market share breakdown"),
        ("./photos/receipt_coffee.jpg",    "coffee shop receipt with itemized purchases"),
        ("./photos/dog_portrait.jpg",      "portrait of a labrador puppy outdoors"),
    ]

    for path, desc in image_catalog:
        engine.add_image(path, desc)

    print(f"  Indexed {len(image_catalog)} images\n")

    queries = [
        "find images of dogs",
        "financial documents and receipts",
        "data visualizations and charts",
    ]

    for query in queries:
        print(f"  Query: '{query}'")
        results = engine.search(query, k=3)
        for rank, r in enumerate(results, 1):
            print(f"    #{rank}: {r['path']:<40}  sim={r['similarity']:.4f}")
            print(f"        '{r['description']}'")
        print()

    # -----------------------------------------------------------------------
    # 6d. Audio Transcription Demo
    # -----------------------------------------------------------------------
    transcribe_audio_demo(simulate=True)

    # -----------------------------------------------------------------------
    # 6e. Token Cost Estimates
    # -----------------------------------------------------------------------
    print("--- IMAGE TOKEN COST ESTIMATES ---\n")

    print("  Image size and approximate token cost (Claude):")
    sizes = [
        ("64×64 (thumbnail)",    64,   64),
        ("256×256 (small)",     256,  256),
        ("512×512 (medium)",    512,  512),
        ("1024×1024 (standard)", 1024, 1024),
        ("2048×2048 (high-res)", 2048, 2048),
    ]
    print(f"  {'Size':<25} {'Pixels':<12} {'~Tokens':<10} {'~Cost @$3/1Mtok'}")
    print("  " + "-" * 60)
    for label, w, h in sizes:
        # Approximate: tiles of 512x512, each ~1601 tokens
        tiles = math.ceil(w / 512) * math.ceil(h / 512)
        tokens = tiles * 1601 + 85  # base tokens
        cost = tokens * 3 / 1_000_000
        print(f"  {label:<25} {w*h:<12,} {tokens:<10,} ${cost:.5f}")
    print()

    print("=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print("""
  1. Use base64 for private images, URLs for public images
  2. VLMs excel at complex document extraction (invoices, forms)
  3. CLIP maps text and images to the same embedding space
  4. Whisper is the standard for speech-to-text
  5. Video = frame sampling + batch image processing
  6. Images are expensive: ~1,000+ tokens each — cache aggressively
  """)


if __name__ == "__main__":
    multimodal_demo()
