# 5.3 Multimodal AI

## Vision-Language Models

Modern frontier LLMs are not text-only. Vision-language models (VLMs) accept both images and text
as input, enabling a wide range of applications that were previously impossible without specialized
computer vision pipelines.

**Current major VLMs:**
- **GPT-4o (OpenAI)** — Strong at OCR, chart reading, diagram interpretation. Accepts images via URL
  or base64. ~1,000–1,700 tokens per image for API cost.
- **Claude 3 / 3.5 / 3.7 Sonnet (Anthropic)** — Excellent document understanding, nuanced description.
  Supports JPEG, PNG, GIF, WEBP. Strong at structured data extraction from images.
- **Gemini 1.5 Pro (Google)** — Supports images, audio, video, and documents natively. Largest context
  window (1M tokens), useful for video analysis.

All three use a **visual encoder** to convert pixel data into token representations that the language
model can reason over, similar to how text tokens are embedded.

---

## Passing Images to the API

### Base64 Encoding

Encode the image file as a base64 string and embed it directly in the API request:

```python
import base64

with open("invoice.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

message = {
    "role": "user",
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            },
        },
        {"type": "text", "text": "Extract all fields from this invoice."},
    ],
}
```

**Use base64 when:** The image is on your server, is private/confidential, or needs to be sent
without exposing a public URL.

### URL References

Pass a publicly accessible URL instead of the image data:

```python
message = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
        {"type": "text", "text": "Describe this chart."},
    ],
}
```

**Use URL when:** The image is already hosted publicly, you want to reduce request payload size,
and you don't mind the API fetching the image from your CDN.

---

## Use Cases

### Document Extraction (Invoices, Forms, Tables)

Traditional OCR tools (Tesseract, AWS Textract) struggle with complex layouts — multi-column invoices,
handwritten fields, mixed fonts, rotated text, tables with merged cells. VLMs handle these naturally
because they understand layout context, not just isolated character recognition.

For invoices: the model can extract vendor name, date, line items, subtotal, tax, and total without
any custom template or field coordinates.

### Visual Q&A

Ask questions about image content: "Is there a fire extinguisher visible in this kitchen photo?"
or "Does this X-ray show any abnormalities?" (with appropriate caveats for medical use).

### Chart and Graph Reading

"What was the highest revenue quarter in this bar chart?" VLMs can read axis labels, identify
data points, and provide quantitative answers from visualizations — useful for automated report
analysis.

### LLM-Based OCR

For documents with challenging layouts, sending the image to a VLM with the prompt "transcribe all
text exactly as it appears" often outperforms traditional OCR. The model understands context
(it knows "M.D." after a name is a title, not a random abbreviation) and produces cleaner output.

---

## Audio: Whisper and TTS

**OpenAI Whisper** is the standard for speech-to-text:
- Open-source, runs locally or via API
- Supports 100+ languages
- Returns transcription with word-level timestamps
- Fine-tunable for domain-specific vocabulary

```python
import openai
with open("audio.mp3", "rb") as f:
    transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
print(transcript.text)
```

**TTS (Text-to-Speech):** OpenAI's TTS API generates natural-sounding speech from text.
Voices: alloy, echo, fable, onyx, nova, shimmer. Outputs MP3/FLAC.

**Diarization:** Speaker separation — identifying which words were spoken by which speaker in
a multi-speaker audio recording. Libraries like pyannote.audio handle this. Combine with Whisper
for full speaker-attributed transcripts.

---

## Video: Frame Sampling Strategy

No current major API accepts video files directly. The standard approach:

1. **Extract frames** at a fixed interval (e.g., 1 frame per second) using OpenCV or ffmpeg
2. **Filter to key frames** — detect scene changes, avoid near-duplicate frames
3. **Send selected frames** as a batch of images with a prompt

```python
# Extract 1 frame per second with ffmpeg
# ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg
```

For a 2-minute video at 1 fps, you get 120 frames. At ~1,000 tokens each, that's 120,000 tokens —
expensive. Use scene-change detection to reduce this to 10–20 representative frames.

Gemini 1.5 Pro has native video understanding in its API, handling frame sampling internally.

---

## Open-Source Vision Models

**LLaVA (Large Language and Vision Assistant):** One of the first open-source VLMs. Uses a
visual encoder (CLIP) connected to a language model (Llama, Mistral). Can run on consumer GPUs.

**Qwen-VL (Alibaba):** Strong performance on document understanding and OCR tasks. Available via
Hugging Face.

**Pixtral (Mistral):** Mistral's multimodal model. 12B parameters, strong on technical documents.

**InternVL:** Competitive with GPT-4V on many benchmarks. Open weights.

Run with:
```python
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
```

---

## CLIP: Image Embeddings for Visual Search

CLIP (Contrastive Language-Image Pre-training, OpenAI) is trained to map images and text into
the same embedding space. A photo of a dog and the text "a golden retriever playing fetch" will
have high cosine similarity.

**Use cases:**
- **Image search by text query:** Store image embeddings in a vector database. Query with text.
- **Zero-shot image classification:** Compare image embedding to text class descriptions.
- **Multimodal RAG:** Embed both images and text, retrieve by either modality.

```python
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed text
inputs = processor(text=["a photo of a cat"], return_tensors="pt")
text_features = model.get_text_features(**inputs)

# Embed image
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)

# Similarity
similarity = (text_features @ image_features.T)
```

---

## Multimodal RAG

Extend standard RAG to handle image-heavy document collections:

1. **Ingest:** For each document page, extract text and generate image embeddings for figures/charts
2. **Index:** Store text embeddings in one vector collection, image embeddings in another
3. **Retrieve:** On query, search both collections; merge results
4. **Generate:** Pass retrieved text chunks + relevant images to the VLM for answer generation

This allows queries like "explain the architecture diagram in the Q3 report" to retrieve the
actual diagram image and pass it to the model.

---

## Limitations and Costs

**Token cost:** Images are expensive compared to text.
- GPT-4o: ~765–1,105 tokens for a standard image (low detail: 85 tokens)
- Claude: images count toward context; a 1024×1024 image ≈ 1,600 tokens

**Maximum image size:**
- Most APIs: max 5MB per image, 2048×2048 pixels
- Higher resolution is downscaled automatically

**No video (most APIs):** Must frame-sample manually for all APIs except Gemini.

**No audio understanding (VLMs):** Audio must go through a separate STT step; VLMs don't directly
process audio waveforms.

**Layout accuracy:** VLMs can misread text in complex layouts or low-resolution images. For
production document extraction, validate outputs against known patterns (regex for amounts, dates).
