"""
5.1 Fine-Tuning — Examples
==========================
Demonstrates LoRA config, dataset format conversion, training skeleton,
OpenAI fine-tuning API simulation, cost estimation, and output comparison.

Run with: python examples.py
All heavy deps (torch, peft, transformers) are wrapped in try/except.
"""

import json
import math

# ---------------------------------------------------------------------------
# 1. LoRA Configuration — annotated to explain every field
# ---------------------------------------------------------------------------

LORA_CONFIG_EXPLAINED = {
    # Which weight matrices to attach LoRA adapters to.
    # "q_proj" and "v_proj" are the query and value projection matrices
    # in the self-attention layers. Adding adapters here is standard.
    # You can also add "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    # for the feed-forward layers — more capacity, more parameters.
    "target_modules": ["q_proj", "v_proj"],

    # Rank of the LoRA decomposition.
    # LoRA update = B @ A where A is (d x r) and B is (r x d).
    # r=8: trains ~0.1% of params. r=64: trains ~0.8% of params.
    # Start with r=8. Use r=16 or r=32 for complex tasks.
    "r": 8,

    # Scaling factor. The actual update is scaled by alpha/r.
    # Setting alpha = 2*r (here: 16) is the most common convention.
    # Higher alpha = larger effective learning rate for LoRA layers.
    "lora_alpha": 16,

    # Dropout applied to LoRA layers for regularization.
    # 0.05 is a safe default. Use 0.1 for small datasets.
    "lora_dropout": 0.05,

    # Do NOT train the base model's original weights.
    # Only train the injected LoRA adapter matrices A and B.
    "bias": "none",

    # Task type tells PEFT how to handle inputs/outputs.
    # CAUSAL_LM for decoder-only models (GPT, Llama, Mistral).
    # SEQ_2_SEQ_LM for encoder-decoder models (T5, BART).
    "task_type": "CAUSAL_LM",
}

def print_lora_config():
    print("=" * 60)
    print("LoRA Configuration (with explanations)")
    print("=" * 60)
    print()
    print("from peft import LoraConfig")
    print()
    print("config = LoraConfig(")
    for key, value in LORA_CONFIG_EXPLAINED.items():
        print(f"    {key}={repr(value)},")
    print(")")
    print()

    # Show the parameter savings math
    d = 4096  # typical hidden dim for a 7B model
    r = 8
    original_params = d * d
    lora_params = 2 * (d * r)  # A matrix + B matrix
    savings_pct = (1 - lora_params / original_params) * 100
    print(f"Parameter math (for one {d}x{d} weight matrix, r={r}):")
    print(f"  Original parameters:  {original_params:,}")
    print(f"  LoRA parameters:      {lora_params:,}  (A: {d}x{r}, B: {r}x{d})")
    print(f"  Reduction:            {savings_pct:.1f}% fewer parameters to train")
    print()


# ---------------------------------------------------------------------------
# 2. Dataset Format Converters
# ---------------------------------------------------------------------------

def chatML_to_alpaca(messages: list[dict]) -> dict:
    """
    Convert a ChatML conversation to Alpaca format.

    ChatML format (used by OpenAI, Mistral, many open models):
        [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    Alpaca format (simpler, used by early LLaMA fine-tunes):
        {"instruction": ..., "input": ..., "output": ...}
    """
    system_text = ""
    user_text = ""
    assistant_text = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    # Alpaca puts system prompt + user message together in "instruction"
    instruction = f"{system_text}\n\n{user_text}".strip() if system_text else user_text
    return {
        "instruction": instruction,
        "input": "",   # Used for secondary context; often left empty
        "output": assistant_text,
    }


def alpaca_to_chatML(example: dict, system_prompt: str = "") -> list[dict]:
    """
    Convert an Alpaca example back to ChatML format.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return messages


def format_as_chatml_string(messages: list[dict]) -> str:
    """Format messages as the raw ChatML string that gets tokenized."""
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    result += "<|im_start|>assistant\n"  # Prompt the model to start responding
    return result


def demo_format_conversion():
    print("=" * 60)
    print("Dataset Format Conversion")
    print("=" * 60)
    print()

    chatml_example = [
        {"role": "system", "content": "You are a SQL expert. Return only valid SQL."},
        {"role": "user", "content": "Get all users who signed up in the last 30 days."},
        {"role": "assistant", "content": "SELECT * FROM users WHERE created_at >= NOW() - INTERVAL '30 days';"},
    ]

    print("INPUT (ChatML format):")
    for msg in chatml_example:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")
    print()

    alpaca = chatML_to_alpaca(chatml_example)
    print("CONVERTED (Alpaca format):")
    print(json.dumps(alpaca, indent=2))
    print()

    restored = alpaca_to_chatML(alpaca, system_prompt="You are a SQL expert. Return only valid SQL.")
    print("RESTORED (ChatML format):")
    for msg in restored:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")
    print()

    print("RAW ChatML STRING (what the tokenizer sees):")
    print(format_as_chatml_string(chatml_example[:2]))  # just system + user
    print()


# ---------------------------------------------------------------------------
# 3. Training Script Skeleton
# ---------------------------------------------------------------------------

def training_skeleton():
    """
    Full LoRA fine-tuning skeleton using PEFT + transformers.
    Uses distilgpt2 (smallest available model) for demonstration.
    Falls back to a printed explanation if torch/peft are not installed.
    """
    print("=" * 60)
    print("LoRA Training Skeleton")
    print("=" * 60)
    print()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import Dataset as TorchDataset

        print("torch + peft found. Running training skeleton...")
        print()

        # Synthetic training examples (5 examples, just enough to show the flow)
        synthetic_data = [
            {"instruction": "What is RAG?", "output": "Retrieval-Augmented Generation retrieves relevant context before generating."},
            {"instruction": "What is LoRA?", "output": "Low-Rank Adaptation adds small adapter matrices to reduce training parameters."},
            {"instruction": "When to use fine-tuning?", "output": "Fine-tune when prompting and RAG are insufficient for your use case."},
            {"instruction": "What is a vector database?", "output": "A database optimized to store and search high-dimensional embeddings."},
            {"instruction": "What is a transformer?", "output": "A neural network architecture using self-attention to process sequences."},
        ]

        MODEL_NAME = "distilgpt2"
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Apply LoRA via PEFT
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],   # GPT-2 uses c_attn instead of q_proj/v_proj
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print()

        # Simple Dataset class
        class InstructionDataset(TorchDataset):
            def __init__(self, examples, tokenizer, max_length=128):
                self.examples = examples
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                ex = self.examples[idx]
                text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}{tokenizer.eos_token}"
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].squeeze()
                return {"input_ids": input_ids, "labels": input_ids.clone()}

        dataset = InstructionDataset(synthetic_data, tokenizer)

        training_args = TrainingArguments(
            output_dir="./lora-adapter-output",
            num_train_epochs=1,
            max_steps=3,               # Just 3 steps for demo purposes
            per_device_train_batch_size=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",        # Don't save in this demo
            report_to="none",
            fp16=False,                # distilgpt2 is small enough for fp32
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        print("Starting training (3 steps)...")
        trainer.train()
        print()
        print("Training complete. In production, you would call:")
        print("  model.save_pretrained('./my-lora-adapter')")
        print("  tokenizer.save_pretrained('./my-lora-adapter')")
        print()
        print("To load and use the adapter later:")
        print("  from peft import PeftModel")
        print("  base = AutoModelForCausalLM.from_pretrained('base-model-name')")
        print("  model = PeftModel.from_pretrained(base, './my-lora-adapter')")
        print()

    except ImportError as e:
        print(f"Optional deps not installed ({e}). Showing code structure instead.")
        print()
        print("Install with: pip install torch peft transformers")
        print()
        print("Training script structure:")
        print("""
  1. tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  2. model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                    load_in_4bit=True)  # QLoRA
  3. lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])
  4. model = get_peft_model(model, lora_config)
  5. trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
  6. trainer.train()
  7. model.save_pretrained("./my-adapter")
        """)


# ---------------------------------------------------------------------------
# 4. OpenAI Fine-Tuning API (Simulated)
# ---------------------------------------------------------------------------

SIMULATE = True   # Set to False to make real API calls (requires OPENAI_API_KEY)

def openai_finetune_workflow():
    """
    Demonstrates the full OpenAI fine-tuning API workflow.
    With SIMULATE=True, all API calls are printed but not executed.
    """
    print("=" * 60)
    print("OpenAI Fine-Tuning API Workflow (SIMULATE=True)")
    print("=" * 60)
    print()

    # Step 1: Prepare JSONL training data
    training_examples = [
        {
            "messages": [
                {"role": "system", "content": "You classify customer support tickets. Reply with: billing | technical | shipping | returns"},
                {"role": "user", "content": "My order hasn't arrived and it's been 2 weeks."},
                {"role": "assistant", "content": "shipping"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You classify customer support tickets. Reply with: billing | technical | shipping | returns"},
                {"role": "user", "content": "I was charged twice for my subscription."},
                {"role": "assistant", "content": "billing"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You classify customer support tickets. Reply with: billing | technical | shipping | returns"},
                {"role": "user", "content": "The app keeps crashing when I try to log in."},
                {"role": "assistant", "content": "technical"},
            ]
        },
    ]

    jsonl_path = "/tmp/training_data.jsonl"
    print(f"Step 1: Saving {len(training_examples)} training examples to {jsonl_path}")
    with open(jsonl_path, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    print(f"  Saved. First line preview:")
    print(f"  {json.dumps(training_examples[0])[:80]}...")
    print()

    if SIMULATE:
        # Step 2: Upload file
        print("Step 2: Upload training file to OpenAI")
        print("  SIMULATED API call:")
        print("  client.files.create(")
        print(f"      file=open('{jsonl_path}', 'rb'),")
        print("      purpose='fine-tune'")
        print("  )")
        fake_file_id = "file-abc123def456"
        print(f"  Response: file.id = '{fake_file_id}'")
        print()

        # Step 3: Create fine-tuning job
        print("Step 3: Create fine-tuning job")
        print("  SIMULATED API call:")
        print("  client.fine_tuning.jobs.create(")
        print(f"      training_file='{fake_file_id}',")
        print("      model='gpt-4o-mini-2024-07-18',")
        print("      hyperparameters={'n_epochs': 3}")
        print("  )")
        fake_job_id = "ftjob-xyz789"
        print(f"  Response: job.id = '{fake_job_id}'")
        print()

        # Step 4: Poll status
        print("Step 4: Poll job status")
        print("  SIMULATED API call:")
        print(f"  client.fine_tuning.jobs.retrieve('{fake_job_id}')")
        print("  Statuses: queued → running → succeeded")
        print("  Typical time: 30 minutes for small datasets")
        print()

        # Step 5: Use fine-tuned model
        fake_model_id = "ft:gpt-4o-mini-2024-07-18:your-org::abc123"
        print("Step 5: Use the fine-tuned model")
        print("  SIMULATED API call:")
        print("  client.chat.completions.create(")
        print(f"      model='{fake_model_id}',")
        print("      messages=[{'role': 'user', 'content': 'I want to return my order.'}]")
        print("  )")
        print("  Simulated response: 'returns'")
        print()
        print("  NOTE: Fine-tuned model needs NO system prompt or few-shot examples.")
        print("  The classification behavior is baked into the weights.")
        print()
    else:
        from openai import OpenAI
        client = OpenAI()
        with open(jsonl_path, "rb") as f:
            file_resp = client.files.create(file=f, purpose="fine-tune")
        print(f"File uploaded: {file_resp.id}")

        job = client.fine_tuning.jobs.create(
            training_file=file_resp.id,
            model="gpt-4o-mini-2024-07-18",
        )
        print(f"Job created: {job.id} — status: {job.status}")


# ---------------------------------------------------------------------------
# 5. Cost Estimator
# ---------------------------------------------------------------------------

def estimate_finetune_cost(
    n_examples: int,
    avg_tokens: int,
    epochs: int = 3,
    price_per_1k: float = 0.008,
) -> dict:
    """
    Estimate the cost of fine-tuning on OpenAI's API.

    Args:
        n_examples:     Number of training examples
        avg_tokens:     Average tokens per example (prompt + completion)
        epochs:         Training epochs (default 3)
        price_per_1k:   Cost per 1,000 training tokens (gpt-4o-mini: $0.003/1K)

    Returns:
        Dict with token count and cost breakdown.
    """
    total_tokens = n_examples * avg_tokens * epochs
    total_cost = (total_tokens / 1000) * price_per_1k

    return {
        "n_examples": n_examples,
        "avg_tokens_per_example": avg_tokens,
        "epochs": epochs,
        "total_training_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "price_per_1k_tokens": price_per_1k,
    }


def demo_cost_estimator():
    print("=" * 60)
    print("Fine-Tuning Cost Estimator")
    print("=" * 60)
    print()

    scenarios = [
        {"n_examples": 1_000, "avg_tokens": 200, "epochs": 3, "label": "Small dataset (1K examples, short responses)"},
        {"n_examples": 10_000, "avg_tokens": 500, "epochs": 3, "label": "Medium dataset (10K examples, medium responses)"},
        {"n_examples": 100_000, "avg_tokens": 300, "epochs": 1, "label": "Large dataset (100K examples, 1 epoch)"},
    ]

    for scenario in scenarios:
        label = scenario.pop("label")
        result = estimate_finetune_cost(**scenario)
        print(f"Scenario: {label}")
        print(f"  Training tokens: {result['total_training_tokens']:,}")
        print(f"  Estimated cost:  ${result['estimated_cost_usd']:.2f}")
        print()

    print("Note: Self-hosted fine-tuning (Unsloth + Runpod A100) costs ~$1-3/hour.")
    print("A 2-hour run on a 7B model typically costs $2-6 total.")
    print()


# ---------------------------------------------------------------------------
# 6. Output Comparison: Base Model vs Fine-Tuned
# ---------------------------------------------------------------------------

# Simulate what a base model vs fine-tuned model would output for classification tasks
SIMULATED_OUTPUTS = {
    "ticket_1": {
        "prompt": "My credit card was charged but the order never went through.",
        "base_model": (
            "I understand you're frustrated. Let me help you resolve this issue. "
            "It sounds like there may have been a processing error. Could you provide "
            "your order number so we can investigate further?"
        ),
        "fine_tuned": "billing",
        "correct": "billing",
        "notes": "Base model gives a helpful response but wrong FORMAT. Fine-tuned gives the exact label needed.",
    },
    "ticket_2": {
        "prompt": "The download link in my email isn't working.",
        "base_model": (
            "Sorry to hear you're having trouble with the download link. "
            "This is often caused by the link expiring. You can usually regenerate "
            "it from your account dashboard under 'My Orders'."
        ),
        "fine_tuned": "technical",
        "correct": "technical",
        "notes": "Base model gives generic help desk advice. Fine-tuned gives correct classification instantly.",
    },
    "ticket_3": {
        "prompt": "I ordered the wrong size and want to exchange it.",
        "base_model": (
            "No problem! We have an easy exchange process. Please note that exchanges "
            "are subject to our return policy and item availability. "
            "Would you like me to walk you through the steps?"
        ),
        "fine_tuned": "returns",
        "correct": "returns",
        "notes": "Base model defaults to conversational mode. Fine-tuned snaps to correct classification.",
    },
}


def demo_output_comparison():
    print("=" * 60)
    print("Output Comparison: Base Model vs Fine-Tuned Model")
    print("=" * 60)
    print()
    print("Task: Classify customer support tickets into:")
    print("      billing | technical | shipping | returns")
    print()

    for key, example in SIMULATED_OUTPUTS.items():
        print(f"Prompt: \"{example['prompt']}\"")
        print()
        print(f"  Base model:   \"{example['base_model'][:80]}...\"")
        print(f"  Fine-tuned:   \"{example['fine_tuned']}\"   <- correct: {example['correct']}")
        print(f"  Note: {example['notes']}")
        print()

    print("Key takeaway:")
    print("  The base model tries to help (which is usually good!) but can't output")
    print("  a specific format reliably. Fine-tuning doesn't improve knowledge —")
    print("  it changes the OUTPUT STYLE to match your exact requirements.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_lora_config()
    demo_format_conversion()
    training_skeleton()
    openai_finetune_workflow()
    demo_cost_estimator()
    demo_output_comparison()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("Decision checklist before fine-tuning:")
    print("  [ ] Have I tried a detailed system prompt?")
    print("  [ ] Have I tried 5-10 few-shot examples?")
    print("  [ ] Have I tried RAG for knowledge retrieval?")
    print("  [ ] Do I have >= 1,000 high-quality labeled examples?")
    print("  [ ] Is the gap a STYLE/FORMAT gap (not a knowledge gap)?")
    print()
    print("If all boxes are checked: fine-tuning is justified.")
    print("Otherwise: keep prompting/RAG and come back when you have more data.")
