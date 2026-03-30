# run_llama3.py

import torch
from airllm import AutoModel
from transformers import AutoTokenizer

# ─────────────────────────────────────────
# 1. AUTO-DETECT HARDWARE
# ─────────────────────────────────────────
def detect_hardware():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   VRAM available: {vram:.1f} GB")
        return "gpu", vram
    else:
        print("⚠️  No GPU detected. Running on CPU (will be slow)")
        return "cpu", 0

# ─────────────────────────────────────────
# 2. PICK BEST SETTINGS BASED ON HARDWARE
# ─────────────────────────────────────────
def get_settings(device, vram):
    if device == "cpu":
        return {
            "compression": "4bit",
            "max_seq_len": 256,
            "max_new_tokens": 100,
            "device": "cpu"
        }
    elif vram < 6:
        # Low VRAM — use aggressive compression
        return {
            "compression": "4bit",
            "max_seq_len": 256,
            "max_new_tokens": 150,
            "device": "cuda"
        }
    elif vram < 10:
        # Medium VRAM
        return {
            "compression": "8bit",
            "max_seq_len": 512,
            "max_new_tokens": 200,
            "device": "cuda"
        }
    else:
        # High VRAM — no compression needed
        return {
            "compression": None,
            "max_seq_len": 1024,
            "max_new_tokens": 300,
            "device": "cuda"
        }

# ─────────────────────────────────────────
# 3. LOAD THE MODEL
# ─────────────────────────────────────────
def load_model(settings):
    print("\n📥 Loading LLaMA 3 (8B)...")
    print("   (First run will download ~16GB — this is a one-time process)")
    print("   Please be patient...\n")

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    if settings["compression"]:
        model = AutoModel.from_pretrained(
            model_id,
            compression=settings["compression"],
            max_seq_len=settings["max_seq_len"]
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            max_seq_len=settings["max_seq_len"]
        )

    # Move model to device
    model = model.to(settings["device"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("✅ Model and tokenizer loaded successfully!\n")
    return model, tokenizer

# ─────────────────────────────────────────
# 4. GENERATE A RESPONSE
# ─────────────────────────────────────────
def generate_response(model, tokenizer, prompt, settings):
    # Format prompt in LLaMA 3 chat style
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    # Tokenize
    inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=settings["max_seq_len"],
        padding=False
    )

    # Generate
    print("🤖 Generating response...\n")
    output = model.generate(
        inputs["input_ids"].to(settings["device"]),
        max_new_tokens=settings["max_new_tokens"],
        use_cache=True,
        return_dict_in_generate=True
    )

    # Decode only the new tokens (not the prompt)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output.sequences[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()

# ─────────────────────────────────────────
# 5. MAIN — RUN EVERYTHING
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Detect hardware
    device, vram = detect_hardware()

    # Get best settings
    settings = get_settings(device, vram)
    print(f"\n⚙️  Settings chosen:")
    print(f"   Compression : {settings['compression'] or 'None (full precision)'}")
    print(f"   Max tokens  : {settings['max_new_tokens']}")
    print(f"   Context len : {settings['max_seq_len']}")

    # Load model
    model, tokenizer = load_model(settings)

    # ── Test Prompts ──────────────────────
    prompts = [
        "What is artificial intelligence in simple words?",
        "Write a short poem about the night sky.",
        "Give me 3 tips to learn programming faster."
    ]

    print("=" * 60)
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📌 Prompt {i}: {prompt}")
        print("-" * 60)
        response = generate_response(model, tokenizer, prompt, settings)
        print(f"💬 Response:\n{response}")
        print("=" * 60)