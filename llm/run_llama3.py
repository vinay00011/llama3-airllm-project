# run_llama3.py

import torch
from airllm import AutoModel
from transformers import AutoTokenizer

# ─────────────────────────────────────────
# 1. AUTO-DETECT HARDWARE
# ─────────────────────────────────────────
def detect_hardware():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
        return {
            "compression": "4bit",
            "max_seq_len": 256,
            "max_new_tokens": 150,
            "device": "cuda"
        }
    elif vram < 10:
        return {
            "compression": "8bit",
            "max_seq_len": 512,
            "max_new_tokens": 200,
            "device": "cuda"
        }
    else:
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
    print("\n📥 Loading LLaMA 3 (8B) via AirLLM...")
    print("   First run will download ~16GB — one-time process.")
    print("   AirLLM streams weights layer-by-layer (low VRAM friendly)\n")

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ── FIX: AirLLM manages device internally — do NOT call .to(device) ──
    if settings["compression"]:
        model = AutoModel.from_pretrained(
            model_id,
            compression=settings["compression"],
            max_seq_len=settings["max_seq_len"],
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            max_seq_len=settings["max_seq_len"],
        )

    # Load tokenizer separately (standard HuggingFace)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure pad token is set (LLaMA 3 may not have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model and tokenizer loaded!\n")
    return model, tokenizer

# ─────────────────────────────────────────
# 4. GENERATE A RESPONSE
# ─────────────────────────────────────────
def generate_response(model, tokenizer, prompt, settings):
    # LLaMA 3 chat format
    formatted_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Tokenize
    inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=settings["max_seq_len"],
        padding=False,
    )

    input_ids = inputs["input_ids"]

    # ── FIX: AirLLM's generate() does NOT support return_dict_in_generate ──
    # It returns a plain tensor of token IDs
    print("🤖 Generating response...\n")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=settings["max_new_tokens"],
        use_cache=True,
        # stop_token_ids are optional but clean up output
        stop_token=tokenizer.eos_token_id,
    )

    # Decode only newly generated tokens (skip the prompt)
    input_length = input_ids.shape[1]
    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()

# ─────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    device, vram = detect_hardware()

    settings = get_settings(device, vram)
    print(f"\n⚙️  Settings chosen:")
    print(f"   Compression : {settings['compression'] or 'None (full precision)'}")
    print(f"   Max tokens  : {settings['max_new_tokens']}")
    print(f"   Context len : {settings['max_seq_len']}")
    print(f"   Device      : {settings['device']}")

    model, tokenizer = load_model(settings)

    prompts = [
        "What is artificial intelligence in simple words?",
        "Write a short poem about the night sky.",
        "Give me 3 tips to learn programming faster.",
    ]

    print("=" * 60)
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📌 Prompt {i}: {prompt}")
        print("-" * 60)
        try:
            response = generate_response(model, tokenizer, prompt, settings)
            print(f"💬 Response:\n{response}")
        except Exception as e:
            print(f"❌ Error on prompt {i}: {e}")
        print("=" * 60)
