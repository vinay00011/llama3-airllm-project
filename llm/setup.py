# setup.py
import subprocess
import sys
import platform

def run(cmd):
    subprocess.check_call(cmd, shell=True)

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("🔧 Setting up LLaMA 3 environment via AirLLM...\n")

# ─────────────────────────────────────────
# STEP 1: Install PyTorch with CUDA support
# ─────────────────────────────────────────
print("📦 Step 1: Installing PyTorch (with CUDA 12.1 support)...")
print("   (Skip this and install CPU-only torch if you have no NVIDIA GPU)\n")

torch_cmd = (
    f"{sys.executable} -m pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu121"
)
subprocess.check_call(torch_cmd, shell=True)
print("✅ PyTorch installed.\n")

# ─────────────────────────────────────────
# STEP 2: Install other packages
# ─────────────────────────────────────────
print("📦 Step 2: Installing remaining dependencies...")

packages = [
    "airllm>=0.9.0",
    "transformers>=4.35.0",
    "accelerate>=0.24.0",
    "huggingface_hub",
    "sentencepiece",   # needed by LLaMA tokenizer
    "protobuf",
]

for pkg in packages:
    print(f"  → Installing {pkg}...")
    pip_install(pkg)

print("\n✅ All dependencies installed!\n")

# ─────────────────────────────────────────
# STEP 3: HuggingFace login reminder
# ─────────────────────────────────────────
print("=" * 55)
print("⚠️  IMPORTANT — HuggingFace Login Required")
print("=" * 55)
print("""
LLaMA 3 is a gated model. Before running run_llama3.py:

  1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  2. Accept Meta's license agreement
  3. Run this in your terminal:

       huggingface-cli login

  4. Paste your HuggingFace access token when prompted
     (Get one at: https://huggingface.co/settings/tokens)

Once logged in, run:

       python run_llama3.py
""")
