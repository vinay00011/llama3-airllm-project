# setup.py
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("🔧 Installing dependencies...")

packages = [
    "airllm",
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub"
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    install(pkg)

print("\n✅ All dependencies installed!")
print("👉 Now run: python run_llama3.py")