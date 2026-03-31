# 🦙 LLaMA 3 Chat — Run AI Locally Like ChatGPT

> A fully local, private, and free AI chatbot powered by **LLaMA 3 (8B)** and **AirLLM** — runs on low VRAM hardware!

---

## ✨ What This Does

This project lets you chat with **Meta's LLaMA 3** model directly on your PC — no internet required after setup, no API costs, 100% private. It works like ChatGPT but runs completely on your own machine.

```
  🦙 LLaMA 3 Chat  —  Running 100% locally & privately
═══════════════════════════════════════════════════════
  Type your message and press Enter to chat.
  Commands:  'clear' = reset chat  |  'quit' = exit
═══════════════════════════════════════════════════════

You: What is machine learning?
🤖 LLaMA: Machine learning is a type of AI that learns from data...

You: Give me an example
🤖 LLaMA: Sure! A great example is spam detection in emails...
```

---

## 🧠 How It Works

```
You type a message
       ↓
Chat history is built (gives it memory like ChatGPT)
       ↓
AirLLM loads LLaMA 3 layer by layer (low VRAM magic)
       ↓
LLaMA 3 generates a response
       ↓
Response is printed + saved to history
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **LLaMA 3 8B** | The AI model (by Meta) |
| **AirLLM** | Runs large models on low VRAM |
| **PyTorch** | Hardware detection & GPU support |
| **Transformers** | Tokenizer (text ↔ numbers) |
| **bitsandbytes** | 4bit/8bit compression |
| **optimum** | Required by AirLLM internally |

---

## 💻 System Requirements

| | Minimum | Recommended |
|---|---|---|
| **RAM** | 8 GB | 16 GB |
| **VRAM** | 4 GB | 8 GB+ |
| **Storage** | 20 GB free | 30 GB free |
| **Python** | 3.11 | 3.11 |
| **OS** | Windows 10/11 | Windows 10/11 |

> ⚠️ Works on CPU too — just slower

---

## ⚙️ Installation

### 1️⃣ Install Python 3.11
Download from: https://www.python.org/downloads/release/python-3119/
- ✅ Check **"Add Python to PATH"** during install

---

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/vinay00011/llama3-airllm-project.git
cd llama3-airllm-project
```

---

### 3️⃣ Create Virtual Environment
```powershell
py -3.11 -m venv llm_env
llm_env\Scripts\activate
```
You should see `(llm_env)` in your terminal.

---

### 4️⃣ Install PyTorch

**With NVIDIA GPU (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

---

### 5️⃣ Install All Dependencies
```bash
pip install airllm transformers==4.40.0 accelerate huggingface_hub sentencepiece protobuf optimum==1.19.0 bitsandbytes
```

---

### 6️⃣ Fix AirLLM Compatibility (Required)

Open this file:
```
llm_env\Lib\site-packages\airllm\airllm_base.py
```

Find **line 18:**
```python
from optimum.bettertransformer import BetterTransformer
```

Replace with:
```python
try:
    from optimum.bettertransformer import BetterTransformer
except (ImportError, ModuleNotFoundError):
    BetterTransformer = None
```

---

### 7️⃣ HuggingFace Login

LLaMA 3 is a gated model — you need to request access:

1. Create a free account at https://huggingface.co/join
2. Accept Meta's license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Get your token at: https://huggingface.co/settings/tokens
4. Login in terminal:
```bash
python -c "from huggingface_hub import login; login(token='your_token_here', add_to_git_credential=False)"
```

---

## ▶️ Run the Chat

```bash
python run_llama3.py
```

> ⚠️ First run downloads ~16GB of model weights — one time only!

---

## 💬 Chat Commands

| Command | Action |
|---|---|
| Type anything | Chat with LLaMA 3 |
| `clear` | Reset conversation memory |
| `quit` | Exit the chat |

---

## 🔧 Auto Hardware Detection

The script automatically picks the best settings for your hardware:

| Hardware | Compression | Context |
|---|---|---|
| CPU | 4bit | 512 tokens |
| GPU < 6GB VRAM | 4bit | 512 tokens |
| GPU 6–10GB VRAM | 8bit | 1024 tokens |
| GPU 10GB+ VRAM | None (full) | 2048 tokens |

---

## 📂 Project Structure

```
llama3-airllm-project/
│
├── llm/
│   ├── run_llama3.py       ← Main chat script
│   ├── setup.py            ← Auto installer
│   └── requirements.txt    ← Dependencies
│
└── README.md
```

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|---|---|
| `No module named optimum.bettertransformer` | Patch airllm_base.py (see Step 6) |
| `401 Unauthorized` | Login to HuggingFace + accept Meta license |
| `fbgemm.dll not found` | Install Visual C++ Redistributable |
| `No module named bitsandbytes` | `pip install bitsandbytes` |
| `Python 3.13 not supported` | Use Python 3.11 (see Step 1) |

---

## 🔮 Future Plans

- [ ] Web UI with Streamlit or Gradio
- [ ] Support multiple LLM models
- [ ] Save and load chat history
- [ ] Voice input/output
- [ ] GPU optimization

---

## 👨‍💻 Author

**Vinay** — [@vinay00011](https://github.com/vinay00011)

*This is my first personal AI project — built from scratch!* 🚀

---

## 📜 License

MIT License — free to use and modify.

---

## ⭐ Acknowledgements

- [LLaMA 3](https://ai.meta.com/llama/) by Meta
- [AirLLM](https://github.com/lyogavin/airllm) by lyogavin
- [HuggingFace Transformers](https://huggingface.co/transformers)
