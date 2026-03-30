# 🚀 LLaMA 3 with AirLLM (Local Inference Project)

## 📌 Overview

This project demonstrates how to run the **LLaMA 3** large language model locally using **AirLLM**, a memory-efficient inference framework. The goal is to enable LLM execution on systems with limited hardware by loading model weights dynamically.

---

## 🎯 Objectives

* Run LLaMA 3 locally without high-end GPU
* Reduce memory usage using AirLLM
* Perform text generation efficiently
* Explore local LLM deployment in VS Code

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** AirLLM
* **Model:** LLaMA 3
* **Libraries:** PyTorch, Transformers
* **IDE:** VS Code

---

## 📂 Project Structure

```
llama3-airllm-project/
│── run_llama3.py
│── setup.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vinay00011/llama3-airllm-project.git
cd llama3-airllm-project
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv llm_env
source llm_env/bin/activate   # Linux/Mac
llm_env\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model Setup

* Download **LLaMA 3 model weights** from official source / Hugging Face
* Place them in your local directory
* Update the model path in `main.py`

---

## ▶️ Usage

Run the model:

```bash
python run_llama3.py
```

### 💡 Example Code

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("path_to_llama3")

response = model.generate("Explain machine learning in simple terms")
print(response)
```

---

## 🧠 How It Works

1. User inputs a prompt
2. AirLLM loads model layers dynamically
3. LLaMA 3 processes the input
4. Output text is generated

---

## 📊 Results

* ✅ Successfully ran LLaMA 3 locally
* ✅ Reduced memory usage
* ⚠️ Slower inference on CPU
* ✅ Works on low-resource systems

---

## ⚠️ Challenges

* Large model size
* Initial setup complexity
* Dependency compatibility
* Slower performance without GPU

---

## 🔮 Future Improvements

* Add GPU acceleration
* Build a web interface (Streamlit/Gradio)
* Optimize inference speed
* Support multiple LLMs

---

## 👨‍💻 Author

**Vinay**

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

* LLaMA 3 by Meta
* AirLLM framework
* Hugging Face Transformers

---
