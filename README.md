# GenAI PDF Assistant (Local-Friendly)

An **interactive** GenAI project that:
- Ingests PDFs (`pdfplumber`, `PyPDF2`)
- Splits text into chunks (`langchain-text-splitters`)
- Stores/queries embeddings with **ChromaDB**
- Calls LLMs via the **OpenAI** API

This version is **Colab-free** and uses a **desktop file picker (Tkinter)** to choose a PDF locally.

---

## 🗂 Recommended Repo Structure

```
genai-pdf-assistant/
├─ app.py
├─ requirements.txt
├─ requirements-pinned.txt   # optional: reproducible installs
├─ README.md
├─ .gitignore
└─ notebooks/                # (optional) keep the original .ipynb here
```

> If you have the original notebook, place it under `notebooks/` for reference.

---

## ⚙️ Setup

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# For reproducible installs (CI, grading), prefer:
# pip install -r requirements-pinned.txt
```

### 3) Configure API keys
Set your **OpenAI** API key as an environment variable before running:

**macOS/Linux**
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

**Windows (PowerShell)**
```powershell
$Env:OPENAI_API_KEY="YOUR_KEY"
```

---

## ▶️ Run the app

```bash
python app.py
```
- A native file dialog will pop up (Tkinter) to select a **PDF**.
- The rest of your pipeline (chunking, vector DB, LLM calls) will run as in the notebook.

> If you prefer command-line usage without GUI dialogs, we can switch to `argparse` (e.g., `python app.py --pdf path/to/file.pdf`).

---

## 🔁 Common Tweaks

- **Headless servers / no display**: Tkinter requires a display. For servers/CI, swap in `argparse` and pass file paths directly.
- **No widgets needed?** Remove `ipywidgets` from requirements and related code.
- **Persistence for ChromaDB**: Consider setting a `persist_directory` so your embeddings survive restarts.

---

## 🧪 Quick sanity checks

1. `python -c "import pdfplumber; print('pdfplumber ok')"`
2. `python -c "import chromadb; print('chromadb ok')"`
3. `python -c "import openai; print('openai ok')"`

---

## 📄 License
MIT (or your preferred license).
