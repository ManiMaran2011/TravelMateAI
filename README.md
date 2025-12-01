# GenAI Project

An interactive GenAI notebook/app for PDF processing and text chunking using `langchain-text-splitters`, vector storage with `chromadb`, and LLM calls via `openai`.

## Repo Structure
- `app.py` – Python script converted from the original notebook.
- `requirements.txt` – Base dependencies.
- `requirements-pinned.txt` – Pinned dependencies from this environment (useful for reproducibility).
- The original notebook can be kept in `notebooks/`.

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# For fully reproducible builds (recommended):
# pip install -r requirements-pinned.txt
```

### 3) Environment variables
Set your OpenAI (or other) API keys as environment variables:
```bash
# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"

# Windows (PowerShell)
$Env:OPENAI_API_KEY="YOUR_KEY"
```

### 4) Run
If you want to run the Python script:
```bash
python app.py
```

Or open the original notebook in Jupyter/VS Code and run cells interactively.

## Notes

- **Colab-only imports**: The notebook uses `from google.colab import files`. For local runs (non‑Colab), either remove those lines or wrap them:
```python
try:
    from google.colab import files
except Exception:
    files = None
```
Then add a local alternative for file selection (e.g., `tkinter.filedialog` or CLI args).

- If you don't need widgets, you can remove `ipywidgets` related code/installs.
- For production, pin exact versions using `requirements-pinned.txt` and enable caching in your CI.

## License
Add your preferred license here (MIT is common for student projects).
