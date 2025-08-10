# Frontend (Streamlit)

## Quick start

```bash
cd frontend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
$env:BACKEND_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

Note: LLM generation is performed by the backend. To enable Gemini, set `GEMINI_API_KEY` in the backend environment before starting the API. 