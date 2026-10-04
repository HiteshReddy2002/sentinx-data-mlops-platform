# Deploy SentinX Demo to Hugging Face Spaces

This guide deploys the SentinX Fraud Detection demo (Streamlit) to [Hugging Face Spaces](https://huggingface.co/spaces).

## Prerequisites

- [Hugging Face account](https://huggingface.co/join)
- `huggingface_hub` CLI: `pip install huggingface_hub`

---

## Steps

### 1. Create a new Streamlit Space

```bash
huggingface-cli login
huggingface-cli repo create sentinx-fraud-demo --type space --space_sdk streamlit
```

### 2. Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentinx-fraud-demo
cd sentinx-fraud-demo
```

### 3. Copy demo + required source files

```bash
# From the sentinx-data-mlops-platform repo root:
cp demo/app.py ../sentinx-fraud-demo/app.py
cp demo/requirements-demo.txt ../sentinx-fraud-demo/requirements.txt

# Copy src modules needed by demo
cp -r src ../sentinx-fraud-demo/src

# Copy trained model artifact (if available after running uv run python -m src.ml.train)
mkdir -p ../sentinx-fraud-demo/models
cp models/sentinx-fraud-detector.joblib ../sentinx-fraud-demo/models/ 2>/dev/null || echo "No model artifact found — demo will use mock scoring"
```

### 4. Train the model first (optional but recommended)

```bash
# In the main repo:
uv sync
uv run python -m src.pipeline.orchestrator
uv run python -m src.ml.train
# This produces models/sentinx-fraud-detector.joblib
```

### 5. Push to Spaces

```bash
cd ../sentinx-fraud-demo
git add .
git commit -m "Initial deploy: SentinX fraud detection Streamlit demo"
git push
```

### 6. View your Space

URL: `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentinx-fraud-demo`

---

## Notes

- Without the model artifact, the demo uses a rule-based mock scorer and still displays correctly.
- Free Spaces provide 2 vCPU / 16 GB RAM — sufficient for this demo (all compute is local DuckDB).
- For the full platform (FastAPI + Streamlit), consider Docker Spaces with the existing `docker-compose.yml`.
