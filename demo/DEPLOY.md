# Deploy SentinX Demo to Hugging Face Spaces

This guide outlines step-by-step instructions to deploy the SentinX Fraud Detection demo to [Hugging Face Spaces](https://huggingface.co/spaces) using the **Gradio** SDK.

---

## Prerequisites

1. A [Hugging Face account](https://huggingface.co/join).
2. A Hugging Face User Access Token with **write** permissions:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Click **Create new token**, name it (e.g., `sentinx-deploy`), select **Write**, and copy the token (`hf_...`).
3. The `huggingface_hub` Python package:
   ```bash
   pip install huggingface_hub
   ```

---

## Deployment Steps

### 1. Authenticate with Hugging Face

Run the CLI login command:
```bash
huggingface-cli login
```
When prompted:
```text
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|    _|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|        _|      
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|      _|_|  
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|              _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|  _|_|_|  

Token: <PASTE_YOUR_HUGGING_FACE_TOKEN_HERE>
```
Paste your token (`hf_...`) and select `Y` to add token as git credential.

### 2. Create the Space on Hugging Face

Create a new Gradio Space under your username:
```bash
huggingface-cli repo create sentinx-fraud-detection --type space --space_sdk gradio
```

### 3. Clone the Space Repository Locally

```bash
git clone https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentinx-fraud-detection
cd sentinx-fraud-detection
```

### 4. Copy SentinX Demo Files

From your local clone of `sentinx-data-mlops-platform`:
```bash
# Copy demo app and requirements
cp ../sentinx-data-mlops-platform/demo/app.py ./app.py
cp ../sentinx-data-mlops-platform/demo/requirements-demo.txt ./requirements.txt

# Copy backend schema & models necessary for live inference
cp -r ../sentinx-data-mlops-platform/src ./src
cp -r ../sentinx-data-mlops-platform/config ./config
cp -r ../sentinx-data-mlops-platform/models ./models 2>/dev/null || true
```

### 5. Commit and Push to Deploy

```bash
git add app.py requirements.txt src/ config/ models/
git commit -m "feat: deploy SentinX fraud detection Gradio demo"
git push origin main
```

### 6. Verify Deployment

1. Navigate to: `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/sentinx-fraud-detection`
2. Wait 1–2 minutes for the build container to install dependencies and launch.
3. Test scoring transactions with the interactive UI!
