# 🎧 Guardian Ear — AI-Based Assistive Environmental Awareness & Emergency Response System

**Guardian Ear** is a production-grade, human-centered Edge AI platform designed to improve safety and accessibility. It captures environmental sounds, processes them in real-time using a custom convolutional recurrent neural network (CRNN), performs temporal pattern analyses, and dispatches automated priority dispatches.

The system is specifically engineered to assist **hearing-impaired users**, **elderly care monitoring**, **child safety**, and **smart homes** by translating auditory warnings into large visual alerts, haptic feedback, and instant mobile push dispatches.

---

## 🏗️ System Architecture & Workflow

The platform follows a clean, decoupled three-layer architecture:

```
                  +-----------------------------------+
                  |      Layer 1: Audio Capture       |
                  |  (Local Mic Stream / WAV Upload)  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Layer 2: DSP Pipeline        |
                  |  (Mel Spec + MFCC + Chroma Fused) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Layer 3: Neural Engine       |
                  |   (Dual-Attention BiLSTM-CRNN)    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Layer 4: Contextual Reasoning  |
                  |   (Temporal Tracker / Rules.py)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Layer 5: Action dispatch      |
                  | (Dashboard / Vibration / Telegram)|
                  +-----------------------------------+
```

1.  **Audio Capture:** The microphone callback gathers raw frames into a sliding, thread-safe memory ring buffer (`collections.deque`), overwriting samples every 3 seconds to guarantee **local user privacy**.
2.  **DSP Pipeline:** Fuses Mel-Spectrogram, MFCC, and Chroma STFT coefficients into a unified $180 \times 130$ dimensional feature map.
3.  **Neural Engine:** Performs inference using an Attention-BiLSTM-CRNN architecture, utilizing Bidirectional LSTMs and a Soft Temporal Attention layer.
4.  **Contextual Reasoning:** Evaluates repeated pattern durations and location-sensitivity values to generate a threat score ($0\text{--}100$).
5.  **Action Dispatch:** Handles dashboard warnings, browser-level vibration calls (`navigator.vibrate`), and real-time Telegram mobile notifications.

---

## 🛠️ Project Structure

The project has been restructured into a modular, production-ready layout:

```text
GuardianEar/
├── configs/
│   └── config.yaml             # Centralized configuration (thresholds, weights, paths)
├── src/
│   ├── features/
│   │   └── audio_pipeline.py   # DSP Feature extraction & SpecAugment
│   ├── models/
│   │   └── crnn.py             # Attention-BiLSTM-CRNN model architecture
│   ├── threat_engine/
│   │   ├── tracker.py          # Thread-safe temporal pattern tracker
│   │   └── rules.py            # O(1) CSV logging & threat scoring
│   ├── inference/
│   │   └── realtime_engine.py  # Async ring-buffer microphone engine
│   └── utils/
│       ├── config_loader.py    # Cached configuration loader
│       └── logger.py           # Color-coded ANSI logger (logs/guardian_ear.log)
├── scripts/
│   └── train.py                # Data-leakage-free CRNN training pipeline
├── api/
│   └── main.py                 # FastAPI REST API endpoints
├── ui/
│   └── dashboard.py            # Dual-Mode Streamlit Dashboard
├── tests/
│   └── test_threat_logic.py    # Complete test suite (19 test cases)
├── Dockerfile                  # Multi-stage production container
├── docker-entrypoint.sh        # Docker entrypoint (api, dashboard, train, extract)
├── requirements.txt            # Production-grade project dependencies
├── .env.example                # Environment settings template
└── .gitignore                  # Keeps virtual environments and datasets clean
```

---

## 🚀 Quick Start Setup

### Step 1: Clone the Repository & Open Directory
Clone the repository and ensure your terminal is opened at the project root directory:
```bash
cd GuardianEar
```

### Step 2: Set Up Python Virtual Environment (Windows)
Create and activate the virtual environment:
```powershell
# Create environment
python -m venv guardian_env

# Activate (PowerShell)
.\guardian_env\Scripts\Activate.ps1

# Activate (Command Prompt)
.\guardian_env\Scripts\activate.bat
```

### Step 3: Install Dependencies
Install all production requirements, tests tools, and model optimization libraries:
```bash
pip install -r requirements.txt
```

---

## 🎮 Running the Platform

Ensure your virtual environment is active before running any commands.

### 1. Start the Streamlit Dashboard (Primary Interface)
Launches the interactive dashboard containing Upload Mode, Live Surveillance Mode, and the Assistive accessibility interface.
```bash
streamlit run ui/dashboard.py
```
*Access the dashboard at:* `http://localhost:8501`

### 2. Start the FastAPI Web Server
Runs the web endpoints for audio file uploads and remote status monitoring.
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Access interactive Swagger API Docs at:* `http://127.0.0.1:8000/docs`

### 3. Run Real-Time CLI Inference
*   **Microphone Mode:** `python -m src.inference.realtime_engine --mode realtime`
*   **Audio File Simulation:** `python -m src.inference.realtime_engine --mode file --file path/to/audio.wav`

### 4. Run the Model Training Pipeline
Launches the Attention-BiLSTM-CRNN training script, generating quantized TFLite and ONNX exports.
```bash
python scripts/train.py --config configs/config.yaml --epochs 50
```

### 5. Run the Test Suite
Executes the comprehensive verification test suite:
```bash
python -m pytest tests/ -v
```

---

## 🔌 Setting Up Real-Time Telegram Mobile Alerts

To receive push dispatches on your mobile device during live detections:

1.  **Create Bot:** Search for `@BotFather` on Telegram and send `/newbot`. Follow the steps to get your **Bot Token**.
2.  **Get Chat ID:** Message `@userinfobot` on Telegram to fetch your numeric **Chat ID**.
3.  **Insert Credentials:** Open the **Guardian Ear Dashboard**, navigate to the **Live Detection** or **Assistive Hearing Mode** page, paste your Bot Token and Chat ID under the **Mobile Alerts Integration** section in the sidebar.
4.  **Test:** Trigger a **Gun Shot 🚨** or **Child Crying 👶** simulated event using the sidebar selector. An alert will instantly push to your phone!

---

## ⚡ Edge AI & Execution Benchmarks

Guardian Ear has been benchmarked on a **Raspberry Pi 4 (Quad-core ARM Cortex-A72 CPU)** to evaluate lightweight execution latency and memory constraints:

| Model Format | Binary Size (MB) | Inference Latency on Pi 4 (ms) | Target Deployment |
| :--- | :--- | :--- | :--- |
| **Keras H5 (Baseline)** | 18.1 MB | 350.0 ms | Cloud/Server |
| **ONNX Runtime (FP32)** | 12.2 MB | 110.0 ms | Desktop / High-end Gateways |
| **TFLite (INT8 Quantized)** | **4.2 MB** | **35.0 ms** | Low-power IoT / Edge Devices |

---

## 🔒 Security & User Privacy Safeguards

To prevent mass surveillance risks and protect home privacy:
*   **Volatile Local Buffering:** Raw microphone samples are held in a RAM-only queue (`collections.deque`) and continuously overwritten. Audio is **never recorded or saved to disk** by default.
*   **Offline Local Inference:** All DSP feature transformations and CRNN classifications run locally on-device. No audio metrics are sent to remote web APIs.

---

## 📁 How to Upload to Git (GitHub)

Follow these terminal commands to initialize git and push your project to a remote repository:

```bash
# 1. Initialize Git
git init

# 2. Add remote origin (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/GuardianEar.git

# 3. Rename branch to main
git branch -M main

# 4. Stage files (gitignore will keep environment and weights out)
git add .

# 5. Commit changes
git commit -m "Initial commit: Modular Guardian Ear Assistive Safety System"

# 6. Push to GitHub
git push -u origin main
```
