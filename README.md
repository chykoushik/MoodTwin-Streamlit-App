# MoodTwin Streamlit App

MoodTwin is a local Streamlit web app for persona-conditioned music emotion and preference prediction. It predicts how a listener may respond to music using valence, arousal, emotion category, and liking scores.

This repository contains only the app files needed to run the web app locally.

---

## What You Need

Before running the app, you need:

1. Python 3.10 installed
2. This GitHub repository downloaded
3. Required Python packages installed
4. The required model and result files inside the `results/` folder
5. Your own DeepSeek API key if you want to use the AI explanation buttons

---

## Project Structure

```text
moodtwin-app/
│
├── app_v3.py
├── predict_v3.py
├── extract_video_emotion_fixed.py
├── requirements.txt
├── README.md
│
├── results/
│   ├── X_v3.npy
│   ├── song_analysis.csv
│   ├── scaler_v3.pkl
│   └── digital_twin_v3.onnx
```

> Do not change these file names unless you also update the paths inside the Python code.

---

## Required Files

The app needs these files to run:

```text
app_v3.py
predict_v3.py
requirements.txt
results/X_v3.npy
results/song_analysis.csv
results/scaler_v3.pkl
results/digital_twin_v3.onnx
```

The webcam mode also requires:

```text
extract_video_emotion_fixed.py
```

---

## Setup Instructions

### Step 1: Download the Project

Click the green **Code** button on GitHub and download the ZIP file. Extract it and open the folder.

### Step 2: Install Python 3.10

Download from: https://www.python.org/downloads/release/python-3100/

During installation, check **Add Python to PATH**.

### Step 3: Open a Terminal in the Project Folder

**Windows:**
1. Open the project folder
2. Click the address bar, type `cmd`, and press Enter

### Step 4: Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal line.

### Step 5: Install Required Packages

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
streamlit
numpy
pandas
plotly
openai
streamlit-webrtc
opencv-python-headless
joblib
onnxruntime
scikit-learn
```

For webcam/DeepFace support, also run:

```bash
pip install deepface tensorflow
```

### Step 6: Add Your DeepSeek API Key

Open `app_v3.py` and find:

```python
DEEPSEEK_API_KEY = "add-your-own-key"
```

Replace it with your actual key:

```python
DEEPSEEK_API_KEY = "your-real-deepseek-api-key"
```

> The DeepSeek key is only needed for the explanation buttons. The prediction model works without it.  
> **Do not upload your real API key to a public repository.**

### Step 7: Run the App

```bash
streamlit run app_v3.py
```

The app will open in your browser. If it doesn't open automatically, navigate to:

```
http://localhost:8501
```

---

## How to Use

1. Open the app in your browser
2. Select a song ID
3. Choose an input mode:
   - Manual input
   - CSV upload
   - Start recording (webcam)
4. Enter listener profile information
5. Click **Run personalized prediction**

The app will display:

- Interpreted persona profile
- Predicted top emotion
- Liking score
- Valence and arousal scores
- Acoustic profile
- Top 10 personalized recommendations
- Valence–arousal map
- Optional DeepSeek explanation
- Downloadable prediction results

---

## Troubleshooting

**Missing file error**

```
FileNotFoundError: results/scaler_v3.pkl
```

Ensure your `results/` folder contains all four required files:

```text
X_v3.npy
song_analysis.csv
scaler_v3.pkl
digital_twin_v3.onnx
```

**Missing package error**

```
ModuleNotFoundError
```

Install the missing package:

```bash
pip install package-name
```

Then rerun:

```bash
streamlit run app_v3.py
```

**DeepSeek explanation fails**

Check that your API key is correct and that your DeepSeek account has sufficient balance. The prediction model will still work without it.

**Webcam mode does not work**

Install the optional packages:

```bash
pip install deepface tensorflow
```

Then restart the app.

---

## Notes

This app is designed to run locally on your own machine. It is not deployed online in this version.

---

## License

This project is for academic and research demonstration purposes.