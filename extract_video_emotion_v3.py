import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

# video path
VIDEO_PATH = r"E:\NeuEng\audi_p_p\dataset\test2.mp4"

# output path
OUTPUT_DIR = r"E:\NeuEng\audi_p_p\results\deepface"

EMOTION_COLUMNS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral", "negative", "valence"]

MIN_VALID_FACE_FRAMES = 3
MIN_FACE_AREA_RATIO = 0.015
MIN_FRAME_BRIGHTNESS = 25.0
MIN_FRAME_STD = 12.0
MIN_FACE_CONFIDENCE = 0.80


def clip(x, low, high):
    return max(low, min(high, x))

def empty_emotion_df():
    return pd.DataFrame(columns=EMOTION_COLUMNS)

def zero_persona(years_training, age, reason="no_face"):
    return {
        "years_training": int(years_training),
        "openness_score": 0.0,
        "empathy_score": 0.0,
        "age": int(age),
        "beck_score": 0.0,
        "western_exposure": "none",
        "mood": "neutral",
        "detected_mean_valence": 0.0,
        "detected_mean_happy": 0.0,
        "detected_mean_sad": 0.0,
        "detected_mean_negative": 0.0,
        "detected_mean_surprise": 0.0,
        "detected_valence_std": 0.0,
        "analyzed_frames": 0,
        "face_detected": 0,
        "recording_status": reason,
    }

def frame_stats(frame):
    if frame is None:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())

def is_black_or_empty_frame(frame):
    mean, std = frame_stats(frame)
    return mean < MIN_FRAME_BRIGHTNESS or std < MIN_FRAME_STD

def validate_face_region(frame, face_obj):
    if frame is None or not isinstance(face_obj, dict):
        return False, 0.0, 0.0
    h, w = frame.shape[:2]
    area = face_obj.get("facial_area", {}) or {}
    fw = float(area.get("w", 0) or 0)
    fh = float(area.get("h", 0) or 0)
    face_area_ratio = (fw * fh) / float(max(h * w, 1))
    confidence = face_obj.get("confidence", None)
    if confidence is None:
        confidence = 1.0
    confidence = float(confidence)
    valid = face_area_ratio >= MIN_FACE_AREA_RATIO and confidence >= MIN_FACE_CONFIDENCE
    return valid, confidence, face_area_ratio

def analyze_frame(frame):
    brightness, std = frame_stats(frame)
    if is_black_or_empty_frame(frame):
        return None
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=True,
            align=False,
        )
        if not faces:
            return None
        valid_faces = []
        for face_obj in faces:
            valid, conf, area_ratio = validate_face_region(frame, face_obj)
            if valid:
                valid_faces.append((face_obj, conf, area_ratio))
        if not valid_faces:
            return None
        valid_faces.sort(key=lambda x: x[2], reverse=True)
        _, face_confidence, face_area_ratio = valid_faces[0]
        res = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        e = res.get("emotion", {})
        happy    = e.get("happy", 0) / 100
        sad      = e.get("sad", 0) / 100
        angry    = e.get("angry", 0) / 100
        fear     = e.get("fear", 0) / 100
        disgust  = e.get("disgust", 0) / 100
        surprise = e.get("surprise", 0) / 100
        neutral  = e.get("neutral", 0) / 100
        negative = sad + angry + fear + disgust
        valence  = happy - negative
        return {
            "happy": happy, "sad": sad, "angry": angry,
            "fear": fear, "disgust": disgust, "surprise": surprise,
            "neutral": neutral, "negative": negative, "valence": valence,
            "face_confidence": face_confidence,
            "face_area_ratio": face_area_ratio,
            "frame_brightness": brightness,
        }
    except Exception:
        return None

def analyze_video(video_path, seconds_per_frame=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    step = max(1, int(fps * seconds_per_frame))
    vals = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step == 0:
            row = analyze_frame(frame)
            if row is not None:
                vals.append(row)
        frame_id += 1
    cap.release()
    if len(vals) == 0:
        return empty_emotion_df()
    return pd.DataFrame(vals)

def compute_persona(df, years_training, age):
    if df is None or len(df) == 0:
        return zero_persona(years_training, age)
    if len(df) < MIN_VALID_FACE_FRAMES:
        return zero_persona(years_training, age, reason="too_few_frames")
    if "face_area_ratio" in df.columns and float(df["face_area_ratio"].mean()) < MIN_FACE_AREA_RATIO:
        return zero_persona(years_training, age, reason="face_too_small")
    if "frame_brightness" in df.columns and float(df["frame_brightness"].mean()) < MIN_FRAME_BRIGHTNESS:
        return zero_persona(years_training, age, reason="too_dark")
    mean_valence  = float(df["valence"].mean())
    mean_happy    = float(df["happy"].mean())
    mean_sad      = float(df["sad"].mean())
    mean_negative = float(df["negative"].mean())
    mean_surprise = float(df["surprise"].mean())
    var = float(df["valence"].std())
    if np.isnan(var):
        var = 0.0
    mood_value = clip(mean_valence, -1, 1)
    if mood_value <= -0.6:
        mood = "very negative"
    elif mood_value <= -0.2:
        mood = "negative"
    elif mood_value < 0.2:
        mood = "neutral"
    elif mood_value < 0.6:
        mood = "positive"
    else:
        mood = "very positive"
    openness      = 100 * clip(0.4 + 0.3 * mean_surprise + 0.2 * var + 0.1 * mean_happy, 0, 1)
    empathy       = 100 * clip(0.4 + 0.3 * mean_sad + 0.1 * mean_happy + 0.2 * var, 0, 1)
    beck          = 63  * clip(0.2 + 0.5 * mean_sad + 0.2 * mean_negative - 0.3 * mean_happy, 0, 1)
    western_score = 100 * clip(0.5 + 0.2 * mean_happy + 0.1 * mean_valence, 0, 1)
    if western_score < 20:
        western = "none"
    elif western_score < 40:
        western = "low"
    elif western_score < 60:
        western = "medium"
    elif western_score < 80:
        western = "high"
    else:
        western = "full"
    return {
        "years_training": int(years_training),
        "openness_score": round(openness, 2),
        "empathy_score": round(empathy, 2),
        "age": int(age),
        "beck_score": round(beck, 2),
        "western_exposure": western,
        "mood": mood,
        "detected_mean_valence": round(mean_valence, 4),
        "detected_mean_happy": round(mean_happy, 4),
        "detected_mean_sad": round(mean_sad, 4),
        "detected_mean_negative": round(mean_negative, 4),
        "detected_mean_surprise": round(mean_surprise, 4),
        "detected_valence_std": round(var, 4),
        "analyzed_frames": int(len(df)),
        "face_detected": 1,
        "recording_status": "face_detected",
    }

# user input
years_training = int(input("years of training: "))
age            = int(input("age: "))

# analyze
print("analyzing video")
emotion_df = analyze_video(VIDEO_PATH)
persona    = compute_persona(emotion_df, years_training, age)

# save
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "persona_input_for_app.csv")
pd.DataFrame([persona]).to_csv(output_path, index=False)

print("done")
print(output_path)
print(persona)