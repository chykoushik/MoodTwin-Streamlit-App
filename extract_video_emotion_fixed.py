import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


def clip(x, low, high):
    return max(low, min(high, x))


def analyze_video(video_path, seconds_per_frame=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video not found or cannot be opened: {video_path}")

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
        raise RuntimeError("No usable face/emotion signal was detected in the video.")

    return pd.DataFrame(vals)


def analyze_frames(frames, every_n_frames=10):
    vals = []
    for idx, frame in enumerate(frames):
        if idx % every_n_frames == 0:
            row = analyze_frame(frame)
            if row is not None:
                vals.append(row)

    if len(vals) == 0:
        raise RuntimeError("No usable face/emotion signal was detected from webcam frames.")

    return pd.DataFrame(vals)


def analyze_frame(frame):
    try:
        res = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        if isinstance(res, list):
            res = res[0]

        e = res["emotion"]
        happy = e.get("happy", 0) / 100
        sad = e.get("sad", 0) / 100
        angry = e.get("angry", 0) / 100
        fear = e.get("fear", 0) / 100
        disgust = e.get("disgust", 0) / 100
        surprise = e.get("surprise", 0) / 100
        neutral = e.get("neutral", 0) / 100

        negative = sad + angry + fear + disgust
        valence = happy - negative

        return {
            "happy": happy,
            "sad": sad,
            "angry": angry,
            "fear": fear,
            "disgust": disgust,
            "surprise": surprise,
            "neutral": neutral,
            "negative": negative,
            "valence": valence,
        }
    except Exception:
        return None


def compute_persona_from_emotions(df, years_training, age):
    mean_valence = float(df["valence"].mean())
    mean_happy = float(df["happy"].mean())
    mean_sad = float(df["sad"].mean())
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

    openness = 100 * clip(0.4 + 0.3 * mean_surprise + 0.2 * var + 0.1 * mean_happy, 0, 1)
    empathy = 100 * clip(0.4 + 0.3 * mean_sad + 0.1 * mean_happy + 0.2 * var, 0, 1)
    beck = 63 * clip(0.2 + 0.5 * mean_sad + 0.2 * mean_negative - 0.3 * mean_happy, 0, 1)
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
    }


def video_to_persona_csv(video_path, output_path, years_training, age, seconds_per_frame=1.0):
    emotion_df = analyze_video(video_path, seconds_per_frame=seconds_per_frame)
    persona = compute_persona_from_emotions(emotion_df, years_training, age)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([persona]).to_csv(output_path, index=False)
    return persona, emotion_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="results/deepface/persona_input_for_app.csv", help="Output CSV path")
    parser.add_argument("--years-training", type=int, required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--seconds-per-frame", type=float, default=1.0)
    args = parser.parse_args()

    persona, _ = video_to_persona_csv(
        video_path=args.video,
        output_path=args.output,
        years_training=args.years_training,
        age=args.age,
        seconds_per_frame=args.seconds_per_frame,
    )

    print("done")
    print(os.path.abspath(args.output))
    print(persona)


if __name__ == "__main__":
    main()
