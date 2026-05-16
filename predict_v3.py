# imports
import numpy as np
import joblib
import onnxruntime as ort

# load
scaler  = joblib.load("results/scaler_v3.pkl")
session = ort.InferenceSession("results/digital_twin_v3.onnx")

emotion_labels = ["joy", "sadness", "calm", "tense"]

def predict(audio_features, persona_features):
    audio_features   = np.asarray(audio_features,   dtype=np.float32)
    persona_features = np.asarray(persona_features, dtype=np.float32)

    if audio_features.ndim == 1:
        audio_features = audio_features.reshape(1, -1)

    if persona_features.ndim == 1:
        persona_features = persona_features.reshape(1, -1)

    audio_features = scaler.transform(audio_features).astype(np.float32)

    outputs = session.run(None, {
        "audio_features":   audio_features,
        "persona_features": persona_features
    })

    valence    = outputs[0].reshape(-1)
    arousal    = outputs[1].reshape(-1)
    logits     = outputs[2]
    liking     = outputs[3].reshape(-1)
    emotion_id = np.argmax(logits, axis=1)
    emotion    = [emotion_labels[i] for i in emotion_id]

    return {
        "valence": valence,
        "arousal": arousal,
        "emotion": emotion,
        "liking":  liking
    }

if __name__ == "__main__":
    X = np.load("results/X_v3.npy").astype(np.float32)

    audio = X[0]

    persona = np.array([
        0.80,
        0.85,
        0.75,
        3.00,
        0.00,
        0.90,
        0.80,
        0.00
    ], dtype=np.float32)

    result = predict(audio, persona)

    print("valence", round(float(result["valence"][0]), 4))
    print("arousal", round(float(result["arousal"][0]), 4))
    print("emotion", result["emotion"][0])
    print("liking",  round(float(result["liking"][0]), 4))