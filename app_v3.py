# imports
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from openai import OpenAI
from predict_v3 import predict

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object

st.set_page_config(
    page_title="Personalized Music Digital Twin",
    layout="wide"
)

DEEPSEEK_API_KEY = "sk-2305f4cb643a4a35af5ddfbe1c83175a"

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

X = np.load("results/X_v3.npy").astype(np.float32)

# load analysis
ANALYSIS_CSV = "results/song_analysis.csv"
song_analysis = pd.read_csv(ANALYSIS_CSV, index_col="song_id")

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHROMA_COLS = [f"chroma_{n}" for n in NOTE_NAMES]

class WebcamEmotionRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frames.append(image)
            if len(self.frames) > 900:
                self.frames = self.frames[-900:]
        return frame

    def get_frames(self):
        with self.lock:
            return list(self.frames)

    def clear(self):
        with self.lock:
            self.frames = []

# helpers 

def level_text(x):
    if x < 0.4: return "low"
    if x < 0.7: return "moderate"
    return "high"

def age_to_group(age):
    if age < 4:  return 0, "infant"
    if age < 8:  return 1, "child"
    if age < 18: return 2, "adolescent"
    if age < 60: return 3, "adult"
    return 4, "older adult"

def depression_to_value(score):
    if score < 10: return 0.0,  "none"
    if score < 20: return 0.33, "mild"
    if score < 30: return 0.66, "moderate"
    return 1.0, "high"

def exposure_to_value(text):
    return {"none": 0.0, "low": 0.25, "medium": 0.50, "high": 0.75, "full": 1.0}[text]

def mood_to_value(text):
    return {
        "very negative": -1.0, "negative": -0.5, "neutral": 0.0,
        "positive": 0.5, "very positive": 1.0
    }[text]

def explain_row(row):
    parts = []
    if row["liking"] >= 0.65:    parts.append("strong preference")
    elif row["liking"] >= 0.45:  parts.append("moderate preference")
    else:                        parts.append("weak preference")
    if row["valence"] > 0.20:    parts.append("positive emotional tone")
    elif row["valence"] < -0.20: parts.append("negative emotional tone")
    else:                        parts.append("balanced emotional tone")
    if row["arousal"] > 0.20:    parts.append("high activation")
    elif row["arousal"] < -0.20: parts.append("low activation")
    else:                        parts.append("moderate activation")
    return ", ".join(parts)

def persona_impact_text(openness, empathy, depression, acculturation, mood):
    points = []
    if mood > 0:        points.append("Positive mood shifts the response toward higher predicted valence.")
    elif mood < 0:      points.append("Negative mood shifts the response toward lower predicted valence.")
    else:               points.append("Neutral mood keeps the response closer to the model baseline.")
    if depression >= 0.66:  points.append("Higher depression score may reduce predicted liking and emotional positivity.")
    elif depression > 0:    points.append("Mild depression score may slightly reduce predicted liking.")
    else:                   points.append("Low depression score supports higher predicted liking.")
    if openness >= 0.7:     points.append("High openness supports broader acceptance of emotionally varied music.")
    elif openness < 0.4:    points.append("Low openness may narrow the preferred emotional range.")
    else:                   points.append("Moderate openness supports balanced music preference.")
    if empathy >= 0.7:      points.append("High empathy can increase engagement with emotionally expressive songs.")
    elif empathy < 0.4:     points.append("Low empathy may lead to a more muted emotional response.")
    else:                   points.append("Moderate empathy supports a stable emotional response.")
    if acculturation >= 0.75:   points.append("High Western music exposure strengthens the major-positive emotion association.")
    elif acculturation <= 0.25: points.append("Low Western music exposure weakens the expected major-positive emotion association.")
    else:                       points.append("Medium Western music exposure creates a moderate positive-emotion bias.")
    return points

# avatar 
AVATAR_SVG = {
    "joy": """
<svg width="110" height="110" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="54" fill="#FFD700" stroke="#333" stroke-width="3"/>
  <circle cx="42" cy="50" r="6" fill="#333"/>
  <circle cx="78" cy="50" r="6" fill="#333"/>
  <path d="M 36 74 Q 60 100 84 74" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round"/>
  <circle cx="36" cy="70" r="9" fill="#FF8C69" opacity="0.45"/>
  <circle cx="84" cy="70" r="9" fill="#FF8C69" opacity="0.45"/>
</svg>""",
    "sadness": """
<svg width="110" height="110" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="54" fill="#87CEEB" stroke="#333" stroke-width="3"/>
  <circle cx="42" cy="50" r="6" fill="#333"/>
  <circle cx="78" cy="50" r="6" fill="#333"/>
  <path d="M 38 88 Q 60 68 82 88" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M 38 46 Q 44 40 50 44" stroke="#555" stroke-width="2" fill="none"/>
  <path d="M 82 46 Q 76 40 70 44" stroke="#555" stroke-width="2" fill="none"/>
  <ellipse cx="46" cy="64" rx="2" ry="5" fill="#6BB8D4" opacity="0.7"/>
  <ellipse cx="74" cy="64" rx="2" ry="5" fill="#6BB8D4" opacity="0.7"/>
</svg>""",
    "calm": """
<svg width="110" height="110" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="54" fill="#90EE90" stroke="#333" stroke-width="3"/>
  <ellipse cx="42" cy="52" rx="7" ry="5" fill="#333"/>
  <ellipse cx="78" cy="52" rx="7" ry="5" fill="#333"/>
  <path d="M 40 78 Q 60 90 80 78" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round"/>
</svg>""",
    "tense": """
<svg width="110" height="110" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="54" fill="#FFB6C1" stroke="#333" stroke-width="3"/>
  <circle cx="42" cy="52" r="6" fill="#333"/>
  <circle cx="78" cy="52" r="6" fill="#333"/>
  <path d="M 40 84 Q 60 74 80 84" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M 34 40 Q 42 34 50 40" stroke="#333" stroke-width="3" fill="none"/>
  <path d="M 86 40 Q 78 34 70 40" stroke="#333" stroke-width="3" fill="none"/>
  <line x1="48" y1="34" x2="50" y2="41" stroke="#333" stroke-width="2"/>
  <line x1="72" y1="34" x2="70" y2="41" stroke="#333" stroke-width="2"/>
</svg>"""
}

EMOTION_META = {
    "joy":     ("Joy",     "🎉", "#FFD700"),
    "sadness": ("Sadness", "💧", "#4169E1"),
    "calm":    ("Calm",    "🍃", "#2E8B57"),
    "tense":   ("Tense",   "⚡", "#DC143C"),
}

def render_avatar(emotion):
    svg = AVATAR_SVG.get(emotion, AVATAR_SVG["calm"])
    label, emoji, color = EMOTION_META.get(emotion, ("Unknown", "❓", "#888"))
    st.markdown(
        f"""<div style="text-align:center; padding:10px;">
            {svg}
            <div style="font-size:1.15em; font-weight:700; color:{color}; margin-top:6px;">
                {emoji} {label}
            </div>
        </div>""",
        unsafe_allow_html=True
    )

# plotly gauges
def make_gauge(title, value, min_val, max_val, color):
    mid = (min_val + max_val) / 2
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 20}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1},
            "bar":  {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [min_val, mid],     "color": "#f0f0f0"},
                {"range": [mid,     max_val], "color": "#e8f4e8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# plotly valence-arousal map
EMOTION_COLORS = {
    "joy":     "#FFD700",
    "sadness": "#4169E1",
    "calm":    "#2E8B57",
    "tense":   "#DC143C",
}

def make_va_map(top10, title="Top 10 Songs — Valence-Arousal Space"):
    top10 = top10.copy()
    top10["hover"] = top10.apply(
        lambda r: (
            f"Song {int(r['song_id'])}<br>Rank: {int(r['rank'])}<br>"
            f"Emotion: {r['emotion']}<br>Valence: {r['valence']:.3f}<br>"
            f"Arousal: {r['arousal']:.3f}<br>Liking: {r['liking']:.3f}"
        ), axis=1
    )

    fig = go.Figure()

    for x0, x1, y0, y1, label, fill in [
        ( 0,  1,  0,  1, "Joy",     "rgba(255,215,  0,0.07)"),
        (-1,  0,  0,  1, "Tense",   "rgba(220, 20, 60,0.07)"),
        (-1,  0, -1,  0, "Sadness", "rgba( 65,105,225,0.07)"),
        ( 0,  1, -1,  0, "Calm",    "rgba( 46,139, 87,0.07)"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line_width=0)
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                           showarrow=False, font=dict(size=11, color="gray"), opacity=0.4)

    for emotion in top10["emotion"].unique():
        sub = top10[top10["emotion"] == emotion]
        fig.add_trace(go.Scatter(
            x=sub["valence"],
            y=sub["arousal"],
            mode="markers+text",
            name=emotion,
            marker=dict(
                size=sub["liking"] * 42 + 10,
                color=EMOTION_COLORS.get(emotion, "#888"),
                opacity=0.82,
                line=dict(width=1.2, color="black")
            ),
            text=sub["rank"].astype(str),
            textposition="middle center",
            textfont=dict(size=9, color="black"),
            hovertext=sub["hover"],
            hoverinfo="text"
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.add_vline(x=0, line_color="black", line_width=0.8)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Valence", range=[-1.05, 1.05], zeroline=False),
        yaxis=dict(title="Arousal", range=[-1.05, 1.05], zeroline=False),
        height=480,
        legend=dict(title="Emotion"),
        plot_bgcolor="white",
        margin=dict(t=50, b=40, l=50, r=30)
    )
    return fig

# deepseek 
def deepseek_interpretation(persona_summary, best_song, top_summary, impact_points):
    prompt = f"""
You are explaining results from a personalized music digital twin.

Rules:
Use simple professional English.
Do not claim clinical diagnosis.
Do not claim real EEG or brain measurement.
Do not say the model proves anything.
Do not use bullet points.
Write 2 short paragraphs explaining the recommendation.
Then write a detailed personal profile for this listener (two paragraphs).

Persona:
{persona_summary}

Top recommendation:
song_id: {int(best_song["song_id"])}
emotion: {best_song["emotion"]}
valence: {best_song["valence"]:.3f}
arousal: {best_song["arousal"]:.3f}
liking: {best_song["liking"]:.3f}

Top 10 summary:
{top_summary}

Persona impact:
{" ".join(impact_points)}

Explain why this recommendation fits the listener profile.
"""
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

def deepseek_acoustic_explanation(selected_song_id, song_analysis, top10):
    if selected_song_id not in song_analysis.index:
        return None

    s = song_analysis.loc[selected_song_id]

    top_ids    = top10["song_id"].tolist()
    top_found  = [sid for sid in top_ids if sid in song_analysis.index]
    top_modes  = [song_analysis.loc[sid]["mode"] for sid in top_found]
    top_keys   = [song_analysis.loc[sid]["key"] for sid in top_found]
    top_tempos = [round(song_analysis.loc[sid]["tempo"], 1) for sid in top_found]

    prompt = f"""
You are explaining acoustic music analysis results in simple professional English.
Do not use bullet points.
Write exactly two paragraphs.
First paragraph explains the selected song only.
Second paragraph explains the top 10 recommended songs and how they compare to the selected song.

Selected song:
song_id: {selected_song_id}
key: {s["key"]}
mode: {s["mode"]}
confidence: {s["confidence"]}
tempo: {s["tempo"]} bpm
energy: {s["rms"]:.4f}
brightness: {s["centroid"]:.0f} Hz

Top 10 recommended songs:
keys: {top_keys}
modes: {top_modes}
tempos: {top_tempos}

Explain what the chroma profile reveals about the selected song, what its key and mode mean emotionally, then explain how the top 10 songs compare acoustically to the selected song.
"""
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content



# prediction helper
def run_prediction(X_data, persona_vec):
    batch = np.repeat(persona_vec.reshape(1, -1), len(X_data), axis=0)
    out = predict(X_data, batch)
    df = pd.DataFrame({
        "song_id": np.arange(len(X_data)),
        "valence": out["valence"].astype(float),
        "arousal": out["arousal"].astype(float),
        "emotion": out["emotion"],
        "liking":  out["liking"].astype(float)
    })
    df = df.sort_values("liking", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["explanation"] = df.apply(explain_row, axis=1)
    return df

# page layout
st.title("Personalized Music Digital Twin")
st.write(
    "This system predicts how a listener profile may respond to music "
    "using persona-conditioned valence, arousal, emotion, and liking scores."
)

st.sidebar.header("Listener Profile")

st.sidebar.markdown("""
<div style="font-family:'Syne',sans-serif; font-size:0.62rem; font-weight:700;
letter-spacing:0.18em; text-transform:uppercase; color:#4a7ace; margin-bottom:2px;">
Song ID
</div>""", unsafe_allow_html=True)

selected_song_id = st.sidebar.number_input("Song ID", min_value=1, max_value=9999, value=10, step=1)

input_mode = st.sidebar.radio(
    "Input mode",
    ["Manual input", "CSV upload", "Start recording"],
    index=0
)

auto_run_prediction = False

years_training = 5
openness_raw = 60
empathy_raw = 60
age_years = 25
beck_score = 0
western_exposure = "medium"
mood_report = "neutral"
motor_impairment = 0.0

if input_mode == "Manual input":
    years_training   = st.sidebar.number_input("Years of music training", 0, 60, 5)
    openness_raw     = st.sidebar.slider("Openness score", 0, 100, 60)
    empathy_raw      = st.sidebar.slider("Empathy score", 0, 100, 60)
    age_years        = st.sidebar.number_input("Age", 1, 100, 25)
    beck_score       = st.sidebar.number_input("Beck depression score", 0, 63, 0)
    western_exposure = st.sidebar.selectbox("Western music exposure",
                                            ["none", "low", "medium", "high", "full"],
                                            index=2)
    mood_report = st.sidebar.selectbox(
    "Current mood",
    ["very negative", "negative", "neutral", "positive", "very positive"],
    index=2
    )
    motor_impairment = st.sidebar.slider("Motor impairment (0 none, 1 severe)", 0.0, 1.0, 0.0, step=0.05)

elif input_mode == "CSV upload":
    uploaded_file    = st.sidebar.file_uploader("CSV file", type=["csv"])
    motor_impairment = st.sidebar.slider("Motor impairment (0 none, 1 severe)", 0.0, 1.0, 0.0, step=0.05, key="csv_motor")

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        row = df_csv.iloc[0]

        years_training   = int(row["years_training"])
        openness_raw     = float(row["openness_score"])
        empathy_raw      = float(row["empathy_score"])
        age_years        = int(row["age"])
        beck_score       = float(row["beck_score"])
        western_exposure = row["western_exposure"]
        mood_report      = row["mood"]
    else:
        st.sidebar.info("Upload a persona CSV to continue.")

else:
    st.sidebar.write("Enter years of training and age, then start camera recording.")
    rec_years_training = st.sidebar.number_input("Years of music training", 0, 60, 5, key="rec_years_training")
    rec_age_years      = st.sidebar.number_input("Age", 1, 100, 25, key="rec_age_years")
    motor_impairment   = st.sidebar.slider("Motor impairment (0 none, 1 severe)", 0.0, 1.0, 0.0, step=0.05, key="rec_motor")

    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc is not installed. Run: pip install streamlit-webrtc")
    else:
        st.subheader("Webcam Recording")
        st.write("Allow camera permission in the browser. Keep your face visible. When finished, click the processing button below.")

        ctx = webrtc_streamer(
            key="webcam-emotion-recorder",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=WebcamEmotionRecorder,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            frame_count = len(ctx.video_processor.get_frames())
            st.caption(f"Recorded frames in memory: {frame_count}")

            if st.button("Finished recording and process", use_container_width=True):
                frames = ctx.video_processor.get_frames()
                if len(frames) == 0:
                    st.error("No webcam frames were recorded yet.")
                else:
                    with st.spinner("Processing webcam recording with DeepFace..."):
                        from extract_video_emotion_fixed import analyze_frames, compute_persona_from_emotions

                        emotion_df = analyze_frames(frames, every_n_frames=10)
                        webcam_persona = compute_persona_from_emotions(
                            emotion_df,
                            years_training=rec_years_training,
                            age=rec_age_years
                        )

                        output_path = Path("results/deepface/persona_input_for_app.csv")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame([webcam_persona]).to_csv(output_path, index=False)

                        st.session_state["webcam_persona"] = webcam_persona
                        st.session_state["webcam_emotion_df"] = emotion_df
                        auto_run_prediction = True

                        if webcam_persona.get("face_detected", 1) == 0:
                            st.warning(
                                "No real human face was detected. The app saved a zero-value persona "
                                "for all inferred fields, while keeping age and years of training."
                            )
                        else:
                            st.success("Human face detected and recording processed successfully.")

                        st.success(f"CSV saved to {output_path}")

    if "webcam_persona" in st.session_state:
        row = st.session_state["webcam_persona"]
        years_training   = int(row["years_training"])
        openness_raw     = float(row["openness_score"])
        empathy_raw      = float(row["empathy_score"])
        age_years        = int(row["age"])
        beck_score       = float(row["beck_score"])
        western_exposure = row["western_exposure"]
        mood_report      = row["mood"]
        

expertise             = float(np.interp(years_training, [0, 3, 7, 12, 20], [0.0, 0.25, 0.50, 0.75, 1.0]))
openness              = float(openness_raw / 100)
empathy               = float(empathy_raw / 100)
age_group, age_label  = age_to_group(age_years)
depression, depression_label = depression_to_value(beck_score)
acculturation         = exposure_to_value(western_exposure)
mood                  = mood_to_value(mood_report)

persona = np.array(
    [expertise, openness, empathy, float(age_group), depression, acculturation, mood, motor_impairment],
    dtype=np.float32
)

st.markdown('<div class="sec-label">00</div><div class="sec-title">Selected Song Analysis</div>', unsafe_allow_html=True)

if selected_song_id in song_analysis.index:
    s = song_analysis.loc[selected_song_id]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Key", f"{s['key']} {s['mode'].capitalize()}")
    c2.metric("Confidence", f"{s['confidence']:.2f}")
    c3.metric("Tempo", f"{s['tempo']:.0f} bpm")
    c4.metric("Energy", f"{s['rms']:.4f}")
    c5.metric("Brightness", f"{s['centroid']:.0f} Hz")
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
else:
    st.warning(f"Song ID {selected_song_id} not found in analysis.")

st.markdown("<hr style='border-color:#1e2a40; margin:24px 0;'>", unsafe_allow_html=True)


# section 1: persona table 
st.subheader("1. Interpreted Persona")

persona_table = pd.DataFrame({
    "Dimension":   ["Music training", "Openness", "Empathy", "Age",
                    "Depression", "Western exposure", "Mood", "Motor impairment"],
    "User input":  [f"{years_training} years", openness_raw, empathy_raw, age_years,
                    beck_score, western_exposure, mood_report, motor_impairment],
    "Model value": [round(expertise, 3), round(openness, 3), round(empathy, 3),
                    age_group, depression, round(acculturation, 3), mood, round(motor_impairment, 3)],
    "Meaning":     [level_text(expertise), level_text(openness), level_text(empathy),
                    age_label, depression_label, western_exposure, mood_report, level_text(motor_impairment)]
})

st.dataframe(persona_table, use_container_width=True, hide_index=True)

persona_summary = (
    f"{age_label} listener, {years_training} years of music training, "
    f"{level_text(openness)} openness, {level_text(empathy)} empathy, "
    f"{western_exposure} Western music exposure, {mood_report} mood, "
    f"{depression_label} depression level, "
    f"{level_text(motor_impairment)} motor impairment"
)
st.write(persona_summary)

# run button
if auto_run_prediction or st.button("Run personalized prediction", use_container_width=True):
    df    = run_prediction(X, persona)
    top10 = df.head(10).copy()
    best  = top10.iloc[0]

    top_summary = (
        f"Top 10 mean liking: {top10['liking'].mean():.3f}, "
        f"mean valence: {top10['valence'].mean():.3f}, "
        f"mean arousal: {top10['arousal'].mean():.3f}, "
        f"dominant emotion: {top10['emotion'].mode()[0]}"
    )

    impact_points = persona_impact_text(openness, empathy, depression, acculturation, mood)

    st.session_state.update({
        "df": df, "top10": top10, "best": best,
        "top_summary": top_summary,
        "impact_points": impact_points,
        "persona_summary": persona_summary,
        "base_persona": persona.copy(),
    })
    st.session_state.pop("interpretation", None)

# results
if "df" in st.session_state:

    df              = st.session_state["df"]
    top10           = st.session_state["top10"]
    best            = st.session_state["best"]
    top_summary     = st.session_state["top_summary"]
    impact_points   = st.session_state["impact_points"]
    persona_summary = st.session_state["persona_summary"]
    base_persona    = st.session_state["base_persona"]

    st.divider()

    # section 2: avatar + gauges
    st.subheader("2. Digital Twin Response Summary")

    col_avatar, col_gauges = st.columns([1, 3])

    with col_avatar:
        render_avatar(best["emotion"])

    with col_gauges:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(
            make_gauge("Top Liking",   best["liking"],           0, 1, "#4CAF50"),
            use_container_width=True
        )
        g2.plotly_chart(
            make_gauge("Mean Valence", top10["valence"].mean(), -1, 1, "#2196F3"),
            use_container_width=True
        )
        g3.plotly_chart(
            make_gauge("Mean Arousal", top10["arousal"].mean(), -1, 1, "#FF9800"),
            use_container_width=True
        )

    st.write(
        f"The digital twin predicts a mainly **{best['emotion']}** response for the strongest "
        f"recommendation. Top 10 mean liking {top10['liking'].mean():.3f}, "
        f"valence {top10['valence'].mean():.3f}, arousal {top10['arousal'].mean():.3f}."
    )

    st.markdown("<hr style='border-color:#1e2a40; margin:24px 0;'>", unsafe_allow_html=True)
    st.subheader("3. Acoustic Profile")
    st.markdown('<div style="font-size:0.8rem; color:#6a8ab0; margin-bottom:8px;">Chroma profile — selected song vs top 10</div>', unsafe_allow_html=True)

    if selected_song_id in song_analysis.index:
        s     = song_analysis.loc[selected_song_id]
        cvals = [s[c] for c in CHROMA_COLS]
        key   = s["key"]
        mode  = s["mode"]

        top_ids   = top10["song_id"].tolist()
        top_found = [sid for sid in top_ids if sid in song_analysis.index]

        combined_fig = go.Figure()

        for sid in top_found:
            ts        = song_analysis.loc[sid]
            tcvals    = [ts[c] for c in CHROMA_COLS]
            rank      = top10[top10["song_id"] == sid]["rank"].values[0]
            emotion   = top10[top10["song_id"] == sid]["emotion"].values[0]
            if emotion == "joy":
                ec = "#000000"
            elif emotion == "calm":
                ec = "#0a4a1e"
            else:
                ec = EMOTION_COLORS.get(emotion, "#888")

            combined_fig.add_trace(go.Scatter(
                x=NOTE_NAMES,
                y=tcvals,
                mode="lines+markers",
                name=f"#{rank} song {sid}",
                line=dict(color=ec, width=1.2),
                marker=dict(size=4, color=ec),
                opacity=0.6,
            ))

        combined_fig.add_trace(go.Scatter(
            x=NOTE_NAMES,
            y=cvals,
            mode="lines+markers",
            name=f"song {selected_song_id} ({key} {mode})",
            line=dict(color="#cc2040", width=3),
            marker=dict(size=8, color="#ff4060"),
        ))

        combined_fig.update_layout(
            title=dict(text=f"Song {selected_song_id} ({key} {mode.capitalize()}) vs Top 10", font=dict(size=13, color="#333", family="Syne")),
            xaxis=dict(
                title=dict(text="pitch class", font=dict(color="#333", size=13, family="Syne")),
                tickfont=dict(color="#333", size=12, family="Syne"),
                gridcolor="#dddddd"
            ),
            yaxis=dict(
                title=dict(text="chroma strength", font=dict(color="#333", size=13, family="Syne")),
                tickfont=dict(color="#333", size=11),
                gridcolor="#dddddd"
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(t=50, b=40, l=50, r=20),
            legend=dict(bgcolor="white", bordercolor="#dddddd", borderwidth=1, font=dict(color="#333", size=10)),
            font=dict(color="#333")
        )
        st.plotly_chart(combined_fig, use_container_width=True)
        if st.button("Generate acoustic explanation", use_container_width=True):
            with st.spinner("Generating..."):
                acoustic_text = deepseek_acoustic_explanation(selected_song_id, song_analysis, top10)
                st.session_state["acoustic_explanation"] = acoustic_text

        if "acoustic_explanation" in st.session_state:
            st.markdown(
                f"""<div style="background:#f8f9fa; border-left:4px solid #2196F3;
                    padding:18px 20px; border-radius:6px; font-size:0.95em; line-height:1.75; margin-top:12px;">
                    {st.session_state['acoustic_explanation'].replace(chr(10), '<br>')}
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("song not found")

    # section 3: top 10 table 

    st.subheader("4. Top 10 Personalized Recommendations")

    st.dataframe(
        top10[["rank", "song_id", "emotion", "liking", "valence", "arousal", "explanation"]],
        use_container_width=True,
        hide_index=True
    )

    # section 4: valence-arousal map 

    st.subheader("5. Valence-Arousal Map")
    st.plotly_chart(make_va_map(top10), use_container_width=True)
    st.caption("Point size = predicted liking. Number inside each point = rank.")

    # section 5: deepseek personality explanation 

    st.subheader("6. DeepSeek Personality Explanation")

    with st.expander("Rule-based persona impact summary", expanded=False):
        for point in impact_points:
            st.write(point)

    if st.button("Generate DeepSeek personality explanation", use_container_width=True):
        with st.spinner("Generating..."):
            interpretation = deepseek_interpretation(
                persona_summary, best, top_summary, impact_points
            )
            st.session_state["interpretation"] = interpretation

    if "interpretation" in st.session_state:
        st.markdown(
            f"""<div style="background:#f8f9fa; border-left:4px solid #4CAF50;
                padding:18px 20px; border-radius:6px; font-size:0.95em; line-height:1.75;">
                {st.session_state['interpretation'].replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )

    #  section 6: what-if mood simulator 
    st.divider()
    st.subheader("7. What-if Mood Simulator")
    st.write(
        "Shift the mood value to see how the top recommendation and valence-arousal map "
        "would change — compared to the original prediction."
    )

    whatif_mood = st.slider(
        "Simulated mood",
        min_value=-1.0, max_value=1.0,
        value=float(base_persona[6]),
        step=0.05,
        format="%.2f"
    )

    whatif_persona    = base_persona.copy()
    whatif_persona[6] = np.float32(whatif_mood)

    whatif_df   = run_prediction(X, whatif_persona)
    whatif_top  = whatif_df.head(10)
    whatif_best = whatif_top.iloc[0]

    orig_top  = st.session_state["top10"]
    orig_best = st.session_state["best"]

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Mood", f"{whatif_mood:+.2f}")
    w2.metric(
        "Top liking",
        f"{whatif_best['liking']:.3f}",
        delta=f"{whatif_best['liking'] - orig_best['liking']:+.3f}"
    )
    w3.metric(
        "Mean valence",
        f"{whatif_top['valence'].mean():.3f}",
        delta=f"{whatif_top['valence'].mean() - orig_top['valence'].mean():+.3f}"
    )
    w4.metric(
        "Mean arousal",
        f"{whatif_top['arousal'].mean():.3f}",
        delta=f"{whatif_top['arousal'].mean() - orig_top['arousal'].mean():+.3f}"
    )

    st.plotly_chart(
        make_va_map(whatif_top, title=f"What-if Mood = {whatif_mood:+.2f} — Valence-Arousal Map"),
        use_container_width=True
    )

    #  section 7: download 
    st.divider()
    st.subheader("8. Download Results")

    st.download_button(
        label="Download full prediction results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="personalized_music_results.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("Set the listener profile and run personalized prediction.")