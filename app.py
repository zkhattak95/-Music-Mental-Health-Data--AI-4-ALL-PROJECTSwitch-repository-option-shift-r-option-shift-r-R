from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.recommendations import GENRE_COLUMN_MAP, recommend_genres, wellness_tips


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "burnout_model.joblib"
MUSIC_PATH = ROOT / "data" / "music_survey_cleaned.csv"
BURNOUT_SAMPLE_PATH = ROOT / "data" / "burnout_dashboard_sample.csv"
CSS_PATH = ROOT / "assets" / "styles.css"


st.set_page_config(
    page_title="Mind & Melody",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_music_data() -> pd.DataFrame:
    if not MUSIC_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MUSIC_PATH)


@st.cache_data
def load_burnout_sample() -> pd.DataFrame:
    if not BURNOUT_SAMPLE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BURNOUT_SAMPLE_PATH)


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Important:</strong> Mind & Melody is an educational wellness project.
        It does not diagnose burnout or any mental-health condition, and it is not a
        substitute for professional support. The app does not save your check-in answers.
        </div>
        """,
        unsafe_allow_html=True,
    )


def home_page() -> None:
    st.markdown(
        """
        <section class="hero">
            <span class="hero-kicker">AI4ALL Group 11C</span>
            <h1>Mind & Melody</h1>
            <p>
                A student-centered wellness experience that combines a burnout-risk
                classifier, practical check-in guidance, and data-informed music
                recommendations in one calm, private interface.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    cards = [
        (
            col1,
            "🧭 Burnout Check-In",
            "Estimate a low, medium, or high burnout-risk pattern from lifestyle and academic inputs."
        ),
        (
            col2,
            "🎵 Music Support",
            "Explore genres frequently reported by survey participants who said music improved their wellbeing."
        ),
        (
            col3,
            "📊 Data Insights",
            "View clear visual summaries of student burnout and music-and-mental-health survey patterns."
        ),
    ]

    for column, title, copy in cards:
        with column:
            st.markdown(
                f'<div class="card"><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("## How it works")
    st.write(
        "Complete a private check-in, review the model's pattern-based estimate, "
        "and explore supportive next steps and music ideas. No account is required."
    )
    render_disclaimer()


def burnout_page() -> None:
    st.markdown('<h1 class="section-title">Burnout Check-In</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Answer based on your recent routine. The model uses patterns from a cleaned student burnout dataset.</p>',
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error("The model file is missing. Add models/burnout_model.joblib and reload the app.")
        return

    with st.form("burnout_checkin"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 29, 20)
            gender = st.selectbox("Gender category used by the dataset", ["Female", "Male", "Other"])
            academic_year = st.selectbox("Academic year", [1, 2, 3, 4], index=1)
            study_hours = st.slider("Study hours per day", 0.0, 14.0, 5.0, 0.5)
            exam_pressure = st.slider("Exam pressure", 1.0, 10.0, 6.0, 0.5)
            academic_performance = st.slider("Academic performance score", 42.0, 98.0, 71.0, 1.0)
            stress_level = st.slider("Stress level", 0.0, 10.0, 5.0, 0.5)
            anxiety_score = st.slider("Anxiety self-rating", 0.0, 10.0, 3.0, 0.5)

        with col2:
            depression_score = st.slider("Low-mood self-rating", 0.0, 9.0, 2.0, 0.5)
            sleep_hours = st.slider("Sleep hours per night", 3.0, 10.0, 7.0, 0.5)
            physical_activity = st.slider("Physical activity hours per day", 0.0, 7.0, 2.0, 0.5)
            social_support = st.slider("Social support", 0.0, 10.0, 6.0, 0.5)
            screen_time = st.slider("Screen time per day", 1.0, 12.0, 5.0, 0.5)
            internet_usage = st.slider("Internet use per day", 1.0, 14.0, 5.0, 0.5)
            financial_stress = st.slider("Financial stress", 0.0, 10.0, 4.0, 0.5)
            family_expectation = st.slider("Family expectation pressure", 0.0, 10.0, 6.0, 0.5)

        submitted = st.form_submit_button("View my check-in result", use_container_width=True)

    if not submitted:
        render_disclaimer()
        return

    row = {
        "age": age,
        "gender": gender,
        "academic_year": academic_year,
        "study_hours_per_day": study_hours,
        "exam_pressure": exam_pressure,
        "academic_performance": academic_performance,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "social_support": social_support,
        "screen_time": screen_time,
        "internet_usage": internet_usage,
        "financial_stress": financial_stress,
        "family_expectation": family_expectation,
    }

    input_df = pd.DataFrame([row])
    predicted_risk = str(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(predicted_risk)])

    risk_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(predicted_risk, "")

    st.markdown(
        f"""
        <div class="risk-box {risk_class}">
            <span class="pill">Pattern-based estimate</span>
            <h2>{predicted_risk} burnout-risk pattern</h2>
            <p>
                Model confidence for this category: <strong>{confidence:.0%}</strong>.
                Confidence is not the same as a diagnosis or certainty.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {
            "Risk level": classes,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    chart = px.bar(
        probability_df,
        x="Risk level",
        y="Probability",
        text_auto=".0%",
        title="Model probability by category",
    )
    chart.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Supportive next steps")
    for tip in wellness_tips(row):
        st.markdown(f"- {tip}")

    if predicted_risk == "High":
        st.info(
            "Consider sharing how you are feeling with a trusted adult, counselor, "
            "health professional, or campus support person. If you feel unsafe or need "
            "urgent help, contact local emergency services."
        )

    render_disclaimer()


def music_page() -> None:
    st.markdown('<h1 class="section-title">Music Support</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Discover a small, data-informed starting point for your next playlist.</p>',
        unsafe_allow_html=True,
    )

    music = load_music_data()
    if music.empty:
        st.error("The music survey file is missing from data/music_survey_cleaned.csv.")
        return

    genres = sorted([genre for genre in GENRE_COLUMN_MAP if genre in set(music["fav_genre"].dropna())])

    col1, col2 = st.columns(2)
    with col1:
        favorite_genre = st.selectbox("Your favorite genre", genres, index=genres.index("Pop") if "Pop" in genres else 0)
    with col2:
        goal = st.selectbox("What would you like the playlist to support?", [
            "Calm down",
            "Focus",
            "Lift my mood",
            "Prepare for sleep",
        ])

    if st.button("Create my music suggestion", use_container_width=True):
        result = recommend_genres(music, favorite_genre, goal)

        st.subheader("Your starting genres")
        genre_cols = st.columns(len(result["genres"]))
        for column, genre in zip(genre_cols, result["genres"]):
            with column:
                st.markdown(
                    f'<div class="card"><h3>🎧 {genre}</h3><p>Add a few familiar tracks first, then explore gradually.</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Listening approach")
        st.write(result["session"])
        st.write(result["approach"])
        st.caption(
            f"Genre ranking used patterns from {result['sample_size']:,} relevant survey responses "
            "among participants who reported that music improved their wellbeing."
        )

    st.markdown("---")
    st.write(
        "Music affects people differently. Stop or switch tracks when listening feels distracting, "
        "uncomfortable, or unhelpful."
    )
    render_disclaimer()


def insights_page() -> None:
    st.markdown('<h1 class="section-title">Data Insights</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Explore high-level patterns from the two project datasets.</p>',
        unsafe_allow_html=True,
    )

    burnout = load_burnout_sample()
    music = load_music_data()

    tab1, tab2 = st.tabs(["Student burnout", "Music and wellbeing"])

    with tab1:
        if burnout.empty:
            st.warning("Burnout dashboard sample is unavailable.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Dashboard sample", f"{len(burnout):,}")
            col2.metric("Average sleep", f"{burnout['sleep_hours'].mean():.1f} hrs")
            col3.metric("Average stress", f"{burnout['stress_level'].mean():.1f}/10")

            risk_counts = burnout["risk_level"].value_counts().rename_axis("Risk level").reset_index(name="Students")
            st.plotly_chart(
                px.bar(risk_counts, x="Risk level", y="Students", title="Burnout-risk distribution"),
                use_container_width=True,
            )

            sleep_by_risk = (
                burnout.groupby("risk_level", as_index=False)["sleep_hours"]
                .mean()
                .rename(columns={"sleep_hours": "Average sleep hours"})
            )
            st.plotly_chart(
                px.bar(
                    sleep_by_risk,
                    x="risk_level",
                    y="Average sleep hours",
                    title="Average sleep hours by burnout-risk level",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                px.box(
                    burnout.sample(min(8000, len(burnout)), random_state=42),
                    x="risk_level",
                    y="stress_level",
                    title="Stress distribution by burnout-risk level",
                    points=False,
                ),
                use_container_width=True,
            )

    with tab2:
        if music.empty:
            st.warning("Music survey data is unavailable.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Survey responses", f"{len(music):,}")
            col2.metric("Reported improvement", f"{(music['music_effects'].eq('Improve').mean()):.0%}")
            col3.metric("Genres represented", f"{music['fav_genre'].nunique()}")

            fav = (
                music["fav_genre"].value_counts()
                .head(10)
                .rename_axis("Favorite genre")
                .reset_index(name="Respondents")
            )
            st.plotly_chart(
                px.bar(
                    fav,
                    x="Respondents",
                    y="Favorite genre",
                    orientation="h",
                    title="Ten most common favorite genres",
                ),
                use_container_width=True,
            )

            effects = music["music_effects"].value_counts().rename_axis("Reported effect").reset_index(name="Respondents")
            st.plotly_chart(
                px.pie(
                    effects,
                    names="Reported effect",
                    values="Respondents",
                    hole=0.55,
                    title="Self-reported effect of music",
                ),
                use_container_width=True,
            )


def about_page() -> None:
    st.markdown('<h1 class="section-title">About the Project</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Mind & Melody was developed as an AI4ALL Group 11C capstone project.</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Project team")
    team = [
        "Nelson Kwesi Xedzro",
        "Estela Acosta Herrera",
        "Zeeshan Khattak",
        "Oyagbemi Oluwatoba",
        "Raheem Kadiku",
        "Sahasra Kothembaka",
    ]
    members = "".join(f'<div class="team-member">{name}</div>' for name in team)
    st.markdown(f'<div class="team-grid">{members}</div>', unsafe_allow_html=True)

    st.subheader("Model approach")
    st.write(
        "The burnout classifier uses a Random Forest pipeline trained on lifestyle, academic, "
        "stress, sleep, activity, and support variables. Direct or derived target columns were "
        "excluded to reduce target leakage."
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Test accuracy", "80.4%")
    metric2.metric("Macro F1", "75.5%")
    metric3.metric("High-risk recall", "79.5%")

    st.subheader("Responsible-use principles")
    st.markdown(
        """
        - The output is an educational pattern estimate, not a diagnosis.
        - User responses remain in the active browser session and are not saved by the app.
        - Self-reported and observational data can contain sampling and measurement bias.
        - Music recommendations are exploratory and may not work the same way for everyone.
        - Human support should take priority when someone feels overwhelmed or unsafe.
        """
    )

    render_disclaimer()


load_css()

with st.sidebar:
    st.markdown("## 🎧 Mind & Melody")
    st.caption("Student wellness through data and music")
    page = st.radio(
        "Navigate",
        ["Home", "Burnout Check-In", "Music Support", "Data Insights", "About & Safety"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("AI4ALL Group 11C · Educational prototype")

if page == "Home":
    home_page()
elif page == "Burnout Check-In":
    burnout_page()
elif page == "Music Support":
    music_page()
elif page == "Data Insights":
    insights_page()
else:
    about_page()
