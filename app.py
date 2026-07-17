from __future__ import annotations

from pathlib import Path
import textwrap

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

RISK_COLORS = {"Low": "#0F766E", "Medium": "#B45309", "High": "#B42318"}
CHART_SEQUENCE = ["#2563EB", "#0F766E", "#B45309", "#0B1F3A", "#0EA5E9", "#B42318"]


st.set_page_config(
    page_title="Mind & Melody",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render presentation HTML safely.

    textwrap.dedent removes the common leading indentation so multiline markup
    can never be parsed as a Markdown indented code block (4+ leading spaces),
    which is what previously caused raw <div>/<h1> tags to display as text.
    """
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


PAGES = ["Home", "Burnout Check-In", "Music Support", "Data Insights", "About & Safety"]

NAV_ITEMS = [
    ("Home", "🏠"),
    ("Burnout Check-In", "🧭"),
    ("Music Support", "🎵"),
    ("Data Insights", "📊"),
    ("About & Safety", "🛡️"),
]


def go_to(page: str) -> None:
    """Set the active page and rerun (native, keyboard-accessible navigation)."""
    st.session_state["page"] = page
    st.rerun()


def render_background() -> None:
    """Fixed, decorative floating orbs behind all content (CSS-only movement)."""
    html(
        """
        <div class="mm-bg" aria-hidden="true">
            <span class="mm-orb mm-orb-a"></span>
            <span class="mm-orb mm-orb-b"></span>
        </div>
        """
    )


def page_header(kicker: str, title: str, subtitle: str) -> None:
    html(
        f"""
        <header class="page-header">
            <span class="kicker">{kicker}</span>
            <h1 class="page-title">{title}</h1>
            <p class="page-subtitle">{subtitle}</p>
        </header>
        """
    )


def apply_chart_theme(fig, *, showlegend: bool = False):
    """Presentation-only theming shared by every chart for a consistent look."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="sans-serif", color="#10233F", size=13),
        title=dict(font=dict(size=16, color="#0B1F3A")),
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=showlegend,
        colorway=CHART_SEQUENCE,
    )
    return fig


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
    html(
        """
        <div class="disclaimer">
        <strong>Important:</strong> Mind &amp; Melody is an educational wellness project.
        It does not diagnose burnout or any mental-health condition, and it is not a
        substitute for professional support. The app does not save your check-in answers.
        </div>
        """
    )


def home_page() -> None:
    # ---- Hero: kicker + headline + lead on the left, animated waveform on the right ----
    hero_left, hero_right = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")

    with hero_left:
        html(
            """
            <div class="mm-hero-text">
                <span class="mm-kicker mm-fade" style="animation-delay:.05s">AI4ALL Group 11C</span>
                <h1 class="mm-hero-title mm-fade" style="animation-delay:.12s">Understand your wellbeing.<br>Find your rhythm.</h1>
                <p class="mm-hero-lead mm-fade" style="animation-delay:.2s">
                    A calm, private space to check in with how you're doing &mdash; then get
                    supportive, data-informed guidance and music built around your rhythm.
                </p>
            </div>
            """
        )
        with st.container(key="mm-hero-cta"):
            cta1, cta2 = st.columns(2)
            with cta1:
                if st.button(
                    "Start Burnout Check-In",
                    key="cta_checkin",
                    type="primary",
                    width="stretch",
                ):
                    go_to("Burnout Check-In")
            with cta2:
                if st.button(
                    "Explore Music Support",
                    key="cta_music",
                    width="stretch",
                ):
                    go_to("Music Support")

    with hero_right:
        html(
            """
            <div class="mm-wave-card mm-fade" style="animation-delay:.34s">
                <span class="mm-wave-label">Your rhythm</span>
                <div class="mm-wave-stage">
                    <svg class="mm-wave-svg mm-wave1" viewBox="0 0 1800 240" preserveAspectRatio="none">
                        <path d="M-360 120 q90 -60 180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0" fill="none" stroke="#2f6fe6" stroke-width="3" opacity=".55"></path>
                    </svg>
                    <svg class="mm-wave-svg mm-wave2" viewBox="0 0 1800 240" preserveAspectRatio="none">
                        <path d="M-360 120 q90 -44 180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0" fill="none" stroke="#7fb0ff" stroke-width="3" opacity=".85"></path>
                    </svg>
                    <svg class="mm-wave-svg mm-wave3" viewBox="0 0 1800 240" preserveAspectRatio="none">
                        <path d="M-360 120 q90 -78 180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0 t180 0" fill="none" stroke="#a9c8ff" stroke-width="1.5" opacity=".35"></path>
                    </svg>
                </div>
            </div>
            """
        )

    # ---- Trust indicators ----
    html(
        """
        <section class="mm-trust mm-fade" style="animation-delay:.4s">
            <div class="mm-trust-card">
                <span class="mm-trust-icon">
                    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.2-2.9 7.4-7 8.5C7.9 18.4 5 15.2 5 11V6z"></path><path d="M9.2 12l2 2 3.6-4"></path></svg>
                </span>
                <div>
                    <div class="mm-trust-title">Private by design</div>
                    <p class="mm-trust-copy">Your check-in responses remain in the current session and are not saved.</p>
                </div>
            </div>
            <div class="mm-trust-card">
                <span class="mm-trust-icon">
                    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5M4 19h16"></path><rect x="7" y="11" width="3" height="5" rx="1" fill="#1a52c6" stroke="none"></rect><rect x="12" y="7" width="3" height="9" rx="1" fill="#1a52c6" stroke="none"></rect><rect x="17" y="13" width="3" height="3" rx="1" fill="#1a52c6" stroke="none"></rect></svg>
                </span>
                <div>
                    <div class="mm-trust-title">Data-informed</div>
                    <p class="mm-trust-copy">Guidance is based on patterns from student wellness and music survey data.</p>
                </div>
            </div>
            <div class="mm-trust-card">
                <span class="mm-trust-icon">
                    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20.5C6.5 16.5 4 13 4 9.5A4 4 0 0 1 12 7a4 4 0 0 1 8 2.5c0 3.5-2.5 7-8 11z"></path></svg>
                </span>
                <div>
                    <div class="mm-trust-title">Student-centered</div>
                    <p class="mm-trust-copy">The experience is supportive, easy to understand, and non-diagnostic.</p>
                </div>
            </div>
        </section>
        """
    )

    # ---- Feature cards (static visual + native navigation button) ----
    html('<h2 class="mm-section-title mm-fade" style="animation-delay:.46s">Explore the experience</h2>')
    features = [
        (
            '<path d="M3 12h4l2 6 4-14 2 8h6"></path>',
            "Assess",
            "Burnout Check-In",
            "Reflect on sleep, stress, academics, activity, and support to receive a pattern-based burnout-risk estimate.",
            "Burnout Check-In",
            "feat_checkin",
        ),
        (
            '<path d="M9 18V6l10-2v12"></path><circle cx="6" cy="18" r="3"></circle><circle cx="16" cy="16" r="3"></circle>',
            "Support",
            "Music Support",
            "Choose your listening goal and favorite genre to explore personalized, data-informed music suggestions.",
            "Music Support",
            "feat_music",
        ),
        (
            '<path d="M21 21H4a1 1 0 0 1-1-1V3"></path><path d="M7 15l4-5 3 3 5-7"></path>',
            "Explore",
            "Data Insights",
            "Explore interactive visualizations showing patterns across the student burnout and music survey datasets.",
            "Data Insights",
            "feat_data",
        ),
    ]
    feature_cols = st.columns(3, gap="medium")
    for column, (icon, tag, title, copy, target, key) in zip(feature_cols, features):
        with column:
            html(
                f"""
                <div class="mm-feature mm-fade" style="animation-delay:.5s">
                    <span class="mm-feature-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1f5fe0" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>
                    </span>
                    <span class="mm-feature-tag">{tag}</span>
                    <h3 class="mm-feature-title">{title}</h3>
                    <p class="mm-feature-copy">{copy}</p>
                </div>
                """
            )
            if st.button(f"Open {title}", key=key, width="stretch"):
                go_to(target)

    # ---- How it works ----
    html(
        """
        <section class="mm-how mm-fade" style="animation-delay:.54s">
            <h2 class="mm-how-title">How it works</h2>
            <p class="mm-how-sub">Three calm steps &mdash; no account, no pressure.</p>
            <div class="mm-steps">
                <span class="mm-steps-line"></span>
                <div class="mm-step">
                    <span class="mm-step-num">1</span>
                    <div class="mm-step-title">Check in privately</div>
                    <p class="mm-step-copy">Answer a short set of lifestyle and academic questions. No account is required.</p>
                </div>
                <div class="mm-step">
                    <span class="mm-step-num">2</span>
                    <div class="mm-step-title">Review the estimate</div>
                    <p class="mm-step-copy">See a pattern-based burnout-risk category with the model's confidence.</p>
                </div>
                <div class="mm-step">
                    <span class="mm-step-num">3</span>
                    <div class="mm-step-title">Explore next steps</div>
                    <p class="mm-step-copy">Get supportive suggestions and data-informed music ideas to try.</p>
                </div>
            </div>
        </section>
        """
    )

    # ---- Disclaimer ----
    html(
        """
        <section class="mm-disclaimer mm-fade" style="animation-delay:.58s">
            <span class="mm-disclaimer-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v5"></path><circle cx="12" cy="16.5" r=".6" fill="#1a52c6"></circle></svg>
            </span>
            <p>
                <strong>Important:</strong> Mind &amp; Melody is an educational wellness project. It does not
                diagnose burnout or any mental-health condition, and it is not a substitute for professional
                support. If you are in crisis, please contact a licensed professional or a local crisis line.
                The app does not save your check-in answers.
            </p>
        </section>
        <div class="mm-home-foot">AI4ALL Group 11C &middot; Educational prototype</div>
        """
    )


# Session-state keys used only by the Burnout Check-In (nothing is persisted to
# disk, logs, or any external service — everything lives in st.session_state and
# is cleared by _reset_burnout_state()).
BURNOUT_RESULT_KEY = "burnout_result"
BURNOUT_WIDGET_PREFIX = "bo_"


def _reset_burnout_state() -> None:
    """Clear every burnout-related session-state key -> return to a blank form."""
    for key in list(st.session_state.keys()):
        if key == BURNOUT_RESULT_KEY or key.startswith(BURNOUT_WIDGET_PREFIX):
            del st.session_state[key]


def _burnout_section(num: str, title: str, helper: str) -> None:
    """Static, numbered section header rendered inside a white form card."""
    html(
        f"""
        <div class="mm-form-head">
            <span class="mm-form-num">{num}</span>
            <div class="mm-form-head-text">
                <h3>{title}</h3>
                <p>{helper}</p>
            </div>
        </div>
        """
    )


def _render_burnout_result(result: dict) -> None:
    """Polished, non-diagnostic result view built entirely from session state."""
    predicted = result["predicted"]
    confidence = result["confidence"]
    classes = result["classes"]
    probabilities = result["probabilities"]
    tips = result["tips"]

    level_class = {
        "Low": "mm-result-low",
        "Medium": "mm-result-medium",
        "High": "mm-result-high",
    }.get(predicted, "")
    level_word = {"Low": "lower", "Medium": "moderate", "High": "higher"}.get(
        predicted, predicted.lower()
    )

    html(
        f"""
        <section class="mm-result {level_class} mm-fade" style="animation-delay:.05s">
            <span class="mm-result-kicker">Pattern-Based Burnout Risk Estimate</span>
            <div class="mm-result-head">
                <span class="mm-result-badge">{predicted}</span>
                <div class="mm-result-headtext">
                    <h2>{predicted} burnout-risk pattern</h2>
                    <p>
                        Based on the lifestyle and academic inputs you entered, your responses
                        most resemble a <strong>{level_word}</strong> burnout-risk pattern in the
                        training data. This is a comparison with patterns &mdash; not a statement
                        about your actual health.
                    </p>
                </div>
            </div>
            <div class="mm-conf">
                <div class="mm-conf-row">
                    <span>Model confidence for this category</span>
                    <strong>{confidence:.0%}</strong>
                </div>
                <div class="mm-conf-track"><span class="mm-conf-fill" style="width:{confidence:.0%}"></span></div>
                <p class="mm-conf-note">
                    Confidence reflects how strongly the model leans toward this category &mdash;
                    it is not a diagnosis or a measure of certainty about how you are doing.
                </p>
            </div>
        </section>
        """
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
        color="Risk level",
        color_discrete_map=RISK_COLORS,
    )
    chart.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(apply_chart_theme(chart), width="stretch")

    tips_html = "".join(f"<li>{tip}</li>" for tip in tips)
    html(
        f"""
        <section class="mm-steps-card mm-fade" style="animation-delay:.12s">
            <h3 class="mm-steps-card-title">Supportive next steps</h3>
            <ul class="mm-tip-list">{tips_html}</ul>
        </section>
        """
    )

    if predicted == "High":
        st.info(
            "Consider sharing how you are feeling with a trusted adult, counselor, "
            "health professional, or campus support person. If you feel unsafe or need "
            "urgent help, contact local emergency services."
        )

    html(
        """
        <section class="mm-disclaimer mm-fade" style="animation-delay:.18s">
            <span class="mm-disclaimer-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v5"></path><circle cx="12" cy="16.5" r=".6" fill="#1a52c6"></circle></svg>
            </span>
            <p>
                <strong>Educational estimate, not medical advice.</strong> Mind &amp; Melody is an
                educational wellness project. This result is a pattern-based estimate from the numbers
                you entered &mdash; it does not diagnose burnout or any condition, does not know your
                actual health, and is not a substitute for professional support. If you are in crisis,
                please contact a licensed professional or a local crisis line. Your answers stay in this
                browser session and are not saved.
            </p>
        </section>
        """
    )


def burnout_page() -> None:
    page_header(
        "Check-In",
        "Burnout Check-In",
        "Answer based on your recent routine. The model uses patterns from a cleaned student burnout dataset.",
    )

    model = load_model()
    if model is None:
        st.error("The model file is missing. Add models/burnout_model.joblib and reload the app.")
        return

    # Persisted result view — stays visible across reruns until the user resets.
    if st.session_state.get(BURNOUT_RESULT_KEY):
        _render_burnout_result(st.session_state[BURNOUT_RESULT_KEY])
        if st.button("Complete Another Check-In", key="bo_reset", type="primary"):
            _reset_burnout_state()
            st.rerun()
        return

    with st.form("burnout_checkin", border=False):
        with st.container(key="mmform_profile"):
            _burnout_section("1", "Student profile", "A little about you and where you are academically.")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 17, 29, 20, key="bo_age")
            with c2:
                gender = st.selectbox("Gender category used by the dataset", ["Female", "Male", "Other"], key="bo_gender")
            with c3:
                academic_year = st.selectbox("Academic year", [1, 2, 3, 4], index=1, key="bo_academic_year")

        with st.container(key="mmform_academic"):
            _burnout_section("2", "Academic experience", "Your recent study load and academic pressure.")
            c1, c2, c3 = st.columns(3)
            with c1:
                study_hours = st.slider("Study hours per day", 0.0, 14.0, 5.0, 0.5, key="bo_study_hours")
            with c2:
                exam_pressure = st.slider("Exam pressure", 1.0, 10.0, 6.0, 0.5, key="bo_exam_pressure")
            with c3:
                academic_performance = st.slider("Academic performance score", 42.0, 98.0, 71.0, 1.0, key="bo_academic_performance")

        with st.container(key="mmform_emotional"):
            _burnout_section("3", "Emotional wellbeing", "Self-ratings for how you have been feeling lately.")
            c1, c2, c3 = st.columns(3)
            with c1:
                stress_level = st.slider("Stress level", 0.0, 10.0, 5.0, 0.5, key="bo_stress_level")
            with c2:
                anxiety_score = st.slider("Anxiety self-rating", 0.0, 10.0, 3.0, 0.5, key="bo_anxiety_score")
            with c3:
                depression_score = st.slider("Low-mood self-rating", 0.0, 9.0, 2.0, 0.5, key="bo_depression_score")

        with st.container(key="mmform_lifestyle"):
            _burnout_section("4", "Daily lifestyle", "Sleep, movement, and screen habits on a typical day.")
            c1, c2 = st.columns(2)
            with c1:
                sleep_hours = st.slider("Sleep hours per night", 3.0, 10.0, 7.0, 0.5, key="bo_sleep_hours")
                screen_time = st.slider("Screen time per day", 1.0, 12.0, 5.0, 0.5, key="bo_screen_time")
            with c2:
                physical_activity = st.slider("Physical activity hours per day", 0.0, 7.0, 2.0, 0.5, key="bo_physical_activity")
                internet_usage = st.slider("Internet use per day", 1.0, 14.0, 5.0, 0.5, key="bo_internet_usage")

        with st.container(key="mmform_support"):
            _burnout_section("5", "Support & environment", "The support around you and any outside pressures.")
            c1, c2, c3 = st.columns(3)
            with c1:
                social_support = st.slider("Social support", 0.0, 10.0, 6.0, 0.5, key="bo_social_support")
            with c2:
                financial_stress = st.slider("Financial stress", 0.0, 10.0, 4.0, 0.5, key="bo_financial_stress")
            with c3:
                family_expectation = st.slider("Family expectation pressure", 0.0, 10.0, 6.0, 0.5, key="bo_family_expectation")

        submitted = st.form_submit_button("View My Check-In Result", type="primary", width="stretch")

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

    # Store the whole check-in outcome in session state only (no files/logs/network).
    st.session_state[BURNOUT_RESULT_KEY] = {
        "row": row,
        "predicted": predicted_risk,
        "classes": classes,
        "probabilities": [float(p) for p in probabilities],
        "confidence": confidence,
        "tips": wellness_tips(row),
    }
    st.rerun()


# Session-state keys used only by Music Support (session-only; nothing is written
# to disk, logs, or any external service — cleared by _reset_music_state()).
MUSIC_RESULT_KEY = "music_result"
MUSIC_WIDGET_PREFIX = "mu_"


def _reset_music_state() -> None:
    """Clear every Music Support session-state key -> return to default selections."""
    for key in list(st.session_state.keys()):
        if key == MUSIC_RESULT_KEY or key.startswith(MUSIC_WIDGET_PREFIX):
            del st.session_state[key]


def _music_section(num: str, title: str, helper: str) -> None:
    """Static, numbered step header rendered inside a white selection card."""
    html(
        f"""
        <div class="mm-form-head">
            <span class="mm-form-num">{num}</span>
            <div class="mm-form-head-text">
                <h3>{title}</h3>
                <p>{helper}</p>
            </div>
        </div>
        """
    )


def _render_music_result(result: dict) -> None:
    """Polished 'Your Music Support Plan' view built entirely from session state."""
    genres = result["genres"]
    sample_size = result["sample_size"]

    html(
        """
        <section class="mm-result mm-result-music mm-fade" style="animation-delay:.05s">
            <span class="mm-result-kicker">Your Music Support Plan</span>
            <div class="mm-result-headtext">
                <h2>A data-informed place to start listening</h2>
                <p>
                    These genres come from patterns in the survey data. Treat them as an exploratory
                    starting point &mdash; keep what feels good and skip what doesn't.
                </p>
            </div>
        </section>
        """
    )

    # Equal-height genre cards, each with a CSS-only animated sound-wave.
    bars = "".join(f'<span style="animation-delay:{i * 0.12:.2f}s"></span>' for i in range(7))
    genre_cards = "".join(
        f"""
        <div class="mm-genre-card">
            <span class="mm-genre-rank">{position}</span>
            <span class="mm-feature-tag">Genre</span>
            <h3 class="mm-genre-name">{genre}</h3>
            <div class="mm-wavebars" aria-hidden="true">{bars}</div>
            <p class="mm-genre-copy">Add a few familiar tracks first, then explore gradually.</p>
        </div>
        """
        for position, genre in enumerate(genres, start=1)
    )
    html(
        f"""
        <section class="mm-genre-grid mm-fade" style="animation-delay:.12s">
            {genre_cards}
        </section>
        """
    )

    # Guidance cards — session + approach, verbatim from recommend_genres().
    html(
        f"""
        <section class="mm-guidance-grid mm-fade" style="animation-delay:.18s">
            <div class="mm-guidance-card">
                <span class="mm-guidance-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1f5fe0" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4l3 2"></path></svg>
                </span>
                <h3 class="mm-guidance-title">Your listening session</h3>
                <p class="mm-guidance-copy">{result["session"]}</p>
            </div>
            <div class="mm-guidance-card">
                <span class="mm-guidance-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1f5fe0" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V6l10-2v12"></path><circle cx="6" cy="18" r="3"></circle><circle cx="16" cy="16" r="3"></circle></svg>
                </span>
                <h3 class="mm-guidance-title">How to listen</h3>
                <p class="mm-guidance-copy">{result["approach"]}</p>
            </div>
        </section>
        """
    )

    html(
        f"""
        <section class="mm-sample mm-fade" style="animation-delay:.24s">
            <span class="mm-sample-figure">{sample_size:,}</span>
            <p class="mm-sample-copy">
                relevant survey responses informed this ranking &mdash; from participants who reported
                that music improved their wellbeing.
            </p>
        </section>
        """
    )

    html(
        """
        <section class="mm-disclaimer mm-fade" style="animation-delay:.3s">
            <span class="mm-disclaimer-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v5"></path><circle cx="12" cy="16.5" r=".6" fill="#1a52c6"></circle></svg>
            </span>
            <p>
                Music affects people differently. Stop or switch tracks when listening feels distracting,
                uncomfortable, or unhelpful. These suggestions are exploratory and do not treat or cure any
                mental-health condition.
            </p>
        </section>
        """
    )
    render_disclaimer()


def music_page() -> None:
    page_header(
        "Music",
        "Music Support",
        "Discover a small, data-informed starting point for your next playlist.",
    )

    music = load_music_data()
    if music.empty:
        st.error("The music survey file is missing from data/music_survey_cleaned.csv.")
        return

    # Persisted result view — stays visible across reruns and navigation until reset.
    if st.session_state.get(MUSIC_RESULT_KEY):
        _render_music_result(st.session_state[MUSIC_RESULT_KEY])
        if st.button("Create Another Suggestion", key="mu_reset", type="primary"):
            _reset_music_state()
            st.rerun()
        return

    html(
        """
        <p class="mm-intro mm-fade" style="animation-delay:.04s">
            Recommendations are <strong>exploratory and data-informed</strong> &mdash; drawn from patterns
            in a student music-and-wellbeing survey. Follow the three steps below to build a starting point
            for your next playlist.
        </p>
        """
    )

    genres = sorted([genre for genre in GENRE_COLUMN_MAP if genre in set(music["fav_genre"].dropna())])

    with st.container(key="mmform_genre"):
        _music_section("1", "Choose your favorite genre", "We'll anchor your suggestions around what you already enjoy.")
        favorite_genre = st.selectbox(
            "Your favorite genre",
            genres,
            index=genres.index("Pop") if "Pop" in genres else 0,
            key="mu_genre",
        )

    with st.container(key="mmform_goal"):
        _music_section("2", "Select your listening goal", "What would you like this listening session to support?")
        goal = st.selectbox(
            "What would you like the playlist to support?",
            [
                "Calm down",
                "Focus",
                "Lift my mood",
                "Prepare for sleep",
            ],
            key="mu_goal",
        )

    with st.container(key="mmform_generate"):
        _music_section("3", "Generate your music suggestion", "Create a small, data-informed plan you can try today.")
        generate = st.button("Create My Music Suggestion", type="primary", width="stretch")

    if generate:
        result = recommend_genres(music, favorite_genre, goal)
        # Store selections + generated recommendation in session state only.
        st.session_state[MUSIC_RESULT_KEY] = {
            "genres": result["genres"],
            "sample_size": result["sample_size"],
            "session": result["session"],
            "approach": result["approach"],
            "favorite_genre": favorite_genre,
            "goal": goal,
        }
        st.rerun()

    render_disclaimer()


def _viz_note(text: str) -> None:
    """Short, causation-free sentence describing what a chart shows."""
    html(f'<p class="mm-viz-note">{text}</p>')


def insights_page() -> None:
    page_header(
        "Insights",
        "Data Insights",
        "Explore high-level patterns from the two project datasets.",
    )

    html(
        """
        <p class="mm-intro mm-fade" style="animation-delay:.04s">
            These charts show <strong>high-level patterns</strong> in the two project datasets &mdash;
            associations and distributions, <strong>not causation or medical conclusions</strong>. The
            survey and dashboard samples are not universally representative of all students, so read the
            visuals as exploratory context rather than definitive findings.
        </p>
        """
    )

    burnout = load_burnout_sample()
    music = load_music_data()

    tab1, tab2 = st.tabs(["Student burnout", "Music and wellbeing"])

    with tab1:
        if burnout.empty:
            st.warning("Burnout dashboard sample is unavailable.")
        else:
            with st.container(key="mmviz_burnout_metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Dashboard sample", f"{len(burnout):,}")
                col2.metric("Average sleep", f"{burnout['sleep_hours'].mean():.1f} hrs")
                col3.metric("Average stress", f"{burnout['stress_level'].mean():.1f}/10")

            risk_counts = burnout["risk_level"].value_counts().rename_axis("Risk level").reset_index(name="Students")
            with st.container(key="mmviz_risk_dist"):
                _viz_note("How the dashboard sample is split across the low, medium, and high burnout-risk categories.")
                st.plotly_chart(
                    apply_chart_theme(
                        px.bar(
                            risk_counts,
                            x="Risk level",
                            y="Students",
                            title="Burnout-risk distribution",
                            color="Risk level",
                            color_discrete_map=RISK_COLORS,
                        )
                    ),
                    width="stretch",
                )

            sleep_by_risk = (
                burnout.groupby("risk_level", as_index=False)["sleep_hours"]
                .mean()
                .rename(columns={"sleep_hours": "Average sleep hours"})
            )
            with st.container(key="mmviz_sleep_by_risk"):
                _viz_note("Average reported sleep hours within each risk group &mdash; a comparison of group averages, not a cause-and-effect link.")
                st.plotly_chart(
                    apply_chart_theme(
                        px.bar(
                            sleep_by_risk,
                            x="risk_level",
                            y="Average sleep hours",
                            title="Average sleep hours by burnout-risk level",
                            color="risk_level",
                            color_discrete_map=RISK_COLORS,
                        )
                    ),
                    width="stretch",
                )

            with st.container(key="mmviz_stress_box"):
                _viz_note("How reported stress levels are distributed within each risk group, including the spread and typical range.")
                st.plotly_chart(
                    apply_chart_theme(
                        px.box(
                            burnout.sample(min(8000, len(burnout)), random_state=42),
                            x="risk_level",
                            y="stress_level",
                            title="Stress distribution by burnout-risk level",
                            points=False,
                            color="risk_level",
                            color_discrete_map=RISK_COLORS,
                        )
                    ),
                    width="stretch",
                )

    with tab2:
        if music.empty:
            st.warning("Music survey data is unavailable.")
        else:
            with st.container(key="mmviz_music_metrics"):
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
            with st.container(key="mmviz_fav_genres"):
                _viz_note("The ten favorite genres reported most often by survey respondents, ordered by number of respondents.")
                st.plotly_chart(
                    apply_chart_theme(
                        px.bar(
                            fav,
                            x="Respondents",
                            y="Favorite genre",
                            orientation="h",
                            title="Ten most common favorite genres",
                        )
                    ),
                    width="stretch",
                )

            effects = music["music_effects"].value_counts().rename_axis("Reported effect").reset_index(name="Respondents")
            with st.container(key="mmviz_effects"):
                _viz_note("How respondents described music's self-reported effect on them &mdash; these are self-reports, not measured outcomes.")
                st.plotly_chart(
                    apply_chart_theme(
                        px.pie(
                            effects,
                            names="Reported effect",
                            values="Respondents",
                            hole=0.55,
                            title="Self-reported effect of music",
                        ),
                        showlegend=True,
                    ),
                    width="stretch",
                )


def about_page() -> None:
    page_header(
        "About",
        "About Mind &amp; Melody",
        "An AI4ALL Group 11C student wellness project &mdash; a calm, non-clinical space to reflect, explore, and learn.",
    )

    # 1 + 2 — Intro + mission
    html(
        """
        <p class="mm-intro mm-fade" style="animation-delay:.04s">
            Mind &amp; Melody is a <strong>student-built, educational wellness project</strong> from AI4ALL
            Group 11C. It is designed to be supportive and easy to understand &mdash; not a clinical tool.
        </p>
        <section class="mm-mission mm-fade" style="animation-delay:.1s">
            <span class="mm-about-kicker">Our purpose</span>
            <h2 class="mm-mission-title">Reflect, explore, and learn &mdash; in one calm space</h2>
            <p>
                The four parts of the app work together: the <strong>burnout check-in</strong> gives a
                pattern-based estimate from your inputs, the <strong>wellness guidance</strong> offers gentle
                next steps, <strong>music support</strong> suggests exploratory, data-informed listening ideas,
                and <strong>data insights</strong> show high-level patterns from the project datasets. Together
                they encourage reflection and learning. The app does <strong>not diagnose, treat, prevent, or
                monitor</strong> burnout or any mental-health condition.
            </p>
        </section>
        """
    )

    # 3 — Team (names preserved exactly; initials-only avatars, no roles)
    team = [
        "Nelson Kwesi Xedzro",
        "Estela Acosta Herrera",
        "Zeeshan Khattak",
        "Oyagbemi Oluwatoba",
        "Raheem Kadiku",
        "Sahasra Kothembaka",
    ]

    def _initials(name: str) -> str:
        parts = [p for p in name.split() if p]
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    member_cards = "".join(
        f"""
        <div class="mm-team-card">
            <span class="mm-team-avatar" aria-hidden="true">{_initials(name)}</span>
            <span class="mm-team-name">{name}</span>
        </div>
        """
        for name in team
    )
    html(
        f"""
        <h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">Project team</h2>
        <section class="mm-team-grid mm-fade" style="animation-delay:.1s">
            {member_cards}
        </section>
        """
    )

    # 4 — How the model works
    html(
        """
        <h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">How the model works</h2>
        <section class="mm-model-card mm-fade" style="animation-delay:.1s">
            <p>
                The burnout check-in uses a <strong>Random Forest</strong> pipeline &mdash; a model that
                combines many decision trees and looks for patterns across lifestyle, academic, stress, sleep,
                activity, and support inputs. From those patterns it places a response into one of three
                pattern-based categories: <strong>Low</strong>, <strong>Medium</strong>, or <strong>High</strong>.
            </p>
            <p>
                Columns that directly or indirectly revealed the answer were <strong>excluded to reduce target
                leakage</strong>, so the estimate reflects general patterns rather than a hidden shortcut. These
                categories are <strong>educational pattern estimates, not diagnoses</strong>, and the model has
                not undergone any clinical validation.
            </p>
        </section>
        """
    )

    # 5 — Model performance (exact metrics preserved)
    html('<h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">Model performance</h2>')
    with st.container(key="mmabout_metrics"):
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Test accuracy", "80.4%")
        metric2.metric("Macro F1", "75.5%")
        metric3.metric("High-risk recall", "79.5%")
    html(
        """
        <p class="mm-about-note mm-fade" style="animation-delay:.12s">
            These results were measured on the <strong>project dataset only</strong> and describe how the model
            performed there. They are <strong>not a clinical validation</strong> and do not indicate accuracy for
            any individual person.
        </p>
        """
    )

    # 6 — Data and limitations
    html(
        """
        <h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">Data and limitations</h2>
        <section class="mm-limit-grid mm-fade" style="animation-delay:.1s">
            <div class="mm-limit-card">
                <span class="mm-limit-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="6" rx="8" ry="3"></ellipse><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"></path><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"></path></svg>
                </span>
                <h3 class="mm-limit-title">Student burnout dataset</h3>
                <p class="mm-limit-copy">Used to train the pattern-based burnout estimate across lifestyle, academic, stress, sleep, activity, and support variables.</p>
            </div>
            <div class="mm-limit-card">
                <span class="mm-limit-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V6l10-2v12"></path><circle cx="6" cy="18" r="3"></circle><circle cx="16" cy="16" r="3"></circle></svg>
                </span>
                <h3 class="mm-limit-title">Music &amp; mental-health survey</h3>
                <p class="mm-limit-copy">Informs the exploratory music suggestions and the wellbeing patterns shown in Data Insights.</p>
            </div>
            <div class="mm-limit-card">
                <span class="mm-limit-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4"></path><circle cx="12" cy="12" r="4"></circle></svg>
                </span>
                <h3 class="mm-limit-title">Sampling &amp; self-report limits</h3>
                <p class="mm-limit-copy">Self-reported data can carry sampling and measurement bias, and answers reflect how people described themselves at one point in time.</p>
            </div>
            <div class="mm-limit-card">
                <span class="mm-limit-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M3 12h18M12 3c2.5 2.7 2.5 15.3 0 18M12 3c-2.5 2.7-2.5 15.3 0 18"></path></svg>
                </span>
                <h3 class="mm-limit-title">Cultural &amp; demographic limits</h3>
                <p class="mm-limit-copy">The data may carry cultural, demographic, and representation bias. It does <strong>not</strong> represent every student population.</p>
            </div>
        </section>
        """
    )

    # 7 — Privacy and responsible use
    html(
        """
        <h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">Privacy and responsible use</h2>
        <section class="mm-model-card mm-fade" style="animation-delay:.1s">
            <p>
                Your check-in inputs and generated recommendations stay <strong>only in the active Streamlit
                browser session</strong>. The app does <strong>not save your responses to files or databases</strong>,
                and nothing is sent to external services. Because results live in session state,
                <strong>navigating between pages or refreshing may clear or reset them</strong>.
            </p>
            <ul class="mm-about-list">
                <li>The output is an educational pattern estimate, not a diagnosis.</li>
                <li>Self-reported data may contain sampling, cultural, measurement, and representation bias.</li>
                <li>Music recommendations are exploratory and may affect people differently.</li>
                <li>Human support should take priority when someone feels overwhelmed or unsafe.</li>
            </ul>
        </section>
        """
    )

    # 8 — Support
    html(
        """
        <h2 class="mm-about-section-title mm-fade" style="animation-delay:.05s">If you need support</h2>
        <section class="mm-support-card mm-fade" style="animation-delay:.1s">
            <span class="mm-support-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1a52c6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20.5C6.5 16.5 4 13 4 9.5A4 4 0 0 1 12 7a4 4 0 0 1 8 2.5c0 3.5-2.5 7-8 11z"></path></svg>
            </span>
            <div class="mm-support-body">
                <h3 class="mm-support-title">This app is not an emergency or crisis service</h3>
                <p>
                    If you feel overwhelmed or unsafe, please reach out to a trusted adult, a counselor, a
                    healthcare professional, a campus support person, or local emergency services. Talking with
                    a real person should always come first.
                </p>
            </div>
        </section>
        """
    )

    render_disclaimer()


load_css()
render_background()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

with st.sidebar:
    html(
        """
        <div class="mm-brand">
            <span class="mm-brand-badge">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#a9c8ff" stroke-width="1.7" stroke-linecap="round"><path d="M4 13v-1a8 8 0 0 1 16 0v1"></path><rect x="3" y="13" width="4" height="7" rx="2" fill="#a9c8ff" stroke="none"></rect><rect x="17" y="13" width="4" height="7" rx="2" fill="#a9c8ff" stroke="none"></rect></svg>
            </span>
            <span class="mm-brand-text">
                <span class="mm-brand-name">Mind &amp; Melody</span>
                <span class="mm-brand-tagline">Student wellness through data and music</span>
            </span>
        </div>
        <p class="mm-nav-label">Navigate</p>
        """
    )
    with st.container(key="mm-nav"):
        for label, icon in NAV_ITEMS:
            active = st.session_state["page"] == label
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                width="stretch",
                type="primary" if active else "secondary",
            ):
                go_to(label)
    html('<div class="mm-side-foot">AI4ALL Group 11C &middot; Educational prototype</div>')

page = st.session_state["page"]

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
