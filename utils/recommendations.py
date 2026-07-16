from __future__ import annotations

from typing import Any
import pandas as pd


GENRE_COLUMN_MAP = {
    "Classical": "frequency_classical_score",
    "Country": "frequency_country_score",
    "EDM": "frequency_edm_score",
    "Folk": "frequency_folk_score",
    "Gospel": "frequency_gospel_score",
    "Hip hop": "frequency_hip_hop_score",
    "Jazz": "frequency_jazz_score",
    "K pop": "frequency_k_pop_score",
    "Latin": "frequency_latin_score",
    "Lofi": "frequency_lofi_score",
    "Metal": "frequency_metal_score",
    "Pop": "frequency_pop_score",
    "R&B": "frequency_randb_score",
    "Rap": "frequency_rap_score",
    "Rock": "frequency_rock_score",
    "Video game music": "frequency_video_game_music_score",
}


GOAL_GUIDANCE = {
    "Calm down": {
        "session": "Try a 15–20 minute listening break at a comfortable volume.",
        "approach": "Choose steady, familiar tracks and reduce other distractions."
    },
    "Focus": {
        "session": "Try one focused 25–40 minute listening session.",
        "approach": "Instrumental or low-lyric music may be less distracting while studying."
    },
    "Lift my mood": {
        "session": "Build a short 10–20 minute playlist you already associate with positive moments.",
        "approach": "Familiar, energizing tracks can make the playlist feel more personal."
    },
    "Prepare for sleep": {
        "session": "Start a softer playlist 20–30 minutes before bedtime.",
        "approach": "Keep the volume low and avoid switching between many tracks."
    },
}


def recommend_genres(
    music_df: pd.DataFrame,
    favorite_genre: str,
    goal: str,
    top_n: int = 3,
) -> dict[str, Any]:
    improved = music_df[music_df["music_effects"].eq("Improve")].copy()

    similar = improved[improved["fav_genre"].eq(favorite_genre)]
    if len(similar) < 8:
        similar = improved

    available_columns = {
        genre: column
        for genre, column in GENRE_COLUMN_MAP.items()
        if column in similar.columns
    }

    scores = {
        genre: float(similar[column].mean())
        for genre, column in available_columns.items()
    }

    ranked = sorted(scores, key=scores.get, reverse=True)
    selected = []

    if favorite_genre in scores:
        selected.append(favorite_genre)

    for genre in ranked:
        if genre not in selected:
            selected.append(genre)
        if len(selected) >= top_n:
            break

    guidance = GOAL_GUIDANCE.get(goal, GOAL_GUIDANCE["Calm down"])

    return {
        "genres": selected[:top_n],
        "sample_size": int(len(similar)),
        "session": guidance["session"],
        "approach": guidance["approach"],
    }


def wellness_tips(inputs: dict[str, float]) -> list[str]:
    tips: list[str] = []

    if inputs["sleep_hours"] < 6:
        tips.append("Protect a consistent sleep window tonight and reduce late-night screen use.")

    if inputs["stress_level"] >= 7 or inputs["exam_pressure"] >= 8:
        tips.append("Break the next task into one small step and schedule a brief reset between study blocks.")

    if inputs["screen_time"] >= 8:
        tips.append("Try one screen-free break and move notifications out of sight while studying.")

    if inputs["social_support"] <= 3:
        tips.append("Consider checking in with a trusted friend, family member, mentor, or campus support person.")

    if inputs["physical_activity"] < 1:
        tips.append("A short walk or gentle movement break may help you reset before returning to work.")

    if inputs["financial_stress"] >= 8:
        tips.append("Write down the immediate concern and identify one trusted campus or community resource to contact.")

    if not tips:
        tips.append("Keep the routines that are working and check in again when your schedule changes.")

    return tips[:4]
