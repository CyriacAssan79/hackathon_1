from pathlib import Path
import ast

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
USERS_PATH = BASE_DIR / "users.csv"
FILMS_PATH = BASE_DIR / "films.csv"
WATCH_HISTORY_PATH = BASE_DIR / "watch_history.csv"


st.set_page_config(
    page_title="Recommandation de films",
    page_icon=":movie_camera:",
    layout="wide",
)


def parse_preferences(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text.startswith("["):
        return ast.literal_eval(text)

    return [genre for genre in text.split("|") if genre]


@st.cache_data
def load_data():
    df_user = pd.read_csv(USERS_PATH)
    df_film = pd.read_csv(FILMS_PATH)
    df_watch_history = pd.read_csv(WATCH_HISTORY_PATH)

    df_user["preference"] = df_user["preference"].apply(parse_preferences)
    df_watch_history["date"] = pd.to_datetime(df_watch_history["date"])
    df_watch_history["year"] = df_watch_history["date"].dt.year
    df_watch_history["month"] = df_watch_history["date"].dt.month

    return df_user, df_film, df_watch_history


def preferences_films(df_film, preferences):
    return df_film[df_film["genre"].isin(preferences)].copy()


def ranking(df_film):
    return df_film.sort_values("rating", ascending=False).copy()


def similarity(genre, preferences):
    if genre in preferences:
        return 1
    return 0


def ranking_similarity(df_film, preferences, top_n=10):
    df_film_similarity = df_film.copy()
    df_film_similarity["score_similarite"] = df_film_similarity["genre"].apply(
        lambda genre: similarity(genre, preferences)
    )

    return df_film_similarity.sort_values(
        by=["score_similarite", "rating", "year"],
        ascending=[False, False, False],
    ).head(top_n)


def get_user_context(df_user, df_watch_history, user_id):
    user = df_user[df_user["user_id"] == user_id].iloc[0]
    user_watch_history = df_watch_history[
        df_watch_history["user_id"] == user_id
    ].sort_values("date")
    return user, user_watch_history


def get_last_seen_title(user_watch_history):
    if user_watch_history.empty:
        return None
    return user_watch_history.iloc[-1]["movie_title"]


def clean_table(df):
    return df.rename(
        columns={
            "movie_id": "ID film",
            "title": "Film",
            "genre": "Genre",
            "year": "Annee",
            "duration": "Duree",
            "rating": "Note",
            "score_similarite": "Score similarite",
            "date": "Date",
            "movie_title": "Film",
            "rate": "Note utilisateur",
        }
    )


df_user, df_film, df_watch_history = load_data()

st.title("Moteur de recommandation de films")

with st.sidebar:
    st.header("Parametres")

    user_labels = df_user.apply(
        lambda row: f"{row['user_id']} - {row['name']}", axis=1
    ).tolist()
    selected_user_label = st.selectbox("Utilisateur", user_labels)
    selected_user_id = int(selected_user_label.split(" - ")[0])

    top_n = st.slider("Nombre de films a afficher", 5, 30, 10)
    selected_genres = st.multiselect(
        "Genres disponibles",
        sorted(df_film["genre"].unique()),
        default=sorted(df_film["genre"].unique()),
    )

filtered_films = df_film[df_film["genre"].isin(selected_genres)].copy()
user, user_watch_history = get_user_context(
    df_user,
    df_watch_history,
    selected_user_id,
)
preferences = user["preference"]
last_seen = get_last_seen_title(user_watch_history)

all_pref = preferences_films(filtered_films, preferences)
df_rating = ranking(filtered_films)
df_similarity = ranking_similarity(filtered_films, preferences, top_n=top_n)

suggestion_similarity = (
    df_similarity.iloc[0]["title"] if not df_similarity.empty else "Aucune suggestion"
)
suggestion_note = (
    df_rating.iloc[0]["title"] if not df_rating.empty else "Aucune suggestion"
)
suggestion_all_pref = (
    all_pref.sort_values("rating", ascending=False).iloc[0]["title"]
    if not all_pref.empty
    else "Aucune suggestion"
)

metric_cols = st.columns(4)
metric_cols[0].metric("Utilisateurs", len(df_user))
metric_cols[1].metric("Films", len(df_film))
metric_cols[2].metric("Historiques", len(df_watch_history))
metric_cols[3].metric("Genres", df_film["genre"].nunique())

st.divider()

profile_col, suggestion_col = st.columns([1, 2])

with profile_col:
    st.subheader("Profil utilisateur")
    st.write(f"**Nom :** {user['name']}")
    st.write(f"**Email :** {user['email']}")
    st.write(f"**Age :** {user['age']} ans")
    st.write("**Preferences :** " + ", ".join(preferences))

with suggestion_col:
    st.subheader("Suggestions personnalisees")

    if last_seen is None:
        st.info("Cet utilisateur n'a pas encore d'historique.")
    else:
        st.write(f"**Dernier film regarde :** {last_seen}")
        st.success(
            f"{user['name']}, apres {last_seen}, vous pourriez aimer "
            f"{suggestion_similarity} selon votre genre."
        )
        st.info(
            f"Selon les meilleures notes, vous pourriez aimer {suggestion_note}."
        )
        st.warning(
            f"Selon vos preferences declarees, vous pourriez aimer "
            f"{suggestion_all_pref}."
        )

st.divider()

tab_pref, tab_similarity, tab_rating, tab_history, tab_stats, tab_data = st.tabs(
    [
        "Préferences",
        "Similaire",
        "Notes",
        "Historique",
        "Statistiques",
        "Donnees",
    ]
)

with tab_pref:
    st.subheader("Films conseilles selon les preferences")
    pref_table = all_pref.sort_values(
        by=["rating", "year"],
        ascending=[False, False],
    ).head(top_n)
    st.dataframe(
        clean_table(pref_table[["movie_id", "title", "genre", "year", "duration", "rating"]]),
        width="stretch",
        hide_index=True,
    )

with tab_similarity:
    st.subheader("Classement selon la similitude de genre")
    st.dataframe(
        clean_table(
            df_similarity[
                ["movie_id", "title", "genre", "rating", "year", "score_similarite"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

with tab_rating:
    st.subheader("Classement selon les notes des utilisateurs")
    st.dataframe(
        clean_table(
            df_rating.head(top_n)[
                ["movie_id", "title", "genre", "year", "duration", "rating"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

with tab_history:
    st.subheader("Historique de visionnage de l'utilisateur")
    history_table = user_watch_history.sort_values("date", ascending=False)
    st.dataframe(
        clean_table(history_table[["date", "movie_title", "genre", "rate"]]),
        width="stretch",
        hide_index=True,
    )

with tab_stats:
    st.subheader("Repartition des genres visionnes")

    if user_watch_history.empty:
        st.info("Aucune donnee d'historique pour cet utilisateur.")
    else:
        genre_counts = user_watch_history["genre"].value_counts()
        st.bar_chart(genre_counts)

        st.subheader("Evaluations du dernier mois du dataset")
        last_date = df_watch_history["date"].max()
        last_month = df_watch_history[
            df_watch_history["date"] >= last_date - pd.DateOffset(months=1)
        ].sort_values("date")

        if last_month.empty:
            st.info("Aucune evaluation sur le dernier mois.")
        else:
            rating_timeline = last_month.set_index("date")["rate"]
            st.line_chart(rating_timeline)

with tab_data:
    st.subheader("Tables sources")
    table_name = st.radio(
        "Table a afficher",
        ["users", "films", "watch_history"],
        horizontal=True,
    )

    if table_name == "users":
        st.dataframe(df_user, width="stretch", hide_index=True)
    elif table_name == "films":
        st.dataframe(df_film, width="stretch", hide_index=True)
    else:
        st.dataframe(df_watch_history, width="stretch", hide_index=True)
