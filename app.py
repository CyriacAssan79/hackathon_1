from pathlib import Path
import ast
import contextlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
NOTEBOOK_PATH = BASE_DIR / "hackathon.ipynb"
USERS_PATH = BASE_DIR / "users.csv"
FILMS_PATH = BASE_DIR / "films.csv"
WATCH_HISTORY_PATH = BASE_DIR / "watch_history.csv"


st.set_page_config(
    page_title="Recommandation de films",
    page_icon=":movie_camera:",
    layout="wide",
)


def fix_text(value):
    if pd.isna(value):
        return value

    text = str(value)
    if "Ã" not in text and "Â" not in text:
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def parse_preferences(value):
    if isinstance(value, list):
        return [fix_text(item) for item in value]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            return [fix_text(item) for item in ast.literal_eval(text)]
        except (SyntaxError, ValueError):
            return []

    separator = "|" if "|" in text else ","
    return [fix_text(item.strip()) for item in text.split(separator) if item.strip()]


def normalize_films(df):
    df = df.copy()
    rename_map = {"title": "movie", "rating": "rate"}
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df})

    for column in ["movie", "genre", "director"]:
        if column in df:
            df[column] = df[column].apply(fix_text)

    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce").astype("Int64")
    df["rate"] = pd.to_numeric(df.get("rate"), errors="coerce")
    df["rate"] = df["rate"].fillna(df["rate"].mean()).round(2)

    return df.drop_duplicates().reset_index(drop=True)


def normalize_history(df):
    df = df.copy()
    rename_map = {
        "movie_title": "movie",
        "date": "watch_date",
        "rate": "rating",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df})

    for column in ["movie", "genre", "director"]:
        if column in df:
            df[column] = df[column].apply(fix_text)

    df["watch_date"] = pd.to_datetime(df["watch_date"], errors="coerce")
    df = df[df["watch_date"].notna()].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].mean()).round(2)
    df["year"] = df["watch_date"].dt.year
    df["month"] = df["watch_date"].dt.month

    return df.sort_values("watch_date").reset_index(drop=True)


def normalize_users(df):
    df = df.copy()
    df = df.rename(columns={"preferences": "preference"})

    for column in ["name", "country", "email"]:
        if column in df:
            df[column] = df[column].fillna("Non renseigne").apply(fix_text)

    if "age" in df:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median()).astype(int)

    if "preference" in df:
        df["preference"] = df["preference"].apply(parse_preferences)
    else:
        df["preference"] = [[] for _ in range(len(df))]

    return df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def load_notebook_frames():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    env = {"__name__": "__streamlit_notebook_data__"}

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if source.lstrip().startswith("preferences = df_users.loc"):
            break

        # The app implements these notebook ideas without SciPy/Matplotlib, so
        # data loading can stay lightweight and focused on dataframe creation.
        source = source.replace("from scipy.stats import mode,pearsonr", "")
        source = source.replace("from scipy.stats import mode", "")
        source = source.replace("from scipy.cluster.vq import whiten, kmeans, vq", "")
        source = source.replace("import matplotlib.pyplot as plt", "")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, env)

    return (
        env["df_users"].copy(),
        env["df_film"].copy(),
        env["df_watch_history"].copy(),
    )


@st.cache_data
def load_data():
    if NOTEBOOK_PATH.exists():
        raw_users, raw_films, raw_history = load_notebook_frames()
        df_user = normalize_users(raw_users)
        df_film = normalize_films(raw_films)
        df_watch_history = normalize_history(raw_history)
    else:
        df_film = normalize_films(pd.read_csv(FILMS_PATH))
        df_watch_history = normalize_history(pd.read_csv(WATCH_HISTORY_PATH))
        df_user = normalize_users(pd.read_csv(USERS_PATH))

    return df_user, df_film, df_watch_history


def recommend_by_preferences(df_film, preferences, top_n):
    return (
        df_film[df_film["genre"].isin(preferences)]
        .sort_values(["rate", "year"], ascending=[False, False])
        .head(top_n)
        .copy()
    )


def rank_by_rating(df_film, top_n):
    return df_film.sort_values(["rate", "year"], ascending=[False, False]).head(top_n).copy()


def rank_by_similarity(df_film, preferences, watched_movies, top_n):
    df = df_film.copy()
    df["score_similarite"] = df["genre"].isin(preferences).astype(int)
    df = df[~df["movie"].isin(watched_movies)]

    return df.sort_values(
        ["score_similarite", "rate", "year"],
        ascending=[False, False, False],
    ).head(top_n)


def pearson_correlation(df_watch_history, user_id_1, user_id_2):
    user1 = df_watch_history[df_watch_history["user_id"] == user_id_1][["movie", "rating"]]
    user2 = df_watch_history[df_watch_history["user_id"] == user_id_2][["movie", "rating"]]
    common_movies = pd.merge(user1, user2, on="movie", suffixes=("_user1", "_user2"))

    if len(common_movies) < 2:
        return None, len(common_movies)

    if (
        common_movies["rating_user1"].nunique() < 2
        or common_movies["rating_user2"].nunique() < 2
    ):
        return None, len(common_movies)

    return common_movies["rating_user1"].corr(common_movies["rating_user2"]), len(common_movies)


def similar_users(df_watch_history, selected_user_id):
    rows = []
    user_ids = sorted(df_watch_history["user_id"].dropna().unique())

    for other_user_id in user_ids:
        if other_user_id == selected_user_id:
            continue

        correlation, common_count = pearson_correlation(
            df_watch_history,
            selected_user_id,
            other_user_id,
        )
        if correlation is None or pd.isna(correlation):
            continue

        rows.append(
            {
                "user_id": int(other_user_id),
                "correlation_pearson": round(float(correlation), 3),
                "films_en_commun": common_count,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["user_id", "correlation_pearson", "films_en_commun"]
        )

    return pd.DataFrame(rows).sort_values(
        ["correlation_pearson", "films_en_commun"],
        ascending=[False, False],
    )


def build_genre_matrix(df_watch_history):
    return (
        df_watch_history.pivot_table(
            index="user_id",
            columns="genre",
            values="rating",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .astype(float)
    )


def simple_kmeans(values, k=3, max_iter=25):
    if len(values) == 0:
        return np.array([], dtype=int)

    k = min(k, len(values))
    centroids = values[:k].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()

        for cluster_id in range(k):
            cluster_values = values[labels == cluster_id]
            if len(cluster_values):
                new_centroids[cluster_id] = cluster_values.mean(axis=0)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def cluster_recommendations(df_watch_history, df_film, selected_user_id, selected_genres, top_n):
    matrix = build_genre_matrix(df_watch_history)
    if selected_user_id not in matrix.index:
        return pd.DataFrame(), None

    std = matrix.std(axis=0).replace(0, 1)
    values = ((matrix - matrix.mean(axis=0)) / std).to_numpy()
    labels = simple_kmeans(values, k=3)
    clusters = pd.Series(labels, index=matrix.index, name="cluster")
    selected_cluster = int(clusters.loc[selected_user_id])
    cluster_user_ids = clusters[clusters == selected_cluster].index
    watched_movies = set(
        df_watch_history[df_watch_history["user_id"] == selected_user_id]["movie"]
    )

    cluster_history = df_watch_history[
        (df_watch_history["user_id"].isin(cluster_user_ids))
        & (~df_watch_history["movie"].isin(watched_movies))
        & (df_watch_history["genre"].isin(selected_genres))
    ]

    if cluster_history.empty:
        return pd.DataFrame(), selected_cluster

    recommendations = (
        cluster_history.groupby(["movie", "genre"], as_index=False)
        .agg(note_moyenne_cluster=("rating", "mean"), vues_cluster=("rating", "size"))
        .sort_values(["note_moyenne_cluster", "vues_cluster"], ascending=[False, False])
        .head(top_n)
    )
    recommendations["note_moyenne_cluster"] = recommendations["note_moyenne_cluster"].round(2)

    film_details = df_film[["movie", "director", "year", "duration", "rate"]].drop_duplicates(
        subset=["movie"]
    )
    return recommendations.merge(film_details, on="movie", how="left"), selected_cluster


def clean_table(df):
    return df.rename(
        columns={
            "history_id": "ID historique",
            "user_id": "ID utilisateur",
            "movie": "Film",
            "genre": "Genre",
            "director": "Realisateur",
            "year": "Annee",
            "month": "Mois",
            "duration": "Duree",
            "rate": "Note film",
            "rating": "Note utilisateur",
            "watch_date": "Date de visionnage",
            "score_similarite": "Score similarite",
            "correlation_pearson": "Correlation Pearson",
            "films_en_commun": "Films en commun",
            "note_moyenne_cluster": "Note moyenne cluster",
            "vues_cluster": "Vues cluster",
        }
    )


df_user, df_film, df_watch_history = load_data()

st.title("Moteur de recommandation de films")
st.caption("Application Streamlit reconstruite a partir du notebook hackathon.")

with st.sidebar:
    st.header("Filtres")

    user_labels = df_user.apply(
        lambda row: f"{int(row['user_id'])} - {row['name']}", axis=1
    ).tolist()
    selected_user_label = st.selectbox("Utilisateur", user_labels)
    selected_user_id = int(selected_user_label.split(" - ")[0])

    top_n = st.slider("Nombre de recommandations", 5, 30, 10)
    selected_genres = st.multiselect(
        "Genres",
        sorted(df_film["genre"].dropna().unique()),
        default=sorted(df_film["genre"].dropna().unique()),
    )

filtered_films = df_film[df_film["genre"].isin(selected_genres)].copy()
user = df_user[df_user["user_id"] == selected_user_id].iloc[0]
user_history = df_watch_history[df_watch_history["user_id"] == selected_user_id].copy()
watched_movies = set(user_history["movie"])
preferences = user["preference"]

recommendations_pref = recommend_by_preferences(filtered_films, preferences, top_n)
recommendations_rating = rank_by_rating(filtered_films, top_n)
recommendations_similarity = rank_by_similarity(
    filtered_films,
    preferences,
    watched_movies,
    top_n,
)
recommendations_cluster, selected_cluster = cluster_recommendations(
    df_watch_history,
    df_film,
    selected_user_id,
    selected_genres,
    top_n,
)
pearson_users = similar_users(df_watch_history, selected_user_id)

last_seen = None
if not user_history.empty:
    last_seen = user_history.sort_values("watch_date").iloc[-1]["movie"]

metric_cols = st.columns(4)
metric_cols[0].metric("Utilisateurs", f"{len(df_user):,}".replace(",", " "))
metric_cols[1].metric("Films", f"{len(df_film):,}".replace(",", " "))
metric_cols[2].metric("Visionnages", f"{len(df_watch_history):,}".replace(",", " "))
metric_cols[3].metric("Genres", df_film["genre"].nunique())

st.divider()

profile_col, recommendation_col = st.columns([1, 2])

with profile_col:
    st.subheader("Profil")
    st.write(f"**Nom :** {user['name']}")
    if pd.notna(user.get("age")):
        st.write(f"**Age :** {int(user['age'])} ans")
    st.write(f"**Pays :** {user.get('country', 'Non renseigne')}")
    st.write(f"**Email :** {user.get('email', 'Non renseigne')}")
    st.write("**Preferences :** " + (", ".join(preferences) if preferences else "Non renseigne"))

with recommendation_col:
    st.subheader("Suggestions")

    if last_seen:
        st.write(f"**Dernier film regarde :** {last_seen}")
    else:
        st.info("Aucun historique de visionnage pour cet utilisateur.")

    best_similarity = (
        recommendations_similarity.iloc[0]["movie"]
        if not recommendations_similarity.empty
        else "Aucune suggestion"
    )
    best_rating = (
        recommendations_rating.iloc[0]["movie"]
        if not recommendations_rating.empty
        else "Aucune suggestion"
    )
    best_preference = (
        recommendations_pref.iloc[0]["movie"]
        if not recommendations_pref.empty
        else "Aucune suggestion"
    )
    best_cluster = (
        recommendations_cluster.iloc[0]["movie"]
        if not recommendations_cluster.empty
        else "Aucune suggestion"
    )

    st.success(f"Par similarite de genre : {best_similarity}")
    st.info(f"Par meilleure note globale : {best_rating}")
    st.warning(f"Par preferences declarees : {best_preference}")
    st.success(f"Par cluster d'utilisateurs : {best_cluster}")

tab_pref, tab_similarity, tab_cluster, tab_pearson, tab_rating, tab_history, tab_stats, tab_data = st.tabs(
    [
        "Preferences",
        "Similarite",
        "Cluster",
        "Pearson",
        "Notes",
        "Historique",
        "Statistiques",
        "Donnees",
    ]
)

with tab_pref:
    st.subheader("Films conseilles selon les preferences")
    columns = ["movie", "genre", "director", "year", "duration", "rate"]
    st.dataframe(clean_table(recommendations_pref[columns]), hide_index=True, width="stretch")

with tab_similarity:
    st.subheader("Classement par similarite")
    columns = ["movie", "genre", "director", "rate", "year", "score_similarite"]
    st.dataframe(clean_table(recommendations_similarity[columns]), hide_index=True, width="stretch")

with tab_cluster:
    st.subheader("Recommandations par cluster")
    if selected_cluster is None or recommendations_cluster.empty:
        st.info("Aucune recommandation disponible pour le cluster de cet utilisateur.")
    else:
        st.write(f"**Cluster de l'utilisateur :** {selected_cluster}")
        columns = [
            "movie",
            "genre",
            "director",
            "year",
            "note_moyenne_cluster",
            "vues_cluster",
            "rate",
        ]
        st.dataframe(
            clean_table(recommendations_cluster[columns]),
            hide_index=True,
            width="stretch",
        )

with tab_pearson:
    st.subheader("Utilisateurs proches par correlation Pearson")
    if pearson_users.empty:
        st.info("Aucun utilisateur comparable avec au moins deux films en commun.")
    else:
        pearson_display = pearson_users.merge(
            df_user[["user_id", "name"]],
            on="user_id",
            how="left",
        ).head(top_n)
        st.dataframe(clean_table(pearson_display), hide_index=True, width="stretch")

with tab_rating:
    st.subheader("Meilleures notes")
    columns = ["movie", "genre", "director", "year", "duration", "rate"]
    st.dataframe(clean_table(recommendations_rating[columns]), hide_index=True, width="stretch")

with tab_history:
    st.subheader("Historique de l'utilisateur")
    if user_history.empty:
        st.info("Aucun film regarde par cet utilisateur.")
    else:
        columns = ["watch_date", "movie", "genre", "director", "rating"]
        table = user_history.sort_values("watch_date", ascending=False)[columns]
        st.dataframe(clean_table(table), hide_index=True, width="stretch")

with tab_stats:
    st.subheader("Genres visionnes par l'utilisateur")
    if user_history.empty:
        st.info("Aucune statistique disponible pour cet utilisateur.")
    else:
        genre_counts = user_history["genre"].value_counts()
        st.bar_chart(genre_counts)

    st.subheader("Notes du dernier mois du dataset")
    last_date = df_watch_history["watch_date"].max()
    last_month = df_watch_history[
        df_watch_history["watch_date"] >= last_date - pd.DateOffset(months=1)
    ].sort_values("watch_date")

    if last_month.empty:
        st.info("Aucune evaluation sur le dernier mois.")
    else:
        timeline = last_month.set_index("watch_date")["rating"]
        st.line_chart(timeline)

    st.subheader("Genres les mieux notes depuis mai 2026")
    popular_genres = df_watch_history[
        (df_watch_history["rating"] >= 4)
        & (df_watch_history["watch_date"] > pd.Timestamp("2026-05-01"))
    ]["genre"].value_counts()

    if popular_genres.empty:
        st.info("Aucun genre avec une note superieure ou egale a 4 depuis mai 2026.")
    else:
        st.bar_chart(popular_genres)

    st.subheader("Genres les moins bien notes depuis mai 2026")
    low_rated_genres = df_watch_history[
        (df_watch_history["rating"] <= 2)
        & (df_watch_history["watch_date"] > pd.Timestamp("2026-05-01"))
    ]["genre"].value_counts()

    if low_rated_genres.empty:
        st.info("Aucun genre avec une note inferieure ou egale a 2 depuis mai 2026.")
    else:
        st.bar_chart(low_rated_genres)

with tab_data:
    st.subheader("Tables sources")
    table_name = st.radio(
        "Table a afficher",
        ["Utilisateurs", "Films", "Historique"],
        horizontal=True,
    )

    if table_name == "Utilisateurs":
        st.dataframe(clean_table(df_user), hide_index=True, width="stretch")
    elif table_name == "Films":
        st.dataframe(clean_table(df_film), hide_index=True, width="stretch")
    else:
        st.dataframe(clean_table(df_watch_history), hide_index=True, width="stretch")
