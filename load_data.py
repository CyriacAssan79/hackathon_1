import ast

import pandas as pd


def parse_preferences(value):
    if pd.isna(value):
        return []

    text = str(value)
    if text.startswith("["):
        return ast.literal_eval(text)

    return [genre for genre in text.split("\"") if genre]


def load_csv_data():
    df_user = pd.read_csv("users.csv")
    df_film = pd.read_csv("films.csv")
    df_watch_history = pd.read_csv("watch_history.csv")

    df_user["preference"] = df_user["preference"].apply(parse_preferences)
    df_watch_history["date"] = pd.to_datetime(df_watch_history["date"])

    return df_user, df_film, df_watch_history
