from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_data():
    target_dir = DATA_DIR / "ml-100k"
    if target_dir.exists():
        return target_dir

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-100k.zip"

    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(DATA_DIR)

    return target_dir


def load_data():
    path = download_data()

    ratings = pd.read_csv(
        path / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        path / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
    )

    return ratings, movies


def temporal_split(ratings, test_fraction=0.2):
    train_parts = []
    test_parts = []

    for _, group in ratings.sort_values("timestamp").groupby("user_id"):
        n_test = max(1, int(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, test


def build_matrix(train):
    matrix = train.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0,
    )
    return matrix


def fit_svd(matrix, n_components=50):
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    model = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = model.fit_transform(matrix.values)
    reconstructed = user_factors @ model.components_
    predictions = pd.DataFrame(
        reconstructed,
        index=matrix.index,
        columns=matrix.columns,
    )
    return model, predictions


def rmse_score(test, predictions):
    y_true = []
    y_pred = []

    users = set(predictions.index)
    items = set(predictions.columns)

    for row in test.itertuples():
        if row.user_id in users and row.item_id in items:
            y_true.append(row.rating)
            y_pred.append(np.clip(predictions.loc[row.user_id, row.item_id], 1, 5))

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def popularity_recommendations(train, user_id, k=10):
    seen = set(train.loc[train["user_id"] == user_id, "item_id"])
    scores = (
        train.assign(relevant=(train["rating"] >= 4).astype(int))
        .groupby("item_id")["relevant"]
        .sum()
        .sort_values(ascending=False)
    )
    return [item for item in scores.index if item not in seen][:k]


def svd_recommendations(train, predictions, user_id, k=10):
    seen = set(train.loc[train["user_id"] == user_id, "item_id"])

    if user_id not in predictions.index:
        return []

    scores = predictions.loc[user_id].sort_values(ascending=False)
    return [item for item in scores.index if item not in seen][:k]


def ranking_metrics(train, test, predictions, k=10):
    precision_scores = []
    recall_scores = []
    baseline_precision_scores = []
    baseline_recall_scores = []

    for user_id, group in test.groupby("user_id"):
        relevant = set(group.loc[group["rating"] >= 4, "item_id"])

        if not relevant:
            continue

        model_recs = svd_recommendations(train, predictions, user_id, k)
        base_recs = popularity_recommendations(train, user_id, k)

        model_hits = len(set(model_recs) & relevant)
        base_hits = len(set(base_recs) & relevant)

        precision_scores.append(model_hits / k)
        recall_scores.append(model_hits / len(relevant))

        baseline_precision_scores.append(base_hits / k)
        baseline_recall_scores.append(base_hits / len(relevant))

    return {
        "precision_at_10": float(np.mean(precision_scores)),
        "recall_at_10": float(np.mean(recall_scores)),
        "baseline_precision_at_10": float(np.mean(baseline_precision_scores)),
        "baseline_recall_at_10": float(np.mean(baseline_recall_scores)),
    }


def save_example(train, predictions, movies):
    user_id = int(predictions.index[0])
    items = svd_recommendations(train, predictions, user_id, 10)

    result = pd.DataFrame({"item_id": items})
    result = result.merge(movies, on="item_id", how="left")
    result.insert(0, "user_id", user_id)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    result.to_csv(ARTIFACTS_DIR / "top_recommendations.csv", index=False)

    return user_id, result


def main():
    ratings, movies = load_data()
    train, test = temporal_split(ratings)
    matrix = build_matrix(train)
    model, predictions = fit_svd(matrix)

    metrics = ranking_metrics(train, test, predictions)
    metrics["rmse"] = rmse_score(test, predictions)
    metrics["explained_variance"] = float(model.explained_variance_ratio_.sum())
    metrics["train_rows"] = int(len(train))
    metrics["test_rows"] = int(len(test))
    metrics["users"] = int(ratings["user_id"].nunique())
    metrics["movies"] = int(ratings["item_id"].nunique())

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    user_id, example = save_example(train, predictions, movies)

    print("Метрики:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nПример рекомендаций для пользователя {user_id}:")
    print(example.to_string(index=False))


if __name__ == "__main__":
    main()
