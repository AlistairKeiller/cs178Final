from typing import Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio


def load_data():
    wine_quality_red = pd.read_csv("winequality-red.csv", delimiter=";")
    wine_quality_white = pd.read_csv("winequality-white.csv", delimiter=";")
    wine_quality_red["color"] = "red"
    wine_quality_white["color"] = "white"
    wine_quality_data = pd.concat(
        [wine_quality_red, wine_quality_white], ignore_index=True
    )
    return wine_quality_data


def get_data(seed, stratified=False, binary=False, deduped=False):
    wine_quality_data = load_data()

    if deduped:
        wine_quality_data.drop_duplicates(inplace=True)

    wine_X = wine_quality_data.drop(columns=["quality"])  # drop quality
    wine_X["color"] = (wine_X["color"] == "red").astype(
        float
    )  # make color a number instead of a string
    scaler = StandardScaler()
    wine_X = scaler.fit_transform(wine_X)  # scaling values

    if binary:
        wine_y = wine_quality_data["quality"] >= 6
    else:
        wine_y = wine_quality_data["quality"]
        wine_y -= wine_y.min()

    wine_X_train_val, wine_X_test, wine_y_train_val, wine_y_test = train_test_split(
        wine_X, wine_y, test_size=0.2, random_state=seed
    )
    if stratified:
        wine_X_train_val, wine_y_train_val = RandomOverSampler(
            random_state=seed
        ).fit_resample(wine_X_train_val, wine_y_train_val)

    wine_X_tr, wine_X_val, wine_y_tr, wine_y_val = train_test_split(
        wine_X_train_val, wine_y_train_val, test_size=0.25, random_state=seed
    )

    return (
        wine_X_tr,
        wine_X_val,
        wine_X_test,
        wine_y_tr,
        wine_y_val,
        wine_y_test,
    )


# def get_binned_stratified_data(seed):
#     np.random.seed(seed=seed)
#     wine_quality_data = load_data()

#     wine_X = wine_quality_data.drop(columns=["quality"])  # drop quality
#     wine_X["color"] = (wine_X["color"] == "red").astype(
#         float
#     )  # make color a number instead of a string
#     scaler = StandardScaler()
#     wine_X = scaler.fit_transform(wine_X)  # scaling values

#     wine_y = wine_quality_data["quality"]
#     wine_y -= wine_y.min()

#     # sorry this isn't general lol, binning the data
#     bins_y = [
#         wine_y[wine_y <= 1],  # classes 3-4
#         wine_y[(wine_y >= 2) & (wine_y <= 3)],  # classes 5-6
#         wine_y[wine_y >= 4],  # classes 7-9
#     ]
#     bins_X = [wine_X[b.index] for b in bins_y]

#     # sampling the data, this is general!
#     size = min(len(y) for y in bins_y)
#     sampled_X = []
#     sampled_y = []
#     for i, (bX, by) in enumerate(zip(bins_X, bins_y)):
#         # randomly sampling by size (minimum length) per bin
#         index = np.random.choice(len(by), size=size, replace=False)
#         # picking X value at these random indices
#         sampled_X.append(bX[index])
#         # they're all in the same 0 indexed bins, so can just make them all quality i
#         sampled_y.append(pd.Series(np.full(size, i)))  # not scuffed at all trust

#     # combining all the sampled data (not random order!)
#     wine_X = np.vstack(sampled_X)
#     wine_y = pd.concat(sampled_y).reset_index(drop=True)

#     wine_X_train_val, wine_X_test, wine_y_train_val, wine_y_test = train_test_split(
#         wine_X, wine_y, test_size=0.2, random_state=seed
#     )

#     wine_X_tr, wine_X_val, wine_y_tr, wine_y_val = train_test_split(
#         wine_X_train_val, wine_y_train_val, test_size=0.25, random_state=seed
#     )

#     return (
#         wine_X_tr,
#         wine_X_val,
#         wine_X_test,
#         wine_y_tr,
#         wine_y_val,
#         wine_y_test,
#     )


def confusion(classifier, X_tr, y_tr, X_val, y_val, save_file: str | None = None):
    classifier.fit(X_tr, y_tr)

    acc_tr = accuracy_score(y_tr, classifier.predict(X_tr))
    acc_te = accuracy_score(y_val, classifier.predict(X_val))

    print(f"Results:")
    print(f"--- Accuracy (train): {100 * acc_tr:.2f}%")
    print(f"--- Accuracy (test): {100 * acc_te:.2f}%")

    cm = confusion_matrix(y_val, classifier.predict(X_val))
    fig = px.imshow(
        cm,
        text_auto=True,
        width=900,
        height=750,
    )
    fig.update_layout(
        xaxis_title="Predicted label",
        yaxis_title="True label",
        xaxis=dict(
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            tickmode="linear",
            dtick=1,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_traces(textfont=dict(size=14))
    if save_file:
        pio.write_image(fig, save_file)
    fig.show()


def print_final_results(classifier, X_tr, y_tr, X_val, y_val, X_te, y_te):
    classifier.fit(X_tr, y_tr)

    sklearn_acc_tr = accuracy_score(y_tr, classifier.predict(X_tr))
    sklearn_acc_val = accuracy_score(y_val, classifier.predict(X_val))
    sklearn_acc_te = accuracy_score(y_te, classifier.predict(X_te))

    print("Results:")
    print(f"--- Accuracy (train): {100 * sklearn_acc_tr:.2f}%")
    print(f"--- Accuracy (validation): {100 * sklearn_acc_val:.2f}%")
    print(
        f"--- Accuracy (test): {100 * sklearn_acc_te:.2f}%"
    )  # this is the final accuracy


def plot_curves(
    lines: list[list[tuple[Any, Any]]],
    labels: list[str],
    x_name: str = "x axis",
    y_name: str = "y axis",
    label_name: str = "label",
    image_name: str | None = None,
    **kwargs,
):
    df = pd.DataFrame(
        [
            {x_name: x, y_name: y, label_name: label}
            for data, label in zip(lines, labels)
            for x, y in data
        ]
    )
    fig = px.line(df, x=x_name, y=y_name, color=label_name, width=1200, **kwargs)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=18),
        legend=dict(font=dict(size=16)),
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    )
    if image_name:
        pio.write_image(fig, f"images/{image_name}.svg")
    fig.show()


def train_and_plot_learning_curves(
    models,
    wine_X_tr,
    wine_y_tr,
    wine_X_val,
    wine_y_val,
    seed: int,
    param_to_test: str,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 15),
):
    data_size = len(wine_X_tr)
    train_accuracies = []
    val_accuracies = []
    for model in models:
        train_acc_curve = []
        val_acc_curve = []
        np.random.seed(seed)
        for train_size in train_sizes:
            indices = np.random.choice(
                data_size, int(train_size * data_size), replace=False
            )
            wine_X_tr_batch = wine_X_tr[indices]
            wine_y_tr_batch = wine_y_tr.iloc[indices]
            model.fit(wine_X_tr_batch, wine_y_tr_batch)
            train_acc_curve.append(
                (train_size, model.score(wine_X_tr_batch, wine_y_tr_batch))
            )
            val_acc_curve.append((train_size, model.score(wine_X_val, wine_y_val)))
        train_accuracies.append(train_acc_curve)
        val_accuracies.append(val_acc_curve)

    param_value_str = [str(getattr(model, param_to_test)) for model in models]
    plot_curves(
        train_accuracies,
        param_value_str,
        "train_size",
        "training accuracy",
        param_to_test,
        f"train_{param_to_test}",
    )
    plot_curves(
        val_accuracies,
        param_value_str,
        "train_size",
        "validation accuracy",
        param_to_test,
        f"validation_{param_to_test}",
    )


def final_test(model, wine_X_test, wine_y_test):
    predictions = model.predict(wine_X_test)
    acc = accuracy_score(wine_y_test, predictions)
    precision = precision_score(wine_y_test, predictions, average="weighted")
    recall = recall_score(wine_y_test, predictions, average="weighted")
    f1 = f1_score(wine_y_test, predictions, average="weighted")
    return pd.DataFrame(
        [
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    )
