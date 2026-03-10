from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np


def get_data(seed):
    wine_quality = fetch_ucirepo(name="Wine Quality")
    wine_quality_data = wine_quality["data"]["original"]

    wine_X = wine_quality_data.drop(columns=["quality"])  # drop quality
    wine_X["color"] = (wine_X["color"] == "red").astype(
        float
    )  # make color a number instead of a string
    scaler = StandardScaler()
    wine_X = scaler.fit_transform(wine_X)  # scaling values

    wine_y = wine_quality_data["quality"]
    wine_y -= wine_y.min()

    wine_X_train_val, wine_X_test, wine_y_train_val, wine_y_test = train_test_split(
        wine_X, wine_y, test_size=0.2, random_state=seed
    )

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


def get_binary_data(seed):
    wine_quality = fetch_ucirepo(name="Wine Quality")
    wine_quality_data = wine_quality["data"]["original"]

    wine_X = wine_quality_data.drop(columns=["quality"])  # drop quality
    wine_X["color"] = (wine_X["color"] == "red").astype(
        float
    )  # make color a number instead of a string
    scaler = StandardScaler()
    wine_X = scaler.fit_transform(wine_X)  # scaling values

    wine_y = wine_quality_data["quality"] >= 6

    wine_X_train_val, wine_X_test, wine_y_train_val, wine_y_test = train_test_split(
        wine_X, wine_y, test_size=0.2, random_state=seed
    )

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

def get_binned_stratified_data(seed):
    wine_quality = fetch_ucirepo(name="Wine Quality")
    wine_quality_data = wine_quality["data"]["original"]

    wine_X = wine_quality_data.drop(columns=["quality"])  # drop quality
    wine_X["color"] = (wine_X["color"] == "red").astype(
        float
    )  # make color a number instead of a string
    scaler = StandardScaler()
    wine_X = scaler.fit_transform(wine_X)  # scaling values

    wine_y = wine_quality_data["quality"]
    wine_y -= wine_y.min()

    #sorry this isn't general lol, binning the data
    bins_y = [
        wine_y[wine_y <= 1], #classes 3-4
        wine_y[(wine_y >= 2) & (wine_y <=3)], #classes 5-6
        wine_y[wine_y >= 4], #classes 7-9
    ]
    bins_X = [wine_X[b.index] for b in bins_y]

    #sampling the data, this is general!
    size = min(len(y) for y in bins_y)
    sampled_X = []
    sampled_y = []
    for i, (bX, by) in enumerate(zip(bins_X, bins_y)):
        #randomly sampling by size (minimum length) per bin
        index = np.random.choice(len(by), size=size, replace=False)
        #picking X value at these random indices
        sampled_X.append(bX[index])
        #they're all in the same 0 indexed bins, so can just make them all quality i
        sampled_y.append(pd.Series(np.full(size, i))) #not scuffed at all trust

    #combining all the sampled data (not random order!) 
    wine_X = np.vstack(sampled_X)
    wine_y = pd.concat(sampled_y).reset_index(drop=True)

    wine_X_train_val, wine_X_test, wine_y_train_val, wine_y_test = train_test_split(
        wine_X, wine_y, test_size=0.2, random_state=seed
    )

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

def confusion(classifier, X_tr, y_tr, X_val, y_val):
    classifier.fit(X_tr, y_tr)

    acc_tr = accuracy_score(y_tr, classifier.predict(X_tr))
    acc_te = accuracy_score(y_val, classifier.predict(X_val))

    print(f'Results:')
    print(f'--- Accuracy (train): {100*acc_tr:.2f}%')
    print(f'--- Accuracy (test): {100*acc_te:.2f}%')

    cm = confusion_matrix(y_val, classifier.predict(X_val))
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot();

def print_final_results(classifier, X_tr, y_tr, X_val, y_val, X_te, y_te):
    classifier.fit(X_tr, y_tr)

    sklearn_acc_tr = accuracy_score(y_tr, classifier.predict(X_tr))
    sklearn_acc_val = accuracy_score(y_val, classifier.predict(X_val))
    sklearn_acc_te = accuracy_score(y_te, classifier.predict(X_te))

    print(f'Results:')
    print(f'--- Accuracy (train): {100*sklearn_acc_tr:.2f}%')
    print(f'--- Accuracy (validation): {100*sklearn_acc_val:.2f}%')
    print(f'--- Accuracy (test): {100*sklearn_acc_te:.2f}%') #this is the final accuracy