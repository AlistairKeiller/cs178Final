from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


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