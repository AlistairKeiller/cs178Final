from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
