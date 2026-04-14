import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_model(student_data):

    df = pd.read_csv(student_data)
    df.columns = df.columns.str.strip()

    # Encoding
    encoded_df = pd.get_dummies(df, drop_first=True)

    X = encoded_df.iloc[:, :-1]
    y = encoded_df.iloc[:, -1]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Random Forest Model...")

    params = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10],
        'min_samples_split': [2, 4]
    }

    grid = GridSearchCV(RandomForestRegressor(), params, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    print("\n📈 Model Performance:")
    print("R2 Score:", round(r2_score(y_test, preds), 4))
    print("MAE:", round(mean_absolute_error(y_test, preds), 2))

    joblib.dump({
        'model': best_model,
        'features': feature_cols
    }, "student_model.pkl")

    print("✅ Model saved as student_model.pkl")


if __name__ == "__main__":
    train_model("dataset/student_data.csv")