import mlflow
import mlflow.sklearn
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Wait for MLflow server to be ready
max_retries = 30
for i in range(max_retries):
    try:
        mlflow.set_experiment("sklearn-experiment")
        print("Successfully connected to MLflow server")
        break
    except Exception as e:
        if i < max_retries - 1:
            print(f"Waiting for MLflow server... ({i+1}/{max_retries})")
            time.sleep(1)
        else:
            raise Exception("Could not connect to MLflow server") from e

X = np.arange(0, 100).reshape(-1, 1)
y = 2 * X.squeeze() + np.random.normal(0, 10, size=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
with mlflow.start_run():
    model = LinearRegression()
    model.fit(
        X_train,
        y_train
    )
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(
        sk_model=model, 
        name="linear_regression_model",
        serialization_format="skops"
    )
    print("Run logged Successfully")