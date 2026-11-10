import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

base = os.path.dirname(os.path.abspath(__file__))
df = joblib.load(os.path.join(base, "development.pkl"))
categorical_features = ["hr", "weekday", "season", "hour_workingday"]
numeric_features = ["workingday", "yr"]
calendar_features = categorical_features + numeric_features
x = df[calendar_features]
y = np.log1p(df["cnt"])
#y = df["cnt"]

tscv = TimeSeriesSplit(n_splits=12, test_size=24*30)

def original_rmse(estimator, X, y):
    y_pred = np.expm1(estimator.predict(X))
    y_true = np.expm1(y)
    #y_pred = estimator.predict(X)
    #y_true = y
    return -root_mean_squared_error(y_true, y_pred)

def build_model():
    preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge())])
    
    search = GridSearchCV(
        model,
        param_grid={"regressor__alpha": [0.1, 1, 10, 100]},
        cv=TimeSeriesSplit(n_splits=3, test_size=24*14),
        scoring=original_rmse) # scoring="neg_root_mean_squared_error"
    return search 

best_alphas = []
def cross_validate(model, x, y, df, tscv):
    rmse_scores = []
    validation_results = []

    for train_idx, val_idx in tscv.split(x):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
        print(
            f"Fold {len(rmse_scores) + 1}: "
            f"Training: {df.iloc[train_idx]['dteday'].min().date()} "
            f"to {df.iloc[train_idx]['dteday'].max().date()}, "
            f"Validation: {df.iloc[val_idx]['dteday'].min().date()} "
            f"to {df.iloc[val_idx]['dteday'].max().date()}"
        )

        model.fit(x_train, y_train)
        best_alphas.append(model.best_params_["regressor__alpha"])

        y_pred = np.expm1(model.predict(x_val))
        y_val = np.expm1(y_val)
        #y_pred = model.predict(x_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        rmse_scores.append(rmse)
        validation_results.append((df.iloc[val_idx]["dteday"], 
                                y_pred, y_val - y_pred))
        
    return rmse_scores, validation_results

def plot_validation_results(validation_results):
    for dates, y_pred, errors in validation_results:
        plt.scatter(dates, errors, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.title("Log: Residuals vs Date")
    #plt.title("Non-Log: Residuals vs Date")
    plt.show()

    for dates, y_pred, errors in validation_results:
        plt.scatter(y_pred, errors, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Count")
    plt.ylabel("Residual")
    plt.title("Log: Residuals vs Predicted Count")
    #plt.title("Non-Log: Residuals vs Predicted Count")
    plt.show()

model = build_model()
rmse_scores, validation_results = cross_validate(model, x, y, df, tscv)

print("/n======= Log Validation Results =======")
#print("======= Non-Log Validation Results =======")


for i, rmse in enumerate(rmse_scores, start=1):
    print(f"Fold {i}: RMSE = {rmse:.4f}")

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores, ddof=1)
se_rmse = std_rmse / np.sqrt(len(rmse_scores))

print(f"Mean RMSE: {mean_rmse:.4f}")
print(f"Std RMSE: {std_rmse:.4f}")
print(f"Standard Error: {se_rmse:.4f}")
print(f"RMSE Range: {min(rmse_scores):.4f} - {max(rmse_scores):.4f}")
print(f"Best Alphas: {best_alphas}")
print(f"Baseline RMSE = {mean_rmse:.4f} ± {se_rmse:.4f} (SE)")

plt.figure(figsize=(5, 4))
plt.errorbar(0, mean_rmse, yerr=se_rmse, fmt="o", capsize=6, markersize=8)
plt.xlim(-1, 1)
plt.xticks([0], ["Model A"])
plt.ylabel("RMSE (rentals/hour)")
plt.title("Baseline RMSE ± SE")
plt.savefig("baseline.png")
plt.show()

same_months = df[df["dteday"].dt.month <= 9]
print(same_months.groupby("yr")["cnt"].mean())
plot_validation_results(validation_results)

