import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, "hour.csv"))

print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
print(df.head())

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
df = df.drop(columns=["casual", "registered", "instant"])
df["hour_workingday"] = (df["hr"].astype(str) + "_" + df["workingday"].astype(str))

calendar_features = [
    "hr", "weekday", "season",
    "workingday", "yr", "hour_workingday"]
weather_features = ["temp", "hum", "windspeed", "weathersit"]

target = "cnt"

train_df = df[df["dteday"] < "2012-10-01"].copy()
test_df = df[df["dteday"] >= "2012-10-01"].copy()

X_train = train_df[calendar_features]
y_train = train_df[target]
X_test = test_df[calendar_features]
y_test = test_df[target]

print("Development:", X_train.shape)
print("Test:", X_test.shape)

joblib.dump(train_df, os.path.join(base, "development.pkl"))
joblib.dump(test_df, os.path.join(base, "test.pkl"))
