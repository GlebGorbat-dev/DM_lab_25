import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "Diagnosis", "ID",
    "Imaginary_Min", "Imaginary_Avg",
    "Real_Min", "Real_Avg",
    "Gender", "Age", "Smoking",
    "extra1", "extra2", "extra3", "extra4",

]

FEATURE_COLUMNS = [
    "Imaginary_Min", "Imaginary_Avg",
    "Real_Min", "Real_Avg",
    "Gender", "Age", "Smoking",
]

def load_data(path: str = "Exasens.csv"):
    df = pd.read_csv(path, skiprows=3, header=None)
    df.columns = RAW_COLUMNS
    df = df.drop(columns=["extra1", "extra2", "extra3", "extra4"])

    df = df.drop(columns=["ID"])

    y = df["Diagnosis"].to_numpy()
    X_df = df[FEATURE_COLUMNS].copy()

    for col in X_df.columns:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].mean())
 
    X_raw = X_df.to_numpy(dtype=float)

    mean = X_raw.mean(axis=0) #среднее
    std = X_raw.std(axis=0, ddof=0) #стандартное отклонение
    X = (X_raw - mean) / std #стандартизация
 
    return X, y, FEATURE_COLUMNS
 
 
if __name__ == "__main__":
    X, y, names = load_data()
    print("Форма X:", X.shape)
    print("Признаки:", names)
    print("Классы:", {c: int((y == c).sum()) for c in np.unique(y)})
