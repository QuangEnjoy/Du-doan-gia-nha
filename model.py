import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "Nhom5_gianha_train.xlsx"
NUM = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea", "FullBath"]
CAT = ["Neighborhood", "KitchenQual"]
CLUSTER_COLS = ["LotArea", "OverallQual", "YearBuilt"]  # giống bảng 4.2 trong báo cáo
SEGMENTS = ["Bình dân", "Trung cấp", "Cao cấp"]


def load():
    return pd.read_excel(DATA)


def _prep():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT),
    ])


def train(df):
    X, y = df[NUM + CAT], df["SalePrice"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": make_pipeline(_prep(), LinearRegression()),
        "Random Forest": make_pipeline(_prep(), RandomForestRegressor(200, random_state=42, n_jobs=-1)),
    }
    metrics = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        metrics[name] = {"MAE": mean_absolute_error(yte, p),
                         "RMSE": mean_squared_error(yte, p) ** 0.5,
                         "R2": r2_score(yte, p)}
        m.fit(X, y)  # huấn luyện lại trên toàn bộ dữ liệu để dự đoán thật

    scaler = StandardScaler().fit(df[CLUSTER_COLS])
    km = KMeans(3, n_init=10, random_state=42).fit(scaler.transform(df[CLUSTER_COLS]))
    # xếp cụm theo giá trung bình tăng dần -> Bình dân / Trung cấp / Cao cấp
    order = df.groupby(km.labels_)["SalePrice"].mean().sort_values().index
    names = {c: SEGMENTS[i] for i, c in enumerate(order)}
    return models, metrics, (scaler, km, names)


def segment(clu, row):
    scaler, km, names = clu
    return names[km.predict(scaler.transform(row[CLUSTER_COLS]))[0]]


if __name__ == "__main__":  # self-check
    df = load()
    models, metrics, clu = train(df)
    print(pd.DataFrame(metrics).T)
    row = df.iloc[[0]]
    print({k: m.predict(row[NUM + CAT])[0] for k, m in models.items()}, segment(clu, row), df.SalePrice[0])
    assert metrics["Linear Regression"]["R2"] > 0.7
