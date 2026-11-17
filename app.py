import pandas as pd
import streamlit as st

import model as M

st.set_page_config(page_title="Dự đoán giá nhà Ames", page_icon="🏠")


@st.cache_resource
def get():
    df = M.load()
    return df, *M.train(df)


df, models, metrics, clu = get()

KHU_VUC = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "IDOTRR": "Iowa DOT & Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
BEP = {"Ex": "Xuất sắc", "Gd": "Tốt", "TA": "Trung bình", "Fa": "Tạm được"}

st.title("🏠 Dự đoán giá nhà – Nhóm 5 (UTH)")
st.caption("Dữ liệu Ames Housing, Iowa (Mỹ). Không dùng cho thị trường khác khi chưa huấn luyện lại.")

tab1, tab2 = st.tabs(["Dự đoán giá", "Hiệu suất mô hình"])

with tab1:
    c1, c2 = st.columns(2)
    d = df
    inp = {
        "OverallQual": c1.slider("Chất lượng tổng thể (1-10)", 1, 10, 6),
        "GrLivArea": c1.number_input("Diện tích sinh hoạt (sqft)", 300, 6000, 1500),
        "TotalBsmtSF": c1.number_input("Diện tích tầng hầm (sqft)", 0, 6000, 1000),
        "LotArea": c1.number_input("Diện tích đất (sqft)", 1000, 100000, 9000),
        "YearBuilt": c2.slider("Năm xây dựng", 1870, 2010, 1995),
        "GarageCars": c2.slider("Sức chứa gara (xe)", 0, 4, 2),
        "FullBath": c2.slider("Số phòng tắm đầy đủ", 0, 4, 2),
        "Neighborhood": c2.selectbox("Khu vực (khu dân cư tại Ames)", sorted(d.Neighborhood.unique()), format_func=lambda k: KHU_VUC.get(k, k)),
        "KitchenQual": c2.selectbox("Chất lượng bếp", list(BEP), index=2, format_func=BEP.get),
    }
    st.info("🌲 Giá được dự đoán bằng mô hình **Random Forest**.")
    choice = "Random Forest"
    if st.button("Dự đoán giá bằng Random Forest", type="primary"):
        row = pd.DataFrame([inp])
        price = models[choice].predict(row[M.NUM + M.CAT])[0]
        st.metric("Giá dự đoán", f"${max(price, 0):,.0f}")
        st.info(f"Phân khúc thị trường (K-Means): **{M.segment(clu, row)}**")

with tab2:
    st.dataframe(pd.DataFrame(metrics).T.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.4f}"}))
    st.caption("Đánh giá trên 20% dữ liệu giữ lại (random_state=42).")
