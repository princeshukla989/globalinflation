import streamlit as st
import pandas as pd
import kagglehub
import numpy as np
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
st.set_page_config(layout="wide", page_title="Global Inflation Analytics")
# --- Data Loading ---
@st.cache_data
def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "ssssws/global-inflation-dynamics-post-covid-20202024",
        "global_inflation_post_covid.csv"
    )
    df['date'] = pd.to_datetime(df['date'])
    df.drop_duplicates(inplace=True)
    return df
# --- Model Training ---
@st.cache_resource
def train_model(df):
    y = df['inflation_rate']
    X = df.drop(['date', 'inflation_rate'], axis=1)
    categorical_features = ['country']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    return model, X_test, y_test, numerical_features
df = load_data()
model_pipeline, X_test, y_test, numerical_cols = train_model(df)
# --- UI Layout ---
st.title("Global Inflation Dynamics")
tabs = st.tabs(["Analytics", "Predictor", "Model Stats"])
# --- Tab 1: Analytics (Native Streamlit Charts) ---
with tabs[0]:
    st.header("Inflation Trends")
    selected_country = st.selectbox("Select Country to Visualize", df['country'].unique(), index=0)
    country_data = df[df['country'] == selected_country].sort_values('date')
    st.subheader(f"Monthly Inflation Rate: {selected_country}")
    st.line_chart(country_data, x='date', y='inflation_rate')
    st.subheader("Economic Indicators Correlation")
    corr = df[numerical_cols + ['inflation_rate']].corr()
    st.dataframe(corr.style.background_gradient(cmap='coolwarm').format(precision=2))
# --- Tab 2: Predictor (Using Sliders) ---
with tabs[1]:
    st.header("Interactive Inflation Predictor")
    st.info("Adjust the sliders to simulate economic shifts.")
    col1, col2 = st.columns(2)
    with col1:
        p_country = st.selectbox('Target Country', df['country'].unique())
        p_interest = st.slider('Interest Rate (%)', -5.0, 20.0, float(df['interest_rate'].mean()))
        p_oil = st.slider('Oil Price (USD)', 20.0, 150.0, float(df['oil_price'].mean()))
        p_gdp = st.slider('GDP Growth (%)', -10.0, 15.0, float(df['gdp_growth'].mean()))
        p_unemp = st.slider('Unemployment Rate (%)', 0.0, 30.0, float(df['unemployment_rate'].mean()))
    with col2:
        p_money = st.slider('Money Supply M2', float(df['money_supply_m2'].min()), float(df['money_supply_m2'].max()), float(df['money_supply_m2'].mean()))
        p_exchange = st.slider('Exchange Rate (vs USD)', 0.0, 500.0, float(df['exchange_rate_usd'].mean()))
        p_food = st.slider('Food Price Index', 50.0, 250.0, float(df['food_price_index'].mean()))
        p_supply = st.slider('Supply Chain Index', -5.0, 5.0, float(df['supply_chain_index'].mean()))
    input_df = pd.DataFrame([{
        'country': p_country, 'interest_rate': p_interest, 'oil_price': p_oil,
        'gdp_growth': p_gdp, 'unemployment_rate': p_unemp, 'money_supply_m2': p_money,
        'exchange_rate_usd': p_exchange, 'food_price_index': p_food, 'supply_chain_index': p_supply
    }])
    prediction = model_pipeline.predict(input_df)[0]
    st.divider()
    st.metric(label=f"Predicted Inflation for {p_country}", value=f"{prediction:.2f}%", delta_color="inverse")
# --- Tab 3: Model Stats ---
with tabs[2]:
    st.header("Model Performance")
    y_pred = model_pipeline.predict(X_test)
    m1, m2, m3= st.columns(3)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    m1.metric("Model Accuracy", f"{1 - mape:.2%}")
    m2.metric("R² Score (Variance Explained)", f"{r2_score(y_test, y_pred):.4f}")
    m3.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f} pts")
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    feat_importances = pd.Series(importances[:len(numerical_cols)], index=numerical_cols)
    st.subheader("Key Economic Drivers")
    st.bar_chart(feat_importances)
