import streamlit as st
import pandas as pd
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

@st.cache_resource
def treinar_modelo():
    df = pd.read_csv("dados_treinamento_organizado.csv")
    df['data'] = pd.to_datetime(df['data'])
    df['dia_semana'] = df['data'].dt.dayofweek
    df['mes'] = df['data'].dt.month
    X = df[['descricao', 'valor', 'dia_semana', 'mes']]
    y = df['categoria']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    preprocessador = ColumnTransformer(transformers=[
        ('desc', TfidfVectorizer(), 'descricao'),
        ('num', StandardScaler(), ['valor', 'dia_semana', 'mes'])
    ])
    modelo = make_pipeline(preprocessador, MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    modelo.fit(X, y_encoded)
    return modelo, le

st.title("🔍 Classificador de Categoria Financeira com MLP")

descricao = st.text_input("Descrição da transação", "Supermercado Extra")
valor = st.number_input("Valor da transação", min_value=0.0, step=1.0)
data_transacao = st.date_input("Data da transação", date.today())

if st.button("Classificar"):
    modelo, le = treinar_modelo()
    nova_entrada = pd.DataFrame([{
        "descricao": descricao,
        "valor": valor,
        "dia_semana": data_transacao.weekday(),
        "mes": data_transacao.month
    }])
    pred = modelo.predict(nova_entrada)[0]
    categoria_predita = le.inverse_transform([pred])[0]
    st.success(f"💡 Categoria sugerida: **{categoria_predita}**")
