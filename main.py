codigo_streamlit = """
import streamlit as st
import pandas as pd
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# === Função para carregar e treinar o modelo ===
@st.cache_resource
def treinar_modelo():
    # Carregar dados
    df = pd.read_csv("dados_treinamento_organizado.csv")
    df['data'] = pd.to_datetime(df['data'])

    # Extração de features da data
    df['dia_semana'] = df['data'].dt.dayofweek
    df['mes'] = df['data'].dt.month

    # Seleção de atributos
    X = df[['descricao', 'valor', 'dia_semana', 'mes']]
    y = df['categoria']

    # Codificação do alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Pipeline de pré-processamento
    preprocessador = ColumnTransformer(transformers=[
        ('desc', TfidfVectorizer(), 'descricao'),
        ('num', StandardScaler(), ['valor', 'dia_semana', 'mes'])
    ])

    # Pipeline do modelo
    modelo = make_pipeline(preprocessador, MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))

    # Treinamento
    modelo.fit(X, y_encoded)

    return modelo, le

# === Interface Streamlit ===
st.title("🔍 Classificador de Categoria Financeira com MLP")

# Inputs do usuário
descricao = st.text_input("Descrição da transação", "Supermercado Extra")
valor = st.number_input("Valor da transação", min_value=0.0, step=1.0)
data_transacao = st.date_input("Data da transação", date.today())

if st.button("Classificar"):
    st.write("Classificando...")

    # Treinar ou carregar modelo
    modelo, le = treinar_modelo()

    # Pré-processar nova entrada
    nova_entrada = pd.DataFrame([{
        "descricao": descricao,
        "valor": valor,
        "dia_semana": data_transacao.weekday(),
        "mes": data_transacao.month
    }])

    # Predição
    pred = modelo.predict(nova_entrada)[0]
    categoria_predita = le.inverse_transform([pred])[0]

    # Resultado
    st.success(f"💡 Categoria sugerida: **{categoria_predita}**")
"""

# Salvar no arquivo
caminho_py = "/mnt/data/classificador_financeiro.py"
with open(caminho_py, "w") as f:
    f.write(codigo_streamlit)

caminho_py
