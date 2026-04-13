from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

modelo = joblib.load("modelo_predicao_clique.pkl")
colunas = joblib.load("colunas_clique.pkl")

app = FastAPI(
    title="API Predição de Clique em Anúncios",
    description="Modelo de Regressão Logística para prever probabilidade de clique em anúncios digitais",
    version="1.0.0"
)

class Usuario(BaseModel):
    daily_time_spent_on_site: float
    age: int
    area_income: float
    daily_internet_usage: float
    male: int

@app.get("/")
def root():
    return {"status": "online", "modelo": "Regressão Logística", "versao": "1.0.0"}

@app.get("/app", response_class=HTMLResponse)
def interface():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(usuario: Usuario):
    dados = {
        "Daily Time Spent on Site": usuario.daily_time_spent_on_site,
        "Age": usuario.age,
        "Area Income": usuario.area_income,
        "Daily Internet Usage": usuario.daily_internet_usage,
        "Male": usuario.male
    }

    df = pd.DataFrame([dados])
    df_final = df.reindex(columns=colunas, fill_value=0)

    predicao = modelo.predict(df_final)[0]
    probabilidade = modelo.predict_proba(df_final)[0]

    return {
        "clicou": int(predicao),
        "resultado": "Alta probabilidade de clique" if predicao == 1 else "Baixa probabilidade de clique",
        "probabilidade_clique": round(float(probabilidade[1]), 4),
        "probabilidade_nao_clique": round(float(probabilidade[0]), 4),
        "modelo": "LogisticRegression"
    }