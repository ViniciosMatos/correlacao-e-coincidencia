# Parte 1 — Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Parte 2 — Carregando o dataset
wine = load_wine(as_frame=True)
dados = wine.frame

print("--- Primeiras linhas do dataset ---")
# print(dados.head())
print("\n")

# Parte 3 — Conhecendo os dados
print("--- Dimensões do dataset (linhas, colunas) ---")
# print(dados.shape)
print("\n--- Nomes das colunas ---")
print(dados.columns)
print("\n")

# Parte 4 — Explorando estatísticas básicas
print("--- Estatísticas básicas ---")
pd.set_option('display.max_columns', None)
print(dados.describe())
print("\n")

# Parte 6 — Calculando a correlação de Pearson
variavel_x = "magnesium"
variavel_y = "alcohol"

correlacao = dados[variavel_x].corr(
    dados[variavel_y],
    method="pearson"
)

print(f"--- Correlação 1: {variavel_x} x {variavel_y} ---")
print("Correlação de Pearson:", correlacao)
print("Correlação (arredondada):", round(correlacao, 3))

# Parte 8 — Classificando a força da correlação
forca = abs(correlacao)

if forca < 0.30:
    print("Força: Correlação fraca")
elif forca < 0.70:
    print("Força: Correlação moderada")
else:
    print("Força: Correlação forte")
print("\n")

# Parte 9 — Construindo um gráfico de dispersão
plt.figure(figsize=(8, 5))
plt.scatter(
    dados[variavel_x],
    dados[variavel_y]
)
plt.xlabel(variavel_x)
plt.ylabel(variavel_y)
plt.title(f"{variavel_x} x {variavel_y}")
plt.grid(alpha=0.3)
plt.show()
#
# # Parte 13 — Procurando outra correlação (Variáveis de exemplo preenchidas)
variavel_x2 = "flavanoids"
variavel_y2 = "total_phenols"

correlacao2 = dados[variavel_x2].corr(
    dados[variavel_y2]
)

print(f"--- Correlação 2: {variavel_x2} x {variavel_y2} ---")
print("Correlação:", round(correlacao2, 3))

# Gráfico da segunda correlação
plt.figure(figsize=(8, 5))
plt.scatter(
    dados[variavel_x2],
    dados[variavel_y2]
)
plt.xlabel(variavel_x2)
plt.ylabel(variavel_y2)
plt.title(f"{variavel_x2} x {variavel_y2}")
plt.grid(alpha=0.3)
plt.show()
print("\n")

# Parte 15 — Desafio: procurando correlações automaticamente
dados_numericos = dados.drop(
    columns=["target"]
)

matriz_correlacao = dados_numericos.corr()

print("--- Matriz de Correlação ---")
print(matriz_correlacao)