import pandas as pd
import os

MATRICULA = 20230013651
SEED = (MATRICULA % 32) - 1 

ARQUIVO_ORIGINAL = "paysim.csv"
ARQUIVO_AMOSTRA = "paysim_sample.csv"

def gerar_amostra_paysim():
    print(f"Iniciando amostragem com semente (seed): {SEED}...")
    
    # Verifica se o arquivo original existe no diretório atual
    if not os.path.exists(ARQUIVO_ORIGINAL):
        raise FileNotFoundError(f"Erro: O arquivo '{ARQUIVO_ORIGINAL}' não foi encontrado no diretório.")

    # Carrega a base completa
    df = pd.read_csv(ARQUIVO_ORIGINAL)

    # Extração balanceada conforme o escopo do projeto
    fraudes = df[df["isFraud"] == 1].sample(n=500, random_state=SEED)
    normais = df[df["isFraud"] == 0].sample(n=2500, random_state=SEED)

    # Concatenação e embaralhamento (shuffle)
    amostra = pd.concat([fraudes, normais])
    amostra = amostra.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Exportação para o novo arquivo CSV
    amostra.to_csv(ARQUIVO_AMOSTRA, index=False)
    
    print(f"Amostra gerada com sucesso! Arquivo salvo como: '{ARQUIVO_AMOSTRA}'.")
    print(f"Total de instâncias: {len(amostra)} (500 fraudes, 2500 normais).")

def carregar_dados_amostra():
    # Verifica se a amostra já existe.
    if not os.path.exists(ARQUIVO_AMOSTRA):
        gerar_amostra_paysim()
    
    return pd.read_csv(ARQUIVO_AMOSTRA)