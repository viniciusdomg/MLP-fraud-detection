import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import glob

import modulos.data_loader as dl
import modulos.preprocessing as pp
from modulos.MLP_network import MLPBinaria
from modulos.utils import EarlyStopping

def pegar_ultimo_csv(prefixo):
    # Recebe o csv mais recente aplicar os gráficos
    arquivos = glob.glob(f"resultados/{prefixo}*.csv")
    if not arquivos:
        return None
    return max(arquivos, key=os.path.getctime)

def gerar_grafico_colunas_f1():
    print("\nGerando Gráfico de Colunas Verticais (F1-Score por Configuração)...")
    
    arquivo_csv = pegar_ultimo_csv("log_20260526_105533")
    if not arquivo_csv:
        print("Erro: Nenhum arquivo de log encontrado na pasta resultados.")
        return
        
    df = pd.read_csv(arquivo_csv)
    
    df['Configuracao'] = df['neuronios'].astype(str) + " N | TAXA: " + df['taxa_aprendizado'].astype(str)
    
    df = df.sort_values(by='f1', ascending=True)
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(df['Configuracao'], df['f1'], color='skyblue', edgecolor='black')
    
    bars[-1].set_color('coral')
    bars[-1].set_edgecolor('black')
    
    plt.ylabel('F1-Score Médio (K-Fold)')
    plt.xlabel('Topologia (Neurônios | Taxa de Aprendizado)')
    plt.title('Comparação de Desempenho entre Configurações da MLP')
    
    plt.ylim(0, 1.0)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    caminho_salvar = "resultados/grafico_colunas_f1.png"
    plt.savefig(caminho_salvar)
    plt.close()
    print(f"-> Salvo em: {caminho_salvar}")


def gerar_boxplots_folds():
    
    caminho_dados_boxplot = "resultados/dados_brutos_boxplots.csv"
    
    # 1. VERIFICA SE OS DADOS JÁ FORAM GERADOS ANTES
    # (Para você não ter que esperar 30 minutos toda vez que quiser ajustar a cor do gráfico)
    if os.path.exists(caminho_dados_boxplot):
        print(f"Dados já encontrados em '{caminho_dados_boxplot}'! Pulando o treinamento e gerando apenas o gráfico...")
        df_box = pd.read_csv(caminho_dados_boxplot)
        
    else:
        print("Dados não encontrados. Iniciando os treinamentos (Aviso: Pode demorar 20~30 minutos. Vá tomar um café!)...")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Suas 9 configurações da Busca em Grade
        neuronios_opcoes = [10, 32, 64]
        taxas_opcoes = [0.1, 0.05, 0.01]
        configuracoes = list(itertools.product(neuronios_opcoes, taxas_opcoes))
        
        MOMENTUM = 0.9
        BATCH_SIZE = 64
        EXECUCOES = 10
        
        df = dl.carregar_dados_amostra()
        df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]

        X_train_full, _, y_train_full, _ = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=dl.SEED
        )

        dados_boxplot = []

        for neuronios, lr in configuracoes:
            print(f"\n[Treinando: {neuronios} Neurônios | LR: {lr}]")
            
            for i in range(EXECUCOES):
                semente_atual = dl.SEED + i
                torch.manual_seed(semente_atual)
                np.random.seed(semente_atual)
                
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente_atual)
                f1_folds = []
                
                for train_idx, val_idx in skf.split(X_train_full, y_train_full):
                    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
                    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

                    pipeline = pp.criar_pipeline_pre_processamento()
                    X_train_processed = pipeline.fit_transform(X_train)
                    X_val_processed = pipeline.transform(X_val)

                    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
                    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

                    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
                    
                    model = MLPBinaria(input_dim=X_train_processed.shape[1], hidden_dim=neuronios).to(DEVICE)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
                    early_stopper = EarlyStopping(parada=15)

                    for epoch in range(150):
                        model.train()
                        for X_batch, y_batch in train_loader:
                            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
                            optimizer.zero_grad()
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        val_loss_total = 0.0
                        with torch.no_grad():
                            for X_val_batch, y_val_batch in DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE):
                                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE).unsqueeze(1)
                                val_loss_total += criterion(model(X_val_batch), y_val_batch).item()
                        
                        early_stopper(val_loss_total, model)
                        if early_stopper.early_stop:
                            model.load_state_dict(early_stopper.best_model_weights)
                            break
                    
                    model.eval()
                    with torch.no_grad():
                        test_predictions = (model(X_val_tensor.to(DEVICE)) > 0).float().cpu().numpy()
                        y_val_true = y_val_tensor.cpu().numpy()
                        
                    f1_folds.append(f1_score(y_val_true, test_predictions, zero_division=0))
                    
                f1_medio_execucao = sum(f1_folds) / 5
                
                dados_boxplot.append({
                    'Neurônios': str(neuronios),
                    'Taxa de Aprendizado': str(lr),
                    'F1-Score': f1_medio_execucao
                })
                print(f"  - Execução {i+1}/10 concluída. F1: {f1_medio_execucao:.4f}")

        df_box = pd.DataFrame(dados_boxplot)
        df_box.to_csv(caminho_dados_boxplot, index=False)
        print(f"\nTodos os treinamentos finalizados! Dados salvos em {caminho_dados_boxplot}")

    plt.figure(figsize=(14, 7))
    
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        sns.boxplot(
            x='Neurônios', 
            y='F1-Score', 
            hue='Taxa de Aprendizado', 
            data=df_box, 
            palette="viridis"
        )
        
    except ImportError:
        print("Por favor, instale o seaborn com 'pip install seaborn' para gerar este gráfico.")
        return
        
    plt.title(f'Variabilidade do F1-Score por Configuração\n(Cada caixa representa a distribuição de 10 execuções independentes)', fontsize=14)
    plt.ylabel('F1-Score Médio (Validação Cruzada 5-Fold)', fontsize=12)
    plt.xlabel('Quantidade de Neurônios na Camada Oculta', fontsize=12)
    plt.legend(title='Taxa de Aprendizado', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    caminho_salvar = "resultados/grafico_boxplot_todas_configuracoes.png"
    plt.savefig(caminho_salvar, dpi=300)
    plt.close()
    print(f"-> GRÁFICO FINAL SALVO EM: {caminho_salvar}")


def gerar_grafico_evolucao_campeao():
    print("\nGerando Gráficos de Evolução (Loss, Accuracy e F1) do Modelo Campeão...")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuração Campeã Absoluta
    NEURONIOS = 10
    taxa_aprendizado = 0.05
    MOMENTUM = 0.9
    BATCH_SIZE = 64
    EPOCHS = 150 

    df = dl.carregar_dados_amostra()
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train_full, _, y_train_full, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=dl.SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=dl.SEED
    )

    pipeline = pp.criar_pipeline_pre_processamento()
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(DEVICE).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(DEVICE).unsqueeze(1)
    
    y_train_true = y_train.values
    y_val_true = y_val.values

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # Modelo e Otimizador
    model = MLPBinaria(input_dim=X_train_processed.shape[1], hidden_dim=NEURONIOS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=taxa_aprendizado, momentum=MOMENTUM)

    hist_train_loss, hist_val_loss = [], []
    hist_train_acc, hist_train_f1 = [], []
    hist_val_acc, hist_val_f1 = [], []

    print("Treinando e coletando métricas a cada época... (Aguarde)")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        
        hist_train_loss.append(train_loss_total / len(train_loader))

        model.eval()
        with torch.no_grad():
            # Predições no Treino
            train_logits = model(X_train_tensor)
            train_preds = (train_logits > 0).float().cpu().numpy()
            hist_train_acc.append(accuracy_score(y_train_true, train_preds))
            hist_train_f1.append(f1_score(y_train_true, train_preds, zero_division=0))

            # Predições na Validação
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
            hist_val_loss.append(val_loss)
            
            val_preds = (val_logits > 0).float().cpu().numpy()
            hist_val_acc.append(accuracy_score(y_val_true, val_preds))
            hist_val_f1.append(f1_score(y_val_true, val_preds, zero_division=0))

    print("Desenhando os gráficos...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Evolução do Erro (Loss)
    ax1.plot(hist_train_loss, label='Treino Loss', color='blue', linewidth=2)
    ax1.plot(hist_val_loss, label='Validação Loss', color='red', linewidth=2)
    ax1.set_title('Evolução do Erro (BCE Loss)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: Evolução de Accuracy e F1
    ax2.plot(hist_train_acc, label='Treino Acc', color='lightblue', linestyle='--')
    ax2.plot(hist_train_f1, label='Treino F1', color='plum', linestyle='--')
    ax2.plot(hist_val_acc, label='Validação Acc', color='blue', linewidth=2)
    ax2.plot(hist_val_f1, label='Validação F1', color='purple', linewidth=2)
    ax2.set_title('Evolução das Métricas (Accuracy e F1-Score)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Pontuação')
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    caminho_salvar = "resultados/grafico_evolucao_campeao.png"
    plt.savefig(caminho_salvar)
    plt.close()
    print(f"-> Salvo em: {caminho_salvar}")