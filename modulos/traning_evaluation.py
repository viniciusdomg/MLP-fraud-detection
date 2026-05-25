import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import itertools
import numpy as np

# Importando seus módulos locais
import data_loader as dl
import preprocessing as pp
from MLP_network import MLPBinaria
from utils import EarlyStopping

def iniciar_experimentos():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando experimentos no dispositivo: {DEVICE}")

    # ==========================================
    # 1. CARREGAMENTO E LIMPEZA INICIAL
    # ==========================================
    df = dl.carregar_dados_amostra()

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # ==========================================
    # 2. HOLDOUT (Separando 20% para teste final)
    # ==========================================
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, 
        y, 
        test_size=0.20, 
        stratify=y, 
        random_state=dl.SEED
    )

    # K-FOLD

    skf = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=dl.SEED
    )

    # ==========================================
    # 4. LOOP DOS EXPERIMENTOS (9 Combinações)
    # ==========================================
    neuronios_opcoes = [10, 32, 64]
    taxas_opcoes = [0.01, 0.005, 0.001]
    combinacoes = list(itertools.product(neuronios_opcoes, taxas_opcoes))
    
    for neuronios, lr in combinacoes:
        print(f"\n[{neuronios} Neurônios | LR: {lr}]")
        
        fold = 1
        for train_idx, val_idx in skf.split(X_train_full, y_train_full):
            print(f"  -> Treinando Fold {fold}...")

            # Separando dados do fold
            X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            # Pré-processamento
            pipeline = pp.criar_pipeline_pre_processamento()
            X_train_processed = pipeline.fit_transform(X_train)
            X_val_processed = pipeline.transform(X_val)

            # Convertendo para Tensores
            X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

            # DataLoaders (Mini-batch padrão para Fase A)
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

            # ==========================================
            # 5. O SEU MOTOR DE TREINAMENTO AQUI
            # ==========================================
            model = MLPBinaria(input_dim=X_train_processed.shape[1], hidden_neurons=neuronios).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            early_stopper = EarlyStopping(patience=15)

            for epoch in range(300): # Máximo de 300 épocas
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                # Validação
                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE).unsqueeze(1)
                        val_outputs = model(X_val_batch)
                        val_loss_total += criterion(val_outputs, y_val_batch).item()
                
                val_loss_media = val_loss_total / len(val_loader)
                
                # Checagem do Early Stopping
                early_stopper(val_loss_media, model)
                if early_stopper.early_stop:
                    # Ocultando o print no meio dos folds para não poluir a tela demais, 
                    # mas o modelo recupera os melhores pesos silenciosamente
                    model.load_state_dict(early_stopper.best_model_weights)
                    break
            
            # (Aqui depois vamos adicionar a coleta das métricas: F1, Accuracy, etc)
            fold += 1