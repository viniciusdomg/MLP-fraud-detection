import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score;

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

    # HOLDOUT - não usa o conjunto de teste para escolha de parâmetro
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

    # LOOP DOS EXPERIMENTOS (9 Combinações)
    
    neuronios_opcoes = [10, 32, 64]
    taxas_opcoes = [0.01, 0.005, 0.001]
    combinacoes = list(itertools.product(neuronios_opcoes, taxas_opcoes))

    resultados_finais = []

    for neuronios, lr in combinacoes:
        print(f"\n[{neuronios} Neurônios | LR: {lr}]")

        metricas_folds = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        
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
                    # modelo recupera os melhores pesos
                    model.load_state_dict(early_stopper.best_model_weights)
                    break
            
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor.to(DEVICE))
                
                # Logit > 0 significa que a rede acha que é Fraude (1)
                val_predictions = (val_logits > 0).float().cpu().numpy()
                y_val_true = y_val_tensor.cpu().numpy()
            
            acc = accuracy_score(y_val_true, val_predictions)
            prec = precision_score(y_val_true, val_predictions, zero_division=0)
            rec = recall_score(y_val_true, val_predictions, zero_division=0)
            f1 = f1_score(y_val_true, val_predictions, zero_division=0)

            metricas_folds['acc'].append(acc)
            metricas_folds['prec'].append(prec)
            metricas_folds['rec'].append(rec)
            metricas_folds['f1'].append(f1)
            
            print(f"  -> Fold {fold} Finalizado | F1-Score: {f1:.4f} | Acurácia: {acc:.4f}")
            # (Aqui depois vamos adicionar a coleta das métricas: F1, Accuracy, etc)
            fold += 1
        
        f1_medio = np.mean(metricas_folds['f1'])
        acc_media = np.mean(metricas_folds['acc'])
        prec_media = np.mean(metricas_folds['prec'])
        rec_medio = np.mean(metricas_folds['rec'])
        
        print(f"\nRESULTADO MÉDIO DA CONFIGURAÇÃO [{neuronios} Neurônios | LR: {lr}]:")
        print(f"F1-Score Médio: {f1_medio:.4f}")
        print(f"Acurácia Média: {acc_media:.4f}")
        
        # Guarda no placar geral para depois acharmos o campeão
        resultados_finais.append({
            'neuronios': neuronios,
            'lr': lr,
            'f1': f1_medio,
            'acc': acc_media,
            'prec': prec_media,
            'rec': rec_medio
        })
    
    resultados_finais.sort(key=lambda x: x['f1'], reverse=True)
    melhor_config = resultados_finais[0]
    
    print("\n" + "="*50)
    print("BUSCA EM GRADE CONCLUÍDA!")
    print(f"MELHOR CONFIGURAÇÃO: {melhor_config['neuronios']} Neurônios com LR de {melhor_config['lr']}")
    print(f"F1-Score: {melhor_config['f1']:.4f}")
    print("="*50)