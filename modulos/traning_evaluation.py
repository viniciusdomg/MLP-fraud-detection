from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score;

import itertools
import os
import time

import pandas as pd
import numpy as np

# Módulos
import modulos.data_loader as dl
import modulos.preprocessing as pp
from modulos.MLP_network import MLPBinaria
from modulos.utils import EarlyStopping

def teste_final():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print("INICIANDO FASE B: COMPARAÇÃO DE BATCHES E TESTE FINAL")
    print("="*50)

    # 1. Carregamento e Separação (Igual ao anterior)
    df = dl.carregar_dados_amostra()
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=dl.SEED
    )

    # 2. Pipeline Final (Treina com os 80% inteiros, aplica nos 20% do cofre)
    pipeline = pp.criar_pipeline_pre_processamento()
    X_train_processed = pipeline.fit_transform(X_train_full)
    X_test_processed = pipeline.transform(X_test)

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # 3. Configuração
    NEURONIOS = 10
    LR = 0.05
    MOMENTUM = 0.9
    EPOCHS = 150

    # 4. Estratégias de gradiente
    estrategias_batch = {
        "Batch (Full)": len(X_train_tensor),
        "Mini-batch (64)": 64,
        "Mini-batch (32)": 32,
        "Stochastic (SGD Puro)": 1
    }

    resultados_batches = []
    historico_losses = {} # Para plotar o gráfico depois

    for nome_estrategia, tamanho_batch in estrategias_batch.items():
        print(f"\n-> Testando Estratégia: {nome_estrategia} (Lote: {tamanho_batch})")
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=tamanho_batch, shuffle=True)
        
        model = MLPBinaria(input_dim=X_train_processed.shape[1], hidden_dim=NEURONIOS).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        
        inicio_tempo = time.time()
        losses_desta_estrategia = []

        model.train()
        for epoch in range(EPOCHS):
            train_loss_total = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item()
            
            # Guarda o loss médio da época para o gráfico
            losses_desta_estrategia.append(train_loss_total / len(train_loader))

        fim_tempo = time.time()
        historico_losses[nome_estrategia] = losses_desta_estrategia
        
        # Teste Final no Holdout (O Cofre)
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor.to(DEVICE))
            test_predictions = (test_logits > 0).float().cpu().numpy()
            y_test_true = y_test_tensor.cpu().numpy()

        acc = accuracy_score(y_test_true, test_predictions)
        f1 = f1_score(y_test_true, test_predictions, zero_division=0)
        
        tempo_gasto = fim_tempo - inicio_tempo
        print(f"   Tempo: {tempo_gasto:.2f}s | F1-Score: {f1:.4f} | Acurácia: {acc:.4f}")
        
        resultados_batches.append({
            'Estrategia': nome_estrategia,
            'Tempo_s': tempo_gasto,
            'F1_Score': f1,
            'Accuracy': acc
        })

        # Salva a matriz de confusão da melhor estratégia esperada (Mini-batch)
        if tamanho_batch == 32:
            matriz_confusao = confusion_matrix(y_test_true, test_predictions)
            print("\nMatriz de Confusão (Mini-batch 32):")
            print(matriz_confusao)

    # 5. Exportar Resultados dos Batches para CSV
    df_batches = pd.DataFrame(resultados_batches)
    df_batches.to_csv("resultados/comparacao_batches.csv", index=False)
    print("\nResultados das estratégias salvos em 'resultados/comparacao_batches.csv'")

    # 6. Gerar o Gráfico de Convergência das Estratégias
    plt.figure(figsize=(10, 6))
    for nome_estrategia, losses in historico_losses.items():
        plt.plot(losses, label=nome_estrategia)
    
    plt.title("Evolução do Erro (Loss) por Estratégia de Gradiente")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (Entropia Cruzada Binária)")
    plt.legend()
    plt.grid(True)
    plt.savefig("resultados/grafico_estrategias_batch.png")
    print("Gráfico de convergência salvo em 'resultados/grafico_estrategias_batch.png'")

def iniciar_experimentos():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando experimentos no dispositivo: {DEVICE}")

    # CARREGAMENTO
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
    
    neuronios_opcoes = [10, 64, 128]
    taxas_opcoes = [0.001, 0.05, 0.1]
    combinacoes = list(itertools.product(neuronios_opcoes, taxas_opcoes))

    resultados_finais = []

    for neuronios, taxa in combinacoes:
        print(f"\n[{neuronios} Neurônios | Taxa aprendizado: {taxa}]")

        metricas_folds = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'epocas_convergencia': []}
        
        inicio_tempo = time.time()

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

            # TREINAMENTO
            model = MLPBinaria(input_dim=X_train_processed.shape[1], hidden_dim=neuronios).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD(model.parameters(), lr=taxa, momentum=0.9)
            early_stopper = EarlyStopping(patience=15)

            epoca_parada = 300

            for epoch in range(300):
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
                    epoca_parada = epoch + 1
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
            metricas_folds['epocas_convergencia'].append(epoca_parada)
            
            print(f"  -> Fold {fold} Finalizado na época {epoca_parada} | F1-Score: {f1:.4f} | Acurácia: {acc:.4f}")
            fold += 1
        
        fim_tempo = time.time()
        tempo_total = fim_tempo - inicio_tempo

        f1_medio = np.mean(metricas_folds['f1'])
        acc_media = np.mean(metricas_folds['acc'])
        prec_media = np.mean(metricas_folds['prec'])
        rec_medio = np.mean(metricas_folds['rec'])
        epocas_media = np.mean(metricas_folds['epocas_convergencia'])
        
        print(f"\nRESULTADO MÉDIO DA CONFIGURAÇÃO [{neuronios} Neurônios | Taxa aprendizado: {taxa}]:")
        print(f"F1-Score Médio: {f1_medio:.4f}")
        print(f"Acurácia Média: {acc_media:.4f}")
        print(f"Épocas médias até convergência: {epocas_media:.1f}")
        print(f"Tempo Total (5 Folds): {tempo_total:.2f} segundos")

        # Guarda no placar geral para depois acharmos o campeão
        resultados_finais.append({
            'neuronios': neuronios,
            'taxa_aprendizado': taxa,
            'f1': f1_medio,
            'acc': acc_media,
            'prec': prec_media,
            'rec': rec_medio,
            'epocas_convergencia': epocas_media,
            'tempo_total': tempo_total
        })
    
    resultados_finais.sort(key=lambda x: x['f1'], reverse=True)
    melhor_config = resultados_finais[0]

    print("\nSalvando log de experimentos...")
    # Verifica se existe a pasta
    os.makedirs("resultados", exist_ok=True) 
    # Cria arquivos com data e hora atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_csv = f"resultados/log_{timestamp}.csv"

    df_resultados = pd.DataFrame(resultados_finais)
    df_resultados.to_csv(caminho_csv, index=False)
    
    print("\n")
    print("BUSCA EM GRADE CONCLUÍDA!")
    print(f"MELHOR CONFIGURAÇÃO: {melhor_config['neuronios']} Neurônios com Taxa de Aprendizado de {melhor_config['taxa_aprendizado']}")
    print(f"F1-Score: {melhor_config['f1']:.4f}")
    print(f"Acurácia: {melhor_config['acc']:.4f}")
    print(f"Precisão: {melhor_config['prec']:.4f}")
    print(f"Revocação: {melhor_config['rec']:.4f}")
    print(f"Épocas médias até convergência: {melhor_config['epocas_convergencia']:.1f}")
    print(f"Tempo Total: {melhor_config['tempo_total']:.2f} segundos")