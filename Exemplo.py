"""
===========================================================
EXEMPLO MÍNIMO
MLP + PYTORCH + STRATIFIED 5-FOLD CV
===========================================================

Destaques:
1. pipeline de pré-processamento
2. geração do modelo
3. treinamento com 5-fold stratified CV
4. métricas
5. treinamento final
6. monitoramento de overfitting
7. gráfico de convergência

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# CONFIGURAÇÕES
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("DEVICE:", DEVICE)

# =========================================================
# DATASET EXEMPLO
# =========================================================

df = pd.DataFrame({

    "idade": [
        20, 25, 30, 35, 40,
        45, 50, 28, 32, 38,
        42, 48, 22, 27, 31,
        36, 41, 46, 29, 33
    ],

    "salario": [
        2000, 2500, 3000, 3500, 4000,
        4500, 5000, 2700, 3200, 3800,
        4200, 4800, 2200, 2600, 3100,
        3600, 4100, 4700, 2900, 3400
    ],

    "cidade": [
        "Natal", "Recife", "Natal",
        "Fortaleza", "Natal",
        "Recife", "Fortaleza",
        "Natal", "Recife",
        "Fortaleza", "Natal",
        "Recife", "Natal",
        "Fortaleza", "Natal",
        "Recife", "Fortaleza",
        "Natal", "Recife",
        "Fortaleza"
    ],

    "comprou": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    
    # classes inconsistentes
    # "comprou":   [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
})

# =========================================================
# X E y
# =========================================================

X = df.drop("comprou", axis=1)
y = df["comprou"]

# =========================================================
# HOLDOUT TESTE FINAL
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# =========================================================
# STRATIFIED 5-FOLD CV
# =========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =========================================================
# MODELO
# =========================================================

class MLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(64, 32)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.output = nn.Linear(32, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x = self.output(x)

        return x

# =========================================================
# MÉTRICAS DOS FOLDS
# =========================================================

accuracies = []
precisions = []
recalls = []
f1s = []

# =========================================================
# CROSS VALIDATION
# =========================================================

fold = 1

for train_idx, val_idx in skf.split(X_train_full, y_train_full):

    print("\n==============================")
    print(f"FOLD {fold}")
    print("==============================")

    # =====================================================
    # DADOS DO FOLD
    # =====================================================

    X_train = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]

    y_train = y_train_full.iloc[train_idx]
    y_val = y_train_full.iloc[val_idx]

    # =====================================================
    # PIPELINE DE PRÉ-PROCESSAMENTO
    # =====================================================

    atributos_numericos = ["idade", "salario"]
    atributos_categoricos = ["cidade"]

    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    pipeline_categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessador = ColumnTransformer([
        ("num", pipeline_numerico, atributos_numericos),
        ("cat", pipeline_categorico, atributos_categoricos)
    ])

    # =====================================================
    # FIT APENAS NO TREINO
    # =====================================================

    X_train = preprocessador.fit_transform(X_train)
    X_val = preprocessador.transform(X_val)

    # =====================================================
    # sparse -> dense
    # =====================================================

    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val

    # =====================================================
    # TENSORES
    # =====================================================

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    # =====================================================
    # DATALOADER
    # =====================================================

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )

    # =====================================================
    # MODELO
    # =====================================================

    input_dim = X_train.shape[1]
    model = MLP(input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # =====================================================
    # TREINAMENTO
    # =====================================================

    epochs = 500

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # =====================================================
    # PREDIÇÕES
    # =====================================================

    model.eval()

    with torch.no_grad():

        logits = model(X_val_tensor.to(DEVICE))

        predictions = torch.argmax(logits, dim=1)

        predictions = predictions.cpu().numpy()

    # =====================================================
    # MÉTRICAS
    # =====================================================

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    fold += 1

# =========================================================
# MÉDIA FINAL DOS FOLDS
# =========================================================

print("\n==============================")
print("MÉDIA DOS 5 FOLDS")
print("==============================")

print(f"Accuracy : {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall   : {np.mean(recalls):.4f}")
print(f"F1       : {np.mean(f1s):.4f}")

# =========================================================
# TREINAMENTO FINAL
# =========================================================

print("\n==============================")
print("TREINAMENTO FINAL")
print("==============================")

# =========================================================
# SPLIT TREINO / VALIDAÇÃO
# =========================================================
# usado apenas para monitorar overfitting

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

# =========================================================
# PIPELINE FINAL
# =========================================================

atributos_numericos = ["idade", "salario"]
atributos_categoricos = ["cidade"]

pipeline_numerico = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

pipeline_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessador = ColumnTransformer([
    ("num", pipeline_numerico, atributos_numericos),
    ("cat", pipeline_categorico, atributos_categoricos)
])

# =========================================================
# FIT APENAS NO TREINO FINAL
# =========================================================

X_train_final = preprocessador.fit_transform(X_train_final)
X_val_final = preprocessador.transform(X_val_final)
X_test_final = preprocessador.transform(X_test)

# =========================================================
# sparse -> dense
# =========================================================

X_train_final = X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final
X_val_final = X_val_final.toarray() if hasattr(X_val_final, "toarray") else X_val_final
X_test_final = X_test_final.toarray() if hasattr(X_test_final, "toarray") else X_test_final

# =========================================================
# TENSORES
# =========================================================

X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_final.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# =========================================================
# DATALOADER
# =========================================================

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    # batch_size=1
    batch_size=len(X_train_tensor),  # batch size igual ao tamanho do treino para simular full batch
    shuffle=True
)

# =========================================================
# MODELO FINAL
# =========================================================

input_dim = X_train_final.shape[1]
model = MLP(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# =========================================================
# HISTÓRICO
# =========================================================

train_losses = []
val_losses = []

# =========================================================
# TREINAMENTO FINAL
# =========================================================

epochs = 250

for epoch in range(epochs):

    # =====================================================
    # TREINO
    # =====================================================

    model.train()
    train_loss_total = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()

    # =====================================================
    # LOSS TREINO
    # =====================================================

    train_loss = train_loss_total / len(train_loader)
    train_losses.append(train_loss)

    # =====================================================
    # VALIDAÇÃO
    # =====================================================

    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor.to(DEVICE))
        val_loss = criterion(outputs, y_val_tensor.to(DEVICE))
        val_loss = val_loss.item()
        val_losses.append(val_loss)

    # =====================================================
    # LOG
    # =====================================================

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )

# =========================================================
# TESTE FINAL
# =========================================================

model.eval()
with torch.no_grad():
    logits = model(X_test_tensor.to(DEVICE))
    predictions = torch.argmax(logits, dim=1)
    predictions = predictions.cpu().numpy()

# =========================================================
# MÉTRICAS FINAIS
# =========================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n==============================")
print("TESTE FINAL")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")

# =========================================================
# GRÁFICO FINAL
# =========================================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Convergência e Overfitting")
plt.legend()
plt.grid()
plt.show()

