"""
2_train_export.py
-----------------
Treina um modelo de aprovação financeira com PyTorch e exporta para ONNX.
Saída:
  model/financial_approval.onnx
  model/feature_dict.json
  model/training_report.txt
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import onnx
import onnxruntime as ort

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_FILE  = "data/transactions.csv"
MODEL_DIR  = "model"
ONNX_FILE  = os.path.join(MODEL_DIR, "financial_approval.onnx")
DICT_FILE  = os.path.join(MODEL_DIR, "feature_dict.json")
REPORT_FILE = os.path.join(MODEL_DIR, "training_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
print("📂  Carregando dados...")
df = pd.read_csv(DATA_FILE)

# Colunas que não entram como feature
DROP_COLS = ["transaction_id", "status", "suggestion", "label", "risk_score"]

CATEGORICAL_COLS = ["transaction_type", "merchant_category", "country", "account_tier"]
NUMERIC_COLS = [
    "amount", "hour", "day_of_week", "is_weekend",
    "account_age_days", "balance", "daily_limit", "monthly_limit",
    "credit_score", "tx_last_24h", "tx_last_7d", "avg_tx_amount_30d",
    "seconds_since_last_tx", "is_new_device", "is_new_beneficiary",
    "failed_attempts_24h", "device_trust_score",
]

# Encoders para categóricas
label_encoders: dict[str, LabelEncoder] = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

ENCODED_COLS = [c + "_enc" for c in CATEGORICAL_COLS]
FEATURE_COLS = NUMERIC_COLS + ENCODED_COLS

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Scaler nas numéricas (colunas 0..len(NUMERIC_COLS))
scaler = StandardScaler()
X[:, :len(NUMERIC_COLS)] = scaler.fit_transform(X[:, :len(NUMERIC_COLS)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"   Features: {len(FEATURE_COLS)}")

# ─────────────────────────────────────────────
# 2. MODELO PYTORCH
# ─────────────────────────────────────────────
INPUT_DIM = len(FEATURE_COLS)

class FinancialApprovalNet(nn.Module):
    """
    Rede feedforward com BatchNorm e Dropout para classificação binária.
    Saída: logit único (usa BCEWithLogitsLoss no treino, sigmoid na inferência).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),   # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️   Dispositivo: {device}")

model = FinancialApprovalNet(INPUT_DIM).to(device)

# Peso para balancear classes (mais negados que aprovados em alguns cenários)
pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# DataLoaders
train_ds = torch.utils.data.TensorDataset(
    torch.tensor(X_train), torch.tensor(y_train)
)
test_ds = torch.utils.data.TensorDataset(
    torch.tensor(X_test), torch.tensor(y_test)
)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=512)

# ─────────────────────────────────────────────
# 3. TREINAMENTO
# ─────────────────────────────────────────────
EPOCHS = 60
THRESHOLD = 0.5

print(f"\n🚀  Treinando por {EPOCHS} epochs...\n")
best_auc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    scheduler.step()

    # --- Eval ---
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(yb)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    auc = roc_auc_score(all_labels, probs)

    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        acc = (preds == all_labels).mean()
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  acc={acc:.4f}  AUC={auc:.4f}")

# Restaura melhor modelo
model.load_state_dict(best_state)
model.eval()

# Relatório final
all_logits, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_logits.append(model(xb.to(device)).cpu())
        all_labels.append(yb)

all_logits = torch.cat(all_logits).numpy()
all_labels = torch.cat(all_labels).numpy()
probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
preds = (probs >= THRESHOLD).astype(int)

report_str = (
    f"AUC-ROC: {roc_auc_score(all_labels, probs):.4f}\n\n"
    + classification_report(all_labels, preds, target_names=["NEGADO", "APROVADO"])
)
print(f"\n📊  Relatório de classificação (test set):\n{report_str}")

with open(REPORT_FILE, "w") as f:
    f.write(report_str)

# ─────────────────────────────────────────────
# 4. EXPORTAÇÃO ONNX
# ─────────────────────────────────────────────
print("\n📦  Exportando para ONNX...")

# Wrapper que aplica sigmoid internamente — a app consumidora recebe probabilidade direta
class FinancialApprovalNetInference(nn.Module):
    def __init__(self, base: FinancialApprovalNet):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x))   # output: [batch] probabilidade de APROVADO

inf_model = FinancialApprovalNetInference(model).to(device)
inf_model.eval()

dummy_input = torch.zeros(1, INPUT_DIM).to(device)

torch.onnx.export(
    inf_model,
    dummy_input,
    ONNX_FILE,
    input_names=["features"],
    output_names=["approval_probability"],
    dynamic_axes={
        "features":              {0: "batch_size"},
        "approval_probability":  {0: "batch_size"},
    },
    opset_version=14,
    dynamo=False,           # usa exporter legado (TorchScript)
    external_data=False,    # força pesos inline — gera .onnx único sem .data
)

# Verifica o modelo
onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
print(f"   ✅  ONNX válido → {ONNX_FILE}")

# ─────────────────────────────────────────────
# 5. DICIONÁRIO DE FEATURES (feature_dict.json)
# ─────────────────────────────────────────────
feature_dict = {
    "version": "1.0.0",
    "model_file": "financial_approval.onnx",
    "threshold": THRESHOLD,
    "input_name": "features",
    "output_name": "approval_probability",
    "feature_order": FEATURE_COLS,
    "numeric_features": NUMERIC_COLS,
    "categorical_features": CATEGORICAL_COLS,
    "scaler": {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_numeric_features": len(NUMERIC_COLS),
    },
    "label_encoders": {
        col: {
            "classes": le.classes_.tolist(),
            "mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        }
        for col, le in label_encoders.items()
    },
    "labels": {
        "0": {"status": "NEGADO",               "color": "red"},
        "1": {"status": "APROVADO",              "color": "green"},
    },
    "suggestion_rules": [
        {"condition": "amount > daily_limit",      "suggestion": "Valor excede o limite diário da conta"},
        {"condition": "amount > balance",           "suggestion": "Saldo insuficiente para realizar a transação"},
        {"condition": "probability < 0.30",         "suggestion": "Perfil de risco elevado detectado"},
        {"condition": "is_new_device == 1",         "suggestion": "Transação de dispositivo não reconhecido — confirme sua identidade"},
        {"condition": "failed_attempts_24h >= 3",   "suggestion": "Múltiplas tentativas falhas detectadas — conta em observação"},
        {"condition": "country not in BR",          "suggestion": "Transação internacional — verifique se você iniciou esta operação"},
    ],
}

with open(DICT_FILE, "w", encoding="utf-8") as f:
    json.dump(feature_dict, f, ensure_ascii=False, indent=2)

print(f"   ✅  Dicionário de features → {DICT_FILE}")

# ─────────────────────────────────────────────
# 6. SMOKE TEST com ONNX Runtime
# ─────────────────────────────────────────────
print("\n🔍  Smoke test ONNX Runtime...")
session = ort.InferenceSession(ONNX_FILE)
test_input = X_test[:5].astype(np.float32)
onnx_out = session.run(["approval_probability"], {"features": test_input})[0]
print(f"   Probabilidades (5 amostras): {np.round(onnx_out, 4)}")
print(f"   Labels reais               : {y_test[:5].astype(int)}")

print(f"\n✅  Artefatos salvos em ./{MODEL_DIR}/")
print(f"   - financial_approval.onnx")
print(f"   - feature_dict.json")
print(f"   - training_report.txt")
