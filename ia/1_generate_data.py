"""
1_generate_data.py
------------------
Gera massa de dados fake para treinamento de aprovação financeira de transações.
Saída: data/transactions.csv
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_SAMPLES = 10_000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Domínios ---
TRANSACTION_TYPES = ["PIX", "TED", "BOLETO", "CARTAO_DEBITO", "CARTAO_CREDITO"]
MERCHANT_CATEGORIES = ["VAREJO", "ALIMENTACAO", "SAUDE", "EDUCACAO", "VIAGEM", "ELETRONICOS", "SERVICOS", "COMBUSTIVEL"]
ACCOUNT_TIERS = ["BASICO", "PADRAO", "PREMIUM", "PRIME"]
COUNTRIES = ["BR", "US", "AR", "CL", "MX", "PT", "GB", "CN"]

TIER_LIMITS = {
    "BASICO":   {"daily": 500,    "monthly": 5_000},
    "PADRAO":   {"daily": 5_000,  "monthly": 50_000},
    "PREMIUM":  {"daily": 20_000, "monthly": 200_000},
    "PRIME":    {"daily": 100_000,"monthly": 1_000_000},
}

def risk_score(row: dict) -> float:
    """Calcula score de risco 0~1 baseado nas features."""
    score = 0.0

    # Valor vs limite diário
    ratio = row["amount"] / row["daily_limit"]
    score += min(ratio * 0.35, 0.35)

    # Horário incomum (22h-5h)
    if row["hour"] >= 22 or row["hour"] <= 5:
        score += 0.15

    # País estrangeiro
    if row["country"] != "BR":
        score += 0.10
        if row["country"] in ["CN"]:
            score += 0.08  # maior risco

    # Muitas transações recentes
    score += min(row["tx_last_24h"] / 20.0 * 0.15, 0.15)

    # Score de crédito baixo
    score += max(0, (600 - row["credit_score"]) / 600) * 0.20

    # Dispositivo novo
    if row["is_new_device"]:
        score += 0.08

    # Velocidade (tempo desde última tx em segundos - muito rápido é suspeito)
    if row["seconds_since_last_tx"] < 30:
        score += 0.12

    # Conta antiga tem menos risco
    score -= min(row["account_age_days"] / 3650 * 0.08, 0.08)

    return min(max(score, 0.0), 1.0)


def decide(score: float, row: dict) -> tuple[int, str, str]:
    """
    Retorna (label, status, suggestion).
    label: 1=aprovado, 0=negado
    """
    # Limite diário ultrapassado => sempre nega
    if row["amount"] > row["daily_limit"]:
        return 0, "NEGADO", "Valor excede o limite diário da conta"

    # Saldo insuficiente
    if row["amount"] > row["balance"]:
        return 0, "NEGADO", "Saldo insuficiente para realizar a transação"

    if score < 0.30:
        return 1, "APROVADO", ""
    elif score < 0.55:
        # Aprovado com leve risco — às vezes nega por ruído
        label = 1 if random.random() > 0.15 else 0
        if label == 1:
            return 1, "APROVADO", ""
        return 0, "NEGADO", "Transação fora do padrão de comportamento"
    elif score < 0.75:
        label = 1 if random.random() > 0.55 else 0
        if label == 1:
            return 1, "APROVADO_COM_RESTRICAO", "Monitoramento ativado para esta transação"
        return 0, "NEGADO", "Perfil de risco elevado detectado"
    else:
        return 0, "NEGADO", "Transação bloqueada por alto risco de fraude"


records = []
for i in range(N_SAMPLES):
    tier = random.choice(ACCOUNT_TIERS)
    limits = TIER_LIMITS[tier]
    daily_limit = limits["daily"] * (1 + random.uniform(-0.1, 0.1))
    monthly_limit = limits["monthly"]

    balance = random.uniform(0, daily_limit * 3)
    amount = random.expovariate(1 / (daily_limit * 0.4))
    amount = round(min(amount, daily_limit * 2.5), 2)  # permite ultrapassar limite ocasionalmente

    credit_score = int(np.clip(np.random.normal(650, 120), 300, 1000))
    country = random.choices(COUNTRIES, weights=[70,5,5,3,3,3,3,8])[0]

    row = {
        "transaction_id":        f"TXN-{i:07d}",
        "amount":                amount,
        "transaction_type":      random.choice(TRANSACTION_TYPES),
        "merchant_category":     random.choice(MERCHANT_CATEGORIES),
        "country":               country,
        "hour":                  random.randint(0, 23),
        "day_of_week":           random.randint(0, 6),       # 0=Segunda
        "is_weekend":            int(random.randint(0, 6) >= 5),
        "account_tier":          tier,
        "account_age_days":      random.randint(1, 5000),
        "balance":               round(balance, 2),
        "daily_limit":           round(daily_limit, 2),
        "monthly_limit":         round(monthly_limit, 2),
        "credit_score":          credit_score,
        "tx_last_24h":           random.randint(0, 30),
        "tx_last_7d":            random.randint(0, 100),
        "avg_tx_amount_30d":     round(random.uniform(50, daily_limit * 0.5), 2),
        "seconds_since_last_tx": random.randint(1, 86400),
        "is_new_device":         int(random.random() < 0.12),
        "is_new_beneficiary":    int(random.random() < 0.25),
        "failed_attempts_24h":   random.randint(0, 5),
        "device_trust_score":    round(random.uniform(0, 1), 4),
    }

    score = risk_score(row)
    row["risk_score"] = round(score, 4)

    label, status, suggestion = decide(score, row)
    row["label"] = label
    row["status"] = status
    row["suggestion"] = suggestion

    records.append(row)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

approved = df["label"].sum()
denied = len(df) - approved
print(f"✅  Dataset gerado: {OUTPUT_FILE}")
print(f"   Total de amostras : {N_SAMPLES:,}")
print(f"   Aprovadas         : {approved:,} ({approved/N_SAMPLES*100:.1f}%)")
print(f"   Negadas           : {denied:,} ({denied/N_SAMPLES*100:.1f}%)")
print(f"   Colunas           : {list(df.columns)}")
