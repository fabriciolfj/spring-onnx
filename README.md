# spring-onnx — Aprovação de Transações Financeiras com IA

Aplicação **Spring Boot** que realiza aprovação de transações financeiras em tempo real utilizando um modelo de Machine Learning exportado no formato **ONNX**. O modelo é carregado e executado diretamente na JVM via **Deep Java Library (DJL)**, sem dependência de serviços externos de inferência.

---

## Visão Geral

```
┌──────────────┐    POST /api/v1/transactions    ┌──────────────────────────┐
│   Client     │ ──────────────────────────────► │  TransactionController   │
└──────────────┘                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │  TransactionTransformer  │
                                                  │  (pré/pós-processamento) │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │  financial_approval.onnx │
                                                  │  (DJL + ONNX Runtime)    │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │     ApprovalResponse     │
                                                  │  aprovado | negado | ...  │
                                                  └──────────────────────────┘
```

O fluxo completo envolve duas etapas distintas:

1. **Treinamento (Python / `ia/`)** — scripts que geram dados sintéticos, treinam uma rede neural com PyTorch e exportam o modelo para ONNX junto com um dicionário de features (`feature_dict.json`).
2. **Inferência (Java / Spring Boot)** — o modelo ONNX e o dicionário são embutidos no classpath da aplicação; a cada requisição o `TransactionTransformer` normaliza as features e o DJL executa a inferência diretamente na JVM.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem (app) | Java 25 |
| Framework | Spring Boot 4.0.6 (Web MVC) |
| Inferência ML | Deep Java Library (DJL) 0.36.0 + ONNX Runtime Engine |
| Documentação API | SpringDoc OpenAPI 3.0.2 (Swagger UI) |
| Boilerplate | Lombok |
| Linguagem (treino) | Python 3.x |
| Framework ML | PyTorch + scikit-learn |
| Formato do modelo | ONNX opset 14 |
| Build | Gradle 8+ |

---

## Estrutura do Projeto

```
spring-onnx/
├── ia/                                 # Scripts Python para geração e treino
│   ├── 1_generate_data.py              # Gera 10.000 transações sintéticas → data/transactions.csv
│   ├── 2_train_export.py               # Treina rede neural e exporta → model/financial_approval.onnx
│   ├── feature_dict.json               # Dicionário de features gerado pelo treino
│   └── financial_approval.onnx         # Modelo ONNX gerado pelo treino
│
└── src/main/
    ├── java/com/github/fabriciolfj/transactions/
    │   ├── TransactionsApplication.java
    │   ├── configurations/
    │   │   ├── FeatureDictionary.java   # Record que desserializa o feature_dict.json
    │   │   ├── ModelConfiguration.java  # Beans DJL: Criteria, ZooModel, Predictor
    │   │   └── TransactionTransformer.java # NoBatchifyTranslator: pré/pós-processamento
    │   ├── controller/
    │   │   └── TransactionController.java  # POST /api/v1/transactions
    │   └── dtos/
    │       ├── TransactionRequest.java  # Payload de entrada (21 features)
    │       ├── ApprovalResponse.java    # Payload de saída
    │       ├── ApprovalStatus.java      # APROVADO | APROVADO_COM_RESTRICAO | NEGADO
    │       ├── RiskLevel.java           # LOW | MEDIUM | HIGH | CRITICAL
    │       ├── TransactionType.java     # PIX | TED | BOLETO | CARTAO_DEBITO | CARTAO_CREDITO
    │       ├── MerchantCategory.java    # ALIMENTACAO | COMBUSTIVEL | ... (8 categorias)
    │       └── AccountTier.java         # BASICO | PADRAO | PREMIUM | PRIME
    └── resources/
        ├── application.yaml
        ├── feature_dict.json           # Dicionário embutido no classpath
        └── financial_approval.onnx     # Modelo embutido no classpath
```

---

## O Modelo de ML

### Arquitetura da Rede Neural

Rede feedforward com **BatchNorm** e **Dropout** para classificação binária:

```
Input (21 features)
    └─► Linear(21→128) → BatchNorm → ReLU → Dropout(0.3)
    └─► Linear(128→64) → BatchNorm → ReLU → Dropout(0.2)
    └─► Linear(64→32)  → BatchNorm → ReLU
    └─► Linear(32→1)   → Sigmoid   → P(aprovado)
```

### Features de Entrada (21 no total)

**Numéricas (17)** — normalizadas com `StandardScaler`:

| # | Feature | Descrição |
|---|---|---|
| 0 | `amount` | Valor da transação (R$) |
| 1 | `hour` | Hora do dia (0–23) |
| 2 | `dayOfWeek` | Dia da semana (0=Seg, 6=Dom) |
| 3 | `isWeekend` | Fim de semana (bool) |
| 4 | `accountAgeDays` | Idade da conta em dias |
| 5 | `balance` | Saldo disponível |
| 6 | `dailyLimit` | Limite diário |
| 7 | `monthlyLimit` | Limite mensal |
| 8 | `creditScore` | Score de crédito (300–1000) |
| 9 | `txLast24h` | Qtd. transações nas últimas 24h |
| 10 | `txLast7d` | Qtd. transações nos últimos 7 dias |
| 11 | `avgTxAmount30d` | Valor médio das transações nos últimos 30 dias |
| 12 | `secondsSinceLastTx` | Segundos desde a última transação |
| 13 | `isNewDevice` | Dispositivo desconhecido (bool) |
| 14 | `isNewBeneficiary` | Beneficiário novo (bool) |
| 15 | `failedAttempts24h` | Tentativas falhas nas últimas 24h |
| 16 | `deviceTrustScore` | Score de confiança do dispositivo (0.0–1.0) |

**Categóricas (4)** — codificadas com `LabelEncoder`:

| # | Feature | Valores |
|---|---|---|
| 17 | `transactionType` | `PIX`, `TED`, `BOLETO`, `CARTAO_DEBITO`, `CARTAO_CREDITO` |
| 18 | `merchantCategory` | `ALIMENTACAO`, `COMBUSTIVEL`, `EDUCACAO`, `ELETRONICOS`, `SAUDE`, `SERVICOS`, `VAREJO`, `VIAGEM` |
| 19 | `country` | Código ISO 2 letras (ex: `BR`, `US`, `AR`) |
| 20 | `accountTier` | `BASICO`, `PADRAO`, `PREMIUM`, `PRIME` |

### Treinamento

| Parâmetro | Valor |
|---|---|
| Dataset | 10.000 amostras sintéticas |
| Split treino/teste | 80% / 20% |
| Epochs | 60 |
| Batch size | 256 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | BCEWithLogitsLoss com pos_weight balanceado |
| Threshold de aprovação | 0.50 |

---

## Pré-requisitos

- **Java 25+**
- **Gradle 8+** (ou use o wrapper `./gradlew`)
- **Python 3.9+** com `torch`, `scikit-learn`, `onnx`, `onnxruntime`, `pandas`, `numpy` (apenas para re-treinar o modelo)

---

## Como Executar

### 1. Clonar e rodar a aplicação

```bash
git clone https://github.com/fabriciolfj/spring-onnx.git
cd spring-onnx
./gradlew bootRun
```

A aplicação sobe na porta **8080**.

### 2. (Opcional) Re-treinar o modelo

```bash
cd ia
pip install torch scikit-learn onnx onnxruntime pandas numpy

python 1_generate_data.py   # Gera data/transactions.csv
python 2_train_export.py    # Treina e exporta model/financial_approval.onnx e feature_dict.json
```

Após o treino, copie os artefatos gerados para o classpath da aplicação:

```bash
cp ia/model/financial_approval.onnx src/main/resources/
cp ia/model/feature_dict.json       src/main/resources/
```

---

## API

### `POST /api/v1/transactions`

Avalia uma transação e retorna a decisão de aprovação.

**Request Body:**

```json
{
  "amount": 200.00,
  "transactionType": "PIX",
  "merchantCategory": "ALIMENTACAO",
  "country": "BR",
  "hour": 14,
  "dayOfWeek": 2,
  "isWeekend": false,
  "accountTier": "PADRAO",
  "accountAgeDays": 365,
  "balance": 1500.00,
  "dailyLimit": 5000.00,
  "monthlyLimit": 50000.00,
  "creditScore": 720,
  "txLast24h": 2,
  "txLast7d": 10,
  "avgTxAmount30d": 250.00,
  "secondsSinceLastTx": 600,
  "isNewDevice": false,
  "isNewBeneficiary": false,
  "failedAttempts24h": 0,
  "deviceTrustScore": 0.95
}
```

**Response (aprovado):**

```json
{
  "approved": true,
  "status": "APROVADO",
  "probability": 0.9231,
  "riskLevel": "LOW",
  "reasons": [],
  "evaluatedAt": "2025-01-15T14:32:00.123Z"
}
```

**Response (negado):**

```json
{
  "approved": false,
  "status": "NEGADO",
  "probability": 0.1842,
  "riskLevel": "CRITICAL",
  "suggestion": "Valor excede o limite diário da conta",
  "reasons": [
    "Valor acima do limite diário",
    "Dispositivo não reconhecido"
  ],
  "evaluatedAt": "2025-01-15T14:32:00.456Z"
}
```

### Níveis de Risco (`riskLevel`)

| Nível | Probabilidade | Descrição |
|---|---|---|
| `LOW` | ≥ 0.80 | Baixo risco — aprovação normal |
| `MEDIUM` | 0.55 – 0.79 | Risco moderado — aprovado com monitoramento |
| `HIGH` | 0.30 – 0.54 | Alto risco — aprovado com restrição ou negado |
| `CRITICAL` | < 0.30 | Risco crítico — transação negada |

### Status de Aprovação (`status`)

| Status | Significado |
|---|---|
| `APROVADO` | Transação aprovada sem restrições |
| `APROVADO_COM_RESTRICAO` | Aprovada, porém com monitoramento ativado |
| `NEGADO` | Transação recusada |

### Regras Hard-Block (bloqueio imediato, independente do modelo)

- `amount > dailyLimit` — valor ultrapassa o limite diário
- `amount > balance` — saldo insuficiente

### Swagger UI

Disponível em: [http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)

---

## Exemplo via cURL

```bash
curl -s -X POST http://localhost:8080/api/v1/transactions \
  -H "Content-Type: application/json" \
  -d @request.json | jq .
```

O arquivo `request.json` na raiz do projeto contém um payload de exemplo pronto para uso.

---

## Como Funciona Internamente

1. **`ModelConfiguration`** carrega o arquivo `financial_approval.onnx` do classpath via DJL `Criteria`, configura o engine `OnnxRuntime` e disponibiliza um `Supplier<Predictor>` como bean Spring.

2. **`FeatureDictionary`** desserializa o `feature_dict.json` contendo os parâmetros do `StandardScaler` (média e desvio) e os mapeamentos dos `LabelEncoders` para as features categóricas.

3. **`TransactionTransformer`** (implementa `NoBatchifyTranslator`) realiza:
   - **`processInput`**: converte o `TransactionRequest` em um vetor float[21], aplica z-score nas 17 features numéricas e codifica as 4 categóricas.
   - **`processOutput`**: extrai a probabilidade de saída do modelo, aplica as regras de hard-block, classifica o nível de risco e monta o `ApprovalResponse` com as razões de risco detectadas.

4. **`TransactionController`** recebe o POST, delega ao `Predictor` e retorna o resultado com status HTTP `202 Accepted`.
