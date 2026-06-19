# Capítulo 21 — MLOps para LLMs On-Premise

## O que é MLOps — e por que é diferente para LLMs

MLOps (Machine Learning Operations) é o conjunto de práticas que leva um modelo de "funciona no meu notebook" para "funciona em produção, 24/7, sem surpresas". É o DevOps do machine learning — mas com camadas extras de complexidade porque o artefato central (o modelo) não é código: é um binário opaco de bilhões de parâmetros que se comporta de forma probabilística.

Para LLMs on-premise, a complexidade aumenta:

| Desafio | ML Clássico | LLM On-Premise |
|---------|-------------|----------------|
| Tamanho do artefato | ~100 MB (XGBoost, RF) | 4–70 GB (7B–70B quantizado) |
| Tempo de carregamento | Milissegundos | 30–120 segundos |
| Hardware | CPU suficiente | GPU obrigatória (VRAM é gargalo) |
| Avaliação | Métricas determinísticas (F1, AUC) | Métricas subjetivas + determinísticas |
| Segurança | Input validation | Input validation + prompt injection |
| Observabilidade | Latência, throughput | Latência, throughput, tokens, traces, custo |

O ciclo MLOps para LLMs on-premise tem **7 etapas**, e cada uma delas aparece no AI-Orchestrator:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Containerização    → Ambiente reprodutível                  │
│  2. Configuração       → Config como código, fail-closed        │
│  3. Model Serving      → Servir modelo com GPU, versionamento   │
│  4. Testes             → Unit + evals de domínio + segurança    │
│  5. Observabilidade    → Traces, métricas, alertas              │
│  6. Segurança Runtime  → Injection detection, rate limiting     │
│  7. CI/CD              → Pipeline automatizado, gates           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Etapa 1: Containerização — ambiente reprodutível

### O problema

"Na minha máquina funciona" é inadmissível em produção. Um LLM on-premise depende de CUDA drivers, bibliotecas Python específicas, modelos SBERT pré-baixados e um frontend compilado. Sem containerização, reproduzir o ambiente é um pesadelo.

### A solução: Docker multi-stage

O AI-Orchestrator usa um Dockerfile multi-stage que separa build e runtime:

```dockerfile
# --- Stage 1: build do frontend (Vite) ---
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* /build/
RUN npm ci || npm install
COPY frontend/ /build/
RUN npm run build

# --- Stage 2: runtime Python ---
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/gateway/requirements.txt

# Pre-download SBERT model (cached na imagem)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', \
    cache_folder='/app/models')"

COPY gateway/ /app/gateway/
COPY evals/golden_routing.jsonl /app/evals/golden_routing.jsonl
COPY --from=frontend /build/dist /app/frontend/dist

RUN useradd -m app
USER app

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Por que cada decisão importa

| Decisão | Razão |
|---------|-------|
| `python:3.12-slim` em vez de `python:3.12` | Imagem 5x menor (~150 MB vs ~900 MB). Menos superfície de ataque |
| `--no-cache-dir` no pip | Elimina cache de download dentro da imagem. Reduz ~200 MB |
| `--extra-index-url pytorch.org/whl/cpu` | Instala PyTorch CPU-only (~200 MB vs ~2 GB com CUDA) |
| Pre-download do SBERT | Primeiro request não espera download de 90 MB |
| `useradd -m app` + `USER app` | Container roda como não-root. Segurança básica obrigatória |
| `COPY --from=frontend` | Frontend compilado sem Node.js no runtime |

### Microserviços com Dockerfile parametrizado

Os 4 domínios (finanças, RH, estoque, vendas) compartilham um Dockerfile com `ARG`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG SERVICE
ENV SERVICE=${SERVICE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ${SERVICE}/ /app/${SERVICE}/
COPY shared/ /app/shared/

# Volume /data já com ownership correto
RUN useradd -m app && mkdir -p /data && chown app:app /data
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn ${SERVICE}.main:app --host 0.0.0.0 --port 8000"]
```

Um Dockerfile, quatro serviços. O `ARG SERVICE` é passado no `docker-compose.yml`:

```yaml
financas:
  build:
    context: ./services
    args:
      SERVICE: financas
```

### Orquestração com Docker Compose

O `docker-compose.yml` orquestra **9 serviços** em uma rede interna:

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐     │
│  │ financas│  │    rh    │  │ estoque │  │  vendas  │     │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘     │
│       └────────────┼────────────┼────────────┘            │
│                    ▼                                        │
│              ┌──────────┐                                   │
│              │ gateway  │◄── healthcheck 30s                │
│              └─────┬────┘                                   │
│           ┌────────┼────────┐                               │
│           ▼        ▼        ▼                               │
│     ┌────────┐ ┌───────┐ ┌─────────┐                       │
│     │ ollama │ │qdrant │ │langfuse │                       │
│     │  GPU   │ │vectors│ │ traces  │                       │
│     └────────┘ └───────┘ └────┬────┘                       │
│                               ▼                             │
│                         ┌───────────┐                       │
│                         │langfuse-db│                       │
│                         │ Postgres  │                       │
│                         └───────────┘                       │
│                                                             │
│  [profile: public] ┌────────────┐                           │
│                     │cloudflared │ → tunnel público          │
│                     └────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

Decisões críticas de segurança na composição:

```yaml
# Todas as portas bound a 127.0.0.1 — sem exposição externa
ports:
  - "127.0.0.1:8100:8000"

# GPU reservation explícita com limite de memória
deploy:
  resources:
    limits:
      memory: 12g
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]

# Healthcheck obrigatório — compose não reinicia sem saber se está vivo
healthcheck:
  test: ["CMD", "python", "-c",
    "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=3)"]
  interval: 30s
  retries: 3
  start_period: 10s
```

---

## Etapa 2: Configuração como código

### O princípio: 12-Factor App

A metodologia [12-Factor](https://12factor.net) diz: **toda configuração deve vir do ambiente**. Nenhum valor hardcoded. Nenhum arquivo de config commitado com segredos.

### Implementação com dataclass imutável

O AI-Orchestrator centraliza toda configuração em um `Settings` imutável:

```python
from dataclasses import dataclass, field
import os

@dataclass(frozen=True)
class Settings:
    """Configuração imutável carregada uma vez no boot."""

    ollama_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11435")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("MODEL", "qwen2.5:7b-instruct-q4_K_M")
    )
    # Semantic router
    semantic_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEMANTIC_ENABLED", "1")
        not in ("0", "false", "False")
    )
    semantic_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SEMANTIC_THRESHOLD", "0.92"))
    )
    # Injection detector
    injection_detector_enabled: bool = field(
        default_factory=lambda: os.environ.get("INJECTION_DETECTOR_ENABLED", "1")
        not in ("0", "false", "False")
    )
    injection_threshold: float = field(
        default_factory=lambda: float(os.environ.get("INJECTION_THRESHOLD", "0.7"))
    )
    # Langfuse observability
    langfuse_enabled: bool = field(
        default_factory=lambda: os.environ.get("LANGFUSE_ENABLED", "1")
        not in ("0", "false", "False")
    )
    # ... 20+ variáveis no total
```

### Por que `frozen=True`?

Configuração mutável em runtime é uma fonte de bugs silenciosos. Com `frozen=True`, qualquer tentativa de alterar um campo após o boot levanta `FrozenInstanceError`. A configuração é um snapshot do ambiente — determinística e auditável.

### Fail-closed: variáveis obrigatórias

No `docker-compose.yml`, variáveis de segurança usam a sintaxe `:?` do Bash:

```yaml
environment:
  - INTERNAL_API_KEY=${INTERNAL_API_KEY:?defina INTERNAL_API_KEY no .env}
  - QDRANT_API_KEY=${QDRANT_API_KEY:?defina QDRANT_API_KEY no .env}
  - LANGFUSE_NEXTAUTH_SECRET=${LANGFUSE_NEXTAUTH_SECRET:?defina LANGFUSE_NEXTAUTH_SECRET}
  - LANGFUSE_SALT=${LANGFUSE_SALT:?defina LANGFUSE_SALT no .env}
```

Se a variável não estiver definida, o container **não sobe**. Isso é fail-closed: a ausência de configuração é um erro, não um fallback silencioso.

### `.env.example` como contrato

O `.env.example` documenta todas as variáveis sem expor segredos:

```bash
# === Autenticação ===
ACCESS_TOKEN=             # Bearer token do /chat. Vazio = aberto (dev)
INTERNAL_API_KEY=         # HMAC entre gateway e microsserviços (obrigatório)

# === Modelo ===
MODEL=qwen2.5:7b-instruct-q4_K_M   # Modelo Ollama (tag ou digest)

# === Semantic Router ===
QDRANT_API_KEY=           # API key do Qdrant (obrigatório)
SEMANTIC_THRESHOLD=0.92   # Cosine similarity mínima para match direto

# === Injection Detector ===
INJECTION_THRESHOLD=0.7   # Score mínimo para classificar como injection

# === Observabilidade ===
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

---

## Etapa 3: Model Serving e versionamento

### Ollama como model server

O AI-Orchestrator usa Ollama para servir LLMs locais. Ollama abstrai a complexidade de carregar modelos GGUF, gerenciar VRAM e expor uma API compatível com OpenAI.

```yaml
ollama:
  image: ollama/ollama:latest
  environment:
    - OLLAMA_KEEP_ALIVE=5m     # Mantém modelo na VRAM por 5min após último request
  deploy:
    resources:
      limits:
        memory: 12g            # Limite rígido — OOM kill se ultrapassar
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

### Seleção de modelo por variável de ambiente

O modelo é configurável sem rebuild da imagem:

```yaml
- MODEL=${MODEL:-qwen2.5:7b-instruct-q4_K_M}
```

Trocar de modelo é uma mudança de `.env`:

```bash
# Modelo menor, mais rápido
MODEL=qwen2.5:3b-instruct-q4_K_M

# Modelo maior, mais capaz (precisa de mais VRAM)
MODEL=qwen2.5:14b-instruct-q4_K_M

# MoE: mais capaz, mas 44% transborda para CPU na RTX 3060
MODEL=qwen2.5:30b-a3b-instruct-q4_K_M
```

### Versionamento: o gap e o caminho

**Situação atual (PoC):** modelo é identificado por tag (`qwen2.5:7b-instruct-q4_K_M`). Tags são mutáveis — o provedor pode atualizar o peso por trás da mesma tag.

**Produção:** pinar por digest SHA256:

```yaml
# PoC (aceitável para desenvolvimento)
MODEL=qwen2.5:7b-instruct-q4_K_M

# Produção (reprodutível)
MODEL=qwen2.5:7b-instruct-q4_K_M@sha256:a8c71be...
```

O digest garante que o modelo é **exatamente** o mesmo binário. Se alguém atualizar a tag, o digest detecta a divergência.

### Comparativo: model servers para LLMs on-premise

| Server | Prós | Contras | Quando usar |
|--------|------|---------|-------------|
| **Ollama** | Setup trivial, API OpenAI-compatível, pull automático | Serializa requests, sem batching | PoC, single-user, prototipagem |
| **vLLM** | Batching contínuo, PagedAttention, throughput alto | Setup complexo, exige mais VRAM | Produção multi-user |
| **TGI** | HuggingFace nativo, streaming, métricas Prometheus | Menos flexível que vLLM | Integração HuggingFace |
| **llama.cpp server** | GGUF nativo, baixo overhead | Sem batching avançado | Edge, recursos limitados |

O AI-Orchestrator usa Ollama porque é PoC single-node. Em produção multi-user, vLLM com batching contínuo seria a escolha.

---

## Etapa 4: Testes automatizados

### A pirâmide de testes para LLMs

Testes de LLM seguem uma pirâmide invertida em relação ao ML clássico:

```
         ┌───────────────┐
         │    Evals      │  ← Mais importante (comportamento do modelo)
         │  domínio/rota │
         ├───────────────┤
         │  Integração   │  ← API end-to-end, SSE, auth
         ├───────────────┤
         │   Unitários   │  ← Lógica de negócio, roteamento, sanitização
         └───────────────┘
```

No ML clássico, unitários são a base. Para LLMs, **evals** são a base — porque o comportamento do modelo é probabilístico e não é testável com assertions determinísticas.

### Unit tests: 182 testes sem LLM

O AI-Orchestrator tem 13 arquivos de teste cobrindo toda a lógica que **não** depende do modelo:

```
gateway/tests/
├── test_main.py              # API endpoints, SSE, healthcheck
├── test_graph.py             # StateGraph transitions
├── test_router.py            # Classificação de domínios
├── test_semantic_router.py   # Qdrant similarity search
├── test_agents.py            # Agent execution, tool calling
├── test_registry.py          # Service registry, discovery
├── test_embedder.py          # SBERT embeddings
├── test_injection_detector.py # BERTimbau classifier
├── test_internal_key.py      # HMAC inter-service auth
├── test_circuit.py           # Circuit breaker states
├── test_sanitize.py          # Input sanitization
├── test_security.py          # Rate limiting, auth
└── __init__.py
```

Padrão do test setup — ASGI in-memory, sem HTTP real:

```python
import httpx
import pytest
from gateway.main import app

@pytest.fixture
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")

@pytest.fixture(autouse=True)
def _open_access(monkeypatch):
    """Desabilita auth para testes unitários."""
    monkeypatch.setenv("ALLOW_OPEN_ACCESS", "1")
```

### Evals de domínio: o golden set

Os evals testam o **comportamento do modelo** contra um conjunto de perguntas com respostas esperadas:

```python
# evals/eval_domains.py — validação de qualidade por domínio
# Gate: >= 80% de acerto por domínio

# O golden set é um JSONL com perguntas e respostas esperadas:
# {"id": "fin-01", "domain": "financas", "question": "...", "expected": "..."}
# {"id": "rh-01", "domain": "rh", "question": "...", "expected": "..."}

# O eval envia cada pergunta para o /chat, compara a resposta com
# keywords esperadas, e computa precisão por domínio.
```

### Evals de roteamento: classificação correta

```python
# evals/eval_routing.py — classificador direciona para o domínio certo?
# Gate: >= 90% de acerto

# 64 perguntas no golden set, cada uma com domínio(s) esperado(s).
# Modo --semantic testa o Qdrant (leave-one-out: remove a pergunta
# do índice antes de classificar, evitando data leakage).
```

### Evals de segurança: prompt injection

```python
# evals/eval_injection.py — modelo vaza dados entre domínios?
# Gate: 0 leakages

# 6 casos adversariais testam se um prompt de finanças consegue
# extrair dados de RH, se ChatML injection funciona, etc.
# Qualquer vazamento = eval falha = deploy bloqueado.
```

### Comparativo: o que cada camada de teste cobre

| Camada | Quantidade | Sem LLM | Gate | O que testa |
|--------|-----------|---------|------|-------------|
| Unit tests | 182 | ✅ | — | Lógica, roteamento, sanitização, auth |
| Eval domínio | 40 | ❌ | ≥80% | Qualidade das respostas |
| Eval routing | 64 | ❌ | ≥90% | Classificação correta |
| Eval injection | 6 | ❌ | 0 leaks | Isolamento entre domínios |

---

## Etapa 5: Observabilidade

### O problema: LLMs são caixas-pretas caras

Em um sistema ML clássico, você monitora latência e throughput. Com LLMs, precisa monitorar:

- **Latência por etapa** (classificação → roteamento → agente → síntese)
- **Consumo de tokens** (input/output/total — tokens = custo)
- **Qualidade das respostas** (scores de avaliação)
- **Traces end-to-end** (qual caminho o request percorreu no grafo)
- **Incidentes de segurança** (injections detectadas, rate limits atingidos)

### Langfuse: observabilidade nativa para LLMs

O AI-Orchestrator usa Langfuse — plataforma de observabilidade projetada especificamente para LLMs. Diferente de ferramentas genéricas (Prometheus, Grafana), Langfuse entende o vocabulário de LLMs: traces, generations, tokens, scores.

```
Langfuse Cloud (padrão)
    ▲
    │ traces, generations, scores
    │
┌───┴────────────────────────────────────┐
│  Gateway                               │
│  ┌──────────────────────────────────┐  │
│  │ MetricsCollector                 │  │
│  │  - query Langfuse SDK            │  │
│  │  - agrega latência (avg/p50/p95) │  │
│  │  - conta tokens (in/out/total)   │  │
│  │  - cache 30s in-memory           │  │
│  │  - graceful degradation          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

### Dual-mode: Cloud + self-hosted fallback

```yaml
# Padrão: Langfuse Cloud (zero infra)
- LANGFUSE_HOST=${LANGFUSE_HOST:-https://us.cloud.langfuse.com}

# Fallback: self-hosted (compose inclui langfuse + postgres)
langfuse:
  image: langfuse/langfuse:2
  depends_on:
    langfuse-db:
      condition: service_healthy

langfuse-db:
  image: postgres:16-alpine
  volumes:
    - langfuse_db_data:/var/lib/postgresql/data
```

Cloud para conveniência. Self-hosted quando os dados não podem sair do perímetro — compliance, regulação, dados sensíveis.

### Métricas coletadas

O `MetricsCollector` agrega e expõe métricas via endpoint `/metrics`:

| Métrica | Descrição | Por que importa |
|---------|-----------|-----------------|
| Latência avg/p50/p95 | Tempo de resposta por percentil | p95 > 30s = UX degradada |
| Tokens input/output | Consumo por request | Custo e otimização de prompts |
| Routing breakdown | Semântico vs. LLM | Taxa de hit do cache semântico |
| Injection blocks | Requests bloqueados por injection | Ataques ativos? |
| Error count | Erros por período | Degradação do serviço |

### Graceful degradation

Se o Langfuse estiver indisponível, o sistema **não para**. O `MetricsCollector` retorna dados em cache (com flag `stale: true`) e continua operando:

```python
# Padrão: se Langfuse falhar, retorna cache stale — nunca crashar
try:
    fresh_metrics = self._query_langfuse()
    self._cache = fresh_metrics
    self._cache_time = time.time()
except Exception:
    # Cache stale é melhor que zero observabilidade
    if self._cache is not None:
        return {**self._cache, "stale": True}
    return self._empty_metrics()
```

---

## Etapa 6: Segurança em runtime

### Defesa em profundidade

Segurança não é uma feature — é uma propriedade do sistema inteiro. O AI-Orchestrator implementa **5 camadas de defesa**:

```
Request → [Rate Limit] → [Auth] → [Sanitização] → [Injection Detection] → [Processamento]
                                                                                   │
                                                    [Isolamento por Domínio] ◄─────┘
```

### Camada 1: Rate limiting

Previne abuso — cada IP tem um limite de requests por hora:

```yaml
RATE_LIMIT_PER_HOUR=${RATE_LIMIT_PER_HOUR:-10}
```

### Camada 2: Autenticação por Bearer token

```yaml
# Boundary público: sem ACCESS_TOKEN o /chat fica aberto (modo dev)
ACCESS_TOKEN=${ACCESS_TOKEN:-}
ALLOW_OPEN_ACCESS=${ALLOW_OPEN_ACCESS:-1}
```

Dois modos: aberto (desenvolvimento) e fechado (produção). A decisão é explícita via variável de ambiente, nunca implícita.

### Camada 3: Sanitização estrutural

Remove tokens que permitem escape do wrapper do prompt (ChatML, tags internas):

```python
import re

_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
_WRAPPER_TAG = re.compile(
    r"<\s*/?\s*(user_question|user_input|plan|agent_answers|context)\s*>",
    re.IGNORECASE,
)

def sanitizar_entrada(texto: str) -> str:
    """Remove sequências de escape estrutural."""
    texto = _SPECIAL_TOKEN.sub(" ", texto)
    texto = _WRAPPER_TAG.sub(" ", texto)
    return re.sub(r"[ \t]{2,}", " ", texto).strip()
```

### Camada 4: Injection Detection com BERTimbau

Um classificador BERTimbau fine-tunado detecta tentativas de prompt injection:

```python
class InjectionDetector:
    """Classificador de injection com lazy loading e fallback seguro."""

    def score(self, text: str) -> float:
        """Retorna probabilidade de injection (0.0 a 1.0).

        Retorna -1.0 se o modelo não estiver disponível.
        Nunca bloqueia o pipeline — apenas sinaliza.
        """

    def is_injection(self, text: str) -> bool:
        """True se score >= threshold (padrão: 0.7)."""
```

Decisões de design:

- **Lazy loading**: modelo carrega no primeiro request, não no boot. Acelera startup
- **Fallback `-1.0`**: se o classificador falhar, retorna score impossível. O sistema **nunca bloqueia** por falha do detector — bloquear erroneamente é pior que deixar passar
- **Threshold configurável**: `INJECTION_THRESHOLD=0.7` é conservador. Ajustar por ambiente

### Camada 5: Isolamento entre domínios

Os microsserviços são autenticados por HMAC (`INTERNAL_API_KEY`). Um agente de finanças não consegue acessar dados de RH:

```yaml
# Cada microsserviço valida a mesma chave
- INTERNAL_API_KEY=${INTERNAL_API_KEY:?defina INTERNAL_API_KEY no .env}
```

O eval de injection (`eval_injection.py`) valida que esse isolamento funciona: 6 casos adversariais com gate de **zero vazamentos**.

### Resumo das camadas

| Camada | Mecanismo | Configuração | Falha = ? |
|--------|-----------|-------------|-----------|
| Rate limit | IP-based throttle | `RATE_LIMIT_PER_HOUR` | Request rejeitado (429) |
| Auth | Bearer token | `ACCESS_TOKEN` | Request rejeitado (401) |
| Sanitização | Regex (ChatML, tags) | Sempre ativo | Input limpo |
| Injection | BERTimbau classifier | `INJECTION_THRESHOLD` | Log + flag (não bloqueia) |
| Isolamento | HMAC inter-service | `INTERNAL_API_KEY` | Microsserviço rejeita (401) |

---

## Etapa 7: CI/CD — da PoC à produção

### O estado atual: PoC sem pipeline

O AI-Orchestrator é uma PoC com testes manuais. Não existe GitHub Actions, não existe Terraform, não existe cobertura de código automatizada. Isso é aceitável para um projeto de portfólio — **inaceitável para produção**.

### O pipeline ideal para LLMs on-premise

```yaml
# .github/workflows/ci.yml — pipeline completo para LLM on-premise
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Gate 1: Código compila e testes unitários passam (sem GPU)
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r gateway/requirements.txt -r requirements-dev.txt
      - run: pytest gateway/tests/ --cov=gateway --cov-fail-under=80

  # Gate 2: Imagens Docker constroem corretamente
  build-images:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build

  # Gate 3: Evals (requer GPU — self-hosted runner ou GPU cloud)
  evals:
    runs-on: [self-hosted, gpu]
    needs: [unit-tests, build-images]
    steps:
      - uses: actions/checkout@v4
      - run: docker compose up -d
      - run: python evals/eval_routing.py --semantic  # Gate: >= 90%
      - run: python evals/eval_domains.py             # Gate: >= 80%
      - run: python evals/eval_injection.py           # Gate: 0 leaks

  # Gate 4: Deploy (só se todos os gates passaram)
  deploy:
    runs-on: [self-hosted, gpu]
    needs: [evals]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: docker compose --profile public up -d --build
```

### Gates de qualidade: o que bloqueia deploy

| Gate | Ferramenta | Threshold | Sem GPU |
|------|-----------|-----------|---------|
| Cobertura de código | pytest-cov | ≥80% | ✅ |
| Build de imagens | docker compose build | Sucesso | ✅ |
| Eval de roteamento | eval_routing.py | ≥90% acerto | ❌ |
| Eval de domínio | eval_domains.py | ≥80% acerto | ❌ |
| Eval de injection | eval_injection.py | 0 leaks | ❌ |

### O desafio dos evals no CI

Evals de LLM precisam de GPU. Runners padrão do GitHub Actions são CPU-only. Soluções:

1. **Self-hosted runner com GPU**: sua máquina roda os evals localmente. Custo zero, controle total, mas a máquina precisa estar online
2. **GPU cloud on-demand**: AWS g4dn, GCP T4, Azure NC-series. Sobe VM, roda evals, destrói VM. Custo: ~$0.50/hora
3. **Evals offline com cache**: roda evals localmente, commita o resultado. CI valida que o resultado existe e é recente

---

## Maturidade MLOps: onde você está?

### Framework de níveis

O Google publicou um framework de maturidade MLOps com 3 níveis. Adaptado para LLMs on-premise:

| Nível | Descrição | Características |
|-------|-----------|-----------------|
| **0 — Manual** | Tudo é manual. Jupyter notebook em produção | Sem CI/CD, sem containerização, modelo em arquivos locais |
| **1 — Pipeline** | Deploy automatizado, mas treino/eval manual | Docker, compose, testes unitários, configuração 12-factor |
| **2 — CI/CD** | Treino, eval e deploy automatizados | GitHub Actions, gates de qualidade, rollback automático |
| **3 — Contínuo** | Feedback loop: produção alimenta retreino | A/B testing, drift detection, retreino automático |

### AI-Orchestrator: Nível 1 com elementos de Nível 2

| Aspecto | Nível alcançado | Evidência |
|---------|----------------|-----------|
| Containerização | ✅ Nível 1 | Docker multi-stage, compose 9 serviços |
| Configuração | ✅ Nível 2 | Pydantic/dataclass imutável, fail-closed, `.env.example` |
| Model serving | ✅ Nível 1 | Ollama + env var (sem registry/digest) |
| Testes unitários | ✅ Nível 1 | 182 testes, mas sem coverage enforcement |
| Evals | ✅ Nível 2 | 3 suites com gates quantitativos |
| Observabilidade | ✅ Nível 2 | Langfuse Cloud + self-hosted fallback |
| Segurança | ✅ Nível 2 | 5 camadas, injection classifier, evals adversariais |
| CI/CD | ❌ Nível 0 | Sem pipeline automatizado |
| IaC | ❌ Nível 0 | Sem Terraform/Bicep/Helm |
| Model registry | ❌ Nível 0 | Sem MLflow/BentoML |

### Roadmap: do Nível 1 ao Nível 2

```
                    Agora                        Nível 2
                     │                             │
  ┌──────────────────┤                             │
  │ ✅ Containers    │  ┌── pytest-cov ≥80% ──────┤
  │ ✅ Config 12F    │  ├── GitHub Actions CI ─────┤
  │ ✅ 182 tests     │  ├── Model digest pin ──────┤
  │ ✅ 3 eval suites │  ├── Terraform/Bicep IaC ───┤
  │ ✅ Langfuse      │  ├── MLflow registry ───────┤
  │ ✅ 5 camadas sec │  └── Structured logging ────┤
  └──────────────────┘                             │
```

As 6 ações para chegar ao Nível 2:

1. **pytest-cov com threshold** — `--cov-fail-under=80` no CI
2. **GitHub Actions** — pipeline de 4 gates (unit → build → eval → deploy)
3. **Model digest pinning** — SHA256 em vez de tag mutável
4. **IaC** — Terraform para infra cloud, Helm para Kubernetes
5. **Model registry** — MLflow para versionar modelos fine-tunados
6. **Structured logging** — JSON em vez de strings formatadas. Parsável por Loki/ELK

---

## MAST Failure Taxonomy: 14 modos de falha em agentes

O catálogo Agents Integration Patterns mapeou 14 modos de falha empiricamente observados em sistemas multi-agente (Cemri et al., NeurIPS 2025, 1.600+ traces analisados, 41-86.7% de taxa de falha). O AI-Orchestrator mitiga os 5 mais relevantes:

| Modo de falha | Categoria | Mitigação no AI-Orchestrator |
|---------------|-----------|------------------------------|
| **Tool parameter mismatch** | Execution | OpenAPI schema validation no `ToolRegistry` — argumentos validados antes do call |
| **Missing tool capability** | Specification | ToolRegistry retorna lista de ferramentas disponíveis como feedback ao modelo |
| **Context overflow** | Communication | `max_seq_length=4096` + chunking de histórico (últimas 20 mensagens) |
| **Agent hallucinated tool call** | Execution | `train_on_responses_only` + LoRA fine-tune + eval gate de domains ≥80% |
| **Unauthorized action attempt** | Security | Least-Privilege Tool Scope — cada domínio só vê suas ferramentas |

**Lição de produção:** "Integration patterns are how you engineer reliability that scaling the model cannot buy." O fine-tuning melhorou o tool-calling, mas foram os padrões de Harness (circuit breaker, tool registry com validação, least-privilege scope) que eliminaram as falhas catastróficas.

---

## Resumo

MLOps para LLMs on-premise não é um luxo — é o que separa uma demo de um produto. As 7 etapas formam um ciclo onde cada uma reforça as outras:

```
Containerização → Config → Serving → Testes → Observabilidade → Segurança → CI/CD
       ▲                                                                       │
       └───────────────────────────────────────────────────────────────────────┘
                              Feedback loop contínuo
```

O AI-Orchestrator demonstra que é possível implementar MLOps real — containerização multi-stage, configuração imutável, 3 suites de eval com gates quantitativos, observabilidade nativa com Langfuse, e 5 camadas de segurança — mesmo em uma PoC single-node com uma RTX 3060.

### Notas de produção

**SSE Heartbeat:** Para manter a conexão Server-Sent Events viva durante delays de LLM (>15s), o gateway em `main.py` envia keepalives periódicos via `asyncio.Queue`:

```python
# gateway/main.py — keepalive SSE via asyncio.Queue
try:
    event, data = await asyncio.wait_for(
        queue.get(), timeout=15.0
    )
except asyncio.TimeoutError:
    yield ": keepalive\n\n"  # Cloudflare 524 timeout prevention
    continue
```

Sem keepalive, proxies como Nginx e Cloudflare fecham conexões inativas após 60-120s, quebrando a experiência de streaming.

**Rate Limiter por IP:** O `gateway/security.py` implementa sliding window rate limiting com `max_entries=10000` e eviction automático:

```python
# gateway/security.py — RateLimiter sliding window + eviction
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_s: float = 3600.0,
                 max_entries: int = 10_000):
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._max_entries = max_entries

    def allow(self, client_ip: str) -> bool:
        now = time.monotonic()
        hits = self._hits.setdefault(client_ip, deque())
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False  # bloqueado
        hits.append(now)
        return True
```

**Cloudflare Tunnel:** Para expor o gateway on-premise sem abrir portas no roteador, o AI-Orchestrator usa `cloudflared tunnel`:

```bash
# Deploy com Cloudflare Tunnel (produção em suasalada.com.br)
cloudflared tunnel create orchestrator
cloudflared tunnel route dns orchestrator suasalada.com.br
cloudflared tunnel run --url http://localhost:8000 orchestrator
```

Isso elimina a necessidade de port forwarding, DDNS, ou certificados SSL manuais — o túnel gerencia tudo.

> **Exercício final:** Documente seu próprio setup MLOps. Desenhe o diagrama de deploy (serviços, portas, dependências). Liste os pontos únicos de falha. Proponha 3 melhorias para produção.

O gap para produção é claro e quantificável: CI/CD automatizado, model registry e IaC. São 6 ações concretas. Nenhuma exige reescrever o sistema — apenas automatizar o que já é feito manualmente.
