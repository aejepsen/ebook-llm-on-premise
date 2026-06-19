# Capítulo 18 — Benchmarking e Monitoramento

## Por que medir performance

"Se você não mede, você não sabe." Essa máxima é ainda mais verdadeira para LLMs on-premise. Diferente de APIs na nuvem (onde o provedor garante SLAs), em infraestrutura própria **você é o provedor**. Se o modelo está lento, se a qualidade caiu após uma troca de quantização, se o throughput não aguenta a carga — é seu problema.

Medir performance serve a três propósitos:

1. **Decisão de modelo.** Qwen2.5:7b-q4 vs q8 vs 14b: qual entrega a melhor relação qualidade/latência para o seu caso de uso?
2. **Capacidade.** Quantos usuários simultâneos o sistema aguenta antes de degradar?
3. **Detecção de regressão.** O modelo que funcionava bem ontem está respondendo pior hoje? Sem métricas, você não sabe.

## Métricas de serving: TTFT, TPS, throughput, latência

### TTFT (Time to First Token)

Tempo entre o envio da requisição e o recebimento do primeiro token da resposta. Crucial para experiência do usuário — é o tempo que ele espera antes de "ver algo acontecendo".

```
Requisição enviada → [TTFT] → Primeiro token → [geração] → Último token
```

- **Bom:** < 500 ms
- **Aceitável:** 500 ms – 2 s
- **Problemático:** > 2 s

TTFT depende de: tamanho do prompt (mais tokens de entrada = mais tempo de prefill), tamanho do modelo, e quantização.

### TPS (Tokens Per Second)

Velocidade de geração: quantos tokens o modelo produz por segundo. Afeta diretamente o tempo total de resposta.

```
TPS = tokens_gerados / tempo_de_geração
```

- **Bom para 7B q4:** 30–60 TPS (GPU RTX 3060)
- **Bom para 7B q8:** 20–40 TPS
- **Bom para 14B q4:** 15–30 TPS

### Throughput

Requisições completas por segundo (ou por minuto) que o sistema processa sob carga. Diferente de TPS (que mede um request), throughput mede o sistema como um todo.

### Latência P50/P95/P99

Percentis de latência sobre todas as requisições:

- **P50 (mediana):** metade das requisições é mais rápida que este valor.
- **P95:** 95% das requisições é mais rápida. Captura os casos "meio ruins".
- **P99:** 99% das requisições é mais rápida. Captura os outliers.

A **média** de latência é enganosa — esconde os casos extremos. Um sistema com média de 1s pode ter P99 de 15s, significando que 1 em 100 usuários espera 15 segundos.

```python
# metricas_latencia.py
# Calcula percentis de latência a partir de medições

import numpy as np

def calcular_percentis(latencias: list[float]) -> dict:
    """Calcula P50, P95, P99 de uma lista de latências em segundos."""
    arr = np.array(latencias)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "media": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# Exemplo: 100 medições simuladas
import random
latencias = [random.uniform(0.5, 2.0) for _ in range(95)]
latencias += [random.uniform(5.0, 15.0) for _ in range(5)]  # 5% lentos

stats = calcular_percentis(latencias)
print(f"P50:   {stats['p50']:.2f}s")
print(f"P95:   {stats['p95']:.2f}s")
print(f"P99:   {stats['p99']:.2f}s")
print(f"Média: {stats['media']:.2f}s")  # Média engana — puxada pelos lentos
```

## Ferramentas de benchmarking

### hey — HTTP load testing simples

```bash
# Instalar
go install github.com/rakyll/hey@latest

# Testar endpoint do Ollama com 10 requests concorrentes
hey -n 50 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Olá"}],"stream":false}' \
  http://localhost:11434/api/chat
```

### Locust — load testing em Python

```python
# locustfile.py
# Teste de carga para o endpoint do gateway

from locust import HttpUser, task, between
import json

class UsuarioGateway(HttpUser):
    """Simula usuários fazendo perguntas ao gateway."""
    wait_time = between(1, 5)  # espera entre requisições

    @task(3)
    def pergunta_simples(self):
        """Pergunta que aciona 1 agente."""
        self.client.post(
            "/chat",
            json={"question": "Qual o saldo do produto SKU-001?"},
            headers={"X-Access-Token": "meu_token"},
            timeout=60,
        )

    @task(1)
    def pergunta_multi_agente(self):
        """Pergunta que aciona múltiplos agentes (mais pesada)."""
        self.client.post(
            "/chat",
            json={"question": "Qual o impacto financeiro das férias do time de vendas?"},
            headers={"X-Access-Token": "meu_token"},
            timeout=120,
        )
```

```bash
# Rodar Locust
locust -f locustfile.py --host http://localhost:8000
# Abrir http://localhost:8089 para o dashboard
```

### Script de benchmark customizado

```python
# benchmark_modelo.py
# Benchmark detalhado de modelo local com TTFT e TPS

import time
import httpx
import json
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434"

@dataclass
class ResultadoBenchmark:
    modelo: str
    prompt_tokens: int
    gerados: int
    ttft_s: float
    total_s: float
    tps: float

def benchmark_modelo(
    modelo: str,
    prompt: str,
    max_tokens: int = 200,
) -> ResultadoBenchmark:
    """Mede TTFT e TPS de um modelo via streaming."""
    inicio = time.monotonic()
    primeiro_token = None
    tokens_gerados = 0
    conteudo = ""

    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"num_predict": max_tokens},
        },
        timeout=120.0,
    ) as resposta:
        for linha in resposta.iter_lines():
            if not linha:
                continue
            dado = json.loads(linha)

            if primeiro_token is None and dado.get("message", {}).get("content"):
                primeiro_token = time.monotonic()

            if dado.get("message", {}).get("content"):
                conteudo += dado["message"]["content"]
                tokens_gerados += 1

            if dado.get("done"):
                break

    fim = time.monotonic()
    ttft = (primeiro_token - inicio) if primeiro_token else (fim - inicio)
    total = fim - inicio
    tps = tokens_gerados / (total - ttft) if total > ttft else 0

    return ResultadoBenchmark(
        modelo=modelo,
        prompt_tokens=len(prompt.split()),  # aproximação
        gerados=tokens_gerados,
        ttft_s=round(ttft, 3),
        total_s=round(total, 3),
        tps=round(tps, 1),
    )

def comparar_modelos(modelos: list[str], prompt: str):
    """Compara TTFT e TPS entre modelos/quantizações."""
    print(f"{'Modelo':<25} {'TTFT(s)':<10} {'TPS':<10} {'Total(s)':<10} {'Tokens':<8}")
    print("-" * 65)

    for modelo in modelos:
        resultado = benchmark_modelo(modelo, prompt)
        print(
            f"{resultado.modelo:<25} "
            f"{resultado.ttft_s:<10} "
            f"{resultado.tps:<10} "
            f"{resultado.total_s:<10} "
            f"{resultado.gerados:<8}"
        )

# Uso
if __name__ == "__main__":
    prompt = "Explique o que é machine learning em 3 parágrafos."
    modelos = [
        "qwen2.5:7b",        # Q4_K_M padrão
        "qwen2.5:7b-instruct-q8_0",  # Q8
        "qwen2.5:14b",       # 14B Q4
    ]
    comparar_modelos(modelos, prompt)
```

## Benchmarking de modelos: comparando quantizações e frameworks

### Quantizações: Q4 vs Q8 vs FP16

| Quantização | Tamanho (7B) | TPS (RTX 3060) | Qualidade | Quando usar |
|-------------|-------------|----------------|-----------|-------------|
| Q4_K_M | ~4 GB | 40–60 | Boa | Produção geral |
| Q5_K_M | ~5 GB | 30–50 | Muito boa | Equilíbrio |
| Q8_0 | ~7 GB | 20–35 | Excelente | Quando qualidade é crítica |
| FP16 | ~14 GB | 10–20 | Referência | Benchmark baseline |

A regra: Q4_K_M é o ponto de partida. Suba para Q8 apenas se os evals mostrarem degradação mensurável no seu caso de uso. FP16 é para medir — não para servir.

### Frameworks de serving

| Framework | Strengths | Batching | Quando usar |
|-----------|----------|----------|-------------|
| **Ollama** | Simples, integrado | Não | Dev, PoC, equipe pequena |
| **vLLM** | Continuous batching, PagedAttention | Sim | Produção com múltiplos usuários |
| **llama.cpp server** | Leve, controle fino | Limitado | Recursos restritos |
| **TGI (HuggingFace)** | Ecossistema HF | Sim | Se já usa HuggingFace |

## Monitoramento em produção: Langfuse, Prometheus, Grafana

### Langfuse — observabilidade LLM

Langfuse é uma plataforma de observabilidade específica para LLMs. Registra cada interação com o modelo: prompt, resposta, latência, tokens, custo, e métricas de qualidade.

```python
# langfuse_integracao.py
# Integração do pipeline com Langfuse para observabilidade

from langfuse import Langfuse

# Inicializa o cliente Langfuse
langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="http://localhost:3000",  # Langfuse self-hosted
)

def pipeline_observavel(pergunta: str) -> str:
    """Pipeline RAG com observabilidade Langfuse."""

    # Cria um trace para toda a requisição
    trace = langfuse.trace(
        name="rag-pipeline",
        input={"pergunta": pergunta},
        metadata={"modelo": "qwen2.5:7b"},
    )

    # Span de retrieval
    span_retrieval = trace.span(
        name="retrieval",
        input={"query": pergunta, "top_k": 5},
    )
    # ... executa busca vetorial ...
    contextos = ["chunk1", "chunk2"]  # exemplo
    span_retrieval.end(output={"chunks": len(contextos)})

    # Generation: registra a chamada ao LLM
    generation = trace.generation(
        name="llm-generation",
        model="qwen2.5:7b",
        input=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": pergunta},
        ],
    )
    resposta = "Resposta gerada pelo modelo..."
    generation.end(
        output=resposta,
        usage={"input": 150, "output": 80},  # tokens
    )

    # Score de qualidade (manual ou automático)
    trace.score(
        name="relevancia",
        value=0.85,
        comment="Resposta relevante para a pergunta",
    )

    trace.update(output={"resposta": resposta})
    return resposta
```

### Prometheus + Grafana — métricas de infraestrutura

Prometheus coleta métricas numéricas (latência, TPS, uso de GPU). Grafana as visualiza em dashboards.

```python
# metricas_prometheus.py
# Exposição de métricas para Prometheus

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
)

# Métricas do pipeline
REQUISICOES_TOTAL = Counter(
    "gateway_requisicoes_total",
    "Total de requisições ao gateway",
    ["dominio", "status"],
)

LATENCIA_PIPELINE = Histogram(
    "gateway_latencia_segundos",
    "Latência do pipeline completo",
    ["dominio"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

TTFT = Histogram(
    "gateway_ttft_segundos",
    "Time to First Token",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5],
)

AGENTES_ATIVOS = Gauge(
    "gateway_agentes_ativos",
    "Agentes em execução no momento",
)

# Uso nas funções do pipeline
import time

def processar_requisicao(pergunta: str, dominio: str):
    """Processa requisição registrando métricas."""
    AGENTES_ATIVOS.inc()
    inicio = time.monotonic()

    try:
        # ... pipeline ...
        resposta = "..."
        REQUISICOES_TOTAL.labels(dominio=dominio, status="sucesso").inc()
    except Exception:
        REQUISICOES_TOTAL.labels(dominio=dominio, status="erro").inc()
        raise
    finally:
        LATENCIA_PIPELINE.labels(dominio=dominio).observe(time.monotonic() - inicio)
        AGENTES_ATIVOS.dec()

    return resposta

# Inicia servidor de métricas na porta 9090
# start_http_server(9090)
```

### Docker Compose para o stack de monitoramento

```yaml
# docker-compose.monitoring.yml
# Stack de monitoramento: Prometheus + Grafana + Langfuse

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://langfuse:langfuse@db:5432/langfuse
      - NEXTAUTH_SECRET=segredo_langfuse
      - SALT=salt_langfuse
    depends_on:
      - db

  db:
    image: postgres:16
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse_db:/var/lib/postgresql/data

volumes:
  grafana_data:
  langfuse_db:
```

## Observabilidade LLM: traces, generations, scores

No AI-Orchestrator, a observabilidade segue o padrão:

- **Trace por request:** cada pergunta do usuário gera um trace com ID único (`trace_id`).
- **Span por nó:** cada nó do grafo (sanitize, classify, dispatch, synthesize) é um span dentro do trace.
- **Generation por LLM call:** cada chamada ao Ollama é registrada como generation com input/output/tokens.
- **Logs estruturados JSON:** cada nó loga `trace_id`, `node`, `latency_ms`, `domains` em JSON — parseável por qualquer ferramenta de log aggregation.

```json
{
  "trace_id": "a1b2c3d4-...",
  "node": "dispatch",
  "latency_ms": 3421.5,
  "domains": ["estoque", "financas"]
}
```

## Alertas: quando o modelo degrada

Degradação de LLM é silenciosa. O modelo não dá erro — dá resposta ruim. Alertas devem cobrir:

### Métricas de infraestrutura

| Métrica | Threshold de alerta | Significado |
|---------|---------------------|-------------|
| TTFT P95 > 3s | Warning | Prefill está lento |
| TPS < 10 | Critical | Modelo não acompanha |
| Latência P99 > 30s | Critical | Usuários esperando demais |
| GPU memory > 95% | Warning | Risco de OOM |
| Requisições com erro > 5% | Critical | Algo quebrou |

### Métricas de qualidade (via Langfuse)

| Métrica | Como medir | Threshold |
|---------|-----------|-----------|
| Taxa de tool_not_found | Logs do agente | > 10% = modelo confuso |
| Stop reason = deadline | Logs do agente | > 5% = modelo lento demais |
| Stop reason = max_iters | Logs do agente | > 10% = modelo em loop |
| Score de relevância | Avaliação manual/automática | Média < 0.7 |

### Exemplo de alerta no Prometheus (Alertmanager)

```yaml
# alertas.yml
# Regras de alerta para degradação do LLM

groups:
  - name: gateway_alertas
    rules:
      - alert: LatenciaAlta
        expr: histogram_quantile(0.95, rate(gateway_latencia_segundos_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 de latência acima de 10s por mais de 5 minutos"

      - alert: TaxaErroAlta
        expr: rate(gateway_requisicoes_total{status="erro"}[5m]) / rate(gateway_requisicoes_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Taxa de erro acima de 5%"
```

## Caso real: Langfuse no AI-Orchestrator

No AI-Orchestrator, o Langfuse é integrado como observabilidade opt-in:

1. O `GatewayGraph` recebe um `Tracer` (wrapper do Langfuse).
2. Cada invocação de `run()` cria um trace com `trace_id` único.
3. Cada nó do grafo abre um span com `trace.span(name="sanitize")`.
4. Cada chamada ao Ollama registra uma generation.
5. O `tool_trace` de cada agente registra nome da ferramenta, argumentos e status HTTP.

Isso permite: "por que a requisição X demorou 15 segundos?" → abrir o trace → ver que o nó `dispatch` demorou 12s → ver que o agente de RH fez 4 tool calls, sendo a terceira um timeout do microsserviço.

Sem observabilidade, a resposta seria "está lento" e ninguém saberia por quê.

## Langfuse em profundidade: observabilidade LLM on-premise

### O que e Langfuse

Langfuse e uma plataforma open-source de observabilidade para LLMs. Diferente de solucoes SaaS como LangSmith ou Helicone, o Langfuse pode ser deployado inteiramente on-premise — requisito para cenarios com dados sensiveis, compliance regulatorio ou air-gapped environments.

Os conceitos centrais:

- **Trace:** representa uma requisicao completa do usuario. Contem metadados (user_id, session_id, tags) e agrega todos os spans e generations.
- **Span:** uma etapa logica dentro do trace. No AI-Orchestrator, cada no do LangGraph (sanitize, classify, dispatch, synthesize) e um span.
- **Generation:** uma chamada especifica ao LLM. Registra modelo, prompt, resposta, tokens de entrada/saida e latencia.

### Arquitetura no AI-Orchestrator

```
Usuario → Gateway /chat → LangGraph Pipeline → Ollama
                |                  |                |
                |    [Langfuse SDK Python]          |
                |         |                        |
                v         v                        v
           1 trace    1 span/no              1 generation/call
                \         |                  /
                 \        |                 /
                  v       v                v
              Langfuse Server (Docker)
                      |
                  PostgreSQL
```

Cada request `POST /chat` cria exatamente 1 trace. Cada nó do grafo (sanitize, classify, dispatch, synthesize) abre 1 span dentro desse trace. Cada chamada ao Ollama — seja para classificação semântica, execução de agente ou síntese — registra 1 generation com input/output/tokens.

### Implementação real: Tracer com degradação graceful

O ponto crítico: Langfuse é observabilidade, não é parte do caminho crítico. Se o Langfuse cair, o gateway **deve continuar funcionando**. A implementação usa o padrão noop handle:

```python
# gateway/tracing.py
# Tracer com degradação graceful — Langfuse offline não afeta requests

from langfuse import Langfuse
import logging

logger = logging.getLogger(__name__)


class TraceHandle:
    """Wrapper seguro sobre trace do Langfuse.

    Nunca lança exceção. Se Langfuse estiver offline,
    todos os métodos são noop — o request continua normalmente.
    """

    def __init__(self, trace):
        self._trace = trace

    def span(self, *, name: str, input: dict | None = None):
        try:
            return SpanHandle(self._trace.span(name=name, input=input))
        except Exception:
            logger.debug("langfuse span noop: %s", name)
            return SpanHandle(None)

    def generation(self, *, name: str, model: str, input: list):
        try:
            return GenerationHandle(
                self._trace.generation(name=name, model=model, input=input)
            )
        except Exception:
            return GenerationHandle(None)

    def update(self, **kwargs):
        try:
            self._trace.update(**kwargs)
        except Exception:
            pass

    def end(self):
        try:
            self._trace.update(status="completed")
        except Exception:
            pass


class SpanHandle:
    """Wrapper seguro sobre span."""

    def __init__(self, span):
        self._span = span

    def end(self, *, output: dict | None = None):
        if self._span:
            try:
                self._span.end(output=output)
            except Exception:
                pass


class GenerationHandle:
    """Wrapper seguro sobre generation."""

    def __init__(self, generation):
        self._gen = generation

    def end(self, *, output: str = "", usage: dict | None = None):
        if self._gen:
            try:
                self._gen.end(output=output, usage=usage)
            except Exception:
                pass


class Tracer:
    """Ponto de entrada para tracing.

    Inicializa conexão com Langfuse. Se falhar,
    opera em modo noop sem impactar o pipeline.
    """

    def __init__(self, settings):
        try:
            self._langfuse = Langfuse(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
            logger.info("Langfuse conectado: %s", settings.langfuse_host)
        except Exception as exc:
            logger.warning("Langfuse indisponivel: %s", exc)
            self._langfuse = None

    def trace(self, *, trace_id: str, name: str = "chat") -> TraceHandle:
        if not self._langfuse:
            return TraceHandle(None)
        try:
            t = self._langfuse.trace(id=trace_id, name=name)
            return TraceHandle(t)
        except Exception:
            return TraceHandle(None)
```

O padrao e simples: cada classe wrappa o objeto real do SDK e captura qualquer excecao. O chamador nunca precisa saber se o Langfuse esta online ou offline. Isso e essencial para producao — observabilidade nao pode ser ponto unico de falha.

### Métricas coletadas via Langfuse

Com traces estruturados, o Langfuse coleta automaticamente:

| Métrica | Onde | Exemplo |
|---------|------|---------|
| Latência por camada | Span duration | sanitize: 12ms, classify: 340ms, dispatch: 8200ms |
| Tokens in/out | Generation usage | input: 1200, output: 380 |
| Modelo utilizado | Generation model | qwen2.5:7b-instruct-q4_K_M |
| Routing layer | Span metadata | semantic (rápido) vs LLM (lento) |
| Injection blocks | Span output | 3 tentativas bloqueadas em sanitize |
| Erros por camada | Span status | timeout no agente de RH |
| Domínios acionados | Trace metadata | ["estoque", "finanças"] |

### Dashboard de métricas: endpoint /metrics

O gateway expõe um endpoint `GET /metrics` que agrega dados do Langfuse com cache de 30 segundos. O frontend React consome e exibe cards em tempo real:

```python
# gateway/metrics.py
# Endpoint de métricas agregadas com cache

import time
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class MetricasAgregadas:
    total_requests: int
    latencia_p50_ms: float
    latencia_p95_ms: float
    tokens_total: int
    routing_semantic_pct: float
    routing_llm_pct: float
    injection_blocks: int
    erro_pct: float
    timestamp: float


class MetricsCollector:
    """Coleta e agrega métricas do Langfuse."""

    CACHE_TTL = 30  # segundos

    def __init__(self, langfuse_client):
        self._langfuse = langfuse_client
        self._cache: MetricasAgregadas | None = None
        self._cache_ts: float = 0

    def get_metrics(self) -> MetricasAgregadas:
        """Retorna métricas agregadas com cache de 30s."""
        agora = time.monotonic()
        if self._cache and (agora - self._cache_ts) < self.CACHE_TTL:
            return self._cache

        # Busca traces das ultimas 24h via API do Langfuse
        traces = self._fetch_recent_traces(hours=24)
        latencias = [t.latency_ms for t in traces if t.latency_ms]

        self._cache = MetricasAgregadas(
            total_requests=len(traces),
            latencia_p50_ms=self._percentil(latencias, 50),
            latencia_p95_ms=self._percentil(latencias, 95),
            tokens_total=sum(t.tokens for t in traces),
            routing_semantic_pct=self._routing_pct(traces, "semantic"),
            routing_llm_pct=self._routing_pct(traces, "llm"),
            injection_blocks=sum(t.blocks for t in traces),
            erro_pct=self._erro_pct(traces),
            timestamp=time.time(),
        )
        self._cache_ts = agora
        return self._cache

    @staticmethod
    def _percentil(valores: list[float], p: int) -> float:
        if not valores:
            return 0.0
        valores_sorted = sorted(valores)
        idx = int(len(valores_sorted) * p / 100)
        return valores_sorted[min(idx, len(valores_sorted) - 1)]

    # _fetch_recent_traces, _routing_pct, _erro_pct omitidos por brevidade
```

### Deploy on-premise com Docker Compose

O stack minimo para Langfuse on-premise requer dois containers: o servidor Langfuse e um PostgreSQL para persistencia.

```yaml
# docker-compose.langfuse.yml
# Langfuse on-premise com persistencia

services:
  langfuse:
    image: langfuse/langfuse:2
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      SALT: ${LANGFUSE_SALT}
      NEXTAUTH_URL: http://localhost:3000
      TELEMETRY_ENABLED: "false"    # desabilita telemetria para ambiente air-gapped
    depends_on:
      langfuse-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/api/public/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  langfuse-db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_DB_PASSWORD}
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  langfuse_pgdata:
```

Variaveis de ambiente no `.env`:

```bash
LANGFUSE_DB_PASSWORD=uma_senha_forte_aqui
LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -base64 32)
LANGFUSE_SALT=$(openssl rand -base64 32)
```

No gateway, as variáveis de conexão:

```bash
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### Por que observabilidade importa em LLM

LLMs sao fundamentalmente diferentes de software deterministico. A mesma entrada pode produzir saidas diferentes. Sem observabilidade estruturada, voce enfrenta problemas que simplesmente nao existem em software tradicional:

1. **Debugging e impossivel sem traces.** Quando um usuario reporta "resposta errada", voce precisa ver: qual foi o prompt exato? Quais agentes foram acionados? Qual tool call falhou? Sem traces, a resposta e "nao sei".

2. **Drift de qualidade e silencioso.** Trocar de Q4 para Q8, atualizar o modelo, mudar o system prompt — qualquer alteracao pode degradar qualidade em cenarios especificos. Langfuse permite comparar scores de relevancia antes e depois.

3. **Otimizacao de prompts requer dados reais.** Prompts escritos no laboratorio funcionam diferente em producao. Com generations registradas, voce pode identificar padroes de falha e iterar com dados reais.

4. **Custo por request precisa ser medido.** Em infraestrutura on-premise, custo nao e por token como na OpenAI — mas tokens impactam latencia e throughput. Saber que a media e 1.200 tokens de entrada e 380 de saida permite dimensionar hardware.

5. **Reproducao de falhas.** Com o trace completo (prompt, contexto, model config), voce pode reproduzir qualquer falha em ambiente de desenvolvimento. Sem isso, bugs de LLM sao historias que ninguem consegue verificar.

---

## Avaliando a qualidade do RAG

Além das métricas de serving (TTFT, TPS, throughput), sistemas RAG precisam de métricas de **qualidade de resposta**. O RAG Cookbook define 4 métricas essenciais:

| Métrica | O que mede | Pergunta respondida |
|---------|-----------|---------------------|
| **Faithfulness** | A resposta é fiel ao contexto recuperado? | "O modelo inventou algo que não está nos documentos?" |
| **Context Precision** | Os chunks recuperados são relevantes? | "Dos 5 chunks retornados, quantos realmente importam?" |
| **Context Recall** | Recuperamos tudo que era relevante? | "Faltou algum documento importante?" |
| **Answer Relevance** | A resposta responde à pergunta? | "O modelo divagou ou foi direto ao ponto?" |

### Exemplo: medindo faithfulness

```python
# Avaliação de faithfulness com LLM-as-Judge
def avaliar_faithfulness(pergunta: str, contexto: list[str], resposta: str) -> float:
    """Usa um LLM para julgar se a resposta é fiel ao contexto."""
    prompt = f"""
    Pergunta: {pergunta}
    Contexto: {contexto}
    Resposta: {resposta}

    A resposta contém alguma informação que NÃO está presente no contexto?
    Responda apenas SIM ou NÃO.
    """
    julgamento = llm.generate(prompt)
    return 0.0 if "SIM" in julgamento else 1.0
```

### Valores de referência (AI-Orchestrator)

| Métrica | Baseline (7B) | LoRA 9B |
|---------|--------------|---------|
| Faithfulness | 0.82 | 0.89 |
| Context precision | 0.88 | 0.92 |
| Context recall | 0.85 | 0.87 |

> **Regra:** Faithfulness < 0.80 → revisar prompts e recuperação. Context precision < 0.80 → melhorar chunking. Context recall < 0.80 → aumentar top-K ou melhorar embeddings.

## Resumo

| Métrica | Ferramenta | Threshold |
|---------|-----------|-----------|
| TTFT | Benchmark custom | < 500 ms (bom) |
| TPS | Benchmark custom | > 30 (7B Q4) |
| Latência P95 | Prometheus | < 10s |
| Qualidade | Langfuse scores | > 0.7 média |
| Throughput | Locust | Depende da carga alvo |

---

## Referências

- Agrawal, A. et al. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. OSDI.
- *Benchmarking LLM serving systems* (PDF). Análise comparativa de vLLM, TGI, Ollama e llama.cpp.
- Langfuse Documentation. https://langfuse.com/docs
- Prometheus Documentation. https://prometheus.io/docs/
- Grafana Documentation. https://grafana.com/docs/
- Projeto AI-Orchestrator — `gateway/graph.py` (logs JSON estruturados por nó, integração Langfuse com trace/span/generation).
- *LLM Engineers Handbook* (PDF) — Capítulos sobre serving e monitoramento.
