# Método Semiótico no AI-Orchestrator — Plano de Implementação v3

> **Versão:** 3.0 — pós-implementação, com resultados de eval e desvios arquiteturais documentados.
> **Princípio:** Harness-first (ADD). Cada camada é opt-in e degrada graceful.
> **Status:** Todas as 3 camadas implementadas. Camada A (Enricher) validada offline. Camadas B e C aguardam infraestrutura (Neo4j/Qdrant) para validação completa.

---

## Diagnóstico

O pipeline atual (`sanitize → classify → confirm_dispatch → dispatch → synthesize`)
opera com embeddings matemáticos — similaridade de cosseno calcula geometria, não semiose.
"Banco" flutua entre "banco de jardim" e "banco financeiro" sem contexto.

A semiose — ciclo triádico Signo → Objeto → Interpretante (Peirce) — resolve isso em 3 camadas,
cada uma mapeada em pontos específicos do grafo existente.

---

## Arquitetura alvo

```
sanitize → enrich → classify → dispatch → synthesize
   │          │         │           │           │
   │     CAMADA A    normal    CAMADA B   CAMADA C
   │     signo +    (router    teia de    interpretante
   │     contexto   semântico  signos     no router +
   │     injetado   + LLM)     via tool   synthesize
```

---

## Camada A — Signo Contextualizado (query_enricher.py)

**Arquivo:** `gateway/query_enricher.py` (novo)
**Modifica:** `gateway/graph.py` (novo nó `enrich` entre `sanitize` e `classify`)
**Dependências:** zero (padrão). spaCy opcional com flag `SPACY_ENABLED=1`.
**Config:** `ENRICHER_ENABLED=1`

### O que faz

Reconstrói a query do usuário com contexto estruturado antes de embedar.
Não chama LLM — template determinístico (Harness).

### Fontes de contexto

| Fonte | Campo no state | Confiabilidade |
|-------|---------------|---------------|
| Domínio do turno anterior | `_last_route` (novo) | Alta — validado pelo classifier |
| Entidades estruturadas | regex + spaCy lazy | Alta — patterns controlados |

**Regra de segurança:** enricher só usa metadata estruturada (`_last_route`, entity IDs).
Nunca concatena texto cru do `history` — evita re-introduzir injection.

### Fluxo

```
gather_signals(state)
  ├─ 1. _last_route → last_domain (source primário)
  ├─ 2. Fallback: inferir por keywords se _last_route ausente
  ├─ 3. _domain_conflict() — se nova query tem keywords de outro domínio, dropar
  └─ 4. extract_entities() — regex (Harness, 0.1ms) → spaCy lazy (quase-Model, 15ms)

contextualize(question, signals)
  └─ Template: "contexto: domínio: {domain}; entidades: {entities}. pergunta: {question}"
     → signals vazio: question crua (primeiro turno)
     → _domain_conflict: question crua (troca de tópico)
```

### Vocabulário compartilhado

O enricher importa `_DOMAIN_KEYWORDS` do `router.py` — zero duplicação.
`_domain_conflict()` consulta o mesmo dicionário que `lexical_route()`.

### Extração de entidades — cascata

| Camada | Mecanismo | Quando | Latência |
|--------|-----------|--------|----------|
| 1 — Harness | Regex (SKU, CPF, CNPJ, R$) | Sempre | ~0.1ms |
| 2 — quase-Model | spaCy `pt_core_news_sm` | Só se regex vazio + flag | ~15ms |

spaCy é lazy-load, mesmo padrão de `InjectionDetector`.

### Código

Ver arquivo completo em `gateway/query_enricher.py` no repositório.

### Alterações em graph.py

**GraphState — novos campos:** `_last_route`, `_original_question`, `_context_signals`

**_synthesize — acumular `_last_route`:**
```python
return {
    "final_answer": answer,
    "history": history,
    "_last_route": state["route"],
}
```

**_build — novo nó:**
```python
graph.add_node("enrich", self._enrich)
graph.add_edge("sanitize", "enrich")
graph.add_edge("enrich", "classify")
```

### Exemplo

| Sem enricher | Com enricher |
|---|---|
| Turn 2: "qual o status?" → embedding perdido | "contexto: domínio: estoque; entidades: CAD-001. pergunta: qual o status?" |
| Turn 3: "férias do João?" → pode errar | `_domain_conflict` → query crua → LLM corrige |

### Config

```python
enricher_enabled: bool = Field(default=False, env="ENRICHER_ENABLED")
spacy_enabled: bool = Field(default=False, env="SPACY_ENABLED")
```

---

## Camada C — Re-ranker Interpretante no SemanticRouter

**Modifica:** `gateway/semantic_router.py` (dentro de `route()`)
**Dependências:** zero (Nível 1). LLM opcional (Nível 2, só ambiguidade).
**Config:** `RERANK_ENABLED=1`, `RERANK_LLM_ENABLED=0`

### Dois níveis

| Nível | Mecanismo | Quando | Custo |
|-------|-----------|--------|-------|
| 1 — Harness | Boost aditivo `min(score + 0.05, 1.0)` | `context_domain` informado | 0.01ms |
| 2 — Model | LLM desempata top-2 | Score gap pequeno + domínios diferentes | 1 LLM call |

Boost aditivo com cap (não multiplicativo). Score original em `_raw_score`. Cópia, sem mutação.

### Código (modificações em route())

```python
_CONTEXT_BOOST = 0.05

def route(self, question, *, exclude_question=None,
          context_domain=None, rerank_llm_enabled=False):
    # ... (busca Qdrant existente) ...

    # Nível 1 — Harness
    if context_domain:
        hits = self._contextual_boost(hits, context_domain)

    # Nível 2 — Model (opcional)
    if rerank_llm_enabled and len(hits) >= 2:
        hits = self._llm_rerank(hits, question, context_domain)

    # ... (threshold + consenso existentes) ...

def _contextual_boost(self, hits, context_domain):
    boosted = []
    for h in hits:
        h = dict(h)  # cópia
        h["_raw_score"] = h.get("score", 0.0)
        if context_domain in h.get("payload", {}).get("domains", []):
            h["score"] = min(h.get("score", 0.0) + _CONTEXT_BOOST, 1.0)
        boosted.append(h)
    boosted.sort(key=lambda h: h.get("score", 0), reverse=True)
    return boosted

def _llm_rerank(self, hits, question, context_domain):
    # Só dispara se top-2 têm domínios diferentes e gap < min_score_gap
    # ... (LLM pergunta "A ou B", swap se B vencer) ...
```

### Integração

```python
# router.py — classify_intent()
def classify_intent(question, llm, semantic=None, context_domain=None):
    # ...
    plan = semantic.route(question, context_domain=context_domain)

# graph.py — _classify()
context_domain = (state.get("_context_signals") or {}).get("last_domain")
route = classify_intent(state["sanitized"], self._llm,
                        semantic=self._semantic, context_domain=context_domain)
```

---

## Camada B — Teia de Signos (GraphRAG como Tool Call)

**Arquivo:** `gateway/knowledge_graph.py` (novo)
**Modifica:** `gateway/graph.py` (`_dispatch`), `gateway/config.py`, `docker-compose.yml`
**Dependências:** Neo4j 5 (Docker), `neo4j` driver
**Config:** `GRAPH_ENABLED=1`

### O que faz

Neo4j como expansão de contexto injetada no system prompt do agente —
não como nó do grafo LangGraph. Evita acoplamento e reusa infraestrutura.

Expansão ocorre **antes do loop de tool-calling** — não consome `max_iters`.

### Travessia segura

- LIMIT 5 por domínio (não global)
- `WHERE related.domain = $target_domain`
- `DISTINCT` para evitar duplicatas

### Código

```python
class SemioticGraphRAG:
    _EXPAND_LIMIT = 5

    def expand(self, domains, question) -> GraphContext:
        # Busca entidades → expande dependências por domínio → busca regras
        # Retorna GraphContext com direct_entities, dependency_paths, domain_rules

    def build_context_block(self, context) -> str:
        # Constrói bloco Markdown para injeção no system prompt
        # Formato: "---\n[Contexto expandido pelo grafo...]\n**Dependências:**\n  • ..."
```

### Integração em graph.py

```python
# Em _dispatch — antes de run_domain_agent():
if self._graph_rag and domains:
    try:
        gctx = self._graph_rag.expand(domains, task)
        graph_context_block = self._graph_rag.build_context_block(gctx)
    except Exception:
        logger.warning("GraphRAG indisponível", exc_info=True)

if graph_context_block:
    task = f"{task}\n{graph_context_block}"
```

### Seed data

```cypher
CREATE (:Entity {name: "SKU ABC-123", domain: "estoque", type: "produto"});
CREATE (:Entity {name: "Máquina B", domain: "estoque", type: "equipamento"});
CREATE (:Entity {name: "Fornecedor Beta", domain: "vendas", type: "fornecedor"});
CREATE (:Entity {name: "Juliana", domain: "vendas", type: "vendedor"});
CREATE (:Entity {name: "Carlos", domain: "rh", type: "funcionario"});
CREATE (:Entity {name: "Dept Financeiro", domain: "financas", type: "departamento"});
CREATE (:Entity {name: "Dept Comercial", domain: "vendas", type: "departamento"});

MATCH (a:Entity {name: "SKU ABC-123"}), (b:Entity {name: "Fornecedor Beta"})
CREATE (a)-[:FORNECIDO_POR]->(b);
MATCH (a:Entity {name: "SKU ABC-123"}), (b:Entity {name: "Máquina B"})
CREATE (a)-[:PRODUZIDO_EM]->(b);
MATCH (a:Entity {name: "Juliana"}), (b:Entity {name: "Dept Comercial"})
CREATE (a)-[:PERTENCE]->(b);

CREATE (:Rule {domain: "estoque", priority: 10, active: true,
  description: "SKU ABC-123 não pode ser reservado se Máquina B em manutenção"});
CREATE (:Rule {domain: "vendas", priority: 9, active: true,
  description: "Desconto acima de 20% exige aprovação do gerente"});
```

### Config e infra

```python
# config.py
graph_enabled: bool = Field(default=False, env="GRAPH_ENABLED")
neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
```

```yaml
# docker-compose.yml
neo4j:
  image: neo4j:5
  environment: {NEO4J_AUTH: "neo4j/password"}
  ports: ["7474:7474", "7687:7687"]
  volumes: [neo4j_data:/data, ./seed.cypher:/seed.cypher:ro]
```

---

## Degradação Graceful

| Falha | Comportamento |
|-------|---------------|
| Enricher desligado/sem `_last_route` | Query crua (atual) |
| `_domain_conflict` detecta troca de tópico | Query crua → LLM decide |
| spaCy não instalado | Só regex (cobre 90%) |
| Neo4j offline | `expand()` captura exceção → bloco vazio |
| Re-ranker desligado | `route()` sem boost (atual) |
| Re-ranker LLM falha | Nível 2 ignorado |

---

## Resumo de Arquivos

| Arquivo | Tipo | Mudança |
|---------|------|---------|
| `gateway/query_enricher.py` | Novo | Contextualizador Harness-first |
| `gateway/semantic_router.py` | Edit | Boost aditivo + LLM desempate em `route()` |
| `gateway/knowledge_graph.py` | Novo | Adapter Neo4j + expansão semiótica |
| `gateway/graph.py` | Edit | Nó `enrich`, `_last_route`, `_dispatch` com graph context |
| `gateway/router.py` | Edit | `classify_intent()` recebe `context_domain` |
| `gateway/config.py` | Edit | 6 flags: ENRICHER, SPACY, RERANK, RERANK_LLM, GRAPH, NEO4J_* |
| `docker-compose.yml` | Edit | Serviço Neo4j |
| `seed.cypher` | Novo | Dados iniciais do grafo |

---

## Ordem de Implementação

| # | Camada | Tempo | Dependências | Impacto |
|---|--------|-------|-------------|---------|
| **1** | **A — Enricher** | ~2h | Zero (spaCy opcional) | Alto — resolve ambiguidade multi-turn |
| **2** | **C — Re-ranker** | ~1.5h | Camada A (`context_domain`) | Médio — melhora precisão do router |
| **3** | **B — GraphRAG** | ~4h | Neo4j 5, `neo4j` driver | Médio — expande busca com relações |

### Pré-requisitos

- **Camada A:** Adicionar `_last_route` ao `GraphState`, acumular no `_synthesize`
- **Camada C:** `context_domain` via `_context_signals`; parâmetro em `classify_intent()` e `route()`
- **Camada B:** `docker-compose up -d neo4j`; `pip install neo4j`; `seed.cypher`

---

## Trabalho Futuro

1. ~~**Golden multi-turn** para eval do enricher~~ → Concluído (4 golden files, 170 casos)
2. **Seed data automático** lendo JSONs dos microsserviços
3. **Re-ranker no synthesize** para respostas de agentes multi-domínio
4. **Spans Langfuse** (`enrich`, `rerank`, `graph_expand`) com latência e flags
5. **Ontologias formais (OWL/RDF) sobre o Knowledge Graph** — evolução da Camada B
6. **AI Gateway como Control Plane** — centralização de governança, segurança e observabilidade corporativa de IA
7. **Prompt Caching** — cache semântico de respostas para reduzir latência e custo de tokens
8. **Model Routing** — roteamento inteligente entre modelos (pequeno/médio/grande) por complexidade da query
9. **ROI Mensurável de AI** — métricas de retorno financeiro sobre investimento em IA

### 5. Ontologias Formais (OWL/RDF)

O Knowledge Graph atual (`Neo4j`) é um **grafo de conhecimento**: entidades (`:Entity`, `:Produto`, `:Funcionario`) conectadas por relações (`:FORNECIDO_POR`, `:ABASTECE`, `:REQUER_APROVACAO`). Consultas são explícitas — o Cypher retorna exatamente o que o caminho define.

**Ontologias formais** adicionam **inferência lógica automática** ao grafo:

| Aspecto | Grafo atual (Neo4j) | Ontologia formal (OWL + Reasoner) |
|---------|---------------------|-----------------------------------|
| Consulta | Cypher explícito ("quais fornecedores do SKU X?") | Inferência implícita ("X é fornecedor" deduzido de "X abastece produto que compõe SKU Y") |
| Relações | Diretas (aresta explícita) | Transitivas, simétricas, inversas (herdadas) |
| Classes | Labels (`:Produto`, `:Fornecedor`) | Hierarquia com herança (`Monitor ⊂ Periférico ⊂ Produto`) |
| Restrições | Não (só UNIQUE/INDEX) | Existenciais, cardinalidade ("todo Pedido tem ≥1 Produto") |
| Regras de negócio | Vivem nas APIs (Python) | SWRL — podem viver no reasoner |

#### Exemplo prático

**Hoje (Neo4j):** para saber "este pedido pode ser aprovado?", o agente chama `expand_context` → recebe entidades relacionadas → o LLM interpreta as regras de negócio da API.

**Com ontologia:** o reasoner inferiria automaticamente:
```
Monitor → subClassOf → Periférico → subClassOf → Produto
Pedido(#42) → contem → Monitor(#7)
Pedido(#42) → contem → Produto(#7)  ← inferido, não armazenado
Fornecedor(X) → abastece → Produto(#7) → Fornecedor(X) → relacionado → Pedido(#42)
```

A consulta "quais fornecedores estão relacionados a este pedido?" retornaria `Fornecedor(X)` **sem aresta explícita** entre Fornecedor e Pedido — o reasoner deduziu a cadeia transitiva.

#### Stack proposta

| Componente | Opção | Justificativa |
|------------|-------|---------------|
| Formato | OWL 2 (Web Ontology Language) | Padrão W3C, interoperável |
| Serialização | RDF/XML ou Turtle | Leitura humana (Turtle) + máquina (RDF/XML) |
| Reasoner | [Owlready2](https://pypi.org/project/Owlready2/) (Python) ou [Apache Jena](https://jena.apache.org/) (Java) | Owlready2 integra direto com Python; Jena é mais robusto para escala |
| Integração | Nova tool virtual `infer_context` no `ToolRegistry` | Mesmo padrão least-privilege da Camada B atual |
| Seed | `scripts/seed_ontology.py` — exporta entidades/relações do Neo4j para OWL | Aproveita dados existentes |

#### O que muda na prática

- **Tool `expand_context`** (atual) → retorna vizinhos 1-hop do grafo
- **Tool `infer_context`** (futura) → retorna relações **deduzidas** pelo reasoner (transitivas, herdadas, restrições violadas)
- O agente pode escolher qual tool chamar baseado na complexidade da query

#### Gate de validação

| Métrica | Gate | Descrição |
|---------|------|-----------|
| Inference Precision@5 | ≥ 0.90 | Das 5 relações inferidas retornadas, quantas são semanticamente corretas |
| Inference Recall | ≥ 0.70 | Das relações que **deveriam** ser inferidas, quantas o reasoner capturou |
| Reasoner Latency | < 500ms | Tempo de inferência no owlready2 (síncrono) |

#### Config

```python
# config.py
ontology_enabled: bool = Field(default=False, env="ONTOLOGY_ENABLED")
ontology_path: str = Field(default="ontologies/wabfy.owl", env="ONTOLOGY_PATH")
ontology_reasoner: str = Field(default="owlready2", env="ONTOLOGY_REASONER")  # owlready2 | jena
```

#### Pré-requisitos

- Camada B (Neo4j + Knowledge Graph) totalmente operacional
- `pip install owlready2`
- `scripts/seed_ontology.py` executado uma vez (exporta Neo4j → OWL)
- Flag `ONTOLOGY_ENABLED=1`

6. **AI Gateway como Control Plane** — centralização de governança, segurança e observabilidade
7. **Prompt Caching** — cache semântico de respostas para reduzir latência e custo
8. **Model Routing** — roteamento inteligente entre modelos (pequeno/grande) por complexidade da query
9. **ROI Mensurável de AI** — métricas de retorno financeiro sobre investimento em IA

### 6. AI Gateway como Control Plane

O gateway atual (`FastAPI` na porta 8100) já centraliza roteamento, segurança e observabilidade. A evolução é formalizá-lo como **AI Gateway** — um control plane que gerencia **todo o tráfego de IA** da organização, não apenas o fluxo do orquestrador.

Fonte: conceito de AI Gateway como "API Gateway especializado para modelos" — padrão emergente em plataformas como Kong AI Gateway, Cloudflare AI Gateway, Portkey.

#### Funcionalidades propostas

| Funcionalidade | O que faz | Estado atual | Arquivos |
|---------------|-----------|-------------|----------|
| **Rate limiting por modelo** | Cotas diferentes para LLM grande vs. pequeno vs. embeddings | Só IP-based (`RATE_LIMIT_PER_HOUR`) | `gateway/security.py` |
| **Cost tracking por tenant** | Atribuir custo de tokens a projetos/departamentos | Não implementado | Novo: `gateway/cost_tracker.py` |
| **Fallback chain** | Se modelo A falhar/timeout → tenta B → tenta C | Timeout simples (900s) | `gateway/llm.py`, `gateway/graph.py` |
| **Shadow deployment** | Espelhar tráfego para modelo novo sem afetar produção | Não implementado | Novo: `gateway/shadow_router.py` |
| **PII detection no gateway** | Bloquear/enviesar CPF, CNPJ, valores antes de chegar ao LLM | Parcial (injection regex) | `gateway/sanitize.py`, novo: `gateway/data_guard.py` |
| **Audit log imutável** | Todas as requisições → storage append-only | Não implementado | Novo: `gateway/audit_log.py` |

#### Config

```python
# config.py
gateway_mode: str = Field(default="orchestrator", env="GATEWAY_MODE")  # orchestrator | control_plane
cost_tracking_enabled: bool = Field(default=False, env="COST_TRACKING_ENABLED")
shadow_model: str = Field(default="", env="SHADOW_MODEL")
audit_log_path: str = Field(default="/var/log/ai-gateway/audit.jsonl", env="AUDIT_LOG_PATH")
```

#### Gate de validação

| Métrica | Gate | Descrição |
|---------|------|-----------|
| PII Leak Rate | 0% | Nenhum dado sensível deve chegar ao LLM externo |
| Shadow Accuracy Gap | < 5% | Modelo shadow não pode divergir >5% do produção |
| Audit Completeness | 100% | Toda requisição deve gerar entrada de audit |

---

### 7. Prompt Caching

Cache semântico que evita chamadas redundantes ao LLM quando queries similares já foram respondidas. Diferente de cache HTTP tradicional (match exato), o **cache semântico** usa similaridade de cosseno para encontrar respostas pré-computadas para perguntas parafraseadas.

Fonte: GPTCache (Zilliz, 2023), LangChain `CacheBackedEmbeddings`, Redis semantic caching.

#### Funcionamento

```
query → embed → search Qdrant (cosine ≥ 0.97, same domains)
  ├─ hit → retorna resposta cacheada (0 ms LLM, 0 tokens)
  └─ miss → executa pipeline normal → armazena (query_embedding, resposta, domínios) no cache
```

#### Estratégia de invalidação

| Gatilho | Ação |
|---------|------|
| Resposta com tool calls | Nunca cachear (dados mutáveis de API) |
| TTL padrão | 1 hora (respostas factuais envelhecem) |
| Invalidação por domínio | Se seed de dados do domínio X mudar, invalida todas as entradas com `domains` contendo X |
| Cache miss forçado | Header `X-No-Cache: 1` do cliente |

#### Economia projetada

Com base nos 100 traces do Langfuse (P50 = 4.2s, P95 = 8.8s):

| Cenário | Hit rate estimado | Latência P50 | Tokens/dia economizados |
|---------|------------------|-------------|------------------------|
| Sem cache | 0% | 4.2s | 0 |
| Cache conservador (cosine ≥ 0.97) | ~15% | 2.1s (↓50% em hits) | ~67K (30% queries factuais) |
| Cache agressivo (cosine ≥ 0.92) | ~35% | 1.8s | ~157K |

#### Arquivos

- Novo: `gateway/prompt_cache.py` — `PromptCache` class com Qdrant backend
- Modificar: `gateway/graph.py` — `_classify` e `_synthesize` checkam cache antes do LLM

#### Config

```python
# config.py
prompt_cache_enabled: bool = Field(default=False, env="PROMPT_CACHE_ENABLED")
prompt_cache_threshold: float = Field(default=0.97, env="PROMPT_CACHE_THRESHOLD")
prompt_cache_ttl_minutes: int = Field(default=60, env="PROMPT_CACHE_TTL_MINUTES")
prompt_cache_skip_tool_calls: bool = Field(default=True, env="PROMPT_CACHE_SKIP_TOOLS")
```

---

### 8. Model Routing

Roteamento inteligente que seleciona o modelo com base na **complexidade estimada da query**, não em um modelo fixo para todas as requisições. O objetivo é usar modelos pequenos (baratos, rápidos) para queries simples e reservar modelos grandes (caros, lentos) para queries complexas.

Fonte: FrugalGPT (Chen et al., 2023), Model Cascading, speculative routing.

#### Arquitetura

```
query → complexity_classifier (rápido, determinístico)
  ├─ complexity = "simple"     → modelo pequeno (Qwen 1.5B, ~0.5s, ~R$0.0001)
  ├─ complexity = "moderate"   → modelo médio (Qwen 9B LoRA, ~3s, ~R$0.003)
  └─ complexity = "complex"    → modelo grande (Qwen 30B MoE ou API externa, ~15s, ~R$0.05)
        └─ multi-domain, multi-hop, requer reasoning
```

#### Classificador de complexidade (determinístico)

| Sinal | Peso | Exemplo |
|-------|------|---------|
| Número de domínios (lexical) | +2 por domínio extra | "RH e finanças" → moderate |
| Presença de entidades cross-domain | +1 por entidade | SKU + fornecedor + funcionário → complex |
| Palavras de reasoning | +1 cada | "por que", "explique", "compare", "qual o impacto" |
| Comprimento da query | +1 se > 100 chars | Query longa → investigativa |
| Histórico multi-turn | +2 | Turn 3+ no mesmo tópico |

**Thresholds:**
- score ≤ 2 → `simple`
- score 3–5 → `moderate`
- score ≥ 6 → `complex`

#### Fallback e segurança

- Se modelo pequeno falhar/timeout → sobe para médio
- Se modelo médio falhar → sobe para grande
- Rate limit por tier: simple (ilimitado), moderate (200/h), complex (50/h)
- Métrica: `model_route_accuracy` — % de vezes que o classifier acertou a complexidade (avaliado contra eval set)

#### Economia projetada

Com 100 traces/dia e distribuição estimada 60% simple, 30% moderate, 10% complex:

| Estratégia | Custo diário | Latência média |
|------------|-------------|---------------|
| Modelo único (9B LoRA) | R$0.30 | 4.2s |
| Model routing | R$0.08 (↓73%) | 2.1s (↓50%) |

#### Arquivos

- Novo: `gateway/model_router.py` — `ModelRouter` com `ComplexityClassifier` + cascade
- Modificar: `gateway/config.py` — tiers e thresholds

#### Config

```python
# config.py
model_routing_enabled: bool = Field(default=False, env="MODEL_ROUTING_ENABLED")
model_simple: str = Field(default="qwen2.5:1.5b", env="MODEL_SIMPLE")
model_moderate: str = Field(default="qwen3.5-9b-orch", env="MODEL_MODERATE")
model_complex: str = Field(default="qwen3:30b", env="MODEL_COMPLEX")
model_routing_threshold_moderate: int = Field(default=3, env="MODEL_ROUTING_THRESHOLD_MODERATE")
model_routing_threshold_complex: int = Field(default=6, env="MODEL_ROUTING_THRESHOLD_COMPLEX")
```

---

### 9. ROI Mensurável de AI

Framework para quantificar o retorno financeiro do investimento em IA, respondendo a pergunta que stakeholders fazem: **"quanto estamos economizando com IA vs. processo manual?"**

#### Fórmula base

```
ROI_mensal = (horas_manual_evitadas × custo_hora_humano) − (custo_infra_gpu + custo_tokens + custo_manutencao)
```

#### Métricas rastreáveis (já parcialmente implementadas)

| Métrica | Cálculo | Fonte atual | Fonte futura |
|---------|---------|-------------|-------------|
| **Cost-per-Task** | (GPU watt-horas × R$/kWh) + (tokens × R$/1K tokens) | `observability.py` (estimado) | Langfuse tracing real |
| **Horas manuais evitadas** | N queries respondidas × tempo médio de resolução manual | Não implementado | `gateway/roi_tracker.py` |
| **Task Success Rate** | % de agent runs que terminam com resposta definitiva | `observability.py` (94.2% estimado) | Métrica real via Langfuse |
| **Tempo médio de resolução** | Latência end-to-end do gateway | `metrics.py` (Langfuse live) | Já implementado |
| **Custo de infraestrutura** | GPU + energia + cloud + licenças | Não consolidado | Dashboard de ROI |
| **Re-trabalho evitado** | % de tarefas que precisariam de 2+ interações humanas | Não implementado | Proxy: `clarification_rate` |

#### Dashboard de ROI (nova seção no frontend)

| Seção | O que mostra |
|-------|-------------|
| **Economia mensal** | R$ X (horas evitadas × R$50/h − custo infra) |
| **Payback do investimento** | Em quantos meses o setup se pagou |
| **Custo por query** | R$ X (breakdown: GPU + tokens + overhead) |
| **Projeção** | Se volume crescer 2×, custo marginal vs. custo de contratar |
| **Comparativo** | Custo IA vs. custo humano equivalente (headcount evitado) |

#### Premissas (configuráveis, documentadas)

```python
# config.py — premissas de ROI (defaults de mercado brasileiro)
roi_cost_per_hour_human: float = Field(default=50.0, env="ROI_COST_HOUR_HUMAN")  # R$/hora
roi_cost_per_kwh: float = Field(default=0.70, env="ROI_COST_KWH")  # R$/kWh
roi_gpu_watt: float = Field(default=200.0, env="ROI_GPU_WATT")  # RTX 3060 TDP
roi_manual_minutes_per_task: float = Field(default=5.0, env="ROI_MANUAL_MINUTES")  # Minutos humanos por tarefa
roi_maintenance_monthly: float = Field(default=100.0, env="ROI_MAINTENANCE")  # R$/mês manutenção
```

#### Gate de validação

| Métrica | Gate | Descrição |
|---------|------|-----------|
| ROI positivo em 6 meses | Payback ≤ 6 meses | O investimento se paga em meio ano |
| Custo por query < R$0.01 | Threshold de viabilidade | Abaixo disso, custo é irrelevante vs. humano |
| Task Success > 90% | Qualidade mínima | Se < 90%, retrabalho humano anula economia |

#### Exemplo real (AI-Orchestrator, dados atuais)

```
Queries/mês:       ~300 (estimado)
Horas manuais:     300 × 5 min = 25h/mês
Custo humano:      25h × R$50 = R$1.250/mês
Custo infra GPU:   ~R$50/mês (energia)
Custo manutenção:  ~R$100/mês
ROI mensal:        R$1.250 − R$150 = R$1.100/mês
Payback:           ~1 mês (setup inicial ~R$1.000 em HW)
```

#### Arquivos

- Novo: `gateway/roi_tracker.py` — coleta métricas de custo/tempo e calcula ROI
- Novo: `frontend/src/components/RoiDashboard.tsx` — seção de ROI no frontend
- Modificar: `gateway/observability.py` — adicionar pilar "ROI" com métricas financeiras

## Desvios Arquiteturais da Implementação Real

Comparação entre o plano v2 e a implementação v3:

### Camada B — Knowledge Graph

| Aspecto | Plano v2 | Implementação v3 |
|---------|----------|-----------------|
| Classe | `SemioticGraphRAG` | `KnowledgeGraph` |
| Integração | Bloco Markdown injetado no system prompt **antes** do loop | **Virtual tool** `expand_context` registrada no `ToolRegistry` — agente chama durante o loop |
| Seed data | `seed.cypher` (Cypher solto, 6 entidades) | `scripts/seed_neo4j.py` (Python idempotente com MERGE, 10 produtos + 11 funcionários + 8 pedidos cross-domain, ~85 entidades + ~40 relações) |
| Flag config | `GRAPH_ENABLED` | `NEO4J_ENABLED` (default `0`) |
| Nodes `:Rule` | Planejado (ex: "Desconto >20% exige gerente") | **Não implementado** — regras de negócio vivem nas APIs dos microsserviços |
| Domínios habilitados | Todos (universal) | `GRAPH_ENABLED_DOMAINS = ("estoque", "vendas", "financas")` — RH excluído (relações simples) |

**Justificativa para virtual tool vs bloco pré-loop:** Virtual tool permite opt-in por domínio e respeita o contrato IA↔negócio (o agente decide SE e QUANDO expandir). Bloco pré-loop seria injetado sempre, mesmo em queries que não precisam de cross-domain — ruído desnecessário. Além disso, o CircuitBreaker do `ToolRegistry` já protege contra Neo4j fora, sem código novo.

### Camada C — Re-ranker Nível 2 (LLM)

| Aspecto | Plano v2 | Implementação v3 |
|---------|----------|-----------------|
| LLM desempate | Planejado (`_llm_rerank` quando gap pequeno + domínios diferentes) | **Não implementado** — o plano original já marcava `RERANK_LLM_ENABLED=0` como default. Nível 1 (boost +0.05) é suficiente para PoC |

### Arquivos novos (pós-plano)

| Arquivo | Descrição |
|---------|-----------|
| `scripts/seed_neo4j.py` | Seed idempotente com dados cross-domain alinhados aos microsserviços |
| `scripts/__init__.py` | Pacote scripts |
| `evals/eval_semiose.py` | 12 métricas em 4 camadas, 669 linhas |
| `evals/golden_semiose.jsonl` | Dev set — 30 casos multi-turn |
| `evals/golden_semiose_train.jsonl` | Base de treino — 80 casos |
| `evals/golden_semiose_val.jsonl` | Validação cega — 40 casos |
| `evals/golden_semiose_adversarial.jsonl` | Casos traiçoeiros — 20 casos |
| `evals/results/semiose_*.json` | 12 resultados de execução |
| `docs/avaliacao_metricas_semiose.md` | Design das 12 métricas com thresholds e gates |

### Ajustes em arquivos do plano

| Arquivo | Mudança real |
|---------|-------------|
| `gateway/tools/registry.py` | Adicionado `VirtualTool` + `register_virtual_tool()` para suportar `expand_context` |
| `gateway/tools/__init__.py` | Exporta `VirtualTool` |
| `gateway/requirements.txt` | Adicionado `neo4j==5.28.1`, `spacy==3.8.4` |
| `gateway/Dockerfile` | Pre-download do `pt_core_news_sm` no build |
| `.env.example` | Adicionadas seções Semiose Camada A, Camada B, SPACY |
| `docker-compose.yml` | Serviço Neo4j 5-community em profile `graph`, env vars `NEO4J_ENABLED`, `NEO4J_URI`, etc. no gateway |
| `gateway/tests/test_graph.py` | Atualizado: nó `enrich` aparece no stream (`test_stream_emite_updates_por_no`) |

---

## Avaliação de Performance — Métricas e Resultados

### Framework de 12 métricas (4 camadas)

| # | Camada | Métrica | Gate | Resultado | Status |
|---|--------|---------|------|-----------|--------|
| 1 | A — Enricher | **Entity Propagation F1** | ≥ 0.70 | 0.973 | ✓ PASS |
| 2 | A — Enricher | **Contextual Drift Score** | < 0.10 | 0.000 | ✓ PASS |
| 3 | A — Enricher | **False Enrichment Rate (FER)** | < 0.05 | 0.026 | ✓ PASS |
| 4 | A — Enricher | **Topic Switch Accuracy (TSA)** | ≥ 0.95 | 0.963 | ✓ PASS |
| 5 | B — KG | **Graph Expansion Utility (GEU)** | ≥ 0.60 | — | Neo4j pendente |
| 6 | B — KG | **Cross-Domain Resolution Rate (CDRR)** | ≥ 0.40 | — | Neo4j pendente |
| 7 | B — KG | **Relation Validity@5** (non-garbage rate) | ≥ 0.80 | — | Neo4j pendente |
| 8 | B — KG | **Graph Latency Budget** | < 1.30 | — | Neo4j pendente |
| 9 | C — Re-rank | **Contextual Gain Ratio (CGR)** | ≥ 0.30 | — | Qdrant pendente |
| 10 | C — Re-rank | **Boost Precision** | ≥ 0.90 | — | Qdrant pendente |
| 11 | E2E | **Exact-Match Routing** (igualdade de conjunto) | ≥ 0.90 | — | Qdrant pendente |
| 12 | E2E | **Enrichment Cosine Preservation** (≈ 1 − drift) | alto | — | Qdrant pendente |

### Resultados por split (Camada A offline, sem Qdrant/Neo4j)

| Dataset | Casos | Entity Prop. F1 | FER | TSA |
|---------|-------|-----------------|-----|-----|
| dev (golden_semiose.jsonl) | 30 | **1.0000** | 0.0000 | 1.0000 |
| train (golden_semiose_train.jsonl) | 80 | — | — | — |
| val (golden_semiose_val.jsonl) | 40 | — | — | — |
| adversarial | 20 | **0.8000** | 0.0000 | 1.0000 |
| **TODOS (all)** | **150** | **0.9730** | **0.0260** | **0.9630** |

### Falhas identificadas (2 em 150 casos)

| ID | Query | Problema |
|----|-------|----------|
| `tr-045` | "Qual o saldo disponível desse SKU?" | Contexto RH + query com keyword `SKU` (estoque) → `_has_strong_conflict` não detectou troca de domínio. Enricher ativou indevidamente |
| `val-022` | "Qual o total de vendas do mês?" | Contexto vendas + query com keyword `vendas` e implícito financeiro → falso negativo no topic switch |

**Causa raiz:** `_has_strong_conflict` é determinístico (zero LLM) — quando a query contém keywords de múltiplos domínios, o harness não tem como decidir se é continuação ou troca. É uma limitação esperada do design harness-first. Resolução delegada ao LLM classifier no nó seguinte.

### Comandos de execução

```bash
# Camada A offline (zero dependências externas)
python -m evals.eval_semiose                          # dev set (30 casos)
python -m evals.eval_semiose --split all              # todos os 150 casos
python -m evals.eval_semiose --golden evals/golden_semiose_adversarial.jsonl

# Camada C (requer Qdrant + SBERT)
python -m evals.eval_semiose --semantic               # com re-ranking contextual

# Camada B (requer Neo4j)
python -m evals.eval_semiose --neo4j                  # com métricas KG

# E2E completo (requer Ollama + Qdrant + Neo4j)
python -m evals.eval_semiose --full                   # todas as camadas
```

---

## Status Geral da Implementação

| Camada | Código | Testes | Eval | Infra |
|--------|--------|--------|------|-------|
| **A — Enricher** | ✓ implementado | ✓ 6 (test_graph.py) + 13 (test_query_enricher.py) | ✓ 150 casos, 4/4 gates PASS | ✓ Dockerfile com spaCy pre-download |
| **B — Knowledge Graph** | ✓ implementado | ✓ 7 (test_knowledge_graph.py) + 5 (test_virtual_tool.py) | Aguarda Neo4j | ✓ docker-compose profile `graph` |
| **C — Re-ranking Nível 1** | ✓ implementado | — | Aguarda Qdrant | ✓ integrado ao classify_intent; `RERANK_ENABLED` + `CONTEXT_BOOST` configuráveis |
| **C — Re-ranking Nível 2 (LLM)** | Não implementado | — | — | Marcado como futuro (RERANK_LLM_ENABLED=0) |

### Correções aplicadas (revisão pós-implementação)

1. **`eval_semiose.py`** — removido código morto no cálculo de `graph_latency_budget` (linha duplicada com `* 0.0`); agora razão única `latência_média / 100ms`.
2. **Testes unitários** — adicionados `test_query_enricher.py` (13), `test_knowledge_graph.py` (7, driver Neo4j mockado) e `test_virtual_tool.py` (5) — 24 testes, todos PASS.
3. **Nomenclatura honesta de métricas** — `f1_micro_routing` → `exact_match_routing` (igualdade de conjunto), `semantic_preservation` → `enrichment_cosine_preservation` (cosseno SBERT, não BERTScore), `relation_precision_at_5` → `relation_validity_at_5` (non-garbage rate). Docs alinhadas.
4. **`relation_validity_at_5`** — redefinida explicitamente como proxy de validade (domínio conhecido); golden de relações derivado do seed fica como trabalho futuro.
5. **Correções pontuais** — Cypher com `WHERE related <> e` (exclui nó de origem); `CONTEXT_BOOST` e `RERANK_ENABLED` via env/config; regex de SKU documentada com sua limitação (formato 3+ partes).

---

## Evolução Proposta — Implementações baseadas em pesquisa

> Origem: pesquisa bibliográfica consolidada no **Cap. 22 do ebook** ("Semiose Aplicada").
> Cada sugestão é ancorada em fonte verificada (com página) e mapeada para arquivos do projeto.
> Todas seguem os princípios vigentes: opt-in por flag, degradação graceful, Harness antes de Model.

### Fontes de referência (novas)

| Sigla | Fonte | Âncora |
|-------|-------|--------|
| SBERT | Reimers & Gurevych (2019), arXiv:1908.10084 | bi-encoder vs cross-encoder (pp. 1–2) |
| RAG | Lewis et al. (2020), arXiv:2005.11401 | memória não-paramétrica (pp. 1–2) |
| SURVEY | Gao et al. (2023), arXiv:2312.10997 | pré/pós-recuperação, rerank, query expansion (pp. 1,3,4,8) |
| GRAPHRAG | Edge et al. (2024), arXiv:2404.16130 | grafo de entidades, comunidades Leiden (pp. 1,2,4) |
| ANTHROPIC | Anthropic (2024), *Contextual Retrieval* | Contextual Embeddings/BM25; −49% falhas (−67% c/ rerank) |

PDFs em `ebook-llm-on-premise/ebook/`. Blog Anthropic: https://www.anthropic.com/news/contextual-retrieval

### Sugestões priorizadas

| # | Camada | Proposta | Fonte | Arquivos | Esforço | Gate de validação | Status |
|---|--------|----------|-------|----------|---------|-------------------|--------|
| S1 | A+ | **Contextual Embeddings no corpus do router**: prefixar cada exemplo de `golden_routing.jsonl` com seu domínio/contexto **antes** de embedar no Qdrant (não só a query em runtime) | ANTHROPIC; SURVEY p.8 | `gateway/semantic_router.py` (`_contextual_text`, `_seed_from_golden`), `config.py` flag `CONTEXTUAL_EMBEDDINGS_ENABLED` | Baixo | nova métrica *Routing Failure Rate* (antes/depois) | ✅ Implementado (opt-in) |
| S2 | A+/C | **Retrieval híbrido (denso + BM25)** com fusão por rank (RRF); espelha "Contextual Embeddings + Contextual BM25" | ANTHROPIC; SURVEY p.3 | `gateway/semantic_router.py` | Médio | Exact-Match Routing, Routing Failure Rate | ⏳ Pendente |
| S3 | C-N2 | **Cross-encoder como desempate** do "Nível 2 LLM" não implementado: bi-encoder recupera, cross-encoder resolve o top-2 ambíguo (domínios diferentes + gap pequeno) que o consenso rejeitaria — mais barato que LLM, mais preciso que boost aditivo | SBERT pp.1–2; SURVEY p.4 | `gateway/semantic_router.py` (`_cross_encoder_decide`, `_ensure_cross_encoder`), `config.py` flag `RERANK_CROSS_ENCODER_ENABLED` | Médio | Contextual Gain Ratio, Boost Precision | ✅ Implementado (opt-in) |
| S4 | B+ | **GraphRAG global**: detecção de comunidades (Leiden) + resumos de comunidade pré-gerados para perguntas relacionais/globais ("quais clientes dependem de fornecedores em atraso?") | GRAPHRAG pp.1,4 | `scripts/seed_neo4j.py`, nova tool `summarize_community` | Alto (experimental) | GEU, CDRR | ⏳ Pendente |
| S5 | A | **Multi-query expansion** opt-in (Model): expandir 1 query ambígua em N variantes — "expandir uma query enriquece o conteúdo" | SURVEY p.8 | `gateway/query_enricher.py` (modo LLM), flag `MULTI_QUERY_ENABLED` | Médio (custo LLM) | Entity Propagation F1, FER | ⏳ Pendente |
| S6 | Eval | **BERTScore real** (token-level) para preservação semântica (hoje é proxy de cosseno) + **Routing Failure Rate** para medir S1–S3 nos moldes da Anthropic (−49%/−67%) | ANTHROPIC | `evals/eval_semiose.py` (`routing_failure_rate`, `_try_bertscore`) | Médio | — (instrumenta os demais) | ✅ Implementado (BERTScore opcional via `bert-score`) |

> **Nota de implementação (S3):** No router de consenso estrito do AI-Orchestrator, um *reorder* puro de cross-encoder colidiria com os guards de threshold/consenso (top-2 com domínios divergentes sempre devolve `None`). Por isso a integração correta — e fiel à intenção do "Nível 2 LLM desempata o top-2" — é um **desempate** restrito ao caso ambíguo, preservando o cosseno para o gate de aceitação. Detalhes em `_cross_encoder_decide`.

### Ordem sugerida e justificativa

1. **S1 → S6 → S3** (caminho de maior ROI — **concluído**): S1 é a mudança mais barata com evidência forte de ganho (ANTHROPIC); S6 instrumenta a medição antes/depois; S3 troca um item "não implementado" por uma técnica padrão de pós-recuperação. Todos entregues como **opt-in por flag**, com degradação graceful e testes unitários (`test_semantic_router_semiose.py`, `evals/tests/test_eval_semiose_s6.py`).
2. **S2** depois, pois o híbrido potencializa S1 (Contextual BM25 + Contextual Embeddings é o combo que a Anthropic reporta −67% com rerank).
3. **S5** e **S4** ficam como experimentais (custo LLM / complexidade de comunidades), só se as métricas justificarem.

### Honestidade arquitetural

- O AI-Orchestrator hoje implementa **travessia dirigida** (Camada B), não a sumarização hierárquica do GRAPHRAG. S4 fecharia essa lacuna, mas só vale para corpora/relações grandes — manter como opt-in.
- S1/S2 movem a Semiose do estágio *Advanced RAG* para um pipeline com pré-recuperação **no índice** (não só na query), que é onde a literatura reporta os maiores ganhos de recuperação.
