# Capítulo 23 — Roadmap de Evolução: do Orquestrador à Plataforma de IA Corporativa

O AI-Orchestrator, como descrito nos capítulos 19 (multi-agente) e 22 (Semiose), é um sistema funcional com **resultados medidos em hardware consumer**: roteamento 90.5%, subagentes 87.5%, injection 0/6 leaks, 182 testes determinísticos (Projeto AI-Orchestrator — `README.md`). Mas ele é um **ponto de partida**, não um destino final.

Este capítulo documenta as evoluções planejadas — todas ancoradas no que já foi construído, com arquivos mapeados, gates de validação e justificativa de ROI. Nada aqui é especulativo: cada sugestão decorre de uma limitação medida ou de uma oportunidade identificada nos evals.

O roteiro está organizado em seis frentes, mais uma seção dedicada ao **retreinamento contínuo dos pesos do modelo**.

---

## 1. Ontologias Formais (OWL/RDF) sobre o Knowledge Graph

### Diagnóstico

A Camada B (Knowledge Graph em Neo4j) implementa **travessia dirigida**: o agente chama `expand_context` e recebe vizinhos 1-hop de uma entidade. Isso resolve "quem fornece o SKU ABC-123?", mas não resolve "este pedido impacta quais fornecedores indiretamente?". A resposta exige percorrer uma cadeia de relações que não está armazenada como aresta explícita.

### Proposta

Adicionar uma camada de **inferência lógica** sobre o grafo existente usando o padrão OWL 2 (Web Ontology Language) com um reasoner Python (Owlready2).

A diferença prática:

| Situação | Grafo atual (Neo4j) | Com ontologia (OWL + Reasoner) |
|----------|---------------------|-------------------------------|
| "Quem fornece o SKU X?" | Cypher direto | Igual (relação explícita) |
| "Quais fornecedores são impactados pelo pedido #42?" | Só se houver aresta explícita `FORNECEDOR → PEDIDO` | Inferido via transitividade: `Fornecedor → abastece → Produto → compoe → Pedido` |
| "Este pedido viola alguma regra de negócio?" | A regra vive na API (Python), o grafo não sabe | SWRL rules no reasoner detectam violações automaticamente |

A integração segue o mesmo padrão da Camada B atual: uma **tool virtual** `infer_context` registrada no `ToolRegistry`. O agente decide se chama `expand_context` (vizinhança 1-hop, rápido) ou `infer_context` (cadeias inferidas, mais pesado). O reasoner roda em memória (Owlready2 não precisa de servidor externo) e consome o OWL exportado do Neo4j via `scripts/seed_ontology.py`.

### Gate de validação

| Métrica | Gate | O que mede |
|---------|------|-----------|
| Inference Precision@5 | ≥ 0.90 | Das 5 relações inferidas, quantas são semanticamente corretas |
| Inference Recall | ≥ 0.70 | Das relações que deveriam ser inferidas, quantas o reasoner capturou |
| Reasoner Latency | < 500ms | Tempo de inferência síncrona (Owlready2) |

### Arquivos e flags

- **Novo:** `gateway/ontology_reasoner.py` — classe `OntologyReasoner` com Owlready2
- **Novo:** `scripts/seed_ontology.py` — exporta Neo4j → OWL (Turtle/RDF-XML)
- **Modificar:** `gateway/tools/registry.py` — registrar `infer_context` como VirtualTool
- **Config:** `ONTOLOGY_ENABLED=1`, `ONTOLOGY_PATH=ontologies/wabfy.owl`

---

## 2. AI Gateway como Control Plane

### Diagnóstico

O gateway atual (`FastAPI` na porta 8100) centraliza roteamento, segurança (token + rate limit) e observabilidade (Langfuse). Mas ele opera como **orquestrador de um fluxo específico** (chat multi-agente), não como **control plane de toda a IA da organização**.

Em um cenário corporativo real, múltiplos consumidores (chat, API interna, dashboard, automação) precisam acessar diferentes modelos com políticas diferentes. Um atendente do suporte pode usar o modelo pequeno (1.5B) para queries simples; um analista financeiro precisa do modelo grande (30B) com acesso a dados sensíveis. Hoje, o gateway trata todos igualmente.

### Proposta

Evoluir o gateway para um **AI Control Plane** com seis capacidades novas:

| Capacidade | O que faz | Por que importa |
|------------|-----------|-----------------|
| **Rate limiting por modelo** | Cotas diferentes para LLM grande vs. pequeno vs. embeddings | Evita que queries triviais consumam cota do modelo caro |
| **Cost tracking por tenant** | Atribuir custo de tokens a projetos/departamentos | Responde "quanto o time de marketing gastou em IA este mês?" |
| **Fallback chain** | Se modelo A falhar → tenta B → tenta C | Resiliência sem intervenção humana |
| **Shadow deployment** | Espelhar tráfego para modelo novo sem afetar produção | Testar novo fine-tune com tráfego real, risco zero |
| **PII detection no gateway** | Bloquear/enviesar CPF, CNPJ antes de chegar ao LLM | Compliance LGPD na borda, não no modelo |
| **Audit log imutável** | Toda requisição → storage append-only | Rastreabilidade para auditoria e debugging |

O modelo mental muda de "um pipeline" para "uma plataforma". O gateway passa a expor um endpoint `/v1/models` que lista modelos disponíveis com suas capacidades e custos, e um header `X-Tenant-ID` que particiona cotas e logs.

### Gate de validação

| Métrica | Gate |
|---------|------|
| PII Leak Rate | 0% (nenhum dado sensível ao LLM externo) |
| Shadow Accuracy Gap | < 5% (modelo shadow não diverge do produção) |
| Audit Completeness | 100% (toda requisição gera entrada de audit) |

### Arquivos e flags

- **Novo:** `gateway/cost_tracker.py`, `gateway/shadow_router.py`, `gateway/data_guard.py`, `gateway/audit_log.py`
- **Modificar:** `gateway/security.py` (rate limiting por modelo), `gateway/main.py` (tenant routing)
- **Config:** `GATEWAY_MODE=control_plane`, `COST_TRACKING_ENABLED=1`, `SHADOW_MODEL=qwen3.5-9b-orch-v2`

---

## 3. Prompt Caching

### Diagnóstico

O Langfuse mostra que 100 traces têm latência P50 = 4.2s e P95 = 8.8s. Uma parcela significativa dessas queries são **factuais e repetitivas**: "qual o saldo do SKU TR-001?", "quantos dias de férias o Carlos tem?", "liste os pedidos do mês". O mesmo SKU é consultado várias vezes ao dia por usuários diferentes — e cada consulta dispara o pipeline completo (embed → classify → dispatch → LLM → synthesize).

### Proposta

**Cache semântico**: antes de executar o pipeline, calcular o embedding da query e buscar no Qdrant por respostas pré-computadas com similaridade de cosseno ≥ 0.97. Se encontrar, retorna a resposta cacheada instantaneamente (0 tokens, 0 ms de LLM).

Diferente de cache HTTP tradicional (match exato de string), o cache semântico reconhece paráfrases: "qual o estoque do TR-001?" e "TR-001 tem quantas unidades?" batem no mesmo cache.

**Estratégia de invalidação** — o que **nunca** vai para cache:

- Respostas que envolveram tool calls (dados mutáveis de API)
- TTL de 1 hora para respostas factuais (envelhecem)
- Invalidação seletiva: se o seed de dados do domínio "estoque" for atualizado, todas as entradas com `domains: ["estoque"]` são invalidadas
- Header `X-No-Cache: 1` permite forçar cache miss

**Economia projetada** (com base nos 100 traces/dia atuais):

| Cenário | Hit rate | Latência P50 | Tokens/dia economizados |
|---------|----------|-------------|------------------------|
| Sem cache | 0% | 4.2s | 0 |
| Conservador (cosine ≥ 0.97) | ~15% | 2.1s (↓50% em hits) | ~67K |
| Agressivo (cosine ≥ 0.92) | ~35% | 1.8s | ~157K |

### Gate de validação

| Métrica | Gate |
|---------|------|
| Cache Hit Rate | ≥ 10% (mínimo para justificar a infra) |
| Cache Staleness Rate | 0% (resposta cacheada nunca pode divergir da resposta fresca em queries factuais) |
| Cache Latency | < 10ms (Qdrant kNN) |

### Arquivos e flags

- **Novo:** `gateway/prompt_cache.py` — classe `PromptCache` com backend Qdrant
- **Modificar:** `gateway/graph.py` — nós `_classify` e `_synthesize` verificam cache
- **Config:** `PROMPT_CACHE_ENABLED=1`, `PROMPT_CACHE_THRESHOLD=0.97`, `PROMPT_CACHE_TTL_MINUTES=60`

---

## 4. Model Routing

### Diagnóstico

O orquestrador atual usa um **modelo único** para todas as queries: Qwen3.5-9B LoRA (5.4 GB Q4_K_M, 100% GPU, ~3s/task). Isso é correto para o PoC, mas ineficiente em escala: queries triviais como "liste os produtos" consomem o mesmo modelo (e a mesma latência) que queries complexas como "qual o impacto financeiro da rotatividade do RH no orçamento de Q3?".

A literatura de **model cascading** (Chen et al., 2023) mostra que rotear queries por complexidade pode reduzir custo em até 70% sem perda de qualidade, usando modelos pequenos para a maioria das queries e reservando modelos grandes para as poucas que realmente precisam.

### Proposta

Um **classificador de complexidade determinístico** que avalia cada query em 5 sinais (zero chamada de LLM) e seleciona o tier:

| Tier | Modelo | Complexidade | Latência típica | Custo/query |
|------|--------|-------------|-----------------|-------------|
| Simple | Qwen 1.5B | score ≤ 2 | ~0.5s | ~R$0.0001 |
| Moderate | Qwen 9B LoRA | score 3–5 | ~3s | ~R$0.003 |
| Complex | Qwen 30B MoE | score ≥ 6 | ~15s | ~R$0.05 |

**Sinais do classificador:**

| Sinal | Peso | Exemplo que pontua |
|-------|------|--------------------|
| Múltiplos domínios (lexical) | +2 por domínio extra | "RH e finanças" → +2 |
| Entidades cross-domain | +1 por entidade | SKU + fornecedor + funcionário → +3 |
| Palavras de reasoning | +1 cada | "por que", "explique", "compare", "impacto" |
| Comprimento da query | +1 se > 100 chars | Query investigativa |
| Histórico multi-turn | +2 | Turn 3+ no mesmo tópico |

**Fallback em cascata:** se o modelo small falhar ou exceder timeout → sobe para moderate → se falhar → sobe para complex. Isso garante que nenhuma query fique sem resposta, mesmo que o classificador erre.

**Economia projetada** (100 queries/dia, distribuição 60/30/10):

| Estratégia | Custo diário | Latência média |
|------------|-------------|---------------|
| Modelo único (9B) | R$0.30 | 4.2s |
| Model routing | R$0.08 (↓73%) | 2.1s (↓50%) |

### Gate de validação

| Métrica | Gate |
|---------|------|
| Model Route Accuracy | ≥ 85% (classificador acerta a complexidade) |
| Cascade Rate | < 5% (quantas queries sobem de tier por falha) |
| Complex Tier Latency | Sem teto (o modelo grande é lento por definição) |

### Arquivos e flags

- **Novo:** `gateway/model_router.py` — `ModelRouter` com `ComplexityClassifier` + cascade
- **Modificar:** `gateway/config.py` — tiers e thresholds
- **Config:** `MODEL_ROUTING_ENABLED=1`, `MODEL_SIMPLE=qwen2.5:1.5b`, `MODEL_MODERATE=qwen3.5-9b-orch`, `MODEL_COMPLEX=qwen3:30b`

---

## 5. ROI Mensurável de AI

### Diagnóstico

O dashboard de observabilidade (cap. 19) mostra latência, tokens e accuracy. Mas a pergunta que stakeholders fazem não é técnica — é financeira: **"quanto estamos economizando com IA vs. processo manual?"**. Sem essa resposta, o projeto é percebido como custo, não como investimento.

### Proposta

Um **framework de ROI** com três camadas:

**Camada 1 — Coleta automática** (já parcialmente implementada):
- Cost-per-Task: GPU watt-horas × R$/kWh + tokens × preço por 1K (via Langfuse tracing)
- Task Success Rate: % de agent runs que terminam com resposta definitiva
- Latência real: P50/P95 do Langfuse (já implementado em `gateway/metrics.py`)

**Camada 2 — Premissas configuráveis:**
- `ROI_COST_HOUR_HUMAN=50.0` (R$/hora do profissional equivalente)
- `ROI_MANUAL_MINUTES=5.0` (minutos que um humano levaria na mesma tarefa)
- `ROI_GPU_WATT=200.0` (TDP da GPU)
- `ROI_COST_KWH=0.70` (R$/kWh)

**Camada 3 — Dashboard de ROI** (nova seção no frontend):

| Indicador | Fórmula | Exemplo com dados atuais |
|-----------|---------|--------------------------|
| Economia mensal | (queries/mês × minutos_manuais/60 × R$/h) − (custo_GPU + manutenção) | 300 × 5/60 × R$50 − R$150 = **R$1.100/mês** |
| Payback | Custo total do setup / economia mensal | ~R$1.000 / R$1.100 = **~1 mês** |
| Custo por query | (GPU + tokens) / queries | R$0.001/query |
| Projeção 2× volume | Economia dobra, custo marginal sobe só a energia | R$2.200/mês |

**Fórmula documentada:**
```
ROI_mensal = (Q × T_manual × C_hora) − (E_gpu × C_kwh + C_manutencao)

Onde:
  Q = queries processadas no mês
  T_manual = horas humanas por query (minutos_manuais / 60)
  C_hora = custo hora do profissional equivalente
  E_gpu = energia consumida (GPU_watt × horas_ativas / 1000 kWh)
  C_kwh = tarifa de energia
  C_manutencao = custo fixo mensal (serviços cloud, domínio, manutenção)
```

> **Nota de honestidade:** O RAOI (Return on AI Investment) não é automatizado — depende de dados reais de operação que ainda não existem. O que o dashboard mostra é uma **projeção documentada**, com todas as premissas explícitas e ajustáveis. Não é um número mágico; é uma fórmula auditável.

### Arquivos e flags

- **Novo:** `gateway/roi_tracker.py`, `frontend/src/components/RoiDashboard.tsx`
- **Modificar:** `gateway/observability.py` — novo pilar "ROI"
- **Config:** `ROI_COST_HOUR_HUMAN=50.0`, `ROI_GPU_WATT=200.0`, `ROI_COST_KWH=0.70`

---

## 6. Retreinamento Contínuo dos Pesos do Modelo

### Diagnóstico

O modelo de produção atual é um **Qwen3.5-9B com LoRA fine-tuned** via Unsloth no Google Colab A100, usando 3.050 exemplos (1.325 trajetórias + 1.569 routing + 156 injection). Ele roda 100% GPU em RTX 3060 a 2-4 s/task e entrega +5 pp nos domínios vs. o baseline 7b (Projeto AI-Orchestrator — `README.md`).

Mas o fine-tune é **estático**. O modelo foi treinado uma vez com dados de junho/2026. Se as regras de negócio mudarem (novos SKUs, novos departamentos, novos padrões de query), o modelo **não aprende** — continua respondendo com o conhecimento congelado no checkpoint LoRA.

Além disso, o dataset atual (3.050 exemplos) não captura a escala real dos domínios: o Knowledge Graph tem 51 nós e 52 relações, mas apenas uma fração disso gerou exemplos de treino. O script `train/build_kg_dataset.py` já está preparado para gerar **4.000+ exemplos** a partir do KG (2-domínios, 3-domínios, 4-domínios, single-domain, injection e trajetórias), mas esse dataset expandido nunca foi usado para retreinar.

### Proposta: Pipeline de Retreinamento Contínuo

O retreinamento não é um evento único — é um **ciclo** com 5 etapas, disparado por gatilhos mensuráveis:

```
[Gatilho] → 1. Expandir dataset → 2. Retreinar LoRA → 3. Avaliar → 4. Shadow deploy → 5. Promover
```

#### 5.1 Gatilhos de retreinamento

O modelo **não** deve ser retreinado em intervalo fixo (ex: "todo mês"). O retreinamento deve ser disparado por **sinais de degradação**:

| Gatilho | Métrica | Threshold | Fonte |
|---------|---------|-----------|-------|
| **Routing accuracy caiu** | Exact-match routing < 85% | Abaixo do gate de 90% por 3 evals consecutivos | `eval_routing.py` |
| **Domain shift detectado** | Distribuição de domínios nas queries mudou > 20% | Novos padrões de pergunta não cobertos pelo dataset atual | `eval_results.py` (domain breakdown) |
| **Tool call failure rate subiu** | Erros 422/404 > 5% | APIs mudaram (novos campos, schemas) e o modelo não sabe | Langfuse traces |
| **Dataset expandido disponível** | `build_kg_dataset.py` gerou ≥ 4.000 exemplos | O KG cresceu e o dataset atual cobre < 50% das entidades | Script de build |
| **Novo modelo base disponível** | Nova versão do Qwen 3/4 lançada | O modelo base melhorou e vale a pena refazer o fine-tune | Monitoring manual |

#### 5.2 Expandir dataset

O script `train/build_kg_dataset.py` gera exemplos de treino a partir do Knowledge Graph vivo (Neo4j). O processo:

1. **Ler o KG atual** (Neo4j) — todas as entidades, relações e domínios
2. **Gerar perguntas sintéticas** usando templates parametrizados com entidades reais:
   - Single-domain: "Qual o saldo do SKU {sku}?" (1.200+ perguntas)
   - Cross-domain 2: "Quanto custa manter o estoque do {produto} que a {cliente} comprou?" (2.500+)
   - Cross-domain 3-4: perguntas complexas multi-hop (500+)
   - Injection: 8% do total (~320 exemplos)
   - Trajetórias: tool-calling completo com agent loop (~200 exemplos)
3. **Balancear** por domínio e complexidade
4. **Dedupicar** contra o dataset atual (evitar overfitting nos mesmos exemplos)
5. **Exportar** no formato Unsloth (chat template com roles)

**Meta de escala:** 4.000+ exemplos (vs. 3.050 atuais), com cobertura > 80% das entidades do KG.

#### 5.3 Retreinar LoRA

Usando o mesmo pipeline do capítulo 11 (Unsloth + Google Colab A100), mas com o dataset expandido:

| Hiperparâmetro | Valor | Justificativa |
|----------------|-------|---------------|
| Base model | `unsloth/Qwen3.5-9B` (ou versão mais recente) | Manter mesma arquitetura |
| LoRA rank | 32 (manter) | Rank atual entrega resultados; aumentar para 64 se dataset crescer > 10K |
| LoRA alpha | 64 (manter) | Alpha = 2× rank é prática padrão |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Cobertura completa dos attention + MLP |
| Epochs | 3 (mesmo) | 3 épocas com 4K exemplos é suficiente; monitorar overfitting |
| Learning rate | 2e-4 com cosine schedule | LR padrão LoRA; reduzir para 1e-4 se loss oscilar |
| Batch size | 8 (manter) | A100 40GB comporta; reduzir para 4 se OOM |
| Max seq length | 2048 (aumentar de 1024) | Dataset expandido tem queries cross-domain mais longas |

**Checkpoint strategy:** salvar checkpoint a cada 0.5 época. Se a loss de validação subir (overfitting), restaurar o melhor checkpoint — não o último.

#### 5.4 Avaliar (antes de deploy)

O modelo retreinado precisa passar pelos **mesmos gates** do modelo atual — e idealmente superá-los:

| Gate | Modelo atual | Meta retreinado | Eval script |
|------|-------------|-----------------|-------------|
| Domínios por subagente | 87.5% (35/40) | ≥ 90% | `eval_domains.py` |
| Roteamento multi-domínio | 90.5% (58/64) | ≥ 93% | `eval_routing.py` |
| Injection | 0/6 leaks | 0 leaks | `eval_injection.py` |
| Semiose — Camada A (4 gates) | 4/4 PASS | 4/4 PASS | `eval_semiose.py` |
| Routing accuracy (153 casos) | 73.9% | ≥ 80% | `eval_results.py` |

**Nova métrica de comparação:** `delta_accuracy` = accuracy_novo_modelo − accuracy_modelo_atual. Se `delta_accuracy < 0` em qualquer gate, o modelo novo **não sobe**.

#### 5.5 Shadow deploy

Antes de substituir o modelo em produção, o novo LoRA roda em **shadow mode**: 10% do tráfego é espelhado para ele, as respostas são logadas mas não enviadas ao usuário. Comparação lado a lado por 48h:

| O que comparar | Como |
|----------------|------|
| Domínios roteados | Concordância entre modelo atual e shadow |
| Latência | P50/P95 de cada modelo |
| Tool calls | Mesmas tools chamadas? Mesmos argumentos? |
| Resposta final | Similaridade de cosseno entre as respostas |

Se o shadow divergir do produção em > 5% dos casos, investigar antes de promover.

#### 5.6 Promover

Se todos os gates passarem e o shadow deploy não mostrar regressão, promover o novo LoRA:

1. **Exportar GGUF** (Q4_K_M, mesmo formato do atual)
2. **Substituir o arquivo** no volume do Ollama (`./models/qwen3.5-9b-orch.Q4_K_M.gguf`)
3. **Hot-reload** no Ollama (`ollama cp qwen3.5-9b-orch qwen3.5-9b-orch-v1; ollama rm qwen3.5-9b-orch-v1`)
4. **Log da transição** no audit log com hash do modelo, métricas comparativas e timestamp

### Ciclo completo — visão anual

Com 4 retreinamentos/ano (gatilhos reais, não calendário):

| Trimestre | Dataset | Ação |
|-----------|---------|------|
| Q3 2026 | 3.050 → 4.000+ | Primeiro retreinamento com dados do KG expandido |
| Q4 2026 | 4.000 → 5.000+ | Incorporar queries reais dos logs (anonymizadas) |
| Q1 2027 | 5.000 → 6.000+ | Adicionar exemplos dos novos domínios (se houver) |
| Q2 2027 | Reavaliar necessidade | Se accuracy > 95% estável, espaçar retreinamentos |

### Gate de validação do pipeline de retreinamento

| Métrica | Gate |
|---------|------|
| Delta accuracy ≥ 0 | Nenhum gate pode regredir |
| Shadow agreement > 95% | Concordância com produção por 48h |
| Retrain latency < 4h | Do build do dataset ao GGUF exportado (Colab A100) |
| Dataset coverage > 80% | % de entidades do KG representadas nos exemplos de treino |

---

## Priorização e ROI

Das seis frentes, três entregam economia mensurável de curto prazo:

| # | Frente | Ganho principal | Esforço |
|---|--------|----------------|---------|
| 7 | Prompt Caching | ↓15% latência, ~67K tokens/dia economizados | Baixo (1 arquivo novo, Qdrant já existe) |
| 8 | Model Routing | ↓73% custo, ↓50% latência | Médio (classificador + cascade) |
| 9 | ROI Mensurável | Transforma projeto técnico em caso de negócio | Baixo (fórmula + dashboard) |
| 5 | Ontologias | Inferência cross-domain sem arestas explícitas | Médio (Owlready2 + seed script) |
| 6 | AI Gateway | Governança corporativa multi-tenant | Alto (6 novos módulos) |
| Retreinamento | Ciclo completo | Previne degradação, escala com os dados | Alto (pipeline end-to-end) |

A ordem recomendada: **ROI (9) → Cache (7) → Model Routing (8) → Ontologias (5) → Retreinamento → Gateway (6)**. O ROI vem primeiro porque justifica o investimento nas demais frentes. O Gateway vem por último porque seu valor é proporcional ao número de consumidores — faz sentido quando houver múltiplos tenants, não antes.

---

## Referências

- Chen, L., Zaharia, M., & Zou, J. (2023). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*. arXiv:2305.05176.
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130.
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
- Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, seção *Trabalho Futuro* (itens 5–9).
- Projeto AI-Orchestrator — `train/build_kg_dataset.py` (gerador de dataset SFT v2, 4.000+ exemplos).
- Projeto AI-Orchestrator — `README.md` (resultados medidos do modelo atual).
- Projeto AI-Orchestrator — `gateway/metrics.py` (Langfuse live metrics, P50/P95/tokens).
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
- Unsloth Documentation (2026). LoRA fine-tuning with Qwen 3.5. https://docs.unsloth.ai.
- W3C (2012). *OWL 2 Web Ontology Language Document Overview (Second Edition)*. https://www.w3.org/TR/owl2-overview/.
