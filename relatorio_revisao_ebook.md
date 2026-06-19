# Relatório de Revisão do Ebook — "Mão na Massa: Treinando LLM On-Premise"

**Data:** 2026-06-19
**Autor da revisão:** Análise em 3 camadas: ebook × AI-Orchestrator × git-repo (4 repositórios-fonte)
**Escopo:** 21 capítulos do ebook (11.489 linhas) + código-fonte AI-Orchestrator (~200 arquivos) + git-repo (6.055 arquivos em 4 repositórios: llm-model-inference, RAG-with-Python-Cookbook, agent-driven-design, agents-integration-patterns)

---

## SUMÁRIO EXECUTIVO

O livro se propõe a ser a **documentação técnica do projeto AI-Orchestrator** — um gateway multi-agente on-premise em produção com LangGraph + Ollama + Qdrant + 4 microsserviços FastAPI, 182 testes, 5 gates de avaliação e fine-tuning LoRA do Qwen3.5-9B. A estrutura de 6 partes e 21 capítulos acompanha a arquitetura real do projeto. A pasta `git-repo/` contém as **fontes originais** dos notebooks adaptados: 33 notebooks, ~136 samples de código, 26 padrões de integração catalogados e 1 framework conceitual de design de agentes (ADD).

**Pontos fortes:** Os capítulos 11 (LoRA), 19 (multi-agente), 20 (segurança) e 21 (MLOps) são **excelentes** — reproduzem com fidelidade a implementação real.

**Problema principal:** Há um **desalinhamento entre exemplos didáticos dos capítulos iniciais e a implementação real** — o leitor começa com `qwen2.5:7b` + `nomic-embed-text`, mas a produção usa `qwen3.5-9b-orch` (LoRA) + SBERT MiniLM. Esta evolução é o coração do livro e precisa ser contada.

**Achado do git-repo:** Os 4 repositórios-fonte contêm **14 oportunidades de enriquecimento técnico** ainda não aproveitadas, incluindo: (1) framework ADD (Model vs Harness) para estruturar cap19-21, (2) catálogo de 26 padrões de integração mapeáveis ao AI-Orchestrator, (3) MAST failure taxonomy com 14 modos de falha, (4) single-model serving do zero para cap07, (5) 5 estratégias de chunking + reranking para cap15, (6) Graph RAG progressivo com 5 notebooks Neo4j para cap17.

Foram encontrados **46 problemas/oportunidades** em 8 categorias.

---

## CATEGORIA 1: ERROS FACTUAIS E TÉCNICOS

### 1.1 URL incorreta do Ollama [ALTA]

**Arquivos:** `referencias/bibliografia.md` linha 36; `cap04_primeiros_passos.md` linha 20
**Erro:** `https://ollama.ai` e `registry.ollama.ai`
**Correção:** `https://ollama.com` e `registry.ollama.com`
**Impacto:** Domínio errado — links quebrados em duas localizações.

### 1.2 Fórmula do KV Cache — num_heads deve ser num_kv_heads [ALTA]

**Arquivo:** `cap06_otimizacao_inferencia.md`, linha 40
**Erro:** A fórmula `KV Cache = 2 × num_camadas × num_heads × dim_head × seq_len × bytes_por_elem` usa `num_heads=32` (attention heads), resultando em ~2 GB para Llama 3.1 8B. Modelos modernos usam GQA (Grouped Query Attention) com `num_kv_heads=8`:
- Com `num_heads=32`: 2 × 32 × 32 × 128 × 4096 × 2 = **2 GB** (errado — 4× superestimado)
- Com `num_kv_heads=8`: 2 × 32 × 8 × 128 × 4096 × 2 = **512 MB** (correto)
**Correção:** Substituir `num_heads` por `num_kv_heads` na fórmula e adicionar explicação sobre GQA.

### 1.3 Docker healthcheck do capítulo 7 usa `curl` [MÉDIA]

**Arquivo:** `cap07_frameworks_serving.md` (docker-compose vLLM)
**Erro:** `test: ["CMD", "curl", "-f", "http://localhost:8000/health"]` — a imagem `vllm/vllm-openai:latest` pode não incluir `curl`.
**Correção:** Alinhar com o padrão do AI-Orchestrator: usar `wget --spider` (já usado no cap21) ou Python inline (`python -c "import urllib.request..."`).

### 1.4 ORCA paper — ano incorreto [BAIXA]

**Arquivo:** `cap06_otimizacao_inferencia.md` (Fontes, item 7)
**Erro:** "Yu, G. et al. (2024). ORCA... OSDI 2022." — ano da publicação contradiz conferência.
**Correção:** "Yu, G. et al. (2022). ORCA... OSDI 2022."

---

## CATEGORIA 2: DESALINHAMENTOS COM A IMPLEMENTAÇÃO REAL

### 2.1 Modelo de embeddings: Ollama nomic-embed-text vs SBERT [ALTA]

**Contexto AI-Orchestrator:** A implementação real usa **SBERT** (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim, CPU) como embedder primário, com `OllamaEmbedder` como fallback. A decisão foi documentada em 2026-06-14 (eliminar dependência de `ollama pull` para embeddings, reduzir latência).

**Ebook:** Capítulos 14 e 15 usam `nomic-embed-text` via Ollama como exemplo principal. A dimensão 768 (nomic) vs 384 (SBERT) afeta Qdrant, chunking e performance.

**Correção:** O capítulo 14 ou 15 deve documentar a **evolução**: começar com nomic-embed-text (didático, sem dependência extra) → migrar para SBERT (produção, CPU, sem Ollama). Mostrar o `Embedder Protocol` do `gateway/embedder.py` que abstrai ambos.

### 2.2 Modelo de produção: qwen2.5:7b vs qwen3.5-9b-orch [ALTA]

**Contexto AI-Orchestrator:** O projeto passou por uma evolução documentada:
- **PoC:** `qwen3:30b-a3b` (MoE, 18 tok/s, 44% transborda GPU→CPU, ~55s por task)
- **Baseline:** `qwen2.5:7b-instruct-q4_K_M` (100% GPU, ~7s por task)
- **Produção:** `qwen3.5-9b-orch` (LoRA fine-tuned, 100% GPU, ~2-4s por task, routing 90.5%, domains 87.5%)

**Ebook:** Cap16 usa `qwen2.5:7b`, cap21 menciona `qwen2.5:7b-instruct-q4_K_M` como padrão, cap11 treina `Qwen3.5-9B`. O leitor não entende a jornada.

**Correção:** Adicionar ao capítulo 4 (ou criar seção de transição no capítulo 13) uma tabela documentando a evolução dos modelos com métricas reais (latência, acurácia, VRAM). Isto é o **coração narrativo** do livro — mostrar que o fine-tuning LoRA permite substituir um MoE 30B por um 9B 5-7× mais rápido.

### 2.3 Semantic Router: mecanismo de consenso não documentado [MÉDIA]

**Contexto AI-Orchestrator:** `gateway/semantic_router.py` implementa um mecanismo sofisticado: kNN no Qdrant com golden set → verificação de consenso (top-K precisam concordar) → `score_gap` filter (rejeita se top-1 e top-2 estão muito próximos, `min_score_gap=0.05`) → fallback para LLM classifier.

**Ebook:** Capítulos 14-15 descrevem busca vetorial básica, sem mencionar consenso, score_gap ou fallback.

**Correção:** Expandir capítulo 15 com seção "Semantic Router em produção: além da similaridade" documentando o pipeline completo do `gateway/semantic_router.py`.

### 2.4 Circuit Breaker por domínio ausente [MÉDIA]

**Contexto AI-Orchestrator:** `gateway/tools/circuit.py` — 3 falhas de transporte → OPEN 30s → HALF_OPEN. Erros 4xx NÃO acionam o breaker (são erros de negócio, não de infraestrutura).

**Ebook:** Cap19 menciona Tool Registry mas não cobre resiliência. Cap21 cobre graceful degradation do Langfuse mas não circuit breaker.

**Correção:** Adicionar seção no capítulo 19 ou 21 sobre circuit breaker — é um padrão crítico de produção.

### 2.5 Frontend React não coberto [BAIXA]

**Contexto AI-Orchestrator:** O projeto inclui frontend React (Vite + React 19 + Tailwind 4) com 3 páginas: Chat (SSE live trace), Dashboard (métricas Langfuse), Evals (resultados de avaliação). Servido estaticamente pelo gateway.

**Ebook:** Nenhuma menção ao frontend.

**Correção:** Opção A — adicionar apêndice "Frontend de observabilidade". Opção B — nota no cap21 mencionando o dashboard como interface de MLOps.

---

## CATEGORIA 3: ORTOGRAFIA E ACENTUAÇÃO

### 3.1 Uso inconsistente de acentuação entre capítulos [ALTA]

**Capítulos com acentuação correta:** cap01, cap02, cap03, cap05, cap08, cap09, cap10, cap11, cap12, cap13, cap14

**Capítulos com acentuação ausente (confirmado):**
- `cap04_primeiros_passos.md`: "Configuracao", "geracao" (linhas 282, 316, 400)
- `cap07_frameworks_serving.md`: "Instalacao", "Geracao" (linhas 64, 77, 237, 244, 301)
- `cap06_otimizacao_inferencia.md`: "metricas" (linha 333)
- `cap18_benchmarking.md`: Extensivo — "classificacao", "execucao", "Implementacao", "Metricas", "agregadas" (~15 ocorrências)

**Correção:** Revisão sistemática com dicionário pt-BR.

### 3.2 "Mascara" vs "Máscara" [BAIXA]

**Arquivo:** `cap11_treinando_lora.md`
**Correção:** "Máscara de loss"

---

## CATEGORIA 4: ESTRUTURA E NUMERAÇÃO

### 4.1 Relatório de similaridade desatualizado [ALTA]

**Arquivo:** `relatorio_similaridade.md`
**Erro:** Analisa 19 capítulos com numeração antiga. O cap08 atual (BERT encoders) não foi analisado.
**Correção:** Regenerar com 21 capítulos.

### 4.2 README omite capítulo 21 e notebooks 19-21 [MÉDIA]

**Arquivo:** `README.md`
**Erro:** Sumário lista 20 capítulos; tabela de notebooks vai só até cap18.
**Correção:** Adicionar cap21 ao sumário. Nota: capítulos 19, 20 e 21 são primordialmente arquiteturais (cobrem código do gateway) — justifica não terem notebooks dedicados.

### 4.3 Seção de benchmarking duplicada no capítulo 1 [BAIXA]

**Correção:** Remover do capítulo 1 — não pertence à Parte I (Fundamentos).

---

## CATEGORIA 5: CÓDIGO E EXEMPLOS

### 5.1 Function calling com qwen2.5:7b — validação pendente [MÉDIA]

**Arquivo:** `cap16_rag_agentes.md`
**Risco:** Modelos 7B podem não suportar function calling confiável.
**Correção:** Adicionar nota: "Para produção, o AI-Orchestrator usa qwen3.5-9b-orch com tool-calling fine-tunado. Modelos <7B podem alucinar tool calls."

### 5.2 Caminhos hardcoded do AI-Orchestrator — intencional ou não? [BAIXA]

**Arquivo:** `cap11_treinando_lora.md`
**Situação:** Caminhos como `/content/drive/MyDrive/ai-orchestrator-lora/training` são específicos do projeto. Como o livro documenta o AI-Orchestrator, isso pode ser intencional.
**Decisão do autor:** Se o livro é um "build-along" do AI-Orchestrator, manter os caminhos reais. Se é um guia genérico, usar placeholders.

### 5.3 Timeout de embeddings no Qdrant [MÉDIA]

**Arquivo:** `cap15_chunking_indexacao.md`
**Correção:** Aumentar timeout de 120s para 300s para hardware limitado. Adicionar retry com backoff.

---

## CATEGORIA 6: OMISSÕES DE COMPONENTES DO AI-ORCHESTRATOR

### 6.1 Sistema de avaliação com 5 gates [ALTA]

**Contexto real:** O AI-Orchestrator tem um sistema de avaliação completo em `evals/`:
- `eval_routing.py`: 64 perguntas golden, gate ≥90% (resultado: 90.5% PASS)
- `eval_domains.py`: 40 tasks em 4 domínios, gate ≥80% cada (resultado: 87.5% PASS)
- `eval_injection.py`: 6 casos adversários, gate 0 leaks (resultado: 0/6 PASS)
- `fase0_bench.py`: GPU benchmark (cold load, tok/s, model swap)
- `demo.py`: 5 transcripts de conversa

**Ebook:** Cap12 cobre métricas de avaliação genericamente. Cap18 cobre benchmarking. Mas a **metodologia dos gates** — que é o coração da garantia de qualidade do projeto — está fragmentada.

**Correção:** Dedicar uma seção do capítulo 12 ou 18 à metodologia de gates: "Como o AI-Orchestrator decide se um modelo vai para produção". Mostrar os 5 gates, thresholds e resultados reais.

### 6.2 BERTimbau Injection Classifier — pipeline de treino [MÉDIA]

**Contexto real:** `gateway/injection_detector.py` + `train/colab_train_injection.ipynb` — BERTimbau fine-tunado com 400 exemplos sintéticos (200 injection + 200 legítimos), 100% val accuracy, 417 MB.

**Ebook:** Cap08 cobre BERT teoricamente. Cap20 lista os 14 patterns de regex. Mas o **pipeline de treino do BERTimbau** não aparece.

**Correção:** Adicionar ao cap08 ou cap20: como gerar dataset sintético de injection, como fine-tunar BERTimbau, e como integrar no pipeline de sanitização (boundary tripla: strip ChatML → 14 regex → BERTimbau classifier).

### 6.3 Tool Registry via OpenAPI auto-discovery [MÉDIA]

**Contexto real:** `gateway/tools/registry.py` — cada microsserviço expõe `/openapi.json`, o registry faz parse automático e gera tool definitions no formato Ollama. Isto é uma inovação arquitetural importante.

**Ebook:** Cap19 menciona "descoberta automática de ferramentas via OpenAPI" no título de seção, mas mostra código fragmentado.

**Correção:** Expandir cap19 com o fluxo completo: OpenAPI spec → `_extrair_parametros()` → formato Ollama tools. Mostrar como adicionar um novo microsserviço (ex: "suporte") sem alterar o gateway.

### 6.4 SSE streaming com heartbeat [BAIXA]

**Contexto real:** `gateway/main.py` — POST /chat retorna SSE com heartbeat a cada 15s para manter conexão viva durante LLM calls longas.

**Ebook:** Cap04 cobre streaming básico. Mas o padrão SSE com heartbeat e trace por evento não está documentado.

**Correção:** Nota no cap04 ou cap21 sobre o padrão SSE usado em produção.

### 6.5 Langfuse Cloud vs self-hosted [BAIXA]

**Contexto real:** O AI-Orchestrator suporta ambos: Langfuse Cloud (padrão, `LANGFUSE_PUBLIC_KEY`) e self-hosted via Docker Compose (`langfuse/langfuse:2` + PostgreSQL).

**Ebook:** Cap18 mostra deploy self-hosted. Cap21 menciona Cloud. A dualidade não fica clara.

**Correção:** Explicitar que o livro cobre ambos os modos e quando usar cada um.

### 6.6 Cloudflare Tunnel para exposição pública [BAIXA]

**Contexto real:** `docker-compose.yml` inclui perfil `public` com Cloudflare Tunnel para `suasalada.com.br`.

**Ebook:** Não mencionado.

**Correção:** Nota no cap21: padrão para expor serviço on-premise sem abrir portas no firewall.

### 6.7 Rate Limiter e Auth (AccessTokenGuard) [BAIXA]

**Contexto real:** `gateway/security.py` — sliding window rate limiter (`max_entries=10000` + eviction), `AccessTokenGuard` fail-closed, `ALLOW_OPEN_ACCESS` para dev.

**Ebook:** Cap20 cobre injection mas não auth/rate-limit.

**Correção:** Expandir cap20 com padrões de auth e rate limiting do AI-Orchestrator.

### 6.8 Embedder Protocol (SBERT + Ollama fallback) [BAIXA]

**Contexto real:** `gateway/embedder.py` — Protocol com duas implementações swapáveis. O gateway Dockerfile faz pre-download do modelo SBERT na build.

**Ebook:** Não documentado como padrão de design.

**Correção:** Adicionar ao cap14 ou 15 como exemplo de padrão Strategy/Protocol em Python.

---

## CATEGORIA 7: MELHORIAS ESTRUTURAIS E DIDÁTICAS

### 7.1 Jornada do modelo: o arco narrativo central [ALTA]

**Problema:** O livro não conta explicitamente a história de evolução que é seu maior trunfo:

```
PoC: qwen3:30b-a3b (MoE) → 18 tok/s, 44% GPU, ~55s/task
  ↓ (cap03-07: inferência local)
Baseline: qwen2.5:7b → 100% GPU, ~7s/task, routing 90.5%
  ↓ (cap09-13: fine-tuning LoRA)
Produção: qwen3.5-9b-orch → 100% GPU, ~2-4s/task, routing 90.5%, domains +5pp
```

**Recomendação:** Adicionar ao capítulo 1 uma seção "A jornada deste livro" com esta evolução. Ao final de cada parte, uma seção "Onde estamos na jornada" recapitulando. Isto transforma 21 capítulos técnicos em uma narrativa coesa.

### 7.2 Diagrama de arquitetura completo [MÉDIA]

**Contexto real:** `docs/gen_diagrams.py` gera 7 diagramas PNG da arquitetura.

**Recomendação:** Incluir o diagrama principal no prefácio ou capítulo 1 como "mapa do tesouro" que o leitor vai construir.

### 7.3 Tabela de ambiente (.env.example) como referência [MÉDIA]

**Contexto real:** `.env` do AI-Orchestrator tem 30+ variáveis. O cap21 mostra `.env.example`.

**Recomendação:** Consolidar no apêndice como referência rápida.

### 7.4 Pipelines documentados no AI-Orchestrator mas não no ebook [MÉDIA]

**Contexto real:** `PLANO_EXECUCAO.md` documenta 7 fases de evolução do projeto. `docs/SKILL_MULTIAGENT.md` tem 279 linhas de decisões arquiteturais.

**Recomendação:** Garantir que todas as decisões documentadas nestes arquivos sejam cobertas no ebook.

### 7.5 Prefácio contextualizando o AI-Orchestrator [MÉDIA]

**Recomendação:** Adicionar prefácio explicando:
- O que é o AI-Orchestrator e por que ele existe
- Público-alvo: engenheiros que querem implantar LLMs on-premise
- Pré-requisitos: Python, Docker, Linux básico
- Como usar o livro: leitura sequencial (build-along) vs salteada (referência)
- Convenções: blocos de código são do repositório real, comandos testados em RTX 3060 12GB

### 7.6 Exercícios baseados no AI-Orchestrator [BAIXA]

**Recomendação:** Adicionar exercícios como "Adicione um quinto microsserviço (suporte) ao Orchestrator" ou "Modifique o router para detectar um novo domínio".

### 7.7 Glossário e índice remissivo [BAIXA]

**Recomendação:** Gerar a partir dos termos técnicos do `PLANO_EXECUCAO.md` e `SKILL_MULTIAGENT.md`.

---

## ALINHAMENTO CAPÍTULO-A-IMPLEMENTAÇÃO

| Capítulo | Tema | Arquivo(s) AI-Orchestrator | Alinhamento |
|----------|------|---------------------------|-------------|
| 01 | O que são LLMs | — (contexto) | ✅ Bom |
| 02 | Arquitetura Transformer | — (contexto) | ✅ Bom |
| 03 | Setup de ambiente | `docker-compose.yml`, `.env` | ✅ Bom |
| 04 | Primeiros passos | `gateway/llm.py` (OllamaClient) | ⚠️ Usa qwen2.5:7b, não mostra evolução |
| 05 | Quantização | — (teórico com exemplos) | ✅ Bom |
| 06 | Otimização de inferência | — (teórico) | ❌ Fórmula KV Cache com num_heads errado |
| 07 | Frameworks de serving | `docker-compose.yml` (ollama) | ✅ Bom |
| 08 | BERT e encoders | `gateway/injection_detector.py`, `gateway/sanitize.py` | ⚠️ Falta pipeline de treino BERTimbau |
| 09 | Conceitos de fine-tuning | `train/build_dataset.py` | ✅ Bom |
| 10 | Preparando datasets | `train/build_dataset.py` (4-stage pipeline) | ⚠️ Não referencia o pipeline real |
| 11 | Treinando LoRA | `train/colab_train_lora.ipynb` | ✅ Excelente — reprodução fiel |
| 12 | Avaliação e métricas | `evals/eval_routing.py`, `evals/eval_domains.py` | ⚠️ Falta metodologia dos 5 gates |
| 13 | Exportando para produção | `train/` → merge + GGUF | ✅ Bom |
| 14 | Pipeline RAG | `gateway/embedder.py`, `gateway/semantic_router.py` | ❌ Usa nomic-embed-text; real usa SBERT |
| 15 | Chunking e indexação | `gateway/semantic_router.py` (Qdrant) | ⚠️ Falta consenso + score_gap |
| 16 | RAG com agentes | `gateway/agents.py`, `gateway/graph.py` | ✅ Bom — function calling + LangGraph |
| 17 | Graph RAG | — (não usado no AI-Orchestrator) | ⚠️ Capítulo standalone, não referencia o projeto |
| 18 | Benchmarking | `evals/fase0_bench.py` | ⚠️ Métricas reais dispersas |
| 19 | Multi-agente | `gateway/graph.py`, `gateway/agents.py`, `gateway/router.py` | ✅ Excelente — arquitetura fiel |
| 20 | Segurança e governança | `gateway/sanitize.py`, `gateway/security.py`, `docs/AUDIT_2026-06-14.md` | ✅ Bom — 14 patterns + BERTimbau. Falta auth/rate-limit |
| 21 | MLOps | `docker-compose.yml`, `gateway/tracing.py`, `gateway/metrics.py` | ✅ Excelente — stack completo |

**Legenda:** ✅ Alinhado | ⚠️ Alinhamento parcial | ❌ Desalinhado

---

## RESUMO QUANTITATIVO

| Categoria | Problemas | Alta | Média | Baixa |
|-----------|-----------|------|-------|-------|
| Erros factuais/técnicos | 4 | 2 | 1 | 1 |
| Desalinhamentos com implementação | 5 | 2 | 2 | 1 |
| Ortografia/acentuação | 2 | 1 | 0 | 1 |
| Estrutura/numeração | 3 | 1 | 1 | 1 |
| Código/exemplos | 3 | 0 | 2 | 1 |
| Omissões AI-Orchestrator | 8 | 1 | 3 | 4 |
| Melhorias estruturais/didáticas | 7 | 1 | 4 | 2 |
| **TOTAL** | **32** | **8** | **13** | **11** |

---

## PLANO DE AÇÃO RECOMENDADO

### Fase 1 — Imediato (antes da publicação, ~2-3 dias)
1. Corrigir URL do Ollama (`ollama.ai` → `ollama.com`) em bibliografia e cap04
2. Corrigir fórmula do KV Cache (`num_heads` → `num_kv_heads`) no cap06
3. Revisar acentuação nos capítulos 4, 6, 7, 18
4. Atualizar sumário do README (adicionar cap21)

### Fase 2 — Alinhamento com AI-Orchestrator (~1 semana)
5. Cap04 ou Cap13: Adicionar seção "A jornada do modelo" com evolução 30B→7B→9B-LoRA
6. Cap14-15: Documentar evolução nomic-embed-text → SBERT + Embedder Protocol
7. Cap15: Adicionar mecanismo de consenso + score_gap do semantic router
8. Cap08/Cap20: Adicionar pipeline de treino do BERTimbau (400 exemplos sintéticos)
9. Cap19: Adicionar circuit breaker + tool registry via OpenAPI completo
10. Cap12/Cap18: Consolidar sistema de avaliação com 5 gates

### Fase 3 — Completude (~2 semanas)
11. Cap19 ou Cap21: Adicionar SSE heartbeat, Rate Limiter, AccessTokenGuard
12. Cap21: Documentar Cloudflare Tunnel, Langfuse Cloud vs self-hosted
13. Cap01: Adicionar seção "A jornada deste livro" com arco narrativo
14. Adicionar prefácio contextualizando o AI-Orchestrator
15. Gerar índice remissivo e glossário
16. Adicionar exercícios baseados em extensões do AI-Orchestrator

---

## CATEGORIA 8: ENRIQUECIMENTO TÉCNICO VIA git-repo/

A pasta `git-repo/` contém 4 repositórios de referência (6.055 arquivos no total) que são as fontes originais dos notebooks adaptados no ebook. A análise revelou **conteúdo técnico substancial ainda não aproveitado** que pode enriquecer significativamente vários capítulos.

### 8.1 Fontes disponíveis no git-repo/

| Repositório | Fonte | Conteúdo |
|-------------|-------|----------|
| `llm-model-inference` | O'Reilly "Hands-On LLM Serving and Optimization" (Wang & Hu, 2025) | 16 notebooks + ~20 scripts Python em 7 capítulos |
| `RAG-with-Python-Cookbook` | O'Reilly "RAG with Python Cookbook" (Polzer) | 17 notebooks + 16 scripts Python em 11 capítulos |
| `agent-driven-design` | Framework conceitual de design de agentes | 5 docs core + 4 patterns + 3 produção + ~20 exemplos |
| `agents-integration-patterns` | Catálogo de 26 padrões de integração multi-agente | 26 patterns + 106 samples (Python/Java/C#/TS) + 28 diagramas |

### 8.2 Enriquecimento por capítulo — llm-model-inference

**Cap02 — Arquitetura Transformer:**
- **Fonte:** `ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb` — walkthrough completo do Transformer, passo a passo com visualizações de atenção
- **Fonte:** `ch02/ch2_Workthrough_LLM_execution.ipynb` — execução completa de LLM com KV cache
- **Fonte:** `ch02/ch2_Batching.ipynb` — estratégias de batching (static, dynamic, continuous)
- **Fonte:** `ch02/ch2_Streaming.ipynb` — streaming inference implementado
- **Enriquecimento proposto:** Adicionar visualizações de self-attention do notebook como figuras no capítulo. Incluir exercício prático: "Execute o notebook e observe os pesos de atenção para diferentes frases em português."

**Cap04 — Primeiros passos com LLMs locais:**
- **Fonte:** `ch02/ch2_Run_LLM_With_vLLM.ipynb` — execução do Qwen com vLLM
- **Enriquecimento proposto:** Adicionar comparação lado a lado: mesmo prompt no Ollama vs vLLM, mostrando diferença de throughput.

**Cap06 — Otimização de inferência:**
- **Fonte:** `ch07/SpecDecode.ipynb` — speculative decoding com Eagle3 e n-gram
- **Fonte:** `ch07/LMCache.ipynb` — KV cache avançado com LMCache para contextos longos (128K+)
- **Enriquecimento proposto:** Expandir seção 6.3 com exemplos concretos do SpecDecode.ipynb. Adicionar seção "KV Cache para contextos longos" usando LMCache.ipynb.

**Cap07 — Frameworks de serving:**
- **Fonte:** `ch03/single_model_llm_serving/` — implementação completa de um servidor LLM do zero (Flask + Python): `model_executor.py`, `model_manager.py`, `model_worker.py`, `workload_manager.py`
- **Fonte:** `ch03/multi_model_serving/` — NVIDIA Triton multi-modelo com DenseNet ONNX
- **Fonte:** `ch08/SGLang.ipynb`, `ch08/TensorRT_LLM.ipynb`, `ch08/llamaCpp.ipynb`
- **Enriquecimento proposto:** O código do single-model serving (ch03) é material de altíssimo valor didático — implementa workload management, batching, streaming, e model executors do zero. **Recomendação forte:** incluir como seção "Construindo um servidor LLM do zero" no cap07, mostrando a arquitetura interna que o Ollama/vLLM abstraem.

**Cap09-13 — Fine-Tuning:**
- **Fonte:** `ch06/quantization_3way_300.ipynb` — comparativo GPTQ vs AWQ vs GGUF com benchmark vLLM
- **Fonte:** `ch09/model_optimization_in_practice.ipynb` — plano de otimização end-to-end para Qwen3-14B
- **Enriquecimento proposto:** O notebook de otimização do Qwen3-14B (ch09) é um **case study completo** que espelha a jornada AI-Orchestrator (30B→7B→9B). Incluir como apêndice ou capítulo bônus: "Otimização na prática: caso Qwen3-14B".

### 8.3 Enriquecimento por capítulo — RAG-with-Python-Cookbook

**Cap14 — Pipeline RAG:**
- **Fonte:** `ch01_RAG_intro/rag_basics.ipynb` — pipeline RAG completo do zero
- **Fonte:** `ch02_generation/generation.ipynb` — prompt engineering para RAG
- **Fonte:** `ch03_loading_data/loading_data_to_RAG.ipynb` — carregamento multimodal: PDF, DOCX, imagens, áudio, vídeo
- **Enriquecimento proposto:** Expandir cap14 com carregamento multimodal (o ebook atual só cobre texto). Adicionar seção "Prompt engineering para RAG" do ch02.

**Cap15 — Chunking e indexação:**
- **Fonte:** `ch04_data_preparation_chunking_data/chunking_data.ipynb` — 5 estratégias de chunking: fixed-size, recursive, semantic, sentence-based, agentic
- **Fonte:** `ch05_text_embedding/text_embeddings.ipynb` — seleção de modelo de embedding, dimensionalidade
- **Fonte:** `ch06_similarity_search_vector_databases/vector_databases.ipynb` — HNSW, PGVector, Qdrant, ChromaDB comparados
- **Fonte:** `ch07_retrieval/retrieval_techniques.ipynb` — metadata filtering, reranking, query decomposition, hybrid search
- **Enriquecimento proposto:** Cap15 atual cobre apenas chunking simples + Qdrant básico. Expandir com: (a) comparação das 5 estratégias de chunking com métricas, (b) reranking pós-retrieval, (c) query decomposition para perguntas complexas.

**Cap16 — RAG com agentes:**
- **Fonte:** `ch08_agentic_rag/8.4_building_agentic_system_function_calling/` — sistema agentic sem framework (function calling puro)
- **Fonte:** `ch08_agentic_rag/8.5_accelerating_agents_asyncio/` — aceleração de agentes com AsyncIO
- **Fonte:** `ch08_agentic_rag/8.7_mcp_tools/` — integração MCP (Playwright, múltiplos servidores)
- **Fonte:** `ch08_agentic_rag/8.8_agentic_system_langgraph/` — LangGraph para agentes
- **Enriquecimento proposto:** Adicionar seção "MCP Tools: expandindo agentes com ferramentas externas" — o AI-Orchestrator usa OpenAPI auto-discovery como alternativa ao MCP, comparar as duas abordagens.

**Cap17 — Graph RAG:**
- **Fonte:** `ch09_graph_rag/` — 5 notebooks Neo4j: criação de grafo SLA, enriquecimento, Cypher queries, embeddings + vector search, extensões
- **Enriquecimento proposto:** Os 5 notebooks do ch09 cobrem Graph RAG de forma muito mais completa que o cap17 atual. Expandir com: (a) criação progressiva do grafo (9.1→9.2→9.3), (b) embeddings no Neo4j para busca semântica, (c) otimização de grafo para RAG.

**Cap18 — Benchmarking:**
- **Fonte:** `ch10_rag_evaluation/rag_evaluation_techniques.ipynb` — métricas de avaliação RAG: faithfulness, relevance, context precision/recall
- **Enriquecimento proposto:** Expandir cap18 com métricas específicas de RAG (além das métricas de serving). Adicionar seção "Avaliando a qualidade do RAG".

### 8.4 Enriquecimento por capítulo — agent-driven-design

Este repositório é **conceitual** (framework de design, não código executável) e oferece vocabulário e princípios que elevariam significativamente os capítulos 16-21.

**Cap19 — Multi-agente:**
- **Enriquecimento proposto:** Estruturar o capítulo usando os 4 conceitos centrais do ADD:
  1. **Model vs Harness:** O LLM é o Model (raciocínio); todo o resto é Harness (prompts, tools, routing, validação). O AI-Orchestrator implementa esta separação: `gateway/graph.py` é o Harness, `gateway/llm.py` é o Model. Esta distinção é fundamental e não está explícita no ebook.
  2. **Topologias:** O AI-Orchestrator usa **Pipeline** (sanitize→classify→dispatch→synthesize) + **Specialist Pool** (router→domain agents). Explicitar estas topologias com os diagramas do ADD.
  3. **Decomposition triggers:** Por que o AI-Orchestrator tem 4 agentes de domínio separados? Resposta ADD: "Domain separation — diferentes vocabulários, invariantes, tool surfaces."
  4. **Agent Context Boundary:** Cada agente de domínio (financas, rh, estoque, vendas) tem seu próprio `ToolRegistry` com ferramentas isoladas — implementação exata do conceito ADD de boundary.

**Cap20 — Segurança:**
- **Enriquecimento proposto:** Mapear defesas do AI-Orchestrator para padrões ADD:

| Defesa AI-Orchestrator | Padrão ADD |
|------------------------|------------|
| `sanitize.py` strip ChatML + 14 regex + BERTimbau | **Prompt Firewall** + **Trust Boundary** |
| `ToolRegistry` com escopo por domínio | **Least-Privilege Tool Scope** |
| `CircuitBreaker` (3 falhas → OPEN 30s) | **Circuit Breaker** |
| `X-Internal-Key` HMAC entre gateway e serviços | **Trust Boundary** |

**Cap21 — MLOps:**
- **Enriquecimento proposto:** Estruturar usando o Eval Pyramid do ADD:
  - **Model Eval:** `eval_routing.py` (acurácia do LLM)
  - **Agent Eval:** `eval_domains.py` (acurácia do agente com tools)
  - **System Eval:** `eval_injection.py` (comportamento sistêmico sob ataque)
  
  Adicionar o princípio ADD: "Failures cascade upward — fix Model evals first, then Agent evals, then System evals."

### 8.5 Enriquecimento por capítulo — agents-integration-patterns

Este catálogo de 26 padrões fornece vocabulário preciso para descrever a arquitetura do AI-Orchestrator.

**Cap19 — Mapeamento de padrões do AI-Orchestrator:**

| Componente AI-Orchestrator | Padrão do catálogo |
|---------------------------|-------------------|
| `router.py` classifica e roteia para domínio | **Content-Based Router** |
| `graph.py` fan-out para múltiplos domínios | **Scatter-Gather** |
| `graph.py` pipeline sanitize→classify→dispatch→synthesize | **Pipeline** |
| `agents.py` loop tool-calling por domínio | **Orchestrator** (coordena tools, não agentes) |
| `ToolRegistry` com OpenAPI auto-discovery | **Tool Provider** |
| `sanitize.py` injeção de contexto do sistema | **Context Injection** (caso inverso — defesa contra injeção maliciosa) |

**Cap20 — Padrões de resiliência implementados:**

| Mecanismo | Padrão |
|-----------|--------|
| `CircuitBreaker` (3 falhas → OPEN 30s) | **Circuit Breaker** |
| `with_deadline(420s)` nos evals | **Checkpoint & Resume** (timeout tratado como checkpoint) |
| `MetricsCollector` com cache stale | **Dead Letter Agent** (métricas degradadas, não perdidas) |
| `X-Internal-Key` HMAC | **Trust Boundary** |
| `sanitize.py` boundary tripla | **Prompt Firewall** |
| `ToolRegistry` escopo por domínio | **Least-Privilege Tool Scope** |

**Enriquecimento proposto para cap19-20:** Incluir uma seção "Padrões de integração em ação" com tabela mapeando cada componente do AI-Orchestrator ao padrão correspondente do catálogo, com o diagrama do padrão (os 28 PNGs estão em `img/`). Isto dá ao leitor vocabulário para discutir arquitetura de agentes além da implementação específica.

**MAST Failure Taxonomy (14 modos de falha):**
O arquivo `FAILURE-MAP.md` mapeia 14 modos de falha empiricamente observados (Cemri et al., NeurIPS 2025) para padrões de mitigação. Incluir no cap20 ou 21:

| Modo de falha | Categoria | Mitigação no AI-Orchestrator |
|---------------|-----------|------------------------------|
| Tool parameter mismatch | Execution | OpenAPI schema validation no `ToolRegistry` |
| Missing tool capability | Specification | `ToolRegistry` fallback com lista de ferramentas disponíveis |
| Context overflow | Communication | `max_seq_length=4096` + chunking no prompt |
| Agent hallucinated tool call | Execution | `train_on_responses_only` + LoRA fine-tune |
| Unauthorized action attempt | Security | `Least-Privilege Tool Scope` por domínio |

### 8.6 Dados técnicos quantitativos para enriquecer capítulos

**Métricas de quantization (ch06/quantization_3way_300.ipynb):**
- GPTQ INT4: tamanho ~4 GB, TPS ~45 (RTX 3060), perplexidade +2.1%
- AWQ INT4: tamanho ~4 GB, TPS ~48 (RTX 3060), perplexidade +1.3%
- GGUF Q4_K_M: tamanho ~4 GB, TPS ~42 (RTX 3060), perplexidade +1.8%
- Estes números concretos enriqueceriam cap05.

**Métricas de speculative decoding (ch07/SpecDecode.ipynb):**
- Eagle3 draft model: aceleração 1.4–1.8× em tarefas de código
- N-gram draft: aceleração 1.1–1.3×, overhead mínimo
- Incluir no cap06.

**Métricas de RAG (ch10_rag_evaluation/):**
- Faithfulness: 0.89 (GPT-4 judge)
- Context precision: 0.92
- Context recall: 0.87
- Incluir no cap18.

### 8.7 Recomendações de novos conteúdos a partir do git-repo/

| Prioridade | Conteúdo | Fonte | Capítulo alvo |
|-----------|----------|-------|---------------|
| **ALTA** | Single-model serving do zero (workload manager, model executor, batching) | `llm-model-inference/ch03/` | Cap07 |
| **ALTA** | Model vs Harness — framework conceitual ADD | `agent-driven-design/core/` | Cap02 + Cap19 |
| **ALTA** | Mapeamento AI-Orchestrator → padrões de integração | `agents-integration-patterns/patterns/` | Cap19-20 |
| **ALTA** | MAST failure taxonomy + mitigação | `agents-integration-patterns/FAILURE-MAP.md` | Cap20-21 |
| **MÉDIA** | 5 estratégias de chunking comparadas com métricas | `RAG-with-Python-Cookbook/ch04/` | Cap15 |
| **MÉDIA** | Reranking + query decomposition | `RAG-with-Python-Cookbook/ch07/` | Cap15 |
| **MÉDIA** | Carregamento multimodal (PDF, imagens, áudio, vídeo) | `RAG-with-Python-Cookbook/ch03/` | Cap14 |
| **MÉDIA** | Graph RAG progressivo (5 notebooks Neo4j) | `RAG-with-Python-Cookbook/ch09/` | Cap17 |
| **MÉDIA** | Métricas de avaliação RAG (faithfulness, relevance) | `RAG-with-Python-Cookbook/ch10/` | Cap18 |
| **MÉDIA** | MCP Tools vs OpenAPI auto-discovery | `RAG-with-Python-Cookbook/ch08.7/` | Cap16 |
| **MÉDIA** | Eval Pyramid (Model→Agent→System) | `agent-driven-design/production/evals/` | Cap12 + Cap21 |
| **BAIXA** | Caso Qwen3-14B end-to-end optimization | `llm-model-inference/ch09/` | Apêndice |
| **BAIXA** | Prompt engineering para RAG | `RAG-with-Python-Cookbook/ch02/` | Cap14 |
| **BAIXA** | Streamlit deployment Docker + AWS | `RAG-with-Python-Cookbook/ch11/` | Cap21 |

---

## PLANO DE AÇÃO ATUALIZADO

### Fase 1 — Imediato (antes da publicação, ~2-3 dias)
1. Corrigir URL do Ollama (`ollama.ai` → `ollama.com`) em bibliografia e cap04
2. Corrigir fórmula do KV Cache (`num_heads` → `num_kv_heads`) no cap06
3. Revisar acentuação nos capítulos 4, 6, 7, 18
4. Atualizar sumário do README (adicionar cap21)

### Fase 2 — Alinhamento com AI-Orchestrator (~1 semana)
5. Cap04 ou Cap13: Adicionar seção "A jornada do modelo" com evolução 30B→7B→9B-LoRA
6. Cap14-15: Documentar evolução nomic-embed-text → SBERT + Embedder Protocol
7. Cap15: Adicionar mecanismo de consenso + score_gap do semantic router
8. Cap08/Cap20: Adicionar pipeline de treino do BERTimbau (400 exemplos sintéticos)
9. Cap19: Adicionar circuit breaker + tool registry via OpenAPI completo
10. Cap12/Cap18: Consolidar sistema de avaliação com 5 gates

### Fase 3 — Enriquecimento via git-repo (~2 semanas)
11. Cap19-20: Adicionar framework ADD (Model vs Harness, 4 topologias, decomposition triggers)
12. Cap19-20: Mapear AI-Orchestrator → 26 padrões de integração com diagramas
13. Cap20-21: Incluir MAST failure taxonomy (14 modos de falha + mitigação)
14. Cap07: Adicionar seção "Construindo um servidor LLM do zero" (single-model serving do llm-model-inference)
15. Cap15: Expandir com 5 estratégias de chunking + reranking + query decomposition
16. Cap17: Expandir Graph RAG com 5 notebooks Neo4j progressivos
17. Cap18: Adicionar métricas de avaliação RAG (faithfulness, context precision/recall)
18. Cap16: Adicionar seção MCP Tools vs OpenAPI auto-discovery

### Fase 4 — Completude (~2 semanas)
19. Cap19 ou Cap21: Adicionar SSE heartbeat, Rate Limiter, AccessTokenGuard
20. Cap21: Documentar Cloudflare Tunnel, Langfuse Cloud vs self-hosted
21. Cap01: Adicionar seção "A jornada deste livro" com arco narrativo
22. Adicionar prefácio contextualizando o AI-Orchestrator
23. Gerar índice remissivo e glossário
24. Adicionar exercícios baseados em extensões do AI-Orchestrator

---

## RESUMO QUANTITATIVO FINAL

| Categoria | Problemas/Oportunidades | Alta | Média | Baixa |
|-----------|------------------------|------|-------|-------|
| Erros factuais/técnicos | 4 | 2 | 1 | 1 |
| Desalinhamentos com implementação | 5 | 2 | 2 | 1 |
| Ortografia/acentuação | 2 | 1 | 0 | 1 |
| Estrutura/numeração | 3 | 1 | 1 | 1 |
| Código/exemplos | 3 | 0 | 2 | 1 |
| Omissões AI-Orchestrator | 8 | 1 | 3 | 4 |
| Melhorias estruturais/didáticas | 7 | 1 | 4 | 2 |
| Enriquecimento via git-repo | 14 | 4 | 7 | 3 |
| **TOTAL** | **46** | **12** | **20** | **14** |

**Resumo dos 4 repositórios:** 16 + 17 + 0 + 0 notebooks, ~136 samples de código, 26 padrões catalogados, 28 diagramas, 14 modos de falha mapeados, 4 topologias de agente, 1 framework conceitual (ADD). Total: ~6.000 arquivos de conteúdo técnico complementar já disponível localmente.
