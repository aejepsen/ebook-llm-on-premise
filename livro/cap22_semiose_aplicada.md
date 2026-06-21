# Capítulo 22 — Semiose Aplicada: Construção de Significado em Pipelines de IA

Nos capítulos anteriores tratamos embeddings (cap. 8, 14, 15), recuperação aumentada (cap. 14–17) e a arquitetura multi-agente do AI-Orchestrator (cap. 19). Este capítulo costura esses fios em torno de um problema que aparece em todo sistema de IA aplicado a contextos complexos: **como o sistema constrói e preserva significado** quando a pergunta isolada não basta para resolvê-lo.

No AI-Orchestrator, esse conjunto de técnicas recebeu o nome de **Semiose** — termo emprestado da semiótica (o estudo de como signos produzem significado), mas tratado aqui de forma estritamente operacional: um processo de engenharia que enriquece, relaciona e reordena representações textuais para que o significado correto sobreviva ao longo do pipeline (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`). Não nos interessa a discussão filosófica; interessa o ganho mensurável de desempenho semântico em dados e diálogos complexos.

## O problema: geometria não é significado

Um banco vetorial calcula similaridade de cosseno entre vetores. Isso é geometria: mede o ângulo entre duas representações, não a intenção por trás delas. A palavra "banco" flutua entre "banco de jardim" e "banco financeiro" se nada no entorno desambiguar o termo — e o cosseno, sozinho, não tem como decidir (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, seção *Diagnóstico*).

Esse é o ponto cego que motiva o capítulo. A recuperação aumentada (RAG) nasceu justamente para dar aos modelos uma "memória não-paramétrica" acessível: no trabalho seminal, a memória paramétrica é um modelo seq2seq pré-treinado e a memória não-paramétrica é um índice vetorial denso da Wikipédia, acessado por um recuperador neural (Lewis et al., 2020, p. 1). O motivo declarado é prático: modelos puramente paramétricos "não conseguem expandir ou revisar sua memória com facilidade, não oferecem visão direta sobre suas previsões e podem produzir 'alucinações'" (Lewis et al., 2020, p. 1).

Mas recuperar não é o mesmo que **compreender em contexto**. A própria Anthropic resume a limitação da abordagem ingênua: "soluções tradicionais de RAG removem contexto ao codificar a informação, o que frequentemente faz o sistema falhar em recuperar a informação relevante" (Anthropic, 2024). Em outras palavras: ao quebrar o documento em chunks e vetorizá-los isoladamente, perdemos as pistas que davam sentido a cada trecho. O mesmo vale para uma conversa: a pergunta "qual o status?" só tem sentido à luz do turno anterior.

> **Princípio:** o significado de um signo (uma query, um chunk, uma entidade) raramente está contido nele mesmo. Está na relação com o contexto que o cerca. Engenharia de semântica é, em boa medida, a engenharia de **reintroduzir** esse contexto de forma controlada.

## Três alavancas para desempenho semântico

A literatura recente de RAG organiza as otimizações de recuperação em dois momentos: **pré-recuperação** (otimizar a query e a indexação) e **pós-recuperação** (reordenar e comprimir o que foi recuperado) (Gao et al., 2023, p. 3). O survey de Gao et al. classifica os sistemas em *Naive RAG*, *Advanced RAG* e *Modular RAG*, sendo as estratégias de pré e pós-recuperação a marca registrada do estágio *Advanced* (Gao et al., 2023, p. 1, 3).

O AI-Orchestrator instancia essas alavancas em três camadas, mapeadas sobre o grafo `sanitize → enrich → classify → dispatch → synthesize` já apresentado no capítulo 19:

```
sanitize → enrich → classify → dispatch → synthesize
              │         │           │
          CAMADA A   CAMADA C    CAMADA B
        enriquecer  reordenar   relacionar
        a query    candidatos  entre domínios
        (pré-      (pós-       (grafo de
        recuperação)recuperação) conhecimento)
```

| Camada | Alavanca semântica | Técnica de referência | Arquivo no projeto |
|--------|--------------------|-----------------------|--------------------|
| A | Enriquecimento contextual da query (pré-recuperação) | Query rewriting/expansion (Gao et al., 2023); Contextual Retrieval (Anthropic, 2024) | `gateway/query_enricher.py` |
| B | Relações cross-domínio via grafo | GraphRAG (Edge et al., 2024) | `gateway/knowledge_graph.py` |
| C | Re-ranking contextual (pós-recuperação) | Bi-encoder + rerank (Reimers & Gurevych, 2019; Gao et al., 2023) | `gateway/semantic_router.py` |

As três camadas são **opt-in** e **degradam graceful** — se a infraestrutura de uma delas estiver ausente, o pipeline cai para o comportamento anterior sem quebrar (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, seção *Degradação Graceful*). Essa disciplina é herança do princípio Harness-first discutido no capítulo 19: lógica determinística primeiro, modelo só quando necessário.

## Camada A — Enriquecer o signo antes de vetorizar

A primeira alavanca ataca o ponto cego da pergunta isolada. Em vez de embedar "qual o status?" cru, o sistema reconstrói a query com o contexto estruturado do diálogo antes de classificá-la.

Essa é a versão conversacional de duas técnicas conhecidas. A primeira é a **transformação de query** na fase de pré-recuperação: "métodos comuns incluem reescrita de query, transformação de query, expansão de query e outras técnicas" (Gao et al., 2023, p. 4). A expansão, em particular, "enriquece o conteúdo da query" para melhorar a recuperação (Gao et al., 2023, p. 8). A segunda é a ideia de **embeddings contextuais**: anexar o contexto que dá sentido ao trecho antes de vetorizá-lo — abordagem que, segundo a Anthropic, reduz falhas de recuperação em 49% e, combinada com reordenação, em 67% (Anthropic, 2024).

### O enricher determinístico

A implementação no AI-Orchestrator é deliberadamente Harness: **zero chamada de LLM**. O enriquecedor reúne sinais estruturados do estado conversacional e monta um prefixo controlado:

```python
# gateway/query_enricher.py — esboço do fluxo (autoral)
# 1) Sinais vêm de metadata validada, nunca de texto cru do histórico
sinais = gather_signals(state, spacy_enabled=False)
#    - last_domain: domínio do turno anterior (_last_route, validado pelo classifier)
#    - recent_entities: entidades extraídas por regex (SKU, CPF, CNPJ, R$)

# 2) Só enriquece se houver contexto E não houver troca de tópico
query, enriquecida = enrich_query(pergunta, sinais)
#    → "[domínio: estoque; entidades: CAD-ERG-001] qual o status?"
```

Duas decisões de projeto merecem destaque, por serem o que separa "enriquecer" de "poluir":

**1. Segurança do contexto.** O enricher só usa metadata estruturada (`_last_route`, IDs de entidade extraídos por regex). Ele **nunca** concatena texto cru do histórico, justamente para não reabrir a porta de prompt injection fechada no nó `sanitize` (cap. 20) (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, Camada A, *Regra de segurança*).

**2. Detecção de troca de tópico.** Enriquecer cegamente é perigoso: se o usuário muda de assunto ("e as férias do João?" depois de falar de estoque), arrastar o domínio anterior degrada a recuperação em vez de melhorá-la. O `query_enricher.py` consulta o mesmo dicionário de palavras-chave do roteador léxico (`_DOMAIN_KEYWORDS`) e, se a nova pergunta tem sinais fortes de outro domínio, **descarta** o enriquecimento e deixa a query crua seguir para o classificador (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, Camada A, *Vocabulário compartilhado*).

A extração de entidades segue uma cascata custo-consciente, no mesmo espírito do detector de injection do capítulo 8:

| Camada | Mecanismo | Quando dispara | Latência |
|--------|-----------|----------------|----------|
| 1 — Harness | Regex (SKU, CPF, CNPJ, R$) | Sempre | ~0,1 ms |
| 2 — quase-Model | spaCy `pt_core_news_sm` | Só se regex vazio + flag `SPACY_ENABLED` | ~15 ms |

O spaCy é *lazy-load* e opcional: o regex cobre os padrões estruturados do domínio, e o NER entra apenas quando há nomes próprios fora desses padrões. É a aplicação literal do princípio "o modelo só quando o harness não resolve".

## Camada B — A teia de signos: relações cross-domínio

Busca vetorial é excelente para "encontre algo parecido com isto", mas fraca para perguntas **relacionais** e **globais** — aquelas cuja resposta depende de conexões espalhadas por vários documentos ou domínios. Um sistema que só tem cosseno não sabe que "o SKU CAD-ERG-001 é fornecido pela Beta, que atende à conta vencida no Financeiro".

É o problema que o GraphRAG ataca. A proposta de Edge et al. usa um LLM para construir, a partir do corpus, um **grafo de conhecimento de entidades** — "nós correspondem a entidades-chave no corpus e arestas representam relações entre essas entidades" (Edge et al., 2024, p. 2). Em seguida, o grafo é particionado em uma hierarquia de comunidades de entidades relacionadas, usando detecção de comunidades (por exemplo, o algoritmo Leiden) (Edge et al., 2024, p. 4), e cada comunidade recebe um resumo pré-gerado (Edge et al., 2024, p. 1). Para perguntas de "sensemaking global" sobre corpora grandes, isso traz melhora substancial em **abrangência e diversidade** das respostas frente a um RAG convencional (Edge et al., 2024, p. 1).

### O recorte do AI-Orchestrator

Aqui é importante ser honesto sobre o que o projeto faz e o que **não** faz. O AI-Orchestrator não implementa a sumarização hierárquica de comunidades do GraphRAG. Ele adota a parte da ideia que resolve seu problema concreto — **travessia dirigida de relações entre domínios** — e a integra de forma frugal: o Neo4j é exposto como uma *virtual tool* chamada `expand_context`, registrada no `ToolRegistry` (cap. 19). O agente decide **se** e **quando** expandir o contexto, e o `CircuitBreaker` já existente protege contra o Neo4j fora do ar (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, *Desvios Arquiteturais*, Camada B).

```python
# gateway/knowledge_graph.py — contrato da virtual tool (autoral)
# expand_context(entity_name, entity_type, target_domain="")
#   → travessia 1..2 saltos a partir da entidade
#   → filtra por domínio-alvo, exclui o nó de origem, LIMIT por domínio
#   → retorna {related: [{name, type, domain, path}], count}
# Neo4j offline → {related: [], note: "indisponível"}  (degradação graceful)
```

Decisões de projeto que tornam a travessia segura e previsível:

- **LIMIT por domínio**, não global — distribuição de resultados previsível.
- **`DISTINCT`** e exclusão do nó de origem — sem duplicatas nem auto-referência.
- **Opt-in por domínio**: `GRAPH_ENABLED_DOMAINS = ("estoque", "vendas", "financas")`. RH fica de fora porque suas relações são simples e não se beneficiam da expansão (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, *Desvios Arquiteturais*).

O resultado é uma "teia de signos" sob demanda: quando o agente de Vendas precisa entender as dependências de um produto, ele puxa do grafo as relações que a busca vetorial jamais traria, porque elas não estão em nenhum texto contíguo — estão na **estrutura** que liga os domínios.

## Camada C — Re-ranking: o interpretante que decide

A última alavanca atua na fase de pós-recuperação. A recuperação vetorial é rápida porque usa um **bi-encoder**: cada texto vira um vetor independente, e a busca é uma comparação de cosseno. Essa eficiência é exatamente o motivo de existir o Sentence-BERT.

Reimers & Gurevych mostram o custo da alternativa ingênua. Um *cross-encoder* como o BERT, que processa os dois textos juntos, é preciso mas não escala para busca por similaridade: encontrar o par mais similar numa coleção de 10.000 sentenças exige cerca de 50 milhões de inferências (~65 horas) (Reimers & Gurevych, 2019, p. 1). O SBERT resolve isso com uma rede siamesa que produz embeddings comparáveis por cosseno, reduzindo o cálculo dos 10.000 embeddings para ~5 segundos, com a comparação de cosseno levando ~0,01 segundo (Reimers & Gurevych, 2019, p. 2). É esse modelo — `paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensões — que alimenta o *semantic router* do AI-Orchestrator (cap. 19; glossário, *SBERT*).

O bi-encoder dá velocidade, mas perde a interação fina entre query e candidato. Daí o padrão consagrado de **recuperar rápido com o bi-encoder e reordenar os melhores candidatos** — os métodos de pós-recuperação incluem justamente "reordenar chunks" (Gao et al., 2023, p. 4). O AI-Orchestrator aplica esse princípio, mas com um *reranker* leve e contextual em vez de um cross-encoder caro:

```python
# gateway/semantic_router.py — boost contextual aditivo (autoral)
# Para cada candidato do Qdrant:
#   _raw_score = score          # preserva o cosseno original
#   se context_domain ∈ candidato.payload["domains"]:
#       score = min(_raw_score + CONTEXT_BOOST, 1.0)   # desempate
# reordena por score
```

A escolha de um **boost aditivo com teto** (`min(score + 0.05, 1.0)`), e não multiplicativo, é proposital: ele **desempata** matches já bons usando o contexto do turno anterior, sem resgatar matches ruins nem destruir a semântica do cosseno. O valor `CONTEXT_BOOST` é configurável por ambiente, e a reordenação preserva o score original em `_raw_score` para auditoria. Quando não há `context_domain` ou o re-ranking está desligado (`RERANK_ENABLED=0`), o roteador opera sem boost — degradação graceful, mais uma vez.

> **Princípio:** recuperar e reordenar são papéis distintos. O bi-encoder responde "o que é parecido?"; o re-ranker responde "o que é relevante **aqui e agora**?". A segunda pergunta é onde o contexto vira desempate.

## Como medir desempenho semântico

Otimizar significado exige medir significado — e métricas de sobreposição lexical (BLEU, ROUGE) são insuficientes para isso, porque o objetivo é justamente capturar equivalência de sentido, não de palavras. O AI-Orchestrator define um framework de avaliação próprio para a Semiose, em `evals/eval_semiose.py`, organizado pelas mesmas camadas (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, seção *Avaliação de Performance*).

Alguns indicadores ilustram o tipo de medição que faz sentido aqui:

| Camada | Métrica | O que mede |
|--------|---------|-----------|
| A | Entity Propagation F1 | Se as entidades certas são propagadas entre turnos |
| A | False Enrichment Rate (FER) | Frequência de enriquecimento indevido (deveria ter ficado cru) |
| A | Topic Switch Accuracy (TSA) | Acerto na detecção de troca de tópico |
| A | Enrichment Cosine Preservation | Cosseno entre query original e enriquecida (≈ 1 − drift): o enriquecimento preserva o sentido? |
| C | Contextual Gain Ratio | Quanto o boost contextual melhora o roteamento |
| C | Exact-Match Routing | Igualdade exata do conjunto de domínios previsto vs. esperado |
| B | Relation Validity@5 | Fração das relações retornadas que pertencem a domínio conhecido |

Dois pontos de método valem registro. Primeiro, **nomes honestos**: a métrica de preservação é cosseno SBERT, não BERTScore token-level, e por isso recebe o nome do que de fato calcula; o "acerto de roteamento" é igualdade de conjunto, não micro-F1 sobre rótulos. Nomear a métrica pelo que ela mede evita a ilusão de rigor. Segundo, **avaliação por camada**: a Camada A roda offline (sem Qdrant/Neo4j), o que permite validar o enriquecedor de forma barata e determinística antes de subir a infraestrutura das outras camadas.

Os resultados offline da Camada A, sobre 150 casos multi-turn, ficaram em Entity Propagation F1 = 0,973, FER = 0,026 e TSA = 0,963 (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, *Resultados por split*). As duas falhas observadas são instrutivas: ambas ocorrem quando a query mistura palavras-chave de dois domínios, e o detector determinístico de troca de tópico não tem como decidir sozinho — situação em que o projeto, coerente com o Harness-first, delega a decisão ao classificador LLM do nó seguinte (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, *Falhas identificadas*). É a fronteira esperada entre o que o harness resolve e o que exige o modelo.

Foi exatamente essa fronteira que orientou a rodada de melhorias seguinte. Em vez de empurrar mais lógica para o harness, o projeto agiu em dois pontos: enriqueceu o grafo da Camada B (canonicalização de SKUs e ligação de fornecedores, que zerou os fornecedores órfãos e levou a *Relation Validity@5* a 0,929) e reforçou o classificador com exemplos *few-shot* de decomposição multi-domínio — instruindo o modelo a quebrar a pergunta em conceitos e unir os domínios envolvidos (faturamento ⇒ `financas` + `vendas`, folha de pagamento ⇒ `rh` + `financas`). Sobre o golden canonicalizado (63 casos), o *Exact-Match Routing* subiu de 88,9% para **93,7% (59/63)**, resultado estável em quatro execuções consecutivas (Projeto AI-Orchestrator — `PLANO_SEMIOSE.md`, *Rodada de melhorias*). A leitura é coerente com o princípio: o harness garante o que é determinístico (entidades, relações do grafo), e o ganho marginal nas perguntas genuinamente ambíguas vem de ensinar o modelo a decompor — não de mais regras.

## Princípios de engenharia que se repetem

Lendo as três camadas juntas, emergem padrões que valem para qualquer pipeline semântico em produção, não só para este projeto:

1. **Reintroduza contexto de forma controlada.** O ganho da Camada A e do Contextual Retrieval (Anthropic, 2024) vem da mesma fonte: devolver ao texto as pistas que a vetorização isolada apagou — mas usando metadata estruturada, não texto cru.
2. **Use a estrutura quando o texto não basta.** Perguntas relacionais e globais pedem grafo, não só vetor (Edge et al., 2024). Adote a parte da técnica que resolve seu problema; não copie a complexidade inteira sem necessidade.
3. **Separe recuperar de reordenar.** Bi-encoder para velocidade, re-ranker para relevância contextual (Reimers & Gurevych, 2019; Gao et al., 2023).
4. **Harness antes de Model.** Regex e boost aditivo custam microssegundos e são auditáveis; o LLM entra apenas onde o determinismo falha.
5. **Degrade graceful e meça honestamente.** Cada camada é opt-in, cai para o comportamento anterior se faltar infraestrutura, e é avaliada por métricas que dizem o que de fato calculam.

## Resumo

| Componente | Decisão |
|------------|---------|
| Conceito | Semiose como processo de engenharia de significado (operacional, não filosófico) |
| Camada A | Enriquecimento contextual determinístico da query (pré-recuperação) |
| Camada B | Grafo de conhecimento (Neo4j) como *virtual tool* para relações cross-domínio |
| Camada C | Re-ranking contextual por boost aditivo no semantic router (pós-recuperação) |
| Embeddings | SBERT bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 384d) |
| Segurança | Enricher só usa metadata estruturada; nunca texto cru do histórico |
| Resiliência | Todas as camadas opt-in e graceful (regex-only, Neo4j offline, rerank off) |
| Avaliação | Framework por camada em `eval_semiose.py`; métricas com nomes honestos |

A Semiose, neste recorte de engenharia, não é uma teoria — é um conjunto de três alavancas práticas (enriquecer, relacionar, reordenar) que devolvem ao pipeline o contexto que a geometria dos embeddings, sozinha, descarta. É o que separa um sistema que "encontra textos parecidos" de um sistema que "entende o que o usuário quis dizer, aqui e agora".

---

## Referências

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401. — Memória paramétrica + não-paramétrica, índice denso, redução de alucinação (pp. 1–2). PDF em `ebook/rag_lewis_2020.pdf`.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084. — Custo do cross-encoder vs. bi-encoder, redes siamesas, busca semântica (pp. 1–3). PDF em `ebook/sbert_reimers_gurevych_2019.pdf`.
- Edge, D. et al. (2024). *From Local to Global: A GraphRAG Approach to Query-Focused Summarization*. arXiv:2404.16130. — Grafo de entidades, comunidades (Leiden), sensemaking global (pp. 1–4). PDF em `ebook/graphrag_edge_2024.pdf`.
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. — Taxonomia Naive/Advanced/Modular; pré e pós-recuperação; query rewriting/expansion; rerank (pp. 1, 3, 4, 8). PDF em `ebook/rag_survey_gao_2023.pdf`.
- Anthropic (2024). *Introducing Contextual Retrieval*. Publicado em 19 set. 2024. https://www.anthropic.com/news/contextual-retrieval — Perda de contexto no RAG ingênuo; Contextual Embeddings + Contextual BM25; ganhos de recuperação (49%/67%).
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media. — Embeddings, busca semântica e representação vetorial de significado. PDF em `ebook/Hands-On LLM.pdf`.
- Arsanjani, A. & Bustos, J.P. (2026). *Agentic Architectural Patterns for Building Multi-Agent Systems*. Packt Publishing. — Roteamento semântico e padrões de coordenação multi-agente.
- Projeto AI-Orchestrator — Allan Eric Jepsen. `gateway/query_enricher.py` (Camada A), `gateway/knowledge_graph.py` (Camada B), `gateway/semantic_router.py` (Camada C), `gateway/graph.py` (nó `enrich`), `evals/eval_semiose.py` (avaliação), e `PLANO_SEMIOSE.md` (diagnóstico, arquitetura, desvios e resultados). https://github.com/aejepsen/AI-Orchestrator
- Capítulo 19 (Arquitetura Multi-Agente) e Capítulo 17 (Graph RAG com Neo4j) deste livro — base de roteamento semântico e de grafos sobre a qual a Semiose é construída.
