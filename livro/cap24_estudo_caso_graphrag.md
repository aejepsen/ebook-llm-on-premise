# Capítulo 24 — Estudo de caso: GraphRAG on-premise em produção (hit@5 0.716 → 0.982)

O Capítulo 17 apresentou a teoria do Graph RAG: grafos de conhecimento, Cypher,
busca híbrida. Este capítulo é o contraponto prático — a construção completa,
medida e auditada de um sistema GraphRAG 100% on-premise sobre um corpus real de
**74 repositórios de código (258.707 chunks)**, numa única GPU de consumo
(RTX 3060, 12 GB), com custo de tokens **zero** em runtime.

Tudo aqui tem número rastreável. Onde houve escolha, houve A/B. Onde houve A/B,
o resultado contrariou a intuição duas vezes — e é por isso que este capítulo
existe.

> Código completo: [github.com/aejepsen/graphrag-onprem-toolkit](https://github.com/aejepsen/graphrag-onprem-toolkit)

## 24.1 O problema e o corpus

Objetivo: uma base de conhecimento consultável por LLMs cobrindo o estado da
arte de treino e serving de modelos — os repositórios do karpathy (nanoGPT,
llm.c, nanochat...), todo o ecossistema deepseek-ai (V3, R1, FlashMLA, DeepEP,
3FS...), gateways (litellm, kong, new-api) e frameworks spec-driven (spec-kit,
OpenSpec, BMAD).

O desafio de retrieval nesse corpus é assimétrico:

- **litellm tem 111.098 chunks; deepseek-ai/profile-data tem 6.** Busca por
  similaridade pura deixa o repo gigante afogar os pequenos.
- Queries são **conceito→código** ("fp8 GEMM kernel", "GRPO"): o usuário
  descreve uma técnica, a resposta é uma implementação.
- Siglas e nomes próprios ("GRPO", "EPLB") são **matches literais** que
  embeddings tratam mal.

Cada uma dessas assimetrias vai exigir um estágio próprio no pipeline — e cada
estágio será validado por eval antes de entrar.

## 24.2 Camada 1 — o grafo (AST primeiro, LLM depois)

A regra de ouro de custo em GraphRAG: **extração estática constrói a estrutura
de graça; o LLM só rotula**.

1. **Extração AST** (ferramenta: graphify): varre o código e emite nós
   (arquivos, classes, funções, chunks de documentação) e **arestas tipadas e
   direcionadas** — `contains` (146k), `calls` (127k), `references` (55k),
   `imports` (37k), `inherits` (5k), além de `rationale_for` e `cites` ligando
   documentação a código. Resultado: **271.779 nós, 515.762 arestas**, sendo
   84% EXTRACTED (determinístico) e 16% INFERRED (heurístico, confiança média
   0,71). Custo de LLM: zero.
2. **Rotulagem em lote com modelo barato**: um passe de LLM econômico nomeia
   271k nós e 10.923 comunidades (detecção Leiden) — "FP8 Quantized GEMM",
   "Expert Parallelism Load Balancer". Custo total: ~600k tokens de entrada,
   ~400k de saída. Uma vez, não por query.

**Armadilha operacional**: re-rodar a extração re-clusteriza e **zera os
rótulos**. Todo update de corpus exige re-rotular — automatize a cadeia inteira
num script (`rebuild_all.sh`) ou alguém vai esquecer um passo e dessincronizar
as camadas silenciosamente.

## 24.3 Auditando um grafo de 271k nós

"O grafo parece bom" não é engenharia. A auditoria estrutural completa custa
uma consulta SQL de segundos — não há desculpa para pular:

```sql
-- auditoria_grafo.sql — checks estruturais sobre o índice SQLite do grafo

-- 1. Arestas dangling (apontam para nó que não existe)
SELECT COUNT(*) FROM edges e
LEFT JOIN nodes n ON e.target = n.id WHERE n.id IS NULL;

-- 2. Arestas duplicadas
SELECT COUNT(*) FROM (
  SELECT source, target, relation, COUNT(*) c
  FROM edges GROUP BY 1,2,3 HAVING c > 1);

-- 3. Nós órfãos (sem nenhuma aresta)
SELECT COUNT(*) FROM nodes n WHERE NOT EXISTS
  (SELECT 1 FROM edges e WHERE e.source = n.id OR e.target = n.id);

-- 4. Amostra reprodutível p/ verificação manual (semente = módulo fixo)
SELECT n1.label, e.relation, n2.label, n1.source_file
FROM edges e
JOIN nodes n1 ON e.source = n1.id
JOIN nodes n2 ON e.target = n2.id
WHERE e.rowid % 51629 = 7 LIMIT 10;
```

Resultados no corpus deste capítulo:

| Check | Resultado |
|---|---|
| Arestas dangling | 0 de 515.762 |
| Arestas duplicadas | 0 |
| Nós órfãos | 0,3% |
| Nós sem rótulo real | 0,06% |
| Amostra manual (semente fixa, 10 tipos) | 10/10 corretas |

A amostra manual pegou arestas como `IBSocket.cc -[imports]-> IBSocket.h` e
`CredentialsPanel() -[calls]-> credentialDeleteCall()` — verificáveis no fonte
em segundos. Semente fixa importa: amostra irreprodutível é auditoria
irrefutável, no pior sentido.

O que a auditoria estrutural **não** cobre: a precisão semântica das 82k
arestas INFERRED. Isso exige amostragem estatística por tipo (100 arestas de
`calls` verificadas por grep → precisão com intervalo de confiança) —
registrado como dívida conhecida, não varrido para debaixo do tapete.

## 24.4 Índices derivados: uma fonte, três projeções

O grafo (JSON de 301 MB) é a fonte de verdade, mas ninguém consulta um JSON de
301 MB por query. Três projeções, cada uma otimizada para um tipo de pergunta:

**1. SQLite** — nós + arestas com índices nas duas pontas:

```sql
CREATE TABLE nodes(id TEXT PRIMARY KEY, label TEXT, community_name TEXT,
                   source_file TEXT, source_location TEXT, repo TEXT,
                   pagerank REAL);
CREATE TABLE edges(source TEXT, target TEXT, relation TEXT, weight REAL);
CREATE INDEX ix_edges_src ON edges(source);
CREATE INDEX ix_edges_dst ON edges(target);
```

Consultas estruturais caem de 14 s (parse do JSON) para **55 ms — 255×**.

**2. FTS5/BM25** sobre exatamente os mesmos chunks do índice vetorial (build:
13 segundos para 258k chunks):

```python
# build_fts.py (essência) — metade léxica da busca híbrida
out.execute("CREATE VIRTUAL TABLE chunks USING "
            "fts5(node_id UNINDEXED, repo UNINDEXED, text, "
            "tokenize='porter unicode61')")
# text = mesmo cabeçalho+código que foi embedado — paridade entre as vias
```

**3. PageRank** como coluna do banco (networkx, 3 s para o grafo inteiro):

```python
# build_centrality.py (essência)
g = nx.DiGraph()
g.add_weighted_edges_from(con.execute(
    "SELECT source, target, weight FROM edges WHERE source != target"))
pr = nx.pagerank(g, alpha=0.85, weight="weight")
con.execute("ALTER TABLE nodes ADD COLUMN pagerank REAL")
con.executemany("UPDATE nodes SET pagerank=? WHERE id=?",
                ((v, k) for k, v in pr.items()))
```

Sanidade do resultado: o nó de maior PageRank do corpus é `UserAPIKeyAuth` do
litellm — o tipo do qual o maior repositório inteiro depende. Quando a métrica
de centralidade aponta algo que um engenheiro reconheceria como "o coração do
sistema", ela está medindo a coisa certa.

## 24.5 Embeddings: o A/B que definiu o projeto

Chunk embedado = cabeçalho estruturado + ±25 linhas de código real:

```
[deepseek-ai__DeepSeek-V3] FP8 Quantized GEMM :: fp8_gemm() (inference/kernel.py:L126)
<código real da função>
```

O cabeçalho vem do grafo (repo, comunidade nomeada, símbolo) — é GraphRAG já na
indexação, não só na consulta. Três embedders, mesmo corpus, mesmo golden set:

| Embedder | hit@5 | MRR@5 | Onde roda |
|---|---|---|---|
| MiniLM-L12 (384d) | 0.716 | 0.608 | local, CPU |
| **Qwen3-Embedding-0.6B (1024d)** | **0.899** | **0.838** | local, GPU 1,2 GB |
| voyage-code-3 (API paga, especialista em código) | 0.872 | 0.817 | nuvem |

Duas lições que valem o capítulo:

1. **O embedder é a alavanca dominante.** +18,3 pontos de hit@5 — 69% de todo
   o ganho do projeto veio desta troca. Antes de construir pipeline sofisticado
   em cima de embedding fraco, troque o embedding.
2. **O modelo local de 0,6B venceu a API especialista.** A hipótese "API
   code-specialist > local pequeno" foi testada e **refutada neste corpus**
   (0.899 vs 0.872). Provável causa: o cabeçalho estruturado do chunk favorece
   o modo instruct assimétrico do Qwen3. Não generalize a conclusão —
   generalize o método: *teste com seu golden antes de assinar contrato com
   API*.

A assimetria instruct merece código, porque errar não lança exceção — só
degrada o ranking silenciosamente:

```python
# Embedders instruct (Qwen3, Voyage): QUERY recebe prompt, DOCUMENTO não.
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
qkw = ({"prompt_name": "query"}
       if "query" in (getattr(model, "prompts", None) or {}) else {})
v_query = model.encode(query, normalize_embeddings=True, **qkw)   # com prompt
v_doc   = model.encode(chunk, normalize_embeddings=True)          # sem prompt
```

### Guards de operação (onde RAG morre em produção)

Duas classes de falha silenciosa, dois guards executáveis:

**Mismatch de embedder** — buscar com modelo errado numa coleção de mesma
dimensão "funciona" e retorna lixo. Solução: manifesto coleção→modelo, e o
consumidor resolve o modelo pela coleção:

```json
// collections.json — fonte de verdade executável
{"llm_wiki_code": {"model": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024,
                   "fingerprint": "b73c2542f02c...", "points": 258707}}
```

```python
def resolve_model(coll, env_model):
    entry = load_manifest().get(coll)
    if not entry:
        return env_model or DEFAULT_MODEL
    if env_model and env_model != entry["model"]:
        sys.exit(f"ERRO: '{coll}' foi embedada com '{entry['model']}' "
                 f"({entry['dim']}d), mas EMBED_MODEL={env_model}.")
    return entry["model"]
```

**Desincronização grafo↔vetores** — o embed é resumível por checkpoint de
offset; se o grafo mudou (update do corpus), o offset aponta para dados
diferentes e o checkpoint mente "tudo pronto". O guard: fingerprint do dataset
(sha256 dos node_ids na ordem de embed) gravado no manifesto; divergência
bloqueia o resume e exige rebuild completo. Dessincronização silenciosa vira
erro barulhento — a única transformação que guards devem fazer.

## 24.6 O funil de retrieval: cinco estágios, cada um com seu número

```
query ─► vetorial (top-25..50) ─┐
                                 ├─► fusão RRF ─► rerank cross-encoder ─► cap por repo ─► top-k
       BM25/FTS5 (top-25..50) ──┘                (Qwen3-Reranker-0.6B)                    │
                                                                                          ▼
                                                              + vizinhos 1-hop do grafo (PageRank)
```

| Estágio adicionado | hit@5 | MRR@5 | O que corrige |
|---|---|---|---|
| Qwen3 puro | 0.899 | 0.838 | — (base) |
| + cap por repo (máx 2 chunks/repo) | 0.927 | 0.849 | litellm afogando repos de 300 chunks |
| + repo-cards | 0.945 | 0.869 | repo com pouco código indexável |
| + híbrido BM25 (fusão RRF) | 0.963 | 0.878 | siglas/termos literais ("GRPO") |
| + rerank n=25 | 0.972 | **0.901** | ordem no topo |
| **+ rerank n=50 (produção)** | **0.982** | 0.896 | recall no pool do reranker |

### Fusão RRF: juntar rankings sem calibrar escalas

Cosseno (0–1) e BM25 (negativo, ilimitado) são incomensuráveis. RRF ignora os
scores e funde **posições**:

```python
def rrf_fuse(rankings, c=60):
    """Reciprocal Rank Fusion: score = soma de 1/(c + posição)."""
    scores = {}
    for ranking in rankings:
        for i, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + i + 1)
    return sorted(scores, key=lambda n: -scores[n])
```

Documento no top de UMA via já entra bem posicionado; documento razoável nas
DUAS vias sobe. É o comportamento desejado: BM25 resgata "GRPO" literal que o
embedding enterra, sem precisar acertar peso de combinação.

### Repo-cards: dar voz a quem tem pouco código

Um chunk sintético por repositório — nome + top-15 comunidades do grafo +
início do README — indexado no vetorial E no FTS. Resolve queries conceito→repo
quando o conceito quase não existe como código ("GRPO" citado no README do
DeepSeek-Math; um repo de profiling com 6 nós). 73 cards, +1,8 ponto de hit@5,
custo de build ~1 minuto. Ids determinísticos (`card::<repo>`) tornam o upsert
idempotente.

### Rerank local: cross-encoder via CausalLM

Bi-encoder compara vetores pré-computados — rápido em 258k documentos, míope no
fino. Cross-encoder lê query e documento **juntos** — preciso, e caro demais
para rodar em tudo. Por isso funil: estágios baratos acham os ~50 prováveis, o
caro decide a ordem. O Qwen3-Reranker-0.6B pontua via probabilidade do token
"yes":

```python
# local_rerank (essência) — score = P("yes") no último token
prompt = (f"{prefix}<Instruct>: Given a code search query, judge whether "
          f"the document answers it.\n<Query>: {query}\n"
          f"<Document>: {doc}{suffix}")
logits = model(**tokenizer(batch, ...)).logits[:, -1, :]
par = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
score = par.log_softmax(1)[:, 1].exp()          # P(relevante)
```

fp16 na GPU: 1,2 GB de VRAM. A escolha do pool (n=25 vs n=50) foi medida:
n=50 troca 0,5 ponto de MRR por 1,0 de hit@5 — para RAG, recall no top-k
importa mais que a posição exata. E a decisão de reranker **local** em vez de
API não foi ideologia: reranker roda a CADA query (não é custo one-time como o
embed), e o A/B do embedder já tinha mostrado a família Qwen3 competitiva.

### K-hop: topologia como contexto adicional

Os top-3 resultados viram sementes; vizinhos 1-hop ranqueados por PageRank são
**anexados** ao resultado — marcados, sem competir com o top-k:

```python
def graph_neighbors(seed_ids, limit=3):
    """Vizinhos 1-hop ranqueados por PageRank (vizinho central > folha)."""
    for nid in seed_ids:
        rows = con.execute(
            "SELECT CASE WHEN e.source=:n THEN e.target ELSE e.source END, "
            "       e.relation, COALESCE(n.pagerank, 0) "
            "FROM edges e JOIN nodes n ON n.id = "
            "  CASE WHEN e.source=:n THEN e.target ELSE e.source END "
            "WHERE e.source=:n OR e.target=:n "
            "ORDER BY n.pagerank DESC LIMIT 40", {"n": nid})
        # dedupe entre sementes, top-`limit` global por PageRank
```

Efeito observado: query "paged kv cache" retorna `flash_attn_with_kvcache()` e
o grafo anexa `_sdpa_attention()` (relação `calls:out` — quem ela chama) e o
arquivo-pai (`contains:in`). É o "mini-mapa" de dependências que similaridade
sozinha não enxerga — e que responde a pergunta seguinte do usuário antes de
ela existir ("e quem chama isso?").

Por que PageRank no ranking dos vizinhos: um nó com 40 vizinhos precisa de
critério para escolher 3. Vizinho de alta centralidade é infraestrutura do
repo; vizinho folha é detalhe. Sem esse ranking, k-hop vira ruído no prompt —
e ruído no prompt custa tokens e atenção do modelo.

## 24.7 O golden set: sem ele, tudo acima é opinião

Nenhum número deste capítulo existiria sem o eval: **109 queries com
repositórios esperados por query**, cobrindo os 74 repos:

```jsonl
{"query": "fp8 GEMM matrix multiplication kernel",
 "expected_repos": ["deepseek-ai__DeepGEMM", "deepseek-ai__TileKernels",
                    "deepseek-ai__FlashMLA"]}
{"query": "group relative policy optimization GRPO",
 "expected_repos": ["deepseek-ai__DeepSeek-Math", "deepseek-ai__DeepSeek-R1"]}
```

Métricas: hit@5 (≥1 repo esperado nos top-5 — a métrica de "RAG não alucina"),
MRR@5 (posição do primeiro acerto) e precision@5 (fração dos top-5 esperada).

Regras de curadoria que separam medição de automaquiagem:

1. **Item só entra no gabarito com prova no corpus** (grep confirmando que o
   repo implementa o conceito) — nunca porque o retrieval o retornou. No
   projeto, três correções passaram no grep (RoPE em dois repos extras,
   backward manual em llm.c) e duas foram **recusadas** (conceitos que só
   existiam em README).
2. **Query não se reescreve para facilitar.** Miss cujo conceito só existe em
   README continua miss: é teto do corpus, e o registro honesto disso orientou
   a solução certa (repo-cards) em vez de esconder o problema.
3. **Aritmética da significância**: com n=109, 1 query = 0,9 ponto. Ganhos de
   +2 pontos entre estágios adjacentes são fracos isoladamente; o que valida a
   série é a monotonicidade em 5 estágios e cada miss resolvido ter explicação
   causal verificada. O primeiro golden tinha 25 queries — 1 query = 4 pontos,
   granularidade que transforma ruído em "melhoria". Expandir o golden **foi**
   uma das correções do projeto: o hit@5 caiu de 0,96 (golden 25) para 0,899
   (golden 109) da noite pro dia, e essa queda era informação, não regressão.
4. **Precision@5 caiu de 0.79 para 0.64 no funil de produção — e está ok.**
   Diversidade troca redundância do repo certo por cobertura; e a métrica
   subestima (chunk relevante de repo fora do gabarito conta como erro). Saber
   o que sua métrica NÃO mede é parte da métrica.

Resultado final: **31 misses → 2** (redução de 94%). Os 2 restantes são
vocabulário de processo (BMAD) e um capítulo de cookbook — teto do corpus
atual, documentado como tal.

## 24.8 Servindo: injeção, não fork

O funil é servido pela API do svc-rag (Capítulo 14) **sem alterar uma linha do
serviço**. O svc-rag expõe um ponto de extensão no construtor do estado:

```python
# rag_wiki.py (launcher, ~90 linhas) — injeta o funil sem fork
class WikiStore(QdrantStore):
    def search(self, collection, qvec, top_k):
        if collection != COLL:                    # outras coleções: herdado
            return super().search(collection, qvec, top_k)
        payloads = retrieve(url, collection, qvec, self._embedder.last_query,
                            top_k, per_repo=2, hybrid=True,
                            rerank=True, rerank_n=50, graph_n=3)
        return [Hit(**mapear(p)) for p in payloads]

app = create_app(settings, state=State(settings,
                                       embedder=WikiQueryEmbedder(...),
                                       store=WikiStore(...)))
```

Um detalhe de design que interfaces genéricas impõem: a interface `VectorStore`
transporta o **vetor** da query, mas o BM25 e o reranker precisam do **texto**.
Solução: o embedder guarda a query corrente em thread-local (encode e search
rodam na mesma thread da request). Documente esse tipo de acoplamento — é ele
que morde o próximo mantenedor.

Operação: systemd user unit (boot automático via linger, `Restart=on-failure`,
logs no journald), Qdrant com `--restart unless-stopped`, chave interna real
(requisição com chave errada → 401, **testado**, não presumido). Um serviço que
só existe enquanto o terminal de quem o subiu estiver aberto não é um serviço.

```ini
# ~/.config/systemd/user/rag-wiki.service (essência)
[Service]
WorkingDirectory=/path/llm-wiki
EnvironmentFile=/path/llm-wiki/.env
ExecStart=/path/venv/bin/uvicorn tools.rag_wiki:app --host 127.0.0.1 --port 8299
Restart=on-failure
[Install]
WantedBy=default.target
```

## 24.9 A decisão Neo4j (ou: quando NÃO usar o banco de grafos)

O Capítulo 17 ensina Neo4j; este projeto decidiu **não** usá-lo — e a
justificativa importa mais que a escolha:

- K-hop de 1–2 saltos com arestas indexadas nas duas pontas: **milissegundos
  em SQLite** para 515k arestas.
- PageRank do grafo inteiro: **3 segundos em networkx**, uma vez por rebuild.
- Neo4j adicionaria: um container disputando RAM com a GPU, uma segunda fonte
  de verdade para sincronizar a cada update, e latência de rede — para
  entregar o que já existia.

Critérios objetivos para migrar: consultas exploratórias ad-hoc frequentes
(Cypher/Bloom), grafo ~10× maior, caminhos profundos (4+ hops) em runtime, ou
necessidade da biblioteca GDS completa. Arquitetura recomendada nesse caso:
**SQLite continua no runtime; Neo4j como espelho analítico** — nunca como
dependência do funil.

## 24.10 Reproduza no seu corpus

Receita completa com o toolkit (os nomes de script são os do repositório):

```bash
# 0. Pré-requisitos: Qdrant local, venv com sentence-transformers/torch/networkx
docker run -d --name qdrant-kb --restart unless-stopped \
  -p 127.0.0.1:6334:6333 -v kb_qdrant:/qdrant/storage qdrant/qdrant

# 1. Grafo (AST grátis + rótulos em lote com LLM barato)
cd /seu/corpus && graphify update . && graphify label . --backend <llm-barato>

# 2. Projeções: SQLite → PageRank → FTS5
python tools/build_sqlite_index.py && python tools/enrich_index.py
python tools/build_centrality.py  && python tools/build_fts.py

# 3. Vetores (resumível por checkpoint; manifesto criado automaticamente)
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B EMBED_DIM=1024 EMBED_DTYPE=float16 \
  WIKI_COLLECTION=minha_kb python tools/build_vectors.py

# 4. Repo-cards
WIKI_COLLECTION=minha_kb python tools/build_repo_cards.py

# 5. SEU golden set (sem ele não há passo 6 honesto)
$EDITOR tools/eval_golden.jsonl   # ~100 queries, gabarito com prova no corpus

# 6. Medir — cada flag é um estágio; ligue um por vez e anote o delta
WIKI_COLLECTION=minha_kb python tools/eval_retrieval.py --k 5
WIKI_COLLECTION=minha_kb python tools/eval_retrieval.py --k 5 --per-repo 2
WIKI_COLLECTION=minha_kb python tools/eval_retrieval.py --k 5 --hybrid --per-repo 2
WIKI_COLLECTION=minha_kb python tools/eval_retrieval.py --k 5 --hybrid --per-repo 2 \
  --rerank --rerank-n 50

# 7. Busca de produção
WIKI_COLLECTION=minha_kb python tools/search_vectors.py "sua pergunta" \
  --k 5 --hybrid --rerank --rerank-n 50 --per-repo 2 --graph 3
```

Ordem de medição importa: ligar tudo de uma vez e ver 0.98 não ensina nada;
ligar um estágio por vez produz a tabela da Seção 24.6 — e a capacidade de
remover estágio que não paga seu custo no SEU corpus.

## 24.11 Checklist do capítulo

- [ ] Estrutura do grafo por AST (grátis); LLM barato só para rotular, em lote.
- [ ] Auditoria estrutural com números: dangling, órfãos, duplicatas, amostra manual com semente fixa.
- [ ] Golden set ≥100 queries ANTES de otimizar; gabarito só com prova no corpus.
- [ ] A/B de embedder no SEU corpus — inclusive contra APIs; não herde benchmark alheio.
- [ ] Assimetria instruct (query≠documento) tratada em código.
- [ ] Manifesto coleção→modelo + fingerprint anti-desync.
- [ ] Funil em camadas: barato filtra, caro ordena; cada estágio entra com delta medido.
- [ ] Métricas honestas: registre trade-offs (precision↓), limitações da métrica e misses restantes.
- [ ] Grafo no runtime: k-hop ranqueado por centralidade como contexto ADICIONAL.
- [ ] Serviço permanente (systemd/container), auth real testada, restart policy.
- [ ] Banco de grafos dedicado só com critério objetivo — não por moda.

**Síntese**: 0.716 → 0.982 de hit@5, R$ 0 de tokens em runtime, uma GPU de
consumo — e nenhum número neste capítulo que não possa ser reproduzido com o
toolkit e um golden set.
