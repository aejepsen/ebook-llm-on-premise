# Capítulo 13 — Pipeline RAG Completo

## O que é RAG e por que usar com LLM local

RAG — Retrieval-Augmented Generation — é a técnica que transforma um LLM genérico em um assistente que responde com base nos **seus** documentos. Em vez de depender exclusivamente do conhecimento congelado no momento do treinamento, o modelo recebe, junto com a pergunta do usuário, trechos relevantes recuperados de uma base de dados própria. O resultado: respostas fundamentadas, atualizadas e auditáveis.

Por que isso importa especialmente para LLMs on-premise? Três razões:

1. **Privacidade.** Seus documentos nunca saem do seu servidor. Não há chamada de API para nuvem, não há risco de exposição de dados sensíveis.
2. **Custo.** Modelos locais via Ollama não cobram por token. O custo é fixo: hardware + energia.
3. **Controle.** Você decide qual modelo, qual versão, qual quantização. Se o modelo degrada, você faz rollback sem depender de terceiros.

O padrão sem RAG é o chamado "closed-book": o modelo responde apenas com o que memorizou durante o treinamento. Com RAG, passamos para "open-book": o modelo consulta uma fonte de dados antes de gerar a resposta. É a diferença entre um aluno fazendo prova de memória e um aluno fazendo prova com consulta — o segundo erra menos, inventa menos, e cita fontes.

## Arquitetura: Embed, Index, Retrieve, Generate

Todo pipeline RAG segue quatro etapas fundamentais:

```
Documentos → [Embed] → [Index] → Base Vetorial
                                       ↓
Pergunta → [Embed] → [Retrieve] → Contexto → [Generate] → Resposta
```

### 1. Embed (Vetorização)

Cada trecho de documento é convertido em um vetor numérico — uma lista de números que captura o significado semântico do texto. Textos com significados parecidos produzem vetores próximos no espaço multidimensional.

### 2. Index (Indexação)

Os vetores são armazenados em um banco vetorial otimizado para busca por similaridade. Essa etapa acontece uma vez (ou incrementalmente quando novos documentos chegam).

### 3. Retrieve (Recuperação)

Quando o usuário faz uma pergunta, ela também é vetorizada. O banco vetorial encontra os k vetores mais próximos — ou seja, os trechos de documento mais semanticamente relevantes.

### 4. Generate (Geração)

Os trechos recuperados são inseridos no prompt, junto com a pergunta do usuário. O LLM gera a resposta fundamentada nesse contexto.

## Embeddings: o que são e modelos populares

Um embedding é uma representação numérica densa de um texto. Diferente de abordagens como bag-of-words ou TF-IDF (que são esparsas e baseadas em contagem de palavras), embeddings capturam relações semânticas. "Cachorro" e "cão" ficam próximos no espaço vetorial, mesmo sendo palavras diferentes.

### Dimensionalidade

Cada modelo de embedding produz vetores de tamanho fixo. Modelos comuns:

| Modelo | Dimensão | Tamanho | Observação |
|--------|----------|---------|------------|
| `nomic-embed-text` | 768 | ~274 MB | Excelente custo-benefício para on-premise. Usado no AI-Orchestrator. |
| `all-MiniLM-L6-v2` | 384 | ~80 MB | Leve, rápido, bom para prototipação. |
| `bge-large-en-v1.5` | 1024 | ~1.3 GB | Alta qualidade, mais pesado. |
| `mxbai-embed-large` | 1024 | ~670 MB | Forte em benchmarks MTEB. |
| `snowflake-arctic-embed` | 1024 | ~1.1 GB | Destaque recente em retrieval. |

### Gerando embeddings com Ollama

```python
# gerando_embeddings.py
# Gera embeddings locais usando Ollama — sem chamada de API externa

import httpx

def gerar_embedding(texto: str, modelo: str = "nomic-embed-text") -> list[float]:
    """Converte texto em vetor numérico via Ollama."""
    resposta = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": modelo, "input": texto},
    )
    resposta.raise_for_status()
    return resposta.json()["embeddings"][0]

# Exemplo de uso
vetor = gerar_embedding("O que é machine learning?")
print(f"Dimensão do vetor: {len(vetor)}")  # 768 para nomic-embed-text
print(f"Primeiros 5 valores: {vetor[:5]}")
```

O Ollama expõe a API de embeddings no endpoint `/api/embed`. O modelo precisa estar baixado (`ollama pull nomic-embed-text`). A operação é rápida: um texto curto leva poucos milissegundos em GPU.

## Bancos vetoriais: Qdrant, ChromaDB, FAISS

Um banco vetorial é um banco de dados otimizado para armazenar vetores e buscar os mais similares a um vetor de consulta. Diferente de um banco relacional (que busca por igualdade ou range), o banco vetorial busca por **proximidade geométrica**.

### Comparativo

| Critério | Qdrant | ChromaDB | FAISS |
|----------|--------|----------|-------|
| **Tipo** | Servidor standalone | Embarcado / servidor | Biblioteca |
| **Linguagem** | Rust | Python | C++ (bindings Python) |
| **Persistência** | Sim, nativa | Sim (SQLite) | Manual (salvar/carregar) |
| **Filtros por metadado** | Sim, avançados | Sim, básicos | Não nativo |
| **API** | REST + gRPC | Python SDK | Python SDK |
| **Escalabilidade** | Clustering nativo | Limitada | Limitada |
| **Ideal para** | Produção on-premise | Prototipação rápida | Pesquisa, benchmarks |
| **Container Docker** | Sim, leve (~50 MB) | Sim | Não aplicável |

### Recomendação para on-premise

**Qdrant** é a escolha para produção. Razões:

- Container Docker leve com volume persistente.
- Filtros por payload permitem buscas híbridas (semântica + metadado).
- API REST simples — no AI-Orchestrator, usamos `httpx` direto, sem SDK pesado.
- Healthcheck via TCP (a imagem não inclui `curl`).
- Performance estável sob carga.

**ChromaDB** serve bem para notebooks e prototipação. **FAISS** é excelente para benchmarks e pesquisa acadêmica, mas não oferece as facilidades de um servidor em produção.

## Busca por similaridade: cosine, dot product, euclidean

Quando buscamos os vetores mais próximos, precisamos definir **como medir proximidade**. As três métricas mais usadas:

### Similaridade por cosseno (Cosine)

Mede o ângulo entre dois vetores, ignorando magnitude. Resultado entre -1 e 1 (1 = idênticos, 0 = ortogonais).

```python
# similaridade_cosseno.py
# Implementação manual para entender a matemática

import numpy as np

def similaridade_cosseno(vetor_a: list[float], vetor_b: list[float]) -> float:
    """Calcula a similaridade por cosseno entre dois vetores."""
    a = np.array(vetor_a)
    b = np.array(vetor_b)
    # cos(θ) = (A · B) / (||A|| * ||B||)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Vetores idênticos → similaridade = 1.0
v1 = [1.0, 2.0, 3.0]
v2 = [1.0, 2.0, 3.0]
print(similaridade_cosseno(v1, v2))  # 1.0

# Vetores ortogonais → similaridade = 0.0
v3 = [1.0, 0.0]
v4 = [0.0, 1.0]
print(similaridade_cosseno(v3, v4))  # 0.0
```

### Produto escalar (Dot Product)

Multiplica elementos correspondentes e soma. Mais rápido que cosseno (sem normalização), mas sensível à magnitude dos vetores. Funciona bem quando os vetores já estão normalizados.

### Distância euclidiana (L2)

Mede a distância geométrica direta entre dois pontos no espaço. Quanto **menor**, mais similares. Intuitiva, mas sensível à escala.

### Qual escolher?

Para embeddings de texto, **cosseno** é o padrão da indústria. Modelos de embedding são treinados para que textos semanticamente próximos tenham alta similaridade por cosseno. No AI-Orchestrator, o Qdrant está configurado com `"distance": "Cosine"`.

## Pipeline completo em Python com Qdrant + Ollama

Vamos construir um pipeline RAG funcional, do zero, usando apenas Qdrant e Ollama — tudo local.

### Pré-requisitos

```bash
# Subir Qdrant em container
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

# Baixar modelos no Ollama
ollama pull nomic-embed-text   # embeddings
ollama pull qwen2.5:7b         # geração
```

### Passo 1 — Carregar e dividir documentos

```python
# passo1_carregar_documentos.py
# Carrega arquivos de texto e divide em chunks

from pathlib import Path

def carregar_documentos(diretorio: str) -> list[dict]:
    """Carrega todos os .txt de um diretório."""
    documentos = []
    for arquivo in Path(diretorio).glob("*.txt"):
        texto = arquivo.read_text(encoding="utf-8")
        documentos.append({
            "nome": arquivo.name,
            "conteudo": texto,
        })
    return documentos

def dividir_em_chunks(texto: str, tamanho: int = 500, overlap: int = 50) -> list[str]:
    """Divide texto em chunks com overlap."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunk = texto[inicio:fim]
        if chunk.strip():  # ignora chunks vazios
            chunks.append(chunk)
        inicio = fim - overlap  # overlap para manter contexto
    return chunks
```

### Passo 2 — Gerar embeddings e indexar no Qdrant

```python
# passo2_indexar.py
# Vetoriza chunks e armazena no Qdrant

import hashlib
import httpx

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLECAO = "documentos_rag"
MODELO_EMBED = "nomic-embed-text"
DIMENSAO = 768

def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    """Gera embeddings em batch via Ollama."""
    resposta = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODELO_EMBED, "input": textos},
        timeout=120.0,
    )
    resposta.raise_for_status()
    return resposta.json()["embeddings"]

def criar_colecao():
    """Cria a coleção no Qdrant se não existir."""
    # Verifica se já existe
    check = httpx.get(f"{QDRANT_URL}/collections/{COLECAO}")
    if check.status_code == 200:
        print(f"Coleção '{COLECAO}' já existe.")
        return

    resposta = httpx.put(
        f"{QDRANT_URL}/collections/{COLECAO}",
        json={"vectors": {"size": DIMENSAO, "distance": "Cosine"}},
    )
    resposta.raise_for_status()
    print(f"Coleção '{COLECAO}' criada.")

def gerar_id(texto: str) -> str:
    """ID determinístico baseado em hash — upsert idempotente."""
    digest = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    # Qdrant aceita UUID ou inteiro; usamos os primeiros 32 hex como UUID
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

def indexar_chunks(chunks: list[str], metadados: list[dict]):
    """Vetoriza e indexa chunks no Qdrant."""
    criar_colecao()
    vetores = gerar_embeddings(chunks)

    pontos = [
        {
            "id": gerar_id(chunk),
            "vector": vetor,
            "payload": {
                "texto": chunk,
                **meta,
            },
        }
        for chunk, vetor, meta in zip(chunks, vetores, metadados)
    ]

    # Upsert em batch
    resposta = httpx.put(
        f"{QDRANT_URL}/collections/{COLECAO}/points?wait=true",
        json={"points": pontos},
        timeout=60.0,
    )
    resposta.raise_for_status()
    print(f"{len(pontos)} chunks indexados.")
```

### Passo 3 — Recuperar e gerar resposta

```python
# passo3_rag_query.py
# Pipeline RAG completo: busca + geração

import httpx

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLECAO = "documentos_rag"
MODELO_EMBED = "nomic-embed-text"
MODELO_CHAT = "qwen2.5:7b"
TOP_K = 5  # quantidade de chunks a recuperar

def buscar_similares(pergunta: str, top_k: int = TOP_K) -> list[dict]:
    """Busca os chunks mais relevantes para a pergunta."""
    # 1. Vetoriza a pergunta
    resp_embed = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODELO_EMBED, "input": pergunta},
        timeout=30.0,
    )
    resp_embed.raise_for_status()
    vetor_pergunta = resp_embed.json()["embeddings"][0]

    # 2. Busca no Qdrant
    resp_busca = httpx.post(
        f"{QDRANT_URL}/collections/{COLECAO}/points/search",
        json={
            "vector": vetor_pergunta,
            "limit": top_k,
            "with_payload": True,
        },
    )
    resp_busca.raise_for_status()
    return resp_busca.json()["result"]

def gerar_resposta(pergunta: str, contextos: list[str]) -> str:
    """Gera resposta fundamentada nos contextos recuperados."""
    # Monta o prompt com os contextos
    contexto_formatado = "\n---\n".join(contextos)

    prompt_sistema = """Você é um assistente que responde perguntas com base
exclusivamente nos documentos fornecidos. Se a informação não estiver nos
documentos, diga que não encontrou a resposta. Nunca invente dados."""

    prompt_usuario = f"""Documentos relevantes:
{contexto_formatado}

Pergunta: {pergunta}

Responda de forma clara e direta, citando os documentos quando possível."""

    resposta = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODELO_CHAT,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            "stream": False,
        },
        timeout=120.0,
    )
    resposta.raise_for_status()
    return resposta.json()["message"]["content"]

def pipeline_rag(pergunta: str) -> str:
    """Pipeline RAG completo: busca → contexto → geração."""
    # Recupera chunks relevantes
    resultados = buscar_similares(pergunta)

    # Extrai textos dos chunks
    contextos = [r["payload"]["texto"] for r in resultados]

    # Exibe scores para debug
    for i, r in enumerate(resultados):
        print(f"  Chunk {i+1}: score={r['score']:.4f}")

    # Gera resposta
    return gerar_resposta(pergunta, contextos)

# Uso
if __name__ == "__main__":
    pergunta = "Quais são os benefícios do modelo on-premise?"
    print(f"\nPergunta: {pergunta}\n")
    resposta = pipeline_rag(pergunta)
    print(f"\nResposta:\n{resposta}")
```

## Métricas de retrieval: Recall@k, MRR, nDCG

Construir o pipeline é metade do trabalho. A outra metade é **medir se ele funciona**. Métricas de retrieval avaliam a qualidade da etapa de recuperação — se os chunks certos estão sendo encontrados.

### Recall@k

"Dos documentos relevantes que existem, quantos apareceram nos top-k resultados?"

```
Recall@k = |documentos relevantes nos top-k| / |total de documentos relevantes|
```

Se existem 3 documentos relevantes e o sistema retornou 2 deles nos top-5, o Recall@5 é 2/3 = 0.667.

### MRR (Mean Reciprocal Rank)

"Qual a posição do primeiro resultado relevante?"

```
RR = 1 / posição_do_primeiro_relevante
MRR = média do RR sobre todas as queries
```

Se o primeiro resultado relevante está na posição 3, o RR é 1/3. MRR alto significa que o sistema coloca resultados relevantes no topo.

### nDCG (normalized Discounted Cumulative Gain)

Considera não apenas se um resultado é relevante, mas **quão relevante** e **em que posição** ele aparece. Resultados mais relevantes em posições mais altas recebem mais peso.

```python
# metricas_retrieval.py
# Implementação das três métricas fundamentais

import numpy as np

def recall_at_k(relevantes: set, recuperados: list, k: int) -> float:
    """Recall@k: proporção de relevantes nos top-k."""
    top_k = set(recuperados[:k])
    encontrados = relevantes & top_k
    return len(encontrados) / len(relevantes) if relevantes else 0.0

def mrr(queries: list[dict]) -> float:
    """Mean Reciprocal Rank sobre múltiplas queries.

    Cada query: {"recuperados": [...], "relevantes": set(...)}
    """
    rr_soma = 0.0
    for q in queries:
        for i, doc in enumerate(q["recuperados"], start=1):
            if doc in q["relevantes"]:
                rr_soma += 1.0 / i
                break
    return rr_soma / len(queries) if queries else 0.0

def dcg_at_k(scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    scores_k = scores[:k]
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores_k))

def ndcg_at_k(relevancia_real: list[float], relevancia_recuperada: list[float], k: int) -> float:
    """nDCG@k: DCG normalizado pelo DCG ideal."""
    dcg = dcg_at_k(relevancia_recuperada, k)
    ideal = dcg_at_k(sorted(relevancia_real, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

# Exemplo de uso
relevantes = {"doc1", "doc3", "doc5"}
recuperados = ["doc2", "doc1", "doc5", "doc4", "doc3"]

print(f"Recall@3: {recall_at_k(relevantes, recuperados, 3):.3f}")  # 2/3 = 0.667
print(f"Recall@5: {recall_at_k(relevantes, recuperados, 5):.3f}")  # 3/3 = 1.000

queries = [
    {"recuperados": ["doc2", "doc1", "doc3"], "relevantes": {"doc1", "doc3"}},  # RR = 1/2
    {"recuperados": ["doc1", "doc2", "doc3"], "relevantes": {"doc1"}},          # RR = 1/1
]
print(f"MRR: {mrr(queries):.3f}")  # (0.5 + 1.0) / 2 = 0.750
```

### Construindo um evaluation set

Para medir essas métricas, você precisa de um **evaluation set** — um conjunto de pares (pergunta, documentos_relevantes) anotados manualmente. Recomendações:

1. **Mínimo 50 pares** para métricas estatisticamente significativas.
2. **Diversidade**: cubra diferentes tipos de pergunta (factual, comparativa, aberta).
3. **Anotação por domínio**: quem conhece os documentos anota melhor.
4. **Versionamento**: o eval set evolui junto com os documentos.

---

## Resumo

| Conceito | Decisão |
|----------|---------|
| Embedding | `nomic-embed-text` (768d) via Ollama |
| Banco vetorial | Qdrant em container Docker |
| Métrica de distância | Cosine |
| Top-k | 5 (ajustar por recall) |
| Métricas de avaliação | Recall@k, MRR, nDCG |

RAG é o padrão de facto para dar contexto a LLMs. Com Ollama + Qdrant, todo o pipeline roda on-premise, sem dependência de serviços externos, sem custo por token, e com controle total sobre os dados.

---

## Referências

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
- Qdrant Documentation. https://qdrant.tech/documentation/
- Ollama API Reference. https://github.com/ollama/ollama/blob/main/docs/api.md
- Nomic AI. *nomic-embed-text: A Reproducible Long Context Text Embedder*. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Projeto AI-Orchestrator — `gateway/semantic_router.py` (uso de Qdrant + nomic-embed-text para roteamento semântico).
- *RAG with Python Cookbook*, Capítulos 1, 5 e 6 — introdução a RAG, embeddings e bancos vetoriais.
