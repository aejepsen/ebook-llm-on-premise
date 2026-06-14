# Capítulo 15 — Chunking e Estratégias de Indexação

## Por que chunking importa

Quando alimentamos documentos em um pipeline RAG, raramente enviamos o documento inteiro para o modelo. Há duas razões fundamentais:

1. **Limite de contexto.** Mesmo modelos com janelas de 128k tokens não devem receber documentos inteiros de forma irrestrita. Quanto mais contexto, maior a latência, maior o custo computacional, e — contraintuitivamente — pior a qualidade da resposta. Modelos sofrem do problema "lost in the middle": informações no meio de contextos longos são frequentemente ignoradas.

2. **Precisão de retrieval.** Se indexarmos documentos inteiros, a busca vetorial retorna documentos inteiros. Um documento de 50 páginas sobre "gestão de projetos" será retornado quando o usuário perguntar sobre "metodologia ágil", mesmo que apenas 2 parágrafos sejam relevantes. Chunks menores permitem retrieval cirúrgico.

O chunking é a ponte entre documentos brutos e vetores buscáveis. A qualidade do chunking determina a qualidade do retrieval. Chunk ruim, RAG ruim — não importa quão bom seja o modelo de embedding ou o LLM.

## Estratégias de chunking

### Fixed-size (tamanho fixo)

A abordagem mais simples: dividir o texto a cada N caracteres (ou N tokens).

```python
# chunking_fixo.py
# Divide texto em chunks de tamanho fixo com overlap

def chunking_fixo(texto: str, tamanho: int = 500, overlap: int = 50) -> list[str]:
    """Chunks de tamanho fixo em caracteres.

    Args:
        texto: texto completo do documento
        tamanho: número de caracteres por chunk
        overlap: sobreposição entre chunks consecutivos
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)
        inicio += tamanho - overlap
    return chunks

# Exemplo
texto = "A" * 1200  # texto com 1200 caracteres
chunks = chunking_fixo(texto, tamanho=500, overlap=50)
print(f"Total de chunks: {len(chunks)}")  # 3 chunks
```

**Vantagens:** simples, previsível, fácil de debugar.
**Desvantagens:** corta no meio de frases, parágrafos, e até palavras. Ignora a estrutura do texto.

### Sentence-based (por sentença)

Divide o texto respeitando fronteiras de sentenças. Agrupa sentenças até atingir o tamanho desejado.

```python
# chunking_sentenca.py
# Divide texto respeitando fronteiras de sentenças

import re

def dividir_sentencas(texto: str) -> list[str]:
    """Divide texto em sentenças usando regex simples."""
    # Padrão: ponto/exclamação/interrogação seguido de espaço e maiúscula
    sentencas = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ú])', texto)
    return [s.strip() for s in sentencas if s.strip()]

def chunking_sentenca(texto: str, max_chars: int = 500) -> list[str]:
    """Agrupa sentenças em chunks sem ultrapassar max_chars."""
    sentencas = dividir_sentencas(texto)
    chunks = []
    chunk_atual = []
    tamanho_atual = 0

    for sentenca in sentencas:
        tam_sentenca = len(sentenca)
        # Se adicionar esta sentença excede o limite, fecha o chunk
        if tamanho_atual + tam_sentenca > max_chars and chunk_atual:
            chunks.append(" ".join(chunk_atual))
            chunk_atual = []
            tamanho_atual = 0

        chunk_atual.append(sentenca)
        tamanho_atual += tam_sentenca + 1  # +1 pelo espaço

    if chunk_atual:
        chunks.append(" ".join(chunk_atual))

    return chunks
```

**Vantagens:** respeita fronteiras linguísticas, chunks fazem sentido isoladamente.
**Desvantagens:** tamanho variável, pode gerar chunks muito pequenos (sentenças curtas) ou muito grandes (sentenças longas).

### Recursive (recursivo)

Estratégia popularizada pelo LangChain: tenta dividir por parágrafos (`\n\n`); se o chunk ainda for grande, divide por sentenças (`.`); se ainda for grande, divide por palavras (` `). É uma cascata de separadores, do mais natural ao mais forçado.

```python
# chunking_recursivo.py
# Divide texto recursivamente por hierarquia de separadores

def chunking_recursivo(
    texto: str,
    max_chars: int = 500,
    overlap: int = 50,
    separadores: list[str] | None = None,
) -> list[str]:
    """Chunking recursivo: tenta separadores na ordem, do mais natural ao mais forçado."""
    if separadores is None:
        separadores = ["\n\n", "\n", ". ", " "]

    # Caso base: texto cabe em um chunk
    if len(texto) <= max_chars:
        return [texto.strip()] if texto.strip() else []

    # Tenta cada separador na ordem
    for sep in separadores:
        partes = texto.split(sep)
        if len(partes) == 1:
            continue  # separador não encontrado, tenta o próximo

        chunks = []
        chunk_atual = ""

        for parte in partes:
            candidato = chunk_atual + sep + parte if chunk_atual else parte

            if len(candidato) <= max_chars:
                chunk_atual = candidato
            else:
                if chunk_atual.strip():
                    chunks.append(chunk_atual.strip())
                # Se a parte individual é maior que max_chars, recursa
                if len(parte) > max_chars:
                    sub_chunks = chunking_recursivo(
                        parte, max_chars, overlap,
                        separadores[separadores.index(sep) + 1:]
                    )
                    chunks.extend(sub_chunks)
                    chunk_atual = ""
                else:
                    chunk_atual = parte

        if chunk_atual.strip():
            chunks.append(chunk_atual.strip())

        if chunks:
            return chunks

    # Fallback: divisão por caractere (último recurso)
    return chunking_fixo_simples(texto, max_chars)

def chunking_fixo_simples(texto: str, tamanho: int) -> list[str]:
    """Fallback: divisão bruta por caractere."""
    return [texto[i:i+tamanho].strip()
            for i in range(0, len(texto), tamanho)
            if texto[i:i+tamanho].strip()]
```

**Vantagens:** equilibra coerência semântica com controle de tamanho. Padrão da indústria.
**Desvantagens:** mais complexo de implementar e debugar.

### Semantic (semântico)

A abordagem mais sofisticada: usa embeddings para detectar mudanças de tópico e dividir nos pontos de transição.

```python
# chunking_semantico.py
# Divide texto detectando mudanças de tópico via embeddings

import numpy as np
import httpx

def gerar_embeddings_batch(textos: list[str]) -> list[list[float]]:
    """Gera embeddings em batch via Ollama."""
    resposta = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": textos},
        timeout=120.0,
    )
    resposta.raise_for_status()
    return resposta.json()["embeddings"]

def similaridade_cosseno(a: list[float], b: list[float]) -> float:
    """Similaridade por cosseno entre dois vetores."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

def chunking_semantico(
    sentencas: list[str],
    threshold: float = 0.75,
    min_chunk_size: int = 2,
) -> list[str]:
    """Agrupa sentenças por similaridade semântica.

    Quando a similaridade entre sentenças consecutivas cai abaixo do
    threshold, inicia um novo chunk.
    """
    if not sentencas:
        return []

    # Gera embeddings de todas as sentenças
    embeddings = gerar_embeddings_batch(sentencas)

    # Calcula similaridade entre sentenças consecutivas
    chunks = []
    grupo_atual = [sentencas[0]]

    for i in range(1, len(sentencas)):
        sim = similaridade_cosseno(embeddings[i-1], embeddings[i])

        if sim < threshold and len(grupo_atual) >= min_chunk_size:
            # Mudança de tópico detectada — fecha o chunk
            chunks.append(" ".join(grupo_atual))
            grupo_atual = [sentencas[i]]
        else:
            grupo_atual.append(sentencas[i])

    if grupo_atual:
        chunks.append(" ".join(grupo_atual))

    return chunks
```

**Vantagens:** chunks semanticamente coerentes, respeita tópicos.
**Desvantagens:** requer chamada de embedding (custo computacional), tamanho imprevisível, threshold precisa de ajuste.

## Overlap: quanto e por quê

Overlap é a sobreposição de conteúdo entre chunks consecutivos. Sem overlap, informações que caem na fronteira entre dois chunks podem ser perdidas no retrieval.

### Quanto overlap usar?

A regra empírica:

| Tamanho do chunk | Overlap recomendado | Proporção |
|------------------|---------------------|-----------|
| 200 caracteres | 20–40 | 10–20% |
| 500 caracteres | 50–100 | 10–20% |
| 1000 caracteres | 100–200 | 10–20% |
| 2000 caracteres | 200–300 | 10–15% |

**10–20%** do tamanho do chunk é o intervalo que funciona na maioria dos cenários. Menos que isso é você perde contexto de fronteira. Mais que isso é você infla o índice com conteúdo duplicado, desperdiçando espaço e poluindo os resultados.

### Quando NÃO usar overlap

- **Chunking por sentença ou semântico**: os chunks já respeitam fronteiras naturais.
- **Documentos estruturados** (JSON, YAML, tabelas): overlap pode gerar chunks inválidos sintaticamente.
- **Quando o tamanho do índice importa**: overlap aumenta o número total de chunks em ~15–25%.

## Metadata: enriquecendo chunks com contexto

Um chunk isolado perde contexto. De qual documento veio? De qual seção? Qual a data? Metadados restauram esse contexto e habilitam filtros poderosos no retrieval.

```python
# metadata_chunks.py
# Enriquece chunks com metadados para busca filtrada

from datetime import datetime
from pathlib import Path

def criar_chunks_com_metadata(
    arquivo: Path,
    chunks: list[str],
    categoria: str = "geral",
) -> list[dict]:
    """Adiciona metadados a cada chunk."""
    resultados = []
    for i, chunk in enumerate(chunks):
        resultados.append({
            "texto": chunk,
            "metadata": {
                "fonte": arquivo.name,
                "caminho": str(arquivo),
                "categoria": categoria,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tamanho_chars": len(chunk),
                "data_indexacao": datetime.now().isoformat(),
            },
        })
    return resultados
```

### Metadados úteis por tipo de documento

| Tipo de documento | Metadados recomendados |
|-------------------|----------------------|
| Documentação técnica | seção, versão, produto |
| Contratos | cláusula, parte, vigência |
| E-mails | remetente, data, assunto |
| Código-fonte | arquivo, função, linguagem |
| Artigos | autor, data, título, DOI |

No Qdrant, metadados são armazenados como **payload** e podem ser usados em filtros:

```python
# busca_com_filtro.py
# Busca vetorial com filtro por metadado no Qdrant

import httpx

def buscar_com_filtro(
    vetor_pergunta: list[float],
    categoria: str,
    top_k: int = 5,
) -> list[dict]:
    """Busca vetorial filtrada por categoria."""
    resposta = httpx.post(
        "http://localhost:6333/collections/documentos_rag/points/search",
        json={
            "vector": vetor_pergunta,
            "limit": top_k,
            "with_payload": True,
            "filter": {
                "must": [
                    {
                        "key": "metadata.categoria",
                        "match": {"value": categoria},
                    }
                ]
            },
        },
    )
    resposta.raise_for_status()
    return resposta.json()["result"]
```

## Indexação: batch vs streaming

### Batch (em lote)

Processa todos os documentos de uma vez. Ideal para carga inicial e reindexações completas.

```python
# indexacao_batch.py
# Indexação em lote de um diretório inteiro

import hashlib
import httpx
from pathlib import Path

QDRANT_URL = "http://localhost:6333"
COLECAO = "documentos_rag"
BATCH_SIZE = 100  # pontos por requisição ao Qdrant

def gerar_id(texto: str) -> str:
    """ID determinístico por hash SHA-256."""
    h = hashlib.sha256(texto.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def indexar_batch(pontos: list[dict], batch_size: int = BATCH_SIZE):
    """Envia pontos ao Qdrant em batches."""
    total = 0
    for i in range(0, len(pontos), batch_size):
        batch = pontos[i:i + batch_size]
        resposta = httpx.put(
            f"{QDRANT_URL}/collections/{COLECAO}/points?wait=true",
            json={"points": batch},
            timeout=60.0,
        )
        resposta.raise_for_status()
        total += len(batch)
        print(f"  Indexados: {total}/{len(pontos)}")
    return total
```

**Vantagens:** throughput alto, controle sobre o processo inteiro.
**Desvantagens:** bloqueante, não processa documentos novos até a próxima execução.

### Streaming (contínuo)

Processa documentos à medida que chegam. Ideal para sistemas que recebem documentos continuamente.

```python
# indexacao_streaming.py
# Indexação contínua: processa documentos à medida que aparecem

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileCreatedEvent, FileSystemEventHandler

class IndexadorHandler(FileSystemEventHandler):
    """Indexa automaticamente novos arquivos .txt."""

    def on_created(self, event):
        if not isinstance(event, FileCreatedEvent):
            return
        if not event.src_path.endswith(".txt"):
            return

        arquivo = Path(event.src_path)
        print(f"Novo arquivo detectado: {arquivo.name}")

        # Aqui entra o pipeline: carregar → chunk → embed → indexar
        texto = arquivo.read_text(encoding="utf-8")
        # chunks = chunking_recursivo(texto)
        # indexar_chunks(chunks, metadados)
        print(f"  Indexado: {arquivo.name}")

def monitorar_diretorio(caminho: str):
    """Monitora diretório e indexa novos arquivos automaticamente."""
    observer = Observer()
    observer.schedule(IndexadorHandler(), caminho, recursive=True)
    observer.start()
    print(f"Monitorando {caminho}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

## Reindexação e atualizações incrementais

Documentos mudam. Novos chegam, antigos são atualizados, alguns são removidos. O índice precisa refletir essas mudanças.

### Estratégia 1: Reindexação completa

Apaga tudo e reconstrói. Simples, mas lento para bases grandes.

### Estratégia 2: Upsert por hash (incremental)

Cada chunk recebe um ID determinístico baseado no hash do conteúdo. Se o conteúdo não mudou, o ID é o mesmo e o upsert é idempotente (sem desperdício). Se mudou, o novo vetor substitui o antigo.

```python
# reindexacao_incremental.py
# Reindexação incremental com purge de vetores obsoletos

import hashlib
import uuid
import httpx

QDRANT_URL = "http://localhost:6333"
COLECAO = "documentos_rag"

def reindexar_com_purge(pontos_novos: list[dict]):
    """Reindexa com batch_id para purge de vetores obsoletos.

    Padrão do AI-Orchestrator: cada reindexação grava um batch_id
    único no payload. Após o upsert, deleta pontos com batch_id
    diferente — removendo vetores de documentos que não existem mais.
    """
    batch_id = str(uuid.uuid4())

    # Marca todos os pontos com o batch_id atual
    for ponto in pontos_novos:
        ponto["payload"]["batch_id"] = batch_id

    # Upsert dos pontos atuais
    httpx.put(
        f"{QDRANT_URL}/collections/{COLECAO}/points?wait=true",
        json={"points": pontos_novos},
        timeout=120.0,
    ).raise_for_status()

    # Purge: deleta pontos com batch_id diferente do atual
    httpx.post(
        f"{QDRANT_URL}/collections/{COLECAO}/points/delete",
        json={
            "filter": {
                "must_not": [
                    {"key": "batch_id", "match": {"value": batch_id}}
                ]
            }
        },
    ).raise_for_status()

    print(f"Reindexação completa. batch_id={batch_id[:8]}...")
```

Este padrão — upsert + purge por `batch_id` — é essencial quando a fonte de dados é mutável. Sem o purge, vetores de documentos deletados permanecem no índice como "fantasmas", poluindo resultados para sempre. No AI-Orchestrator, essa lição foi aprendida na prática: o golden set de roteamento é estático, mas documentos corporativos não são.

## Na prática: indexando documentos corporativos

Vamos juntar tudo em um script que indexa documentos de um diretório corporativo.

```python
# indexar_corporativo.py
# Pipeline completo: carregar → chunk → enriquecer → indexar

import hashlib
from pathlib import Path
import httpx

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLECAO = "docs_corporativos"
MODELO_EMBED = "nomic-embed-text"
DIMENSAO = 768
TAMANHO_CHUNK = 500
OVERLAP = 50

def pipeline_indexacao(diretorio: str, categoria: str = "geral"):
    """Pipeline completo de indexação corporativa."""
    diretorio_path = Path(diretorio)

    # 1. Cria coleção se necessário
    check = httpx.get(f"{QDRANT_URL}/collections/{COLECAO}")
    if check.status_code != 200:
        httpx.put(
            f"{QDRANT_URL}/collections/{COLECAO}",
            json={"vectors": {"size": DIMENSAO, "distance": "Cosine"}},
        ).raise_for_status()

    # 2. Processa cada arquivo
    todos_pontos = []
    extensoes = {".txt", ".md", ".csv"}

    for arquivo in diretorio_path.rglob("*"):
        if arquivo.suffix not in extensoes:
            continue
        if arquivo.is_dir():
            continue

        print(f"Processando: {arquivo.name}")
        texto = arquivo.read_text(encoding="utf-8", errors="ignore")

        # Chunking recursivo simplificado
        chunks = []
        inicio = 0
        while inicio < len(texto):
            fim = inicio + TAMANHO_CHUNK
            chunk = texto[inicio:fim].strip()
            if chunk:
                chunks.append(chunk)
            inicio += TAMANHO_CHUNK - OVERLAP

        if not chunks:
            continue

        # Gera embeddings
        embeddings = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODELO_EMBED, "input": chunks},
            timeout=120.0,
        ).json()["embeddings"]

        # Monta pontos com metadata
        for i, (chunk, vetor) in enumerate(zip(chunks, embeddings)):
            h = hashlib.sha256(chunk.encode()).hexdigest()
            ponto_id = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
            todos_pontos.append({
                "id": ponto_id,
                "vector": vetor,
                "payload": {
                    "texto": chunk,
                    "fonte": arquivo.name,
                    "categoria": categoria,
                    "chunk_index": i,
                },
            })

    # 3. Indexa em batch
    BATCH = 100
    for i in range(0, len(todos_pontos), BATCH):
        batch = todos_pontos[i:i+BATCH]
        httpx.put(
            f"{QDRANT_URL}/collections/{COLECAO}/points?wait=true",
            json={"points": batch},
            timeout=60.0,
        ).raise_for_status()

    print(f"\nTotal indexado: {len(todos_pontos)} chunks de {diretorio}")

# Uso
if __name__ == "__main__":
    pipeline_indexacao("/caminho/para/documentos", categoria="politicas")
```

---

## Resumo

| Estratégia | Quando usar |
|------------|-------------|
| Fixed-size | Prototipação rápida, dados sem estrutura |
| Sentence | Textos narrativos, artigos |
| Recursive | Padrão geral, documentação técnica |
| Semantic | Quando coerência de tópico é crítica |

| Decisão | Recomendação |
|---------|-------------|
| Tamanho do chunk | 300–800 caracteres (ajustar por recall) |
| Overlap | 10–20% do tamanho |
| Metadados | Sempre incluir fonte, data, categoria |
| Reindexação | Upsert por hash + purge por batch_id |

---

## Referências

- Langchain Documentation. *Text Splitters*. https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Pinecone. *Chunking Strategies for LLM Applications*. https://www.pinecone.io/learn/chunking-strategies/
- *RAG with Python Cookbook*, Capítulo 4 — Data Preparation and Chunking.
- Projeto AI-Orchestrator — `gateway/semantic_router.py` (padrão de upsert idempotente por hash e purge por batch_id).
- Qdrant Documentation. *Points and Payloads*. https://qdrant.tech/documentation/concepts/points/
