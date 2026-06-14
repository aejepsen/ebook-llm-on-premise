# Capítulo 16 — Graph RAG com Neo4j

## Limitações do RAG vetorial puro

O RAG vetorial funciona excepcionalmente bem para perguntas que podem ser respondidas por **trechos isolados** de texto. "O que diz a política de férias?" → busca os chunks relevantes → responde. Mas há classes de perguntas que o RAG vetorial não resolve bem:

1. **Perguntas que exigem conexões entre entidades.** "Quais clientes compraram produtos do mesmo fornecedor que a empresa X?" — a resposta não está em nenhum chunk individual. Está na **relação** entre clientes, produtos e fornecedores.

2. **Perguntas multi-hop.** "Quem é o gerente do departamento que mais contratou no último trimestre?" — requer navegar: departamentos → contratações → gerentes. Cada hop é uma relação.

3. **Raciocínio sobre estrutura.** "Quais equipes têm dependências circulares?" — impossível responder olhando textos. Requer análise de grafo.

4. **Contexto holístico.** O RAG vetorial retorna top-k chunks. Se a informação relevante está espalhada em 50 documentos, os top-5 capturam apenas uma fração. Knowledge graphs condensam essa informação em relações navegáveis.

O RAG vetorial vê **textos**. O Graph RAG vê **conexões**.

## Knowledge Graphs: entidades, relações, propriedades

Um knowledge graph (grafo de conhecimento) é uma estrutura de dados que modela o mundo como:

- **Nós (entidades):** pessoas, empresas, produtos, departamentos, documentos.
- **Arestas (relações):** "trabalha_em", "fornece_para", "escrito_por", "depende_de".
- **Propriedades:** atributos dos nós e arestas (nome, data, valor).

```
(João) -[TRABALHA_EM]→ (Departamento Vendas) -[GERENCIADO_POR]→ (Maria)
(João) -[VENDEU]→ (Produto X) -[FORNECIDO_POR]→ (Fornecedor ABC)
```

Essa estrutura permite queries como: "todos os produtos vendidos por funcionários do departamento gerenciado por Maria" — uma travessia natural no grafo, impossível com busca vetorial.

### Grafo vs Tabela relacional

| Aspecto | Banco relacional | Knowledge graph |
|---------|-----------------|-----------------|
| Modelo | Tabelas + JOINs | Nós + arestas |
| Consulta relacional | JOINs custosos em profundidade | Travessias nativas O(1) por hop |
| Schema | Rígido (DDL) | Flexível (schema-less) |
| Descoberta de padrões | Difícil | Natural (pathfinding, clustering) |
| Ideal para | Dados estruturados, ACID | Dados conectados, exploração |

## Neo4j: instalação e Cypher básico

Neo4j é o banco de grafos mais maduro e documentado. Suporta ACID, índices, full-text search e, com extensões, busca vetorial.

### Instalação via Docker

```bash
# Subir Neo4j em container
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -v neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/senha_segura_123 \
  -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
  neo4j:5-community

# Acessar: http://localhost:7474 (Neo4j Browser)
# Bolt: bolt://localhost:7687
```

### Cypher: a linguagem de consulta

Cypher é para grafos o que SQL é para tabelas. A sintaxe é visual — parênteses para nós, setas para relações.

```cypher
// Criar nós
CREATE (j:Pessoa {nome: "João", cargo: "Vendedor"})
CREATE (m:Pessoa {nome: "Maria", cargo: "Gerente"})
CREATE (d:Departamento {nome: "Vendas"})
CREATE (p:Produto {sku: "SKU-001", nome: "Widget"})
CREATE (f:Fornecedor {nome: "ABC Ltda"})

// Criar relações
CREATE (j)-[:TRABALHA_EM]->(d)
CREATE (m)-[:GERENCIA]->(d)
CREATE (j)-[:VENDEU {data: date("2026-01-15"), valor: 1500}]->(p)
CREATE (p)-[:FORNECIDO_POR]->(f)
```

```cypher
// Consultas

// 1. Quem trabalha no departamento de Vendas?
MATCH (p:Pessoa)-[:TRABALHA_EM]->(d:Departamento {nome: "Vendas"})
RETURN p.nome, p.cargo

// 2. Qual o gerente do departamento onde João trabalha?
MATCH (j:Pessoa {nome: "João"})-[:TRABALHA_EM]->(d)<-[:GERENCIA]-(g)
RETURN g.nome AS gerente, d.nome AS departamento

// 3. Quais fornecedores estão conectados a funcionários de Vendas?
// (query multi-hop: pessoa → produto → fornecedor)
MATCH (p:Pessoa)-[:TRABALHA_EM]->(d:Departamento {nome: "Vendas"}),
      (p)-[:VENDEU]->(prod)-[:FORNECIDO_POR]->(f:Fornecedor)
RETURN DISTINCT f.nome AS fornecedor

// 4. Caminho mais curto entre duas entidades
MATCH caminho = shortestPath(
  (a:Pessoa {nome: "João"})-[*]-(b:Fornecedor {nome: "ABC Ltda"})
)
RETURN caminho
```

### Conectando via Python

```python
# neo4j_basico.py
# Conecta ao Neo4j e executa queries Cypher

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USUARIO = "neo4j"
SENHA = "senha_segura_123"

driver = GraphDatabase.driver(URI, auth=(USUARIO, SENHA))

def executar_query(query: str, parametros: dict = None) -> list[dict]:
    """Executa uma query Cypher e retorna os resultados."""
    with driver.session() as sessao:
        resultado = sessao.run(query, parametros or {})
        return [registro.data() for registro in resultado]

# Exemplo: buscar todas as pessoas
pessoas = executar_query("MATCH (p:Pessoa) RETURN p.nome, p.cargo")
for p in pessoas:
    print(f"{p['p.nome']} - {p['p.cargo']}")

# Exemplo: buscar relações de uma pessoa
relacoes = executar_query(
    "MATCH (p:Pessoa {nome: $nome})-[r]->(alvo) RETURN type(r), alvo",
    {"nome": "João"},
)
for r in relacoes:
    print(f"  → {r['type(r)']}: {r['alvo']}")

driver.close()
```

## Construindo grafos a partir de texto (NER + relation extraction)

Na maioria dos cenários reais, você não tem dados estruturados — tem texto. Documentos, e-mails, relatórios. Para construir um knowledge graph a partir de texto, precisamos de duas etapas:

### 1. NER (Named Entity Recognition)

Identifica as entidades mencionadas no texto: pessoas, organizações, locais, produtos.

### 2. Relation Extraction

Identifica como as entidades se relacionam: "João trabalha na empresa X", "o produto Y é fornecido por Z".

### Usando LLM para extração

LLMs são surpreendentemente bons em extração estruturada. Em vez de modelos NER dedicados, podemos usar o próprio LLM local:

```python
# extrair_grafo_llm.py
# Usa LLM local para extrair entidades e relações de texto

import json
import httpx

OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen2.5:7b"

PROMPT_EXTRACAO = """Analise o texto abaixo e extraia TODAS as entidades e relações.

Retorne um JSON com o formato:
{
  "entidades": [
    {"id": "e1", "tipo": "Pessoa|Empresa|Produto|Local|Departamento", "nome": "..."}
  ],
  "relacoes": [
    {"origem": "e1", "destino": "e2", "tipo": "TRABALHA_EM|FORNECE|GERENCIA|LOCALIZADO_EM|..."}
  ]
}

Regras:
- Extraia SOMENTE o que está explícito no texto — nunca infira relações.
- Use IDs curtos (e1, e2, e3...).
- Tipos de relação em MAIÚSCULAS com underscore.
- Retorne APENAS o JSON, sem explicação.

Texto:
{texto}"""

def extrair_entidades_relacoes(texto: str) -> dict:
    """Extrai entidades e relações de um texto usando LLM local."""
    resposta = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODELO,
            "messages": [
                {"role": "user", "content": PROMPT_EXTRACAO.format(texto=texto)},
            ],
            "stream": False,
            "format": "json",  # força output em JSON
        },
        timeout=120.0,
    )
    resposta.raise_for_status()
    conteudo = resposta.json()["message"]["content"]
    return json.loads(conteudo)

def inserir_no_neo4j(driver, dados: dict):
    """Insere entidades e relações extraídas no Neo4j."""
    with driver.session() as sessao:
        # Cria entidades
        for ent in dados["entidades"]:
            sessao.run(
                f"MERGE (n:{ent['tipo']} {{nome: $nome}})",
                {"nome": ent["nome"]},
            )

        # Cria relações
        for rel in dados["relacoes"]:
            origem = next(e for e in dados["entidades"] if e["id"] == rel["origem"])
            destino = next(e for e in dados["entidades"] if e["id"] == rel["destino"])
            query = (
                f"MATCH (a:{origem['tipo']} {{nome: $nome_a}}), "
                f"(b:{destino['tipo']} {{nome: $nome_b}}) "
                f"MERGE (a)-[:{rel['tipo']}]->(b)"
            )
            sessao.run(query, {"nome_a": origem["nome"], "nome_b": destino["nome"]})

# Exemplo
texto = """
João Silva é gerente do departamento de Engenharia na TechCorp.
Maria Santos trabalha como desenvolvedora no mesmo departamento.
A TechCorp está localizada em São Paulo e fornece serviços para a RetailMax.
"""

dados = extrair_entidades_relacoes(texto)
print(json.dumps(dados, indent=2, ensure_ascii=False))
```

## Busca híbrida: vetorial + grafo

O verdadeiro poder aparece quando combinamos busca vetorial (para encontrar documentos relevantes) com travessia de grafo (para expandir o contexto com relações).

```python
# busca_hibrida.py
# Combina busca vetorial (Qdrant) + travessia de grafo (Neo4j)

import httpx
from neo4j import GraphDatabase

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
NEO4J_URI = "bolt://localhost:7687"

driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "senha_segura_123"))

def busca_hibrida(pergunta: str, top_k: int = 5) -> dict:
    """Pipeline de busca híbrida: vetorial + grafo.

    1. Busca vetorial identifica chunks relevantes.
    2. Extrai entidades mencionadas nos chunks.
    3. Expande contexto via travessia no grafo.
    """
    # Passo 1: Busca vetorial
    vetor = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "nomic-embed-text", "input": pergunta},
        timeout=30.0,
    ).json()["embeddings"][0]

    resultados = httpx.post(
        f"{QDRANT_URL}/collections/documentos_rag/points/search",
        json={"vector": vetor, "limit": top_k, "with_payload": True},
    ).json()["result"]

    chunks = [r["payload"]["texto"] for r in resultados]

    # Passo 2: Extrai entidades dos chunks (simplificado)
    entidades = extrair_nomes_proprios(chunks)

    # Passo 3: Expande via grafo
    contexto_grafo = []
    with driver.session() as sessao:
        for entidade in entidades:
            # Busca vizinhos de 1 hop
            resultado = sessao.run(
                """
                MATCH (n {nome: $nome})-[r]-(vizinho)
                RETURN n.nome, type(r) AS relacao, vizinho.nome
                LIMIT 10
                """,
                {"nome": entidade},
            )
            for registro in resultado:
                contexto_grafo.append(
                    f"{registro['n.nome']} → {registro['relacao']} → {registro['vizinho.nome']}"
                )

    return {
        "chunks_vetoriais": chunks,
        "relacoes_grafo": contexto_grafo,
        "contexto_combinado": chunks + contexto_grafo,
    }

def extrair_nomes_proprios(textos: list[str]) -> list[str]:
    """Extrai nomes próprios simples (heurística por capitalização)."""
    import re
    nomes = set()
    for texto in textos:
        # Palavras capitalizadas que não iniciam sentenças (heurística)
        palavras = re.findall(r'(?<!\.\s)\b([A-Z][a-záéíóú]+(?:\s[A-Z][a-záéíóú]+)*)\b', texto)
        nomes.update(palavras)
    return list(nomes)
```

### O fluxo combinado

```
Pergunta do usuário
       ↓
[Busca vetorial] → top-k chunks
       ↓
[Extração de entidades] → entidades mencionadas
       ↓
[Travessia no grafo] → relações e entidades conectadas
       ↓
[Contexto combinado] = chunks + relações
       ↓
[LLM] → Resposta fundamentada em texto E estrutura
```

## GraphRAG: o approach da Microsoft

Em 2024, a Microsoft Research publicou o GraphRAG, uma abordagem que automatiza a construção de knowledge graphs a partir de documentos e os usa para responder perguntas que exigem visão holística.

### O pipeline GraphRAG

1. **Extração de entidades e relações** por LLM, documento por documento.
2. **Construção de comunidades** — agrupamento automático de entidades densamente conectadas (usando algoritmo de Leiden).
3. **Geração de sumários por comunidade** — cada comunidade recebe um resumo gerado por LLM que captura o tema central.
4. **Query time**: perguntas globais ("qual o tema principal desse corpus?") são respondidas consultando os sumários das comunidades. Perguntas locais usam busca vetorial + travessia.

### Quando usar GraphRAG vs busca vetorial simples

```
Pergunta local ("o que diz o artigo 5?")
  → RAG vetorial basta

Pergunta global ("quais são os temas recorrentes nos relatórios de 2025?")
  → GraphRAG com comunidades e sumários

Pergunta relacional ("quais clientes compartilham fornecedores?")
  → Travessia de grafo (Cypher)

Pergunta híbrida ("resuma as reclamações dos clientes do fornecedor X")
  → Busca vetorial + travessia de grafo + síntese por LLM
```

### Implementação simplificada on-premise

```python
# graph_rag_simples.py
# GraphRAG simplificado: extração → grafo → comunidades → sumários

from neo4j import GraphDatabase
import httpx
import json

OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen2.5:7b"
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "senha_segura_123"))

def construir_grafo_de_documentos(documentos: list[str]):
    """Extrai entidades/relações de cada documento e insere no Neo4j."""
    for i, doc in enumerate(documentos):
        print(f"Processando documento {i+1}/{len(documentos)}...")
        dados = extrair_entidades_relacoes(doc)
        inserir_no_neo4j(driver, dados)

def detectar_comunidades():
    """Usa Graph Data Science para detectar comunidades."""
    with driver.session() as sessao:
        # Projeta o grafo em memória
        sessao.run("""
            CALL gds.graph.project(
                'meu_grafo',
                '*',
                {ALL: {orientation: 'UNDIRECTED'}}
            )
        """)

        # Detecta comunidades com Leiden
        sessao.run("""
            CALL gds.leiden.write('meu_grafo', {
                writeProperty: 'comunidade',
                maxLevels: 10,
                gamma: 1.0
            })
        """)

        # Recupera comunidades
        resultado = sessao.run("""
            MATCH (n)
            RETURN n.comunidade AS comunidade, collect(n.nome) AS membros
            ORDER BY size(collect(n.nome)) DESC
        """)
        return [r.data() for r in resultado]

def gerar_sumarios_comunidades(comunidades: list[dict]) -> list[str]:
    """Gera sumário para cada comunidade via LLM."""
    sumarios = []
    for com in comunidades:
        membros = ", ".join(com["membros"][:20])  # limita para prompt
        prompt = (
            f"As seguintes entidades formam uma comunidade: {membros}. "
            "Descreva em 2-3 frases o tema central que as conecta."
        )
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODELO,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60.0,
        ).json()
        sumarios.append(resp["message"]["content"])
    return sumarios
```

## Quando usar Graph RAG vs RAG vetorial

| Critério | RAG vetorial | Graph RAG |
|----------|-------------|-----------|
| Perguntas factoids locais | Excelente | Excessivo |
| Perguntas relacionais | Fraco | Excelente |
| Perguntas globais/tema | Limitado | Forte (comunidades) |
| Complexidade de setup | Baixa | Alta |
| Custo de indexação | Baixo | Alto (extração por LLM) |
| Manutenção | Simples | Requer curadoria do grafo |
| Latência | Baixa | Maior (travessia + busca) |

### Regra prática

Comece com RAG vetorial. Se as perguntas dos seus usuários envolvem relações entre entidades, adicione a camada de grafo. Não construa um knowledge graph porque é elegante — construa porque as perguntas exigem.

No AI-Orchestrator, a camada de roteamento semântico usa Qdrant (RAG vetorial simples) porque o problema é de similaridade entre perguntas, não de relações entre entidades. Se os domínios tivessem dependências cruzadas (um pedido de vendas que verifica estoque é impacta financeiro simultaneamente), um grafo de dependências entre domínios seria justificado.

---

## Resumo

| Componente | Tecnologia | Quando usar |
|------------|-----------|-------------|
| Busca vetorial | Qdrant + nomic-embed-text | Perguntas locais, factoids |
| Knowledge graph | Neo4j + Cypher | Perguntas relacionais, multi-hop |
| Extração | LLM local (Qwen2.5) | Construção de grafo a partir de texto |
| Comunidades | Neo4j GDS (Leiden) | Perguntas globais, temas |
| Busca híbrida | Qdrant + Neo4j | Melhor dos dois mundos |

---

## Referências

- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. Microsoft Research.
- Neo4j Documentation. https://neo4j.com/docs/
- Neo4j Graph Data Science Library. https://neo4j.com/docs/graph-data-science/current/
- *RAG with Python Cookbook*, Capítulo 9 — Graph RAG (notebooks 9.1–9.5).
- Traag, V. A. et al. (2019). *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports.
- Projeto AI-Orchestrator — `gateway/semantic_router.py` (decisão de usar Qdrant vetorial simples onde grafo seria over-engineering).
