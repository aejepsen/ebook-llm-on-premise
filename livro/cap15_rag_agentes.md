# Capítulo 15 — RAG com Agentes

## O que são agentes LLM

Um LLM sozinho é um gerador de texto: recebe prompt, produz resposta. Um **agente** LLM é um sistema que usa o modelo como motor de decisão para executar ações no mundo real — consultar APIs, ler bancos de dados, manipular arquivos, enviar notificações.

A diferença fundamental:

| LLM puro | Agente LLM |
|----------|-----------|
| Recebe pergunta, gera texto | Recebe pergunta, decide o que fazer |
| Conhecimento estático (treinamento) | Acessa dados em tempo real (ferramentas) |
| Uma rodada de geração | Loop de raciocínio + ação |
| Não pode verificar fatos | Consulta fontes antes de responder |

Um agente RAG combina o melhor dos dois mundos: usa retrieval para buscar contexto relevante e usa ferramentas para executar ações quando necessário. A pergunta "qual o saldo do estoque do produto X?" não é respondida por retrieval — é respondida por uma chamada de API ao sistema de estoque.

## Function Calling: o modelo escolhe a ferramenta

Function calling (ou tool calling) é o mecanismo pelo qual o modelo, em vez de gerar texto, **solicita a execução de uma função**. O modelo recebe a descrição das ferramentas disponíveis e decide qual usar, com quais argumentos.

### Como funciona

1. O sistema envia ao modelo a pergunta do usuário + a lista de ferramentas disponíveis.
2. O modelo analisa a pergunta e decide se precisa de uma ferramenta.
3. Se sim, retorna um JSON estruturado com o nome da ferramenta e os argumentos.
4. O sistema executa a ferramenta e devolve o resultado ao modelo.
5. O modelo gera a resposta final com base no resultado.

```python
# function_calling_basico.py
# Exemplo de function calling com Ollama

import httpx
import json

OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen2.5:7b"

# Definição das ferramentas disponíveis
ferramentas = [
    {
        "type": "function",
        "function": {
            "name": "consultar_estoque",
            "description": "Consulta o saldo atual de um produto no estoque pelo SKU",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Código SKU do produto",
                    }
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_documentos",
            "description": "Busca documentos relevantes na base de conhecimento",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Texto da consulta para busca semântica",
                    }
                },
                "required": ["consulta"],
            },
        },
    },
]

def chat_com_ferramentas(pergunta: str) -> dict:
    """Envia pergunta com ferramentas ao modelo."""
    resposta = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODELO,
            "messages": [
                {"role": "user", "content": pergunta},
            ],
            "tools": ferramentas,
            "stream": False,
        },
        timeout=120.0,
    )
    resposta.raise_for_status()
    return resposta.json()

# Teste: pergunta que requer ferramenta
resultado = chat_com_ferramentas("Qual o saldo do produto SKU-001?")
mensagem = resultado["message"]

if mensagem.get("tool_calls"):
    for chamada in mensagem["tool_calls"]:
        print(f"Ferramenta: {chamada['function']['name']}")
        print(f"Argumentos: {chamada['function']['arguments']}")
else:
    print(f"Resposta direta: {mensagem['content']}")
```

### Ferramentas no AI-Orchestrator

No AI-Orchestrator, as ferramentas são **descobertas automaticamente** a partir dos endpoints dos microsserviços. O `ToolRegistry` consulta os schemas OpenAPI dos microsserviços de cada domínio (finanças, RH, estoque, vendas) e converte cada endpoint em uma ferramenta que o modelo pode chamar. É o padrão de **tool discovery via OpenAPI** — o agente não precisa conhecer as ferramentas em tempo de compilação.

## ReAct: Reason + Act — o loop fundamental

ReAct (Reasoning + Acting) é o padrão que estrutura como um agente pensa e age. O modelo alterna entre **raciocínio** (pensar sobre o que fazer) e **ação** (executar uma ferramenta), em um loop iterativo.

```
Pensamento: "O usuário quer saber o saldo do produto X. Preciso consultar o estoque."
Ação: consultar_estoque(sku="SKU-001")
Observação: {"saldo": 42, "unidade": "peças"}
Pensamento: "Tenho o saldo. Posso responder."
Resposta: "O produto SKU-001 tem 42 peças em estoque."
```

### Implementação do loop ReAct

```python
# react_loop.py
# Loop ReAct completo: raciocínio → ação → observação → resposta

import json
import time
import httpx

OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen2.5:7b"
MAX_ITERACOES = 6  # limite de segurança
DEADLINE_S = 30.0  # timeout total

def executar_ferramenta(nome: str, argumentos: dict) -> dict:
    """Executa uma ferramenta e retorna o resultado.

    Em produção, isso chamaria a API do microsserviço correspondente.
    """
    # Simulação para exemplo
    ferramentas_mock = {
        "consultar_estoque": lambda args: {
            "status": 200,
            "body": {"sku": args["sku"], "saldo": 42, "unidade": "peças"},
        },
        "buscar_documentos": lambda args: {
            "status": 200,
            "body": {"resultados": ["Doc 1: política de estoque...", "Doc 2: regras..."]},
        },
    }
    fn = ferramentas_mock.get(nome)
    if fn:
        return fn(argumentos)
    return {"status": 404, "body": {"error": f"Ferramenta '{nome}' não encontrada"}}

def loop_react(pergunta: str, ferramentas: list[dict]) -> str:
    """Loop ReAct: itera até resposta final ou limite."""
    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente que usa ferramentas para responder perguntas. "
                "Use exclusivamente as ferramentas fornecidas. Nunca invente dados."
            ),
        },
        {"role": "user", "content": pergunta},
    ]

    inicio = time.monotonic()

    for iteracao in range(MAX_ITERACOES):
        # Verifica deadline
        if time.monotonic() - inicio > DEADLINE_S:
            return "Não foi possível concluir dentro do tempo limite."

        # Chama o modelo
        resposta = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODELO,
                "messages": mensagens,
                "tools": ferramentas,
                "stream": False,
            },
            timeout=60.0,
        )
        resposta.raise_for_status()
        msg = resposta.json()["message"]

        # Se não há tool_calls, o modelo decidiu responder diretamente
        if not msg.get("tool_calls"):
            return msg["content"]

        # Adiciona a mensagem do assistente (com tool_calls)
        mensagens.append({
            "role": "assistant",
            "content": msg.get("content", ""),
            "tool_calls": msg["tool_calls"],
        })

        # Executa cada ferramenta chamada
        for chamada in msg["tool_calls"]:
            nome = chamada["function"]["name"]
            args = chamada["function"]["arguments"]
            print(f"  [{iteracao+1}] Executando: {nome}({args})")

            resultado = executar_ferramenta(nome, args)

            # Devolve o resultado ao modelo como mensagem de tool
            mensagens.append({
                "role": "tool",
                "tool_name": nome,
                "content": json.dumps(resultado, ensure_ascii=False, default=str),
            })

    return "Limite de iterações atingido sem resposta final."
```

Este padrão — o loop com `max_iters` + `deadline_s` — é exatamente o que o AI-Orchestrator usa no `run_domain_agent`. Dois limites de segurança (iterações e tempo) garantem que o agente nunca entra em loop infinito.

## LangGraph: grafos de estado para orquestração

LangGraph é o framework de orquestração que modela o fluxo do agente como um **grafo de estados**. Cada nó é uma função, cada aresta define a transição. Diferente de uma cadeia linear (LangChain), um grafo permite bifurcações, loops controlados e paralelismo.

### Por que grafos?

Considere o fluxo do AI-Orchestrator:

```
sanitize → classify → (clarificação? → responder)
                       (senão → confirmar → dispatch → synthesize)
```

Isso não é linear. Há uma decisão condicional após `classify`: se a pergunta é ambígua, pede clarificação; se é clara, despacha para os agentes. Um grafo modela isso naturalmente.

```python
# grafo_basico.py
# Exemplo mínimo de LangGraph StateGraph

from typing import TypedDict
from langgraph.graph import END, StateGraph

# 1. Define o estado do grafo
class Estado(TypedDict):
    pergunta: str
    contexto: str
    resposta: str

# 2. Define os nós (funções que transformam o estado)
def buscar_contexto(estado: Estado) -> Estado:
    """Nó de retrieval: busca documentos relevantes."""
    # Em produção, faria busca vetorial no Qdrant
    contexto = f"Contexto recuperado para: {estado['pergunta']}"
    return {"contexto": contexto}

def gerar_resposta(estado: Estado) -> Estado:
    """Nó de geração: produz resposta com base no contexto."""
    # Em produção, chamaria o LLM com o contexto
    resposta = f"Resposta baseada em: {estado['contexto']}"
    return {"resposta": resposta}

# 3. Monta o grafo
grafo = StateGraph(Estado)
grafo.add_node("buscar", buscar_contexto)
grafo.add_node("gerar", gerar_resposta)
grafo.set_entry_point("buscar")
grafo.add_edge("buscar", "gerar")
grafo.add_edge("gerar", END)

# 4. Compila e executa
app = grafo.compile()
resultado = app.invoke({"pergunta": "O que é RAG?", "contexto": "", "resposta": ""})
print(resultado["resposta"])
```

### Conditional edges (arestas condicionais)

O poder do LangGraph está nas arestas condicionais. No AI-Orchestrator:

```python
# O grafo decide o caminho com base no estado
graph.add_conditional_edges(
    "classify",
    # Função que decide para onde ir
    lambda state: "respond_clarification"
        if state["route"].get("clarification")
        else "confirm_dispatch",
    {
        "respond_clarification": "respond_clarification",
        "confirm_dispatch": "confirm_dispatch",
    },
)
```

Se o classificador detecta ambiguidade, o grafo vai para `respond_clarification`. Senão, vai para `confirm_dispatch`. A decisão está no estado, não no código — o grafo é declarativo.

## Ferramentas (tools): definição, execução, tratamento de erro

### Definição de ferramentas

Uma ferramenta é uma função com schema JSON que o modelo pode chamar. O schema descreve: nome, descrição, parâmetros e tipos.

### Tratamento de erros

Quando uma ferramenta falha (API retorna 422, 404, timeout), o erro **deve ser reinjetado no modelo como observação**, não como exceção. O modelo então decide: tentar novamente com argumentos diferentes, usar outra ferramenta, ou informar o usuário.

```python
# tratamento_erro_ferramentas.py
# Padrão de tratamento de erro do AI-Orchestrator

def executar_com_fallback(registry, dominio: str, chamada) -> dict:
    """Executa ferramenta com fallback para ferramentas desconhecidas.

    Padrão do AI-Orchestrator: se a ferramenta não existe no domínio,
    retorna uma mensagem estruturada que o modelo entende — listando
    as ferramentas disponíveis.
    """
    try:
        return registry.execute(dominio, chamada.name, chamada.arguments)
    except ToolNotFound as exc:
        # Não é exceção fatal — é feedback para o modelo
        return {
            "status": 0,
            "body": {
                "error": "unknown_tool",
                "detail": (
                    f"A ferramenta '{exc.name}' não existe neste domínio. "
                    f"Disponíveis: {', '.join(exc.known)}."
                ),
                "rule": "use somente as ferramentas fornecidas",
            },
        }
```

Este padrão é crucial: o modelo vai errar nomes de ferramentas. Em vez de crashar, o sistema informa quais ferramentas existem. O modelo autocorrige na próxima iteração.

### Regra de negócio na API, não no prompt

Decisão de arquitetura do AI-Orchestrator que merece destaque: **a regra de negócio vive na API, nunca no modelo**. Se o desconto máximo é 15%, a API rejeita descontos maiores (HTTP 422). O modelo recebe o erro e informa o usuário. Não dependa do prompt para impor regras — o modelo pode ignorar.

## Caso real: agentes do AI-Orchestrator (4 domínios)

O AI-Orchestrator implementa 4 agentes especializados, cada um com acesso exclusivo ao microsserviço do seu domínio:

| Domínio | Ferramentas | Exemplo de pergunta |
|---------|------------|---------------------|
| **Finanças** | contas a pagar/receber, fluxo de caixa, alçada | "Qual o total a receber este mês?" |
| **RH** | funcionários, férias CLT, reembolsos, headcount | "Quantos funcionários há no departamento de vendas?" |
| **Estoque** | produtos por SKU, saldo, reservas, ponto de reposição | "Qual o saldo do SKU-001?" |
| **Vendas** | pedidos, política de desconto, comissão | "Liste os pedidos pendentes." |

Cada agente recebe um system prompt que define:

1. **Identidade**: "Você é o agente especialista de {domínio}."
2. **Data atual**: `{today}` — evita que o modelo invente datas.
3. **Regra anti-fabricação**: "Nunca invente valores, IDs, SKUs ou totais."
4. **Regra de escrita**: "Para operações de escrita sem todos os campos obrigatórios, liste os campos e peça os valores — nunca fabrique."

A regra de escrita nasceu de um incidente real: sem ela, o agente criava funcionários com nomes inventados ao receber "incluir um funcionário" sem detalhes. O modelo preenchia os campos obrigatórios com dados fabricados.

## Multi-agente: fan-out/fan-in, síntese de respostas

Quando uma pergunta envolve múltiplos domínios — "qual o impacto financeiro das férias do time de vendas?" — o AI-Orchestrator ativa múltiplos agentes em paralelo (fan-out) e sintetiza as respostas em uma resposta unificada (fan-in).

### Fan-out: execução paralela

```python
# fan_out.py
# Execução paralela de agentes com ThreadPoolExecutor

from concurrent.futures import ThreadPoolExecutor

def dispatch_paralelo(dominios: list[str], pergunta: str) -> dict[str, str]:
    """Executa agentes em paralelo e coleta resultados.

    O ThreadPoolExecutor paraleliza as chamadas. Mesmo que o Ollama
    serialize gerações (GPU única), o paralelismo vale para os
    microsserviços (I/O bound) e prepara a arquitetura para múltiplas
    GPUs no futuro.
    """
    def executar(dominio: str) -> tuple[str, str]:
        resultado = run_domain_agent(dominio, pergunta)
        return dominio, resultado.final_answer

    with ThreadPoolExecutor(max_workers=len(dominios)) as pool:
        resultados = dict(pool.map(executar, dominios))

    return resultados
```

### Fan-in: síntese

Com 1 domínio, a resposta do agente é devolvida diretamente (sem chamada extra ao LLM). Com múltiplos domínios, o orquestrador sintetiza:

```
"Você recebeu as respostas de {N} agentes especialistas.
Sintetize UMA resposta final em português, curta e direta,
fundamentada EXCLUSIVAMENTE nas respostas dos agentes —
nunca invente números ou fatos que não estejam nelas."
```

Essa regra — "fundamentada exclusivamente" — é a defesa contra fabricação na camada de síntese. O LLM não inventa se o prompt proíbe explicitamente e o contexto fornece dados concretos.

---

## Resumo

| Conceito | Implementação |
|----------|---------------|
| Function calling | Modelo decide ferramenta + argumentos |
| ReAct loop | `max_iters` + `deadline_s` para segurança |
| LangGraph | StateGraph com conditional edges |
| Erro de ferramenta | Reinjetar como observação, não exceção |
| Multi-agente | ThreadPoolExecutor para fan-out, LLM para fan-in |
| Anti-fabricação | System prompt + regra de negócio na API |

---

## Referências

- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.
- LangGraph Documentation. https://langchain-ai.github.io/langgraph/
- Projeto AI-Orchestrator — `gateway/agents.py` (loop de tool-calling com max_iters + deadline), `gateway/graph.py` (StateGraph com fan-out/fan-in).
- *RAG with Python Cookbook*, Capítulo 8 — Agentic RAG.
- Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.
- Qwen2.5 Tool Calling Documentation. https://qwen.readthedocs.io/
