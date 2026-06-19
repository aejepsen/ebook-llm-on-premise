# Capítulo 19 — Arquitetura Multi-Agente On-Premise

## Design patterns: router, orchestrator, hierarchical

Quando um sistema precisa de múltiplos agentes, a pergunta fundamental é: **quem decide o que?** Os três padrões principais:

### Router (roteador)

Um classificador direciona a pergunta para o agente especialista correto. Não há coordenação entre agentes — cada um resolve sua tarefa isoladamente.

```
Pergunta → [Router] → Agente A → Resposta
                     → Agente B → Resposta
                     → Agente C → Resposta
```

**Quando usar:** domínios independentes, uma pergunta aciona apenas um agente.

### Orchestrator (orquestrador)

Um nó central decompõe a pergunta, despacha para múltiplos agentes em paralelo, e sintetiza as respostas. É o padrão do AI-Orchestrator.

```
Pergunta → [Classificar] → [Despachar] → Agente A ─┐
                                        → Agente B ─┤→ [Sintetizar] → Resposta
                                        → Agente C ─┘
```

**Quando usar:** perguntas podem envolver múltiplos domínios, resposta precisa ser unificada.

### Hierarchical (hierárquico)

Agentes supervisionam outros agentes. Um "gerente" distribui tarefas para "trabalhadores" e avalia os resultados.

```
Pergunta → [Gerente] → [Trabalhador A] → resultado
                      → [Trabalhador B] → resultado
                      → [Gerente avalia] → (satisfatório? → Resposta)
                                         → (insatisfatório? → redistribui)
```

**Quando usar:** tarefas complexas que exigem iteração e qualidade garantida.

### Comparativo

| Padrão | Complexidade | Latência | Controle | Caso de uso |
|--------|-------------|----------|----------|-------------|
| Router | Baixa | Baixa | Baixo | FAQ, suporte L1 |
| Orchestrator | Média | Média | Alto | Sistemas corporativos |
| Hierarchical | Alta | Alta | Muito alto | Pesquisa, análise profunda |

O AI-Orchestrator usa o padrão **Orchestrator** com elementos de **Router**: o classificador pode rotear para 1 ou N agentes. Se a pergunta é simples (1 domínio), comporta-se como router. Se é complexa (N domínios), faz fan-out/fan-in.

## LangGraph StateGraph: construindo o grafo

O StateGraph do LangGraph é a espinha dorsal da orquestração. Cada nó é uma função que recebe e retorna estado. As arestas definem o fluxo.

### Definição do estado

```python
# estado_grafo.py
# Definição do estado tipado do grafo multi-agente

from typing import TypedDict

class EstadoGrafo(TypedDict):
    """Estado compartilhado entre todos os nós do grafo.

    REGRA CRÍTICA: apenas tipos serializáveis.
    Callbacks, funções e objetos complexos NUNCA entram no estado.
    O checkpointer serializa tudo via msgpack — funções não serializam.
    """
    # Entrada
    question: str                    # pergunta original
    sanitized: str                   # pergunta após sanitização
    trace_id: str                    # ID único da requisição

    # Classificação
    route: dict                      # {domains: [...], plan: "...", clarification: "..."}

    # Resultados
    agent_results: dict              # {dominio: "resposta do agente"}
    final_answer: str                # resposta sintetizada
    error: str | None                # erro, se houver

    # Conversação
    history: list[dict]              # histórico de mensagens

    # HITL
    pending_confirmation: dict | None  # dados pendentes de aprovação humana
```

### Armadilha: estado residual entre turns

Com checkpointer ativo, campos como `final_answer` e `agent_results` persistem entre turnos de conversa. Se o grafo tem uma aresta condicional que verifica `if state["final_answer"]`, ela será enganada por valores do turno anterior.

A solução usada no AI-Orchestrator: o primeiro nó (`_sanitize`) **zera todos os campos de resultado** a cada novo turno.

```python
# limpar_estado.py
# Primeiro nó do grafo: limpa estado residual

def sanitize(estado: EstadoGrafo) -> EstadoGrafo:
    """Sanitiza a entrada e limpa estado residual de turns anteriores."""
    return {
        "sanitized": sanitize_question(estado["question"]),
        "final_answer": "",        # limpa
        "agent_results": {},       # limpa
        "route": {},               # limpa
        "error": None,             # limpa
        "pending_confirmation": None,  # limpa
    }
```

## O framework ADD: Model vs Harness

Antes de mergulhar na implementação, é essencial entender uma distinção conceitual que estrutura todo o AI-Orchestrator: a separação entre **Model** e **Harness**, vinda do framework Agent-Driven Design (ADD).

| Conceito | O que é | Responsabilidade | Exemplo no AI-Orchestrator |
|----------|---------|------------------|----------------------------|
| **Model** | O LLM (Qwen, Llama, etc.) | Raciocínio, geração, julgamento | `gateway/llm.py` — OllamaClient |
| **Harness** | Todo o código ao redor | Prompts, tools, routing, validação, controle de fluxo | `gateway/graph.py`, `router.py`, `agents.py`, `sanitize.py` |
| **Agente** | Model + Harness juntos | Executar tarefas com ferramentas | DomainAgentRunner (finanças, RH, estoque, vendas) |

**Regra de ouro do ADD:** "Does this require reasoning/judgment?" → Model. "Is this structure, flow, or contract?" → Harness.

O AI-Orchestrator implementa esta separação rigorosamente:
- O **Model** nunca decide qual ferramenta é segura — o **Harness** (ToolRegistry) impõe escopo por domínio
- O **Model** nunca escreve em memória persistente — o **Harness** (SQLite checkpointer) gerencia estado
- O **Model** nunca define rotas — o **Harness** (semantic router → LLM classifier → lexical fallback) controla o fluxo

### As 4 topologias de agentes

O ADD cataloga 4 topologias. O AI-Orchestrator usa duas:

1. **Pipeline:** sanitize → classify → dispatch → synthesize (sequencial, cada etapa tem contrato explícito)
2. **Specialist Pool:** router classifica o domínio → dispatcher ativa o agente especialista (finanças OU rh OU estoque OU vendas)

**Topologias NÃO usadas** (e por quê):
3. **Hierarchical (Orchestrator + Workers):** overkill para 4 domínios com escopos bem definidos
4. **Event-Driven (Mesh):** complexidade de debug não justificada em sistema single-node

### Por que 4 agentes separados?

O ADD define 5 gatilhos para decompor um agente monolítico em múltiplos. O AI-Orchestrator usou dois:

- **Domain separation:** cada domínio tem vocabulário, invariantes e tool surfaces diferentes. O agente de finanças não precisa saber sobre férias (RH).
- **Fault isolation:** se o serviço de RH cair, o circuit breaker isola o domínio — vendas continua funcionando.

> **Princípio ADD:** "Start with a single agent. Decompose only when a concrete problem exists." O AI-Orchestrator começou com um agente monolítico. A decomposição em 4 veio quando tool calls cruzados começaram a confundir o modelo.

## Classificador de intenção: LLM + semântico + léxico

O classificador decide quais agentes ativar. No AI-Orchestrator, opera em cascata:

```
Pergunta → [Semântico] → (confiante? → rota)
                        → (incerto? ↓)
         → [LLM]       → (classificou? → rota)
                        → (ambíguo? ↓)
         → [Léxico]    → rota fallback
```

### Camada 1: Semântico (Qdrant + embeddings)

A pergunta é vetorizada e comparada com exemplos rotulados do golden set. Se o top-1 tem score acima do threshold **e** há consenso de domínios no top-k, a rota é aceita.

```python
# classificador_semantico.py
# Classificação por similaridade semântica (primeira camada)

import httpx

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
THRESHOLD = 0.85
TOP_K = 5

def classificar_semantico(pergunta: str) -> dict | None:
    """Classifica por similaridade com golden set.

    Retorna rota se confiante, None se incerto (fallback para LLM).
    """
    # Vetoriza a pergunta
    vetor = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "nomic-embed-text", "input": pergunta},
        timeout=30.0,
    ).json()["embeddings"][0]

    # Busca no golden set
    resultados = httpx.post(
        f"{QDRANT_URL}/collections/routing_examples/points/search",
        json={"vector": vetor, "limit": TOP_K, "with_payload": True},
    ).json()["result"]

    if not resultados:
        return None

    # Verifica threshold
    top1 = resultados[0]
    if top1["score"] < THRESHOLD:
        return None

    # Verifica consenso: todos os hits confiantes apontam o mesmo domínio?
    confiantes = [r for r in resultados if r["score"] >= THRESHOLD]
    dominios_top = tuple(top1["payload"]["domains"])

    for hit in confiantes:
        if tuple(hit["payload"]["domains"]) != dominios_top:
            return None  # sem consenso → LLM decide

    return {
        "domains": list(dominios_top),
        "plan": f'Roteado por similaridade (score {top1["score"]:.2f})',
        "clarification": None,
    }
```

### Camada 2: LLM (classificação por prompt)

Se a camada semântica não tem certeza, o LLM classifica. Recebe a pergunta e a lista de domínios disponíveis, retorna quais domínios ativar.

### Camada 3: Léxico (fallback determinístico)

Palavras-chave simples como última rede de segurança. "estoque" → domínio estoque. "funcionário" → domínio RH. Robusto e rápido.

### Lição aprendida no AI-Orchestrator

A camada semântica com threshold 0.92 e consenso restritivo resultou em **0 acionamentos** — o LLM resolvia em 1.6s. A camada semântica só compensa se o LLM de routing é lento ou caro. Meça antes de tunar.

## Fan-out/Fan-in: paralelizando agentes

### Fan-out

Quando o classificador identifica múltiplos domínios, os agentes são despachados em paralelo.

```python
# fan_out_fan_in.py
# Despacho paralelo de agentes com ThreadPoolExecutor

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass
class ResultadoAgente:
    dominio: str
    resposta: str
    latencia_s: float
    ferramentas_usadas: list[str]

def despachar_agentes(
    dominios: list[str],
    pergunta: str,
    timeout_total: float = 60.0,
) -> dict[str, ResultadoAgente]:
    """Fan-out: executa agentes em paralelo.

    ThreadPoolExecutor paraleliza as chamadas. Mesmo com GPU única
    (Ollama serializa gerações), o paralelismo beneficia:
    - Chamadas HTTP aos microsserviços (I/O bound)
    - Preparação futura para múltiplas GPUs
    - Fan-out/fan-in correto por design
    """
    def executar(dominio: str) -> tuple[str, ResultadoAgente]:
        inicio = time.monotonic()
        resultado = run_domain_agent(dominio, pergunta, deadline_s=timeout_total)
        return dominio, ResultadoAgente(
            dominio=dominio,
            resposta=resultado.final_answer,
            latencia_s=round(time.monotonic() - inicio, 2),
            ferramentas_usadas=[t.name for t in resultado.tool_trace],
        )

    with ThreadPoolExecutor(max_workers=len(dominios)) as pool:
        resultados = dict(pool.map(executar, dominios))

    return resultados
```

### Fan-in (síntese)

```python
# sintese.py
# Síntese de respostas de múltiplos agentes

PROMPT_SINTESE = """Você recebeu as respostas de {n} agentes especialistas
que responderam à pergunta do usuário, cada um no seu domínio.
Sintetize UMA resposta final em português, curta e direta, fundamentada
EXCLUSIVAMENTE nas respostas dos agentes — nunca invente números ou fatos
que não estejam nelas.
Se algum agente reportou impedimento ou erro, reflita isso na resposta final."""

def sintetizar(pergunta: str, resultados: dict[str, str]) -> str:
    """Fan-in: sintetiza respostas de múltiplos agentes.

    Com 1 domínio: retorna a resposta diretamente (zero chamada extra ao LLM).
    Com N domínios: LLM sintetiza.
    """
    if len(resultados) == 1:
        # Otimização: sem síntese para single-agent
        return next(iter(resultados.values()))

    # Monta contexto com respostas dos agentes
    respostas_formatadas = "\n\n".join(
        f"[{dominio}]: {resposta}"
        for dominio, resposta in resultados.items()
    )

    prompt = f"""{PROMPT_SINTESE.format(n=len(resultados))}

Respostas dos agentes:
{respostas_formatadas}

Pergunta original: {pergunta}"""

    # Chama LLM para sintetizar
    resposta = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODELO,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=60.0,
    ).json()
    return resposta["message"]["content"]
```

## Tool Registry: descoberta automática de ferramentas via OpenAPI

Cada microsserviço expõe um schema OpenAPI. O `ToolRegistry` lê esses schemas e converte cada endpoint em uma ferramenta que o modelo pode chamar. Isso significa que adicionar uma ferramenta é adicionar um endpoint — sem tocar no código do agente.

```python
# tool_registry.py
# Descoberta automática de ferramentas via OpenAPI

import httpx
from dataclasses import dataclass

@dataclass
class Ferramenta:
    nome: str
    descricao: str
    parametros: dict
    endpoint: str
    metodo: str

class RegistroFerramentas:
    """Descobre e gerencia ferramentas a partir de schemas OpenAPI."""

    def __init__(self, urls_servicos: dict[str, str], timeout_s: float = 10.0):
        """
        Args:
            urls_servicos: {dominio: url_base} ex: {"estoque": "http://localhost:8001"}
        """
        self._urls = urls_servicos
        self._timeout = timeout_s
        self._ferramentas: dict[str, list[Ferramenta]] = {}
        self._carregar_todas()

    def _carregar_todas(self):
        """Carrega schemas OpenAPI de todos os serviços."""
        for dominio, url in self._urls.items():
            try:
                resp = httpx.get(f"{url}/openapi.json", timeout=self._timeout)
                resp.raise_for_status()
                schema = resp.json()
                self._ferramentas[dominio] = self._extrair_ferramentas(schema, url)
            except httpx.HTTPError as e:
                print(f"Erro ao carregar schema de {dominio}: {e}")
                self._ferramentas[dominio] = []

    def _extrair_ferramentas(self, schema: dict, url_base: str) -> list[Ferramenta]:
        """Converte endpoints OpenAPI em ferramentas para o modelo."""
        ferramentas = []
        for caminho, metodos in schema.get("paths", {}).items():
            for metodo, detalhes in metodos.items():
                if metodo not in ("get", "post", "put", "delete"):
                    continue
                ferramentas.append(Ferramenta(
                    nome=detalhes.get("operationId", f"{metodo}_{caminho}"),
                    descricao=detalhes.get("summary", ""),
                    parametros=self._extrair_parametros(detalhes, schema),
                    endpoint=f"{url_base}{caminho}",
                    metodo=metodo.upper(),
                ))
        return ferramentas

    def _extrair_parametros(self, detalhes: dict, schema: dict) -> dict:
        """Extrai schema de parâmetros para formato de tool calling."""
        params = {"type": "object", "properties": {}, "required": []}
        for param in detalhes.get("parameters", []):
            nome = param["name"]
            params["properties"][nome] = {
                "type": param.get("schema", {}).get("type", "string"),
                "description": param.get("description", ""),
            }
            if param.get("required"):
                params["required"].append(nome)
        return params

    def ferramentas_para(self, dominio: str) -> list[dict]:
        """Retorna ferramentas em formato de tool calling do Ollama."""
        return [
            {
                "type": "function",
                "function": {
                    "name": f.nome,
                    "description": f.descricao,
                    "parameters": f.parametros,
                },
            }
            for f in self._ferramentas.get(dominio, [])
        ]

    def executar(self, dominio: str, nome: str, argumentos: dict) -> dict:
        """Executa uma ferramenta e retorna o resultado."""
        ferramentas = self._ferramentas.get(dominio, [])
        ferramenta = next((f for f in ferramentas if f.nome == nome), None)

        if not ferramenta:
            nomes = [f.nome for f in ferramentas]
            return {
                "status": 0,
                "body": {
                    "error": "unknown_tool",
                    "detail": f"'{nome}' não existe. Disponíveis: {', '.join(nomes)}.",
                },
            }

        try:
            resp = httpx.request(
                ferramenta.metodo,
                ferramenta.endpoint,
                params=argumentos if ferramenta.metodo == "GET" else None,
                json=argumentos if ferramenta.metodo != "GET" else None,
                timeout=self._timeout,
            )
            return {"status": resp.status_code, "body": resp.json()}
        except httpx.HTTPError as e:
            return {"status": 0, "body": {"error": str(e)}}
```

## Estado conversacional: checkpointer, threading

Para manter contexto entre turnos de conversa, o LangGraph usa **checkpointers** — mecanismos de persistência de estado.

```python
# estado_conversacional.py
# Configuração de checkpointer para estado entre turnos

from langgraph.graph import StateGraph, END

# Opção 1: MemorySaver (in-memory, desenvolvimento)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# Opção 2: SqliteSaver (persistente, produção single-node)
# from langgraph.checkpoint.sqlite import SqliteSaver
# checkpointer = SqliteSaver.from_conn_string("gateway_threads.db")

# Compilar o grafo com checkpointer
grafo = StateGraph(EstadoGrafo)
# ... adicionar nós e arestas ...
app = grafo.compile(checkpointer=checkpointer)

# Invocar com thread_id para manter conversa
config = {"configurable": {"thread_id": "usuario_123"}}

# Turno 1
resultado1 = app.invoke({"question": "Quantos funcionários há no RH?"}, config)

# Turno 2 (mesmo thread_id = mesma conversa)
resultado2 = app.invoke({"question": "E no departamento de vendas?"}, config)
# O modelo tem contexto do turno anterior
```

## HITL (Human-in-the-Loop): quando pausar para aprovação

Nem toda ação deve ser automática. Operações de escrita (criar funcionário, aprovar despesa, alterar estoque) devem pausar para aprovação humana.

```python
# hitl.py
# Human-in-the-Loop com interrupt() do LangGraph

from langgraph.types import interrupt

def confirmar_despacho(estado: EstadoGrafo) -> EstadoGrafo:
    """Pausa o grafo para aprovação humana antes do despacho.

    O interrupt() suspende a execução. O frontend exibe os domínios
    e o plano ao usuário. Após aprovação, o grafo retoma.
    """
    dominios = estado["route"]["domains"]
    plano = estado["route"]["plan"]

    # Decisão: auto-aprovar ou pausar?
    # No AI-Orchestrator, HITL é desabilitado por padrão e ativado
    # apenas quando há detecção confiável de write intent.
    if not precisa_confirmacao(dominios, plano):
        return {}  # auto-aprovado, continua o fluxo

    # Pausa para aprovação humana
    aprovacao = interrupt({
        "tipo": "confirmacao_despacho",
        "dominios": dominios,
        "plano": plano,
        "mensagem": f"Deseja prosseguir com {', '.join(dominios)}?",
    })

    if not aprovacao.get("aprovado"):
        return {"final_answer": "Operação cancelada pelo usuário."}

    return {}

def precisa_confirmacao(dominios: list[str], plano: str) -> bool:
    """Heurística: operações de escrita precisam de confirmação."""
    palavras_escrita = ["criar", "incluir", "atualizar", "excluir", "deletar", "aprovar"]
    plano_lower = plano.lower()
    return any(p in plano_lower for p in palavras_escrita)
```

### Armadilha: null payload no interrupt

Quando `interrupt()` suspende o grafo, o stream pode yieldar payloads `None`. O código que processa o stream deve tratar isso:

```python
# stream_com_interrupt.py
# Processamento seguro do stream com interrupt

for evento in app.stream(entrada, config):
    if not evento:  # payload None do interrupt
        continue
    if "final_answer" in evento:
        print(evento["final_answer"])
```

## Caso real completo: AI-Orchestrator

O AI-Orchestrator implementa o pipeline completo:

```
sanitize → classify → confirm_dispatch → dispatch → synthesize
```

### Sanitize

Remove tokens especiais ChatML (`<|im_start|>`, `<|endoftext|>`) e tags de wrapper (`</user_question>`, `<plan>`) que poderiam permitir escape estrutural. Detecta (sem bloquear) 14 padrões de injection semântica.

### Classify

Cascata semântico → LLM → léxico. Retorna: domínios a ativar, plano de execução, ou pedido de clarificação.

### Confirm dispatch

HITL opcional. Se configurado, pausa para aprovação antes do despacho.

### Dispatch

Fan-out via `ThreadPoolExecutor`. Cada agente recebe system prompt escopado + ferramentas do seu domínio. Loop ReAct com `max_iters=6` + `deadline_s`.

### Synthesize

Single-domain: resposta direta. Multi-domain: LLM sintetiza, fundamentado exclusivamente nas respostas dos agentes.

## System prompt engineering: anti-fabricação e regras de negócio

### Anti-fabricação

```
"Nunca invente valores, IDs, SKUs ou totais."
"Para operações de escrita sem todos os campos obrigatórios,
liste os campos e peça os valores — nunca fabrique."
```

Essa regra nasceu de um incidente real: sem ela, o agente criava funcionários com nomes inventados.

### Regras de negócio na API

O desconto máximo não é no prompt — é na API (HTTP 422). O modelo recebe o erro e informa o usuário. O prompt diz "use as ferramentas"; a API diz "desconto máximo é 15%". Separação de responsabilidades.

### Case-sensitivity

LLMs enviam parâmetros em lowercase. Se o banco usa case-sensitive, a query retorna vazio. Todo filtro textual deve usar `COLLATE NOCASE` (SQLite) ou `ILIKE` (Postgres).

---

## Resiliência: Circuit Breaker por domínio

Em produção, microsserviços falham. Se o serviço de RH estiver offline e o gateway continuar enviando requests, o sistema inteiro degrada. O AI-Orchestrator usa **Circuit Breaker** — padrão clássico de resiliência adaptado para agentes:

```python
# gateway/tools/circuit.py — Circuit Breaker por domínio
import time
from enum import Enum

class State(Enum):
    CLOSED = "closed"       # operação normal
    OPEN = "open"           # bloqueado — falhas recentes
    HALF_OPEN = "half_open" # testando recuperação

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown: float = 30.0):
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._state = State.CLOSED
        self._failures = 0
        self._last_failure = 0.0

    def call(self, func, *args, **kwargs):
        if self._state == State.OPEN:
            if time.monotonic() - self._last_failure > self._cooldown:
                self._state = State.HALF_OPEN  # testa recuperação
            else:
                raise CircuitOpenError("Circuito aberto — domínio indisponível")

        try:
            result = func(*args, **kwargs)
            if self._state == State.HALF_OPEN:
                self._state = State.CLOSED  # recuperado!
            self._failures = 0
            return result
        except Exception as e:
            self._failures += 1
            self._last_failure = time.monotonic()
            if self._failures >= self._threshold:
                self._state = State.OPEN
            raise e
```

**Regras do AI-Orchestrator:**
- **3 falhas de transporte** (timeout, connection refused) → circuito ABERTO por 30s
- **Erros 4xx NÃO contam** — são erros de negócio (produto não encontrado, saldo insuficiente), não de infraestrutura
- **Half-open:** após 30s, o próximo request serve como teste — se passar, fecha o circuito

Cada domínio (finanças, RH, estoque, vendas) tem seu próprio breaker independente. Se o RH cair, vendas continua funcionando normalmente.

## Resumo

| Componente | Decisão |
|------------|---------|
| Padrão | Orchestrator (fan-out/fan-in) |
| Grafo | LangGraph StateGraph |
| Classificação | Semântico → LLM → léxico |
| Paralelismo | ThreadPoolExecutor |
| Tool discovery | OpenAPI automático |
| Estado | Checkpointer (Memory/SQLite) |
| HITL | Interrupt() para write ops |
| Anti-fabricação | System prompt + API validation |

---

## Referências

- LangGraph Documentation. *StateGraph, Checkpointing, Interrupt*. https://langchain-ai.github.io/langgraph/
- Projeto AI-Orchestrator — `gateway/graph.py` (grafo completo: sanitize → classify → dispatch → synthesize), `gateway/agents.py` (loop ReAct com tool calling), `gateway/semantic_router.py` (classificação em cascata).
- *Hands-on LLM-based Agents* (PDF) — Arquiteturas multi-agente e design patterns.
- Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. Microsoft Research.
- SKILL_MULTIAGENT.md — Fases 0–7 de construção multi-agente, anti-padrões, gotchas.
