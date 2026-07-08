# Capítulo 25 — Spec-Driven Development para sistemas de LLM on-premise

Os capítulos anteriores construíram as peças: inferência, fine-tuning, RAG,
agentes, MLOps. Este capítulo trata do que os une — **como especificar, medir e
evoluir um sistema de LLM sem que ele vire um amontoado de experimentos**.

A tese: sistemas de LLM precisam de Spec-Driven Development (SDD) **mais** que
software convencional, porque o componente central é não-determinístico. Quando
o código é determinístico, o teste unitário é a spec executável. Quando o
núcleo é um modelo, a única spec executável possível é **um golden set com
métricas-alvo** — e todo o resto do método se reorganiza em torno disso.

## 25.1 O que é SDD — e o que muda com LLMs

SDD inverte a ordem tradicional: a especificação vem antes do código e
permanece como fonte de verdade viva. O ciclo canônico (spec-kit, do GitHub):

```
constituição → /specify (o QUÊ e o PORQUÊ) → /plan (o COMO) → /tasks → implementação
```

Variações do ecossistema — todas presentes no corpus do Capítulo 24, o que
permite estudá-las pelo código, não pelo marketing:

| Ferramenta | Ideia central | Quando encaixa |
|---|---|---|
| **spec-kit** | constituição + specify/plan/tasks; agente executa | projeto novo, time pequeno |
| **OpenSpec** | specs como *deltas* versionados (proposta → aprovação → arquivo) | sistema vivo, mudanças incrementais |
| **BMAD-METHOD** | personas ágeis (PM, arquiteto, dev) + story files | simular um time completo com agentes |
| **ralph / loop-engineering** | loop autônomo: agente roda até o PRD passar | tarefas bem especificadas, CI de agente |

Com LLMs, três cláusulas se tornam obrigatórias em qualquer variação:

1. **Critério de aceite = métrica em golden set**, não frase ("busca deve ser
   boa" → "hit@5 ≥ 0,90 nas 109 queries do golden v2").
2. **Contratos de dados explícitos** entre camadas (payload, embedder, dims) —
   porque a falha típica de LLM stack é silenciosa, não uma exceção.
3. **Registro de decisão com número** — cada escolha técnica referencia o A/B
   que a justificou, porque a intuição comprovadamente falha neste domínio
   (o Capítulo 24 refutou duas intuições razoáveis por medição).

## 25.2 Walkthrough: o ciclo spec-kit num projeto do zero

Nada explica SDD melhor que executá-lo. Projeto exemplo: **`ask-local`** — um
endpoint `/v1/ask` que responde perguntas sobre uma base de documentos usando
o RAG do Capítulo 14 e um modelo local do Capítulo 7. Pequeno o bastante para
caber numa seção; real o bastante para expor cada artefato do ciclo.

O spec-kit instala um conjunto de comandos para o agente de código
(`specify`, `clarify`, `plan`, `tasks`, `implement`, `analyze`, `checklist`,
`constitution` — a lista real de `templates/commands/`). O ciclo:

**Passo 0 — Constituição** (uma vez por projeto). Princípios inegociáveis que
TODAS as specs herdam. Para sistemas de LLM on-premise, a nossa teria:

```markdown
# constitution.md — ask-local
1. Runtime 100% on-premise; nenhuma chamada externa no caminho da resposta.
2. Toda meta de qualidade é métrica em golden set versionado.
3. Contratos de dados escritos antes do segundo consumidor existir.
4. Falha silenciosa é bug de arquitetura: mismatch/desync → erro explícito.
5. VRAM ≤ 12 GB; latência p95 ≤ 8 s.
```

**Passo 1 — `/specify`**: o QUÊ e o PORQUÊ, sem tecnologia. O template real do
spec-kit organiza por **user stories priorizadas e independentemente
testáveis** — cada uma um MVP viável sozinha:

```markdown
# Feature Specification: ask-local
**Status**: Draft

### User Story 1 - Pergunta com resposta fundamentada (Priority: P1)
Como engenheiro, pergunto em linguagem natural e recebo resposta com as
fontes (arquivo:linha) que a sustentam.
**Why this priority**: sem fundamentação verificável o produto não existe —
resposta sem fonte é chatbot, não ferramenta de engenharia.
**Independent Test**: 20 perguntas do golden; resposta cita ≥1 fonte correta.

**Acceptance Scenarios**:
1. **Given** corpus indexado, **When** pergunto "onde valida o token JWT?",
   **Then** resposta cita o arquivo real e a resposta menciona a função.
2. **Given** pergunta fora do corpus, **When** pergunto sobre clima,
   **Then** sistema responde "fora do escopo da base" (não alucina).

### User Story 2 - Latência interativa (Priority: P2)
...
```

Repare no cenário 2: **o comportamento fora de distribuição é especificado**
— com LLMs, o que o sistema NÃO deve responder é requisito de primeira classe
(é o guard OOD do Capítulo 20 nascendo como spec, não como remendo).

**Passo 2 — `/clarify`**: o agente interroga a spec procurando ambiguidade
("qual corpus? qual formato de fonte? o que conta como 'citar'?") e grava as
respostas na spec. Ambiguidade resolvida em texto custa minutos; resolvida em
código custa dias.

**Passo 3 — `/plan`**: o COMO — decisões técnicas com justificativa,
herdando a constituição. Aqui entram: embedder local (com o A/B do Capítulo
24 como precedente), funil de retrieval, modelo gerador quantizado, e o
**contrato do payload** entre indexador e servidor.

**Passo 4 — `/tasks`**: o plano vira tarefas pequenas, ordenadas por
dependência, cada uma com critério de done executável:

```markdown
- [ ] T01 golden set v1 (20 perguntas + fontes esperadas) — done: arquivo
      versionado, 3 revisões de pares
- [ ] T02 indexador + contrato payload — done: 100% chunks com fonte;
      teste de mismatch de embedder falha ruidosamente
- [ ] T03 retrieval — done: hit@5 ≥ 0,85 no golden v1
- [ ] T04 geração fundamentada — done: 90% das respostas citam fonte correta
- [ ] T05 recusa OOD — done: 10/10 perguntas fora do corpus recusadas
```

T01 é o golden — **primeira tarefa, não última**. É a inversão que define SDD
para LLMs: o instrumento de medição existe antes do sistema medido.

**Passo 5 — `/implement` (+ `/analyze`)**: o agente executa tarefa a tarefa,
rodando o critério de done de cada uma; `/analyze` audita consistência entre
spec, plano e código ao final. O loop da Seção 25.6 é este passo
industrializado.

O ciclo inteiro para o ask-local: ~1 hora de specify/clarify/plan/tasks antes
da primeira linha de código. O retorno: nenhuma discussão de escopo durante a
implementação — a discussão já aconteceu, com custo de texto.

## 25.3 OpenSpec vs BMAD: dois SDDs, trechos reais

As duas filosofias dominantes de SDD-com-agentes resolvem problemas
diferentes. Comparação com artefatos reais do corpus do Capítulo 24.

**OpenSpec: a unidade é o CHANGE (delta versionado).** Um diretório por
mudança — `proposal.md`, `design.md`, `tasks.md`, `specs/` — que nasce em
`changes/`, é aprovado, implementado e termina em `changes/archive/` com data.
Trecho real (`archive/2025-12-28-add-artifact-workflow-cli/proposal.md`):

```markdown
## Why
The ArtifactGraph (Slice 1) and InstructionLoader (Slice 3) provide
programmatic APIs [...]. Users currently have no CLI interface to:
- See artifact completion status for a change
- Discover what artifacts are ready to create [...]

## What Changes
- **NEW**: `openspec status --change <id>` shows artifact completion state
- **NEW**: `openspec next --change <id>` shows artifacts ready to create [...]

**Experimental isolation**: All commands are implemented in a single file
(`src/commands/artifact-workflow.ts`) for easy removal if the feature
doesn't work out. Help text marks them as experimental.
```

Três traços de maturidade nesse trecho real: o **Why antes do What** (proposta
sem justificativa não passa), o What como **lista verificável de superfícies
novas**, e a cláusula de **isolamento experimental** — a mudança já nasce
sabendo como ser removida. O arquivamento datado cria o histórico que
respondeu, no nosso caso real, "por que Qwen3 e não a API?" seis meses depois.

**BMAD: a unidade é a STORY, produzida por um TIME de personas.** O método
instala agentes nomeados com papéis ágeis — tabela real da documentação:

| Agente | Skill ID | Workflows principais |
|---|---|---|
| Analyst (Mary) | `bmad-agent-analyst` | Brainstorm, Market/Domain Research, Brief |
| Product Manager (John) | `bmad-agent-pm` | PRD, Epics & Stories, Correct Course |
| Architect (Winston) | `bmad-agent-architect` | Create Architecture, Implementation Readiness |
| Developer (Amelia) | `bmad-agent-dev` | Dev Story, Code Review, Sprint Planning |
| UX Designer (Sally) / Tech Writer (Paige) | ... | UX Design / Documentação |

O fluxo: Mary pesquisa → John escreve o PRD e o quebra em epics/stories →
Winston decide arquitetura → Amelia implementa story a story. O formato de
story do template real de PRD:

> *"As a [persona], I can [action] [under conditions].
> Acceptance: [testable criteria]."* — numeradas Story-1, Story-2, ...

E o template real de PRD traz a seção que mais previne desastre em projeto de
LLM — **non-goals explícitos**: "what this product is *not* [...] prevents the
'let me also add this nearby thing' failure mode at every level (epic, ticket,
code)". Scope creep em sistema de LLM não adiciona só código: adiciona
superfície de alucinação.

**Comparativo:**

| Dimensão | OpenSpec | BMAD |
|---|---|---|
| Unidade de trabalho | change (delta sobre sistema existente) | story (fatia de produto novo) |
| Pergunta central | "o que muda e por quê?" | "o que construir e para quem?" |
| Rastreabilidade | arquivo datado por mudança | PRD → epic → story → código |
| Cerimônia | mínima (3–4 arquivos por change) | alta (personas, workflows, PRD completo) |
| Brilha em | sistema VIVO evoluindo (nosso caso: KB #1→#12) | produto NOVO com escopo difuso |
| Risco típico | proposta rasa ("Why" de uma linha) | cerimônia sem medição (story sem métrica) |

**Escolha prática para sistemas de LLM**: greenfield com escopo claro →
spec-kit; greenfield com escopo difuso (precisa de descoberta) → BMAD;
sistema em evolução contínua → OpenSpec. O caso real deste livro usou, sem
saber o nome, o modelo OpenSpec: doze deltas aprovados, medidos e arquivados
num STATE.md. Em todos os três, as cláusulas da Seção 25.1 (golden set,
contratos, decisão com número) são o adaptador que os torna adequados a LLMs —
nenhum dos três as traz de fábrica.

## 25.4 Anatomia de uma spec para sistema de LLM

Cinco blocos, todos verificáveis. Abaixo, uma spec **real e completa** — a que
governou o incremento #7 do sistema do Capítulo 24:

```markdown
# SPEC #7 — Embedder v2 (retrieval de código)
**Status:** ativo · **Data:** 2026-07-08 · **Depende de:** #6 (baseline medido)

## 1. Objetivo mensurável
hit@5 ≥ 0,90 no golden set corrente (eval_retrieval.py --k 5).
Baseline a bater: 0,76 (MiniLM texto, medido em #6).

## 2. Abordagem e restrições
- Embedder de código local (candidato: Qwen3-Embedding-0.6B; nomic-embed-code
  7B DESCARTADO: não cabe em 12 GB de VRAM).
- Coleção PARALELA (llm_wiki_code) — a baseline llm_wiki não é destruída;
  A/B permanece possível.
- 100% on-premise em runtime. Re-embed pode usar GPU por horas (one-time).

## 3. Critérios de aceite (executáveis)
- [ ] 258.707 pontos na coleção nova (contagem Qdrant = view nodes_clean)
- [ ] eval hit@5 ≥ 0,90; resultado versionado em graphify-out/eval_*.json
- [ ] Query usa prompt instruct assimétrico (verificar em eval e search)
- [ ] Payload idêntico ao contrato SPEC_PAYLOAD.md (nenhum consumidor quebra)

## 4. Fora de escopo
Reranking, busca híbrida (specs futuras se a meta não for atingida).

## 5. Alavancas permitidas vs proibidas (anti-maquiagem)
PERMITIDO: trocar embedder, prompt de query, parâmetros de indexação.
PROIBIDO: editar golden p/ facilitar; aceitar repo no gabarito sem prova grep.

## Resultado (preenchido ao fechar)
hit@5 = 0,96 · MRR = 0,873 · eval_retrieval_20260708_092921.json
Decisão derivada: endurecer régua → spec #8 (golden ≥ 100 queries).
```

Repare no bloco 5, o mais incomum e o mais importante: ele declara **de
antemão** o que pode ser ajustado para bater a meta. Sem essa cláusula, a
pressão para fechar a spec corrompe a métrica — e uma métrica corrompida é
pior que nenhuma, porque autoriza decisões erradas com selo de "medido".

O bloco "Resultado" transforma a spec em registro histórico: specs fechadas
não se apagam, se **arquivam** (padrão OpenSpec). Seis meses depois, a pergunta
"por que usamos Qwen3 e não a API X?" tem resposta com número e data.

## 25.5 Caso real: a KB do Capítulo 24 como sequência de specs

O sistema GraphRAG do capítulo anterior não nasceu de um plano monolítico —
nasceu de **doze incrementos especificados** (#1 a #12 num STATE.md), cada um
com objetivo mensurável, implementação e resultado registrado no mesmo
documento:

| Incremento (spec) | Critério de aceite | Resultado |
|---|---|---|
| #1 índice SQLite | query estrutural < 100 ms | 14 s → 55 ms (255×) |
| #2 índice vetorial | payload conforme contrato; consulta via API existente | 258k pontos, testado |
| #6 acurácia medida | golden + eval versionado | baseline 0,76 estabelecido |
| #7 embedder v2 | hit@5 ≥ 0,90 sem destruir baseline | 0,96 (golden 25) |
| #8 hardening | golden ≥ 100 queries; guards com teste negativo | 0,927; mismatch → erro |
| #10 funil completo | cada estágio entra com delta medido | 0,982 |
| #11 GraphRAG | k-hop sem poluir top-k; PageRank com sanidade | verificado via API |
| #12 operação | serviço sobrevive a reboot; auth real testada | systemd + 401 |

Três mecânicas fazem esse método funcionar:

**STATE.md como spec viva + handoff.** Um único documento com estado, decisões,
comandos de reprodução e pendências — qualquer sessão (humana ou agente) retoma
o trabalho sem arqueologia. É a "constituição" do spec-kit fundida com o log de
deltas do OpenSpec. Estrutura mínima:

```markdown
# STATE — <sistema> (handoff resumível)
## O que é              ← 3 linhas, links p/ contratos
## Ambiente/dependências ← versões, caminhos, chaves (NUNCA o valor da chave)
## Etapas               ← specs #N: status, critério, resultado, comando
## Pendências           ← dívidas DECLARADAS, com contexto p/ retomar
## Decisões             ← cada uma com o número que a justificou
```

**Meta que resiste ao próprio teste.** Quando a meta de hit@5 foi batida no
golden de 25 queries (0,96), a spec seguinte **endureceu a régua** (golden 109)
em vez de declarar vitória — e o número caiu para 0,899, honestamente, antes de
subir de novo com trabalho real. Time que só mede quando espera boa notícia não
está medindo; está fazendo cerimônia.

**Dívida declarada em vez de descoberta.** As 82k arestas INFERRED sem
auditoria estatística, os 2 misses residuais do golden, a ausência de eval de
geração fim-a-fim — tudo listado em "Pendências" com contexto. O custo de
declarar é um parágrafo; o custo de descobrir em produção é um incidente.

## 25.6 SDD com agentes de código

SDD é o método natural para desenvolvimento com agentes (Claude Code, Codex e
afins), porque specs executáveis são exatamente o que um agente consegue
verificar sozinho:

```
# loop de agente sobre spec (padrão ralph/loop-engineering)
enquanto (critérios da spec não passam) e (iterações < N):
    ler STATE.md + spec corrente
    implementar o menor incremento que ataca o critério que falha
    rodar eval; registrar resultado no STATE.md
    se travado 2 iterações no mesmo erro: escalar para humano
```

Condições para o loop não degenerar — todas aprendidas na prática:

1. **Eval barato e determinístico** — o agente roda a cada iteração; eval de
   10 minutos mata o loop, eval flaky o envenena.
2. **Critério de parada numérico** — "até ficar bom" não termina; "hit@5 ≥
   0,90 ou 3 alavancas esgotadas com relatório honesto" termina.
3. **Estado externo ao contexto** (STATE.md) — sessões de agente são efêmeras;
   o projeto não pode ser. No caso real deste livro, um re-embed de 2 horas
   sobreviveu a três sessões diferentes graças a checkpoint + STATE.md.
4. **Fronteiras de autorização explícitas** — o que o agente decide sozinho
   (implementação, correção de bug no próprio loop) vs o que escala (mudança
   de escopo, custo externo, apagar dados, commit/push). É o HITL do Capítulo
   19 aplicado ao desenvolvimento, não só ao runtime.
5. **Exit codes honestos** — anti-padrão real deste projeto: um job em
   background com `; echo "exit=$?"` no final SEMPRE retorna 0; o embed morreu
   a 37% e o eval seguinte mediu uma coleção incompleta (0,606 — número
   inválido que quase virou conclusão). A spec de qualquer job longo inclui:
   exit code propagado, log em arquivo, critério de completude verificável
   (contagem esperada = contagem observada).

## 25.7 Meta-caso: treinar uma LLM para GERAR specs

Fechando o círculo: se specs são o artefato central e agentes as consomem, o
passo natural é treinar um modelo on-premise **especialista em escrevê-las**.
Este meta-caso exercita metade do livro (Capítulos 10, 11, 12, 14 e 24) e cabe
numa GPU de 12 GB.

**1. Dataset extraído da KB (Capítulo 24 como fonte).** Os repos spec-driven
contêm centenas de specs, PRDs e story files reais. O índice estrutural
localiza os arquivos; o par de treino é `pedido → spec` num **formato canônico
único** (misturar spec-kit com BMAD no mesmo dataset degrada o formato de
saída):

```jsonl
{"messages": [
  {"role": "user", "content": "Crie a spec para: rate limiting por API key
   no gateway, com burst configurável e resposta 429 padronizada"},
  {"role": "assistant", "content": "# SPEC — Rate limiting por API key\n
   ## 1. Objetivo mensurável\n- p99 do overhead do limiter < 2 ms ...\n
   ## 3. Critérios de aceite\n- [ ] 429 com Retry-After correto sob burst\n
   - [ ] contadores por key isolados (teste: 2 keys, 1 estoura, outra passa)..."}
]}
```

Ampliação por self-instruct: o modelo local gera variações de PEDIDOS sobre
templates reais (nunca variações de specs — o formato-alvo é sagrado). Alvo:
500–2.000 pares limpos, não 20.000 sujos. Curadoria segue a regra do golden:
cada par validado contra o checklist estrutural antes de entrar.

**2. QLoRA 4-bit sobre modelo de código de 7B** (Capítulo 11). O LoRA ensina o
**formato**: seções obrigatórias, IDs de requisito, critérios testáveis,
métricas-alvo. Conteúdo de domínio não se memoriza — vem do RAG.

**3. RAG na inferência** (Capítulos 14 e 24): a geração consulta a KB para
puxar templates e código do domínio do pedido. Divisão de trabalho: LoRA =
forma; retrieval = conteúdo; modelo-base = língua e raciocínio.

**4. RAFT (Retrieval-Augmented Fine-Tuning)** — o refinamento que liga treino
e inferência: treinar os pares **já com o contexto recuperado no prompt**,
incluindo ~20% de documentos distratores deliberados:

```
prompt de treino = pedido
                 + top-3 chunks REAIS do retrieval (relevantes)
                 + 1 chunk distrator (irrelevante, de outro domínio)
resposta-alvo    = spec canônica
```

Treino e inferência na mesma distribuição; o modelo aprende inclusive a
**ignorar** contexto irrelevante — a habilidade que separa RAG que funciona de
RAG que alucina com confiança.

**5. Eval estrutural como spec do próprio modelo.** Um validador programático
— determinístico, barato, rodável em loop de agente — checa cada spec gerada:

```python
# eval_spec.py (essência) — conformidade estrutural de specs geradas
CHECKS = [
    ("objetivo mensurável", r"##\s*1?\.?\s*Objetivo",           True),
    ("métrica numérica",    r"(≥|<=?|>=?)\s*[\d.,]+",           True),
    ("criterios de aceite", r"- \[ \] ",                         True),
    ("IDs de requisito",    r"\b(RF|RNF|REQ)-?\d+",              False),
    ("fora de escopo",      r"[Ff]ora de escopo",                True),
]
def score(spec_md: str) -> float:
    obrig = [c for c in CHECKS if c[2]]
    return sum(bool(re.search(rx, spec_md)) for _, rx, req in obrig
               ) / len(obrig)
# meta da spec do modelo: média ≥ 0,90 em N pedidos de teste held-out
```

SDD especificando o treinamento do modelo que escreve specs — o método validando
a si mesmo. Se a média não bater 0,90 após esgotar as alavancas declaradas
(mais dados, RAFT, prompt), o relatório diz onde parou e por quê — a mesma
honestidade exigida do retrieval no Capítulo 24.

## 25.8 Checklist do capítulo

- [ ] Toda meta de componente LLM expressa como métrica em golden set versionado.
- [ ] Contratos de dados escritos ANTES da segunda coleção/segundo consumidor existir.
- [ ] Um STATE.md por sistema: estado, decisões com números, comandos de reprodução, pendências declaradas.
- [ ] Specs como deltas incrementais; bloco "Resultado" preenchido ao fechar; specs fechadas arquivadas, nunca apagadas.
- [ ] Cláusula anti-maquiagem em cada spec: alavancas permitidas vs proibidas.
- [ ] Loop de agente só com eval determinístico + critério de parada numérico + fronteiras de autorização + exit codes honestos.
- [ ] Ao bater a meta, endurecer a régua antes de comemorar.
- [ ] Dataset de fine-tuning curado pela mesma disciplina do golden (prova antes de entrar; formato canônico único).

**Síntese**: em sistemas de LLM, a spec não é burocracia que precede o trabalho
— é o único instrumento que distingue progresso de ruído num domínio onde o
componente central é estocástico e as falhas são silenciosas. O time (ou o
agente) que não mede contra uma spec não sabe se melhorou; só sabe que mudou.
