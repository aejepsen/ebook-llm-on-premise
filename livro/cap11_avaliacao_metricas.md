# Capítulo 11 — Avaliação e Métricas de Modelos Fine-Tuned

## Métricas de avaliação: perplexidade, acurácia, F1

Avaliar um modelo fine-tuned é tão importante quanto treiná-lo. Sem avaliação rigorosa, você não sabe se o fine-tuning melhorou, piorou ou não fez diferença. Existem métricas gerais (aplicáveis a qualquer modelo) e métricas específicas da tarefa.

### Perplexidade (PPL)

A perplexidade mede a "surpresa" do modelo diante dos dados de validação. Quanto menor, melhor — significa que o modelo prevê os tokens com mais confiança.

```
PPL = exp(loss médio nos dados de validação)
```

Para o AI-Orchestrator:
- Val loss = 0.089 → PPL = exp(0.089) = **1.093**
- Isso é extremamente baixo, indicando que o modelo aprendeu muito bem os padrões do dataset.

**Limitação**: PPL baixa não garante que o modelo faz a coisa certa em produção. Um modelo pode ter PPL baixíssima no val set e ainda errar em casos reais que não estão no dataset.

### Acurácia

Proporção de respostas corretas sobre o total de perguntas. Simples e direto para tarefas de classificação.

```python
# Acurácia de roteamento
# O modelo deve classificar a pergunta nos domínios corretos
acertos = 0
total = 0
for caso in golden_routing:
    predicao = modelo.classificar(caso["pergunta"])
    if predicao.domains == caso["domains_esperados"]:
        acertos += 1
    total += 1

acuracia = acertos / total
print(f"Routing accuracy: {acuracia:.1%}")
# AI-Orchestrator LoRA: 90.9%
```

### F1-Score

Combinação de precisão e recall, especialmente útil quando as classes são desbalanceadas:

```
Precisão = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)
Recall   = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)
F1       = 2 × (Precisão × Recall) / (Precisão + Recall)
```

Para avaliação multi-classe (como roteamento em 4 domínios), use F1 macro (média simples entre classes) ou F1 micro (pondera pelo tamanho de cada classe).

---

## Avaliação task-specific vs geral

### Avaliação task-specific

Testa exatamente o que o modelo precisa fazer em produção. No AI-Orchestrator, três avaliações específicas:

1. **Routing**: o modelo classifica corretamente as perguntas nos domínios?
2. **Injection**: o modelo resiste a tentativas de prompt injection?
3. **Domains**: o modelo executa tool-calling corretamente em cada domínio?

Essas avaliações usam o **golden set** — exemplos de teste criados manualmente, com respostas verificadas, que **nunca** entraram no dataset de treino.

### Avaliação geral

Testa capacidades amplas do modelo (raciocínio, conhecimento, linguagem) usando benchmarks padronizados. Útil para verificar se o fine-tuning não causou **catastrophic forgetting** — quando o modelo perde capacidades gerais ao se especializar.

> **Regra prática**: se o modelo fine-tuned perde significativamente em benchmarks gerais (>5% de queda), o fine-tuning foi agressivo demais. Considere reduzir epochs, rank ou dataset.

---

## Benchmarks: MMLU, HellaSwag, ARC

Benchmarks padronizados permitem comparar modelos entre si e verificar degradação.

### MMLU (Massive Multitask Language Understanding)

Testa conhecimento em 57 disciplinas (matemática, história, medicina, direito). O modelo responde perguntas de múltipla escolha.

```
Pergunta: Qual é a capital da Austrália?
A) Sydney  B) Melbourne  C) Canberra  D) Brisbane
Resposta correta: C
```

Scores típicos:
- GPT-4: ~86%
- Llama 3.3 70B: ~82%
- Qwen3.5-9B: ~72%
- Modelos 7B: ~60-65%

### HellaSwag

Testa compreensão de linguagem e senso comum. O modelo deve completar frases de forma natural.

### ARC (AI2 Reasoning Challenge)

Testa raciocínio científico com perguntas de nível escolar.

### Como rodar benchmarks

A forma mais prática é usar o **lm-evaluation-harness** do EleutherAI:

```bash
# Instalação
pip install lm-eval

# Rodar MMLU no modelo local via Ollama
# (requer adapter para API local — ou use o modelo HF direto)
lm_eval --model hf \
    --model_args pretrained=./merged_model \
    --tasks mmlu \
    --batch_size 8
```

> **No AI-Orchestrator**: não rodamos benchmarks gerais porque o objetivo era estritamente task-specific (routing + tool-calling). O golden set de 44 perguntas de routing + 40 tasks de domínio era suficiente para validar o modelo na tarefa real.

---

## Avaliação com LLM-as-Judge

Quando a resposta correta não é um valor exato (como em geração de texto livre), um LLM mais poderoso pode atuar como "juiz" avaliando a qualidade da resposta.

### Como funciona

```python
# Prompt para o juiz (modelo maior, ex.: Claude ou GPT-4)
JUDGE_PROMPT = """Avalie a resposta do assistente para a pergunta abaixo.

Pergunta: {pergunta}
Resposta esperada: {resposta_esperada}
Resposta do assistente: {resposta_modelo}

Critérios:
1. Correção factual (0-10): os fatos estão corretos?
2. Completude (0-10): a resposta aborda tudo que foi perguntado?
3. Clareza (0-10): a resposta é clara e bem estruturada?
4. Fundamentação (0-10): a resposta cita fontes/dados quando relevante?

Responda em JSON: {{"correcao": N, "completude": N, "clareza": N,
"fundamentacao": N, "nota_final": N, "justificativa": "..."}}"""
```

### Vantagens e limitações

| Aspecto | Vantagem | Limitação |
|---------|----------|-----------|
| Escalabilidade | Avalia milhares de respostas automaticamente | Custo da API do modelo juiz |
| Subjetividade | Mais nuançado que métricas numéricas | Bias do modelo juiz |
| Reprodutibilidade | Determinístico com temperature=0 | Pode mudar entre versões do modelo |

### Quando usar

- Geração de texto livre (resumos, explicações, redação)
- Avaliação de qualidade de diálogos multi-turno
- Comparação A/B entre modelos quando não há resposta "certa"

### Quando NÃO usar

- Tarefas com resposta exata (classificação, routing) — use acurácia
- Tarefas numéricas — use erro absoluto/relativo
- Quando o modelo juiz é menor ou igual ao modelo avaliado

---

## Comparando modelo base vs fine-tuned

A comparação direta é o teste mais importante. Rode o mesmo golden set nos dois modelos e compare:

### Framework de comparação

```python
import json
import time

def avaliar_modelo(modelo_nome, golden_set):
    """Avalia um modelo contra o golden set completo."""
    resultados = {
        "modelo": modelo_nome,
        "acertos": 0,
        "erros": 0,
        "latencia_media_ms": 0,
        "detalhes": []
    }

    latencias = []
    for caso in golden_set:
        inicio = time.monotonic()
        predicao = classificar(modelo_nome, caso["pergunta"])
        latencia = (time.monotonic() - inicio) * 1000
        latencias.append(latencia)

        correto = predicao == caso["esperado"]
        resultados["acertos" if correto else "erros"] += 1
        resultados["detalhes"].append({
            "pergunta": caso["pergunta"],
            "esperado": caso["esperado"],
            "obtido": predicao,
            "correto": correto,
            "latencia_ms": round(latencia, 1),
        })

    total = resultados["acertos"] + resultados["erros"]
    resultados["acuracia"] = resultados["acertos"] / total
    resultados["latencia_media_ms"] = sum(latencias) / len(latencias)
    return resultados

# Compara os três modelos
for modelo in ["qwen3:7b", "qwen3.5:9b", "qwen3.5-9b-orch"]:
    r = avaliar_modelo(modelo, golden_routing)
    print(f"{modelo}: {r['acuracia']:.1%} | {r['latencia_media_ms']:.0f}ms")
```

### Resultados reais — AI-Orchestrator

| Modelo | Routing | Injection | Domains | Latência média |
|--------|---------|-----------|---------|---------------|
| Qwen3 7B (base) | 90.5% | 0/6 | 82.5% | ~800ms |
| Qwen3.5 9B (base) | 95.5% | 0/6 | 87.5% | ~1200ms |
| Qwen3.5 9B LoRA | 90.9% | 0/6 | 87.5% | ~1200ms |

Observações:
- O modelo LoRA manteve a acurácia de domínios (87.5%) com tool-calling mais consistente
- Routing caiu de 95.5% para 90.9% — trade-off aceitável porque o dataset de routing do professor (30B) tinha ~7% de erro, herdado pelo aluno
- Injection manteve 0 leaks — o treino com 156 exemplos de injection sintética funcionou

---

## Caso real: eval de roteamento (accuracy, latency)

O AI-Orchestrator define **gates** — critérios mínimos que o modelo deve atingir para ser promovido a produção:

### Gates de qualidade

```
Gate 1 — Routing:   >= 90% de acurácia
Gate 2 — Injection: 0 leaks (zero tolerância)
Gate 3 — Domains:   >= 80% por domínio (finanças, RH, estoque, vendas)
```

### Como os evals funcionam

Cada eval é um script Python independente que:
1. Carrega o golden set (perguntas com respostas esperadas)
2. Envia cada pergunta ao modelo via Ollama
3. Compara a resposta com o esperado
4. Imprime relatório com acurácia, erros detalhados e latência

```bash
# Rodar os 3 gates (do diretório raiz do projeto)
MODEL=qwen3.5-9b-orch python evals/eval_routing.py
MODEL=qwen3.5-9b-orch python evals/eval_injection.py
MODEL=qwen3.5-9b-orch python evals/eval_domains.py
```

### Critério de adoção

O modelo LoRA é promovido para produção **apenas se**:
1. Passa em todos os 3 gates
2. Não regride em nenhuma métrica comparado ao modelo base
3. Resultados são **consistentes** em pelo menos 2 runs (eliminando variância da geração)

> **No AI-Orchestrator**: o modelo LoRA foi avaliado em 2 runs consecutivos com resultados idênticos (90.9%, 0/6, 87.5%). Passou nos 3 gates. Foi promovido para produção com `.env MODEL=qwen3.5-9b-orch`.

### Watchdog de avaliação

Cada caso de teste roda com um **deadline** (timeout). Se o modelo travar ou demorar demais, o caso é marcado como falha:

```python
from common import with_deadline, CaseTimeout

DEFAULT_CASE_DEADLINE_S = 60  # 60 segundos por caso

try:
    resultado = with_deadline(
        classificar_intent,  # função a executar
        pergunta,            # argumento
        llm,                 # cliente Ollama
        deadline_s=DEFAULT_CASE_DEADLINE_S
    )
except CaseTimeout:
    print(f"TIMEOUT: {pergunta[:50]}...")
    erros += 1
```

Isso evita que um caso travado segure toda a avaliação — o pipeline continua e reporta o timeout como erro.

---

## Golden set: construindo conjunto de teste confiável

O golden set é a pedra angular da avaliação. Se o golden set for ruim, toda avaliação é inútil.

### Princípios de um bom golden set

1. **Criado manualmente**: não gerado por LLM (evita bias circular)
2. **Verificado por humano**: cada resposta esperada foi conferida
3. **Representativo**: cobre todos os cenários da tarefa (happy path + edge cases)
4. **Isolado**: **nunca** vaza para o dataset de treino
5. **Estável**: não muda entre avaliações (permite comparar modelos ao longo do tempo)
6. **Documentado**: cada caso tem ID, categoria e justificativa

### Estrutura recomendada

```json
{
  "id": "routing-01",
  "question": "Qual o faturamento total do mês passado?",
  "expected_domains": ["financeiro"],
  "expected_plan": "Consultar faturamento mensal",
  "category": "single_domain",
  "difficulty": "easy",
  "notes": "Pergunta direta, domínio claro"
}
```

### Tamanho recomendado

| Tipo de tarefa | Mínimo | Ideal |
|---------------|--------|-------|
| Classificação binária | 50 | 200+ |
| Classificação multi-classe (4 classes) | 40 (10/classe) | 200+ (50/classe) |
| Geração de texto | 30 | 100+ |
| Tool-calling | 40 (10/domínio) | 160+ (40/domínio) |

### Golden set do AI-Orchestrator

- **44 perguntas de routing**: cobrem single-domain, multi-domain, ambíguas e injections
- **40 tasks de domínio**: 10 por domínio (finanças, RH, estoque, vendas), incluindo edge cases
- **6 injections**: tentativas de jailbreak que o modelo deve ignorar

```
golden_routing.jsonl  → 44 linhas  (eval de roteamento)
golden_domains.jsonl  → 40 linhas  (eval de tool-calling por domínio)
```

### Anti-contaminação automática

O pipeline de geração de dados verifica automaticamente que nenhuma pergunta do golden vazou para o treino:

```python
# Carrega normalizações do golden COMPLETO
golden = {_normalize(seed.text) for seed in _load_seeds()}

# Na geração de variações, rejeita matches
for question in variations:
    if _normalize(question) in golden:
        continue  # DESCARTA — contaminação do eval
```

---

## Relatórios de avaliação: o que incluir

Um relatório de avaliação bem estruturado documenta o experimento de forma que qualquer pessoa possa reproduzir e auditar os resultados.

### Template de relatório

```markdown
# Relatório de Avaliação — [Nome do Modelo]

## Configuração
- Modelo base: Qwen3.5-9B
- Fine-tuning: LoRA bf16, r=16, alpha=32
- Dataset: 3.050 exemplos (2.745 train / 305 val)
- Treino: 2 epochs, 344 steps, 148 min (A100 40GB)
- Val loss: 0.089

## Resultados

| Eval       | Score   | Gate    | Status |
|------------|---------|---------|--------|
| Routing    | 90.9%   | >= 90%  | PASS   |
| Injection  | 0/6     | 0 leaks | PASS   |
| Domains    | 87.5%   | >= 80%  | PASS   |

### Detalhamento por domínio
| Domínio    | Acurácia | Gate   |
|------------|----------|--------|
| Financeiro | 90%      | PASS   |
| RH         | 90%      | PASS   |
| Estoque    | 90%      | PASS   |
| Vendas     | 80%      | PASS   |

## Comparativo com baseline

| Modelo        | Routing | Domains | Injection |
|---------------|---------|---------|-----------|
| Qwen3 7B      | 90.5%   | 82.5%   | 0/6       |
| Qwen3.5 9B    | 95.5%   | 87.5%   | 0/6       |
| Qwen3.5 LoRA  | 90.9%   | 87.5%   | 0/6       |

## Erros detalhados
[Lista de cada caso que errou, com pergunta, resposta esperada
e resposta obtida — para análise qualitativa]

## Decisão
Modelo APROVADO para produção. Trade-off aceito: routing
caiu 4.6pp mas domain accuracy se manteve e tool-calling
ficou mais consistente.

## Reprodução
1. `ollama create qwen3.5-9b-orch -f Modelfile`
2. `MODEL=qwen3.5-9b-orch python evals/eval_routing.py`
3. `MODEL=qwen3.5-9b-orch python evals/eval_injection.py`
4. `MODEL=qwen3.5-9b-orch python evals/eval_domains.py`
```

### O que incluir sempre

1. **Configuração completa**: modelo, hiperparâmetros, dataset, hardware
2. **Métricas agregadas**: acurácia, F1, latência média
3. **Detalhamento**: por classe/domínio/categoria
4. **Comparativo**: modelo base vs fine-tuned
5. **Erros**: lista completa de falhas para análise qualitativa
6. **Decisão**: aprovado/reprovado, com justificativa
7. **Reprodução**: comandos exatos para reproduzir os resultados

---

## Resumo do capítulo

1. **Perplexidade** mede a confiança do modelo, mas não garante qualidade em produção.
2. **Acurácia e F1** são as métricas primárias para tarefas de classificação.
3. **Benchmarks gerais** (MMLU, HellaSwag) verificam se houve catastrophic forgetting.
4. **LLM-as-Judge** é útil para avaliar geração de texto livre.
5. Compare **sempre** o modelo fine-tuned contra o baseline — em múltiplos runs.
6. O **golden set** é sagrado: criado manualmente, verificado, e nunca contaminado.
7. **Gates de qualidade** definem critérios claros de aprovação/rejeição.
8. **Documente tudo**: configuração, resultados, erros, decisão e comandos de reprodução.

---

## Fontes

- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media.
- Labonne, M. (2025). *LLM Engineer's Handbook*. Packt Publishing.
- Hendrycks, D. et al. (2021). *Measuring Massive Multitask Language Understanding*. arXiv:2009.03300.
- Zellers, R. et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?*. arXiv:1905.07830.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685.
- Gao, L. et al. (2023). *A Framework for Few-shot Language Model Evaluation (lm-evaluation-harness)*. GitHub.
- Projeto AI-Orchestrator — `evals/eval_routing.py`, `evals/eval_injection.py`, `evals/eval_domains.py`, `docs/PLANO_LORA_9B.md`.
