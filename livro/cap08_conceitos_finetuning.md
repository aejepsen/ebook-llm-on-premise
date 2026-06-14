# Capítulo 8 — Conceitos de Fine-Tuning

## O que é fine-tuning é quando usar

Fine-tuning é o processo de continuar o treinamento de um modelo de linguagem pré-treinado usando dados específicos da sua tarefa. Enquanto o pré-treinamento ensina o modelo a "entender" linguagem a partir de trilhões de tokens da internet, o fine-tuning o especializa em um domínio, formato de saída ou comportamento particular.

Pense assim: o pré-treinamento é a faculdade — o modelo aprende fundamentos amplos. O fine-tuning é a residência médica — ele se especializa em uma área com dados curados e supervisão direta.

Quando fazer fine-tuning:

- **Formato de saída consistente**: você precisa que o modelo sempre responda em JSON, siga um schema específico ou use um template fixo.
- **Comportamento especializado**: roteamento de requisições, classificação de intenções, tool-calling em um domínio fechado.
- **Conhecimento proprietário profundo**: quando os dados do seu negócio precisam estar "nos pesos" do modelo, não apenas no contexto.
- **Latência e custo**: um modelo menor fine-tuned pode superar um modelo maior com prompting, com fração do custo de inferência.

No projeto AI-Orchestrator, fizemos fine-tuning do Qwen3.5-9B para duas tarefas: **roteamento multi-agente** (classificar perguntas em domínios como finanças, RH, estoque, vendas) e **tool-calling** (executar ferramentas contra microsserviços reais). O modelo base já acertava 95.5% no roteamento — mas precisávamos que ele também executasse tool-calling com consistência e mantivesse robustez contra prompt injection.

---

## Transfer Learning: aproveitando conhecimento pré-treinado

Transfer learning é o princípio fundamental por trás do fine-tuning. Em vez de treinar um modelo do zero (o que custaria milhões de dólares e semanas de GPU), você parte de um modelo que já aprendeu:

- **Gramática e sintaxe** de dezenas de idiomas
- **Raciocínio lógico** básico e matemático
- **Conhecimento factual** até a data de corte do treinamento
- **Capacidade de seguir instruções** (se for um modelo instruction-tuned)

Esse conhecimento está codificado em bilhões de parâmetros — pesos numéricos organizados em camadas de atenção e feedforward. Quando fazemos fine-tuning, **não descartamos** esse conhecimento. Apenas ajustamos uma parte dos pesos para que o modelo se comporte de forma diferente em situações específicas.

A analogia clássica: ensinar um médico generalista a ser cardiologista é muito mais rápido do que formar alguém que nunca estudou medicina. O generalista já sabe anatomia, fisiologia e farmacologia — precisa apenas de especialização.

```
Pré-treinamento (meses, milhões de $):
  Texto da internet → Modelo generalista (7B-70B parâmetros)

Fine-tuning (horas, dezenas de $):
  Dados específicos → Modelo especialista (mesmos parâmetros, comportamento diferente)
```

---

## Full Fine-Tuning vs Parameter-Efficient (PEFT)

Existem duas abordagens fundamentais para fine-tuning:

### Full Fine-Tuning

Atualiza **todos** os parâmetros do modelo durante o treinamento. Para um modelo de 9 bilhões de parâmetros, isso significa:

- **Memória**: ~36 GB só para os pesos em fp32 (4 bytes x 9B), mais gradientes (~36 GB) e estados do otimizador (~72 GB). Total: **~144 GB de VRAM** — impossível em GPUs consumer.
- **Risco de catastrophic forgetting**: o modelo pode "esquecer" capacidades gerais ao se especializar demais.
- **Dados necessários**: tipicamente dezenas de milhares a milhões de exemplos para convergir sem degradação.

### Parameter-Efficient Fine-Tuning (PEFT)

Congela a maioria dos parâmetros e treina apenas um subconjunto pequeno. Vantagens:

- **Memória drasticamente menor**: treina 0.1%-2% dos parâmetros originais.
- **Preserva conhecimento base**: os pesos originais ficam congelados — o modelo não "esquece".
- **Modularidade**: os parâmetros treinados (adapters) podem ser salvos separadamente (~50-200 MB vs ~18 GB do modelo completo) e trocados em runtime.

As técnicas PEFT mais relevantes hoje:

| Técnica | Ano | Princípio | Parâmetros treináveis |
|---------|-----|-----------|----------------------|
| LoRA | 2021 | Matrizes de baixo rank | 0.1%-1% |
| QLoRA | 2023 | LoRA + quantização 4-bit | 0.1%-1% |
| DoRA | 2024 | LoRA com decomposição de magnitude/direção | 0.1%-1% |
| Prefix Tuning | 2021 | Tokens virtuais prepended | <0.1% |
| Adapters | 2019 | Camadas bottleneck inseridas | 0.5%-2% |

Para GPUs consumer (RTX 3060, 4090, ou Colab gratuito com T4), **PEFT é o único caminho viável**. E dentro de PEFT, LoRA domina o ecossistema por simplicidade, eficácia e suporte universal.

---

## LoRA: Low-Rank Adaptation — como funciona

LoRA (Low-Rank Adaptation), proposto por Hu et al. (2021), é a técnica PEFT mais utilizada no mundo. A intuição é elegante:

> As mudanças nos pesos durante o fine-tuning têm **baixo rank intrínseco** — ou seja, podem ser aproximadas por matrizes muito menores do que as originais.

### A matemática simplificada

Em um Transformer, cada camada de atenção tem matrizes de projeção (Q, K, V, O) e cada camada feedforward tem matrizes (gate, up, down). Essas matrizes são enormes — por exemplo, a matriz Q do Qwen3.5-9B tem dimensão `5120 x 5120` = 26.2 milhões de parâmetros.

LoRA congela a matriz original `W` e adiciona uma **decomposição de baixo rank** `BA`:

```
Saída = W·x + (B·A)·x
         │        │
    congelado   treinável
```

Onde:
- `W` é a matriz original (congelada), dimensão `d x d`
- `A` é uma matriz "down-projection", dimensão `r x d` (r << d)
- `B` é uma matriz "up-projection", dimensão `d x r`
- `r` é o **rank** — tipicamente 8, 16 ou 32

### Diagrama: como LoRA modifica uma camada

```
                    ┌─────────────────────┐
                    │   Entrada (x)       │
                    │   dim: [batch, d]   │
                    └─────────┬───────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            │            ▼
        ┌────────────┐       │    ┌────────────────┐
        │  W (d x d) │       │    │   A (r x d)    │
        │  CONGELADO │       │    │   treinável     │
        │ 26.2M param│       │    │   r=16: 81K     │
        └─────┬──────┘       │    └───────┬────────┘
              │              │            │
              │              │            ▼
              │              │    ┌────────────────┐
              │              │    │   B (d x r)    │
              │              │    │   treinável     │
              │              │    │   r=16: 81K     │
              │              │    └───────┬────────┘
              │              │            │
              ▼              │            ▼
        ┌─────────┐         │    ┌─────────────┐
        │  W · x  │         │    │ (B·A)·x     │
        │         │         │    │ × (α/r)     │
        └────┬────┘         │    └──────┬──────┘
             │              │           │
             └──────────┬───────────────┘
                        │ soma
                        ▼
                ┌───────────────┐
                │  Saída final  │
                │  W·x + BA·x  │
                └───────────────┘
```

### Os hiperparâmetros de LoRA

- **rank (r)**: controla a "capacidade" do adapter. `r=16` é o padrão mais usado — bom equilíbrio entre qualidade e custo. Aumentar para 32 ou 64 pode ajudar em tarefas complexas, mas aumenta VRAM e risco de overfitting.
- **alpha (α)**: fator de escala aplicado ao adapter. A fórmula é `saída_LoRA = (α/r) × BA·x`. Regra prática: `alpha = 2 × rank`. No AI-Orchestrator usamos `r=16, alpha=32`.
- **dropout**: regularização aplicada aos adapters. `0.05` é um valor padrão seguro.
- **target_modules**: quais matrizes recebem adapters LoRA. Quanto mais módulos, mais expressivo o adapter — mas mais VRAM e parâmetros.

No AI-Orchestrator, os target_modules foram:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # atenção
    "gate_proj", "up_proj", "down_proj",        # MLP/feedforward
]
```

As camadas DeltaNet (híbridas, específicas do Qwen3.5) ficaram **fora** dos adapters — decisão alinhada com a documentação oficial da Unsloth.

### Economia de parâmetros

Para o Qwen3.5-9B com `r=16`:
- Parâmetros totais: ~9.4 bilhões
- Parâmetros treináveis (LoRA): ~160 milhões (~1.7%)
- Tamanho do adapter salvo: ~320 MB (vs ~18 GB do modelo completo)

---

## QLoRA: LoRA + quantização 4-bit

QLoRA (Dettmers et al., 2023) combina LoRA com **quantização NormalFloat 4-bit** do modelo base. A ideia:

1. Carrega o modelo base quantizado em 4-bit (reduz VRAM de ~18 GB para ~5 GB para um modelo 9B)
2. Aplica adapters LoRA em precisão bf16/fp16 sobre os pesos quantizados
3. Gradientes fluem apenas pelos adapters — os pesos 4-bit ficam congelados

**Vantagem**: treinar modelos de 7-13B em GPUs com apenas 8-16 GB de VRAM (T4, RTX 3060).

**Desvantagem**: a quantização introduz ruído nos pesos base, o que pode degradar a qualidade do fine-tuning — especialmente em arquiteturas mais novas.

> **Caso real — AI-Orchestrator**: a Unsloth **contraindica QLoRA 4-bit** no Qwen3.5 devido à arquitetura híbrida (DeltaNet + Attention). Usamos LoRA bf16 com `load_in_4bit=False`. Isso exigiu uma A100 40GB no Colab em vez de uma T4 16GB. A decisão foi acertada: treinar com os pesos em precisão completa resultou em val loss de 0.089 — excelente para um dataset de 3.050 exemplos.

```python
# AI-Orchestrator: LoRA bf16 (NÃO QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3.5-9B",
    max_seq_length = 4096,
    dtype          = torch.bfloat16,
    load_in_4bit   = False,   # QLoRA desabilitado
    load_in_16bit  = True,    # pesos em precisão completa
)
```

---

## DoRA: Weight-Decomposed LoRA

DoRA (Liu et al., 2024) é uma evolução do LoRA que decompõe os pesos em dois componentes:

- **Magnitude**: escalar que controla "quanto" cada neurônio ativa
- **Direção**: vetor unitário que controla "para onde" a ativação aponta

O LoRA padrão modifica magnitude e direção simultaneamente. O DoRA treina a magnitude separadamente e aplica LoRA apenas na direção — imitando mais fielmente o comportamento do full fine-tuning com a eficiência do LoRA.

Em benchmarks publicados, DoRA mostra ganhos de 1-3% sobre LoRA em tarefas de NLU, especialmente com ranks baixos (r=4 ou r=8).

> **No AI-Orchestrator**: DoRA foi avaliado mas **não utilizado** no treinamento final. A Unsloth na versão utilizada não garantia estabilidade do DoRA com as camadas DeltaNet do Qwen3.5. Como o LoRA padrão já atingiu os gates de qualidade (routing >= 90%, injection 0 leaks, domains >= 80% por domínio), a complexidade adicional não se justificou.

---

## Quando NÃO fazer fine-tuning (RAG pode ser suficiente)

Fine-tuning não é a resposta para tudo. Antes de investir tempo e GPU, considere se **Retrieval-Augmented Generation (RAG)** resolve o seu problema:

| Cenário | RAG | Fine-Tuning |
|---------|-----|-------------|
| Perguntas sobre documentos que mudam frequentemente | Ideal | Inadequado (retreinar a cada mudança) |
| Formato de saída rígido (JSON schema) | Funciona com prompt engineering | Melhor solução |
| Tool-calling consistente | Frágil com prompt-only | Ideal |
| Conhecimento factual atualizado | Ideal (busca em tempo real) | Desatualiza rápido |
| Comportamento/personalidade customizada | Parcial | Ideal |
| Base de conhecimento com milhares de documentos | Ideal | Impraticável (não cabe no contexto) |

**Regra prática**: se o problema é de **conhecimento** (o modelo não sabe algo), use RAG. Se o problema é de **comportamento** (o modelo sabe, mas não faz do jeito certo), use fine-tuning. Se é ambos, combine: fine-tuning para o comportamento + RAG para o conhecimento.

No AI-Orchestrator, a arquitetura usa **ambos**: o fine-tuning ensina o modelo a rotear e chamar tools corretamente; os microsserviços (que são as "tools") consultam bancos de dados reais — funcionando como um RAG estruturado.

---

## Custos: GPU, tempo, dados — planejamento realista

Fine-tuning exige planejamento de recursos. Aqui está o que esperar para modelos de 7-13B com LoRA:

### GPU

| GPU | VRAM | Adequada para |
|-----|------|---------------|
| T4 (Colab Free) | 16 GB | QLoRA 4-bit em modelos 7B |
| L4 (Colab Pro) | 24 GB | LoRA bf16 em modelos 7B, QLoRA em 13B |
| A100 (Colab Pro) | 40 GB | LoRA bf16 em modelos 9-13B |
| RTX 3060 | 12 GB | QLoRA 4-bit em modelos 7B (com paciência) |
| RTX 4090 | 24 GB | LoRA bf16 em modelos 7B-9B |

### Tempo

Com dataset de ~3.000 exemplos e modelo 9B:
- **A100 40GB**: ~2.5 horas (2 epochs)
- **L4 24GB**: ~4-6 horas (estimado)
- **T4 16GB**: 8-12 horas (QLoRA, batch menor)

### Dados

- **Mínimo viável**: 500-1.000 exemplos de alta qualidade
- **Recomendado**: 1.500-5.000 exemplos
- **Diminishing returns**: acima de 10.000, a qualidade do dado importa mais que a quantidade

### Custo real — AI-Orchestrator

| Recurso | Valor |
|---------|-------|
| GPU | A100 40GB (Colab, ~25 unidades de ~87 disponíveis) |
| Tempo de treino | 148 minutos (2 epochs, 344 steps) |
| VRAM pico | 31.8 GB de 40 GB |
| Dataset | 3.050 exemplos (2.745 train / 305 val) |
| Tamanho do adapter | ~320 MB |
| Tamanho GGUF final | 5.4 GB (Q4_K_M) |

---

## Comparativo: LoRA vs QLoRA vs Full Fine-Tuning

| Aspecto | Full Fine-Tuning | LoRA | QLoRA |
|---------|-----------------|------|-------|
| **Parâmetros treinados** | 100% | 0.1%-2% | 0.1%-2% |
| **VRAM (modelo 9B)** | ~144 GB | ~30-36 GB | ~8-12 GB |
| **GPU mínima** | A100 80GB (ou multi-GPU) | A100 40GB / RTX 4090 | T4 16GB / RTX 3060 |
| **Qualidade máxima** | Referência | 95-100% do full | 90-98% do full |
| **Risco de forgetting** | Alto | Baixo (pesos congelados) | Baixo |
| **Adapter separado** | Não (modelo inteiro muda) | Sim (~200 MB) | Sim (~200 MB) |
| **Merge necessário** | Não | Sim (para deploy) | Sim (para deploy) |
| **Suporte a modelos novos** | Universal | Universal | Depende da arquitetura |
| **Complexidade** | Baixa (só treinar) | Média | Média-Alta |
| **Custo Colab (3K exemplos)** | Inviável | ~$10-15 (A100) | ~$3-5 (T4/L4) |

### Recomendação por cenário

- **Hobby/aprendizado**: QLoRA no Colab Free (T4) com modelo 7B
- **Produção com GPU consumer**: LoRA bf16 no Colab Pro (A100) ou RTX 4090
- **Produção enterprise**: Full fine-tuning com cluster multi-GPU (ou usar LoRA mesmo — a diferença é marginal na maioria das tarefas)
- **Modelo com arquitetura nova/experimental**: LoRA bf16 (QLoRA pode ter bugs de quantização)

---

## Resumo do capítulo

1. Fine-tuning especializa um modelo pré-treinado com seus dados.
2. Transfer learning permite partir de bilhões de parâmetros já treinados.
3. PEFT (especialmente LoRA) torna fine-tuning viável em GPUs acessíveis.
4. LoRA adiciona matrizes de baixo rank — treina <2% dos parâmetros com resultados próximos ao full fine-tuning.
5. QLoRA reduz VRAM via quantização 4-bit, mas pode não funcionar em arquiteturas novas.
6. DoRA é uma evolução promissora, mas menos testada em produção.
7. Antes de fine-tuning, avalie se RAG resolve. Combine ambos quando necessário.
8. Planeje GPU, tempo e dados realisticamente — o caso do AI-Orchestrator mostra que 3.050 exemplos e 2.5 horas numa A100 são suficientes para resultados de produção.

---

## Fontes

- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Liu, S.-Y. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media. ISBN 978-1-098-15096-9.
- Labonne, M. (2025). *LLM Engineer's Handbook*. Packt Publishing.
- Unsloth Documentation (2025). *Qwen3.5 Fine-tuning Guide*. https://docs.unsloth.ai
- Projeto AI-Orchestrator — `docs/PLANO_LORA_9B.md`, `train/colab_train_lora.ipynb`. Dados reais de treino e avaliação.
