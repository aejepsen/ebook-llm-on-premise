# Capítulo 11 — Treinando LoRA na Prática

## Unsloth: por que é a melhor opção para GPU consumer

Unsloth é uma biblioteca open-source que acelera o fine-tuning de LLMs em até 2x e reduz o consumo de VRAM em até 60%, sem perda de qualidade. Ela consegue isso através de:

- **Kernels CUDA otimizados**: reescreve operações críticas (atenção, RMSNorm, cross-entropy) em Triton/CUDA, eliminando ineficiências do PyTorch padrão.
- **Gradient checkpointing inteligente**: libera memória de ativações intermediárias e recomputa sob demanda, com overhead mínimo.
- **Integração nativa com HuggingFace**: usa `transformers`, `trl` e `peft` por baixo — seu código existente funciona com mudanças mínimas.
- **Suporte a modelos recentes**: Qwen3.5, Llama 3.3, Gemma 2, Mistral — atualizado constantemente.
- **Modelos pré-quantizados**: oferece variantes `unsloth/Qwen3.5-9B` já otimizadas para LoRA.

Alternativas como `axolotl` e `LLaMA-Factory` são mais configuráveis, mas Unsloth é a escolha ideal para quem tem **uma GPU consumer** e quer resultados rápidos com o mínimo de complexidade.

> **Caso real**: no AI-Orchestrator, Unsloth permitiu treinar LoRA bf16 no Qwen3.5-9B em apenas 148 minutos numa A100 40GB, com VRAM pico de 31.8 GB. Sem Unsloth, o mesmo treino exigiria ~50+ GB de VRAM.

---

## Setup no Google Colab (A100/T4)

O Google Colab é a forma mais acessível de treinar LoRA sem investir em hardware. A versão Pro oferece acesso a A100 40GB e L4 24GB.

### Célula 1 — Instalação (rode primeiro, depois reinicie a sessão)

```python
# Instalação pinada — Qwen3.5 exige transformers v5
# IMPORTANTE: após rodar, faça Ambiente de execução → Reiniciar sessão
# (NÃO "Desconectar e excluir"), depois rode a célula de verificação

# Remove torchaudio para liberar espaço
!pip uninstall -y -q torchaudio 2>/dev/null

# Instala Unsloth e dependências
!pip install -q --no-cache-dir "unsloth" "unsloth_zoo"

# Pina versões compatíveis com Qwen3.5
!pip install -q \
    "transformers>=5.2.0,<=5.5.0" \
    "trl>=0.18.2,<=0.24.0" \
    "datasets>=3.4.1,<4.4.0" \
    "accelerate>=1.2" \
    "peft>=0.16" \
    sentencepiece \
    "protobuf<7"

print("INSTALADO — agora: Reiniciar sessão, depois rode a verificação")
```

### Célula 2 — Verificação pós-restart

```python
# Rode APÓS reiniciar a sessão
import unsloth  # importar primeiro, antes de transformers
import transformers, trl, datasets as ds_lib

print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('datasets:', ds_lib.__version__)

# Garante que transformers v5 está ativo (obrigatório para Qwen3.5)
assert transformers.__version__.split('.')[0] == '5', \
    'transformers versão errada — reinstale'
print('OK — ambiente pronto')
```

> **Gotcha**: modelos da série Qwen3.5 exigem `transformers >= 5.2`. Se você usar a versão padrão do Colab (v4.x), o modelo carrega mas gera lixo. Sempre verifique a versão após o restart.

### Célula 3 — Montar o Drive e carregar dataset

```python
# Drive: seus dados e checkpoints ficam seguros entre sessões
from google.colab import drive
drive.mount('/content/drive')

import json

DATA_DIR = '/content/drive/MyDrive/ai-orchestrator-dataset'

def load_rows(path):
    """Carrega JSONL manualmente — Arrow falha com schema heterogêneo."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

raw = {
    'train': load_rows(f'{DATA_DIR}/orch_sft_train.jsonl'),
    'val':   load_rows(f'{DATA_DIR}/orch_sft_val.jsonl'),
}

# Mostra contagem e estrutura do primeiro exemplo
print({k: len(v) for k, v in raw.items()})
print('Roles:', [m['role'] for m in raw['train'][0]['messages']])
```

---

## Carregando modelo base (Qwen, Llama)

A Unsloth oferece modelos pré-otimizados no formato `unsloth/NomeDoModelo`. Veja como carregar:

```python
import torch
from unsloth import FastLanguageModel

MAX_SEQ = 4096  # tamanho máximo de sequência (tokens)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3.5-9B",  # modelo base otimizado
    max_seq_length = MAX_SEQ,
    dtype          = torch.bfloat16,  # precisão bf16 (melhor para A100)
    load_in_4bit   = False,           # False = LoRA bf16, True = QLoRA 4-bit
    load_in_16bit  = True,            # carrega pesos em 16-bit
    full_finetuning = False,          # vamos usar LoRA, não full FT
)
```

### Modelos populares disponíveis

| Modelo | Parâmetros | VRAM (LoRA bf16) | VRAM (QLoRA 4-bit) | Uso |
|--------|-----------|-----------------|--------------------|----|
| `unsloth/Qwen3.5-9B` | 9B | ~30 GB | Não recomendado | Tool-calling, raciocínio |
| `unsloth/Llama-3.3-8B` | 8B | ~24 GB | ~8 GB | Uso geral |
| `unsloth/Gemma-2-9B` | 9B | ~28 GB | ~10 GB | Multilíngue |
| `unsloth/Mistral-7B-v0.3` | 7B | ~20 GB | ~6 GB | Eficiência |

---

## Configurando LoRA: rank, alpha, target_modules

Após carregar o modelo, aplique os adapters LoRA:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,      # rank: capacidade do adapter
    lora_alpha     = 32,      # fator de escala (regra: 2x o rank)
    lora_dropout   = 0.05,    # regularização (0.05 é conservador)
    target_modules = [        # quais camadas recebem LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",   # atenção
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    bias              = "none",       # não treina bias
    use_gradient_checkpointing = "unsloth",  # modo otimizado do Unsloth
    random_state      = 42,           # reprodutibilidade
    max_seq_length    = MAX_SEQ,
)

# Mostra quantos parâmetros serão treinados
model.print_trainable_parameters()
# Saída esperada: trainable params: ~160M (1.7% of 9.4B)
```

### Guia de target_modules

| Módulos | Parâmetros extras | Quando usar |
|---------|------------------|-------------|
| `q_proj, v_proj` | Mínimo | Tarefas simples, pouca VRAM |
| `q_proj, k_proj, v_proj, o_proj` | Médio | Classificação, roteamento |
| `+ gate_proj, up_proj, down_proj` | Máximo | Tool-calling, geração complexa |

> **Decisão AI-Orchestrator**: usamos o conjunto **máximo** (atenção + MLP) porque tool-calling exige que o modelo aprenda tanto a selecionar ferramentas (atenção) quanto a formatar argumentos (MLP). As camadas DeltaNet (híbridas do Qwen3.5) ficaram fora — recomendação da documentação Unsloth.

### Como escolher o rank

```
r=4  → adapter mínimo (~40M params) → tarefas simples
r=8  → padrão conservador (~80M params) → classificação, sentimento
r=16 → padrão recomendado (~160M params) → tool-calling, routing
r=32 → alta capacidade (~320M params) → tarefas muito complexas
r=64 → quase full FT em expressividade → raro, risco de overfit
```

---

## Hiperparâmetros: lr, epochs, batch_size, warmup

O treinamento é configurado pelo `SFTConfig` (Supervised Fine-Tuning Configuration):

```python
from trl import SFTTrainer, SFTConfig
import os

# Checkpoints no Drive — sobrevivem reset de sessão do Colab
DRIVE_OUT = '/content/drive/MyDrive/ai-orchestrator-lora/training'
os.makedirs(DRIVE_OUT, exist_ok=True)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset  = dataset["val"],
    args = SFTConfig(
        # --- Campo de texto ---
        dataset_text_field = "text",   # coluna com o texto formatado
        max_length         = MAX_SEQ,  # trunca sequências acima de 4096 tokens

        # --- Batch ---
        per_device_train_batch_size = 2,   # exemplos por step (limitado pela VRAM)
        gradient_accumulation_steps = 8,   # acumula 8 steps antes de atualizar
        # batch efetivo = 2 x 8 = 16 exemplos por atualização de pesos

        # --- Epochs ---
        num_train_epochs = 2,  # passes completos pelo dataset
        # 2-3 epochs é o sweet spot para LoRA; mais que 5 = overfit provável

        # --- Learning Rate ---
        learning_rate     = 2e-4,     # taxa de aprendizado (padrão LoRA)
        lr_scheduler_type = "cosine", # decai suavemente até ~0
        warmup_ratio      = 0.03,     # 3% dos steps com lr crescente

        # --- Otimizador ---
        optim        = "adamw_8bit",  # AdamW quantizado (economiza VRAM)
        weight_decay = 0.01,          # regularização L2

        # --- Precisão ---
        bf16 = True,  # bfloat16 (A100/H100). Use fp16=True para T4.

        # --- Logging e avaliação ---
        logging_steps  = 10,       # loga loss a cada 10 steps
        eval_strategy  = "epoch",  # avalia no val set a cada epoch
        save_strategy  = "epoch",  # salva checkpoint a cada epoch

        # --- Saída ---
        output_dir = DRIVE_OUT,  # DRIVE, não disco efêmero do Colab!
        seed       = 42,
        report_to  = "none",  # desabilita WandB/TensorBoard
    ),
)
```

### Máscara de loss — treinando apenas nas respostas

```python
# Mascara loss fora dos turnos do assistant
# Tool results e system prompts não contribuem para o loss
from unsloth.chat_templates import train_on_responses_only

if "<|im_start|>" in (tokenizer.chat_template or ""):
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",      # delimita contexto
        response_part    = "<|im_start|>assistant\n",  # delimita resposta
    )
    print("train_on_responses_only: ATIVO")
```

### Guia de hiperparâmetros

| Parâmetro | Valor recomendado | Por quê |
|-----------|------------------|---------|
| `learning_rate` | 1e-4 a 3e-4 | Muito alto = instabilidade; muito baixo = não converge |
| `num_train_epochs` | 2-3 | Mais = overfit em datasets pequenos (<5K) |
| `batch_size efetivo` | 8-32 | Abaixo de 8 = gradientes ruidosos; acima de 32 = pouco ganho |
| `warmup_ratio` | 0.03-0.10 | Evita saltos de loss no início |
| `weight_decay` | 0.01-0.1 | Regularização contra overfit |
| `lr_scheduler` | cosine | Decaimento suave, melhor que linear na maioria dos casos |

---

## Monitorando o treino: loss curves, overfitting

Após configurar tudo, inicie o treinamento:

```python
import torch

# Reset de métricas de VRAM para medir o pico real
torch.cuda.reset_peak_memory_stats()

# Treino (a barra de progresso mostra loss em tempo real)
stats = trainer.train()

# Métricas finais
print(f"Train loss final: {stats.metrics.get('train_loss'):.4f}")
print(f"Tempo total: {stats.metrics.get('train_runtime', 0)/60:.1f} min")

# Histórico detalhado — train e val loss por step/epoch
for row in trainer.state.log_history:
    if "loss" in row or "eval_loss" in row:
        print(row)

# VRAM pico atingido
vram = torch.cuda.max_memory_reserved() / 1024**3
print(f"VRAM pico: {vram:.1f} GB")
```

### Como interpretar as loss curves

```
Loss ideal (sem overfit):
  Epoch 1: train_loss=0.091 | val_loss=0.097  ← val próximo do train
  Epoch 2: train_loss=0.071 | val_loss=0.089  ← ambos caindo

Loss com overfit (PARE o treino):
  Epoch 1: train_loss=0.091 | val_loss=0.097
  Epoch 2: train_loss=0.050 | val_loss=0.110  ← val SUBINDO
  Epoch 3: train_loss=0.020 | val_loss=0.150  ← divergência crescente
```

No AI-Orchestrator, os resultados reais foram:
- **Epoch 1**: train 0.091 / val 0.097
- **Epoch 2**: train 0.071 / val 0.089

Sem overfit — ambas as curvas descendo. O treino poderia continuar para um epoch 3, mas o risco de overfitting com apenas 3.050 exemplos não justificava.

### Sinais de treino saudável

| Sinal | Significado |
|-------|------------|
| Train loss caindo gradualmente | Modelo aprendendo |
| Val loss acompanhando train loss | Sem overfit |
| Val loss estável (não cai mais) | Convergiu — pode parar |
| Val loss subindo com train caindo | OVERFIT — pare imediatamente |
| Loss = NaN | Bug — veja seção de debug |

---

## Caso real: LoRA no Qwen3.5-9B para roteamento multi-agente

O treino do AI-Orchestrator produziu os seguintes resultados:

| Métrica | Valor |
|---------|-------|
| Dataset | 3.050 exemplos (2.745 train / 305 val) |
| Epochs | 2 |
| Steps totais | 344 |
| Tempo | 148 minutos |
| VRAM pico | 31.8 GB (de 40 GB disponíveis na A100) |
| Train loss final | 0.071 |
| Val loss final | 0.089 |
| Adapter salvo | ~320 MB |

### Resultados de avaliação pós-treino

| Avaliação | LoRA 9B | Baseline 9B | Baseline 7B | Gate |
|-----------|---------|-------------|-------------|------|
| Routing | 90.9% | 95.5% | 90.5% | >=90% PASS |
| Injection | 0/6 leaks | 0/6 | 0/6 | 0 leaks PASS |
| Domains | 87.5% | 87.5% | 82.5% | >=80%/dom PASS |

O modelo LoRA passou em todos os gates de qualidade. Curiosamente, o routing caiu de 95.5% para 90.9% — mas a acurácia de domínios melhorou de 82.5% (7B) para 87.5% e o tool-calling ficou mais consistente. O modelo foi promovido para produção.

---

## Dicas de debug: OOM, NaN loss, convergência lenta

### Out of Memory (OOM)

```
CUDA out of memory. Tried to allocate 2.00 GiB
```

Soluções, em ordem de preferência:
1. Reduza `per_device_train_batch_size` (de 2 para 1)
2. Aumente `gradient_accumulation_steps` (para manter batch efetivo)
3. Reduza `max_seq_length` (de 4096 para 2048)
4. Reduza `r` do LoRA (de 16 para 8)
5. Use `load_in_4bit=True` (QLoRA) se o modelo suportar
6. Limpe cache HF antes do treino:

```python
# Libera cache do HuggingFace (modelo já está na VRAM)
import shutil, os
hf_cache = '/root/.cache/huggingface/hub'
if os.path.exists(hf_cache):
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, dn, fn in os.walk(hf_cache) for f in fn)
    shutil.rmtree(hf_cache, ignore_errors=True)
    print(f'Cache HF removido: {sz/1024**3:.1f} GB liberados')
```

### NaN loss

Loss vira `nan` durante o treinamento. Causas comuns:

1. **Learning rate muito alto**: reduza de 2e-4 para 5e-5
2. **Dados corrompidos**: JSON malformado, campos ausentes, textos vazios
3. **Sequências muito longas**: tokens acima do `max_seq_length` causam overflow
4. **Precisão numérica**: use `bf16=True` em vez de `fp16=True` (bf16 tem range maior)

### Convergência lenta

O loss não cai ou cai muito devagar:

1. **Learning rate muito baixo**: aumente de 1e-5 para 1e-4
2. **Batch efetivo muito pequeno**: aumente `gradient_accumulation_steps`
3. **Target modules insuficientes**: adicione mais camadas ao LoRA
4. **Rank muito baixo**: aumente `r` de 8 para 16 ou 32
5. **Dataset muito pequeno ou pouco diverso**: adicione mais exemplos

### CUDA IllegalMemoryAccess (específico Qwen3.5)

```
RuntimeError: CUDA error: an illegal memory access was encountered
```

Isso acontece ao chamar `trainer.evaluate()` pós-treino nas camadas DeltaNet do Qwen3.5 (bug da Unsloth). **Solução**: use o val loss computado **durante** o treino (`eval_strategy="epoch"`) e **não** chame `evaluate()` separadamente.

### Checkpoints perdidos

No Colab, o disco `/content` é **efêmero** — se a sessão cair, tudo se perde. **Sempre** salve no Drive:

```python
# CORRETO: output no Drive
output_dir = '/content/drive/MyDrive/meu-projeto/training'

# ERRADO: output no disco efêmero (será perdido!)
output_dir = '/content/output'  # NÃO FAÇA ISSO
```

> **Incidente real no AI-Orchestrator**: 2h17 de treino foram perdidas porque o `output_dir` apontava para `/content` em vez do Drive. A sessão do Colab caiu e os checkpoints sumiram. Desde então, **todo output vai para o Drive**.

---

## Código completo comentado passo a passo

Abaixo está o notebook completo condensado, com cada bloco explicado:

```python
# ============================================================
# BLOCO 1: Instalação (rodar e reiniciar a sessão)
# ============================================================
!pip uninstall -y -q torchaudio 2>/dev/null
!pip install -q --no-cache-dir "unsloth" "unsloth_zoo"
!pip install -q "transformers>=5.2.0,<=5.5.0" "trl>=0.18.2,<=0.24.0" \
    "datasets>=3.4.1,<4.4.0" "accelerate>=1.2" "peft>=0.16" \
    sentencepiece "protobuf<7"
# >>> REINICIE A SESSÃO AQUI <<<

# ============================================================
# BLOCO 2: Verificação + Drive + Dataset
# ============================================================
import unsloth
import transformers, json
assert transformers.__version__.split('.')[0] == '5'

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/ai-orchestrator-dataset'
raw = {
    'train': [json.loads(l) for l in open(f'{DATA_DIR}/orch_sft_train.jsonl')
              if l.strip()],
    'val':   [json.loads(l) for l in open(f'{DATA_DIR}/orch_sft_val.jsonl')
              if l.strip()],
}

# ============================================================
# BLOCO 3: Modelo base + LoRA
# ============================================================
import torch
from unsloth import FastLanguageModel

MAX_SEQ = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-9B", max_seq_length=MAX_SEQ,
    dtype=torch.bfloat16, load_in_4bit=False, load_in_16bit=True,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    bias="none", use_gradient_checkpointing="unsloth",
    random_state=42, max_seq_length=MAX_SEQ,
)

# ============================================================
# BLOCO 4: Formatação com chat template
# ============================================================
from datasets import Dataset, DatasetDict

def format_row(row):
    msgs = []
    for m in row['messages']:
        mc = dict(m)
        # Garante que content é sempre string (tool results podem ser dict)
        if not isinstance(mc.get('content'), str):
            mc['content'] = json.dumps(mc.get('content'), ensure_ascii=False)
        msgs.append(mc)
    return tokenizer.apply_chat_template(msgs, tokenize=False)

dataset = DatasetDict({
    'train': Dataset.from_dict({'text': [format_row(r) for r in raw['train']]}),
    'val':   Dataset.from_dict({'text': [format_row(r) for r in raw['val']]}),
})

# ============================================================
# BLOCO 5: Trainer + máscara de loss
# ============================================================
from trl import SFTTrainer, SFTConfig
import os

DRIVE_OUT = '/content/drive/MyDrive/ai-orchestrator-lora/training'
os.makedirs(DRIVE_OUT, exist_ok=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"], eval_dataset=dataset["val"],
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_SEQ,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=2, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        optim="adamw_8bit", weight_decay=0.01, bf16=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        output_dir=DRIVE_OUT, seed=42, report_to="none",
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ============================================================
# BLOCO 6: Treino
# ============================================================
torch.cuda.reset_peak_memory_stats()
stats = trainer.train()
print(f"Loss: {stats.metrics['train_loss']:.4f}")
print(f"Tempo: {stats.metrics['train_runtime']/60:.1f} min")
print(f"VRAM: {torch.cuda.max_memory_reserved()/1024**3:.1f} GB")
```

---

## Resumo do capítulo

1. **Unsloth** é a melhor opção para GPU consumer — acelera 2x, reduz VRAM 60%.
2. **Colab Pro** com A100 é suficiente para LoRA bf16 em modelos de 9B.
3. **Pinar versões** de transformers/trl é obrigatório para modelos recentes.
4. **Checkpoints no Drive** — nunca no disco efêmero do Colab.
5. **train_on_responses_only** melhora qualidade mascarando loss em contexto.
6. **2-3 epochs** com lr 2e-4 cosine é o sweet spot para datasets de ~3K exemplos.
7. Monitore val loss: se subir enquanto train cai, **pare** (overfit).
8. O AI-Orchestrator treinou em 148 min com val loss de 0.089 — sem overfit.

---

## Fontes

- Unsloth Documentation (2025). *Getting Started with LoRA*. https://docs.unsloth.ai
- Han, D. & Unsloth Team (2024). *Unsloth: Fast LoRA Fine-tuning*. GitHub.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media.
- Labonne, M. (2025). *LLM Engineer's Handbook*. Packt Publishing.
- HuggingFace TRL Documentation (2025). *SFTTrainer*. https://huggingface.co/docs/trl
- Projeto AI-Orchestrator — `train/colab_train_lora.ipynb` (notebook completo), `docs/PLANO_LORA_9B.md`.
