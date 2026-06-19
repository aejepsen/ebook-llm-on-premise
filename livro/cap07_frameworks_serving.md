# Capítulo 7 -- Frameworks de Serving: Escolhendo a Ferramenta Certa

Você já entende quantização, KV cache, Paged Attention e speculative decoding. Agora a pergunta e: **qual framework implementa tudo isso da melhor forma para o seu cenario?** Este capítulo compara os cinco frameworks mais relevantes para servir LLMs localmente: Ollama, vLLM, llama.cpp, SGLang e TensorRT-LLM. Cada um tem forcas distintas e contextos ideais de uso.

---

## 7.1 Panorama dos frameworks

O ecossistema de serving de LLMs cresceu rapidamente entre 2023 e 2026. Os frameworks diferem em backend de execução, formatos suportados, otimizações implementadas e facilidade de uso.

| Framework | Backend | Formato | API | GPU | CPU | Foco |
|-----------|---------|---------|-----|-----|-----|------|
| **Ollama** | llama.cpp | GGUF | REST proprietaria | Sim | Sim | Simplicidade |
| **vLLM** | PyTorch | HF/GPTQ/AWQ/FP8 | OpenAI-compatible | Sim | Não | Throughput em produção |
| **llama.cpp** | GGML/GGUF | GGUF | Própria + OpenAI-compat. | Sim | Sim | Eficiência em hardware limitado |
| **SGLang** | PyTorch | HF/GPTQ/AWQ | OpenAI-compatible | Sim | Não | Programação estruturada + velocidade |
| **TensorRT-LLM** | TensorRT | Checkpoints otimizados | Própria | Sim (NVIDIA only) | Não | Performance máxima NVIDIA |

---

## 7.2 llama.cpp: inferência em CPU, GGUF e llama-server

O llama.cpp e o projeto que democratizou LLMs locais. Escrito em C/C++, roda em praticamente qualquer hardware: CPU x86, ARM, Apple Silicon, GPUs NVIDIA, AMD e Intel.

### Caracteristicas principais

- **Formato:** GGUF exclusivamente
- **Quantização nativa:** Q2_K até Q8_0, integrado no próprio runtime
- **Offloading hibrido:** parte do modelo na GPU, parte na CPU
- **Zero dependencias Python:** binario compilado, sem PyTorch/CUDA runtime
- **Tamanho:** binario de ~5 MB vs gigabytes dos frameworks baseados em PyTorch

### Instalação e uso básico

```bash
# Compilar llama.cpp com suporte a CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Inferencia via CLI
./build/bin/llama-cli \
  -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
  -p "Explique o que e um LLM:" \
  -n 256 \
  --gpu-layers 35

# Iniciar servidor HTTP (OpenAI-compatible)
./build/bin/llama-server \
  -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-layers 35
```

### Uso via Python (llama-cpp-python)

O binding Python permite integrar llama.cpp em aplicações:

```python
"""
Exemplo de inferencia com llama-cpp-python.
Instalação: pip install llama-cpp-python
Com GPU: CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python
"""
from llama_cpp import Llama

# Carregar modelo GGUF
llm = Llama(
    model_path="./models/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,   # -1 = todas as camadas na GPU
    n_ctx=4096,         # tamanho do contexto
    verbose=False,
)

# Geração simples
resposta = llm(
    "Explique o que e quantizacao em 2 frases.",
    max_tokens=128,
    temperature=0.7,
)
print(resposta["choices"][0]["text"])

# Chat completion (formato OpenAI)
resposta_chat = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Voce e um assistente tecnico."},
        {"role": "user", "content": "O que e KV cache?"},
    ],
    max_tokens=256,
)
print(resposta_chat["choices"][0]["message"]["content"])
```

### Quando usar llama.cpp

- Hardware sem GPU ou com GPU limitada (< 6 GB VRAM)
- Apple Silicon (Mac M1/M2/M3/M4) -- excelente performance com Metal
- Dispositivos embarcados (Raspberry Pi, Jetson)
- Quando você quer controle total sobre offloading CPU/GPU
- Quando tamanho do deploy importa (binario minusculo)

---

## 7.3 vLLM: Paged Attention, continuous batching, API OpenAI-compatible

O vLLM e o framework de referência para serving de LLMs em produção com GPUs NVIDIA. Introduziu o Paged Attention e se tornou o padrão da industria para alto throughput.

### Caracteristicas principais

- **Paged Attention:** gerenciamento de memória do KV cache (Cap. 6)
- **Continuous Batching:** slots reutilizados dinamicamente
- **Prefix Caching:** compartilhamento de KV cache entre requisicoes com mesmo prefixo
- **Speculative Decoding:** suporte a n-gram, Eagle3, draft models
- **Tensor Parallelism:** distribui modelo entre multiplas GPUs automaticamente
- **API OpenAI-compatible:** drop-in replacement para aplicações que usam a API da OpenAI

### Iniciando o servidor

```bash
# Instalar
pip install vllm

# Servir modelo HuggingFace (FP16)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --disable-log-requests

# Servir modelo quantizado (GPTQ INT4)
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --disable-log-requests

# Servir modelo quantizado (FP8)
vllm serve RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic \
  --disable-log-requests

# Servir com tensor parallelism (2 GPUs)
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --disable-log-requests
```

### Consumindo a API (compatível com OpenAI)

```python
"""
Cliente vLLM usando a biblioteca openai.
O vLLM expoe a mesma API da OpenAI -- basta trocar a base_url.
"""
from openai import OpenAI

# Conectar ao servidor vLLM local
client = OpenAI(
    api_key="EMPTY",            # vLLM nao exige chave
    base_url="http://localhost:8000/v1",
)

# Chat completion
resposta = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "Responda em portugues."},
        {"role": "user", "content": "O que e Paged Attention?"},
    ],
    max_tokens=256,
    temperature=0.7,
    stream=True,  # streaming habilitado
)

# Imprimir com streaming
for chunk in resposta:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### Benchmark com vLLM

O vLLM inclui ferramentas de benchmark integradas:

```bash
# Baixar dataset de benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Executar benchmark
python -m vllm.entrypoints.openai.api_server &

vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --max-concurrency 50
```

### Quando usar vLLM

- Produção com GPU NVIDIA (12 GB+ VRAM)
- Alto throughput e muitos usuarios simultaneos
- Quando você precisa de API compatível com OpenAI
- Multi-GPU serving (tensor parallelism)
- Quando prefix caching importa (ex: mesmo system prompt para todos)

---

## 7.4 SGLang: RadixAttention e programação estruturada

O SGLang (Structured Generation Language) combina um runtime de serving eficiente com uma linguagem de programação para LLMs. Seu diferencial e o **RadixAttention**, uma evolução do prefix caching.

### RadixAttention

Enquanto o prefix caching do vLLM compartilha KV cache apenas para prefixos exatos, o RadixAttention usa uma **radix tree** (arvore de prefixos) para compartilhar KV cache de forma mais granular:

```
Requisicoes:
  1. "Voce e um assistente. Explique Python."
  2. "Voce e um assistente. Explique JavaScript."
  3. "Voce e um assistente. O que e Docker?"

RadixAttention armazena:
  "Voce e um assistente." -> KV cache compartilhado (raiz da arvore)
    ├── "Explique Python." -> branch especifico
    ├── "Explique JavaScript." -> branch especifico
    └── "O que e Docker?" -> branch especifico
```

Para workflows com muitas requisicoes compartilhando prefixos (ex: RAG, few-shot prompting, agentes), o RadixAttention reduz significativamente o tempo de prefill.

### Uso básico

```python
"""
SGLang -- exemplo de inferencia offline.
Instalação: pip install "sglang[all]"
"""
import sglang as sgl

# Inicializar engine offline
llm = sgl.Engine(model_path="Qwen/Qwen3-8B-AWQ")

# Geração em lote
prompts = [
    "Explique machine learning em uma frase.",
    "O que e transfer learning?",
    "Qual a diferenca entre GPU e CPU para IA?",
]

# Parametros de amostragem
sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 128}

# Gerar respostas
resultados = llm.generate(prompts, sampling_params)

for prompt, resultado in zip(prompts, resultados):
    print(f"Prompt: {prompt}")
    print(f"Resposta: {resultado['text']}\n")
```

### Streaming com SGLang

```python
from sglang.utils import stream_and_merge

prompt = "Escreva um guia rapido sobre Docker."
sampling_params = {"temperature": 0.3, "top_p": 0.9}

# stream_and_merge retorna o texto completo com streaming interno
texto = stream_and_merge(llm, prompt, sampling_params)
print(texto)
```

### Quando usar SGLang

- Workflows com prefix sharing intensivo (RAG, agentes, few-shot)
- Quando você precisa de geração estruturada (JSON, código)
- Alta concorrência com prefixos repetidos
- Pesquisa e experimentação com LLMs

---

## 7.5 TensorRT-LLM: otimização NVIDIA, FP8, inflight batching

O TensorRT-LLM e o framework da NVIDIA para extrair performance máxima de GPUs NVIDIA. Compila o modelo em um grafo otimizado específico para a sua GPU.

### Caracteristicas principais

- **Compilação de grafo:** converte o modelo em operações otimizadas para a GPU específica
- **FP8:** quantização nativa em 8 bits de ponto flutuante (Hopper/Ada GPUs)
- **Inflight Batching:** variação do continuous batching otimizada para a arquitetura NVIDIA
- **Multi-GPU:** suporte a tensor parallelism e pipeline parallelism
- **KV Cache otimizado:** implementação nativa com paging e eviction

### Exemplo básico

```python
"""
TensorRT-LLM -- exemplo minimo.
Instalação: pip install tensorrt_llm
Requer GPU NVIDIA com drivers recentes.
"""
from tensorrt_llm import LLM, SamplingParams

# Carregar modelo -- aceita modelos HuggingFace, quantizados FP8 ou checkpoints TRT
# Exemplo com modelo FP8 pre-quantizado:
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")

# Alternativa com modelo HuggingFace padrao:
# llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Definir prompts
prompts = [
    "O que e TensorRT?",
    "Explique a diferenca entre FP16 e FP8.",
]

# Parametros de amostragem
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

# Gerar respostas
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Resposta: {output.outputs[0].text}\n")
```

### Quando usar TensorRT-LLM

- GPUs NVIDIA datacenter (A100, H100, H200)
- Quando throughput máximo e prioridade absoluta
- GPUs com suporte a FP8 (RTX 4090, Ada, Hopper)
- Produção enterprise com SLA rigoroso
- **Não usar quando:** hardware não-NVIDIA, prototipação rapida, GPU de consumo com pouca VRAM

---

## 7.6 Comparativo de performance

A tabela abaixo sintetiza benchmarks tipicos em uma GPU RTX 3090 (24 GB) com o modelo Qwen 2.5 7B:

| Métrica | Ollama (GGUF Q4) | llama.cpp (GGUF Q4) | vLLM (GPTQ Int4) | SGLang (AWQ) | TensorRT-LLM (FP8) |
|---------|------------------|---------------------|-------------------|--------------|---------------------|
| **TTFT** (1 req) | ~120 ms | ~100 ms | ~80 ms | ~75 ms | ~50 ms |
| **TPS** (1 req) | ~70 tok/s | ~75 tok/s | ~90 tok/s | ~95 tok/s | ~120 tok/s |
| **Throughput** (50 req) | ~200 tok/s | ~220 tok/s | ~1200 tok/s | ~1400 tok/s | ~1800 tok/s |
| **VRAM usada** | ~5 GB | ~5 GB | ~6 GB | ~6 GB | ~7 GB |
| **Setup time** | 1 min | 5 min | 3 min | 3 min | 15 min |
| **Dificuldade** | Trivial | Moderada | Moderada | Moderada | Alta |

**Observações importantes:**

- Ollama e llama.cpp tem throughput similar em usuario único, mas não escalam bem com muitos usuarios
- vLLM e SGLang dominam em cenarios de alta concorrência gracas ao continuous batching e Paged Attention
- TensorRT-LLM lidera em performance bruta mas exige mais esforco de setup
- Benchmarks reais variam significativamente com hardware, modelo e workload

---

## 7.7 Quando usar cada um: fluxograma de decisao

```
Voce tem GPU NVIDIA?
├── NAO
│   ├── Tem Apple Silicon? -> llama.cpp (Metal) ou Ollama
│   └── Apenas CPU? -> llama.cpp ou Ollama
│
└── SIM
    ├── Prototipo / uso pessoal?
    │   └── Ollama (mais simples)
    │
    ├── Producao com muitos usuarios?
    │   ├── Prefixos compartilhados? (RAG, agentes)
    │   │   └── SGLang (RadixAttention)
    │   ├── API OpenAI-compatible necessaria?
    │   │   └── vLLM
    │   └── Performance maxima em NVIDIA datacenter?
    │       └── TensorRT-LLM
    │
    └── Recursos limitados (< 8 GB VRAM)?
        └── llama.cpp com offloading parcial
```

**Regra simplificada:**

1. **Comecando agora?** Ollama
2. **Indo para produção?** vLLM
3. **Precisa escalar prefix caching?** SGLang
4. **Hardware NVIDIA enterprise + máximo desempenho?** TensorRT-LLM
5. **Hardware limitado ou não-NVIDIA?** llama.cpp

---

## 7.8 Deploy com Docker: exemplo prático

Docker e a forma mais confiável de deployar LLMs em produção. Aqui está um exemplo completo com vLLM:

### docker-compose.yml

```yaml
# docker-compose.yml
# Deploy de LLM local com vLLM + API OpenAI-compatible
version: "3.8"

services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      # Cache de modelos HuggingFace (evita re-download)
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
      --max-model-len 4096
      --gpu-memory-utilization 0.90
      --disable-log-requests
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  huggingface_cache:
```

### Executando

```bash
# Criar arquivo .env com seu token HuggingFace (se necessario)
echo "HF_TOKEN=hf_xxxxx" > .env

# Subir o servico
docker compose up -d

# Verificar logs
docker compose logs -f vllm

# Testar
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Ola!"}],
    "max_tokens": 128
  }'
```

### Deploy com Ollama (alternativa mais simples)

```yaml
# docker-compose-ollama.yml
version: "3.8"

services:
  ollama:
    image: ollama/ollama:latest
    runtime: nvidia
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

```bash
# Subir
docker compose -f docker-compose-ollama.yml up -d

# Baixar modelo dentro do container
docker exec -it ollama ollama pull llama3.2:3b

# Testar
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Ola!"}],
  "stream": false
}'
```

---

## Construindo um servidor LLM do zero

Antes de depender de frameworks como Ollama e vLLM, é instrutivo entender o que acontece dentro deles. O repositório `llm-model-inference` (Wang & Hu, 2025) contém uma implementação completa de um servidor LLM em Python puro que ilustra todos os conceitos que discutimos:

```
servidor_llm/
├── model_executor.py    # Executa forward pass do modelo (GPU/CPU)
├── model_manager.py     # Carrega/descarrega modelos da memória
├── model_worker.py      # Worker que processa requests individuais
├── workload_manager.py  # Orquestra batching, fila, prioridades
└── main.py              # API Flask que expõe /generate e /chat
```

### Workload Manager: o cérebro do servidor

O workload manager decide quando formar batches, quais requests agrupar e quando fazer streaming:

```python
# Padrão simplificado de workload manager
class WorkloadManager:
    def __init__(self, max_batch_size: int = 8):
        self._fila: list[Request] = []
        self._max_batch = max_batch_size

    def submit(self, request: Request):
        self._fila.append(request)
        if len(self._fila) >= self._max_batch:
            self._process_batch()

    def _process_batch(self):
        batch = self._fila[:self._max_batch]
        self._fila = self._fila[self._max_batch:]
        # Prefill: processa todos os prompts em paralelo
        prefill_outputs = model.prefill([r.prompt for r in batch])
        # Decode: geração autoregressiva com KV cache
        for r, kv in zip(batch, prefill_outputs):
            response = model.decode(r.max_tokens, kv_cache=kv)
            r.callback(response)  # streaming para o cliente
```

### O que Ollama e vLLM adicionam

Construir o servidor do zero revela o que os frameworks abstraem:

| Componente | Implementação manual | Ollama | vLLM |
|-----------|---------------------|--------|------|
| Carregamento de modelo | `torch.load` + `to(device)` | `ollama pull` | HF auto-download |
| KV cache | Alocado manualmente | Gerenciado pelo llama.cpp | PagedAttention |
| Batching | Lógica manual de fila | Interno (opaco) | Continuous batching |
| API HTTP | Flask/FastAPI manual | REST na porta 11434 | OpenAI-compatible |
| Quantização | Conversão manual GGUF | Integrado | GPTQ/AWQ/FP8 |

> **Exercício:** Implemente um servidor mínimo com Flask + Transformers que aceita POST /generate e retorna `{"response": "..."}`. Depois compare a performance com o Ollama. A diferença de throughput vai deixar claro por que usamos frameworks em produção.

## Resumo do capítulo

1. **Ollama** e a porta de entrada -- simples, funciona em 1 minuto, ideal para uso pessoal e prototipação
2. **llama.cpp** e o motor por tras do Ollama -- use diretamente quando precisa de controle fino sobre offloading e hardware não-NVIDIA
3. **vLLM** e o padrão de produção -- Paged Attention, continuous batching, API OpenAI-compatible, tensor parallelism
4. **SGLang** se destaca em workflows com prefix sharing via RadixAttention e geração estruturada
5. **TensorRT-LLM** extrai performance máxima de GPUs NVIDIA enterprise com compilação de grafo e FP8
6. **Docker** e a forma recomendada de deployar qualquer um desses frameworks em produção

Nos proximos capitulos, você vai aprender a configurar um gateway de API para gerenciar multiplos modelos e implementar RAG (Retrieval-Augmented Generation) com seu LLM local.

---

## Fontes

1. Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*. O'Reilly Media. Cap. 8 (LLM Serving Frameworks Overview).
2. Wang, C. & Hu, P. (2025). Notebooks: `llamaCpp.ipynb`, `SGLang.ipynb`, `TensorRT_LLM.ipynb`. Repositório: github.com/orca3/llm-model-serving.
3. Troyer, L. (2026). *Benchmarking LLM Serving Systems*. Johannes Kepler University. Secoes 2.9 (LLM Serving Systems), 3.5 (LLM Serving System Selection), Cap. 5 (Results).
4. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
5. Zheng, L. et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs*. Disponível em: https://arxiv.org/abs/2312.07104
6. NVIDIA (2025). *TensorRT-LLM Documentation*. Disponível em: https://nvidia.github.io/TensorRT-LLM/
7. Repositório llama.cpp. Disponível em: https://github.com/ggerganov/llama.cpp
8. Repositório Ollama. Disponível em: https://github.com/ollama/ollama
9. Repositório vLLM. Disponível em: https://github.com/vllm-project/vllm
