# Capitulo 6 -- Otimizacao de Inferencia: Fazendo Seu LLM Voar

Quantizacao reduz o tamanho do modelo. Mas isso e apenas metade da historia. Para realmente maximizar a performance de um LLM local, voce precisa entender as tecnicas que otimizam o **processo de geracao** -- como o modelo usa a GPU a cada token gerado. Este capitulo cobre as seis tecnicas fundamentais: KV Cache, Paged Attention, Speculative Decoding, Continuous Batching, Flash Attention e as metricas para medir tudo isso.

---

## 6.1 KV Cache: o que e, por que acelera, como funciona

Para entender o KV cache, voce precisa entender como um Transformer gera texto.

### O problema

Quando um modelo Transformer gera o token N+1, ele precisa calcular a **atencao** (attention) sobre todos os N tokens anteriores. O mecanismo de atencao usa tres vetores por token: **Query (Q)**, **Key (K)** e **Value (V)**.

Sem cache, a cada novo token o modelo recalcula K e V para **todos** os tokens anteriores. Se voce esta no token 1000, o modelo recalcula 999 pares K,V que ja foram calculados antes. Isso e um desperdicio enorme.

### A solucao: KV Cache

O KV cache armazena os vetores K e V ja calculados na memoria da GPU. Quando o modelo gera o proximo token, ele so precisa:

1. Calcular K e V para o **novo** token
2. Recuperar K e V dos tokens anteriores do cache
3. Calcular Q apenas para o novo token
4. Computar a atencao usando Q do novo token contra K,V de todos os tokens

```
Sem KV cache (token 1000):
  Calcular K,V para tokens 1..1000 = 1000 computacoes

Com KV cache (token 1000):
  Recuperar K,V dos tokens 1..999 do cache (rapido, so leitura)
  Calcular K,V apenas para o token 1000 = 1 computacao
```

### Quanto de memoria o KV cache consome

A formula para estimar o consumo de memoria do KV cache:

```
KV Cache (bytes) = 2 x num_camadas x num_heads x dim_head x seq_len x bytes_por_elem
```

Para um Llama 3.1 8B (32 camadas, 32 heads, dim_head=128, FP16):

```
KV Cache para 4096 tokens = 2 x 32 x 32 x 128 x 4096 x 2 bytes
                          = 2,147,483,648 bytes
                          ≈ 2 GB
```

**Isso e por requisicao.** Se voce tem 10 usuarios simultaneos com contexto de 4096 tokens, o KV cache sozinho consome 20 GB. E por isso que o gerenciamento de memoria do KV cache e tao critico.

---

## 6.2 Paged Attention (vLLM) -- gerenciamento inteligente de memoria

O KV cache cria um problema classico de alocacao de memoria: cada requisicao precisa de um bloco contiguo de memoria, mas o tamanho final da resposta e desconhecido antecipadamente.

### O problema da alocacao tradicional

Sem Paged Attention, o sistema precisa alocar memoria para o comprimento **maximo** possivel de cada requisicao:

```
Requisicao A: prompt=100 tokens, resposta real=50 tokens
  Alocado: 4096 tokens (maximo do contexto)
  Desperdicado: 3946 tokens = 96% de desperdicio!
```

Multiplique isso por dezenas de requisicoes simultaneas e a GPU fica cheia de memoria desperdicada.

### A solucao: Paged Attention

O vLLM resolveu isso inspirando-se no gerenciamento de memoria virtual dos sistemas operacionais. Em vez de alocar blocos contiguos, o KV cache e dividido em **paginas** (blocks) de tamanho fixo (tipicamente 16 tokens por pagina).

```
Memoria da GPU (Paged Attention):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ Pag1 │ Pag2 │ Pag3 │ Pag4 │ Pag5 │ Pag6 │
│ Req A│ Req B│ Req A│ Req C│ Req B│ livre│
└──────┴──────┴──────┴──────┴──────┴──────┘

Tabela de paginas:
  Req A -> [Pag1, Pag3]       (2 paginas = 32 tokens)
  Req B -> [Pag2, Pag5]       (2 paginas = 32 tokens)
  Req C -> [Pag4]             (1 pagina  = 16 tokens)
```

Beneficios:

1. **Sem fragmentacao:** paginas nao precisam ser contiguas
2. **Alocacao sob demanda:** paginas sao alocadas conforme o modelo gera tokens, nao antecipadamente
3. **Compartilhamento de memoria:** requisicoes com o mesmo prefixo (ex: system prompt) podem compartilhar paginas -- tecnica chamada **prefix caching**
4. **Throughput 2-4x maior:** menos memoria desperdicada = mais requisicoes simultaneas

O Paged Attention foi introduzido pelo paper do vLLM (Kwon et al., 2023) e se tornou o padrao da industria. Hoje, alem do vLLM, frameworks como SGLang e TensorRT-LLM implementam variacoes dessa tecnica.

---

## 6.3 Speculative Decoding: draft model + verificacao

A geracao de tokens em LLMs e inerentemente **sequencial**: cada token depende de todos os anteriores. Isso significa que a GPU fica subutilizada na fase de decode, pois processa apenas um token por vez.

### A ideia central

Em vez de gerar um token por vez com o modelo grande (chamado **target model**), use um modelo menor e mais rapido (chamado **draft model**) para "chutar" varios tokens de uma vez. Depois, o modelo grande verifica todos os tokens chutados em paralelo.

```
Sem speculative decoding:
  Target Model: token1 -> token2 -> token3 -> token4 -> token5
  Tempo: 5 passos sequenciais

Com speculative decoding:
  Draft Model:   [token1, token2, token3, token4, token5]  (rapido, 1 passo)
  Target Model:  verifica todos de uma vez                  (1 passo paralelo)
  Aceitos: token1 ✓, token2 ✓, token3 ✓, token4 ✗
  Resultado: 3 tokens em 2 passos (em vez de 3)
```

### Metodos disponiveis no vLLM

O vLLM suporta varios metodos de speculative decoding:

**1. N-gram Speculation:** usa padroes de n-gramas do proprio prompt para prever os proximos tokens. Nao requer modelo adicional.

```bash
# vLLM com n-gram speculation
vllm serve Qwen/Qwen3-32B \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5}'
```

**2. Eagle3:** modelo especulador treinado especificamente para um modelo target. Atinge taxas de aceitacao de 70-85%.

```bash
# vLLM com Eagle3
vllm serve Qwen/Qwen3-32B \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-32B-speculator.eagle3",
    "num_speculative_tokens": 3
  }' \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95
```

**3. Draft Model (modelo rascunho):** usa um modelo menor da mesma familia como especulador.

### Quando usar

- **Baixa concorrencia (1-4 usuarios):** speculative decoding brilha. Reduz latencia em 1.5-2.5x
- **Alta concorrencia (50+ usuarios):** o ganho diminui porque a GPU ja esta saturada com batching. O draft model compete por recursos

---

## 6.4 Continuous Batching vs Static Batching

Como visto no Capitulo 4, batching agrupa requisicoes para processamento paralelo. A diferenca entre static e continuous batching e critica para a eficiencia do sistema.

### Static Batching

```
Tempo ->
Req A: [████████████ DONE ▒▒▒▒▒▒▒▒▒▒▒]  <- GPU ociosa apos terminar
Req B: [████████████████████████████ DONE]  <- mais longa define o lote
Req C: [ESPERANDO NA FILA...              ]  <- so entra no proximo lote
```

O lote inteiro espera pela requisicao mais demorada. Recursos desperdicados.

### Continuous Batching

```
Tempo ->
Req A: [████████████ DONE]
Req C: [               ██████████████ DONE]  <- entra quando A sai
Req B: [████████████████████████████ DONE  ]
Req D: [                             █████ ]  <- entra quando B sai
```

Cada slot e imediatamente reutilizado. A GPU nunca espera. O continuous batching e implementado pelo vLLM, SGLang e TensorRT-LLM como mecanismo padrao.

O resultado pratico e um **throughput 2-10x maior** comparado ao static batching, especialmente com requisicoes de tamanhos variados (o caso real mais comum).

---

## 6.5 Prefill vs Decode -- as duas fases da geracao

Toda requisicao de LLM passa por duas fases distintas, cada uma com perfil computacional diferente:

### Fase de Prefill (processamento do prompt)

- **O que faz:** processa todos os tokens do prompt de entrada em paralelo
- **Perfil:** **compute-bound** -- saturacao da GPU (alta utilizacao de FLOPs)
- **Complexidade:** O(n^2) em relacao ao tamanho do prompt (atencao quadratica)
- **Analogia:** ler e compreender a pergunta inteira de uma vez

### Fase de Decode (geracao de tokens)

- **O que faz:** gera tokens um a um, autoregressivamente
- **Perfil:** **memory-bound** -- gargalo e a largura de banda de memoria
- **Complexidade:** O(n) por token (usando KV cache)
- **Analogia:** escrever a resposta palavra por palavra

### Por que a distincao importa

```
Fase          | GPU Compute | Memory Bandwidth | Bottleneck
--------------+-------------+-------------------+----------
Prefill       | ALTO        | Medio             | Compute
Decode        | BAIXO       | ALTO              | Bandwidth
```

Frameworks avancados separam as duas fases para otimizar cada uma:

- **Prefill:** pode ser paralelizado com chunked prefill (processar o prompt em pedacos enquanto intercala com decoding de outras requisicoes)
- **Decode:** beneficia-se de batching grande (mais requisicoes no mesmo lote amortizam o custo de carregar pesos da memoria)

Essa separacao e uma das razoes pelas quais o vLLM e o TensorRT-LLM conseguem throughput muito superior ao HuggingFace Transformers vanilla.

---

## 6.6 Flash Attention: a revolucao na eficiencia

A atencao (attention) padrao em Transformers tem complexidade O(n^2) em memoria, onde n e o comprimento da sequencia. Para contextos longos (32K, 128K tokens), isso rapidamente estoura a VRAM.

### O problema da atencao padrao

```python
# Atencao padrao (simplificada)
# Q, K, V: tensores de shape [seq_len, dim]
S = Q @ K.T           # shape [seq_len, seq_len] -- ENORME!
P = softmax(S)        # shape [seq_len, seq_len] -- precisa materializar tudo
O = P @ V             # shape [seq_len, dim]
```

Para um contexto de 128K tokens com FP16:
```
Matriz S = 128K x 128K x 2 bytes = 32 GB  (!) -- so para uma camada!
```

### A solucao: Flash Attention

Flash Attention (Dao et al., 2022) reformula o calculo da atencao para **nunca materializar** a matriz S completa. Em vez disso, processa a atencao em blocos (tiles) que cabem no SRAM (cache rapida) da GPU:

1. Divide Q, K, V em blocos pequenos
2. Calcula a atencao bloco por bloco no SRAM (20x mais rapido que HBM)
3. Acumula os resultados de forma numericamente estavel
4. Nunca materializa a matriz S completa na memoria global

Resultados:
- **Memoria:** O(n) em vez de O(n^2)
- **Velocidade:** 2-4x mais rapido que a atencao padrao
- **Contextos longos:** viabiliza contextos de 128K+ tokens que seriam impossiveis antes

Flash Attention v2 e v3 refinam a abordagem com melhor paralelismo e suporte a novos tipos de dados (FP8). Todos os frameworks modernos (vLLM, SGLang, TensorRT-LLM, llama.cpp) utilizam alguma variacao de Flash Attention.

---

## 6.7 Metricas: TTFT, TPS, throughput, latencia P50/P95

Para saber se suas otimizacoes estao funcionando, voce precisa medir. Estas sao as metricas fundamentais para avaliar performance de LLMs em producao.

### TTFT -- Time to First Token

- **O que mede:** tempo entre o envio da requisicao e o recebimento do primeiro token
- **Importancia:** define a percepcao de "velocidade" do usuario. TTFT alto = usuario espera uma tela em branco
- **Valores tipicos:** 50-500 ms (GPU), 500-5000 ms (CPU)
- **Depende de:** tamanho do prompt (fase de prefill) e carga do servidor

### TPS -- Tokens Per Second

- **O que mede:** velocidade de geracao de tokens apos o primeiro
- **Dois niveis:**
  - **Per-user TPS:** tokens/segundo que um usuario individual recebe (~30-100 tok/s local)
  - **System TPS:** total de tokens/segundo que o sistema gera para todos os usuarios
- **Depende de:** modelo, quantizacao, GPU, batch size

### Throughput

- **O que mede:** numero total de requisicoes completadas por unidade de tempo
- **Unidade:** requisicoes/segundo ou tokens/segundo (sistema inteiro)
- **Depende de:** continuous batching, Paged Attention, tamanho medio das respostas

### Latencia P50/P95

- **P50 (mediana):** metade das requisicoes sao mais rapidas que este valor
- **P95:** 95% das requisicoes sao mais rapidas que este valor
- **P99:** 99% das requisicoes sao mais rapidas que este valor
- **Por que P95/P99 importam:** um P50 de 200 ms com P95 de 5000 ms significa que 1 em cada 20 usuarios espera 25x mais. Em producao, isso e inaceitavel

### Outras metricas relevantes

| Metrica | Descricao |
|---------|-----------|
| **TTLT** (Time to Last Token) | Tempo total desde a requisicao ate o ultimo token |
| **TPOT** (Time Per Output Token) | Tempo medio por token gerado (fase decode) |
| **TBT** (Time Between Tokens) | Tempo entre tokens consecutivos -- afeta "fluidez" do streaming |
| **MTPOT** (Max Time Per Output Token) | Pior caso de tempo por token -- indica stalls |
| **Normalized Latency** | Latencia total / numero de tokens -- permite comparar respostas de tamanhos diferentes |

---

## 6.8 Na pratica: medindo performance do seu setup

Vamos medir as metricas principais do seu LLM local. O script abaixo usa Python puro para medir TTFT, TPS e latencia:

```python
"""
benchmark_ollama.py
Mede TTFT, TPS e latencia de um modelo no Ollama.
"""

import json
import time
import requests
from dataclasses import dataclass


@dataclass
class ResultadoBenchmark:
    """Armazena os resultados de uma medicao."""
    ttft_ms: float          # Time to First Token em milissegundos
    tps: float              # Tokens por segundo (decode)
    latencia_total_ms: float  # Tempo total da requisicao
    tokens_gerados: int     # Numero de tokens na resposta


def medir_requisicao(
    prompt: str,
    modelo: str = "llama3.2:3b",
    url: str = "http://localhost:11434/api/generate",
) -> ResultadoBenchmark:
    """
    Envia uma requisicao com streaming e mede as metricas.
    """
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": True,
    }

    inicio = time.perf_counter()
    primeiro_token_recebido = False
    ttft = 0.0
    num_tokens = 0

    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for linha in resp.iter_lines():
            if not linha:
                continue
            dados = json.loads(linha)
            token = dados.get("response", "")

            if token and not primeiro_token_recebido:
                ttft = (time.perf_counter() - inicio) * 1000  # ms
                primeiro_token_recebido = True

            if token:
                num_tokens += 1

            if dados.get("done", False):
                break

    fim = time.perf_counter()
    latencia_total = (fim - inicio) * 1000  # ms

    # TPS = tokens gerados / tempo de decode (exclui prefill/TTFT)
    tempo_decode = (latencia_total - ttft) / 1000  # segundos
    tps = num_tokens / tempo_decode if tempo_decode > 0 else 0

    return ResultadoBenchmark(
        ttft_ms=round(ttft, 1),
        tps=round(tps, 1),
        latencia_total_ms=round(latencia_total, 1),
        tokens_gerados=num_tokens,
    )


def main() -> None:
    """Executa benchmark com diferentes prompts."""
    prompts = [
        "Explique o que e deep learning em 2 frases.",
        "Escreva um codigo Python que calcula o fatorial de um numero.",
        "Quais sao as vantagens de rodar LLMs localmente?",
    ]

    modelo = "llama3.2:3b"
    print(f"Benchmark: {modelo}")
    print(f"{'Prompt':<55} {'TTFT':>8} {'TPS':>8} {'Total':>8} {'Tokens':>8}")
    print("-" * 95)

    resultados = []
    for prompt in prompts:
        r = medir_requisicao(prompt, modelo=modelo)
        resultados.append(r)
        print(
            f"{prompt[:53]:<55} "
            f"{r.ttft_ms:>6.0f}ms "
            f"{r.tps:>6.1f} "
            f"{r.latencia_total_ms:>6.0f}ms "
            f"{r.tokens_gerados:>6}"
        )

    # Resumo
    print("-" * 95)
    ttft_medio = sum(r.ttft_ms for r in resultados) / len(resultados)
    tps_medio = sum(r.tps for r in resultados) / len(resultados)
    print(f"{'Media':<55} {ttft_medio:>6.0f}ms {tps_medio:>6.1f}")


if __name__ == "__main__":
    main()
```

### Saida esperada (RTX 3060 12GB, Llama 3.2 3B Q4_K_M)

```
Benchmark: llama3.2:3b
Prompt                                                     TTFT      TPS    Total   Tokens
-----------------------------------------------------------------------------------------------
Explique o que e deep learning em 2 frases.                 87ms    72.3   1138ms       76
Escreva um codigo Python que calcula o fatorial de u...    102ms    68.5   2847ms      188
Quais sao as vantagens de rodar LLMs localmente?            91ms    70.1   1821ms      121
-----------------------------------------------------------------------------------------------
Media                                                       93ms    70.3
```

### Interpretando os resultados

- **TTFT < 200 ms:** excelente para aplicacoes interativas
- **TPS > 50:** fluido o suficiente para streaming em tempo real (humanos leem ~3-5 palavras/segundo)
- **TPS > 30:** aceitavel para chatbots
- **TPS < 15:** o usuario percebe lentidao

---

## Resumo do capitulo

1. **KV Cache** elimina recalculos na geracao autoregressiva, mas consome memoria proporcional ao contexto
2. **Paged Attention** resolve o desperdicio de memoria do KV cache com alocacao paginada (vLLM)
3. **Speculative Decoding** usa um modelo pequeno para "chutar" tokens e o modelo grande para verificar, reduzindo latencia em 1.5-2.5x
4. **Continuous Batching** reutiliza slots imediatamente, sem esperar o lote inteiro terminar
5. **Flash Attention** reduz a complexidade de memoria da atencao de O(n^2) para O(n), viabilizando contextos longos
6. **Metricas** essenciais: TTFT, TPS, throughput e latencia P50/P95/P99

No proximo capitulo, comparamos os frameworks que implementam essas otimizacoes: Ollama, vLLM, llama.cpp, SGLang e TensorRT-LLM.

---

## Fontes

1. Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*. O'Reilly Media. Cap. 5 (Performance Challenges), Cap. 7 (LLM-Specific Optimization Techniques).
2. Wang, C. & Hu, P. (2025). Notebooks: `LMCache.ipynb` (KV cache com vLLM), `SpecDecode.ipynb` (speculative decoding com Eagle3 e n-gram). Repositorio: github.com/orca3/llm-model-serving.
3. Troyer, L. (2026). *Benchmarking LLM Serving Systems*. Johannes Kepler University. Secoes 2.2.2 (Decode/Prefill), 2.3 (Performance Metrics), 2.8 (Optimizations).
4. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
5. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
6. Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
7. Yu, G. et al. (2024). *ORCA: A Continuous Batching Framework for Large Language Model Serving*. OSDI 2022.
