# Capítulo 4 -- Primeiros Passos com LLMs Locais

Até aqui você já entende o que são LLMs, por que roda-los localmente e qual hardware precisa. Agora e hora de colocar a mão na massa. Neste capítulo, você vai baixar seu primeiro modelo, conversar com ele pelo terminal, consumir sua API REST e construir um chatbot simples em Python. Tudo rodando na sua maquina, sem depender de nenhum serviço na nuvem.

---

## 4.1 Baixando seu primeiro modelo com Ollama

O Ollama e a forma mais simples de rodar LLMs localmente. Com um único comando, ele baixa o modelo, configura o runtime e expoe uma API REST. Pense nele como o "Docker dos LLMs".

Depois de instalar o Ollama (veja o Capítulo 3), abra o terminal e execute:

```bash
# Baixar e executar o modelo Llama 3.2 de 3 bilhoes de parametros
ollama pull llama3.2:3b
```

O que acontece nos bastidores:

1. O Ollama consulta o registro de modelos (registry.ollama.com)
2. Baixa os arquivos do modelo no formato GGUF (quantizado)
3. Armazena em `~/.ollama/models/`
4. Verifica a integridade via hash SHA256

O download do `llama3.2:3b` e de aproximadamente 2 GB. Para modelos maiores como o `llama3.1:70b`, espere algo em torno de 40 GB.

Para listar os modelos já baixados:

```bash
ollama list
```

Saida esperada:

```
NAME              ID            SIZE    MODIFIED
llama3.2:3b       a80c4f17acd5  2.0 GB  2 minutes ago
```

Para remover um modelo que não usa mais:

```bash
ollama rm llama3.2:3b
```

---

## 4.2 Modelos populares: Llama, Qwen, Mistral, Phi

Nem todo modelo serve para tudo. A tabela abaixo compara as familias mais relevantes para uso local em 2025/2026:

| Familia | Desenvolvedor | Tamanhos disponiveis | Forca principal | Licença |
|---------|--------------|---------------------|-----------------|---------|
| **Llama 3.2/3.3** | Meta | 1B, 3B, 8B, 70B, 405B | Equilibrio geral, multilingual | Llama 3.3 Community |
| **Qwen 2.5/3** | Alibaba | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Código, matemática, multilingual | Apache 2.0 |
| **Mistral** | Mistral AI | 7B, 8x7B (MoE), 8x22B | Eficiência por parâmetro | Apache 2.0 |
| **Phi-3/4** | Microsoft | 3.8B, 7B, 14B | Modelos pequenos com alta qualidade | MIT |
| **Gemma 2/3** | Google | 2B, 9B, 27B | Pesquisa, raciocinio | Gemma License |
| **DeepSeek-R1** | DeepSeek | 1.5B, 7B, 8B, 14B, 32B, 70B, 671B | Raciocinio profundo (chain-of-thought) | MIT |

**Qual escolher para comecar?**

- **Maquina com 8 GB de VRAM (RTX 3060/4060):** Qwen 2.5:7B ou Llama 3.2:3B
- **Maquina com 16 GB de VRAM (RTX 4080):** Qwen 2.5:14B ou Llama 3.1:8B
- **Maquina com 24 GB de VRAM (RTX 3090/4090):** Qwen 2.5:32B (quantizado Q4) ou DeepSeek-R1:14B
- **Apenas CPU (16 GB RAM):** Phi-3:3.8B ou Llama 3.2:1B

---

## 4.3 Tamanhos de modelo: o que cabe na sua GPU

O tamanho de um modelo em memória depende de dois fatores: o número de parâmetros e a precisão numerica (quantização). A regra prática e:

```
Memoria (GB) ≈ Parametros (bilhoes) x Bytes por parametro
```

| Parâmetros | FP16 (2 bytes) | Q8 (1 byte) | Q4 (0.5 byte) |
|-----------|---------------|-------------|---------------|
| 1B | 2 GB | 1 GB | 0.5 GB |
| 3B | 6 GB | 3 GB | 1.5 GB |
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 30B | 60 GB | 30 GB | 15 GB |
| 70B | 140 GB | 70 GB | 35 GB |

**Atenção:** esses valores são apenas para os pesos do modelo. Na prática, você precisa de memória adicional para o KV cache (cache de atenção), que cresce com o tamanho do contexto. Uma margem segura e reservar 20-30% além do tamanho do modelo.

**Exemplo prático:** um modelo de 7B quantizado em Q4_K_M ocupa cerca de 4.4 GB. Numa GPU com 8 GB de VRAM, isso deixa ~3.6 GB para o KV cache, o que permite contextos de até ~8.000 tokens confortavelmente.

Para verificar quanto de VRAM sua GPU tem:

```bash
# NVIDIA
nvidia-smi

# Saida relevante:
# |   0  NVIDIA GeForce RTX 3060   |   0 MiB / 12288 MiB |
```

---

## 4.4 Primeira conversa: ollama run

O comando mais simples para interagir com um LLM local:

```bash
ollama run llama3.2:3b
```

Isso abre um prompt interativo. Digite sua pergunta e pressione Enter:

```
>>> Explique o que e machine learning em 3 frases simples.

Machine learning e um ramo da inteligencia artificial onde computadores
aprendem padroes a partir de dados, sem serem programados explicitamente
para cada tarefa. O modelo analisa exemplos, identifica regularidades e
usa esse conhecimento para fazer previsoes sobre dados novos. Quanto mais
dados de qualidade o modelo recebe, melhores suas previsoes se tornam.

>>> /bye
```

Opcoes uteis:

```bash
# Definir o system prompt diretamente
ollama run llama3.2:3b --system "Voce e um assistente tecnico que responde em portugues."

# Passar uma pergunta sem modo interativo
ollama run llama3.2:3b "O que e uma GPU?"
```

---

## 4.5 API REST do Ollama

O Ollama expoe uma API REST na porta 11434. Isso é o que torna possível integrar LLMs locais com qualquer linguagem de programação.

### 4.5.1 /api/generate -- Geração de texto

A rota mais básica. Envia um prompt, recebe a resposta:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "O que e quantizacao em modelos de linguagem?",
  "stream": false
}'
```

Resposta (simplificada):

```json
{
  "model": "llama3.2:3b",
  "response": "Quantizacao e o processo de reduzir a precisao numerica...",
  "done": true,
  "total_duration": 2841503200,
  "eval_count": 156,
  "eval_duration": 2301000000
}
```

Campos importantes da resposta:
- `eval_count`: número de tokens gerados
- `eval_duration`: tempo de geração em nanosegundos
- **Tokens por segundo** = `eval_count / (eval_duration / 1e9)` = 156 / 2.3 ≈ 67.8 tok/s

### 4.5.2 /api/chat -- Conversa com histórico

Para aplicações de chat, use a rota que aceita mensagens com roles:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "system", "content": "Voce e um assistente tecnico."},
    {"role": "user", "content": "O que e um transformer?"},
    {"role": "assistant", "content": "Um transformer e uma arquitetura de rede neural..."},
    {"role": "user", "content": "E o mecanismo de atencao?"}
  ],
  "stream": false
}'
```

### 4.5.3 /api/embed -- Gerar embeddings

Para aplicações de busca semântica e RAG (Retrieval-Augmented Generation):

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "llama3.2:3b",
  "input": "LLMs on-premise oferecem controle total sobre os dados"
}'
```

A resposta contém um vetor numerico (embedding) que representa o significado semântico do texto. Esses vetores podem ser armazenados em bancos vetoriais como Qdrant, ChromaDB ou Milvus.

---

## 4.6 Batching: processando multiplas requisicoes

Quando você envia uma única pergunta por vez, a GPU fica subutilizada. **Batching** e a técnica de agrupar multiplas requisicoes para processamento simultaneo, maximizando o uso do hardware.

Existem dois tipos fundamentais:

### Static Batching (lote estático)

Todas as requisicoes do lote são processadas juntas. O lote só e liberado quando **todas** as respostas terminam. Se uma resposta tem 10 tokens e outra tem 500, a GPU fica ociosa esperando a mais longa.

### Continuous Batching (lote continuo)

Assim que uma requisicao termina, seu slot e imediatamente ocupado por uma nova requisicao da fila. A GPU nunca fica ociosa esperando. Essa e a abordagem usada pelo vLLM e outros frameworks modernos.

```
Static Batching:
[Req A ████████████████████         ]  <- esperando B terminar
[Req B ██████████████████████████████]
[Req C                               ]  <- esperando na fila

Continuous Batching:
[Req A ████████████ | Req C ██████████████]  <- C entra quando A sai
[Req B ██████████████████████████████████ ]
```

Na prática com Ollama, você pode enviar multiplas requisicoes simultaneas e o servidor gerência a fila internamente. Para maior controle de batching, frameworks como vLLM são mais adequados (veja Capítulo 7).

---

## 4.7 Streaming: respostas em tempo real

Por padrão, a API do Ollama retorna a resposta completa de uma vez. Com **streaming**, os tokens são enviados conforme são gerados, permitindo que o usuario veja a resposta sendo construida em tempo real -- exatamente como acontece no ChatGPT.

### Como funciona

O streaming usa o padrão **Server-Sent Events (SSE)**: o servidor mantém a conexao HTTP aberta e envia fragmentos de dados conforme ficam prontos.

```bash
# Com stream habilitado (padrao no curl)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Conte uma historia curta sobre IA",
  "stream": true
}'
```

Cada linha da resposta é um JSON independente:

```json
{"model":"llama3.2:3b","response":"Era","done":false}
{"model":"llama3.2:3b","response":" uma","done":false}
{"model":"llama3.2:3b","response":" vez","done":false}
...
{"model":"llama3.2:3b","response":"","done":true,"total_duration":...}
```

### Por que streaming importa

- **Experiência do usuario:** a percepcao de velocidade melhora dramaticamente. O usuario comeca a ler enquanto o modelo ainda gera
- **Time to First Token (TTFT):** com streaming, o TTFT e o tempo até o primeiro token aparecer, geralmente menos de 200 ms em modelos pequenos
- **Aplicações interativas:** chatbots, assistentes de código e interfaces conversacionais dependem de streaming para parecerem naturais

---

## 4.8 Exemplo prático: chatbot simples com Python + Ollama

Vamos construir um chatbot funcional que mantém histórico de conversa e usa streaming. Este código roda com Python 3.10+ e a biblioteca `requests` (já inclusa na maioria das instalações):

```python
"""
chatbot_ollama.py
Chatbot local com streaming usando Ollama.
Requisitos: Python 3.10+, Ollama rodando na porta 11434.
"""

import json
import requests
from typing import Generator

# Configuração do modelo
OLLAMA_URL = "http://localhost:11434/api/chat"
MODELO = "llama3.2:3b"

def enviar_mensagem(
    mensagens: list[dict],
    modelo: str = MODELO,
) -> Generator[str, None, None]:
    """
    Envia mensagens para o Ollama e retorna tokens via streaming.

    Args:
        mensagens: Lista de dicts com 'role' e 'content'.
        modelo: Nome do modelo no Ollama.

    Yields:
        Cada token gerado pelo modelo.
    """
    payload = {
        "model": modelo,
        "messages": mensagens,
        "stream": True,
    }

    # stream=True no requests permite ler a resposta linha por linha
    with requests.post(OLLAMA_URL, json=payload, stream=True) as resposta:
        resposta.raise_for_status()
        for linha in resposta.iter_lines():
            if linha:
                dados = json.loads(linha)
                # Cada chunk contem um fragmento da resposta
                token = dados.get("message", {}).get("content", "")
                if token:
                    yield token
                # Quando done=True, a geração terminou
                if dados.get("done", False):
                    break


def main() -> None:
    """Loop principal do chatbot."""
    print("=" * 60)
    print(f"  Chatbot Local | Modelo: {MODELO}")
    print(f"  Digite 'sair' para encerrar.")
    print("=" * 60)

    # Historico da conversa -- mantem contexto entre turnos
    historico: list[dict] = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente tecnico especializado em IA. "
                "Responda em portugues brasileiro de forma clara e direta."
            ),
        }
    ]

    while True:
        # Captura input do usuario
        print()
        pergunta = input("Voce: ").strip()

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit", "/bye"):
            print("\nAte logo!")
            break

        # Adiciona a pergunta ao historico
        historico.append({"role": "user", "content": pergunta})

        # Envia para o modelo e imprime com streaming
        print(f"\n{MODELO}: ", end="", flush=True)
        resposta_completa = []

        for token in enviar_mensagem(historico):
            print(token, end="", flush=True)
            resposta_completa.append(token)

        print()  # Quebra de linha apos a resposta

        # Adiciona a resposta ao historico para manter contexto
        historico.append({
            "role": "assistant",
            "content": "".join(resposta_completa),
        })

        # Limita o historico para nao estourar o contexto do modelo
        # Mantemos o system prompt + ultimas 20 mensagens
        if len(historico) > 21:
            historico = [historico[0]] + historico[-20:]


if __name__ == "__main__":
    main()
```

### Executando o chatbot

```bash
# Terminal 1: certifique-se de que o Ollama esta rodando
ollama serve

# Terminal 2: execute o chatbot
python chatbot_ollama.py
```

### Saida esperada

```
============================================================
  Chatbot Local | Modelo: llama3.2:3b
  Digite 'sair' para encerrar.
============================================================

Voce: O que e KV cache?

llama3.2:3b: KV cache (Key-Value cache) e uma tecnica de otimizacao
usada em modelos Transformer durante a fase de geração de texto.
Quando o modelo gera tokens sequencialmente, ele precisa recalcular
a atencao sobre todos os tokens anteriores. O KV cache armazena os
vetores Key e Value ja computados, evitando recalculos redundantes
e acelerando significativamente a inferencia.

Voce: E por que ele usa tanta memoria?

llama3.2:3b: O KV cache cresce linearmente com o comprimento do
contexto e com o numero de camadas do modelo...
```

---

## Resumo do capítulo

Neste capítulo você:

1. **Baixou e gerenciou modelos** com `ollama pull` e `ollama list`
2. **Comparou familias de modelos** (Llama, Qwen, Mistral, Phi) e seus casos de uso
3. **Entendeu a relação entre tamanho do modelo, quantização e VRAM**
4. **Conversou com um LLM** diretamente pelo terminal
5. **Explorou a API REST** do Ollama: geração, chat e embeddings
6. **Aprendeu sobre batching e streaming** -- os dois pilares de performance em inferência
7. **Construiu um chatbot funcional** em Python com streaming e histórico de contexto

No próximo capítulo, vamos mergulhar na **quantização** -- a técnica que torna possível rodar modelos de 70 bilhoes de parâmetros em uma única GPU de consumo.

> **A jornada deste livro:** O projeto AI-Orchestrator — que serve de fio condutor para este livro — passou por uma evolução real de modelos que ilustra perfeitamente os trade-offs que você vai enfrentar:
>
> | Fase | Modelo | VRAM | Latência/task | Acurácia Routing | Por que mudou |
> |------|--------|------|---------------|------------------|---------------|
> | **PoC** | `qwen3:30b-a3b` (MoE Q4) | 12 GB GPU + 8 GB RAM | ~55s | 90.5% | Mais capaz, mas 44% transbordava para CPU |
> | **Baseline** | `qwen2.5:7b-instruct-q4_K_M` | 100% GPU | ~7s | 90.5% | 8× mais rápido, mesma acurácia de routing |
> | **Produção** | `qwen3.5-9b-orch` (LoRA) | 100% GPU | ~2-4s | 93.7%* | Fine-tuned + roteamento multi-domínio e Knowledge Graph (Semiose); 14× mais rápido que PoC |
>
> *\*Medido no golden canonicalizado (63 casos) com decomposição multi-domínio e enriquecimento via Knowledge Graph (cap. 22). As fases PoC/Baseline foram medidas no golden anterior (44 casos) — por isso o salto reflete tanto o fine-tuning quanto a evolução do pipeline de roteamento, e não o modelo isoladamente.*
>
> **Lição:** O modelo mais capaz (30B MoE) era o pior para produção. O fine-tuning LoRA de um modelo 9B produziu o melhor equilíbrio entre latência, acurácia e custo de hardware. Esta jornada — escolher, medir, otimizar e especializar — é exatamente o que você aprenderá nos próximos capítulos.

---

## Exercícios

1. **Escolha seu modelo:** Instale 3 modelos de famílias diferentes (Llama, Qwen, Phi) via `ollama pull`. Teste cada um com a mesma pergunta complexa ("Explique o que é backpropagation e por que ela é importante"). Qual respondeu melhor? Qual foi mais rápido? Qual usou mais VRAM? (monitore com `nvidia-smi`).

2. **API em outra linguagem:** Reimplemente o chatbot da seção 4.8 em uma linguagem diferente (JavaScript/Node.js, Go, Rust). A API REST do Ollama é agnóstica de linguagem.

3. **Batching manual:** Use o script de benchmark da seção 6.8 para enviar 3 prompts simultaneamente (threads ou `asyncio.gather`). Compare o throughput total com o benchmark sequencial.

4. **Custo do contexto:** Gere uma resposta curta (50 tokens) e uma longa (500 tokens) no mesmo modelo. Use `nvidia-smi` para medir a VRAM usada. Quanto de VRAM adicional a resposta longa consumiu? (Isso é o KV cache em ação — Capítulo 6.)

## Fontes

1. Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization: Hosting LLMs at Scale*. O'Reilly Media. Cap. 1-2.
2. Troyer, L. (2026). *Benchmarking LLM Serving Systems*. Master Thesis, Johannes Kepler University Linz. Cap. 2 (LLM Serving System, Performance Metrics).
3. Ollama Documentation. Disponível em: https://github.com/ollama/ollama/blob/main/docs/api.md
4. Wang, C. & Hu, P. (2025). Notebooks de referência: `ch2_Streaming.ipynb`, `ch2_Batching.ipynb`. Repositório: github.com/orca3/llm-model-serving.
5. Meta AI (2024). *Llama 3.2 Model Card*. Disponível em: https://github.com/meta-llama/llama-models
6. Qwen Team (2025). *Qwen2.5 Technical Report*. Alibaba Group.
