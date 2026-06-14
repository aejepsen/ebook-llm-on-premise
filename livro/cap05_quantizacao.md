# Capitulo 5 -- Quantizacao: Rodando Modelos Gigantes em Hardware Modesto

Quantizacao e a tecnica que torna possivel rodar um modelo de 70 bilhoes de parametros numa unica GPU de consumo. Sem ela, voce precisaria de um servidor com 140 GB de VRAM so para carregar os pesos. Com ela, o mesmo modelo cabe em 35 GB -- ou ate menos. Este capitulo explica o que e quantizacao, como funciona e quando cada formato faz sentido.

---

## 5.1 O que e quantizacao e por que importa

Quando um modelo de linguagem e treinado, cada parametro (peso) e armazenado como um numero de ponto flutuante com 32 bits de precisao (FP32). Um modelo com 7 bilhoes de parametros ocupa, portanto, 28 GB em FP32.

**Quantizacao** e o processo de representar esses numeros com menos bits. Em vez de usar 32 bits por peso, podemos usar 16, 8 ou ate 4 bits. Cada reducao pela metade corta o tamanho do modelo (e o consumo de memoria) pela metade.

```
FP32:  1 parametro = 4 bytes   -> 7B modelo = 28 GB
FP16:  1 parametro = 2 bytes   -> 7B modelo = 14 GB
INT8:  1 parametro = 1 byte    -> 7B modelo =  7 GB
INT4:  1 parametro = 0.5 byte  -> 7B modelo =  3.5 GB
```

A grande pergunta e: **quanto de qualidade voce perde?** A resposta depende do metodo de quantizacao, do modelo e da tarefa. Com metodos modernos como GPTQ e AWQ, a perda de qualidade em modelos com 7B+ parametros e frequentemente imperceptivel para a maioria das aplicacoes praticas.

---

## 5.2 Tipos de dados: FP32, FP16, BF16, INT8, INT4

Cada tipo de dado tem suas caracteristicas. Entender a diferenca e fundamental para escolher a quantizacao certa.

### FP32 (Float 32 bits)

- **Precisao:** Maxima. 1 bit de sinal + 8 bits de expoente + 23 bits de mantissa
- **Uso de memoria:** 4 bytes por parametro
- **Quando usar:** Treinamento. Raramente faz sentido para inferencia

### FP16 (Float 16 bits)

- **Precisao:** Alta. 1 + 5 + 10 bits
- **Uso de memoria:** 2 bytes por parametro
- **Quando usar:** Inferencia em GPU com Tensor Cores (RTX 3000+). Bom equilibrio entre velocidade e qualidade
- **Risco:** Range numerico menor que FP32, pode causar overflow em alguns modelos

### BF16 (BFloat 16 bits)

- **Precisao:** Semelhante ao FP16 em expoente, menor em mantissa. 1 + 8 + 7 bits
- **Uso de memoria:** 2 bytes por parametro
- **Quando usar:** Preferivel ao FP16 para modelos grandes. Mesmo range do FP32 com menos precisao na mantissa. Suportado em GPUs Ampere (RTX 3000) e superiores
- **Vantagem:** Nao sofre de overflow como FP16

### INT8 (Inteiro de 8 bits)

- **Precisao:** Boa. Valores inteiros de -128 a 127
- **Uso de memoria:** 1 byte por parametro
- **Quando usar:** Boa reducao de tamanho com perda minima de qualidade. Funciona bem para a maioria dos modelos 7B+

### INT4 (Inteiro de 4 bits)

- **Precisao:** Aceitavel. Valores inteiros de -8 a 7 (ou 0 a 15 sem sinal)
- **Uso de memoria:** 0.5 byte por parametro
- **Quando usar:** Quando a VRAM e severamente limitada. Perda de qualidade perceptivel em modelos pequenos (< 3B), aceitavel em modelos grandes (7B+)

### Tabela comparativa

| Tipo | Bits | Bytes/param | 7B modelo | Velocidade relativa | Qualidade |
|------|------|------------|-----------|-------------------|-----------|
| FP32 | 32 | 4.0 | 28 GB | 1.0x (baseline) | Maxima |
| FP16 | 16 | 2.0 | 14 GB | ~2x | Excelente |
| BF16 | 16 | 2.0 | 14 GB | ~2x | Excelente |
| INT8 | 8 | 1.0 | 7 GB | ~2-3x | Muito boa |
| INT4 | 4 | 0.5 | 3.5 GB | ~3-4x | Boa |

---

## 5.3 GPTQ vs AWQ vs GGUF -- quando usar cada um

Existem tres formatos dominantes de quantizacao. Cada um foi projetado para cenarios diferentes.

### GPTQ (GPT Quantization)

- **O que e:** Metodo de quantizacao pos-treinamento (PTQ) que usa um pequeno dataset de calibracao para minimizar o erro de quantizacao camada por camada
- **Formato:** Modelos salvos em formato Safetensors/HuggingFace
- **Runtime:** vLLM, SGLang, HuggingFace Transformers
- **Precisao tipica:** INT4, INT8
- **Melhor para:** Inferencia em GPU com frameworks como vLLM
- **Exemplo de modelo:** `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`

### AWQ (Activation-Aware Weight Quantization)

- **O que e:** Evolucao do GPTQ. Em vez de tratar todos os pesos igualmente, o AWQ identifica quais pesos sao mais importantes para as ativacoes do modelo e os preserva com maior precisao
- **Formato:** Safetensors/HuggingFace
- **Runtime:** vLLM, SGLang, HuggingFace Transformers
- **Precisao tipica:** INT4 (W4A16 -- pesos em 4 bits, ativacoes em 16 bits)
- **Melhor para:** Quando voce quer a melhor qualidade possivel com INT4 em GPU
- **Exemplo de modelo:** `Qwen/Qwen2.5-7B-Instruct-AWQ`

### GGUF (GPT-Generated Unified Format)

- **O que e:** Formato criado pelo projeto llama.cpp. Um unico arquivo binario que contem tudo: pesos quantizados, tokenizer, metadados
- **Runtime:** llama.cpp, Ollama (que usa llama.cpp internamente)
- **Precisao tipica:** De Q2_K ate Q8_0 (veja secao 5.7)
- **Melhor para:** Inferencia em CPU ou GPU com pouca VRAM. Formato padrao do Ollama
- **Exemplo de modelo:** `bartowski/Qwen2.5-7B-Instruct-GGUF`

### Tabela de decisao

| Cenario | Formato recomendado |
|---------|-------------------|
| GPU NVIDIA + vLLM/SGLang | AWQ ou GPTQ |
| GPU NVIDIA + Ollama | GGUF |
| CPU only | GGUF |
| Apple Silicon (Mac M1/M2/M3) | GGUF |
| AMD GPU (ROCm) | GPTQ via vLLM |
| Maximo throughput em producao | AWQ + vLLM |

---

## 5.4 Impacto na qualidade: perplexidade antes e depois

**Perplexidade** e a metrica padrao para medir a qualidade de um modelo de linguagem. Quanto menor, melhor. Ela mede o quao "surpreso" o modelo fica ao ver texto real -- um modelo com perplexidade 5 esta muito mais confiante (e preciso) que um com perplexidade 50.

Resultados tipicos de perplexidade para um modelo Llama 3.1 8B (medidos no dataset WikiText-2):

| Quantizacao | Perplexidade | Degradacao |
|-------------|-------------|------------|
| FP16 (baseline) | 6.14 | -- |
| Q8_0 | 6.16 | +0.3% |
| Q6_K | 6.18 | +0.7% |
| Q5_K_M | 6.23 | +1.5% |
| Q4_K_M | 6.39 | +4.1% |
| Q3_K_M | 6.95 | +13.2% |
| Q2_K | 8.42 | +37.1% |

**Conclusao pratica:** Q4_K_M oferece o melhor equilibrio entre tamanho e qualidade para a maioria dos casos. A degradacao de ~4% e imperceptivel em conversas e tarefas gerais. Ja Q2_K degrada significativamente e so deve ser usado quando a memoria e extremamente limitada.

Para modelos maiores (70B+), a degradacao por quantizacao e menor. Um Llama 70B em Q4_K_M mantem qualidade proxima ao FP16 de um modelo de 13B.

---

## 5.5 Na pratica: convertendo um modelo para GGUF

O processo de conversao requer o repositorio llama.cpp e Python:

```bash
# 1. Clonar o llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. Instalar dependencias Python
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# 3. Baixar o modelo original do HuggingFace
# (requer huggingface-cli instalado e autenticado)
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/llama-3.2-3b-instruct

# 4. Converter para GGUF (FP16)
python convert_hf_to_gguf.py ./models/llama-3.2-3b-instruct \
  --outfile ./models/llama-3.2-3b-instruct-f16.gguf \
  --outtype f16

# 5. Quantizar para Q4_K_M
./llama-quantize \
  ./models/llama-3.2-3b-instruct-f16.gguf \
  ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
  Q4_K_M
```

Resultado:

```
Arquivo original (FP16): 6.4 GB
Arquivo quantizado (Q4_K_M): 2.0 GB   <- reducao de 69%
```

Para usar o modelo quantizado no Ollama, crie um Modelfile:

```dockerfile
# Modelfile
FROM ./models/llama-3.2-3b-instruct-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM """Voce e um assistente util que responde em portugues."""
```

```bash
# Registrar no Ollama
ollama create meu-llama -f Modelfile

# Testar
ollama run meu-llama "Ola, como voce esta?"
```

---

## 5.6 Tabela: modelo x quantizacao x VRAM necessaria

Referencia pratica para decidir o que roda no seu hardware:

| Modelo | Q4_K_M | Q5_K_M | Q8_0 | FP16 | GPU recomendada |
|--------|--------|--------|------|------|----------------|
| Phi-3 3.8B | 2.4 GB | 2.8 GB | 4.1 GB | 7.6 GB | RTX 3060 (8 GB) |
| Llama 3.2 3B | 2.0 GB | 2.4 GB | 3.4 GB | 6.4 GB | RTX 3060 (8 GB) |
| Qwen 2.5 7B | 4.4 GB | 5.1 GB | 7.6 GB | 14.2 GB | RTX 4060 Ti (16 GB) |
| Llama 3.1 8B | 4.9 GB | 5.7 GB | 8.5 GB | 16.1 GB | RTX 4060 Ti (16 GB) |
| Qwen 2.5 14B | 8.7 GB | 10.1 GB | 15.0 GB | 28.0 GB | RTX 3090 (24 GB) |
| Qwen 2.5 32B | 19.9 GB | 23.1 GB | 34.2 GB | 64.0 GB | 2x RTX 3090 |
| Llama 3.1 70B | 40.6 GB | 47.5 GB | 74.0 GB | 140 GB | 2x A100 (80 GB) |

**Nota:** valores aproximados incluindo overhead do KV cache para contexto de 4096 tokens.

---

## 5.7 Q4_K_M, Q5_K_M, Q8_0 -- o que significam esses nomes

Os nomes dos niveis de quantizacao GGUF seguem um padrao. Vamos decodifica-lo:

```
Q4_K_M
│ │ │
│ │ └─ M = Medium (qualidade media dentro da faixa)
│ │      S = Small (menor/mais compacto)
│ │      L = Large (maior/mais preciso)
│ │
│ └─── K = K-quant (metodo de quantizacao avancado)
│       Usa k-means clustering para minimizar erro
│       Diferente do metodo simples (sem K)
│
└───── 4 = 4 bits por peso (media)
        Pode variar entre camadas no K-quant
```

Os principais niveis disponiveis, do mais compacto ao mais preciso:

| Nivel | Bits medios | Tamanho relativo | Qualidade | Uso recomendado |
|-------|-----------|-----------------|-----------|----------------|
| Q2_K | 2.5 | 0.30x do FP16 | Baixa | Emergencia de memoria |
| Q3_K_S | 3.0 | 0.35x | Aceitavel | Testes rapidos |
| Q3_K_M | 3.4 | 0.38x | Razoavel | Dispositivos moveis |
| Q4_K_S | 4.2 | 0.46x | Boa | Uso geral (memoria limitada) |
| **Q4_K_M** | **4.6** | **0.50x** | **Boa** | **Uso geral (recomendado)** |
| Q5_K_S | 5.2 | 0.57x | Muito boa | Quando qualidade importa |
| **Q5_K_M** | **5.6** | **0.61x** | **Muito boa** | **Equilibrio ideal** |
| Q6_K | 6.5 | 0.70x | Excelente | Quase sem perda |
| **Q8_0** | **8.0** | **0.85x** | **Quase FP16** | **Maximo com quantizacao** |

**Dica:** o sufixo `K` indica que o metodo usa quantizacao por k-means, que distribui os bits de forma inteligente entre as camadas. Camadas mais sensiveis recebem mais bits. Isso e significativamente melhor que distribuir bits uniformemente.

**Por que Q4_K_M e a escolha mais popular?**

1. Reduz o tamanho para ~50% do FP16
2. Degradacao de perplexidade tipicamente abaixo de 5%
3. Velocidade de inferencia superior ao FP16 (menos dados para mover na memoria)
4. Permite rodar modelos maiores no mesmo hardware

---

## 5.8 Trade-off qualidade vs velocidade vs memoria

A quantizacao nao e apenas sobre economizar memoria. Ela tambem afeta velocidade e qualidade de formas nao intuitivas.

### Velocidade

Modelos quantizados sao frequentemente **mais rapidos** que modelos FP16 porque:

1. **Menos dados na memoria:** a largura de banda de memoria (memory bandwidth) e o gargalo principal na inferencia de LLMs. Transferir 4 bits por peso e 4x mais rapido que transferir 16 bits
2. **Melhor uso do cache da GPU:** modelos menores cabem melhor nos caches L2 da GPU
3. **Maior batch size:** com menos VRAM por modelo, sobra mais espaco para processar mais requisicoes simultaneas

### Quando a quantizacao prejudica

- **Modelos pequenos (< 3B):** cada parametro carrega mais informacao. Quantizar agressivamente (Q3 ou menos) causa degradacao perceptivel
- **Tarefas de raciocinio complexo:** matematica, logica formal e programacao sofrem mais com quantizacao que tarefas de texto livre
- **Linguas de baixo recurso:** se o modelo tem poucos dados de treinamento em uma lingua, quantizacao amplifica as deficiencias

### Regra pratica

```
Se VRAM permite -> use Q8_0 ou FP16
Se VRAM e limitada -> use Q4_K_M (melhor custo-beneficio)
Se VRAM e muito limitada -> use Q4_K_S
Nunca use Q2_K em producao
```

---

## Resumo do capitulo

1. **Quantizacao** reduz a precisao numerica dos pesos para diminuir o uso de memoria
2. **FP16/BF16** oferecem qualidade proxima ao FP32 com metade da memoria
3. **INT4 (Q4_K_M)** e o ponto ideal para a maioria dos cenarios praticos
4. **GPTQ e AWQ** sao para GPU + vLLM/SGLang; **GGUF** e para llama.cpp/Ollama
5. A degradacao de qualidade e geralmente aceitavel em modelos 7B+ com Q4_K_M
6. Modelos quantizados sao frequentemente **mais rapidos** por reduzir gargalos de bandwidth

No proximo capitulo, exploramos as tecnicas de otimizacao de inferencia que vao alem da quantizacao: KV cache, Paged Attention, Speculative Decoding e Flash Attention.

---

## Fontes

1. Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*. O'Reilly Media. Cap. 6 (Quantization).
2. Wang, C. & Hu, P. (2025). Notebook de referencia: `quantization_3way_300.ipynb` -- comparativo GPTQ vs AWQ vs FP8 com benchmark vLLM. Repositorio: github.com/orca3/llm-model-serving.
3. Troyer, L. (2026). *Benchmarking LLM Serving Systems*. Johannes Kepler University. Secoes 2.8.16 (Quantization), 3.4 (Quantization Selection).
4. Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023.
5. Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024.
6. Repositorio llama.cpp. *GGUF format specification*. Disponivel em: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
7. Repositorio llama.cpp. *Quantization types and perplexity benchmarks*. Disponivel em: https://github.com/ggerganov/llama.cpp/discussions/2094
