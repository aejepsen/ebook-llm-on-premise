# Capítulo 5 -- Quantização: Rodando Modelos Gigantes em Hardware Modesto

Quantização e a técnica que torna possível rodar um modelo de 70 bilhoes de parâmetros numa única GPU de consumo. Sem ela, você precisaria de um servidor com 140 GB de VRAM só para carregar os pesos. Com ela, o mesmo modelo cabe em 35 GB -- ou até menos. Este capítulo explica o que é quantização, como funciona é quando cada formato faz sentido.

---

## 5.1 O que é quantização e por que importa

Quando um modelo de linguagem e treinado, cada parâmetro (peso) é armazenado como um número de ponto flutuante com 32 bits de precisão (FP32). Um modelo com 7 bilhoes de parâmetros ocupa, portanto, 28 GB em FP32.

**Quantização** e o processo de representar esses números com menos bits. Em vez de usar 32 bits por peso, podemos usar 16, 8 ou até 4 bits. Cada redução pela metade corta o tamanho do modelo (e o consumo de memória) pela metade.

```
FP32:  1 parametro = 4 bytes   -> 7B modelo = 28 GB
FP16:  1 parametro = 2 bytes   -> 7B modelo = 14 GB
INT8:  1 parametro = 1 byte    -> 7B modelo =  7 GB
INT4:  1 parametro = 0.5 byte  -> 7B modelo =  3.5 GB
```

A grande pergunta e: **quanto de qualidade você perde?** A resposta depende do método de quantização, do modelo e da tarefa. Com métodos modernos como GPTQ e AWQ, a perda de qualidade em modelos com 7B+ parâmetros é frequentemente imperceptível para a maioria das aplicações praticas.

---

## 5.2 Tipos de dados: FP32, FP16, BF16, INT8, INT4

Cada tipo de dado tem suas caracteristicas. Entender a diferença é fundamental para escolher a quantização certa.

### FP32 (Float 32 bits)

- **Precisão:** Maxima. 1 bit de sinal + 8 bits de expoente + 23 bits de mantissa
- **Uso de memória:** 4 bytes por parâmetro
- **Quando usar:** Treinamento. Raramente faz sentido para inferência

### FP16 (Float 16 bits)

- **Precisão:** Alta. 1 + 5 + 10 bits
- **Uso de memória:** 2 bytes por parâmetro
- **Quando usar:** Inferência em GPU com Tensor Cores (RTX 3000+). Bom equilibrio entre velocidade e qualidade
- **Risco:** Range numerico menor que FP32, pode causar overflow em alguns modelos

### BF16 (BFloat 16 bits)

- **Precisão:** Semelhante ao FP16 em expoente, menor em mantissa. 1 + 8 + 7 bits
- **Uso de memória:** 2 bytes por parâmetro
- **Quando usar:** Preferível ao FP16 para modelos grandes. Mesmo range do FP32 com menos precisão na mantissa. Suportado em GPUs Ampere (RTX 3000) e superiores
- **Vantagem:** Não sofre de overflow como FP16

### INT8 (Inteiro de 8 bits)

- **Precisão:** Boa. Valores inteiros de -128 a 127
- **Uso de memória:** 1 byte por parâmetro
- **Quando usar:** Boa redução de tamanho com perda mínima de qualidade. Funciona bem para a maioria dos modelos 7B+

### INT4 (Inteiro de 4 bits)

- **Precisão:** Aceitável. Valores inteiros de -8 a 7 (ou 0 a 15 sem sinal)
- **Uso de memória:** 0.5 byte por parâmetro
- **Quando usar:** Quando a VRAM e severamente limitada. Perda de qualidade perceptível em modelos pequenos (< 3B), aceitável em modelos grandes (7B+)

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

Existem tres formatos dominantes de quantização. Cada um foi projetado para cenarios diferentes.

### GPTQ (GPT Quantization)

- **O que e:** Método de quantização pós-treinamento (PTQ) que usa um pequeno dataset de calibração para minimizar o erro de quantização camada por camada
- **Formato:** Modelos salvos em formato Safetensors/HuggingFace
- **Runtime:** vLLM, SGLang, HuggingFace Transformers
- **Precisão tipica:** INT4, INT8
- **Melhor para:** Inferência em GPU com frameworks como vLLM
- **Exemplo de modelo:** `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`

### AWQ (Activation-Aware Weight Quantization)

- **O que e:** Evolução do GPTQ. Em vez de tratar todos os pesos igualmente, o AWQ identifica quais pesos são mais importantes para as ativações do modelo e os preserva com maior precisão
- **Formato:** Safetensors/HuggingFace
- **Runtime:** vLLM, SGLang, HuggingFace Transformers
- **Precisão tipica:** INT4 (W4A16 -- pesos em 4 bits, ativações em 16 bits)
- **Melhor para:** Quando você quer a melhor qualidade possível com INT4 em GPU
- **Exemplo de modelo:** `Qwen/Qwen2.5-7B-Instruct-AWQ`

### GGUF (GPT-Generated Unified Format)

- **O que e:** Formato criado pelo projeto llama.cpp. Um único arquivo binario que contém tudo: pesos quantizados, tokenizer, metadados
- **Runtime:** llama.cpp, Ollama (que usa llama.cpp internamente)
- **Precisão tipica:** De Q2_K até Q8_0 (veja seção 5.7)
- **Melhor para:** Inferência em CPU ou GPU com pouca VRAM. Formato padrão do Ollama
- **Exemplo de modelo:** `bartowski/Qwen2.5-7B-Instruct-GGUF`

### Tabela de decisao

| Cenario | Formato recomendado |
|---------|-------------------|
| GPU NVIDIA + vLLM/SGLang | AWQ ou GPTQ |
| GPU NVIDIA + Ollama | GGUF |
| CPU only | GGUF |
| Apple Silicon (Mac M1/M2/M3) | GGUF |
| AMD GPU (ROCm) | GPTQ via vLLM |
| Máximo throughput em produção | AWQ + vLLM |

---

## 5.4 Impacto na qualidade: perplexidade antes e depois

**Perplexidade** e a métrica padrão para medir a qualidade de um modelo de linguagem. Quanto menor, melhor. Ela mede o quao "surpreso" o modelo fica ao ver texto real -- um modelo com perplexidade 5 está muito mais confiante (é preciso) que um com perplexidade 50.

Resultados tipicos de perplexidade para um modelo Llama 3.1 8B (medidos no dataset WikiText-2):

| Quantização | Perplexidade | Degradação |
|-------------|-------------|------------|
| FP16 (baseline) | 6.14 | -- |
| Q8_0 | 6.16 | +0.3% |
| Q6_K | 6.18 | +0.7% |
| Q5_K_M | 6.23 | +1.5% |
| Q4_K_M | 6.39 | +4.1% |
| Q3_K_M | 6.95 | +13.2% |
| Q2_K | 8.42 | +37.1% |

**Conclusão prática:** Q4_K_M oferece o melhor equilibrio entre tamanho e qualidade para a maioria dos casos. A degradação de ~4% e imperceptível em conversas e tarefas gerais. Já Q2_K degrada significativamente e só deve ser usado quando a memória e extremamente limitada.

Para modelos maiores (70B+), a degradação por quantização e menor. Um Llama 70B em Q4_K_M mantém qualidade próxima ao FP16 de um modelo de 13B.

---

## 5.5 Na prática: convertendo um modelo para GGUF

O processo de conversão requer o repositório llama.cpp e Python:

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

## 5.6 Tabela: modelo x quantização x VRAM necessária

Referência prática para decidir o que roda no seu hardware:

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

Os nomes dos niveis de quantização GGUF seguem um padrão. Vamos decodifica-lo:

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

| Nível | Bits medios | Tamanho relativo | Qualidade | Uso recomendado |
|-------|-----------|-----------------|-----------|----------------|
| Q2_K | 2.5 | 0.30x do FP16 | Baixa | Emergência de memória |
| Q3_K_S | 3.0 | 0.35x | Aceitável | Testes rapidos |
| Q3_K_M | 3.4 | 0.38x | Razoável | Dispositivos moveis |
| Q4_K_S | 4.2 | 0.46x | Boa | Uso geral (memória limitada) |
| **Q4_K_M** | **4.6** | **0.50x** | **Boa** | **Uso geral (recomendado)** |
| Q5_K_S | 5.2 | 0.57x | Muito boa | Quando qualidade importa |
| **Q5_K_M** | **5.6** | **0.61x** | **Muito boa** | **Equilibrio ideal** |
| Q6_K | 6.5 | 0.70x | Excelente | Quase sem perda |
| **Q8_0** | **8.0** | **0.85x** | **Quase FP16** | **Máximo com quantização** |

**Dica:** o sufixo `K` indica que o método usa quantização por k-means, que distribui os bits de forma inteligente entre as camadas. Camadas mais sensiveis recebem mais bits. Isso é significativamente melhor que distribuir bits uniformemente.

**Por que Q4_K_M e a escolha mais popular?**

1. Reduz o tamanho para ~50% do FP16
2. Degradação de perplexidade tipicamente abaixo de 5%
3. Velocidade de inferência superior ao FP16 (menos dados para mover na memória)
4. Permite rodar modelos maiores no mesmo hardware

---

## 5.8 Trade-off qualidade vs velocidade vs memória

A quantização não é apenas sobre economizar memória. Ela também afeta velocidade e qualidade de formas não intuitivas.

### Velocidade

Modelos quantizados são frequentemente **mais rapidos** que modelos FP16 porque:

1. **Menos dados na memória:** a largura de banda de memória (memory bandwidth) e o gargalo principal na inferência de LLMs. Transferir 4 bits por peso e 4x mais rapido que transferir 16 bits
2. **Melhor uso do cache da GPU:** modelos menores cabem melhor nos caches L2 da GPU
3. **Maior batch size:** com menos VRAM por modelo, sobra mais espaço para processar mais requisicoes simultaneas

### Quando a quantização prejudica

- **Modelos pequenos (< 3B):** cada parâmetro carrega mais informação. Quantizar agressivamente (Q3 ou menos) causa degradação perceptível
- **Tarefas de raciocinio complexo:** matemática, lógica formal e programação sofrem mais com quantização que tarefas de texto livre
- **Linguas de baixo recurso:** se o modelo tem poucos dados de treinamento em uma lingua, quantização amplifica as deficiencias

### Regra prática

```
Se VRAM permite -> use Q8_0 ou FP16
Se VRAM e limitada -> use Q4_K_M (melhor custo-beneficio)
Se VRAM e muito limitada -> use Q4_K_S
Nunca use Q2_K em producao
```

---

## Resumo do capítulo

1. **Quantização** reduz a precisão numerica dos pesos para diminuir o uso de memória
2. **FP16/BF16** oferecem qualidade próxima ao FP32 com metade da memória
3. **INT4 (Q4_K_M)** e o ponto ideal para a maioria dos cenarios praticos
4. **GPTQ e AWQ** são para GPU + vLLM/SGLang; **GGUF** e para llama.cpp/Ollama
5. A degradação de qualidade é geralmente aceitável em modelos 7B+ com Q4_K_M
6. Modelos quantizados são frequentemente **mais rapidos** por reduzir gargalos de bandwidth

No próximo capítulo, exploramos as técnicas de otimização de inferência que vao além da quantização: KV cache, Paged Attention, Speculative Decoding e Flash Attention.

---

## Fontes

1. Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*. O'Reilly Media. Cap. 6 (Quantization).
2. Wang, C. & Hu, P. (2025). Notebook de referência: `quantization_3way_300.ipynb` -- comparativo GPTQ vs AWQ vs FP8 com benchmark vLLM. Repositório: github.com/orca3/llm-model-serving.
3. Troyer, L. (2026). *Benchmarking LLM Serving Systems*. Johannes Kepler University. Secoes 2.8.16 (Quantization), 3.4 (Quantization Selection).
4. Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pré-trained Transformers*. ICLR 2023.
5. Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024.
6. Repositório llama.cpp. *GGUF format specification*. Disponível em: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
7. Repositório llama.cpp. *Quantization types and perplexity benchmarks*. Disponível em: https://github.com/ggerganov/llama.cpp/discussions/2094
