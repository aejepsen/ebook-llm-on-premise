# Glossário

## A

**Acurácia:** Proporção de previsões corretas em relação ao total. No AI-Orchestrator, medida pelos gates de routing (≥90%) e domains (≥80%).

**Agent Context Boundary:** Escopo do que um agente conhece, pode fazer e possui — incluindo vocabulário, ferramentas, objetivos e estado. Conceito do framework ADD.

**Agent-Driven Design (ADD):** Framework conceitual para projetar sistemas onde agentes baseados em LLM são cidadãos de primeira classe. Define a separação Model vs Harness e 4 topologias de agente.

**Atenção (Attention):** Mecanismo que permite ao Transformer pesar a importância de diferentes tokens da sequência de entrada.

**AWQ (Activation-Aware Weight Quantization):** Método de quantização que preserva pesos importantes com maior precisão, baseado na análise das ativações do modelo.

## B

**Batching:** Técnica de agrupar múltiplas requisições para processamento paralelo na GPU. Static batching espera todas terminarem; continuous batching reutiliza slots imediatamente.

**BERT (Bidirectional Encoder Representations from Transformers):** Modelo encoder-only do Google (2018), bidirecional, otimizado para tarefas de compreensão de linguagem.

**BERTimbau:** BERT pré-treinado para português brasileiro (`neuralmind/bert-base-portuguese-cased`). Usado no AI-Orchestrator como detector de injection.

**BF16 (BFloat 16):** Formato de ponto flutuante de 16 bits com o mesmo range do FP32 mas metade da precisão. Melhor que FP16 para treino (range maior evita overflow).

## C

**ChatML:** Formato de marcação de mensagens usado por modelos como Qwen: `<|im_start|>role\ncontent<|im_end|>`.

**Circuit Breaker:** Padrão de resiliência que interrompe chamadas a um serviço após N falhas consecutivas. No AI-Orchestrator: 3 falhas → OPEN 30s → HALF_OPEN.

**Continuous Batching:** Estratégia de batching onde slots de GPU são reutilizados assim que uma requisição termina, sem esperar o lote inteiro. Implementado por vLLM, SGLang e TensorRT-LLM.

**Cosine Similarity:** Medida de similaridade entre vetores, usada em bancos vetoriais para busca semântica. Varia de -1 a 1.

**CUDA:** Plataforma de computação paralela da NVIDIA para GPUs.

## D

**Decode (fase):** Fase de geração autoregressiva de tokens, um por um. Memory-bound — gargalo é largura de banda de memória.

**DeltaNet:** Camadas de atenção alternativas do Qwen3.5, complementares ao Transformer tradicional. Não recebem adapters LoRA no AI-Orchestrator.

## E

**Embedding:** Representação vetorial de texto que captura significado semântico. Usado em busca vetorial e RAG.

**Encoder:** Metade do Transformer que processa a entrada bidirecionalmente. BERT é encoder-only.

## F

**Flash Attention:** Algoritmo de atenção que reduz complexidade de memória de O(n²) para O(n) usando tiling em SRAM. Viabiliza contextos de 128K+ tokens.

**FP16 (Float 16):** Precisão de meia-ponto flutuante. Metade da memória do FP32.

**FP32 (Float 32):** Precisão de ponto flutuante padrão (32 bits).

**Function Calling:** Capacidade do LLM de solicitar a execução de uma função externa, retornando nome da função e argumentos em formato estruturado.

## G

**GGUF (GPT-Generated Unified Format):** Formato de arquivo para modelos quantizados, usado por llama.cpp e Ollama. Suporta múltiplos níveis de quantização (Q4_K_M, Q5_K_M, Q8_0, etc.).

**GQA (Grouped Query Attention):** Variante de atenção onde múltiplos Q-heads compartilham o mesmo par K,V. Reduz o tamanho do KV cache. Usado em Llama, Qwen, Mistral.

**GPTQ (GPT Quantization):** Método de quantização pós-treinamento que usa dataset de calibração para minimizar erro de quantização camada por camada.

## H

**Harness:** Todo o código que envolve o LLM: prompts, ferramentas, memória, roteamento, validação, controle de fluxo. Termo do framework ADD.

**HNSW (Hierarchical Navigable Small World):** Algoritmo de busca aproximada de vizinhos mais próximos usado em bancos vetoriais.

## I

**INT4/INT8:** Precisão inteira de 4 ou 8 bits. Usada em quantização agressiva para reduzir VRAM.

## K

**KV Cache:** Cache que armazena vetores Key e Value já calculados, evitando recomputação durante geração autoregressiva. Complexidade de memória: O(num_camadas × num_kv_heads × dim_head × seq_len).

## L

**LangGraph:** Framework para construir grafos de estado com agentes. Usado no AI-Orchestrator para o pipeline sanitize → classify → dispatch → synthesize.

**Langfuse:** Plataforma de observabilidade nativa para LLMs. Suporta traces, spans, generations e scores.

**Least-Privilege Tool Scope:** Padrão de segurança que concede a cada agente apenas as ferramentas necessárias para seu domínio.

**LoRA (Low-Rank Adaptation):** Método de fine-tuning que adiciona pequenas matrizes adaptadoras (adapters) em camadas específicas, treinando ~1-2% dos parâmetros totais.

## M

**MAST Failure Taxonomy:** Catálogo de 14 modos de falha empiricamente observados em sistemas multi-agente (Cemri et al., NeurIPS 2025).

**MCP (Model Context Protocol):** Protocolo para conectar agentes a ferramentas e fontes de dados externas.

**Model vs Harness:** Distinção fundamental do ADD — o Model (LLM) raciocina e gera; o Harness (código) estrutura, valida e controla.

**MoE (Mixture of Experts):** Arquitetura onde apenas uma fração dos parâmetros é ativada por token. Modelos MoE (ex: Qwen3 30B-A3B) têm muitos parâmetros totais mas uso eficiente de computação.

## O

**Ollama:** Ferramenta para rodar LLMs localmente com um comando. Abstrai download, quantização e serving via API REST.

**On-Premise:** Infraestrutura executada localmente, sem dependência de serviços cloud.

## P

**Paged Attention:** Técnica do vLLM que gerencia KV cache em páginas de tamanho fixo (16 tokens), inspirada em memória virtual. Elimina fragmentação e permite sharing de prefixos.

**Prefill (fase):** Fase de processamento paralelo de todos os tokens do prompt. Compute-bound — satura a GPU.

**Prompt Firewall:** Padrão de segurança que valida e sanitiza prompts antes de chegarem ao modelo.

**Prompt Injection:** Ataque onde o usuário insere instruções maliciosas no prompt para subverter o comportamento do modelo.

## Q

**QLoRA:** LoRA com quantização de 4 bits (NF4) + Double Quantization. Permite fine-tuning de modelos de 65B+ em uma única GPU de 48GB.

**Qdrant:** Banco de dados vetorial usado no AI-Orchestrator para semantic router e busca de embeddings.

**Quantização:** Processo de reduzir a precisão numérica dos pesos do modelo (ex: FP16 → INT4) para reduzir uso de VRAM.

## R

**RAG (Retrieval-Augmented Generation):** Técnica que combina busca em documentos com geração de texto, conectando o LLM a bases de conhecimento externas.

**ReAct:** Padrão de loop de agente: Reasoning (pensar) → Acting (executar ferramenta) → Observation (observar resultado).

**Reranking:** Reordenação de resultados de busca por relevância semântica mais precisa usando cross-encoder.

**RMSNorm (Root Mean Square Normalization):** Variante de Layer Normalization usada em Llama, Qwen e Mistral. Mais eficiente computacionalmente.

## S

**SBERT (Sentence-BERT):** Modelo de embeddings que gera vetores semanticamente significativos para sentenças. O AI-Orchestrator usa `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões).

**Semantic Router:** Roteador que classifica a intenção do usuário usando similaridade de embeddings, com fallback para LLM classifier.

**Speculative Decoding:** Técnica que usa um modelo menor (draft) para "chutar" múltiplos tokens, depois o modelo grande verifica em paralelo. Aceleração típica: 1.4-1.8×.

**SwiGLU:** Função de ativação usada em FFN de modelos modernos (Llama, Qwen, Mistral). Substitui ReLU/GELU do Transformer original.

## T

**TensorRT-LLM:** Framework de inferência da NVIDIA otimizado para GPUs NVIDIA, com suporte a FP8, inflight batching e kernels customizados.

**Tool Calling:** Ver Function Calling.

**TPS (Tokens Per Second):** Métrica de velocidade de geração. 30-100 tok/s local (GPU), 5-15 tok/s (CPU).

**TTFT (Time to First Token):** Tempo entre o envio da requisição e o recebimento do primeiro token. Afeta percepção de velocidade do usuário.

## U

**Unsloth:** Biblioteca open-source que acelera fine-tuning LoRA em até 2× e reduz VRAM em até 60% via kernels CUDA otimizados.

## V

**vLLM:** Framework de serving de LLMs com PagedAttention, continuous batching e API OpenAI-compatible. Focado em throughput de produção.

**VRAM (Video RAM):** Memória da GPU. Limita o tamanho do modelo que pode ser carregado.
