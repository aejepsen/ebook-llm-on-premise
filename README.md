# Mão na Massa: Treinando LLM On-Premise

Livro prático de AI Engineering para iniciantes — do zero ao deploy de modelos de linguagem em infraestrutura local.

## Sumário

### Parte I — Fundamentos
1. [O que são LLMs e por que rodar on-premise](livro/cap01_o_que_sao_llms.md)
2. [Arquitetura Transformer passo a passo](livro/cap02_arquitetura_transformer.md)
3. [Setup do ambiente local (GPU, Docker, Ollama, vLLM)](livro/cap03_setup_ambiente.md)

### Parte II — Inferência Local
4. [Primeiros passos com modelos open-source](livro/cap04_primeiros_passos.md)
5. [Quantização: rodando modelos grandes em hardware limitado](livro/cap05_quantizacao.md)
6. [Otimização de inferência (KV cache, speculative decoding)](livro/cap06_otimizacao_inferencia.md)
7. [Frameworks de serving (llama.cpp, SGLang, TensorRT-LLM)](livro/cap07_frameworks_serving.md)

### Parte III — Fine-Tuning
8. [Conceitos de fine-tuning (full, LoRA, QLoRA)](livro/cap08_conceitos_finetuning.md)
9. [Preparando datasets de treino](livro/cap09_preparando_datasets.md)
10. [Treinando com Unsloth + LoRA na prática](livro/cap10_treinando_lora.md)
11. [Avaliação e métricas do modelo treinado](livro/cap11_avaliacao_metricas.md)
12. [Exportando para produção (GGUF, merge, deploy)](livro/cap12_exportando_producao.md)

### Parte IV — RAG (Retrieval-Augmented Generation)
13. [Pipeline RAG completo (embeddings, vector DB, retrieval)](livro/cap13_pipeline_rag.md)
14. [Chunking e estratégias de indexação](livro/cap14_chunking_indexacao.md)
15. [RAG com agentes (function calling, LangGraph)](livro/cap15_rag_agentes.md)
16. [Graph RAG com Neo4j](livro/cap16_graph_rag.md)

### Parte V — Produção
17. [Benchmarking e monitoramento](livro/cap17_benchmarking.md)
18. [Arquitetura multi-agente on-premise](livro/cap18_multi_agente.md)
19. [Segurança e governança de LLMs locais](livro/cap19_seguranca_governanca.md)

### Referências
- [Bibliografia e fontes](referencias/bibliografia.md)

## Notebooks

Cada capítulo possui um notebook correspondente na pasta `notebooks/`, com explicações detalhadas em português e prontos para execução no Google Colab.

| Notebook | Capítulo | Tema |
|----------|----------|------|
| `cap02_transformers.ipynb` | 2 | Arquitetura Transformer na prática |
| `cap03_setup_ollama.ipynb` | 3 | Instalação e configuração do ambiente |
| `cap04_inferencia_local.ipynb` | 4 | Inferência com modelos open-source |
| `cap05_quantizacao.ipynb` | 5 | Quantização de modelos (GPTQ, AWQ, GGUF) |
| `cap06_otimizacao.ipynb` | 6 | KV cache e speculative decoding |
| `cap07_serving.ipynb` | 7 | Frameworks de serving na prática |
| `cap08_finetuning_intro.ipynb` | 8 | LoRA e QLoRA — conceitos aplicados |
| `cap09_datasets.ipynb` | 9 | Construção e curadoria de datasets |
| `cap10_lora_unsloth.ipynb` | 10 | Treino LoRA com Unsloth (RTX 3060) |
| `cap11_avaliacao.ipynb` | 11 | Avaliação e benchmarks |
| `cap12_export_gguf.ipynb` | 12 | Export GGUF e deploy no Ollama |
| `cap13_rag_pipeline.ipynb` | 13 | Pipeline RAG completo |
| `cap14_chunking.ipynb` | 14 | Estratégias de chunking |
| `cap15_rag_agentes.ipynb` | 15 | RAG agentic com LangGraph |
| `cap16_graph_rag.ipynb` | 16 | Graph RAG com Neo4j |
| `cap17_benchmarking.ipynb` | 17 | Benchmarking de serving |
| `cap18_multi_agente.ipynb` | 18 | Orquestrador multi-agente |

## Como usar

```bash
# Clone o repositório
git clone git@github.com:aejepsen/ebook-llm-on-premise.git

# Leia os capítulos em ordem na pasta livro/
# Execute os notebooks na pasta notebooks/ (local ou Colab)
```

## Requisitos mínimos recomendados

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | GTX 1660 (6 GB) | RTX 3060 (12 GB) |
| RAM | 16 GB | 32 GB |
| Disco | 50 GB livres | 100 GB SSD |
| SO | Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.10+ | 3.12 |

## Licença

Conteúdo original licenciado sob [CC BY-NC-SA 4.0](LICENSE).
Notebooks adaptados mantêm atribuição aos autores originais conforme licenças de origem.

## Autor

Adaptado e comentado por **Anderson Ejepsen** — AI Engineer.

---

> *"A melhor forma de aprender é fazendo. E a melhor forma de ensinar é mostrando cada passo."*
