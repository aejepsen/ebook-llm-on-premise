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

### Parte III — Modelos Encoder
8. [BERT e modelos encoder: classificação, embeddings e detecção](livro/cap08_bert_encoders.md)

### Parte IV — Fine-Tuning
9. [Conceitos de fine-tuning (full, LoRA, QLoRA)](livro/cap09_conceitos_finetuning.md)
10. [Preparando datasets de treino](livro/cap10_preparando_datasets.md)
11. [Treinando com Unsloth + LoRA na prática](livro/cap11_treinando_lora.md)
12. [Avaliação e métricas do modelo treinado](livro/cap12_avaliacao_metricas.md)
13. [Exportando para produção (GGUF, merge, deploy)](livro/cap13_exportando_producao.md)

### Parte V — RAG (Retrieval-Augmented Generation)
14. [Pipeline RAG completo (embeddings, vector DB, retrieval)](livro/cap14_pipeline_rag.md)
15. [Chunking e estratégias de indexação](livro/cap15_chunking_indexacao.md)
16. [RAG com agentes (function calling, LangGraph)](livro/cap16_rag_agentes.md)
17. [Graph RAG com Neo4j](livro/cap17_graph_rag.md)

### Parte VI — Produção
18. [Benchmarking e monitoramento](livro/cap18_benchmarking.md)
19. [Arquitetura multi-agente on-premise](livro/cap19_multi_agente.md)
20. [Segurança e governança de LLMs locais](livro/cap20_seguranca_governanca.md)
21. [MLOps para LLMs On-Premise](livro/cap21_mlops_llm_on_premise.md)

### Referências
- [Bibliografia e fontes](referencias/bibliografia.md)
- [Glossário](glossario.md)

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
| `cap08_bert_encoders.ipynb` | 8 | BERTimbau, SBERT e detecção de injection |
| `cap09_finetuning_intro.ipynb` | 9 | LoRA e QLoRA — conceitos aplicados |
| `cap10_datasets.ipynb` | 10 | Construção e curadoria de datasets |
| `cap11_lora_unsloth.ipynb` | 11 | Treino LoRA com Unsloth (RTX 3060) |
| `cap12_avaliacao.ipynb` | 12 | Avaliação e benchmarks |
| `cap13_export_gguf.ipynb` | 13 | Export GGUF e deploy no Ollama |
| `cap14_rag_pipeline.ipynb` | 14 | Pipeline RAG completo |
| `cap15_chunking.ipynb` | 15 | Estratégias de chunking |
| `cap16_rag_agentes.ipynb` | 16 | RAG agentic com LangGraph |
| `cap17_graph_rag.ipynb` | 17 | Graph RAG com Neo4j |
| `cap18_benchmarking.ipynb` | 18 | Benchmarking de serving |

> **Nota:** Os capítulos 19 (Multi-Agente), 20 (Segurança) e 21 (MLOps) documentam a arquitetura e código do AI-Orchestrator — não possuem notebooks dedicados, pois seu conteúdo prático está no código-fonte do projeto (`gateway/`, `services/`, `evals/`).

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

Adaptado e comentado por **Allan Eric Jepsen** — AI Engineer.

---

> *"A melhor forma de aprender é fazendo. E a melhor forma de ensinar é mostrando cada passo."*
