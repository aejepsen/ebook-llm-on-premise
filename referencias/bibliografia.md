# Bibliografia e Fontes

## Livros e Publicações

1. **Hands-On Large Language Models** — Jay Alammar, Maarten Grootendorst. O'Reilly Media, 2024. Fundamentos de LLMs, tokenização, embeddings, fine-tuning e geração de texto.

2. **LLM Engineer's Handbook** — Paul Iusztin, Maxime Labonne. Packt Publishing, 2024. Guia completo de engenharia de LLMs: design de sistemas, treinamento, avaliação, deploy e monitoramento em produção.

3. **Hands-On LLM Serving and Optimization** — Guia prático de otimização e serving de modelos de linguagem em escala, cobrindo técnicas de quantização, batching e frameworks de inferência.

4. **Benchmarking LLM Serving Systems** — Análise comparativa de sistemas de serving de LLMs, métricas de performance (throughput, latência, TTFT) e metodologias de benchmark.

5. **LLM Serving** — Primer de arquitetura Transformer, otimização de inferência, throughput vs latência, e técnicas de aceleração (KV cache, paged attention).

6. **Hands-on LLM-based Agents: A Tutorial for General Audiences** — Tutorial sobre design de agentes baseados em LLM, planejamento, execução e uso de ferramentas.

## Repositórios de Código (Notebooks Adaptados)

7. **RAG with Python Cookbook** — Hamza Farooq. Repositório com 17 notebooks cobrindo o pipeline RAG completo: carregamento de dados, chunking, embeddings, bancos vetoriais, retrieval, RAG agentic com LangGraph, Graph RAG com Neo4j e avaliação.
   - Fonte: https://github.com/hamzafarooq/RAG-with-Python-Cookbook
   - Licença: verificar repositório original

8. **LLM Model Inference** — Repositório com 16 notebooks sobre inferência de LLMs: arquitetura Transformer, batching, streaming, vLLM, quantização, KV cache, speculative decoding, llama.cpp, SGLang e TensorRT-LLM.
   - Fonte: https://github.com/PacktPublishing/llm-model-inference (ou origem equivalente)
   - Licença: verificar repositório original

## Projetos Práticos Referenciados

9. **AI-Orchestrator** — Anderson Ejepsen. Gateway multi-agente on-premise com LangGraph, Ollama, Qdrant e 4 microsserviços FastAPI. Fine-tuning LoRA do Qwen3.5-9B para roteamento. Produção em suasalada.com.br.
   - Fonte: https://github.com/aejepsen/AI-Orchestrator

10. **Agent-Driven Design (ADD)** — Framework conceitual para design de agentes baseados em LLM: separação Model vs Harness, 4 topologias de agente, eval pyramid. Disponível em: https://github.com/Architecting-AI-Agents-In-Production/agent-driven-design

11. **Agentic Architectural Patterns for Building Multi-Agent Systems** — Ali Arsanjani, Juan Pablo Bustos. Packt Publishing, 2026. Catálogo de padrões arquiteturais para sistemas multi-agente: coordenação (Agent Router, Supervisor, Consensus, Negotiation, Conflict Resolution), robustez (Circuit Breaker, Canary Testing, Trust Decay), compliance (Instruction Fidelity Auditing, Persistent Instruction Anchoring), e maturidade (GenAI Maturity Model, Self-Improvement Flywheel, R⁵ Model).
   - Fonte: Packt Publishing, ISBN 978-1-80602-957-0
   - Licença: Todos os direitos reservados. Citações e referências conforme fair use acadêmico.

## Ferramentas e Frameworks Citados

| Ferramenta | Uso no livro | Site |
|-----------|-------------|------|
| Ollama | Inferência local de LLMs | https://ollama.com |
| vLLM | Serving de alta performance | https://vllm.ai |
| Unsloth | Fine-tuning LoRA otimizado | https://unsloth.ai |
| LangGraph | Orquestração de agentes | https://langchain-ai.github.io/langgraph/ |
| Qdrant | Banco vetorial | https://qdrant.tech |
| Neo4j | Banco de grafos (Graph RAG) | https://neo4j.com |
| llama.cpp | Inferência CPU/GPU em C++ | https://github.com/ggerganov/llama.cpp |
| SGLang | Framework de serving | https://github.com/sgl-project/sglang |
| TensorRT-LLM | Otimização NVIDIA | https://github.com/NVIDIA/TensorRT-LLM |
| Hugging Face Transformers | Ecossistema de modelos | https://huggingface.co |

## Nota sobre Atribuição

Os notebooks neste repositório são **adaptações comentadas em português** dos materiais originais listados acima. Todo conteúdo original mantém a atribuição aos respectivos autores. As adaptações, explicações adicionais e o texto do livro são de autoria de Anderson Ejepsen, licenciados sob CC BY-NC-SA 4.0.
