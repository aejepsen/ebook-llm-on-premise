# Relatório de Similaridade — 4-gram Overlap

Data: 2026-06-19 18:10

## Resumo

| Métrica | Valor |
|---|---|
| Capítulos analisados | 21 |
| Parágrafos analisados (top-5/cap) | 102 |
| Parágrafos suspeitos (>40% match) | 0 |
| Fontes PDF | 8 |
| Fontes Notebook | 0 |
| N-grams únicos nas fontes | 439,321 |

## Originalidade por Capítulo

| Capítulo | Parágrafos | Avg 4-gram match | Originalidade | Suspeitos |
|---|---|---|---|---|
| cap01_o_que_sao_llms | 21 | 2.3% | 97.7% | 0 |
| cap02_arquitetura_transformer | 8 | 6.4% | 93.6% | 0 |
| cap03_setup_ambiente | 3 | 4.9% | 95.1% | 0 |
| cap04_primeiros_passos | 8 | 3.4% | 96.6% | 0 |
| cap05_quantizacao | 14 | 4.2% | 95.8% | 0 |
| cap06_otimizacao_inferencia | 7 | 5.4% | 94.6% | 0 |
| cap07_frameworks_serving | 9 | 4.7% | 95.3% | 0 |
| cap08_bert_encoders | 14 | 0.0% | 100.0% | 0 |
| cap09_conceitos_finetuning | 15 | 3.7% | 96.3% | 0 |
| cap10_preparando_datasets | 5 | 1.2% | 98.8% | 0 |
| cap11_treinando_lora | 8 | 0.0% | 100.0% | 0 |
| cap12_avaliacao_metricas | 8 | 5.1% | 94.9% | 0 |
| cap13_exportando_producao | 10 | 0.0% | 100.0% | 0 |
| cap14_pipeline_rag | 7 | 1.5% | 98.5% | 0 |
| cap15_chunking_indexacao | 4 | 0.5% | 99.5% | 0 |
| cap16_rag_agentes | 9 | 3.3% | 96.7% | 0 |
| cap17_graph_rag | 6 | 2.3% | 97.7% | 0 |
| cap18_benchmarking | 12 | 0.0% | 100.0% | 0 |
| cap19_multi_agente | 11 | 4.1% | 95.9% | 0 |
| cap20_seguranca_governanca | 6 | 3.3% | 96.7% | 0 |
| cap21_mlops_llm_on_premise | 16 | 0.0% | 100.0% | 0 |

## Parágrafos Suspeitos (>40% 4-gram overlap)

Nenhum parágrafo suspeito encontrado.
## Notas Metodológicas

- **N-gram size**: 4 palavras
- **Threshold de suspeita**: 40% de 4-grams do parágrafo presentes nas fontes
- **Seleção de parágrafos**: top-5 mais longos por capítulo (>50 palavras, sem código)
- **Normalização**: lowercase, remoção de pontuação, colapso de espaços
- Overlap alto em parágrafos técnicos pode ser natural (terminologia compartilhada)
- PDFs que falharam na leitura foram ignorados na comparação
