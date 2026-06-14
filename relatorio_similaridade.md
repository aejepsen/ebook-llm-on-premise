# Relatório de Similaridade — 4-gram Overlap

Data: 2026-06-13 21:23

## Resumo

| Métrica | Valor |
|---|---|
| Capítulos analisados | 19 |
| Parágrafos analisados (top-5/cap) | 83 |
| Parágrafos suspeitos (>40% match) | 0 |
| Fontes PDF | 6 |
| Fontes Notebook | 33 |
| N-grams únicos nas fontes | 290,244 |

## Originalidade por Capítulo

| Capítulo | Parágrafos | Avg 4-gram match | Originalidade | Suspeitos |
|---|---|---|---|---|
| cap01_o_que_sao_llms | 4 | 9.9% | 90.1% | 0 |
| cap02_arquitetura_transformer | 3 | 8.9% | 91.1% | 0 |
| cap03_setup_ambiente | 2 | 7.4% | 92.6% | 0 |
| cap04_primeiros_passos | 7 | 4.0% | 96.0% | 0 |
| cap05_quantizacao | 14 | 4.6% | 95.4% | 0 |
| cap06_otimizacao_inferencia | 7 | 6.7% | 93.3% | 0 |
| cap07_frameworks_serving | 8 | 5.1% | 94.9% | 0 |
| cap08_conceitos_finetuning | 15 | 3.7% | 96.3% | 0 |
| cap09_preparando_datasets | 5 | 1.2% | 98.8% | 0 |
| cap10_treinando_lora | 8 | 0.0% | 100.0% | 0 |
| cap11_avaliacao_metricas | 6 | 5.1% | 94.9% | 0 |
| cap12_exportando_producao | 10 | 0.0% | 100.0% | 0 |
| cap13_pipeline_rag | 6 | 1.7% | 98.3% | 0 |
| cap14_chunking_indexacao | 2 | 1.0% | 99.0% | 0 |
| cap15_rag_agentes | 9 | 3.6% | 96.4% | 0 |
| cap16_graph_rag | 6 | 2.6% | 97.4% | 0 |
| cap17_benchmarking | 9 | 0.0% | 100.0% | 0 |
| cap18_multi_agente | 3 | 4.0% | 96.0% | 0 |
| cap19_seguranca_governanca | 4 | 1.1% | 98.9% | 0 |

## Parágrafos Suspeitos (>40% 4-gram overlap)

Nenhum parágrafo suspeito encontrado.
## Notas Metodológicas

- **N-gram size**: 4 palavras
- **Threshold de suspeita**: 40% de 4-grams do parágrafo presentes nas fontes
- **Seleção de parágrafos**: top-5 mais longos por capítulo (>50 palavras, sem código)
- **Normalização**: lowercase, remoção de pontuação, colapso de espaços
- Overlap alto em parágrafos técnicos pode ser natural (terminologia compartilhada)
- Todos os 6 PDFs e 33 notebooks foram lidos com sucesso
