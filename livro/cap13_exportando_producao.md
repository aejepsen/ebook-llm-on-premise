# Capítulo 13 — Exportando e Colocando em Produção

## Merge: LoRA adapter + modelo base

Após o treinamento, você tem dois artefatos separados: o **modelo base** (congelado, ~18 GB) e o **adapter LoRA** (treinado, ~320 MB). Para deploy, existem duas opções:

### Opção 1: Merge (recomendado para produção)

Funde os pesos do adapter de volta no modelo base, criando um modelo único. Vantagens:
- **Sem overhead de inferência**: não precisa carregar e somar o adapter em cada forward pass
- **Compatível com qualquer runtime**: o modelo merged funciona como qualquer modelo padrão
- **Simplifica o deploy**: um único artefato para versionar e distribuir

```python
# Merge dos adapters LoRA no modelo base em 16-bit
# Cria um modelo completo com os pesos atualizados
model.save_pretrained_merged(
    '/content/merged',      # diretório de saída
    tokenizer,              # tokenizer do modelo
    save_method='merged_16bit'  # precisão do merge
)
# Saída: ~18 GB de pesos em safetensors + config + tokenizer
```

### Opção 2: Adapter separado (útil para desenvolvimento)

Mantém o adapter separado e carrega em runtime. Vantagens:
- **Menor armazenamento**: salva apenas ~320 MB (adapter) vs ~18 GB (merged)
- **Troca rápida**: pode alternar entre adapters sem recarregar o modelo base
- **Experimenta variações**: múltiplos adapters para tarefas diferentes sobre o mesmo modelo base

```python
# Salva apenas o adapter LoRA (sem merge)
model.save_pretrained('/content/adapter')
# Saída: adapter_config.json + adapter_model.safetensors (~320 MB)

# Para carregar depois:
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
model = PeftModel.from_pretrained(base_model, '/content/adapter')
```

> **AI-Orchestrator**: usamos **merge** porque o modelo vai rodar localmente no Ollama, que espera um modelo completo em formato GGUF. O adapter separado seria útil se tivéssemos múltiplas variantes do LoRA para A/B testing rápido.

---

## Exportando para GGUF (llama.cpp format)

GGUF (GGML Universal Format) é o formato binário usado pelo llama.cpp e Ollama. É otimizado para inferência em CPU e GPU consumer, com suporte a quantização integrada.

### Por que GGUF

| Formato | Runtime | Otimizado para | Tamanho (9B, Q4) |
|---------|---------|----------------|-------------------|
| Safetensors (HF) | transformers, vLLM | GPU server | ~18 GB (fp16) |
| GGUF | llama.cpp, Ollama | CPU + GPU consumer | ~5.4 GB (Q4_K_M) |
| ONNX | ONNX Runtime | Edge devices | Variável |

### Conversão com Unsloth

A Unsloth integra a conversão para GGUF diretamente:

```python
# Converte o modelo merged para GGUF com quantização Q4_K_M
# Unsloth compila llama.cpp automaticamente na primeira chamada
model.save_pretrained_gguf(
    '/content/gguf',          # diretório de saída
    tokenizer,                # tokenizer
    quantization_method='q4_k_m'  # método de quantização
)
```

> **Gotcha real**: a Unsloth adiciona `_gguf` ao diretório de saída. Se você informar `/content/gguf`, o arquivo será gerado em `/content/gguf_gguf/`. Sempre use glob para encontrar o arquivo:

```python
import glob, os

# Busca o arquivo GGUF gerado (Unsloth pode mudar o path)
ggufs = sorted(
    glob.glob('/content/gguf_gguf/**/*.gguf', recursive=True) +
    glob.glob('/content/gguf/**/*.gguf', recursive=True),
    key=os.path.getsize,
    reverse=True  # maior arquivo primeiro (o modelo, não metadados)
)

# Filtra pelo método de quantização
src = [g for g in ggufs if 'Q4_K_M' in g][0]
print(f"GGUF gerado: {src} ({os.path.getsize(src)/1024**3:.1f} GB)")
# GGUF gerado: .../unsloth.Q4_K_M.gguf (5.4 GB)
```

### Conversão manual (sem Unsloth)

Se precisar converter fora do Colab, use o `convert_hf_to_gguf.py` do llama.cpp:

```bash
# Clone o llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Converte de HF safetensors para GGUF fp16
python convert_hf_to_gguf.py /caminho/para/merged/ \
    --outfile modelo-fp16.gguf \
    --outtype f16

# Quantiza para Q4_K_M
./build/bin/llama-quantize modelo-fp16.gguf modelo-Q4_K_M.gguf Q4_K_M
```

---

## Quantizando o modelo merged

Quantização reduz a precisão dos pesos para diminuir tamanho e acelerar inferência. Os métodos mais usados em GGUF:

### Tabela de métodos de quantização

| Método | Bits/peso | Tamanho (9B) | Qualidade | Uso |
|--------|-----------|-------------|-----------|-----|
| F16 | 16 | ~18 GB | Referência | Server com VRAM sobrando |
| Q8_0 | 8 | ~9.5 GB | ~99% do F16 | Melhor qualidade quantizada |
| Q6_K | 6 | ~7.3 GB | ~98% do F16 | Equilíbrio premium |
| **Q4_K_M** | **4** | **~5.4 GB** | **~95% do F16** | **Melhor custo-benefício** |
| Q4_K_S | 4 | ~5.0 GB | ~93% do F16 | Economia de espaço |
| Q3_K_M | 3 | ~4.1 GB | ~88% do F16 | GPU muito limitada |
| Q2_K | 2 | ~3.2 GB | ~80% do F16 | Extremo, perda notável |

### Recomendação por hardware

```
RTX 3060 (12 GB): Q4_K_M — cabe com folga, qualidade excelente
RTX 4060 (8 GB):  Q4_K_S ou Q3_K_M — depende do contexto necessário
RTX 4090 (24 GB): Q6_K ou Q8_0 — aproveita a VRAM extra
CPU only:         Q4_K_M — llama.cpp roda bem em CPU com AVX2
Apple M1/M2/M3:   Q4_K_M ou Q6_K — Metal acceleration
```

> **AI-Orchestrator**: usamos **Q4_K_M** porque a RTX 3060 (12 GB) do servidor de produção comporta o modelo (5.4 GB) com folga para contexto. A diferença de qualidade em relação ao F16 é imperceptível na tarefa de routing/tool-calling.

---

## Registrando no Ollama (Modelfile)

O Ollama usa um `Modelfile` para registrar modelos locais. Ele define: arquivo de pesos, template de chat, parâmetros de inferência e tokens de parada.

### Anatomia do Modelfile

```dockerfile
# 1. Arquivo de pesos GGUF
FROM ./qwen3.5-9b-orch.Q4_K_M.gguf

# 2. Template de chat (Go template — sintaxe do Ollama)
# Define como mensagens são formatadas para o modelo
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- end }}
{{- if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

# 3. Tokens de parada
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# 4. Parâmetros de inferência
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.0
PARAMETER num_ctx 8192
```

### Pontos críticos do Modelfile

1. **Template de chat**: deve ser **idêntico** ao usado no treino. Se o treino usou ChatML (`<|im_start|>/<|im_end|>`), o Modelfile deve usar os mesmos tokens.

2. **Tool-calling**: o template precisa formatar corretamente os blocos `<tools>` e `<tool_call>`. Se essa parte estiver errada, o modelo não vai conseguir chamar ferramentas.

3. **Tokens de parada**: `<|im_start|>` e `<|im_end|>` devem estar nos stop tokens. Sem eles, o modelo pode gerar infinitamente ou incluir lixo na resposta.

4. **num_ctx**: tamanho do contexto em tokens. Deve ser compatível com o `max_seq_length` do treino (4096) ou maior.

### Registrando no Ollama

```bash
# Copie o .gguf e o Modelfile para o mesmo diretório
# Depois registre o modelo:
cd /caminho/para/modelos/
ollama create qwen3.5-9b-orch -f Modelfile

# Verifique que o modelo foi registrado:
ollama list
# NAME                       SIZE    MODIFIED
# qwen3.5-9b-orch:latest    5.4 GB  10 seconds ago
```

> **Gotcha real — AI-Orchestrator**: o Ollama 0.24 não suportava a arquitetura híbrida DeltaNet do Qwen3.5. Foi necessário atualizar para o Ollama 0.30.8. Outro bug: o parâmetro `keep_alive` do código Python enviava `"-1"` (string), mas o Ollama 0.30 exige inteiro (`-1`). Sempre teste a comunicação entre seu código e o Ollama após atualizações.

---

## Deploy local: testando o modelo treinado

Após registrar no Ollama, teste antes de colocar em produção:

### Teste rápido via CLI

```bash
# Teste de geração simples
ollama run qwen3.5-9b-orch "Qual o faturamento do mês passado?"

# Teste com system prompt
ollama run qwen3.5-9b-orch \
    --system "Você é o roteador do AI-Orchestrator." \
    "Quero ver as vendas do último trimestre e os funcionários do RH."
```

### Teste via API

```python
import httpx
import json

# Teste de chat via API do Ollama
resposta = httpx.post("http://localhost:11434/api/chat", json={
    "model": "qwen3.5-9b-orch",
    "stream": False,
    "messages": [
        {"role": "system", "content": "Você é o roteador do AI-Orchestrator."},
        {"role": "user", "content": "Qual o estoque do SKU-001?"}
    ]
})

dados = resposta.json()
print(dados["message"]["content"])
```

### Teste com tools

```python
# Teste de tool-calling
resposta = httpx.post("http://localhost:11434/api/chat", json={
    "model": "qwen3.5-9b-orch",
    "stream": False,
    "messages": [
        {"role": "system", "content": "Você é o agente de estoque."},
        {"role": "user", "content": "Qual a quantidade do SKU-001?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_estoque",
            "description": "Consulta estoque de um produto",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Código SKU"}
                },
                "required": ["sku"]
            }
        }
    }]
})

dados = resposta.json()
tool_calls = dados["message"].get("tool_calls", [])
print(f"Tools chamadas: {json.dumps(tool_calls, indent=2)}")
```

### Checklist de teste pré-produção

- [ ] Modelo carrega sem erro no Ollama
- [ ] Responde a perguntas simples de forma coerente
- [ ] Tool-calling funciona (gera JSON válido para chamadas de ferramenta)
- [ ] Template de chat está correto (tokens especiais aparecem onde esperado)
- [ ] Latência aceitável (< 2s para primeira resposta em GPU consumer)
- [ ] VRAM dentro do limite (modelo + contexto devem caber na GPU)
- [ ] Não gera conteúdo de sistema/contexto indesejado na resposta

---

## Versionamento de modelos

Modelos treinados são artefatos tão importantes quanto código. Versioná-los corretamente evita confusões e permite rollback.

### Estratégia de versionamento

```
modelos/
├── qwen3.5-9b-orch-v1/
│   ├── qwen3.5-9b-orch.Q4_K_M.gguf    (5.4 GB)
│   ├── Modelfile
│   ├── training_config.json             (hiperparâmetros)
│   ├── eval_results.json                (resultados dos 3 gates)
│   └── CHANGELOG.md                     (o que mudou nesta versão)
├── qwen3.5-9b-orch-v2/
│   └── ...
```

### O que versionar

| Artefato | Onde guardar | Por quê |
|----------|-------------|---------|
| GGUF | Google Drive / S3 / NAS | Modelo de produção |
| Adapter LoRA | Git LFS ou Drive | Permite re-merge com modelo base atualizado |
| Dataset (train + val) | Git LFS ou Drive | Reprodutibilidade do treino |
| Config de treino | Git (JSON) | Hiperparâmetros exatos |
| Resultados de eval | Git (JSON/MD) | Baseline para comparação futura |
| Modelfile | Git | Template de chat e parâmetros |

### Naming convention

```
{modelo_base}-{tarefa}-v{versao}.{quantizacao}.gguf

Exemplos:
qwen3.5-9b-orch-v1.Q4_K_M.gguf
llama-3.3-8b-routing-v2.Q6_K.gguf
```

> **AI-Orchestrator**: o README do projeto mantém uma tabela de comparação com todos os modelos testados (7B base, 9B base, 9B LoRA, 30B base), com métricas de cada um. Isso permite decidir rapidamente qual modelo usar em cada cenário.

---

## A/B testing: modelo base vs fine-tuned em produção

Antes de substituir completamente o modelo base, considere rodar ambos em paralelo:

### Estratégia de A/B testing

```
                    ┌─────────────────┐
                    │   Load Balancer  │
                    │   (ou gateway)   │
                    └────────┬────────┘
                             │
                    ┌────────┼────────┐
                    │ 80%    │    20% │
                    ▼        │        ▼
           ┌─────────────┐  │  ┌─────────────┐
           │ Modelo LoRA  │  │  │ Modelo Base  │
           │ (challenger) │  │  │ (incumbent)  │
           └─────────────┘  │  └─────────────┘
                             │
                    ┌────────┴────────┐
                    │   Logging de    │
                    │   métricas      │
                    └─────────────────┘
```

### Métricas para A/B testing

1. **Acurácia em produção**: porcentagem de respostas corretas (validadas por humano ou automaticamente)
2. **Latência**: tempo de resposta (p50, p95, p99)
3. **Taxa de erro**: respostas malformadas, timeouts, crashes
4. **Satisfação do usuário**: se aplicável (thumbs up/down)

### Implementação simples

```python
import random

def escolher_modelo():
    """80% das requisições vão para o LoRA, 20% para o base."""
    if random.random() < 0.8:
        return "qwen3.5-9b-orch"   # modelo LoRA (challenger)
    else:
        return "qwen3.5:9b"        # modelo base (incumbent)

# No gateway, log qual modelo atendeu cada requisição
modelo = escolher_modelo()
resposta = chamar_ollama(modelo, mensagens)
log_metrica(modelo=modelo, latencia=resposta.latencia,
            sucesso=resposta.ok)
```

### Critérios de promoção

O modelo LoRA substitui completamente o base quando:
1. **Acurácia >= baseline** por pelo menos 7 dias
2. **Latência <= baseline** (não pode ficar mais lento)
3. **Zero falhas críticas** (injection leaks, crashes, respostas malformadas)

> **AI-Orchestrator**: o A/B testing não foi implementado formalmente porque o ambiente é local (uma única GPU). A promoção foi feita diretamente após passar nos 3 gates em 2 runs consecutivos. Em ambientes com mais tráfego, A/B testing é essencial.

---

## Checklist de deploy

Antes de colocar o modelo fine-tuned em produção, passe por este checklist completo:

### Preparação

- [ ] **Modelo merged**: adapter LoRA fundido no modelo base (`save_pretrained_merged`)
- [ ] **GGUF gerado**: conversão para formato GGUF com quantização adequada
- [ ] **Modelfile criado**: template de chat, stop tokens e parâmetros configurados
- [ ] **Modelo registrado**: `ollama create` executado com sucesso

### Validação

- [ ] **Gates de qualidade**: todos os evals task-specific passam
- [ ] **Consistência**: resultados idênticos em pelo menos 2 runs
- [ ] **Sem regressão**: nenhuma métrica pior que o baseline
- [ ] **Teste de inferência**: modelo responde corretamente via CLI e API
- [ ] **Tool-calling**: ferramentas são chamadas com argumentos corretos
- [ ] **Injection**: modelo resiste a tentativas de prompt injection

### Performance

- [ ] **VRAM**: modelo + contexto cabem na GPU de produção
- [ ] **Latência**: tempo de resposta aceitável para a aplicação
- [ ] **Throughput**: suporta a carga esperada de requisições
- [ ] **Estabilidade**: sem crashes após horas de operação contínua

### Operacional

- [ ] **Versionamento**: modelo versionado com config e resultados de eval
- [ ] **Rollback**: procedimento documentado para voltar ao modelo anterior
- [ ] **Monitoramento**: logging de métricas em produção (latência, erros, uso de VRAM)
- [ ] **Backup**: GGUF e dataset armazenados em local seguro (Drive, S3, NAS)
- [ ] **Documentação**: README atualizado com tabela de resultados

### Deploy

```bash
# 1. Parar o serviço atual
sudo systemctl stop ai-orchestrator

# 2. Atualizar o .env com o novo modelo
echo 'MODEL=qwen3.5-9b-orch' >> .env

# 3. Verificar que o Ollama reconhece o modelo
ollama list | grep qwen3.5-9b-orch

# 4. Reiniciar o serviço
sudo systemctl start ai-orchestrator

# 5. Verificar saúde
curl http://localhost:8000/health

# 6. Rodar smoke test
python evals/eval_routing.py  # deve passar >= 90%
```

### Rollback de emergência

Se algo der errado em produção, voltar ao modelo anterior deve ser instantâneo:

```bash
# Rollback: voltar para o modelo base
echo 'MODEL=qwen3.5:9b' >> .env
sudo systemctl restart ai-orchestrator

# Verificar
curl http://localhost:8000/health
```

> **AI-Orchestrator**: o deploy foi feito com `.env MODEL=qwen3.5-9b-orch`, rebuild do gateway e verificação dos 3 gates em produção. O modelo anterior (`qwen3.5:9b`) permanece disponível no Ollama para rollback imediato.

---

## Resumo do capítulo

1. **Merge** os adapters no modelo base antes de exportar para produção.
2. **GGUF** é o formato ideal para deploy local com Ollama/llama.cpp.
3. **Q4_K_M** é o melhor custo-benefício de quantização para GPUs consumer.
4. O **Modelfile** define template de chat e parâmetros — deve ser idêntico ao treino.
5. **Teste exaustivamente** antes de promover: CLI, API, tool-calling, injection.
6. **Versione** modelos como código: GGUF + config + eval results + Modelfile.
7. **A/B testing** em produção reduz risco de regressão.
8. **Rollback** deve ser instantâneo — mantenha o modelo anterior disponível.

---

## Fontes

- Gerganov, G. et al. (2024). *llama.cpp: LLM inference in C/C++*. GitHub. https://github.com/ggerganov/llama.cpp
- Ollama Documentation (2025). *Modelfile Reference*. https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- Unsloth Documentation (2025). *Export to GGUF*. https://docs.unsloth.ai
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media.
- Labonne, M. (2025). *LLM Engineer's Handbook*. Packt Publishing.
- Projeto AI-Orchestrator — `train/colab_train_lora.ipynb` (export + Modelfile), `docs/PLANO_LORA_9B.md` (Fases 3-5).
