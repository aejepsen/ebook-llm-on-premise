# Capítulo 3 — Setup do Ambiente

## 3.1 Requisitos de hardware

Antes de instalar qualquer software, você precisa saber o que seu hardware aguenta. Rodar um LLM é fundamentalmente diferente de rodar uma aplicação web — o modelo precisa caber inteiro na memória (RAM ou VRAM da GPU) e a velocidade de geração depende diretamente do poder de processamento.

### Tabela de recomendações por cenário

```
+-------------------+----------------+----------+----------+-----------------------+
| Cenário           | GPU            | VRAM     | RAM      | Modelos que rodam     |
+-------------------+----------------+----------+----------+-----------------------+
| Estudo/hobby      | RTX 3060       | 12 GB    | 16 GB    | 7B q4, 3B fp16        |
| Desenvolvimento   | RTX 4070 Ti    | 16 GB    | 32 GB    | 7B fp16, 13B q4       |
| Produção leve     | RTX 4090       | 24 GB    | 64 GB    | 13B fp16, 70B q4      |
| Produção média    | A100 40GB      | 40 GB    | 128 GB   | 70B fp16              |
| Produção pesada   | 2x A100 80GB   | 160 GB   | 256 GB   | 70B fp16, 405B q4     |
| Sem GPU           | CPU only       | -        | 32+ GB   | 7B q4 (lento)         |
+-------------------+----------------+----------+----------+-----------------------+
```

**Regra prática para VRAM**: um modelo com N bilhões de parâmetros em FP16 (meia precisão) precisa de aproximadamente `N × 2` GB de VRAM. Em quantização 4-bit, cai para aproximadamente `N × 0.5` GB para os pesos, mais ~10-15% de overhead para metadados de quantização (escalas, offsets).

> **Importante**: os valores acima referem-se apenas ao carregamento dos pesos. Em inferência, o **KV Cache** adiciona consumo significativo de VRAM que cresce com o comprimento do contexto (ver Seção 2.8). Para o LLaMA 3 8B com contexto 8K: ~1 GB extra. Com contexto 128K: 10-15 GB extras. Considere essa margem ao dimensionar hardware para produção.

```
Exemplos:
- LLaMA 3 8B em FP16:  ~16 GB de VRAM
- LLaMA 3 8B em Q4:    ~4.5 GB de VRAM
- LLaMA 3 70B em FP16: ~140 GB de VRAM
- LLaMA 3 70B em Q4:   ~40 GB de VRAM
```

### Disco

Modelos ocupam espaço significativo em disco:

```
- Modelo 7B (Q4):    ~4 GB
- Modelo 7B (FP16):  ~14 GB
- Modelo 70B (Q4):   ~40 GB
- Modelo 70B (FP16): ~140 GB
```

Recomendação mínima: **SSD com pelo menos 100 GB livres**. HDD funciona para armazenamento, mas o tempo de carregamento do modelo será significativamente maior.

### CPU-only: é possível?

Sim, mas com expectativas realistas. Rodando um modelo 7B quantizado em 4-bit na CPU, você obterá algo entre 2-5 tokens por segundo, dependendo do processador. Para comparação, uma RTX 4090 gera 50-100 tokens por segundo com o mesmo modelo.

CPU-only serve para testes e prototipagem, não para produção.

## 3.2 Instalando drivers NVIDIA e CUDA

Se você tem uma GPU NVIDIA, precisa de dois componentes: o **driver da GPU** e o **CUDA toolkit**.

### Verificando sua GPU

```bash
# Verifica se o sistema detecta a GPU NVIDIA
lspci | grep -i nvidia

# Saída esperada (exemplo):
# 01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090]
```

### Instalando o driver NVIDIA (Ubuntu/Debian)

```bash
# Atualizar lista de pacotes
sudo apt update

# Instalar versão estável para servidor (recomendado para produção)
sudo apt install nvidia-driver-570-server

# OU instalar driver desktop (para estações de trabalho com monitor)
sudo apt install nvidia-driver-570

# Nota: evite `sudo ubuntu-drivers autoinstall` em servidores — pode instalar
# pacotes de desktop instáveis. Prefira especificar a versão explicitamente.

# Reiniciar o sistema (obrigatório após instalação do driver)
sudo reboot
```

### Verificando a instalação do driver

```bash
# Verificar se o driver está funcionando
nvidia-smi

# Saída esperada (versões podem variar):
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 570.xx    Driver Version: 570.xx    CUDA Version: 12.8           |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  GeForce RTX 4090    Off  | 00000000:01:00.0  On |                  Off |
# | 30%   35C    P8    20W / 450W |    512MiB / 24564MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### Instalando CUDA Toolkit

O CUDA Toolkit é necessário para frameworks como PyTorch compilarem kernels GPU. A maioria das ferramentas modernas (Ollama, vLLM) já inclui CUDA embutido, mas é bom ter instalado para desenvolvimento.

> **Nota**: verifique a versão mais recente do CUDA compatível com seu driver em [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). Em 2026, o ecossistema trabalha com CUDA 12.6+ / 12.8+. Os comandos abaixo usam 12.8 como exemplo — ajuste conforme seu ambiente.

```bash
# Instalar CUDA Toolkit (Ubuntu 22.04/24.04/26.04)
# Via repositório NVIDIA (recomendado)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8

# Adicionar ao PATH (colocar no ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Recarregar bashrc
source ~/.bashrc

# Verificar instalação
nvcc --version
# Saída esperada: Cuda compilation tools, release 12.8, V12.8.xxx
```

## 3.3 Docker: por que usar, instalação, conceitos básicos

### Por que Docker para LLMs?

Docker resolve o problema clássico "funciona na minha máquina". Com LLMs, as dependências são particularmente complexas: versões específicas de CUDA, cuDNN, PyTorch, bibliotecas de quantização. Docker encapsula tudo isso em um container reproduzível.

Além disso, ferramentas como Ollama e vLLM oferecem imagens Docker oficiais que já vêm com tudo configurado.

### Instalando Docker

```bash
# Remover versões antigas (se existirem)
sudo apt remove docker docker-engine docker.io containerd runc

# Instalar dependências
sudo apt update
sudo apt install ca-certificates curl gnupg

# Adicionar chave GPG oficial do Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Adicionar repositório
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Permitir uso sem sudo
sudo usermod -aG docker $USER

# Aplicar mudança de grupo (ou faça logout/login)
newgrp docker

# Verificar instalação
docker run hello-world
```

### NVIDIA Container Toolkit (para GPU no Docker)

Para que containers Docker acessem a GPU, você precisa do NVIDIA Container Toolkit:

```bash
# Adicionar repositório NVIDIA (método unificado estável — funciona em Ubuntu 22.04/24.04/26.04)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar
sudo apt update
sudo apt install nvidia-container-toolkit

# Configurar Docker para usar runtime NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker

# Reiniciar Docker
sudo systemctl restart docker

# Testar acesso à GPU dentro de container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

> **Nota**: o método anterior baseado em `distribution=$(. /etc/os-release;echo $ID$VERSION_ID)` foi depreciado pela NVIDIA e pode falhar em Ubuntu 24.04+. O repositório unificado `stable/deb` acima funciona em qualquer versão do Ubuntu.

## 3.4 Ollama: instalação, primeiro modelo, comandos essenciais

**Ollama** é a forma mais simples de rodar LLMs localmente. Pense nele como o "Docker para modelos de linguagem" — abstrai toda a complexidade de configuração.

### Instalação

```bash
# Instalação com script oficial (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Verificar instalação
ollama --version
# Saída: ollama version 0.x.x
```

### Baixando e rodando seu primeiro modelo

```bash
# Baixar e rodar o LLaMA 3.2 3B (modelo leve, ideal para começar)
ollama run llama3.2

# O download acontece automaticamente (~2 GB)
# Após o download, você entra no modo interativo:
# >>> Olá! Como você está?
# Olá! Estou bem, obrigado por perguntar...

# Para sair: /bye ou Ctrl+D
```

### Comandos essenciais

```bash
# Listar modelos instalados
ollama list

# Baixar modelo sem executar
ollama pull mistral

# Remover modelo
ollama rm mistral

# Ver informações de um modelo
ollama show llama3.2

# Rodar e ativar modo verbose (dentro do prompt interativo, digite /set verbose)
ollama run llama3.2

# Pré-baixar modelo sem executar (útil para scripts de setup)
ollama pull llama3.2

# Servir como API (roda em background na porta 11434)
ollama serve
```

### Usando a API do Ollama

```python
# Opção 1: biblioteca oficial do Ollama (recomendada)
# pip install ollama
import ollama

resposta = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "Responda de forma concisa."},
        {"role": "user", "content": "Explique o que é um LLM em 3 frases."}
    ]
)
print(resposta["message"]["content"])
```

```python
# Opção 2: via requests (didático — mostra que Ollama é uma API HTTP)
import requests

def perguntar(modelo: str, pergunta: str) -> str:
    """
    Envia uma pergunta ao Ollama via API local.
    O Ollama precisa estar rodando (ollama serve).
    """
    resposta = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": modelo,
            "messages": [
                {"role": "user", "content": pergunta}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500,
            }
        }
    )
    return resposta.json()["message"]["content"]

resposta = perguntar("llama3.2", "Explique o que é um LLM em 3 frases.")
print(resposta)
```

### Modelos recomendados para começar

```
+---------------------+--------+--------+------------------------------------------+
| Modelo              | Params | VRAM   | Melhor para                              |
+---------------------+--------+--------+------------------------------------------+
| llama3.2:1b         | 1B     | ~2 GB  | Testes rápidos, hardware fraco           |
| llama3.2            | 3B     | ~3 GB  | Uso geral leve                           |
| llama3.1:8b         | 8B     | ~5 GB  | Uso geral, bom custo-benefício           |
| mistral             | 7B     | ~5 GB  | Europeu, bom em múltiplos idiomas        |
| qwen2.5:7b          | 7B     | ~5 GB  | Forte em código e raciocínio             |
| qwen2.5-coder:7b    | 7B     | ~5 GB  | Especializado em código (substitui       |
|                     |        |        | CodeLlama, superado em 2024)             |
| deepseek-coder-v2   | 16B*   | ~10 GB | Melhor para código (*MoE: 16B total,     |
|                     |        |        | 2.4B ativos por token)                   |
| llama3.1:70b-q4     | 70B    | ~40 GB | Qualidade próxima a GPT-4                |
+---------------------+--------+--------+------------------------------------------+
```

## 3.5 vLLM: quando usar, instalação, diferenças do Ollama

**vLLM** é um servidor de inferência de alta performance. Enquanto Ollama é otimizado para facilidade de uso, vLLM é otimizado para **throughput** — servir muitas requisições simultâneas com eficiência máxima.

### Quando usar vLLM em vez de Ollama

- **Ollama**: desenvolvimento local, prototipagem, uso pessoal, poucos usuários
- **vLLM**: produção, múltiplos usuários simultâneos, alta demanda, integração com APIs OpenAI-compatible

### Diferenças técnicas principais

```
+---------------------+----------------------+------------------------+
| Característica      | Ollama               | vLLM                   |
+---------------------+----------------------+------------------------+
| Facilidade de uso   | Alta                 | Média                  |
| Throughput          | Moderado             | Alto                   |
| Continuous batching | Básico (fila FIFO)   | Sim (PagedAttention)   |
| PagedAttention      | Não                  | Sim                    |
| API compatível      | Própria + OpenAI     | OpenAI-compatible      |
| Formato de modelo   | GGUF (via llama.cpp) | HuggingFace            |
| Quantização         | GGUF (Q4, Q8)        | AWQ, GPTQ, FP8        |
| Multi-GPU           | Limitado             | Tensor + Pipeline Par. |
| Ideal para          | Dev/pessoal          | Produção               |
+---------------------+----------------------+------------------------+
```

### Instalação do vLLM

```bash
# Opção 1: pip (requer CUDA instalado)
pip install vllm

# Opção 2: Docker (recomendado — já inclui CUDA)
# Usando Qwen (modelo aberto, sem restrição de download):
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct

# Para modelos gated (ex: LLaMA), é necessário passar o token do HuggingFace:
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=seu_token_aqui \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

> **Nota**: modelos da família LLaMA da Meta são *gated* no HuggingFace — exigem aceitar os termos de uso e passar um `HF_TOKEN`. Sem ele, o download falha com erro de acesso. Para evitar esse atrito, os exemplos deste livro usam Qwen como modelo padrão (totalmente aberto).

### Usando vLLM como servidor

```bash
# Iniciar servidor vLLM com API compatível com OpenAI
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Agora você pode usar a API idêntica à da OpenAI:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "O que é um transformer?"}
        ],
        "max_tokens": 200
    }'
```

### Usando vLLM com Python (client OpenAI)

```python
from openai import OpenAI

# Aponta para o servidor vLLM local
# A mesma biblioteca usada com a API da OpenAI funciona aqui
cliente = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="nao-necessario"  # vLLM não exige chave por padrão (apenas sandbox local)
)

resposta = cliente.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "Você é um assistente técnico."},
        {"role": "user", "content": "Explique PagedAttention em 3 frases."}
    ],
    max_tokens=300,
    temperature=0.7
)

print(resposta.choices[0].message.content)
```

> **Segurança em produção**: o padrão sem chave (`api_key="nao-necessario"`) é inseguro para deploys expostos na rede. Em produção, inicie o vLLM com `--api-key sua_chave_secreta` e passe a mesma chave no cliente.

## 3.6 Python: venv, pip, jupyter — setup completo

### Instalando Python (se necessário)

A maioria das distribuições Linux já vem com Python 3. Verifique:

```bash
python3 --version
# Saída esperada: Python 3.10+ (recomendado 3.11 ou 3.12)

# Se não tiver, instale:
sudo apt install python3 python3-pip python3-venv
```

### Criando um ambiente virtual

Ambientes virtuais isolam dependências de cada projeto. **Nunca instale pacotes no Python do sistema.**

```bash
# Criar diretório do projeto
mkdir ~/llm-on-premise && cd ~/llm-on-premise

# Criar ambiente virtual
python3 -m venv .venv

# Ativar o ambiente (fazer isso sempre que abrir o terminal)
source .venv/bin/activate

# Verificar que está usando o Python do venv
which python
# Saida: /home/seu-usuario/llm-on-premise/.venv/bin/python

# Atualizar pip
pip install --upgrade pip
```

### Instalando pacotes essenciais

```bash
# Pacotes para trabalhar com LLMs localmente
pip install \
    torch \                    # framework de deep learning
    transformers \             # biblioteca da Hugging Face
    accelerate \               # otimização de inferência
    bitsandbytes \             # quantização 4-bit/8-bit
    sentencepiece \            # tokenização
    protobuf \                 # serialização de modelos
    ollama \                   # client oficial Ollama
    openai \                   # client para APIs OpenAI-compat (vLLM)
    requests \                 # chamadas HTTP (Ollama)
    jupyter \                  # notebooks interativos
    ipywidgets                 # widgets para Jupyter
```

### Configurando Jupyter

```bash
# Instalar e iniciar Jupyter
pip install jupyter

# Iniciar notebook (abre no navegador)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Ou usar JupyterLab (interface mais moderna)
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

## 3.7 Google Colab como alternativa

Se você não tem GPU local, o **Google Colab** oferece acesso gratuito a GPUs T4 (16 GB de VRAM) — suficiente para rodar modelos de até 7B em FP16 ou 13B em Q4.

### Limitações do Colab gratuito

- Sessões duram no máximo 12 horas (frequentemente menos)
- GPU pode não estar disponível em horários de pico
- Disco efêmero — dados são perdidos ao encerrar a sessão
- RAM limitada a ~12 GB (versão gratuita)

### Setup básico no Colab

```python
# Célula 1: verificar GPU disponível
!nvidia-smi

# Célula 2: instalar Ollama no Colab
!curl -fsSL https://ollama.com/install.sh | sh

# Célula 3: iniciar Ollama em background
import subprocess
processo = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Célula 4: aguardar inicialização (polling robusto em vez de sleep fixo)
import time, requests
for tentativa in range(30):
    try:
        requests.get("http://localhost:11434/api/tags")
        print("✓ Servidor Ollama inicializado com sucesso!")
        break
    except requests.exceptions.ConnectionError:
        time.sleep(1)

# Célula 5: baixar modelo
!ollama pull llama3.2:1b

# Célula 6: testar
!ollama run llama3.2:1b "O que é machine learning? Responda em 2 frases."
```

### Quando usar Colab vs local

```
+---------------------+------------------+------------------+
| Critério            | Colab            | Local            |
+---------------------+------------------+------------------+
| Custo               | Grátis (T4)      | Hardware próprio  |
| Disponibilidade     | Incerta          | Sempre           |
| Persistência        | Efêmera          | Permanente       |
| Privacidade         | Google vê dados  | Total            |
| Modelos grandes     | Até ~13B q4      | Depende do HW    |
| Ideal para          | Aprender/testar  | Desenvolver/prod |
+---------------------+------------------+------------------+
```

## 3.8 Verificando se tudo funciona: checklist de validação

Após instalar tudo, execute este checklist para garantir que o ambiente está funcional:

```bash
#!/bin/bash
# Script de validação do ambiente LLM on-premise
# Salve como: verificar_ambiente.sh
# Execute com: bash verificar_ambiente.sh

echo "=== CHECKLIST DE VALIDAÇÃO ==="
echo ""

# 1. Driver NVIDIA
echo "[1/7] Driver NVIDIA..."
if nvidia-smi &> /dev/null; then
    echo "  OK - $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    echo "  FALHA - Driver NVIDIA não encontrado"
fi

# 2. CUDA
echo "[2/7] CUDA Toolkit..."
if nvcc --version &> /dev/null; then
    echo "  OK - $(nvcc --version | grep release | awk '{print $6}')"
else
    echo "  AVISO - nvcc não encontrado (pode não ser necessário)"
fi

# 3. Docker
echo "[3/7] Docker..."
if docker --version &> /dev/null; then
    echo "  OK - $(docker --version)"
else
    echo "  FALHA - Docker não instalado"
fi

# 4. Docker GPU
echo "[4/7] Docker com GPU..."
if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    echo "  OK - Container consegue acessar GPU"
else
    echo "  AVISO - NVIDIA Container Toolkit pode não estar configurado"
fi

# 5. Ollama
echo "[5/7] Ollama..."
if ollama --version &> /dev/null; then
    echo "  OK - $(ollama --version)"
else
    echo "  FALHA - Ollama não instalado"
fi

# 6. Python e venv
echo "[6/7] Python..."
if python3 --version &> /dev/null; then
    echo "  OK - $(python3 --version)"
else
    echo "  FALHA - Python3 não encontrado"
fi

# 7. PyTorch com CUDA
echo "[7/7] PyTorch + CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  OK - PyTorch {torch.__version__} com CUDA {torch.version.cuda}')
    print(f'       GPU: {torch.cuda.get_device_name(0)}')
    print(f'       VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  AVISO - PyTorch instalado mas CUDA não disponível')
" 2>/dev/null || echo "  FALHA - PyTorch não instalado"

echo ""
echo "=== FIM DO CHECKLIST ==="
```

### Teste rápido com Ollama

```bash
# Baixar modelo leve para teste
ollama pull llama3.2:1b

# Testar geração
ollama run llama3.2:1b "Responda apenas com 'OK' se você está funcionando."

# Testar API
curl -s http://localhost:11434/api/chat \
    -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"Diga olá"}],"stream":false}' | \
    python3 -m json.tool
```

### Teste rápido com Python

```python
# teste_ambiente.py
# Executa verificações básicas do ambiente

import sys
print(f"Python: {sys.version}")

# Verifica PyTorch e CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Teste rápido de computação na GPU
        tensor = torch.randn(1000, 1000, device="cuda")
        resultado = torch.matmul(tensor, tensor)
        print(f"Teste de computação GPU: OK ({resultado.shape})")
except ImportError:
    print("PyTorch: NÃO INSTALADO")

# Verifica conexão com Ollama
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    modelos = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama: OK - {len(modelos)} modelo(s) instalado(s)")
    for m in modelos:
        print(f"  - {m}")
except Exception as e:
    print(f"Ollama: NÃO ACESSÍVEL ({e})")
```

## 3.9 Troubleshooting: problemas comuns e soluções

### Problema: `nvidia-smi` retorna erro

```
Causa: driver NVIDIA não instalado ou não carregado
Solução:
  sudo apt install nvidia-driver-570-server
  sudo reboot
```

### Problema: `CUDA out of memory`

```
Causa: modelo grande demais para a VRAM disponível (inclui pesos + KV Cache)
Soluções:
  1. Usar modelo menor (ex: 7B em vez de 13B)
  2. Usar quantização (Q4 em vez de FP16)
  3. Reduzir max_tokens ou context_length
  4. Fechar outros programas que usam GPU (navegador, jogos)

  # Ver uso atual de VRAM:
  nvidia-smi
  # Ou em tempo real:
  watch -n 1 nvidia-smi
```

### Problema: Ollama lento na primeira execução

```
Causa: modelo sendo carregado em memória (normal)
Solução: aguardar. Primeira inferência é lenta, seguintes são rápidas.
O Ollama mantém o modelo em memória por ~5 minutos de inatividade.

  # Manter modelo carregado permanentemente:
  curl http://localhost:11434/api/generate \
      -d '{"model":"llama3.2","keep_alive":-1}'
```

### Problema: Docker não acessa GPU

```
Causa: NVIDIA Container Toolkit não configurado
Solução:
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  # Testar:
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### Problema: `pip install torch` instala versão sem CUDA

```
Causa: pip instala versão CPU por padrão
Solução: instalar versão específica com CUDA

  pip install torch --index-url https://download.pytorch.org/whl/cu124
  # Ajuste cu124 para a versão CUDA do seu ambiente (cu126, cu128, etc.)
```

### Problema: modelo gera texto em inglês quando pergunto em português

```
Causa: modelos menores (1B-3B) têm capacidade limitada em português
Soluções:
  1. Usar modelos maiores (7B+) que têm melhor suporte multilingual
  2. Adicionar instrução explícita no prompt: "Responda em português brasileiro"
  3. Usar modelos com bom suporte PT-BR: Mistral, Qwen2.5, Sabia (Maritaca AI)
```

### Problema: vLLM não inicia — `ValueError: model not found`

```
Causa: modelo não baixado, nome incorreto, ou modelo gated sem HF_TOKEN
Solução:
  # Para modelos abertos (sem restrição):
  vllm serve Qwen/Qwen2.5-7B-Instruct

  # Para modelos gated (LLaMA), definir token antes:
  export HF_TOKEN=seu_token_aqui
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
  vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### Problema: permissão negada no Docker

```
Causa: usuário não está no grupo docker
Solução:
  sudo usermod -aG docker $USER
  # Fazer logout e login novamente
  # Ou executar: newgrp docker
```

---

## Resumo do capítulo

- VRAM é o recurso mais crítico: modelo N bilhões de parâmetros em FP16 precisa de ~N×2 GB (mais KV Cache)
- Driver NVIDIA + CUDA são pré-requisitos para uso de GPU
- Docker com NVIDIA Container Toolkit simplifica deploys reproduzíveis
- Ollama é a porta de entrada: `curl | sh` + `ollama run llama3.2` e você está rodando
- vLLM é para produção: continuous batching, PagedAttention, API OpenAI-compatible
- Ambientes virtuais Python (`venv`) isolam dependências e evitam conflitos
- Google Colab é uma alternativa viável para aprendizado quando não se tem GPU
- Sempre valide o ambiente completo antes de começar a desenvolver

---

## Fontes

- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 2: Tooling and Installation.
- Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*, O'Reilly. Cap. 1 e 5.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Prefácio (setup Colab).
- Notebooks de referência: `ch02/ch2_Run_LLM_With_vLLM.ipynb`, `ch03/` (setup).
- Documentação oficial: [Ollama](https://ollama.com), [vLLM](https://docs.vllm.ai), [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).
