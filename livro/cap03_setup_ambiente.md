# Capitulo 3 — Setup do Ambiente

## 3.1 Requisitos de hardware

Antes de instalar qualquer software, voce precisa saber o que seu hardware aguenta. Rodar um LLM e fundamentalmente diferente de rodar uma aplicacao web — o modelo precisa caber inteiro na memoria (RAM ou VRAM da GPU) e a velocidade de geracao depende diretamente do poder de processamento.

### Tabela de recomendacoes por cenario

```
+-------------------+----------------+----------+----------+-----------------------+
| Cenario           | GPU            | VRAM     | RAM      | Modelos que rodam     |
+-------------------+----------------+----------+----------+-----------------------+
| Estudo/hobby      | RTX 3060       | 12 GB    | 16 GB    | 7B q4, 3B fp16        |
| Desenvolvimento   | RTX 4070 Ti    | 16 GB    | 32 GB    | 7B fp16, 13B q4       |
| Producao leve     | RTX 4090       | 24 GB    | 64 GB    | 13B fp16, 70B q4      |
| Producao media    | A100 40GB      | 40 GB    | 128 GB   | 70B fp16              |
| Producao pesada   | 2x A100 80GB   | 160 GB   | 256 GB   | 70B fp16, 405B q4     |
| Sem GPU           | CPU only       | -        | 32+ GB   | 7B q4 (lento)         |
+-------------------+----------------+----------+----------+-----------------------+
```

**Regra pratica para VRAM**: um modelo com N bilhoes de parametros em FP16 (meia precisao) precisa de aproximadamente `N * 2` GB de VRAM. Em quantizacao 4-bit, cai para aproximadamente `N * 0.5` GB.

```
Exemplos:
- LLaMA 3 8B em FP16:  ~16 GB de VRAM
- LLaMA 3 8B em Q4:    ~4.5 GB de VRAM
- LLaMA 3 70B em FP16: ~140 GB de VRAM
- LLaMA 3 70B em Q4:   ~40 GB de VRAM
```

### Disco

Modelos ocupam espaco significativo em disco:

```
- Modelo 7B (Q4):    ~4 GB
- Modelo 7B (FP16):  ~14 GB
- Modelo 70B (Q4):   ~40 GB
- Modelo 70B (FP16): ~140 GB
```

Recomendacao minima: **SSD com pelo menos 100 GB livres**. HDD funciona para armazenamento, mas o tempo de carregamento do modelo sera significativamente maior.

### CPU-only: e possivel?

Sim, mas com expectativas realistas. Rodando um modelo 7B quantizado em 4-bit na CPU, voce obtera algo entre 2-5 tokens por segundo, dependendo do processador. Para comparacao, uma RTX 4090 gera 50-100 tokens por segundo com o mesmo modelo.

CPU-only serve para testes e prototipagem, nao para producao.

## 3.2 Instalando drivers NVIDIA e CUDA

Se voce tem uma GPU NVIDIA, precisa de dois componentes: o **driver da GPU** e o **CUDA toolkit**.

### Verificando sua GPU

```bash
# Verifica se o sistema detecta a GPU NVIDIA
lspci | grep -i nvidia

# Saida esperada (exemplo):
# 01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090]
```

### Instalando o driver NVIDIA (Ubuntu/Debian)

```bash
# Atualizar lista de pacotes
sudo apt update

# Instalar driver recomendado automaticamente
sudo ubuntu-drivers autoinstall

# OU instalar versao especifica (recomendado para producao)
sudo apt install nvidia-driver-555

# Reiniciar o sistema (obrigatorio apos instalacao do driver)
sudo reboot
```

### Verificando a instalacao do driver

```bash
# Verificar se o driver esta funcionando
nvidia-smi

# Saida esperada:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 555.42    Driver Version: 555.42    CUDA Version: 12.5           |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  GeForce RTX 4090    Off  | 00000000:01:00.0  On |                  Off |
# | 30%   35C    P8    20W / 450W |    512MiB / 24564MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### Instalando CUDA Toolkit

O CUDA Toolkit e necessario para frameworks como PyTorch compilarem kernels GPU. A maioria das ferramentas modernas (Ollama, vLLM) ja inclui CUDA embutido, mas e bom ter instalado para desenvolvimento.

```bash
# Instalar CUDA Toolkit (Ubuntu 22.04/24.04)
# Opcao 1: via repositorio NVIDIA (recomendado)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-5

# Adicionar ao PATH (colocar no ~/.bashrc)
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

# Recarregar bashrc
source ~/.bashrc

# Verificar instalacao
nvcc --version
# Saida esperada: Cuda compilation tools, release 12.5, V12.5.xxx
```

## 3.3 Docker: por que usar, instalacao, conceitos basicos

### Por que Docker para LLMs?

Docker resolve o problema classico "funciona na minha maquina". Com LLMs, as dependencias sao particularmente complexas: versoes especificas de CUDA, cuDNN, PyTorch, bibliotecas de quantizacao. Docker encapsula tudo isso em um container reprodutivel.

Alem disso, ferramentas como Ollama e vLLM oferecem imagens Docker oficiais que ja vem com tudo configurado.

### Instalando Docker

```bash
# Remover versoes antigas (se existirem)
sudo apt remove docker docker-engine docker.io containerd runc

# Instalar dependencias
sudo apt update
sudo apt install ca-certificates curl gnupg

# Adicionar chave GPG oficial do Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Adicionar repositorio
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Permitir uso sem sudo
sudo usermod -aG docker $USER

# Aplicar mudanca de grupo (ou faca logout/login)
newgrp docker

# Verificar instalacao
docker run hello-world
```

### NVIDIA Container Toolkit (para GPU no Docker)

Para que containers Docker acessem a GPU, voce precisa do NVIDIA Container Toolkit:

```bash
# Adicionar repositorio NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar
sudo apt update
sudo apt install nvidia-container-toolkit

# Configurar Docker para usar runtime NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker

# Reiniciar Docker
sudo systemctl restart docker

# Testar acesso a GPU dentro de container
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

## 3.4 Ollama: instalacao, primeiro modelo, comandos essenciais

**Ollama** e a forma mais simples de rodar LLMs localmente. Pense nele como o "Docker para modelos de linguagem" — abstrai toda a complexidade de configuracao.

### Instalacao

```bash
# Instalacao com script oficial (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Verificar instalacao
ollama --version
# Saida: ollama version 0.x.x
```

### Baixando e rodando seu primeiro modelo

```bash
# Baixar e rodar o LLaMA 3.2 3B (modelo leve, ideal para comecar)
ollama run llama3.2

# O download acontece automaticamente (~2 GB)
# Apos o download, voce entra no modo interativo:
# >>> Ola! Como voce esta?
# Ola! Estou bem, obrigado por perguntar...

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

# Ver informacoes de um modelo
ollama show llama3.2

# Rodar com parametros customizados
ollama run llama3.2 --verbose

# Servir como API (roda em background na porta 11434)
ollama serve
```

### Usando a API do Ollama

```python
import requests

def perguntar(modelo: str, prompt: str) -> str:
    """
    Envia uma pergunta ao Ollama via API local.
    O Ollama precisa estar rodando (ollama serve).
    """
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,  # resposta completa de uma vez
            "options": {
                "temperature": 0.7,  # criatividade (0 = deterministico, 1 = criativo)
                "num_predict": 500,  # maximo de tokens na resposta
            }
        }
    )
    return resposta.json()["response"]

# Exemplo de uso
resposta = perguntar("llama3.2", "Explique o que e um LLM em 3 frases.")
print(resposta)
```

### Modelos recomendados para comecar

```
+-------------------+--------+--------+----------------------------------+
| Modelo            | Params | VRAM   | Melhor para                      |
+-------------------+--------+--------+----------------------------------+
| llama3.2:1b       | 1B     | ~2 GB  | Testes rapidos, hardware fraco   |
| llama3.2          | 3B     | ~3 GB  | Uso geral leve                   |
| llama3.1:8b       | 8B     | ~5 GB  | Uso geral, bom custo-beneficio   |
| mistral           | 7B     | ~5 GB  | Europeu, bom em multiplos idiomas|
| qwen2.5:7b        | 7B     | ~5 GB  | Forte em codigo e raciocinio     |
| codellama         | 7B     | ~5 GB  | Especializado em codigo          |
| deepseek-coder-v2 | 16B    | ~10 GB | Melhor para codigo               |
| llama3.1:70b-q4   | 70B    | ~40 GB | Qualidade proxima a GPT-4        |
+-------------------+--------+--------+----------------------------------+
```

## 3.5 vLLM: quando usar, instalacao, diferencas do Ollama

**vLLM** e um servidor de inferencia de alta performance. Enquanto Ollama e otimizado para facilidade de uso, vLLM e otimizado para **throughput** — servir muitas requisicoes simultaneas com eficiencia maxima.

### Quando usar vLLM em vez de Ollama

- **Ollama**: desenvolvimento local, prototipagem, uso pessoal, poucos usuarios
- **vLLM**: producao, multiplos usuarios simultaneos, alta demanda, integracao com APIs OpenAI-compatible

### Diferencas tecnicas principais

```
+---------------------+------------------+------------------+
| Caracteristica      | Ollama           | vLLM             |
+---------------------+------------------+------------------+
| Facilidade de uso   | Alta             | Media            |
| Throughput          | Moderado         | Alto             |
| Continuous batching | Nao              | Sim              |
| PagedAttention      | Nao              | Sim              |
| API compativel      | Propria          | OpenAI-compat.   |
| Formato de modelo   | GGUF             | HuggingFace      |
| Quantizacao         | GGUF (Q4, Q8)    | AWQ, GPTQ, FP8   |
| Multi-GPU           | Limitado         | Tensor Parallel  |
| Ideal para          | Dev/pessoal      | Producao         |
+---------------------+------------------+------------------+
```

### Instalacao do vLLM

```bash
# Opcao 1: pip (requer CUDA instalado)
pip install vllm

# Opcao 2: Docker (recomendado — ja inclui CUDA)
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Usando vLLM como servidor

```bash
# Iniciar servidor vLLM com API compativel com OpenAI
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Agora voce pode usar a API identica a da OpenAI:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "O que e um transformer?"}
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
    api_key="nao-necessario"  # vLLM nao exige chave
)

resposta = cliente.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "Voce e um assistente tecnico."},
        {"role": "user", "content": "Explique PagedAttention em 3 frases."}
    ],
    max_tokens=300,
    temperature=0.7
)

print(resposta.choices[0].message.content)
```

## 3.6 Python: venv, pip, jupyter — setup completo

### Instalando Python (se necessario)

A maioria das distribuicoes Linux ja vem com Python 3. Verifique:

```bash
python3 --version
# Saida esperada: Python 3.10+ (recomendado 3.11 ou 3.12)

# Se nao tiver, instale:
sudo apt install python3 python3-pip python3-venv
```

### Criando um ambiente virtual

Ambientes virtuais isolam dependencias de cada projeto. **Nunca instale pacotes no Python do sistema.**

```bash
# Criar diretorio do projeto
mkdir ~/llm-on-premise && cd ~/llm-on-premise

# Criar ambiente virtual
python3 -m venv .venv

# Ativar o ambiente (fazer isso sempre que abrir o terminal)
source .venv/bin/activate

# Verificar que esta usando o Python do venv
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
    accelerate \               # otimizacao de inferencia
    bitsandbytes \             # quantizacao 4-bit/8-bit
    sentencepiece \            # tokenizacao
    protobuf \                 # serializacao de modelos
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

Se voce nao tem GPU local, o **Google Colab** oferece acesso gratuito a GPUs T4 (16 GB de VRAM) — suficiente para rodar modelos de ate 7B.

### Limitacoes do Colab gratuito

- Sessoes duram no maximo 12 horas (frequentemente menos)
- GPU pode nao estar disponivel em horarios de pico
- Disco efemero — dados sao perdidos ao encerrar a sessao
- RAM limitada a ~12 GB (versao gratuita)

### Setup basico no Colab

```python
# Celula 1: verificar GPU disponivel
!nvidia-smi

# Celula 2: instalar Ollama no Colab
!curl -fsSL https://ollama.com/install.sh | sh

# Celula 3: iniciar Ollama em background
import subprocess
processo = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Celula 4: aguardar inicializacao e baixar modelo
import time
time.sleep(5)  # aguarda o servidor iniciar
!ollama pull llama3.2:1b

# Celula 5: testar
!ollama run llama3.2:1b "O que e machine learning? Responda em 2 frases."
```

### Quando usar Colab vs local

```
+---------------------+------------------+------------------+
| Criterio            | Colab            | Local            |
+---------------------+------------------+------------------+
| Custo               | Gratis (T4)      | Hardware proprio  |
| Disponibilidade     | Incerta          | Sempre           |
| Persistencia        | Efemera          | Permanente       |
| Privacidade         | Google ve dados  | Total            |
| Modelos grandes     | Ate ~13B q4      | Depende do HW    |
| Ideal para          | Aprender/testar  | Desenvolver/prod |
+---------------------+------------------+------------------+
```

## 3.8 Verificando se tudo funciona: checklist de validacao

Apos instalar tudo, execute este checklist para garantir que o ambiente esta funcional:

```bash
#!/bin/bash
# Script de validacao do ambiente LLM on-premise
# Salve como: verificar_ambiente.sh
# Execute com: bash verificar_ambiente.sh

echo "=== CHECKLIST DE VALIDACAO ==="
echo ""

# 1. Driver NVIDIA
echo "[1/7] Driver NVIDIA..."
if nvidia-smi &> /dev/null; then
    echo "  OK - $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    echo "  FALHA - Driver NVIDIA nao encontrado"
fi

# 2. CUDA
echo "[2/7] CUDA Toolkit..."
if nvcc --version &> /dev/null; then
    echo "  OK - $(nvcc --version | grep release | awk '{print $6}')"
else
    echo "  AVISO - nvcc nao encontrado (pode nao ser necessario)"
fi

# 3. Docker
echo "[3/7] Docker..."
if docker --version &> /dev/null; then
    echo "  OK - $(docker --version)"
else
    echo "  FALHA - Docker nao instalado"
fi

# 4. Docker GPU
echo "[4/7] Docker com GPU..."
if docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "  OK - Container consegue acessar GPU"
else
    echo "  AVISO - NVIDIA Container Toolkit pode nao estar configurado"
fi

# 5. Ollama
echo "[5/7] Ollama..."
if ollama --version &> /dev/null; then
    echo "  OK - $(ollama --version)"
else
    echo "  FALHA - Ollama nao instalado"
fi

# 6. Python e venv
echo "[6/7] Python..."
if python3 --version &> /dev/null; then
    echo "  OK - $(python3 --version)"
else
    echo "  FALHA - Python3 nao encontrado"
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
    print('  AVISO - PyTorch instalado mas CUDA nao disponivel')
" 2>/dev/null || echo "  FALHA - PyTorch nao instalado"

echo ""
echo "=== FIM DO CHECKLIST ==="
```

### Teste rapido com Ollama

```bash
# Baixar modelo leve para teste
ollama pull llama3.2:1b

# Testar geracao
ollama run llama3.2:1b "Responda apenas com 'OK' se voce esta funcionando."

# Testar API
curl -s http://localhost:11434/api/generate \
    -d '{"model":"llama3.2:1b","prompt":"Diga ola","stream":false}' | \
    python3 -m json.tool
```

### Teste rapido com Python

```python
# teste_ambiente.py
# Executa verificacoes basicas do ambiente

import sys
print(f"Python: {sys.version}")

# Verifica PyTorch e CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Teste rapido de computacao na GPU
        tensor = torch.randn(1000, 1000, device="cuda")
        resultado = torch.matmul(tensor, tensor)
        print(f"Teste de computacao GPU: OK ({resultado.shape})")
except ImportError:
    print("PyTorch: NAO INSTALADO")

# Verifica conexao com Ollama
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    modelos = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama: OK - {len(modelos)} modelo(s) instalado(s)")
    for m in modelos:
        print(f"  - {m}")
except Exception as e:
    print(f"Ollama: NAO ACESSIVEL ({e})")
```

## 3.9 Troubleshooting: problemas comuns e solucoes

### Problema: `nvidia-smi` retorna erro

```
Causa: driver NVIDIA nao instalado ou nao carregado
Solucao:
  sudo apt install nvidia-driver-555
  sudo reboot
```

### Problema: `CUDA out of memory`

```
Causa: modelo grande demais para a VRAM disponivel
Solucoes:
  1. Usar modelo menor (ex: 7B em vez de 13B)
  2. Usar quantizacao (Q4 em vez de FP16)
  3. Reduzir max_tokens ou context_length
  4. Fechar outros programas que usam GPU (navegador, jogos)

  # Ver uso atual de VRAM:
  nvidia-smi
  # Ou em tempo real:
  watch -n 1 nvidia-smi
```

### Problema: Ollama lento na primeira execucao

```
Causa: modelo sendo carregado em memoria (normal)
Solucao: aguardar. Primeira inferencia e lenta, seguintes sao rapidas.
O Ollama mantém o modelo em memoria por ~5 minutos de inatividade.

  # Manter modelo carregado permanentemente:
  curl http://localhost:11434/api/generate \
      -d '{"model":"llama3.2","keep_alive":-1}'
```

### Problema: Docker nao acessa GPU

```
Causa: NVIDIA Container Toolkit nao configurado
Solucao:
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  # Testar:
  docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

### Problema: `pip install torch` instala versao sem CUDA

```
Causa: pip instala versao CPU por padrao
Solucao: instalar versao especifica com CUDA

  pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Problema: modelo gera texto em ingles quando pergunto em portugues

```
Causa: modelos menores (1B-3B) tem capacidade limitada em portugues
Solucoes:
  1. Usar modelos maiores (7B+) que tem melhor suporte multilingual
  2. Adicionar instrucao explicita no prompt: "Responda em portugues brasileiro"
  3. Usar modelos com bom suporte PT-BR: Mistral, Qwen2.5, Sabia (Maritaca AI)
```

### Problema: vLLM nao inicia — `ValueError: model not found`

```
Causa: modelo nao baixado ou nome incorreto
Solucao:
  # Baixar modelo antes de servir
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

  # Ou usar modelo de repositorio publico (nao precisa de token):
  vllm serve Qwen/Qwen2.5-7B-Instruct
```

### Problema: permissao negada no Docker

```
Causa: usuario nao esta no grupo docker
Solucao:
  sudo usermod -aG docker $USER
  # Fazer logout e login novamente
  # Ou executar: newgrp docker
```

---

## Resumo do capitulo

- VRAM e o recurso mais critico: modelo N bilhoes de parametros em FP16 precisa de ~N*2 GB
- Driver NVIDIA + CUDA sao pre-requisitos para uso de GPU
- Docker com NVIDIA Container Toolkit simplifica deploys reprodutiveis
- Ollama e a porta de entrada: `curl | sh` + `ollama run llama3.2` e voce esta rodando
- vLLM e para producao: continuous batching, PagedAttention, API OpenAI-compatible
- Ambientes virtuais Python (`venv`) isolam dependencias e evitam conflitos
- Google Colab e uma alternativa viavel para aprendizado quando nao se tem GPU
- Sempre valide o ambiente completo antes de comecar a desenvolver

---

## Fontes

- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 2: Tooling and Installation.
- Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*, O'Reilly. Cap. 1 e 5.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Prefacio (setup Colab).
- Notebooks de referencia: `ch02/ch2_Run_LLM_With_vLLM.ipynb`, `ch03/` (setup).
- Documentacao oficial: [Ollama](https://ollama.com), [vLLM](https://docs.vllm.ai), [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).
