# Capítulo 2 — Arquitetura Transformer

## 2.1 O paper "Attention is All You Need" — contexto histórico

Em junho de 2017, oito pesquisadores do Google publicaram um artigo que mudaria a inteligência artificial para sempre: **"Attention is All You Need"** (Vaswani et al., 2017). O título era uma provocação deliberada.

Na época, o estado da arte em processamento de linguagem natural (NLP) era dominado por redes recorrentes — RNNs e LSTMs. Esses modelos processavam texto sequencialmente, palavra por palavra, como alguém lendo um livro da esquerda para a direita sem poder voltar a página. Funcionava, mas tinha dois problemas graves:

1. **Lentidão**: por ser sequencial, não aproveitava o paralelismo massivo das GPUs
2. **Memória curta**: em textos longos, informações do início se perdiam até chegar ao final

O paper propunha algo radical: descartar completamente a recorrência e usar apenas **mecanismos de atenção**. O resultado foi o **Transformer** — uma arquitetura que processa todas as palavras simultaneamente e aprende quais palavras devem "prestar atenção" em quais.

Os resultados foram imediatos. O Transformer superou todos os modelos existentes em tradução automática (inglês-alemão, inglês-francês) e fez isso treinando em uma fração do tempo. Em menos de três anos, a maior parte da indústria de NLP migrou para variantes do Transformer.

## 2.2 Encoder vs Decoder — quando usar cada um

O Transformer original tem duas metades: o **Encoder** e o **Decoder**. Pense neles como duas funções complementares:

- **Encoder**: lê e compreende o texto de entrada. Produz uma representação numérica rica do significado.
- **Decoder**: gera texto novo, palavra por palavra, usando a representação do encoder (ou sozinho).

### Analogia da tradução

Imagine um tradutor humano trabalhando com um documento:

1. Primeiro, ele **lê o documento inteiro** em português e compreende o significado (encoder)
2. Depois, ele **escreve a tradução** em inglês, palavra por palavra, consultando sua compreensão (decoder)

O Transformer original usava ambos para tradução. Mas pesquisadores descobriram que cada metade, sozinha, era poderosa para tarefas diferentes:

```
+------------------+------------+---------------------------+
| Arquitetura      | Exemplos   | Melhor para               |
+------------------+------------+---------------------------+
| Encoder-only     | BERT       | Classificação, NER,       |
|                  |            | busca semântica           |
+------------------+------------+---------------------------+
| Decoder-only     | GPT,       | Geração de texto,         |
|                  | LLaMA      | chatbots, código          |
+------------------+------------+---------------------------+
| Encoder-Decoder  | T5, BART   | Tradução, sumarização,    |
|                  |            | tarefas seq-to-seq        |
+------------------+------------+---------------------------+
```

Hoje, a maioria dos LLMs que você vai rodar on-premise (LLaMA, Mistral, Qwen, Phi) são **decoder-only**. Eles recebem um texto de entrada (prompt) e geram a continuação.

## 2.3 Self-Attention explicado com analogia

Self-Attention é o coração do Transformer. Vamos entender com uma analogia antes de ver a mecânica.

### A analogia da sala de aula

Imagine uma sala de aula onde cada aluno é uma palavra em uma frase. A frase é:

> "O gato sentou no tapete porque ele estava cansado"

Cada aluno (palavra) precisa entender sua relação com todos os outros alunos. O mecanismo de atenção funciona assim:

1. O embedding de cada aluno (palavra) é **projetado em três vetores diferentes** usando matrizes de pesos aprendidas:
   - **Query (Q)**: "O que eu estou procurando?" — representa a busca desta palavra
   - **Key (K)**: "O que eu tenho para oferecer?" — representa o que esta palavra disponibiliza
   - **Value (V)**: "Qual informação eu carrego?" — o conteúdo real desta palavra

2. Cada aluno compara sua **Query** com as **Keys** de todos os outros alunos
3. As comparações que dão "match" alto recebem mais peso
4. O resultado final de cada aluno é uma mistura ponderada dos **Values** de todos

No exemplo: quando o modelo processa a palavra "ele", a Query de "ele" vai ter alta compatibilidade com a Key de "gato" — o modelo aprende que "ele" se refere a "gato", não a "tapete". Isso é **resolução de correferência**, e o Transformer faz isso naturalmente.

### A matemática (simplificada)

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Passo a passo:

1. `Q * K^T` — multiplica queries por keys, gerando scores de compatibilidade
2. `/ sqrt(d_k)` — normaliza para evitar valores muito grandes (d_k = dimensão das keys)
3. `softmax(...)` — transforma scores em probabilidades (somam 1.0)
4. `× V` — multiplica probabilidades pelos values, gerando a saída ponderada

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V, causal=True):
    """
    Self-attention com suporte a máscara causal (decoder-only).
    Q, K, V: tensores de forma (..., seq_len, d_k)
    causal: se True, impede atenção a tokens futuros (autorregressivo)
    """
    d_k = K.shape[-1]

    # Passo 1: calcular scores de compatibilidade
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Passo 2: aplicar máscara causal (decoder-only)
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    # Passo 3: converter scores em pesos (probabilidades)
    pesos_atencao = F.softmax(scores, dim=-1)

    # Passo 4: mistura ponderada dos values
    saida = torch.matmul(pesos_atencao, V)

    return saida, pesos_atencao
```

> **Nota**: A máscara causal (`torch.triu`) zera a atenção a tokens futuros preenchendo com `-inf` antes do softmax. Sem ela, o código implementaria atenção bidirecional (estilo BERT/encoder). Para modelos decoder-only (GPT, LLaMA, Mistral), a máscara causal é obrigatória.

## 2.4 Multi-Head Attention — por que múltiplas "perspectivas"

Uma única operação de atenção captura um tipo de relação. Mas linguagem é multifacetada. Na frase "O banco do parque perto do banco financeiro", a palavra "banco" tem relações diferentes com "parque" (banco de sentar) e "financeiro" (instituição).

**Multi-Head Attention** resolve isso executando a atenção múltiplas vezes em paralelo, cada uma com seus próprios pesos (Q, K, V) aprendidos. Cada "cabeça" (head) pode focar em um aspecto diferente:

- Uma cabeça pode aprender relações sintáticas (sujeito-verbo)
- Outra pode capturar relações semânticas (sinônimos, antônimos)
- Outra pode focar em distância posicional (palavras vizinhas)
- Outra pode rastrear correferências (pronomes e seus referentes)

```
                    Input
                      |
         +------------+------------+
         |            |            |
      Head 1       Head 2       Head N
     (sintaxe)   (semântica)   (posição)
         |            |            |
         +------------+------------+
                      |
                   Concat
                      |
                Linear final
                      |
                    Output
```

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: executa atenção em paralelo
    com diferentes projeções dos mesmos dados.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensão por cabeça

        # Projeções lineares para Q, K, V e saída
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # 1. Projetar input em Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Dividir em múltiplas cabeças: reshape para (batch, heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Aplicar atenção em cada cabeça (com máscara causal)
        saida, pesos = self_attention(Q, K, V, causal=True)

        # 4. Concatenar cabeças: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        saida = saida.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 5. Projeção final
        return self.W_o(saida)
```

> **Nota sobre implementações reais**: a projeção `nn.Linear(d_model, d_model)` é matematicamente equivalente a projetar diretamente para `d_k * num_heads`. Implementações como `nn.MultiheadAttention` do PyTorch e HuggingFace usam a segunda forma por eficiência.

O modelo GPT-3, por exemplo, usa 96 cabeças de atenção, cada uma com 128 dimensões, totalizando d_model = 12.288 (96 × 128). O LLaMA 3 70B possui d_model = 8.192, distribuído entre 64 cabeças de Query (128 dimensões cada). Modelos modernos como LLaMA 3 usam **Grouped-Query Attention (GQA)**: as 64 cabeças de Query compartilham apenas 8 cabeças de Key/Value, reduzindo drasticamente o consumo de memória durante a inferência sem perda significativa de qualidade.

## 2.5 Positional Encoding — por que a ordem importa

Durante o treinamento e a fase de prefill, o Transformer processa todas as palavras simultaneamente. Isso é ótimo para velocidade, mas cria um problema: sem informação de ordem, "o gato comeu o rato" é idêntico a "o rato comeu o gato". O modelo não sabe qual palavra veio primeiro.

**Positional Encoding** resolve isso adicionando um sinal matemático único a cada posição. É como numerar as cadeiras em um cinema — cada assento tem uma coordenada única.

O paper original usou funções senoidais:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))
```

Onde `pos` é a posição da palavra e `i` é a dimensão. Essas funções geram um padrão único para cada posição, e o modelo aprende a interpretar esses padrões como informação de ordem.

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Gera matriz de positional encoding.
    Cada posição recebe um vetor único baseado em senos e cossenos.
    """
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Dimensões pares: seno
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            # Dimensões ímpares: cosseno
            pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return pe

# Exemplo: encoding para 10 posições com 8 dimensões
pe = positional_encoding(10, 8)
# Cada linha é única — o modelo sabe "esta é a posição 3"
```

Modelos modernos como LLaMA usam **RoPE (Rotary Position Embedding)**, uma evolução que codifica posições relativas (distância entre palavras) em vez de posições absolutas, permitindo que o modelo generalize melhor para sequências mais longas que as vistas no treinamento.

## 2.6 Feed-Forward Network

Após a atenção determinar **quais** palavras são relevantes, a **Feed-Forward Network (FFN)** processa **o que fazer** com essa informação. É uma rede neural densa aplicada independentemente a cada posição.

### FFN clássica (Transformer original, 2017)

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

A FFN tipicamente expande a dimensão (ex: de 4096 para 16384), aplica uma não-linearidade, e projeta de volta para a dimensão original. Essa expansão temporária permite ao modelo computar transformações mais ricas.

```python
import torch.nn as nn

class FeedForwardClassic(nn.Module):
    """
    FFN clássica (Transformer original): expande, ativa, projeta de volta.
    Aplicada independentemente em cada posição.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # expansão (ex: 4096 -> 16384)
        self.linear2 = nn.Linear(d_ff, d_model)     # projeção (ex: 16384 -> 4096)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)       # expande dimensão
        x = self.activation(x)    # não-linearidade
        x = self.linear2(x)       # volta à dimensão original
        return x
```

### SwiGLU (LLaMA, Qwen, Mistral — padrão moderno)

Modelos recentes como LLaMA usam **SwiGLU**, uma variante estruturalmente diferente que usa **três projeções lineares** com gating element-wise (produto de Hadamard). Não é apenas trocar ReLU por SiLU — o mecanismo de gating é fundamentalmente diferente:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}} \right) W_{\text{down}}$$

onde $\text{SiLU}(z) = z \cdot \sigma(z)$ e $\odot$ = produto de Hadamard (element-wise)

```python
class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN: três projeções lineares com gating.
    Padrão em LLaMA, Qwen, Mistral e modelos modernos.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # projeção gate
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)  # projeção up
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # projeção down

    def forward(self, x):
        gate = F.silu(self.w_gate(x))   # SiLU(xW_gate)
        up = self.w_up(x)               # xW_up
        return self.w_down(gate * up)    # (gate ⊙ up) W_down
```

> **Nota**: SwiGLU não usa bias por padrão e introduz uma projeção extra (W_gate), mas o tamanho de d_ff é reduzido para compensar (ex: LLaMA 7B usa d_ff = 11.008 em vez de 4 × 4096 = 16.384), mantendo o total de parâmetros similar.

## 2.7 Layer Normalization

Redes neurais profundas sofrem de um problema: a escala dos valores muda conforme os dados fluem pelas camadas. Uma camada pode produzir valores entre -0.01 e 0.01, a próxima entre -100 e 100. Isso torna o treinamento instável.

**Layer Normalization** resolve isso normalizando os valores em cada camada para ter média 0 e variância 1:

```
LayerNorm(x) = gamma * (x - media) / sqrt(variancia + epsilon) + beta
```

Onde `gamma` e `beta` são parâmetros aprendíveis. O modelo pode aprender a escala e o deslocamento ideais, mas parte de uma base normalizada.

O Transformer original aplica LayerNorm **depois** de cada sub-camada (Post-LN). Modelos modernos usam **Pré-LN** — normalizam **antes** de cada sub-camada — porque isso torna o treinamento mais estável em modelos muito profundos.

```
Post-LN (original):   x + LayerNorm(SubLayer(x))
Pre-LN  (moderno):    x + SubLayer(LayerNorm(x))
```

A diferença parece sutil, mas Pré-LN permite treinar modelos com centenas de camadas sem que o gradiente exploda ou desapareça. LLaMA, Mistral e a maioria dos modelos recentes usam **RMSNorm**, uma variante mais eficiente que ignora a média e normaliza apenas pela variância.

## 2.8 O fluxo completo: input -> tokens -> embeddings -> attention -> output

Vamos acompanhar uma frase completa passando por um Transformer decoder-only (como GPT ou LLaMA):

```
FLUXO COMPLETO DE UM TRANSFORMER DECODER-ONLY

Entrada: "O gato sentou"

1. TOKENIZACAO
   "O gato sentou" -> [128, 9542, 18793]
   (texto vira números inteiros, cada um representando uma sub-palavra)

2. EMBEDDING
   [128, 9542, 18793] -> [[0.12, -0.34, ...], [0.56, 0.78, ...], [0.91, -0.23, ...]]
   (cada token vira um vetor denso de alta dimensão, ex: 4096 dimensões)

3. POSITIONAL ENCODING
   Adiciona informação de posição a cada embedding
   embedding[0] += PE(posição=0)
   embedding[1] += PE(posição=1)
   embedding[2] += PE(posição=2)

4. CAMADAS DO TRANSFORMER (repete N vezes, ex: 32 camadas no LLaMA 7B)
   Para cada camada:
   +-----------------------------------------------+
   |  a. RMSNorm (normaliza)                       |
   |  b. Multi-Head Self-Attention (com máscara      |
   |     causal: só vê tokens anteriores)            |
   |  c. Conexão residual (soma com input)           |
   |  d. RMSNorm (normaliza de novo)                 |
   |  e. Feed-Forward Network (SwiGLU)               |
   |  f. Conexão residual (soma com input)           |
   +-----------------------------------------------+

5. NORMALIZAÇÃO FINAL
   RMSNorm na saída da última camada

6. PROJEÇÃO PARA VOCABULÁRIO
   Vetor de 4096 dimensões -> vetor de 32000 dimensões (tamanho do vocabulário)
   Cada dimensão = score para uma palavra no vocabulário

7. SOFTMAX + AMOSTRAGEM
   Scores -> probabilidades
   Seleciona próxima palavra (ex: "no" com 42% de probabilidade)

8. REPETE
   Adiciona "no" ao input e volta ao passo 4 (com KV Cache — ver abaixo)
   "O gato sentou no" -> próximo token -> ...
```

### Diagrama ASCII do bloco Transformer

```
         Input (embeddings + posição)
              |
              v
    +---------+---------+
    |     RMSNorm       |
    +---------+---------+
              |
              v
    +---------+---------+
    |   Multi-Head      |
    |   Self-Attention  |
    |   (masked)        |
    +---------+---------+
              |
         +----+----+
         |  Soma   |<-------- Conexão Residual
         +----+----+
              |
              v
    +---------+---------+
    |     RMSNorm       |
    +---------+---------+
              |
              v
    +---------+---------+
    |   Feed-Forward    |
    |   (SwiGLU)        |
    +---------+---------+
              |
         +----+----+
         |  Soma   |<-------- Conexão Residual
         +----+----+
              |
              v
         Output (para próxima camada ou projeção final)
```

### Máscara causal

Um detalhe crucial em modelos decoder-only: a **máscara causal** impede que o modelo "veja o futuro". Ao processar a posição 3, ele só pode prestar atenção nas posições 0, 1 e 2. Isso garante que a geração seja autorregressiva — cada palavra gerada depende apenas das anteriores.

```
Matriz de máscara causal (1 = pode ver, 0 = bloqueado):

        pos0  pos1  pos2  pos3
pos0  [  1     0     0     0  ]
pos1  [  1     1     0     0  ]
pos2  [  1     1     1     0  ]
pos3  [  1     1     1     1  ]
```

### Prefill vs Decode — as duas fases da inferência

Na prática, a passagem pelo Transformer durante a inferência não é uniforme. Existem duas fases com comportamentos de GPU radicalmente diferentes:

**Fase 1 — Prefill (leitura do prompt)**:
- O prompt inteiro é processado **em paralelo** (todas as posições ao mesmo tempo)
- **Compute-bound**: satura a GPU com operações de matmul
- É aqui que o Transformer brilha em paralelismo

**Fase 2 — Decode (geração token a token)**:
- Gera **um token por vez**, de forma estritamente sequencial
- **Memory-bound**: a GPU fica subutilizada, esperando leitura de memória
- Sem otimização, cada novo token reprocessaria toda a sequência (O(n²))

Ferramentas como vLLM otimizam essas fases separadamente. Entender essa diferença é essencial para dimensionar corretamente hardware on-premise.

### KV Cache — por que a inferência é viável

Sem otimização, para gerar o 100º token de uma resposta, o modelo precisaria reprocessar os 99 tokens anteriores do zero — recalculando Keys e Values de todas as posições em todas as camadas. Isso tornaria a geração exponencialmente mais lenta a cada palavra.

O **KV Cache** resolve isso: durante a fase de decode, as Keys e Values calculadas para tokens anteriores são **armazenadas na VRAM** e reutilizadas. A cada novo token, o modelo só precisa calcular Q, K, V do token atual e consultar o cache dos anteriores.

```
Sem KV Cache (ingênuo):
  Token 1: processa [1]                          → 1 cálculo
  Token 2: processa [1, 2]                       → 2 cálculos
  Token 3: processa [1, 2, 3]                    → 3 cálculos
  Token N: processa [1, 2, ..., N]               → N cálculos
  Total: O(N²) — cresce quadraticamente

Com KV Cache:
  Token 1: calcula Q₁,K₁,V₁ → atenção → salva K₁,V₁ no cache    → 1 cálculo
  Token 2: calcula Q₂,K₂,V₂ → atenção Q₂ vs [K₁,K₂] → salva K₂,V₂  → 1 cálculo
  Token 3: calcula Q₃,K₃,V₃ → atenção Q₃ vs [K₁₋₂,K₃] → salva K₃,V₃ → 1 cálculo
  Token N: calcula Qₙ,Kₙ,Vₙ → atenção Qₙ vs cache completo      → 1 cálculo
  Total: O(N) — cresce linearmente
```

**Impacto prático para on-premise**: o KV Cache consome VRAM proporcional ao comprimento da sequência × número de camadas × dimensão do modelo. Para o LLaMA 3 8B com contexto de 8192 tokens, o KV Cache de uma única requisição ocupa ~1 GB de VRAM. Com múltiplas requisições simultâneas, esse consumo escala rapidamente — e é o principal fator limitante do throughput em produção.

### Hiperparâmetros de modelos reais

Esses são os valores que você encontrará nos arquivos `config.json` dos modelos no HuggingFace:

```
+-------------------+--------+--------+---------+-----------+----------+
| Hiperparâmetro    | LLaMA  | LLaMA  | Qwen    | Mistral   | Phi-3    |
|                   | 3 8B   | 3 70B  | 2.5 72B | 7B v0.3   | mini     |
+-------------------+--------+--------+---------+-----------+----------+
| d_model           | 4096   | 8192   | 8192    | 4096      | 3072     |
| num_layers        | 32     | 80     | 80      | 32        | 32       |
| num_heads (Q)     | 32     | 64     | 64      | 32        | 32       |
| num_kv_heads      | 8      | 8      | 8       | 8         | 8        |
| d_ff              | 14336  | 28672  | 29568   | 14336     | 8192     |
| vocab_size        | 128256 | 128256 | 152064  | 32000     | 32064    |
| context_length    | 8192   | 8192   | 131072  | 32768     | 131072   |
+-------------------+--------+--------+---------+-----------+----------+
```

Note que `num_kv_heads < num_heads` em todos — isso é GQA (Grouped-Query Attention) em ação, reduzindo o tamanho do KV Cache sem sacrificar qualidade.

## 2.9 GPT vs BERT vs T5 — diferenças práticas

Três famílias de modelos emergiram do Transformer, cada uma com foco diferente:

### BERT (Bidirectional Encoder Representations from Transformers)

- **Arquitetura**: encoder-only
- **Treinamento**: mascara palavras aleatórias e pede ao modelo para prevê-las (Masked Language Modeling)
- **Direção**: bidirecional — vê o texto inteiro ao mesmo tempo
- **Força**: compreensão profunda do texto
- **Uso**: classificação, busca semântica, NER, análise de sentimento
- **Limitação**: não gera texto novo

```
Treinamento BERT:
"O [MASK] sentou no [MASK]" -> prever "gato" e "tapete"
```

### GPT (Generative Pré-trained Transformer)

- **Arquitetura**: decoder-only
- **Treinamento**: prever a próxima palavra (Causal Language Modeling)
- **Direção**: unidirecional — só vê palavras anteriores
- **Força**: geração de texto fluente e coerente
- **Uso**: chatbots, geração de código, escrita criativa, instruções
- **Limitação**: compreensão pode ser inferior ao BERT para tarefas específicas

```
Treinamento GPT:
"O gato sentou no" -> prever "tapete"
```

### T5 (Text-to-Text Transfer Transformer)

- **Arquitetura**: encoder-decoder completo
- **Treinamento**: toda tarefa é formulada como texto-para-texto
- **Direção**: encoder bidirecional + decoder autorregressivo
- **Força**: versatilidade — uma arquitetura para tudo
- **Uso**: tradução, sumarização, Q&A, classificação

```
Treinamento T5:
Input:  "traduza inglês para português: The cat sat on the mat"
Output: "O gato sentou no tapete"
```

### Comparação rápida

```
+----------+----------------+-------------------+-----------------+
| Modelo   | Arquitetura    | Direção           | Tarefa principal|
+----------+----------------+-------------------+-----------------+
| BERT     | Encoder-only   | Bidirecional      | Compreensão     |
| GPT      | Decoder-only   | Unidirecional     | Geração         |
| T5       | Enc-Dec        | Bi + Uni          | Seq-to-seq      |
+----------+----------------+-------------------+-----------------+
```

### A convergência para decoder-only (2023–presente)

Desde 2023, a esmagadora maioria dos modelos SOTA são **decoder-only**: GPT-4, Claude, Gemini, LLaMA, Mistral, DeepSeek, Qwen. O ecossistema convergiu para essa arquitetura porque, em escala suficiente, modelos decoder-only conseguem cobrir tanto geração quanto compreensão com qualidade competitiva — simplificando significativamente a stack de produção.

BERT e T5 permanecem relevantes em nichos específicos: BERT para embeddings de busca semântica (onde a bidirecionalidade é vantajosa) e T5 para tarefas de tradução pura e seq-to-seq especializado. Mas para a maioria das aplicações on-premise em 2026 — chat, código, análise, geração — modelos decoder-only (família LLaMA/Mistral/Qwen/DeepSeek) são o padrão de mercado.

---

## Resumo do capítulo

- O Transformer substituiu RNNs/LSTMs ao processar texto em paralelo com mecanismos de atenção
- Self-Attention permite que cada palavra "olhe" para todas as outras e determine relevância
- Multi-Head Attention executa múltiplas atenções em paralelo, capturando relações diferentes
- Positional Encoding adiciona informação de ordem ao processamento paralelo
- Feed-Forward Networks processam a informação capturada pela atenção
- Layer Normalization estabiliza o treinamento de modelos profundos
- O fluxo completo: texto → tokens → embeddings → N camadas de atenção+FFN → probabilidades → próxima palavra
- A inferência tem duas fases: prefill (paralelo, compute-bound) e decode (sequencial, memory-bound)
- O KV Cache torna a geração viável ao reutilizar Keys/Values de tokens anteriores
- BERT (compreensão), GPT (geração) e T5 (seq-to-seq) são as três famílias históricas; decoder-only domina desde 2023

---

## Fontes

- Vaswani, A. et al. (2017). "Attention is All You Need". *NeurIPS 2017*.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Cap. 3: Looking Inside Large Language Models.
- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 1 e 2.
- Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL 2019*.
- Radford, A. et al. (2018). "Improving Language Understanding by Generative Pre-Training". *OpenAI*.
- Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". *JMLR*.
- Notebooks de referência: `ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb`, `ch02/ch2_Workthrough_LLM_execution.ipynb`.
