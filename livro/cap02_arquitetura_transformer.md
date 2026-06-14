# Capítulo 2 — Arquitetura Transformer

## 2.1 O paper "Attention is All You Need" — contexto histórico

Em junho de 2017, oito pesquisadores do Google publicaram um artigo que mudaria a inteligência artificial para sempre: **"Attention is All You Need"** (Vaswani et al., 2017). O título era uma provocação deliberada.

Na epoca, o estado da arte em processamento de linguagem natural (NLP) era dominado por redes recorrentes — RNNs e LSTMs. Esses modelos processavam texto sequencialmente, palavra por palavra, como alguém lendo um livro da esquerda para a direita sem poder voltar a página. Funcionava, mas tinha dois problemas graves:

1. **Lentidao**: por ser sequencial, não aproveitava o paralelismo massivo das GPUs
2. **Memória curta**: em textos longos, informações do início se perdiam até chegar ao final

O paper propunha algo radical: descartar completamente a recorrência e usar apenas **mecanismos de atenção**. O resultado foi o **Transformer** — uma arquitetura que processa todas as palavras simultaneamente e aprende quais palavras devem "prestar atenção" em quais.

Os resultados foram imediatos. O Transformer superou todos os modelos existentes em traducao automática (ingles-alemao, ingles-frances) e fez isso treinando em uma fração do tempo. Em menos de dois anos, toda a industria de NLP migrou para variantes do Transformer.

## 2.2 Encoder vs Decoder — quando usar cada um

O Transformer original tem duas metades: o **Encoder** e o **Decoder**. Pense neles como duas funções complementares:

- **Encoder**: le e compreende o texto de entrada. Produz uma representação numerica rica do significado.
- **Decoder**: gera texto novo, palavra por palavra, usando a representação do encoder (ou sozinho).

### Analogia da traducao

Imagine um tradutor humano trabalhando com um documento:

1. Primeiro, ele **le o documento inteiro** em portugues e compreende o significado (encoder)
2. Depois, ele **escreve a traducao** em ingles, palavra por palavra, consultando sua compreensão (decoder)

O Transformer original usava ambos para traducao. Mas pesquisadores descobriram que cada metade, sozinha, era poderosa para tarefas diferentes:

```
+------------------+------------+---------------------------+
| Arquitetura      | Exemplos   | Melhor para               |
+------------------+------------+---------------------------+
| Encoder-only     | BERT       | Classificacao, NER,       |
|                  |            | busca semantica           |
+------------------+------------+---------------------------+
| Decoder-only     | GPT,       | Geracao de texto,         |
|                  | LLaMA      | chatbots, codigo          |
+------------------+------------+---------------------------+
| Encoder-Decoder  | T5, BART   | Traducao, sumarizacao,    |
|                  |            | tarefas seq-to-seq        |
+------------------+------------+---------------------------+
```

Hoje, a maioria dos LLMs que você vai rodar on-premise (LLaMA, Mistral, Qwen, Phi) são **decoder-only**. Eles recebem um texto de entrada (prompt) e geram a continuação.

## 2.3 Self-Attention explicado com analogia

Self-Attention e o coração do Transformer. Vamos entender com uma analogia antes de ver a mecânica.

### A analogia da sala de aula

Imagine uma sala de aula onde cada aluno é uma palavra em uma frase. A frase e:

> "O gato sentou no tapete porque ele estava cansado"

Cada aluno (palavra) precisa entender sua relação com todos os outros alunos. O mecanismo de atenção funciona assim:

1. Cada aluno faz **tres perguntas**:
   - **Query (Q)**: "O que eu estou procurando?" — o que está palavra precisa saber
   - **Key (K)**: "O que eu tenho para oferecer?" — o que está palavra representa
   - **Value (V)**: "Qual informação eu carrego?" — o conteúdo real desta palavra

2. Cada aluno compara sua **Query** com as **Keys** de todos os outros alunos
3. As comparações que dao "match" alto recebem mais peso
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
4. `* V` — multiplica probabilidades pelos values, gerando a saida ponderada

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    """
    Implementacao simplificada de self-attention.
    Q, K, V: tensores de forma (seq_len, d_model)
    """
    d_k = K.shape[-1]

    # Passo 1: calcular scores de compatibilidade
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Passo 2: converter scores em pesos (probabilidades)
    pesos_atencao = F.softmax(scores, dim=-1)

    # Passo 3: mistura ponderada dos values
    saida = torch.matmul(pesos_atencao, V)

    return saida, pesos_atencao
```

## 2.4 Multi-Head Attention — por que multiplas "perspectivas"

Uma única operação de atenção captura um tipo de relação. Mas linguagem e multifacetada. Na frase "O banco do parque perto do banco financeiro", a palavra "banco" tem relações diferentes com "parque" (banco de sentar) e "financeiro" (instituicao).

**Multi-Head Attention** resolve isso executando a atenção multiplas vezes em paralelo, cada uma com seus próprios pesos (Q, K, V) aprendidos. Cada "cabeca" (head) pode focar em um aspecto diferente:

- Uma cabeca pode aprender relações sintaticas (sujeito-verbo)
- Outra pode capturar relações semanticas (sinonimos, antonimos)
- Outra pode focar em distância posicional (palavras vizinhas)
- Outra pode rastrear correferencias (pronomes e seus referentes)

```
                    Input
                      |
         +------------+------------+
         |            |            |
      Head 1       Head 2       Head N
     (sintaxe)   (semantica)   (posicao)
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
class MultiHeadAttention:
    """
    Multi-Head Attention: executa atencao em paralelo
    com diferentes projecoes dos mesmos dados.
    """
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensao por cabeca

        # Cada cabeca tem suas proprias projecoes
        self.W_q = Linear(d_model, d_model)  # projecao de queries
        self.W_k = Linear(d_model, d_model)  # projecao de keys
        self.W_v = Linear(d_model, d_model)  # projecao de values
        self.W_o = Linear(d_model, d_model)  # projecao final

    def forward(self, x):
        # 1. Projetar input em Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Dividir em multiplas cabecas
        # 3. Aplicar atencao em cada cabeca
        # 4. Concatenar resultados
        # 5. Projecao final
        pass  # implementacao completa no notebook do cap. 2
```

O modelo GPT-3, por exemplo, usa 96 cabecas de atenção. Cada uma com 128 dimensões, totalizando 12.288 dimensões (96 x 128 = 12.288). O LLaMA 3 70B usa 64 cabecas com 128 dimensões cada.

## 2.5 Positional Encoding — por que a ordem importa

O Transformer processa todas as palavras simultaneamente. Isso é ótimo para velocidade, mas cria um problema: sem informação de ordem, "o gato comeu o rato" e identico a "o rato comeu o gato". O modelo não sabe qual palavra veio primeiro.

**Positional Encoding** resolve isso adicionando um sinal matemático único a cada posição. E como numerar as cadeiras em um cinema — cada assento tem uma coordenada única.

O paper original usou funções senoidais:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))
```

Onde `pos` e a posição da palavra e `i` e a dimensão. Essas funções geram um padrão único para cada posição, e o modelo aprende a interpretar esses padrões como informação de ordem.

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Gera matrix de positional encoding.
    Cada posicao recebe um vetor unico baseado em senos e cossenos.
    """
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Dimensoes pares: seno
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            # Dimensoes impares: cosseno
            pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return pe

# Exemplo: encoding para 10 posicoes com 8 dimensoes
pe = positional_encoding(10, 8)
# Cada linha e unica — o modelo sabe "esta e a posicao 3"
```

Modelos modernos como LLaMA usam **RoPE (Rotary Position Embedding)**, uma evolução que codifica posições relativas (distância entre palavras) em vez de posições absolutas, permitindo que o modelo generalize melhor para sequencias mais longas que as vistas no treinamento.

## 2.6 Feed-Forward Network

Após a atenção determinar **quais** palavras são relevantes, a **Feed-Forward Network (FFN)** processa **o que fazer** com essa informação. E uma rede neural densa aplicada independentemente a cada posição:

```
FFN(x) = ReLU(x * W1 + b1) * W2 + b2
```

A FFN tipicamente expande a dimensão (ex: de 4096 para 16384), aplica uma não-linearidade (ReLU, GELU ou SiLU), e projeta de volta para a dimensão original. Essa expansão temporaria permite ao modelo computar transformações mais ricas.

```python
class FeedForward:
    """
    Rede feed-forward: expande, ativa, projeta de volta.
    Aplicada independentemente em cada posicao.
    """
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)    # expansao (ex: 4096 -> 16384)
        self.linear2 = Linear(d_ff, d_model)     # projecao (ex: 16384 -> 4096)
        self.activation = GELU()                  # funcao de ativacao

    def forward(self, x):
        x = self.linear1(x)       # expande dimensao
        x = self.activation(x)    # nao-linearidade
        x = self.linear2(x)       # volta a dimensao original
        return x
```

Modelos recentes como LLaMA usam **SwiGLU**, uma variante que substitui a ativação simples por um mecanismo de gating, melhorando a capacidade do modelo sem aumentar parâmetros significativamente.

## 2.7 Layer Normalization

Redes neurais profundas sofrem de um problema: a escala dos valores muda conforme os dados fluem pelas camadas. Uma camada pode produzir valores entre -0.01 e 0.01, a próxima entre -100 e 100. Isso torna o treinamento instável.

**Layer Normalization** resolve isso normalizando os valores em cada camada para ter media 0 e variância 1:

```
LayerNorm(x) = gamma * (x - media) / sqrt(variancia + epsilon) + beta
```

Onde `gamma` e `beta` são parâmetros aprendiveis. O modelo pode aprender a escala e o deslocamento ideais, mas parte de uma base normalizada.

O Transformer original aplica LayerNorm **depois** de cada sub-camada (Post-LN). Modelos modernos usam **Pré-LN** — normalizam **antes** de cada sub-camada — porque isso torna o treinamento mais estável em modelos muito profundos.

```
Post-LN (original):   x + LayerNorm(SubLayer(x))
Pre-LN  (moderno):    x + SubLayer(LayerNorm(x))
```

A diferença parece sutil, mas Pré-LN permite treinar modelos com centenas de camadas sem que o gradiente exploda ou desapareca. LLaMA, Mistral e a maioria dos modelos recentes usam **RMSNorm**, uma variante mais eficiente que ignora a media e normaliza apenas pela variância.

## 2.8 O fluxo completo: input -> tokens -> embeddings -> attention -> output

Vamos acompanhar uma frase completa passando por um Transformer decoder-only (como GPT ou LLaMA):

```
FLUXO COMPLETO DE UM TRANSFORMER DECODER-ONLY

Entrada: "O gato sentou"

1. TOKENIZACAO
   "O gato sentou" -> [128, 9542, 18793]
   (texto vira numeros inteiros, cada um representando um sub-palavra)

2. EMBEDDING
   [128, 9542, 18793] -> [[0.12, -0.34, ...], [0.56, 0.78, ...], [0.91, -0.23, ...]]
   (cada token vira um vetor denso de alta dimensao, ex: 4096 dimensoes)

3. POSITIONAL ENCODING
   Adiciona informacao de posicao a cada embedding
   embedding[0] += PE(posicao=0)
   embedding[1] += PE(posicao=1)
   embedding[2] += PE(posicao=2)

4. CAMADAS DO TRANSFORMER (repete N vezes, ex: 32 camadas no LLaMA 7B)
   Para cada camada:
   +-----------------------------------------------+
   |  a. RMSNorm (normaliza)                       |
   |  b. Multi-Head Self-Attention (com mascara     |
   |     causal: so ve tokens anteriores)           |
   |  c. Conexao residual (soma com input)          |
   |  d. RMSNorm (normaliza de novo)                |
   |  e. Feed-Forward Network (SwiGLU)              |
   |  f. Conexao residual (soma com input)          |
   +-----------------------------------------------+

5. NORMALIZACAO FINAL
   RMSNorm na saida da ultima camada

6. PROJECAO PARA VOCABULARIO
   Vetor de 4096 dimensoes -> vetor de 32000 dimensoes (tamanho do vocabulario)
   Cada dimensao = score para uma palavra no vocabulario

7. SOFTMAX + AMOSTRAGEM
   Scores -> probabilidades
   Seleciona proxima palavra (ex: "no" com 42% de probabilidade)

8. REPETE
   Adiciona "no" ao input e volta ao passo 1
   "O gato sentou no" -> proximo token -> ...
```

### Diagrama ASCII do bloco Transformer

```
         Input (embeddings + posicao)
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
         |  Soma   |<-------- Conexao Residual
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
         |  Soma   |<-------- Conexao Residual
         +----+----+
              |
              v
         Output (para proxima camada ou projecao final)
```

### Mascara causal

Um detalhe crucial em modelos decoder-only: a **mascara causal** impede que o modelo "veja o futuro". Ao processar a posição 3, ele só pode prestar atenção nas posições 0, 1 e 2. Isso garante que a geração seja autorregressiva — cada palavra gerada depende apenas das anteriores.

```
Matriz de mascara causal (1 = pode ver, 0 = bloqueado):

        pos0  pos1  pos2  pos3
pos0  [  1     0     0     0  ]
pos1  [  1     1     0     0  ]
pos2  [  1     1     1     0  ]
pos3  [  1     1     1     1  ]
```

## 2.9 GPT vs BERT vs T5 — diferenças praticas

Tres familias de modelos emergiram do Transformer, cada uma com foco diferente:

### BERT (Bidirectional Encoder Representations from Transformers)

- **Arquitetura**: encoder-only
- **Treinamento**: mascara palavras aleatorias e pede ao modelo para preve-las (Masked Language Modeling)
- **Direcao**: bidirecional — ve o texto inteiro ao mesmo tempo
- **Forca**: compreensão profunda do texto
- **Uso**: classificação, busca semântica, NER, análise de sentimento
- **Limitação**: não gera texto novo

```
Treinamento BERT:
"O [MASK] sentou no [MASK]" -> prever "gato" e "tapete"
```

### GPT (Generative Pré-trained Transformer)

- **Arquitetura**: decoder-only
- **Treinamento**: prever a próxima palavra (Causal Language Modeling)
- **Direcao**: unidirecional — só ve palavras anteriores
- **Forca**: geração de texto fluente e coerente
- **Uso**: chatbots, geração de código, escrita criativa, instruções
- **Limitação**: compreensão pode ser inferior ao BERT para tarefas especificas

```
Treinamento GPT:
"O gato sentou no" -> prever "tapete"
```

### T5 (Text-to-Text Transfer Transformer)

- **Arquitetura**: encoder-decoder completo
- **Treinamento**: toda tarefa e formulada como texto-para-texto
- **Direcao**: encoder bidirecional + decoder autorregressivo
- **Forca**: versatilidade — uma arquitetura para tudo
- **Uso**: traducao, sumarização, Q&A, classificação

```
Treinamento T5:
Input:  "traduza ingles para portugues: The cat sat on the mat"
Output: "O gato sentou no tapete"
```

### Comparação rapida

```
+----------+----------------+-------------------+-----------------+
| Modelo   | Arquitetura    | Direcao           | Tarefa principal|
+----------+----------------+-------------------+-----------------+
| BERT     | Encoder-only   | Bidirecional      | Compreensao     |
| GPT      | Decoder-only   | Unidirecional     | Geracao         |
| T5       | Enc-Dec        | Bi + Uni          | Ambas           |
+----------+----------------+-------------------+-----------------+
```

Para uso on-premise em 2024-2026, a maioria das aplicações usa modelos **decoder-only** (familia GPT): LLaMA, Mistral, Qwen, Phi, DeepSeek. Eles cobrem geração de texto, chat, código e — com as técnicas certas — também tarefas de compreensão.

---

## Resumo do capítulo

- O Transformer substituiu RNNs/LSTMs ao processar texto em paralelo com mecanismos de atenção
- Self-Attention permite que cada palavra "olhe" para todas as outras e determine relevância
- Multi-Head Attention executa multiplas atencoes em paralelo, capturando relações diferentes
- Positional Encoding adiciona informação de ordem ao processamento paralelo
- Feed-Forward Networks processam a informação capturada pela atenção
- Layer Normalization estabiliza o treinamento de modelos profundos
- O fluxo completo: texto -> tokens -> embeddings -> N camadas de atenção+FFN -> probabilidades -> próxima palavra
- BERT (compreensão), GPT (geração) e T5 (ambos) são as tres familias principais

---

## Fontes

- Vaswani, A. et al. (2017). "Attention is All You Need". *NeurIPS 2017*.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Cap. 3: Looking Inside Large Language Models.
- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 1 e 2.
- Devlin, J. et al. (2019). "BERT: Pré-training of Deep Bidirectional Transformers for Language Understanding". *NAACL 2019*.
- Radford, A. et al. (2018). "Improving Language Understanding by Generative Pré-Training". *OpenAI*.
- Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". *JMLR*.
- Notebooks de referência: `ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb`, `ch02/ch2_Workthrough_LLM_execution.ipynb`.
