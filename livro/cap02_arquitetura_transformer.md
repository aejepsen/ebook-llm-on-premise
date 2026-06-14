# Capitulo 2 — Arquitetura Transformer

## 2.1 O paper "Attention is All You Need" — contexto historico

Em junho de 2017, oito pesquisadores do Google publicaram um artigo que mudaria a inteligencia artificial para sempre: **"Attention is All You Need"** (Vaswani et al., 2017). O titulo era uma provocacao deliberada.

Na epoca, o estado da arte em processamento de linguagem natural (NLP) era dominado por redes recorrentes — RNNs e LSTMs. Esses modelos processavam texto sequencialmente, palavra por palavra, como alguem lendo um livro da esquerda para a direita sem poder voltar a pagina. Funcionava, mas tinha dois problemas graves:

1. **Lentidao**: por ser sequencial, nao aproveitava o paralelismo massivo das GPUs
2. **Memoria curta**: em textos longos, informacoes do inicio se perdiam ate chegar ao final

O paper propunha algo radical: descartar completamente a recorrencia e usar apenas **mecanismos de atencao**. O resultado foi o **Transformer** — uma arquitetura que processa todas as palavras simultaneamente e aprende quais palavras devem "prestar atencao" em quais.

Os resultados foram imediatos. O Transformer superou todos os modelos existentes em traducao automatica (ingles-alemao, ingles-frances) e fez isso treinando em uma fracao do tempo. Em menos de dois anos, toda a industria de NLP migrou para variantes do Transformer.

## 2.2 Encoder vs Decoder — quando usar cada um

O Transformer original tem duas metades: o **Encoder** e o **Decoder**. Pense neles como duas funcoes complementares:

- **Encoder**: le e compreende o texto de entrada. Produz uma representacao numerica rica do significado.
- **Decoder**: gera texto novo, palavra por palavra, usando a representacao do encoder (ou sozinho).

### Analogia da traducao

Imagine um tradutor humano trabalhando com um documento:

1. Primeiro, ele **le o documento inteiro** em portugues e compreende o significado (encoder)
2. Depois, ele **escreve a traducao** em ingles, palavra por palavra, consultando sua compreensao (decoder)

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

Hoje, a maioria dos LLMs que voce vai rodar on-premise (LLaMA, Mistral, Qwen, Phi) sao **decoder-only**. Eles recebem um texto de entrada (prompt) e geram a continuacao.

## 2.3 Self-Attention explicado com analogia

Self-Attention e o coracao do Transformer. Vamos entender com uma analogia antes de ver a mecanica.

### A analogia da sala de aula

Imagine uma sala de aula onde cada aluno e uma palavra em uma frase. A frase e:

> "O gato sentou no tapete porque ele estava cansado"

Cada aluno (palavra) precisa entender sua relacao com todos os outros alunos. O mecanismo de atencao funciona assim:

1. Cada aluno faz **tres perguntas**:
   - **Query (Q)**: "O que eu estou procurando?" — o que esta palavra precisa saber
   - **Key (K)**: "O que eu tenho para oferecer?" — o que esta palavra representa
   - **Value (V)**: "Qual informacao eu carrego?" — o conteudo real desta palavra

2. Cada aluno compara sua **Query** com as **Keys** de todos os outros alunos
3. As comparacoes que dao "match" alto recebem mais peso
4. O resultado final de cada aluno e uma mistura ponderada dos **Values** de todos

No exemplo: quando o modelo processa a palavra "ele", a Query de "ele" vai ter alta compatibilidade com a Key de "gato" — o modelo aprende que "ele" se refere a "gato", nao a "tapete". Isso e **resolucao de correferencia**, e o Transformer faz isso naturalmente.

### A matematica (simplificada)

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Passo a passo:

1. `Q * K^T` — multiplica queries por keys, gerando scores de compatibilidade
2. `/ sqrt(d_k)` — normaliza para evitar valores muito grandes (d_k = dimensao das keys)
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

Uma unica operacao de atencao captura um tipo de relacao. Mas linguagem e multifacetada. Na frase "O banco do parque perto do banco financeiro", a palavra "banco" tem relacoes diferentes com "parque" (banco de sentar) e "financeiro" (instituicao).

**Multi-Head Attention** resolve isso executando a atencao multiplas vezes em paralelo, cada uma com seus proprios pesos (Q, K, V) aprendidos. Cada "cabeca" (head) pode focar em um aspecto diferente:

- Uma cabeca pode aprender relacoes sintaticas (sujeito-verbo)
- Outra pode capturar relacoes semanticas (sinonimos, antonimos)
- Outra pode focar em distancia posicional (palavras vizinhas)
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

O modelo GPT-3, por exemplo, usa 96 cabecas de atencao. Cada uma com 128 dimensoes, totalizando 12.288 dimensoes (96 x 128 = 12.288). O LLaMA 3 70B usa 64 cabecas com 128 dimensoes cada.

## 2.5 Positional Encoding — por que a ordem importa

O Transformer processa todas as palavras simultaneamente. Isso e otimo para velocidade, mas cria um problema: sem informacao de ordem, "o gato comeu o rato" e identico a "o rato comeu o gato". O modelo nao sabe qual palavra veio primeiro.

**Positional Encoding** resolve isso adicionando um sinal matematico unico a cada posicao. E como numerar as cadeiras em um cinema — cada assento tem uma coordenada unica.

O paper original usou funcoes senoidais:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))
```

Onde `pos` e a posicao da palavra e `i` e a dimensao. Essas funcoes geram um padrao unico para cada posicao, e o modelo aprende a interpretar esses padroes como informacao de ordem.

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

Modelos modernos como LLaMA usam **RoPE (Rotary Position Embedding)**, uma evolucao que codifica posicoes relativas (distancia entre palavras) em vez de posicoes absolutas, permitindo que o modelo generalize melhor para sequencias mais longas que as vistas no treinamento.

## 2.6 Feed-Forward Network

Apos a atencao determinar **quais** palavras sao relevantes, a **Feed-Forward Network (FFN)** processa **o que fazer** com essa informacao. E uma rede neural densa aplicada independentemente a cada posicao:

```
FFN(x) = ReLU(x * W1 + b1) * W2 + b2
```

A FFN tipicamente expande a dimensao (ex: de 4096 para 16384), aplica uma nao-linearidade (ReLU, GELU ou SiLU), e projeta de volta para a dimensao original. Essa expansao temporaria permite ao modelo computar transformacoes mais ricas.

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

Modelos recentes como LLaMA usam **SwiGLU**, uma variante que substitui a ativacao simples por um mecanismo de gating, melhorando a capacidade do modelo sem aumentar parametros significativamente.

## 2.7 Layer Normalization

Redes neurais profundas sofrem de um problema: a escala dos valores muda conforme os dados fluem pelas camadas. Uma camada pode produzir valores entre -0.01 e 0.01, a proxima entre -100 e 100. Isso torna o treinamento instavel.

**Layer Normalization** resolve isso normalizando os valores em cada camada para ter media 0 e variancia 1:

```
LayerNorm(x) = gamma * (x - media) / sqrt(variancia + epsilon) + beta
```

Onde `gamma` e `beta` sao parametros aprendiveis. O modelo pode aprender a escala e o deslocamento ideais, mas parte de uma base normalizada.

O Transformer original aplica LayerNorm **depois** de cada sub-camada (Post-LN). Modelos modernos usam **Pre-LN** — normalizam **antes** de cada sub-camada — porque isso torna o treinamento mais estavel em modelos muito profundos.

```
Post-LN (original):   x + LayerNorm(SubLayer(x))
Pre-LN  (moderno):    x + SubLayer(LayerNorm(x))
```

A diferenca parece sutil, mas Pre-LN permite treinar modelos com centenas de camadas sem que o gradiente exploda ou desapareca. LLaMA, Mistral e a maioria dos modelos recentes usam **RMSNorm**, uma variante mais eficiente que ignora a media e normaliza apenas pela variancia.

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

Um detalhe crucial em modelos decoder-only: a **mascara causal** impede que o modelo "veja o futuro". Ao processar a posicao 3, ele so pode prestar atencao nas posicoes 0, 1 e 2. Isso garante que a geracao seja autorregressiva — cada palavra gerada depende apenas das anteriores.

```
Matriz de mascara causal (1 = pode ver, 0 = bloqueado):

        pos0  pos1  pos2  pos3
pos0  [  1     0     0     0  ]
pos1  [  1     1     0     0  ]
pos2  [  1     1     1     0  ]
pos3  [  1     1     1     1  ]
```

## 2.9 GPT vs BERT vs T5 — diferencas praticas

Tres familias de modelos emergiram do Transformer, cada uma com foco diferente:

### BERT (Bidirectional Encoder Representations from Transformers)

- **Arquitetura**: encoder-only
- **Treinamento**: mascara palavras aleatorias e pede ao modelo para preve-las (Masked Language Modeling)
- **Direcao**: bidirecional — ve o texto inteiro ao mesmo tempo
- **Forca**: compreensao profunda do texto
- **Uso**: classificacao, busca semantica, NER, analise de sentimento
- **Limitacao**: nao gera texto novo

```
Treinamento BERT:
"O [MASK] sentou no [MASK]" -> prever "gato" e "tapete"
```

### GPT (Generative Pre-trained Transformer)

- **Arquitetura**: decoder-only
- **Treinamento**: prever a proxima palavra (Causal Language Modeling)
- **Direcao**: unidirecional — so ve palavras anteriores
- **Forca**: geracao de texto fluente e coerente
- **Uso**: chatbots, geracao de codigo, escrita criativa, instrucoes
- **Limitacao**: compreensao pode ser inferior ao BERT para tarefas especificas

```
Treinamento GPT:
"O gato sentou no" -> prever "tapete"
```

### T5 (Text-to-Text Transfer Transformer)

- **Arquitetura**: encoder-decoder completo
- **Treinamento**: toda tarefa e formulada como texto-para-texto
- **Direcao**: encoder bidirecional + decoder autorregressivo
- **Forca**: versatilidade — uma arquitetura para tudo
- **Uso**: traducao, sumarizacao, Q&A, classificacao

```
Treinamento T5:
Input:  "traduza ingles para portugues: The cat sat on the mat"
Output: "O gato sentou no tapete"
```

### Comparacao rapida

```
+----------+----------------+-------------------+-----------------+
| Modelo   | Arquitetura    | Direcao           | Tarefa principal|
+----------+----------------+-------------------+-----------------+
| BERT     | Encoder-only   | Bidirecional      | Compreensao     |
| GPT      | Decoder-only   | Unidirecional     | Geracao         |
| T5       | Enc-Dec        | Bi + Uni          | Ambas           |
+----------+----------------+-------------------+-----------------+
```

Para uso on-premise em 2024-2026, a maioria das aplicacoes usa modelos **decoder-only** (familia GPT): LLaMA, Mistral, Qwen, Phi, DeepSeek. Eles cobrem geracao de texto, chat, codigo e — com as tecnicas certas — tambem tarefas de compreensao.

---

## Resumo do capitulo

- O Transformer substituiu RNNs/LSTMs ao processar texto em paralelo com mecanismos de atencao
- Self-Attention permite que cada palavra "olhe" para todas as outras e determine relevancia
- Multi-Head Attention executa multiplas atencoes em paralelo, capturando relacoes diferentes
- Positional Encoding adiciona informacao de ordem ao processamento paralelo
- Feed-Forward Networks processam a informacao capturada pela atencao
- Layer Normalization estabiliza o treinamento de modelos profundos
- O fluxo completo: texto -> tokens -> embeddings -> N camadas de atencao+FFN -> probabilidades -> proxima palavra
- BERT (compreensao), GPT (geracao) e T5 (ambos) sao as tres familias principais

---

## Fontes

- Vaswani, A. et al. (2017). "Attention is All You Need". *NeurIPS 2017*.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Cap. 3: Looking Inside Large Language Models.
- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 1 e 2.
- Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL 2019*.
- Radford, A. et al. (2018). "Improving Language Understanding by Generative Pre-Training". *OpenAI*.
- Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". *JMLR*.
- Notebooks de referencia: `ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb`, `ch02/ch2_Workthrough_LLM_execution.ipynb`.
