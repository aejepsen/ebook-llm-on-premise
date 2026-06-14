# Capítulo 1 — O Que São LLMs e Por Que Rodar On-Premise

## 1.1 O que é um modelo de linguagem

Imagine que você está escrevendo uma mensagem no celular e o teclado sugere a próxima palavra. "Bom" -> "dia". "Tudo" -> "bem". Essa funcionalidade simples e, na essência, um **modelo de linguagem**: um sistema que calcula a probabilidade da próxima palavra dado um contexto anterior.

Um modelo de linguagem é uma função matemática treinada em grandes volumes de texto. Ele aprende padrões estatisticos da linguagem humana — quais palavras tendem a aparecer juntas, como frases são estruturadas, quais ideias se conectam. Não "entende" o texto como nos entendemos, mas captura regularidades com precisão suficiente para gerar texto coerente, responder perguntas e até escrever código.

A formula fundamental e simples:

```
P(proxima_palavra | palavras_anteriores)
```

Dado um histórico de palavras, qual é a probabilidade de cada palavra possível ser a próxima? O modelo calcula essa distribuição de probabilidade e escolhe (ou amostra) a próxima palavra. Repete o processo, e você tem geração de texto.

Quando dizemos **Large Language Model (LLM)**, estamos falando de modelos de linguagem com bilhoes de parâmetros, treinados em terabytes de texto. A escala e o que transforma um corretor ortografico sofisticado em algo que parece "inteligente". Modelos como GPT-4, LLaMA, Mistral e Qwen operam nessa escala.

## 1.2 De Bayes a Transformers — os gigantes por trás dos LLMs

A ideia de modelar linguagem com matemática não nasceu ontem. LLMs são o resultado de séculos de trabalho de pensadores brilhantes. Conhecer essas raízes ajuda a entender por que os modelos funcionam como funcionam.

### Thomas Bayes e a probabilidade condicional (1763)

O reverendo **Thomas Bayes** desenvolveu o que hoje chamamos de **Teorema de Bayes** — uma forma de atualizar a probabilidade de uma hipótese com base em novas evidências. Bayes morreu em 1761 sem publicar o trabalho; seu amigo **Richard Price** encontrou os manuscritos e os apresentou à Royal Society em 1763. A fórmula:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

Isso é a pedra fundamental de tudo que veio depois. Quando um LLM calcula "qual a próxima palavra mais provável dado o contexto", ele está fazendo essencialmente inferência bayesiana. Toda a modelagem estatística de linguagem — de n-gramas a Transformers — descende dessa ideia.

### Alan Turing e a máquina que pensa (1950)

**Alan Turing** não só inventou a computação moderna com a Máquina de Turing (1936), como em 1950 publicou o artigo seminal *"Computing Machinery and Intelligence"*, propondo o famoso **Teste de Turing**: se uma máquina consegue conversar com um humano sem que ele perceba que está falando com uma máquina, ela pode ser considerada "inteligente".

LLMs são, em certo sentido, a primeira tecnologia que se aproxima de passar nesse teste de forma consistente. Turing imaginou isso 73 anos antes do ChatGPT.

### Claude Shannon e a teoria da informação (1948)

**Claude Shannon** fundou a **Teoria da Informação** e aplicou-a diretamente à linguagem. Ele demonstrou que o inglês tem aproximadamente 1 bit de entropia por caractere — ou seja, a linguagem é altamente previsível e redundante. Shannon já usava modelos estatísticos de n-gramas para prever a próxima letra de um texto nos anos 1950.

O conceito de **entropia** que ele definiu é usado até hoje: a *perplexidade* (métrica padrão para avaliar modelos de linguagem) é derivada diretamente da entropia de Shannon.

### Cadeias de Markov (1906)

**Andrey Markov** propôs que a probabilidade de um evento depende apenas dos eventos imediatamente anteriores. Aplicado a texto: a próxima palavra depende apenas das últimas N palavras. É simples, mas limitado — não captura contexto de longo alcance.

```python
# Exemplo conceitual de modelo de Markov para texto
# Dado "o gato", qual a próxima palavra mais provável?
transicoes = {
    ("o", "gato"): {"sentou": 0.4, "dormiu": 0.3, "comeu": 0.3},
    ("gato", "sentou"): {"no": 0.7, "na": 0.3},
}
# Limitação: só olha 2 palavras pra trás
```

### Warren McCulloch e Walter Pitts — o primeiro neurônio artificial (1943)

Antes mesmo de Turing publicar sobre inteligência artificial, **McCulloch e Pitts** propuseram o primeiro modelo matemático de um neurônio. Simples — apenas operações lógicas AND, OR, NOT — mas plantou a semente das redes neurais que, 80 anos depois, dariam origem aos LLMs.

### Frank Rosenblatt e o Perceptron (1958)

**Rosenblatt** construiu o **Perceptron**, a primeira rede neural que aprendia ajustando pesos automaticamente. Era limitado (só resolvia problemas linearmente separáveis), mas provou que máquinas podiam aprender a partir de dados. O conceito de ajustar pesos por gradiente é exatamente o que acontece no treino de um LLM — só que em escala de bilhões de parâmetros.

### N-gramas e modelos estatísticos — Jelinek e a IBM (1980-2000)

**Frederick Jelinek** e sua equipe na IBM aplicaram modelos de n-gramas ao reconhecimento de fala nos anos 1980, transformando a ideia teórica de Shannon em ferramenta prática. N-gramas consideram sequências de N palavras consecutivas. Trigramas (N=3) e quadrigramas (N=4) foram o estado da arte por décadas. O problema: quanto maior o N, mais dados você precisa, e a memória explode exponencialmente. A famosa frase de Jelinek resume a era: *"Toda vez que demito um linguista, o desempenho do sistema melhora."*

### Redes neurais recorrentes — Elman e Jordan (1986-1990)

A ideia de redes recorrentes surgiu com **David Rumelhart**, **Geoffrey Hinton** e **Ronald Williams** em 1986, que publicaram o algoritmo de **backpropagation through time**. **Jeffrey Elman** (1990) e **Michael Jordan** (1986) formalizaram as arquiteturas de RNN mais conhecidas. RNNs processam texto palavra por palavra, mantendo um "estado oculto" que funciona como uma memória. O problema: essa memória degrada ao longo de sequências longas — o chamado *vanishing gradient problem*. Uma RNN lendo um parágrafo inteiro frequentemente "esquece" o que leu no início.

### LSTMs e GRUs — Hochreiter, Schmidhuber e Cho (1997-2014)

**Sepp Hochreiter** e **Jürgen Schmidhuber** publicaram a Long Short-Term Memory (LSTM) em 1997, resolvendo o problema do gradiente que desaparece com mecanismos de "portão" que controlam o que lembrar e o que esquecer. Em 2014, **Kyunghyun Cho** et al. propuseram a Gated Recurrent Unit (GRU), uma versão simplificada da LSTM com desempenho comparável e treino mais rápido. LSTM e GRU foram o padrão para tradução automática, chatbots e geração de texto até 2017.

### Hinton, LeCun e Bengio — os padrinhos do Deep Learning (2006-2015)

**Geoffrey Hinton**, **Yann LeCun** e **Yoshua Bengio** — ganhadores do Prêmio Turing em 2018 — foram os responsáveis por manter viva a chama das redes neurais durante as "invernos da IA" e, a partir de 2006, provaram que redes profundas (*deep learning*) podiam superar todas as outras abordagens. Sem o trabalho deles, Transformers não existiriam.

### Transformers (2017 - presente)

**Ashish Vaswani**, **Noam Shazeer**, **Niki Parmar** e colegas do Google Brain publicaram *"Attention is All You Need"* em 2017 — o paper que mudou tudo. Em vez de processar texto sequencialmente (palavra por palavra), o Transformer processa todas as palavras ao mesmo tempo, usando um mecanismo chamado **self-attention** para determinar quais palavras são relevantes para quais.

Isso trouxe duas vantagens decisivas:

1. **Paralelismo**: treinar em GPUs ficou ordens de magnitude mais rápido
2. **Contexto de longo alcance**: o modelo consegue "olhar" para qualquer parte do texto de uma vez

Todos os LLMs modernos — GPT, LLaMA, Mistral, Claude, Gemini — são baseados na arquitetura Transformer.

```
Linha do tempo — dos gigantes aos LLMs:

1763  Bayes           Teorema de Bayes (probabilidade condicional)
1906  Markov          Cadeias de Markov (probabilidade sequencial)
1936  Turing          Máquina de Turing (computação universal)
1943  McCulloch/Pitts Primeiro neurônio artificial
1948  Shannon         Teoria da Informação (entropia na linguagem)
1950  Turing          "Computing Machinery and Intelligence"
1958  Rosenblatt      Perceptron (primeira rede neural treinável)
1980  Jelinek/IBM     N-gramas aplicados a linguagem
1986  Rumelhart/Hinton Backpropagation through time (RNNs)
1990  Elman           Rede recorrente para sequências
1997  Hochreiter/Schmidhuber  LSTM (memória de longo prazo)
2006  Hinton          Deep Learning (redes profundas viáveis)
2014  Cho et al.      GRU (LSTM simplificada)
2017  Vaswani et al.  Transformer (atenção paralela — revolução)
2018  Google/OpenAI   BERT (encoder) e GPT (decoder)
2020  OpenAI          GPT-3 (escala massiva, 175B parâmetros)
2023  Meta            LLaMA (modelos abertos competitivos)
2024  Mistral/Qwen    Modelos eficientes, rodando local
```

## 1.3 Por que rodar on-premise: privacidade, custo, controle, latência

Usar uma API como OpenAI ou Anthropic e conveniente. Você faz uma chamada HTTP e recebe a resposta. Mas essa conveniência tem custos que nem sempre são obvios.

### Privacidade e conformidade regulatória

Quando você envia dados para uma API externa, esses dados cruzam redes publicas e são processados em servidores que você não controla. Para muitas industrias, isso é inaceitável:

- **Saude**: dados de pacientes (LGPD, HIPAA) não podem sair da infraestrutura controlada
- **Financeiro**: informações de clientes e transações tem regulamentação rigorosa
- **Juridico**: documentos confidenciais de clientes não podem ser enviados a terceiros
- **Governo**: dados classificados tem restrições legais de processamento

Rodar on-premise significa que os dados nunca saem da sua rede. O modelo roda no seu servidor, processa os dados localmente e retorna a resposta sem que nenhum byte cruze a internet.

### Custo em escala

APIs cobram por token processado. Para uso esporádico, é barato. Para uso intensivo, a conta cresce rápido:

```
Comparativo de custo (referência: junho/2026, preços em USD)

═══ Cloud API ═══
OpenAI GPT-4o:      $2.50/1M tokens input, $10.00/1M tokens output
Anthropic Claude:   $3.00/1M tokens input, $15.00/1M tokens output
Google Gemini 1.5:  $1.25/1M tokens input, $5.00/1M tokens output

Cenário: 1M tokens/dia (input+output mix)
- GPT-4o:  ~$6/dia  = ~$180/mês  = ~$2.160/ano
- Claude:  ~$9/dia  = ~$270/mês  = ~$3.240/ano

Cenário: 10M tokens/dia
- GPT-4o:  ~$60/dia = ~$1.800/mês = ~$21.600/ano
- Claude:  ~$90/dia = ~$2.700/mês = ~$32.400/ano

═══ On-Premise (TCO — Total Cost of Ownership) ═══
Hardware: 1x RTX 4090 24GB (~$1.600 usada, ~$2.000 nova)
Depreciação: 3 anos linear = ~$56/mês (nova) ou ~$44/mês (usada)
Eletricidade: 450W × 24h × 30d × $0.12/kWh = ~$39/mês
Manutenção (estimativa): ~$20/mês
─────────────────────────────────────────────────
TCO mensal: ~$115/mês (~$1.380/ano)
Tokens: ilimitados (limitado apenas pelo throughput da GPU)
Throughput estimado (Qwen 7B Q4): ~30 tok/s = ~2.6M tokens/dia

Break-even vs GPT-4o: ~2 meses (no cenário 10M tok/dia)
Break-even vs Claude:  ~1.5 meses (no cenário 10M tok/dia)
```

**Nota importante**: o custo on-premise inclui **depreciação do hardware** (vida útil de 3 anos), eletricidade e manutenção. Muitas comparações ignoram a depreciação, o que dá uma falsa vantagem ao on-premise. Mesmo assim, a partir de ~2M tokens/dia, on-premise já é mais barato — e a vantagem cresce linearmente com o volume.

### Controle total

Com uma API, você depende do provedor. Ele pode mudar o modelo, alterar precos, modificar politicas de uso ou até descontinuar o serviço. Rodando on-premise:

- Você escolhe exatamente qual modelo usar
- Você controla versões e atualizações
- Você decide os parâmetros de geração (temperatura, top-p, etc.)
- Você pode fazer fine-tuning para seu dominio específico
- Nenhuma dependência externa critica

### Latência

Uma chamada de API envolve: serializar a requisicao, enviar pela internet, esperar na fila do provedor, processar, retornar pela internet. Isso adiciona latência variável que pode ir de 200ms a vários segundos.

On-premise, a latência e deterministica é geralmente menor — especialmente critico para aplicações em tempo real como assistentes de voz, análise de logs em produção ou chatbots com expectativa de resposta instantanea.

## 1.4 Cloud vs On-Premise: trade-offs reais

Não existe solução universal. A decisao depende do seu contexto:

```
+----------------------+-------------------+-------------------+
| Criterio             | Cloud (API)       | On-Premise        |
+----------------------+-------------------+-------------------+
| Setup inicial        | Minutos           | Horas/dias        |
| Custo baixo volume   | Baixo             | Alto              |
| Custo alto volume    | Alto              | Baixo             |
| Privacidade          | Limitada          | Total             |
| Controle do modelo   | Nenhum            | Total             |
| Manutencao           | Zero              | Voce gerencia     |
| Escala               | Elastica          | Limitada ao HW    |
| Modelos disponiveis  | Proprietarios     | Open-source       |
| Latencia             | Variavel          | Deterministica    |
| Fine-tuning          | Limitado/caro     | Livre             |
| Disponibilidade      | Depende provedor  | Depende voce      |
+----------------------+-------------------+-------------------+
```

**Quando cloud faz mais sentido**: prototipagem rapida, volume baixo, equipe pequena sem expertise em infra, necessidade de modelos proprietarios de ponta (GPT-4o, Claude Opus).

**Quando on-premise faz mais sentido**: volume alto de requisicoes, dados sensiveis, necessidade de customização profunda, latência critica, orcamento previsível.

**Abordagem hibrida**: muitas empresas usam ambos. API para tarefas que exigem os melhores modelos proprietarios, on-premise para tarefas de volume alto com modelos open-source adequados.

## 1.5 Casos de uso corporativos

LLMs on-premise já são realidade em diversas aplicações empresariais:

### Atendimento ao cliente

Chatbots internos que respondem perguntas sobre produtos, politicas e procedimentos. O modelo e treinado (fine-tuned) com a base de conhecimento da empresa e responde sem expor dados de clientes a terceiros.

```python
# Exemplo conceitual: chatbot de atendimento com Ollama
import requests

def responder_cliente(pergunta: str) -> str:
    """Envia pergunta ao modelo local e retorna resposta."""
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",  # modelo rodando localmente
            "prompt": f"Voce e um atendente. Responda: {pergunta}",
            "stream": False
        }
    )
    return resposta.json()["response"]
```

### Documentação e sumarização

Resumir contratos, extrair clausulas-chave de documentos juridicos, gerar relatorios a partir de dados brutos. Tudo processado internamente, sem risco de vazamento.

### Assistente de código

Modelos como CodeLlama, DeepSeek Coder e Qwen2.5-Coder rodam localmente e oferecem autocompletar, revisao de código e geração de testes — sem enviar seu código proprietario para servidores externos.

### Análise de dados e BI

LLMs que interpretam dashboards, geram consultas SQL a partir de perguntas em linguagem natural e explicam anomalias em métricas de negocio.

### Processamento de documentos

Extração de informações de notas fiscais, laudos medicos, formularios escaneados. Combinando OCR com LLM, você transforma documentos não-estruturados em dados estruturados.

## 1.6 O que você vai aprender neste livro

Este livro é um guia prático para rodar LLMs na sua própria infraestrutura. Cada capítulo constroi sobre o anterior:

1. **O que são LLMs** (este capítulo) — fundamentos e motivação
2. **Arquitetura Transformer** — como esses modelos funcionam por dentro
3. **Setup do ambiente** — hardware, software e configuração completa
4. **Tokenização e embeddings** — como texto vira números
5. **Inferência com Ollama e vLLM** — rodando modelos na prática
6. **Quantização** — reduzindo modelos para hardware modesto
7. **Fine-tuning** — adaptando modelos ao seu dominio
8. **RAG (Retrieval-Augmented Generation)** — conectando modelos a bases de conhecimento
9. **Serving em produção** — APIs, load balancing, monitoramento
10. **Segurança e governanca** — guardrails, auditoria, compliance

Ao final, você tera a capacidade de avaliar, instalar, configurar e colocar em produção um LLM on-premise — desde um projeto pessoal em uma única GPU até um deploy corporativo multi-node.

Não é necessário ser especialista em machine learning. Se você sabe Python básico e tem familiaridade com terminal Linux, você tem o suficiente para comecar.

---

## Resumo do capítulo

- Um modelo de linguagem calcula a probabilidade da próxima palavra dado um contexto
- LLMs são modelos de linguagem com bilhoes de parâmetros, baseados na arquitetura Transformer
- A evolução foi: Markov -> N-gramas -> RNNs -> LSTM -> Transformers
- Rodar on-premise oferece privacidade, controle, custo previsível e latência deterministica
- A decisao cloud vs on-premise depende de volume, sensibilidade dos dados e capacidade da equipe
- Casos de uso corporativos incluem atendimento, documentação, código, análise e processamento de documentos

---

## Fontes

- Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances". *Philosophical Transactions of the Royal Society*.
- Turing, A. (1950). "Computing Machinery and Intelligence". *Mind*, 59(236), 433-460.
- Shannon, C. (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*, 27, 379-423.
- McCulloch, W. & Pitts, W. (1943). "A Logical Calculus of Ideas Immanent in Nervous Activity". *Bulletin of Mathematical Biophysics*, 5, 115-133.
- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain". *Psychological Review*, 65(6), 386-408.
- Vaswani, A. et al. (2017). "Attention is All You Need". *Advances in Neural Information Processing Systems*.
- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Cap. 1.
- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 1.
- Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*, O'Reilly. Cap. 1.
