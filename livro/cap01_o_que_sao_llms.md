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

## 1.2 De Markov a Transformers — linha do tempo simplificada

A ideia de modelar linguagem com probabilidades não é nova. Vamos percorrer as etapas principais:

### Cadeias de Markov (1906)

Andrey Markov propos que a probabilidade de um evento depende apenas dos eventos imediatamente anteriores. Aplicado a texto: a próxima palavra depende apenas das ultimas N palavras. E simples, mas limitado — não captura contexto de longo alcance.

```python
# Exemplo conceitual de modelo de Markov para texto
# Dado "o gato", qual a proxima palavra mais provavel?
transicoes = {
    ("o", "gato"): {"sentou": 0.4, "dormiu": 0.3, "comeu": 0.3},
    ("gato", "sentou"): {"no": 0.7, "na": 0.3},
}
# Limitacao: so olha 2 palavras pra tras
```

### N-gramas e modelos estatisticos (1980-2000)

Extensoes das cadeias de Markov que consideram sequencias de N palavras. Trigramas (N=3) e quadrigramas (N=4) foram o estado da arte por decadas. O problema: quanto maior o N, mais dados você precisa, e a memória explode exponencialmente.

### Redes neurais recorrentes — RNNs (2010-2015)

A revolucao comecou quando redes neurais foram aplicadas a sequencias. RNNs processam texto palavra por palavra, mantendo um "estado oculto" que funciona como uma memória. O problema: essa memória degrada ao longo de sequencias longas. Uma RNN lendo um paragrafo inteiro frequentemente "esquece" o que leu no início.

### LSTMs e GRUs (2014-2017)

Long Short-Term Memory (LSTM) e Gated Recurrent Units (GRU) resolveram parcialmente o problema da memória com mecanismos de "portao" que controlam o que lembrar e o que esquecer. Foram o padrão para traducao automática, chatbots e geração de texto até 2017.

### Transformers (2017 - presente)

O paper "Attention is All You Need" (Vaswani et al., 2017) mudou tudo. Em vez de processar texto sequencialmente (palavra por palavra), o Transformer processa todas as palavras ao mesmo tempo, usando um mecanismo chamado **self-attention** para determinar quais palavras são relevantes para quais.

Isso trouxe duas vantagens decisivas:

1. **Paralelismo**: treinar em GPUs ficou ordens de magnitude mais rapido
2. **Contexto de longo alcance**: o modelo consegue "olhar" para qualquer parte do texto de uma vez

Todos os LLMs modernos — GPT, LLaMA, Mistral, Claude, Gemini — são baseados na arquitetura Transformer.

```
Linha do tempo simplificada:

1906  Markov          Probabilidade condicional simples
1980  N-gramas        Estatistica com janela fixa
2010  RNNs            Redes neurais sequenciais
2014  LSTM/GRU        Memoria de longo prazo
2017  Transformer     Atencao paralela (revolucao)
2018  BERT            Encoder-only, representacao
2018  GPT             Decoder-only, geracao
2020  GPT-3           Escala massiva (175B parametros)
2023  LLaMA           Modelos abertos competitivos
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

APIs cobram por token processado. Para uso esporadico, e barato. Para uso intensivo, a conta cresce rapido:

```
Exemplo de custo com API (valores ilustrativos):
- 1 milhao de tokens/dia = ~R$ 150/dia = R$ 4.500/mes
- 10 milhoes de tokens/dia = ~R$ 1.500/dia = R$ 45.000/mes

Exemplo de custo on-premise:
- 1 GPU RTX 4090 (usada): ~R$ 12.000 (custo unico)
- Eletricidade: ~R$ 300/mes
- Tokens ilimitados apos o investimento inicial
```

A partir de certo volume, o custo fixo de hardware se paga em poucos meses.

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

- Alammar, J. & Grootendorst, M. (2024). *Hands-On Large Language Models*, O'Reilly. Cap. 1: An Introduction to Large Language Models.
- Iusztin, P. & Labonne, M. (2024). *LLM Engineer's Handbook*, Packt. Cap. 1: Understanding the LLM Twin Concept and Architecture.
- Wang, C. & Hu, P. (2025). *Hands-On LLM Serving and Optimization*, O'Reilly. Cap. 1: Introduction to Model Serving and Optimization.
- Vaswani, A. et al. (2017). "Attention is All You Need". *Advances in Neural Information Processing Systems*.
