# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Rodrigo Martins

### 1️⃣ Resumo da Arquitetura do Modelo

CNN simples no padrão VGG-mini, com 3 blocos convolucionais e Global Average Pooling antes do classificador. Fluxo completo:

- Entrada 32×32×3, normalizada em `[0, 1]`.
- **Data augmentation** embutida no modelo (`Sequential` interno) com `RandomFlip("horizontal")`, `RandomRotation(0.1)` e `RandomZoom(0.1)`. Como o dataset tem apenas 45.000 imagens de treino após o split, aumentar a variabilidade sem coletar dados adicionais foi fundamental para reduzir overfitting.
- **Bloco 1:** Conv2D(32) → BN → Conv2D(32) → BN → MaxPool(2×2)
- **Bloco 2:** Conv2D(64) → BN → Conv2D(64) → BN → MaxPool(2×2)
- **Bloco 3:** Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2×2)
- **GlobalAveragePooling2D** → Dropout(0.5) → Dense(64, relu) → Dense(10, softmax)

Total: **297.706 parâmetros** treináveis (~1,13 MB antes de otimizar).

**Justificativas técnicas de hiperparâmetros:**

- `GlobalAveragePooling2D` no lugar de `Flatten` corta em ~90% os parâmetros antes do classificador denso e alinha o modelo ao objetivo Edge AI.
- `ReduceLROnPlateau(factor=0.5, patience=2)` combinado com `Adam` foi escolhido porque a perda de validação oscilava com LR fixo de 1e-3. Durante o treino final, o schedule reduziu o LR duas vezes (épocas 17 e 23) e cada corte produziu queda visível em `val_loss`.
- `batch_size=128` estabiliza as estatísticas de BatchNormalization e reduz o overhead por step em CPU, comparado ao batch 64 utilizado no experimento inicial.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow** 2.21.0 — treino, `TFLiteConverter` e `tf.lite.Interpreter`
- **NumPy** 2.4.6 — manipulação de arrays na inferência
- **Python** 3.11

### 3️⃣ Técnica de Otimização do Modelo

**Dynamic Range Quantization** via TensorFlow Lite. Em `optimize_model.py`, o `TFLiteConverter` é configurado com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, o que quantiza os pesos do modelo para inteiros de 8 bits enquanto mantém entrada/saída em float32. É a técnica de otimização mais leve de implementar (não exige dataset representativo) e já produz reduções substanciais de tamanho — adequada para o cenário Edge deste projeto.

### 4️⃣ Resultados Obtidos

| Métrica | Valor |
|---|---|
| Acurácia final de validação | **0,8230 (82,30%)** |
| Perda final de validação | 0,5281 |
| Tamanho `model.h5` | 3,52 MB |
| Tamanho `model.tflite` | 314 KB |
| Redução após quantização | **91,29%** |

### 5️⃣ Comentários Adicionais

O projeto foi desenvolvido em dois experimentos:

- **Experimento 1** — CNN baseline com 1 Conv2D por bloco, batch 64, sem schedule de learning rate. Atingiu val_acc = 73,58% em 25 épocas, com `val_loss` ainda descendo na última época — sinal de que faltava tempo efetivo de aprendizado, não capacidade.
- **Experimento 2 (final)** — mesma quantidade de blocos, mas com 2 Conv2D cada, batch 128 e `ReduceLROnPlateau`. Ganho de **+8,72 pontos percentuais** sem exceder as regras (3 blocos, mesmos componentes obrigatórios).

**Decisão técnica interessante:** em vez de adicionar um 4º bloco, optei por dobrar as convoluções dentro de cada bloco (padrão VGG-mini). Isso aumenta a capacidade de extração de features sem alterar o campo receptivo espacial nem exceder o limite de blocos, e mantém o `.tflite` em ~314 KB — ainda apropriado para Edge.

**Limitação real:** o `val_loss` continuou caindo até a última época (25), então o teto de épocas — não a arquitetura — passou a ser o gargalo de acurácia. `EarlyStopping` foi mantido por ser requisito, mas não chegou a disparar em nenhum dos experimentos. Estimo que 40-50 épocas dariam mais 2-3 pp de acurácia com o mesmo modelo.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao rodar `python run_inference.py`:

```
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat      | real=cat
Amostra 2: predito=ship     | real=ship
Amostra 3: predito=ship     | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog     | real=frog
```

Todas as 5 amostras foram classificadas corretamente.