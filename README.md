# Disciplina de Inteligência Artificial - Professor Munif - Unicesumar 2026

## Classificação de Potencial Apícola em Flores utilizando Inteligência Artificial (Projeto BeeAI)

### Integrantes da Equipe
* **Lucas de Oliveira Lima - RA: 23000810-2**
* **Alexandre Lozano de Souza - RA: 23003803-2**
* **Gabriel do Nascimento Cano Andrade - RA: 23000555-2** 
* **Daniel Andrade Mendonça -
RA:23000397-2**

---

### 1. Objetivo do Projeto, Contextualização e Hipótese

#### Contextualização e Objetivo
Na apicultura comercial e no ecossistema de **Agrotech**, a seleção do local para instalação de apiários é um fator crítico para o sucesso da produção de mel. O mapeamento manual da flora local consome muito tempo e exige conhecimento botânico especializado. 

O objetivo deste projeto é desenvolver uma solução automatizada de Visão Computacional capaz de analisar imagens de flores capturadas no campo e determinar se elas possuem potencial apícola real. O sistema classifica as espécies detectadas em duas categorias:
1. **Adequadas (Melíferas):** Espécies com alta oferta de néctar e pólen, essenciais para o sustento e produtividade das colmeias (*Apis mellifera*).
2. **Inadequadas (Não-Melíferas):** Espécies com baixo ou nulo valor apícola comercial, atuando como controle negativo.

#### Hipótese Científica
**Hipótese:** Modelos de Deep Learning baseados em **Redes Neurais Convolucionais (CNNs)** apresentarão um desempenho estatisticamente superior (Acurácia e F1-Score) em comparação com algoritmos tradicionais e estatísticos (como KNN e Naive Bayes).

*Justificativa:* Algoritmos clássicos exigem o achatamento (*flatten*) de dados multidimensionais de imagem em uma única linha linear. Esse processo destrói completamente as relações espaciais e de vizinhança entre os pixels, impedindo que os modelos tradicionais capturem texturas complexas e padrões geométricos das pétalas. A CNN, por meio de seus filtros convolucionais e camadas de subamostragem (*pooling*), consegue aprender hierarquias de características estruturais automaticamente, adaptando-se muito melhor à variabilidade biológica do campo.

---

### 2. Dataset e Preparação dos Dados (Explicação da Base)

#### Origem e Reestruturação da Base
O projeto utiliza a base pública **Flowers Recognition**. Para adaptar o problema de classificação botânica tradicional ao escopo apícola, realizamos um mapeamento binário das classes do dataset:
* **Classe Adequadas (Rótulo 1):** `sunflower` (Girassol) e `dandelion` (Dente-de-leão) — espécies com alto índice de visitação por abelhas e excelente retorno de néctar.
* **Classe Inadequadas (Rótulo 0):** `rose` (Rosa), `tulip` (Tulipa) e `daisy` (Margarida) — espécies com baixo apelo ou utilizadas como controle de classificação negativo no nosso pipeline.

#### Engenharia e Pré-processamento de Dados
Para o correto funcionamento dos modelos, todas as imagens passam por um pipeline automatizado que executa:
1. **Filtração de Extensão:** Varredura a nível de arquivo para aceitar estritamente formatos de imagem válidos (`.jpg`, `.jpeg`, `.png`), descartando metadados indesejados.
2. **Redimensionamento:** Ajuste uniforme de todas as dimensões de imagem para **128x128 pixels**, garantindo matrizes de tamanho idêntico na entrada dos classificadores.
3. **Normalização Min-Max:** Divisão dos valores brutos dos pixels (escala RGB de 0 a 255) por 255.0. Isso reescala os dados para o intervalo flutuante **[0.0, 1.0]**, otimizando a estabilidade matemática do gradiente descende das Redes Neurais e os cálculos de distância vetorial.
4. **Divisão Amostral:** Separação randômica e estratificada de **80% dos dados para Treinamento** e **20% para Teste (Avaliação)**.

---

### 3. Métodos da Parte 1 da Disciplina

Cumprindo as especificações da primeira etapa da matéria de IA, foram implementados:
* **K-Nearest Neighbors (KNN):** Um classificador baseado em distância. Ele mapeia o vetor linearizado de 49.152 atributos da flor de teste e calcula as distâncias Euclidianas mais próximas de $k=5$ vizinhos do conjunto de treino para definir a classe por voto majoritário.
* **Gaussian Naive Bayes:** Um classificador estatístico baseado no Teorema de Bayes. Ele assume uma premissa simplificada de independência total entre os pixels da imagem para calcular as probabilidades *a priori* e determinar a chance de a flor ser adequada para mel.

---

### 4. Métodos da Parte 2 da Disciplina

Para a segunda etapa da avaliação, expandimos os testes aplicando arquiteturas de maior complexidade estrutural e aprendizado profundo:
* **Support Vector Machine (SVM):** Cria um hiperplano linear de separação máxima tentando isolar as flores com valor apícola positivo das negativas dentro do espaço vetorial de alta dimensionalidade.
* **Random Forest (Ensemble):** Um método de comitê baseado em 100 árvores de decisão treinadas de forma paralela com subamostragem de dados. A classificação final ocorre pela média das decisões das árvores, mitigando o risco de *overfitting*.
* **Rede Neural Convolucional (CNN):** Modelo de Deep Learning projetado especificamente para visão computacional, composto por:
  * 3 Camadas de Convolução (`Conv2D`) com funções de ativação ReLU, extraindo gradualmente bordas, padrões geométricos e texturas de pétalas.
  * Camadas de Max Pooling (`MaxPooling2D`) para reduzir a dimensionalidade espacial, mantendo os recursos invariantes.
  * Camada de Achatamento (`Flatten`) interligando os mapas de feições extraídos a uma camada densa oculta de 64 neurônios.
  * 1 Camada de Saída com ativação **Sigmoide**, responsável por gerar um valor probabilístico exato entre 0 e 1 para a classificação binária.

---

### 5. Funcionamento Prático da Arquitetura do Código

A aplicação foi projetada dividindo responsabilidades e evitando acoplamento excessivo. O ecossistema está organizado em 4 módulos fundamentais:

* **`tratar_imagens.py`:** Orquestra a infraestrutura de arquivos. Em vez de copiar fisicamente gigabytes de dados no disco rígido do servidor, o script lê a pasta de origem do Kaggle e cria **links simbólicos (symlinks)** apontando para as novas pastas de classes binárias (`adequadas` e `inadequadas`).
* **`carregar_Imagens.py`:** Responsável pelo pipeline de Entrada/Saída (I/O). Utiliza a biblioteca `OpenCV` para carregar as matrizes de imagem de forma eficiente, executa o redimensionamento matemático e a divisão vetorial da normalização para $[0, 1]$, retornando os conjuntos prontos estruturados em arrays `NumPy`.
* **`gerar_graficos.py`:** Módulo exclusivo para avaliação estatística e visual. Isola a biblioteca `Matplotlib` e gera de forma automatizada a curva de aprendizado de perda (*Loss Curve*) da Rede Neural profunda, a matriz de confusão e o gráfico de desempenho comparativo geral.
* **`main.py`:** O arquivo central. Ele importa as funções dos arquivos, faz a divisão dos dados em subconjuntos através do `train_test_split`, remodela tridimensionalmente os arrays para a CNN, aplica a função de achatamento (*flatten*) para alimentar os modelos tradicionais, realiza os treinamentos e dispara a plotagem final de métricas.

---

### 6. Comparação dos Resultados e Conclusão


| Método de IA | Tipo | Acurácia Alcançada |
| :--- | :--- | :--- |
| **KNN** | Parte 1 | `0.52` |
| **Naive Bayes** | Parte 1 | `0.69` |
| **SVM** | Parte 2 | `0.65` |
| **Random Forest** | Parte 2 | `0.76` |
| **Rede Neural (CNN)** | Parte 2 | `0.85` |

#### Análise Crítica e Conclusão
Os testes práticos confirmaram a nossa hipótese inicial. O modelo baseado em **Redes Neurais Convolucionais (CNN)** alcançou a maior acurácia de classificação, demonstrando sua robustez superior em capturar o arranjo biológico tridimensional das pétalas sem perder a correlação de proximidade entre os pixels. 

O Random Forest apresentou-se como a melhor alternativa entre os algoritmos que utilizam o vetor plano (*flatten*), se sobressaindo ao SVM e ao KNN pela eficiência das decisões combinadas. O Naive Bayes apresentou limitações significativas ao assumir independência entre dados estruturados de imagem.

Concluímos que a integração de Visão Computacional profunda em sistemas de **Agrotech** é altamente viável, viabilizando soluções confiáveis de monitoramento ecológico e planejamento para a apicultura sustentável e industrial de mel.

---

### 🛠️ Como Instalar e Rodar o Ecossistema

1. Certifique-se de ter clonado a estrutura de arquivos modulares completa:
   ```text
   ├── main.py
   ├── tratar_imagens.py
   ├── carregar_Imagens.py
   ├── gerar_graficos.py
   ├── README.md
   ├── flowers/ (dataset)
   └── requirements.txt
