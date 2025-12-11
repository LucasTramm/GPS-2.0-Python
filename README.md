## Sistema Inteligente de Otimização de Rotas de Entrega

Este projeto integra diferentes técnicas de Inteligência Artificial e algoritmos de otimização para resolver desafios reais de logística, como encontrar a rota mais curta, definir a melhor ordem para múltiplas entregas e adaptar-se a condições variáveis de tráfego.

## 🚀 Objetivo do Projeto

-Calcular rotas eficientes entre múltiplos pontos de entrega.

-Adaptar os pesos das rotas conforme horário, clima e trânsito.

-Avaliar a qualidade final da rota com base em critérios humanos.

-Apoiar a tomada de decisão em sistemas de entrega e logística.

## 🧠 Tecnologias de IA Utilizadas
1. A*

-Responsável por calcular a menor distância entre dois pontos.

-Usado para obter o custo real de cada trecho da rota.

2. Algoritmo Genético (GA)

-Gera combinações possíveis de ordem de entregas.

-Evolui essas sequências até encontrar a ordem com menor custo total.

-Define a ordem em que o A* será executado.

3. Redes Neurais Artificiais (RNA)

-Preveem o fator de congestionamento baseado em:

-Hora do dia

-Dia da semana

-Condição climática

-Ajustam o peso das arestas antes da execução do A*.

4. Lógica Fuzzy

-Avalia qualitativamente a rota final:

"Boa", "Regular" ou "Cansativa".

-Usa distância e tempo estimado como entradas.

-Gera um score (0 a 10) para apoiar a decisão final.

## 🧩 Funcionamento Geral

-Usuário informa Origem + 4 Pontos de Entrega.

-O GA cria populações de possíveis ordens de visita.

-A RNA prevê fator de congestionamento com base no contexto.

-Os pesos do grafo são ajustados conforme essa previsão.

-O A* calcula as distâncias entre os pontos na ordem indicada pelo GA.

-A Lógica Fuzzy classifica a rota final e gera o score.

-O sistema exibe o resultado no console.

## 📊 Entradas e Saídas

Entradas:

Pontos de entrega

Hora do dia

Dia da semana

Condição climática

Saídas:

Rota final mais eficiente

Quilometragem total

Tempo estimado

Score Fuzzy (0 a 10)

Classificação da rota (Boa / Regular / Cansativa)

## 🛠️ Tecnologias e Bibliotecas

Python

Scikit-Learn (RNA)

Numpy

Custom A* implementation

Implementação própria de Algoritmo Genético
