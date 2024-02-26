# LSTM-Estudos
Este desafio foi proposto como solução no curso AI900 da Microsoft em parseria com a DIO.

Este desafio foi proposto no curso AI900 da Microsoft em parseria com a DIO no curso AI-900.

# Tecnologias Utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pytorch](https://img.shields.io/badge/python-gray?style=for-the-badge&logo=pytorch&logoColor=red)

## tree do projeto
| Pasta   | Conteúdo   |
|--------|------------|
|network| Contém toda a nossa rede neural|
|data| Contem o código que lida com o dataset diretamente do pytorch|
|notes| Contém uma explicação sobre as redes neurais recorrentes e as LSTMs|

# Passos do Projeto
após estudar sobre RNN, em especial as LSTM, desenvolvi um código basico em pytorch para lidar com um dataset de imagens (de forma a atingir os objetivos do desafio AI-900).
a abordagem foi não convencional, mas definitivamente obteve-se sucessos, como podemos ver:
Testando a acurácia do modelo nas 10000 imagens de teste: 97.63 % utilizando optim SGD, momentum 0.9 com shuffle no dataset
Testando a acurácia do modelo nas 10000 imagens de teste: 97.96 % utilizando optim Adam com shuffle no dataset
Testando a acurácia do modelo nas 10000 imagens de teste: 98.75 % utilizando optim Adamax e sem shuffle no dataset
Testando a acurácia do modelo nas 10000 imagens de teste: 98.89 % utilizando optim Adamax com shuffle no dataset
