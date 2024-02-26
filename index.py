from data.importData import getData
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from network.network import RNN
import torch

train_data, test_data = getData()

data = {
                      #dataset, tamanho do batch, embaralhar, multi threads
    'test': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=5),
    'validation': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=5)
}

#variáveis globais do treinamento da rede
epochs = 10
lr = 0.01

net = RNN(input_size=28, hidden_size=128, num_layers=1, num_classes=10)
gradienteErro = nn.CrossEntropyLoss()
correcaoErro = optim.Adamax(net.parameters(), lr=lr)



#para cada época da rede
for epoch in range(epochs):

    #para cada batch de dados
    for i, (images, labels) in enumerate(data['test']):

        #mostra a época atual
        #print("epoch: ", epoch)

        #mostra o batch atual
        #print(f"images: {images[0]}")

        #mostra as classes esperadas do batch atual
        #print(f"labels: {labels}")
        
        """ 
            redimensiona o batch que é do modelo:
            [
                [
                    [img1]
                ]
                [
                    [img2]
                ]
                ...
            ]
            para o modelo:
            [
                [img1]
                [img2]
                ...
            ]
        """
        images = images.reshape(-1, 28, 28)

        #print(f"imagem após reshpae: {images[0]}")

        #passa o batch pela rede
        outputs = net(images)
        
        #zera o gradiente de erro
        correcaoErro.zero_grad()

        #calcula o erro
        loss = gradienteErro(outputs, labels)

        #calcula o gradiente do erro
        loss.backward()

        #atualiza os pesos
        correcaoErro.step()

        if (i+1) % 100 == 0:
            print(f'Época [{epoch+1}/{epochs}], Step {i+1}/{len(data["test"])}, Loss: {loss.item()}')

            if loss.item() < 0.09:
                lr /= 3

#avaliando o modelo
net.eval()

#passando os dados da rede sem calular gradientes
with torch.no_grad():

    #total de acertos
    correct = 0

    #total de exemplos
    total = 0

    #para cada batch de dados de validação
    for images, labels in data['validation']:

        #redimensiona o batch como feito nos testes
        images = images.reshape(-1, 28, 28)

        #passa o batch pela rede
        outputs = net(images)

        #pega a posição da classe com maior probabilidade
        _, predicted = torch.max(outputs.data, 1)

        #atualiza o total de exemplos
        total = total + labels.size(0)

        #atualiza o total de acertos (soma 1 para cada acerto)
        #predicted == labels retorna um tensor com 1 para cada acerto e 0 para cada erro
        #sum() retorna a soma de todos os valores do tensor gerado por predicted == labels
        #item() retorna o valor do tensor gerado por sum()
        correct = correct + (predicted == labels).sum().item()

print(f'Testando a acurácia do modelo nas 10000 imagens de teste: {100 * correct / total} %')