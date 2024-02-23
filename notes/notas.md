# Estudos sobre LSTM
*   ## O que são redes LSTM
    Para compreendermos as redes LSTM (Long Short-Term Memory), devemos primeiro compreender as redes neurais recorrentes.

    ### O que são Redes Neurais Recorrentes
    Imaginemos o seguinte caso: queremos prever o resultado de uma certa empresa X amanhã na bolsa de valores.

    Sabemos que:

    Se ontem e hoje a empresa aumentou de valor, amanhã ela também aumentará.
    Se ontem e hoje a empresa diminuiu de valor, amanhã ela também diminuirá.
    Se ontem e hoje a empresa manteve o seu valor, amanhã ela também manterá.
    <sup>(consideremos que os casos acima são os únicos possíveis por conveniência)</sup>

    Neste caso, devemos considerar o desempenho de 2 dias da empresa para prever seu 3º. Como fazer para uma RNA praticar este comportamento?

    Basicamente, o que vamos fazer é pegar o resultado do dia de ontem, somar a resposta da rede com a entrada do dia de hoje, e então passar esta soma para a rede neural.

    Por fim, a resposta deste 2 input na rede neural será a resposta do nosso dia de amanhã.

    Acompanhe abaixo:

    ```mermaid
    flowchart TD
    A[X1]-->B[Multiplica por Peso1]-->S[Somatorio = EntradasAnteriores + EntradaAtual + Bias]-->F[Função de ativação] -->C{Falta alguma entrada?}-->|Não|E[Saida Rede];
                                                           C-->|Sim|P2[Multiplica por Peso2] -->|Lembre-se de \n Ao mesmo tempo que este \n resultado retorna, \n Inserir o próximo Xn na entrada| S;

                                                           K[Passa para a próxima entrada: X2, X3, X4, ...] -->B;
  
    ```

    ou em código:
    ```python
    #ontem foi 1 e hoje foi 2, queremos saber amanhã
    entradas = [1, 2, ]
    peso1 = float()
    peso2 = float()
    somatorio = float()
    bias1 = float()
    bias2 = float()
    pesoSaida = float()
    for c in range(0, len(entradas)):
        #multiplica por peso1
        multiplicação1 = entradas[c] * peso1

        #soma com as entradas anteriores
        somatorio += multiplicação1 + bias1

        #passa o somatório para a função de ativação
        res = funcAtivacao(somatorio)

        #verifica se falta alguma entrada
        if c != len(entradas) - 1:

            #multiplica por peso2
            multiplicação2 = res * peso2

            #soma com o somatório
            somatorio += multiplicação2
        else:
            #passa o resultado para a camada de saída
            res *= pesoSaida
            res += bias2

    #mostra o resultado da rede quando não houverem mais entradas, prevendo o valor de amanhã
    print(res)
    ```

    Esta é a forma como funciona uma rede neural recorrente. Ela considera resultados anteriores para prever o próximo resultado, e isto é fantástico pois permite termos respostas com contexto, e não previsões separadas.

    O problema deste tipo de rede neural está no fato de que é fácil perceber que caso peso2 > 1 a quantidade de vezes que ele vai multiplicar saídas de rede resultará em um número absurdamente grande, e na hora de fazer a retropropagação do erro, teríamos um problema de gradiente explode, ou seja, o gradiente do erro seria tão grande que não conseguiríamos ajustá-lo corretamente, pois ele ficaria variando em quantidades muito grandes, "saltando" pelo mínimo global sem atingi-lo.

    Por outro lado, caso peso2 < 1, isso gera o problema inverso; um número tão pequeno que faz o erro estagnar e não descer até o mínimo global não importando o quanto treinemos a rede.

    ### Para solucionar estes problemas temos as Redes de Memória de Curto e Longo prazo (LSTM)

* ## O que são LSTM (parte 2)
    Agora que entendemos o funcionamento das redes neurais recorrentes, podemos tentar compreender as redes de memórias de curto e longo prazo.

    Basicamente, estas redes trabalham com 2 "caminhos": um de curto prazo, que retorna uma resposta para cada entrada que colocamos, e um de longo prazo, o qual não alteramos diretamente e que serve como memória dos acontecimentos anteriores para direcionar a próxima resposta de curto prazo.

    Estas redes utilizam funções de ativação sigmoide (sign) e tangente hiperbólica (tanh). Estas funções são escolhidas porque a tanh pode facilmente produzir um "score" para um valor, sempre entre -1 e 1, de forma a evitar explosão do gradiente. Já a função sign permite filtrarmos as entradas dando uma porcentagem, posto que sua resposta é sempre entre 0 e 1.

    Assim, podemos começar a entender as dinâmicas por trás de cada parte deste tipo de rede neural.

    Observemos a imagem:
    <img src='https://www.mdpi.com/sensors/sensors-21-05625/article_deploy/html/images/sensors-21-05625-g001.png'/>
    <sup>fonte: https://www.mdpi.com/sensors/sensors-21-05625/article_deploy/html/images/sensors-21-05625-g001.png</sup>

    Pode parecer muito complicada a princípio, mas vamos explicá-la por partes para ser entendida.

    Começando pela estrutura da nossa rede, podemos ver os dois caminhos mencionados anteriormente: um em cima e um embaixo (ambos nomeados como Ck-1 nesta imagem).

    A linha superior, também chamada de cell state, representa a nossa memória de longo prazo. Já a parte de baixo, chamada de short memory, representa nossa memória de curto prazo.

    Observe que a memória de longo prazo nunca é alterada diretamente, diferente da memória de curto prazo que passa por funções de ativação.

    Repare também que devemos passar nossos dados por 4 gates (portões), um por vez. Cada um deles é responsável por uma ação específica.
    
    * #### Forget Gate (portão do esquecimento):

        É onde tudo irá começar. **Começamos a ler o fluxograma de baixo para cima.**

        Na parte de baixo podemos ver nossa memória de curto prazo, e Xk-1 que é o input X no momento k-1.

        Este input será multiplicado com os pesos Wf. Já nossa memória de curto prazo será multiplicada com os pesos Rf. Em seguida, serão somados e acrescentado um bias, e por fim serão passados a uma função de ativação sigmoid.

        Este é o **exato funcionamento** de uma camada **MLP**.

        Em seguida, a saída desta camada será multiplicada com a nossa memória de longo prazo. Como o resultado de nossa saída da rede é um número entre 0 e 1 (posto que a função utilizada é a sigmoid), quando multiplicamos com a memória de longo prazo, estamos exatamente dizendo quantos por cento da memória de longo prazo queremos considerar.

        Agora que sabemos o quanto da memória de longo prazo que queremos considerar, partimos para o próximo portão.

    * #### Input Gate (portão de entrada) e State Candidate Gate (portão do candidato de estado):

        Estes dois portões funcionam juntos. Inclusive em alguns lugares eles são chamados em conjunto de Input Gate. O objetivo é computar quanto da memória de curto prazo deve ser adicionada à memória de longo prazo.

        Então vamos começar com o Input Gate.

        **Começando como sempre lendo de baixo para cima o fluxograma.** Nele temos exatamente o mesmo que no portão anterior: uma camada de rede neural com a função de ativação sigmoid e pesos Wi multiplicando a entrada, e Ri multiplicando a memória de curto prazo. O bias desta camada é bi. O resultado desta camada é jogado para um multiplicador.

        Agora vamos ver o segundo termo desta multiplicação no portão State Candidate. Temos novamente uma camada de rede neural com pesos Wg para a entrada, e Rg para a memória de curto prazo. O bias desta camada é bg.

        Repare que ao invés de termos uma função de ativação sigmoid, temos como função de ativação a tanh.

        Diferente da sigmoid, a função tanh gera resultados entre -1 e 1, sendo um bom indicador de score, ou seja, se algo deve ser considerado(1), ou evitado(-1). Sabendo que este portão busca adicionar dados à memória de longo prazo, esta função é a responsável por dizer quais dados devem ser inseridos na memória de longo prazo.

        Após computar a saída da função tanh, multiplicamos ela pela saída da função sigmoid. Assim, vamos ter como resposta quantos por cento dos dados gerados pela função tanh devem efetivamente ser adcionados para a memória de longo prazo.

        Em seguida, adicionamos estes dados à memória de longo prazo, por meio de uma soma.
    
    * #### Output Gate (Portão de Saída):
        Aqui vamos ter a **resposta** da nossa iteração k-1 em nossa rede neural LSTM: a **nova memória de curto prazo** que será calculada utilizando a memória de longo prazo que atualizamos.

        **Novamente vamos ler o fluxograma de baixo para cima.**

        Vemos que temos novamente o exato funcionamento de uma camada de rede neural, sendo os pesos Wo para a entrada, Ro para a memória de curto prazo, e utilizando o bias bo. Nesta etapa, é utilizado a função sigmoid, e o resultado é passado para um multiplicador.

        **Agora pela única vez vamos olhar de cima para baixo no fluxograma**

        Temos a memória de longo prazo sendo passada diretamente para uma função tanh, gerando uma resposta, e agora calculamos quantos por cento desta resposta será atribuída à nova memória de curto prazo, colocando a saída desta função também no multiplicador.

        O resultado desta multiplicação é tanto a resposta da rede para a entrada k-1 quanto a nova memória de curto prazo.