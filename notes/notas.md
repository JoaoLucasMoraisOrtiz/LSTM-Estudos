# Estudos sobre LSTM
*   ## O que são redes LSTM
    para compreendermos as redes LSTM (Long-Short Time Memory) devemos primeiramente compreender as redes neurais recorrentes.

    ### O que são Redes Neurais Recorrentes
    Imaginemos o seguinte caso:
    Queremos prever o resultado de uma certa empresa X amanhã na bolsa de valores.

    Sabemos que:
    
    * Se **ontem e hoje** a empresa **aumentou** de valor, amanhã ela também aumentar.
    * Se **ontem e hoje** a empresa
    **diminuiu** de valor, amanhá ela também diminuirá
    * Se **ontem e hoje** a empresa **manteve** o seu valor, amanhã ela também manterá.

        <small>consideremos que os casos acima são os únicos possíveis por conveniência</small>
    
    Neste caso, devemos considerar o desempenho de 2 dias da empresa para prever seu 3º. Como fazer para uma RNA praticar este comportamento?

    Basicamente o que vamos fazer é pessar o resultado do dia de ontem, e somar a resposta da rede com a entrada do dia de hoje, e então passar esta soma para a rede neural.

    Por fim, a resposta deste 2 input na rede neural será a resposta do nosso dia de amanhã.

    Acompanhe abaixo:

    ```mermaid
    graph TD;
    X1-->Peso1-->Func.At.-->X2+X1;
    X2+Anterior-->Peso1-->Func.At-->X3+X2+X1;
    X3+X2+X1-->Peso1-->Func.At.-->SaidaRede;
    ```