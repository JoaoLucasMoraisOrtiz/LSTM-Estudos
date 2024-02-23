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
    flowchart TD
    A[X1]-->B[Multiplica por Peso1]-->S[Somatorio = EntradasAnteriores + EntradaAtual + Bias]-->F[Função de ativação] -->C{Falta alguma entrada?}-->|Não|E[Saida Rede];
                                                           C-->|Sim|P2[Multiplica por Peso2] -->|Lembre-se de \n Ao mesmo tempo que este \n resultado retorna, \n Inserir o próximo Xn na entrada| S;

                                                           K[Passa para a próxima entrada: X2, X3, X4, ...] -->B;
  
    ```