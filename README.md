# Aplicação: Machine Translation with Transformer

## Objetivo: 

A proposta do trabalho é implementar o modelo Transformer para Machine Translation com base no paper Attention Is All You Need, principal referência sobre o modelo Transfomer. Avaliar a execução da aplicação de forma distribuída e com provisionamento dinâmico em clusters compostos de instâncias EC2 da AWS


## Metodologia: 

Através do uso dos pacotes mxnet e horovod habilitou-se a aplicação disponibilizada em https://github.com/eric-haibin-lin/AMLC19-GluonNLP/tree/master/03_machine_translation para execução distribuída e com o auxilio da ferramenta CLAP foi implementado o provisionamento automatico da aplicação em cluster da AWS e, posteriormente, foi feita a otimização com base no custo de cada tipo de instância da AWS na unidade de tempo.

Habilitação de Execução Distribuída

A aplicação foi adaptada para uso com o pacote `Horovod` segundo o esqueleto de blocos de código fornecido em https://github.com/horovod/horovod/blob/master/docs/mxnet.rst
A compatibilidade entre pacotes do `Horovod` e `MXNet` com o sistema operacional Ubuntu foi o maior desafio nessa fase. Por fim, o setup que se comportou melhor em termos de facilidade de configuração foi Ubuntu 20.04 com `Horovod 0.22.1` e `MXNet 1.8.0.post0`.

Orquestração dos Nodes e Clusters

Um script de python notebook foi desenvolvido pra utilização do software CLAP com o fim de orquestrar o provisiomento de forma dinâmica dos nodes e clusters hosteados na AWS. O notebook script `machine-translation.ipynb` pode ser encontrado em https://github.com/robertopossidente/optimizer-clap-app e os passos para reprodução do experimento podem ser verificados na última seção desse relatório.

Critério para provisionamento dinâmico: 

* `preço da instância / hora` obtido através de um arquivo .yaml
* desempenho de processamento medido através dos `tempos de iteração` 

Experimento: 

O experimento realizado nesse trabalho foi composto das seguintes etapas:
* Provisionamento de um cluster heterogêneo composto por 6 instâncias EC2 da AWS
* Setup e execução da aplicação Transformer de forma distribuída
* Obtenção dos tempos de iterações de cada nó executando a aplicação, sendo que esses valores indicam o desempenho de processamento durante a execução da aplicação
* Cálculo dos preços de cada instância em `Paramount Iteration / dólar` baseado nos tempos de iterações de cada nó multiplicado pelo `custo de cada instância / hora`
* Otimização do cluster seguindo o critério para provisionamento dinâmico mencionado anteriormente
	
Otimização:

O processo de otimização foi implementado através do módulo `optimizer` utilizado conforme descrito no enunciado do trabalho. A função `optimize_it` do módulo `optimizer` é responsável pelo gerenciamento do processo de otimização através da chamada dos métodos implementados das classes `Reporter` e `Optimizer` em intervalos de 60 segundos. 
Assim, os tempos de iterações de execução da aplicação nos nós são obtidos através do método get_metrics da classe `Reporter` usando o recurso de `ansible local facts` via execução de uma playbook `getfacts.yml` de leitura dos paramêtros de forma remota ṕelo CLAP. Em seguida, os valores de preços correspondentes a cada nó são computados levando em conta o `custo da instância / hora` multiplicado pelo `tempo de iteração` em cada um dos nós. O método retorno um dicionário com identificadores dos nós e respectivos preços calculados.
Com essa informação o método run da classe Optimizer implementa uma lógica de decisão baseada na comparação de preços dos nós, identificando os ids dos nós correspondentes ao maior e menor valor e, posteriormete, executando comandos de adição/grow e, após sucesso dessa operação, remoção/stop de 1 nó do cluster.



## Análise

O processo de otimização é repetido em intervalos de 60 segundos e a otimização se encerra quando todos os nós do cluster possuem um mesmo flavor, o qual apresentou o menor custo durante o experimento. 
Na prática, para um cluster com 6 instâncias de flavors diferentes, a cada 60 segundos o otimizador iŕá adicionar 1 instância ao cluster e, após sucesso, irá remover 1 instância do cluster. Se não houver sucesso na adição da instância, nenhum nó é removido do cluster. Após 5 iterações em intervalos de 60 segundos a expectativa é que o cluster seja homogêneo e da 6 iteração em diante o otimizador não atue, adicionando e removendo instâncias do cluster, até a execução da aplicação finalizar.
O otimizador funcionou perfeitamente durante o experimento, sendo possível obter ao final um cluster homogêneo de 6 instâncias com o flavor que apresentou menor preço segundo os criteŕios de otimização empregados nesse trabalho. Por motivo de simplicidade o preço da instância foi calculado atráves da multiplicação de `tempo de iteração` e `custo da instância / hora`.
Na fase de habilitação de execução distribuida da aplicação com Horovod foram encontradas dificuldades em executar a aplicação em instâncias t2.micro e t2.small devido a pouca memória disponivel. Porém, a msg de erro interna era pouco explicativa e levou-se muito tempo para entende ra causa do problema.
Durante o desenvolvimento do notebook script através do CLAP foram encontradas grandes dificuldades de uso da API de forma adequada e de entendimento de alguns parâmetros de configuração necessários que algumas vezes levaram a uma perda de tempo enorme. Porém, a ferrenta demonstrou-se eficiente e ficou evidente que o problema era conhecimento e experiência com a ferramenta. 
Vale ressaltar que a execução da aplicação de forma distribuída usando `Horovod` e `MXNet` facilita as fases de desenvolvimento e validação do funciomento da aplicação, ao mesmo tempo que trazem grande de performance sobretudo quando empregadas em casos de grande volume de dados. Esse fato dá mais valor ao retorno obtido com a execução do trabalho.

Procedimentos para reprodução do experimento:
* Clone o repositório https://github.com/robertopossidente/optimizer-clap-app no seu diretório de trabalho, ex: /home/ubuntu/MO833-trabalho
* Copie o script optimizer.py para o diretório em que o CLAP foi instalado, sob o path `app/cli/modules`, conforme orientado no enunciado do trabalho
* Copie o notebook script `machine-translation.ipynb` para o diretório em que o CLAP foi instalado.
* Execute o script ./install.sh, o qual fará a configuração do ambiente para execução da aplicação


## Referências 

Relevância do modelo na área de Machine Translation
https://paperswithcode.com/task/machine-translation

Paper mais citado: Attention Is All You Need
https://arxiv.org/abs/1706.03762

MXNet: Distributed Training in MXNet
https://mxnet.apache.org/versions/1.8.0/api/faq/distributed_training

Pacote  de modelos Gluon - Machine Translation with Transformer
https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/packages/gluon/text/transformer.html

Aplicação de Referência: AMLC 2019: Dive into Deep Learning for Natural Language Processing
https://github.com/eric-haibin-lin/AMLC19-GluonNLP/tree/master/03_machine_translation

Treinamento Distribuído: Horovod with MXNet
https://horovod.readthedocs.io/en/stable/mxnet.html
https://github.com/horovod/horovod

