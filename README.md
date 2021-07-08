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
O processo de otimização foi implementado através do módulo `optimizer` utilizado conforme descrito no enunciado do trabalho. O método ru

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

