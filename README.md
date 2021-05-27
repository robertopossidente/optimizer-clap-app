# optimizer-clap-app

Optimizer clapp app é uma aplicação para o CLAP que visa otimizar a execução de 
aplicação em um aglomerado dinamicamente. Estes repositório é um **modelo** e pode 
ser alterado como desejado.

Para usar este app, coloque o arquivo `optimizer.py` no diretório onde o CLAP foi 
instalado (feito o git clone) dentro de `app/cli/modules/`.

Com isso você consegue utilizar esta aplicação diretamente pela linha de comando 
do CLAP. Esta aplicação pode ser chamada utilizando o comando `clapp optimizer run`.
Esta aplicação recebe os seguintes parametros:
* `--cluster-id`: Identificador de um Cluster no CLAP executando a aplicação.
* `--experiment-id`: Nome único do experimento.
* `-vm-price`: Arquivo YAML contendo o preço por hora das VMs (veja o arquivo de exemplo `vm_prices.yaml`)
* `--root-dir` (opcional): Direrótio raiz para salvar informações do experimentos. Irá criar os seguintes diretórios: `<root_dir>/<expriment_id>/app_results`, `<root_dir>/<expriment_id>/optimizer_logs` and `<root_dir>/<expriment_id>/PIs_logs`.
* `--report-time` (opcional): Tempo de espera antes para executar o reporter e o otimizador.

Ao executar a aplicação através do comando `clapp optimizer run` a função `optimizer_run` será executada como ponto de entrada. 

