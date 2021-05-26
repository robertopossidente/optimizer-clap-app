import os
import time
import click

from typing import Dict

from clap.utils import yaml_load, path_extend
from app.cli.cliapp import clap_command
from app.cli.modules.node import get_node_manager, get_config_db
from app.cli.modules.role import get_role_manager
from app.cli.modules.cluster import get_cluster_config_db, get_cluster_manager

# Get configuration templates of instances (instances.yaml) and
# clusters (~/.clap/configs/clusters)
instances_configuration = get_config_db()
clusters_configuration = get_cluster_config_db()

# Get node, role and cluster managers
node_manager = get_node_manager()
role_manager = get_role_manager()
cluster_manager = get_cluster_manager()


# Class Reporter
class Reporter:
    def get_metrics(self, cluster_id: str, experiment_id: str,
                    pi_logs_dir: str, instance_costs: Dict[str, float]) -> \
            Dict[str, float]:
        pass

    def terminated(self, cluster_id: str, experiment_id: str) -> bool:
        pass

    def fetch_results(self, cluster_id: str, experiment_id: str,
                      output_dir: str):
        pass


# Class Optimizer
class Optimizer:
    def run(self, cluster_id: str, experiment_id: str,
            metrics: Dict[str, float]) -> bool:
        pass


def optimize_it(cluster_id: str, experiment_id: str, vm_price_file: str,
                root_dir: str, report_time: int = 60) -> int:
    experiment_dir = path_extend(root_dir, experiment_id, str(int(time.time())))
    app_results_dir = path_extend(experiment_dir, 'app_results')
    optimizer_logs_dir = path_extend(experiment_dir, 'optimizer_logs')
    pis_logs_dir = path_extend(experiment_dir, 'PIs_logs')
    os.makedirs(app_results_dir, exist_ok=True)
    os.makedirs(optimizer_logs_dir, exist_ok=True)
    os.makedirs(pis_logs_dir, exist_ok=True)

    print(f"Cluster ID: {cluster_id}, Experiment ID: {experiment_id}, "
          f"VM Price File: {vm_price_file}, APP Result Dir: {app_results_dir}, "
          f"Optimizer Logs Dir: {optimizer_logs_dir}, PI Logs Dir: {pis_logs_dir}")
    reporter_obj = Reporter()
    optimizer_obj = Optimizer()
    prices = yaml_load(vm_price_file)

    while True:
        time.sleep(report_time)
        if reporter_obj.terminated(cluster_id, experiment_id):
            reporter_obj.fetch_results(cluster_id, experiment_id, app_results_dir)
            cluster_manager.stop_cluster(cluster_id)
            return 0
        else:
            metrics = reporter_obj.get_metrics(
                cluster_id, experiment_id, pis_logs_dir, prices)
            changed = optimizer_obj.run(cluster_id, experiment_id, metrics)
            print(f"Does cluster changed? {changed}")


# Command-line interface
@clap_command
@click.group(help='Control and manage cluster of nodes using optimizer')
def optimizer():
    pass


@optimizer.command('run')
@click.option('-c', '--cluster-id', default=None, show_default=False,
              type=str, required=True,
              help="Id of the cluster running the application")
@click.option('-e', '--experiment-id', default=None, show_default=False,
              type=str, required=True,
              help="Symbolic name of the experiment")
@click.option('-v', '--vm-price', default=None, show_default=False,
              type=str, required=True,
              help='Path to the YAML file with the price of the VMs')
@click.option('-r', '--root-dir', default=None, show_default=False,
              type=str, required=True,
              help='Root directory where experiment directories will be created')
@click.option('-rt', '--report-time', default=60, show_default=True,
              type=int, required=True,
              help='Time to wait before calling reporter')
def optimizer_run(cluster_id: str, experiment_id: str, vm_price: str,
                  root_dir: str, report_time: int):
    return optimize_it(cluster_id, experiment_id, vm_price, root_dir, report_time)
