import os
import time
import click

from typing import Dict

from clap.utils import yaml_load, path_extend
from app.cli.cliapp import clap_command
from app.cli.modules.node import get_node_manager, get_config_db
from app.cli.modules.role import get_role_manager
from app.cli.modules.cluster import get_cluster_config_db, get_cluster_manager

import yaml
from dataclasses import asdict
from app.cli.modules.node import get_config_db, get_node_manager
from clap.utils import float_time_to_string, path_extend
from clap.executor import SSHCommandExecutor, AnsiblePlaybookExecutor

# Get configuration templates of instances (instances.yaml) and
# clusters (~/.clap/configs/clusters)
instances_configuration = get_config_db()
clusters_configuration = get_cluster_config_db()

# Get node, role and cluster managers
node_manager = get_node_manager()
role_manager = get_role_manager()
cluster_manager = get_cluster_manager()

configuration_db = get_config_db()
# Private's path (usually ~/.clap/private/) will be used for other methods
private_path = node_manager.private_path

# Class Reporter
# You must implement these 3 methods as speficied in the document
class Reporter:
    def get_metrics(self, cluster_id: str, experiment_id: str,
                    pi_logs_dir: str, instance_costs: Dict[str, float]) -> \
            Dict[str, float]:
        cluster = cluster_manager.get_cluster_by_id(cluster_id)
        cluster_dict = asdict(cluster)
        print(yaml.dump(cluster_dict, indent=4))
        cluster_nodes = cluster_manager.get_all_cluster_nodes(cluster_id)
        print(cluster_nodes)
        cluster_nodes_with_type = cluster_manager.get_cluster_nodes_types(cluster_id)
        print(cluster_nodes_with_type)
        cluster_nodes_with_type_dump = cluster_manager.get_cluster_nodes_types(cluster_id)
        print(yaml.dump(cluster_nodes_with_type_dump))

        playbook_file = path_extend('~/.clap/roles/roles/getfacts.yml')
        inventory = AnsiblePlaybookExecutor.create_inventory(cluster_nodes, private_path)
        executor = AnsiblePlaybookExecutor(playbook_file, private_path, inventory=inventory)
        result = executor.run()

        print(f"Did the playbook executed? {result.ok}")
        print(f"Ansible playbook return code: {result.ret_code}")
        print(f"Let's check how nodes executed: ")
        for node_id, status in result.hosts.items():
            print(f"    Node {node_id}: {status}")
        print(f"Let's check variables set using set_fact module: ")
        for node_id, facts in result.vars.items():
            print(f"    Node {node_id}: {facts}")
            iteration_times = { node_id, facts}
            timestamp_dir = path_extend(experiment_dir, 'timestamp', pi_logs_dir)
            timestamp_file = path_extend(timestamp_dir, str(int(time.time())), '.out')
            with open(timestamp_file) as outfile:
                yaml.dump(iteration_times, outfile, default_flow_style=False)

        instance_cost = yaml_load('vm_prices.yaml')
        instances = instance_cost.getKeys()

        node_prices = instance_cost.getValues() * iteration_times.getValues()

        return node_prices

    def terminated(self, cluster_id: str, experiment_id: str) -> bool:
        pass

    def fetch_results(self, cluster_id: str, experiment_id: str,
                      output_dir: str):
        pass


# Class Optimizer
# You must implement one method as speficied in the document
class Optimizer:
    def run(self, cluster_id: str, experiment_id: str,
            metrics: Dict[str, float]) -> bool:
        pass


# Function used to dynamic optimization
def optimize_it(cluster_id: str, experiment_id: str, vm_price_file: str,
                root_dir: str, report_time: int = 60) -> int:
    # Create experiments directory
    experiment_dir = path_extend(root_dir, experiment_id, str(int(time.time())))
    timestamp_dir = path_extend(experiment_dir, 'timestamp')
    app_results_dir = path_extend(timestamp_dir, 'app_results')
    optimizer_logs_dir = path_extend(timestamp_dir, 'optimizer_logs')
    pis_logs_dir = path_extend(timestamp_dir, 'PIs_logs')
    os.makedirs(timestamp_dir, exist_ok=True)
    os.makedirs(app_results_dir, exist_ok=True)
    os.makedirs(optimizer_logs_dir, exist_ok=True)
    os.makedirs(pis_logs_dir, exist_ok=True)

    # Print some information...
    print(f"Cluster ID: {cluster_id}, Experiment ID: {experiment_id}, "
          f"VM Price File: {vm_price_file}, APP Result Dir: {app_results_dir}, "
          f"Optimizer Logs Dir: {optimizer_logs_dir}, PI Logs Dir: {pis_logs_dir}")
    
    # Get reporter and optimizer objects
    reporter_obj = Reporter()
    optimizer_obj = Optimizer()
    # Read VM prices from vm_price_file. The result is a dictionary
    prices = yaml_load(vm_price_file)

    # You may want to print nodes init time here....
    # .....

    # Continue until application terminates
    while True:
	# Sleep for report_time seconds
        time.sleep(report_time)
        # Check if the application already terminated.
        if reporter_obj.terminated(cluster_id, experiment_id):
            # Fetch results and terminate cluster
            reporter_obj.fetch_results(cluster_id, experiment_id, app_results_dir)
            cluster_manager.stop_cluster(cluster_id)
            return 0
        else:
            # Get cost for nodes
            metrics = reporter_obj.get_metrics(
                cluster_id, experiment_id, pis_logs_dir, prices)
            # Optimize it!            
            changed = optimizer_obj.run(cluster_id, experiment_id, metrics)
            print(f"Does cluster changed? {changed}")
    # This should never be reached...    
    return 1


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
@click.option('-r', '--root-dir', default='.', show_default=False,
              type=str, required=False,
              help='Root directory where experiment directories will be created')
@click.option('-rt', '--report-time', default=60, show_default=True,
              type=int, required=False,
              help='Time to wait before calling reporter')
def optimizer_run(cluster_id: str, experiment_id: str, vm_price: str,
                  root_dir: str, report_time: int):
    return optimize_it(cluster_id, experiment_id, vm_price, root_dir, report_time)

