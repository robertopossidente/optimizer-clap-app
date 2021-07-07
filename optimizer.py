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
        print('cluster -> yaml.dump \n')
        print(yaml.dump(cluster_dict, indent=4))
        cluster_nodes = cluster_manager.get_all_cluster_nodes(cluster_id)
        print('cluster_nodes \n')
        print(cluster_nodes)
        cluster_nodes_with_type = cluster_manager.get_cluster_nodes_types(cluster_id)
        print('cluster_nodes_with_type \n')
        print(cluster_nodes_with_type)
        print('cluster_nodes_with_type -> yaml.dump \n')
        print(yaml.dump(cluster_nodes_with_type))

        nodes = node_manager.get_nodes_by_id(cluster_nodes)
 
        playbook_file = path_extend('/home/ubuntu/.clap/roles/roles/getfacts.yml')
        inventory = AnsiblePlaybookExecutor.create_inventory(nodes, private_path)
        executor = AnsiblePlaybookExecutor(playbook_file, private_path, inventory=inventory)
        result = executor.run()

        print(f"Did the playbook executed? {result.ok}")
        print(f"Ansible playbook return code: {result.ret_code}")
        print(f"Let's check how nodes executed: ")
        times={}
        for node_id, status in result.hosts.items():
            print(f"    Node {node_id}: {status}")
        print(f"Let's check variables set using set_fact module: ")
        for node_id, facts in result.vars.items():
            print(f"    Node {node_id}: {facts}")
            times[node_id] = facts['iteration_time']
        print('%s' % str(times))
        # Dump dictionary in YAML format
        print(yaml.dump(times, indent=4, sort_keys=True))

        timestamp_file = pi_logs_dir + str(int(time.time())) +  '.out'
        print('%s' % str(timestamp_file))
        with open(timestamp_file, 'a') as outfile:
            yaml.dump(times, outfile, default_flow_style=False)

        node_prices={}
        for node in nodes:
            print('---------')
            print(f"Node Id: {node.node_id}, created at {float_time_to_string(node.creation_time)}; Status: {node.status}")
            print('---------')
            # Or can be converted to a dict
            node_dict = asdict(node)
            # Printing dict in YAML format
            #print(yaml.dump(node_dict, indent=4))
            #print('**********')
            instance_flavor = node_dict['configuration']['instance']['flavor']
            node_prices[node.node_id] = float(times[node.node_id]) * float(instance_costs[instance_flavor])
            print(f"Instance Flavor: {instance_flavor}, Instance Cost: {instance_costs[instance_flavor]}, Iteration Time: {times[node.node_id]}, Node Price: {node_prices[node.node_id]}")
            print('---------')
        print('Node Prices')
        print(str(node_prices))

        return node_prices

    def terminated(self, cluster_id: str, experiment_id: str) -> bool:
        return False

    def fetch_results(self, cluster_id: str, experiment_id: str,
                      output_dir: str):
        pass


# Class Optimizer
# You must implement one method as speficied in the document
class Optimizer:
    def run(self, cluster_id: str, experiment_id: str,
            metrics: Dict[str, float]) -> bool:
        return False


# Function used to dynamic optimization
def optimize_it(cluster_id: str, experiment_id: str, vm_price_file: str,
                root_dir: str, report_time: int = 60) -> int:
    # Create experiments directory
    experiment_dir = path_extend(root_dir, experiment_id, str(int(time.time())))
    app_results_dir = path_extend(experiment_dir, 'app_results/')
    optimizer_logs_dir = path_extend(experiment_dir, 'optimizer_logs/')
    pis_logs_dir = path_extend(experiment_dir, 'PIs_logs/')
    os.makedirs(experiment_dir, exist_ok=True)
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
    print(str(prices))

    # You may want to print nodes init time here....
    # .....

    # Continue until application terminates
    #while True:
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

