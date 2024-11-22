import sys
import pandas as pd
import os
import time
from pathlib import Path
import yaml  # Ensure PyYAML is installed
import logging

from src.simulator import Simulator_Plebiscito
from src.config import DebugLevel, SchedulingAlgorithm, Utility
from src.dataset_loader import init_go_, poisson_arrivals
from src.topology_nx import SpineLeafTopology as SpineLeafTopology_mps
from src.topology_nx_v1 import SpineLeafTopology as SpineLeafTopology_sps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

def validate_config(config: dict, required_keys: list):
    """
    Validate that all required keys are present in the configuration.

    Args:
        config (dict): The configuration dictionary.
        required_keys (list): List of keys that must be present.

    Raises:
        KeyError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration parameters: {', '.join(missing_keys)}")

    # Validate topology configurations
    if 'LeafSpine' not in config:
        raise KeyError("Missing 'LeafSpine' topology configuration.")

    leaf_spine_config = config['LeafSpine']
    required_leaf_spine_keys = [
        'num_spine_switches',
        'num_leaf_switches',
        'host_per_leaf',
        'max_spine_capacity',
        'max_leaf_capacity',
        'max_node_bw',
        'max_leaf_to_spine_bw',
        'infinite_bw'
    ]
    missing_leaf_spine_keys = [key for key in required_leaf_spine_keys if key not in leaf_spine_config]
    if missing_leaf_spine_keys:
        raise KeyError(f"Missing LeafSpine configuration parameters: {', '.join(missing_leaf_spine_keys)}")

    # Validate nested infinite_bw configurations
    infinite_bw = leaf_spine_config.get('infinite_bw', {})
    required_bw_keys = [
        'max_spine_capacity',
        'max_leaf_capacity',
        'max_node_bw',
        'max_leaf_to_spine_bw'
    ]
    missing_bw_keys = [key for key in required_bw_keys if key not in infinite_bw]
    if missing_bw_keys:
        raise KeyError(f"Missing infinite_bw parameters in LeafSpine: {', '.join(missing_bw_keys)}")

def get_bandwidth_config(leaf_spine_config: dict, with_bw: bool) -> dict:
    """
    Select the bandwidth configuration based on with_bw flag.

    Args:
        leaf_spine_config (dict): LeafSpine topology configuration.
        with_bw (bool): Flag indicating whether to use limited bandwidth.

    Returns:
        dict: Selected bandwidth configuration.
    """
    if with_bw:
        bw_config = {
            'max_spine_capacity': leaf_spine_config['max_spine_capacity'],
            'max_leaf_capacity': leaf_spine_config['max_leaf_capacity'],
            'max_node_bw': leaf_spine_config['max_node_bw'],
            'max_leaf_to_spine_bw': leaf_spine_config['max_leaf_to_spine_bw']
        }
    else:
        infinite_bw = leaf_spine_config['infinite_bw']
        bw_config = {
            'max_spine_capacity': infinite_bw['max_spine_capacity'],
            'max_leaf_capacity': infinite_bw['max_leaf_capacity'],
            'max_node_bw': infinite_bw['max_node_bw'],
            'max_leaf_to_spine_bw': infinite_bw['max_leaf_to_spine_bw']
        }
    logging.debug(f"Bandwidth configuration selected: {'Limited' if with_bw else 'Infinite'}")
    return bw_config

def create_topology(leaf_spine_config: dict, singleps: bool, with_bw: bool):
    """
    Initialize the topology based on the configuration.

    Args:
        leaf_spine_config (dict): LeafSpine topology configuration.
        singleps (bool): Flag for single spine leaf topology.
        with_bw (bool): Flag indicating whether to apply bandwidth constraints.

    Returns:
        SpineLeafTopology: Initialized topology object.
    """
    bw_config = get_bandwidth_config(leaf_spine_config, with_bw)

    topology_cls = SpineLeafTopology_sps if singleps else SpineLeafTopology_mps

    topology = topology_cls(
        num_spine=leaf_spine_config['num_spine_switches'],
        num_leaf=leaf_spine_config['num_leaf_switches'],
        num_hosts_per_leaf=leaf_spine_config['host_per_leaf'],
        spine_bw=bw_config['max_spine_capacity'] * 100,
        leaf_bw=bw_config['max_leaf_capacity'] * 100,
        link_bw_leaf_to_node=bw_config['max_node_bw'] * 100,
        link_bw_leaf_to_spine=bw_config['max_leaf_to_spine_bw'] * 100
    )
    logging.debug(f"Topology created with {'SPS' if singleps else 'MPS'} and {'with BW' if with_bw else 'without BW'}")
    return topology

def run_simulation(rep: str, config: dict, dataset: pd.DataFrame, utility: Utility, 
                  scheduling_algorithm: SchedulingAlgorithm, singleps: bool, with_bw: bool):
    """
    Configure and run a single simulation.

    Args:
        rep (str): Replication identifier.
        config (dict): Configuration dictionary.
        dataset (pd.DataFrame): Dataset for the simulation.
        utility (Utility): Utility function.
        scheduling_algorithm (SchedulingAlgorithm): Scheduling algorithm.
        singleps (bool): Flag for single spine leaf topology.
        with_bw (bool): Flag indicating whether to apply bandwidth constraints.
    """
    topology = create_topology(config['LeafSpine'], singleps, with_bw)

    simulator = Simulator_Plebiscito(
        filename=rep,
        n_nodes=config['num_nodes'],
        n_jobs=config['num_jobs'],
        dataset=dataset,
        scheduling_algorithm=scheduling_algorithm,
        utility=utility,
        debug_level=DebugLevel.TRACE,
        topology=topology,
        with_bw=with_bw,
        discard_job=config['discard_job'],
        heterogeneous_nodes=config['heterogeneous_nodes'],
        fix_duration=config['fix_duration'],
        singleps=singleps
    )

    logging.info(f"Starting simulation: Rep={rep}, Utility={utility.name}, "
                 f"Scheduling={scheduling_algorithm.name}, SinglePS={singleps}, WithBW={with_bw}")
    simulator.run()
    logging.info(f"Simulation completed: Rep={rep}, Utility={utility.name}, "
                 f"Scheduling={scheduling_algorithm.name}, SinglePS={singleps}, WithBW={with_bw}")

def main(rep: str):
    # config_file = 'config_100_100n_100bw.yaml'
    # config_file = 'config_200_100n_25bw.yaml'
    # config_file = 'config_300_50n_100bw.yaml'
    config_file = 'config_400_50n_25bw.yaml'
    config = load_config(config_file)

    # Define all required top-level configuration keys
    required_keys = [
        'num_jobs',
        'num_nodes',
        'n_failure',
        'csv_file_path',
        'csv_file',
        'utils',
        'sched',
        'with_bw',
        'discard_job',
        'heterogeneous_nodes',
        'fix_duration',
        'topology_type',
        'LeafSpine'
    ]

    # Validate configuration
    try:
        validate_config(config, required_keys)
        logging.info("Configuration validation successful.")
    except (KeyError, ValueError) as e:
        logging.error(f"Configuration Error: {e}")
        sys.exit(1)

    # Assign general variables from config
    NUM_JOBS = config['num_jobs']
    NUM_NODES = config['num_nodes']
    n_failure = config['n_failure']

    CSV_FILE_PATH = Path(__file__).parent / config['csv_file_path']
    CSV_FILE = config['csv_file']

    # Initialize dataset
    dataset = init_go_(NUM_JOBS, CSV_FILE, rep, config['fix_duration'])
    df_dataset_full = pd.DataFrame(dataset)
    random_state = int(time.time())
    dataset_plebi = df_dataset_full.sample(n=NUM_JOBS, random_state=random_state)
    dataset_plebi = poisson_arrivals(dataset_plebi, total_time=500, total_jobs=NUM_JOBS)
    logging.info("Dataset initialized and loaded.")

    # Simulation Parameters from config
    utils = config['utils']
    sched = config['sched']

    # Iterate through experiment configurations
    for util in utils:
        utility = getattr(Utility, util, None)
        if utility is None:
            logging.warning(f"'{util}' is not a valid Utility member. Skipping.")
            continue

        for sched_alg in sched:
            scheduling_algorithm = getattr(SchedulingAlgorithm, sched_alg, None)
            if scheduling_algorithm is None:
                logging.warning(f"'{sched_alg}' is not a valid SchedulingAlgorithm member. Skipping.")
                continue

            # Define experiment scenarios
            experiment_scenarios = [
                {'singleps': False, 'with_bw': True},
                {'singleps': True, 'with_bw': True},
                {'singleps': True, 'with_bw': False}
            ]

            for scenario in experiment_scenarios:
                singleps = scenario['singleps']
                with_bw = scenario['with_bw']

                # Run the simulation
                run_simulation(
                    rep=rep,
                    config=config,
                    dataset=dataset_plebi,
                    utility=utility,
                    scheduling_algorithm=scheduling_algorithm,
                    singleps=singleps,
                    with_bw=with_bw
                )

if __name__ == '__main__':
    # Ensure the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <rep>")
        sys.exit(1)
    
    rep = sys.argv[1]
    main(rep)
