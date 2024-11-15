import sys
import pandas as pd
import os
import time
from pathlib import Path
import yaml  # Ensure PyYAML is installed

from src.simulator import Simulator_Plebiscito
from src.config import ApplicationGraphType, DebugLevel, SchedulingAlgorithm, Utility
from src.dataset_builder import generate_dataset
from src.dataset_loader import init_go_, poisson_arrivals
from src.topology_nx import SpineLeafTopology
from src.network_topology import FatTreeTopology

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

    # Additionally validate nested keys for bandwidth configurations
    # bw_mode = config.get('bw_mode', None)
    # if bw_mode not in ['limited', 'infinite']:
        # raise ValueError("Invalid 'bw_mode' value. Must be 'limited' or 'infinite'.")
    for with_bw in config.get('with_bw_options', None):
        bw_required_keys = ['max_spine_capacity', 'max_leaf_capacity', 'max_node_bw', 'max_leaf_to_spine_bw']
        if with_bw:
            missing_bw = [key for key in bw_required_keys if key not in config.get('limited_bw', {})]
            if missing_bw:
                raise KeyError(f"Missing limited_bw parameters: {', '.join(missing_bw)}")
        else:
            missing_bw = [key for key in bw_required_keys if key not in config.get('infinite_bw', {})]
            if missing_bw:
                raise KeyError(f"Missing infinite_bw parameters: {', '.join(missing_bw)}")

def get_bandwidth_config(config: dict, with_bw) -> dict:
    """
    Select the bandwidth configuration based on bw_mode.

    Args:
        config (dict): The entire configuration dictionary.

    Returns:
        dict: Selected bandwidth configuration.
    """
    if with_bw:
        return config['limited_bw']
    else:
        return config['infinite_bw']

if __name__ == '__main__':
    # Ensure the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <rep>")
        sys.exit(1)

    # Load configuration
    rep = sys.argv[1]
    config_file = 'config.yaml'
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
        'fix_duration'
    ]

    # Assuming 'fat_tree' is a section of the YAML config
    fat_tree_config = config.get('fat_tree')

    # Ensure that fat_tree_config exists in the loaded config
    if not fat_tree_config:
        raise KeyError("Missing 'fat_tree' configuration parameters in the YAML file.")

    # Proceed to initialize FatTreeTopology with the loaded config
    fat_tree = FatTreeTopology(
        num_core_switches=fat_tree_config['num_core_switches'],
        num_agg_switches=fat_tree_config['num_agg_switches'],
        num_edge_switches=fat_tree_config['num_edge_switches'],
        num_hosts=fat_tree_config['num_hosts']
    )
    fat_tree.create_topology()

    # Validate configuration
    # try:
    #     validate_config(config, required_keys)
    # except (KeyError, ValueError) as e:
    #     print(f"Configuration Error: {e}")
    #     sys.exit(1)

    # Assign general variables from config
    NUM_JOBS = config['num_jobs']
    NUM_NODES = config['num_nodes']
    n_failure = config['n_failure']

    CSV_FILE_PATH = Path(__file__).parent / config['csv_file_path']
    CSV_FILE = config['csv_file']
    
    # Get the selected topology type
    topology_type = config['topology_type']

    # Set the variables based on the selected topology
    if topology_type == 'LeafSpine':
        topology_config = config['LeafSpine']
        
        # Topology-specific settings
        NUM_SPINE_SWITCHES = topology_config['num_spine_switches']
        NUM_LEAF_SWITCHES = topology_config['num_leaf_switches']
        HOST_PER_LEAF = topology_config['host_per_leaf']
        MAX_SPINE_CAPACITY = topology_config['max_spine_capacity']
        MAX_LEAF_CAPACITY = topology_config['max_leaf_capacity']
        MAX_NODE_BW = topology_config['max_node_bw']
        MAX_LEAF_TO_SPINE_BW = topology_config['max_leaf_to_spine_bw']

        # Bandwidth selection
        with_bw = config['with_bw']
        if with_bw:
            # Use limited bandwidth
            SPINE_CAPACITY = topology_config['max_spine_capacity']
            LEAF_CAPACITY = topology_config['max_leaf_capacity']
            NODE_BW = topology_config['max_node_bw']
            LEAF_TO_SPINE_BW = topology_config['max_leaf_to_spine_bw']
        else:
            # Use infinite bandwidth
            infinite_bw_config = topology_config['infinite_bw']
            SPINE_CAPACITY = infinite_bw_config['max_spine_capacity']
            LEAF_CAPACITY = infinite_bw_config['max_leaf_capacity']
            NODE_BW = infinite_bw_config['max_node_bw']
            LEAF_TO_SPINE_BW = infinite_bw_config['max_leaf_to_spine_bw']
            
        print("Topology Type:", topology_type)
        print("Number of Spine Switches:", NUM_SPINE_SWITCHES)
        print("Number of Leaf Switches:", NUM_LEAF_SWITCHES)
        print("Hosts per Leaf:", HOST_PER_LEAF)
        topology = SpineLeafTopology(
            num_spine=NUM_SPINE_SWITCHES,
            num_leaf=NUM_LEAF_SWITCHES,
            num_hosts_per_leaf=HOST_PER_LEAF,
            spine_bw=SPINE_CAPACITY,
            leaf_bw=LEAF_CAPACITY,
            link_bw_leaf_to_node=NODE_BW,
            link_bw_leaf_to_spine=LEAF_TO_SPINE_BW
        )

        # Example Usage
        adj_matrix = topology.calculate_host_to_host_adjacency_matrix()
        print("Adjacency Matrix:", adj_matrix)
        



    # Assign topology variables from config
    HETEROGENEOUS_NODES = config['heterogeneous_nodes']
    FIX_DURATION = config['fix_duration']


    # JOBS settings
    DISCARD_JOB = config['discard_job']

    # Initialize dataset
    dataset = init_go_(NUM_JOBS, CSV_FILE, rep, FIX_DURATION)
    df_dataset_full = pd.DataFrame(dataset)

    # Simulation Parameters from config
    utils = config['utils']
    sched = config['sched']
    with_bw = config['with_bw']

    for rep_ in range(1):
        # Sample jobs
        random_state = int(time.time())
        # random_state = 42
        dataset_plebi_ = df_dataset_full.sample(n=NUM_JOBS, random_state=random_state)
        dataset_plebi_ = poisson_arrivals(dataset_plebi_, total_time=500, total_jobs=NUM_JOBS)
        print(dataset_plebi_.describe())

        dec_factor = [0]
        for u in utils:
            utility = getattr(Utility, u, None)
            if utility is None:
                print(f"Warning: '{u}' is not a valid Utility member.")
                continue

            for s in sched:
                scheduling_algorithm = getattr(SchedulingAlgorithm, s, None)
                if scheduling_algorithm is None:
                    print(f"Warning: '{s}' is not a valid SchedulingAlgorithm member.")
                    continue
                print(utility)
                # for withbw_option in with_bw:
                # print(withbw_option)
                # Assign bandwidth variables based on bw_mode from config
                # bw_config = get_bandwidth_config(config, withbw_option)

                # Debugging: Print loaded configuration
                # print("Loaded Configuration:")
                # print(f"Replication: {rep}")
                # print(f"Number of Jobs: {NUM_JOBS}")
                # print(f"Discard Jobs: {DISCARD_JOB}")
                # print(f"HETEROGENEOUS NODES: {HETEROGENEOUS_NODES}")
                # print(f"Number of Nodes: {NUM_NODES}")
                # print(f"bw_config: {bw_config}")
                # print(f"Number of Spine Switches: {NUM_SPINE_SWITCHES}")
                # print(f"Number of Leaf Switches: {NUM_LEAF_SWITCHES}")
                # print(f"Hosts per Leaf: {HOST_PER_LEAF}")
                # print(f"Max Spine Capacity: {MAX_SPINE_CAPACITY}")
                # print(f"Max Leaf Capacity: {MAX_LEAF_CAPACITY}")
                # print(f"Max Node BW: {MAX_NODE_BW}")
                # print(f"Max Leaf to Spine BW: {MAX_LEAF_TO_SPINE_BW}")
                # print(f"Utilities: {config['utils']}")
                # print(f"Scheduling Algorithms: {config['sched']}")
                # print(f"With Bandwidth Options: {config['with_bw']}")
                topology = SpineLeafTopology(num_spine=NUM_SPINE_SWITCHES,
                                        num_leaf=NUM_LEAF_SWITCHES,
                                        num_hosts_per_leaf=HOST_PER_LEAF,
                                        spine_bw=MAX_SPINE_CAPACITY*100,
                                        leaf_bw=MAX_LEAF_CAPACITY*100,
                                        link_bw_leaf_to_node=MAX_NODE_BW*100,
                                        link_bw_leaf_to_spine=MAX_LEAF_TO_SPINE_BW*100)

                simulator = Simulator_Plebiscito(
                    filename=rep,
                    n_nodes=NUM_NODES,
                    n_jobs=NUM_JOBS,
                    dataset=dataset_plebi_,
                    scheduling_algorithm=scheduling_algorithm,
                    utility=utility,
                    debug_level=DebugLevel.TRACE,
                    topology=topology,
                    # enable_logging=True,
                    with_bw=with_bw,
                    discard_job=DISCARD_JOB,
                    heterogeneous_nodes=HETEROGENEOUS_NODES,
                    fix_duration=FIX_DURATION
                )
                simulator.run()

                # # Determine result filename based on bandwidth option
                # result_filename = 'BW_results.csv' if with_bw else 'results.csv'
                # simulator.save_res(result_filename, rep)
