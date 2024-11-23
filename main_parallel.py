import sys
import pandas as pd
import os
import time
from pathlib import Path
import yaml
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.simulator import Simulator_Plebiscito
from src.config import DebugLevel, SchedulingAlgorithm, Utility
from src.dataset_loader import init_go_, poisson_arrivals
from src.topology_nx import SpineLeafTopology as SpineLeafTopology_mps
from src.topology_nx_v1 import SpineLeafTopology as SpineLeafTopology_sps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> dict:
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
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration parameters: {', '.join(missing_keys)}")

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

def run_simulation(args):
    """
    Wrapper function to run a single simulation. This function is necessary
    because ProcessPoolExecutor can only map functions with a single argument.
    
    Args:
        args (tuple): Contains all necessary arguments for the simulation.
    """
    (rep, config, dataset_path, utility, scheduling_algorithm, singleps, with_bw) = args
    # Load dataset within the worker to avoid pickling large DataFrame
    dataset_plebi = pd.read_csv(dataset_path)
    # dataset_plebi = pd.read_csv('/home/cc/PlebyNet/traces/302_70J_50N_NFD_HN_NDJ_MPS_BW_TETRIS_FIFO_dataset.csv')
    
    
    topology = create_topology(config['LeafSpine'], singleps, with_bw)

    simulator = Simulator_Plebiscito(
        filename=rep,
        n_nodes=config['num_nodes'],
        n_jobs=config['num_jobs'],
        dataset=dataset_plebi,
        scheduling_algorithm=scheduling_algorithm,
        utility=utility,
        debug_level=DebugLevel.TRACE,
        topology=topology,
        with_bw=with_bw,
        discard_job=config['discard_job'],
        heterogeneous_nodes=config['heterogeneous_nodes'],
        fix_duration=config['fix_duration'],
        singleps=singleps,
        # enable_logging=True
    )

    logging.info(f"Starting simulation: Rep={rep}, Utility={utility.name}, "
                 f"Scheduling={scheduling_algorithm.name}, SinglePS={singleps}, WithBW={with_bw}")
    simulator.run()
    logging.info(f"Simulation completed: Rep={rep}, Utility={utility.name}, "
                 f"Scheduling={scheduling_algorithm.name}, SinglePS={singleps}, WithBW={with_bw}")
    return f"Completed: Rep={rep}, Utility={utility.name}, Scheduling={scheduling_algorithm.name}, SinglePS={singleps}, WithBW={with_bw}"

def main(rep: str):
    config_file = 'config_100_100n_100bw.yaml'
    # config_file = 'config_200_100n_25bw.yaml'
    # config_file = 'config_300_50n_100bw.yaml'
    # config_file = 'config_400_50n_25bw.yaml'
    
    
    config = load_config(config_file)

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
    dataset_full_path = CSV_FILE_PATH / CSV_FILE

    # Initialize dataset
    # Note: Instead of passing the DataFrame to each process, save it to a CSV and let each worker load it.
    # This avoids the overhead of pickling large DataFrames.
    dataset = init_go_(NUM_JOBS, CSV_FILE, rep, config['fix_duration'])
    df_dataset_full = pd.DataFrame(dataset)
    random_state = int(time.time())
    dataset_plebi = df_dataset_full.sample(n=NUM_JOBS, random_state=random_state)
    dataset_plebi = poisson_arrivals(dataset_plebi, total_time=500, total_jobs=NUM_JOBS)
    
    # Save the prepared dataset to a temporary CSV file
    temp_dataset_path = Path(f"temp_dataset_rep_{rep}.csv")
    dataset_plebi.to_csv(temp_dataset_path, index=False)
    logging.info(f"Dataset initialized and saved to {temp_dataset_path}.")

    # Simulation Parameters from config
    utils = config['utils']
    sched = config['sched']

    # Define all experiment configurations
    experiment_tasks = []
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

                # Prepare arguments for the simulation
                args = (
                    rep,
                    config,
                    temp_dataset_path,  # Pass the dataset path
                    utility,
                    scheduling_algorithm,
                    singleps,
                    with_bw
                )
                experiment_tasks.append(args)

    # Determine the number of workers (you can adjust this based on your CPU cores)
    max_workers = os.cpu_count() or 4

    # Execute simulations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_simulation, task): task for task in experiment_tasks}
        
        # Optionally, display progress
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                logging.info(result)
            except Exception as exc:
                logging.error(f"Simulation generated an exception: {exc}")

    # Clean up the temporary dataset file
    if temp_dataset_path.exists():
        os.remove(temp_dataset_path)
        logging.info(f"Temporary dataset file {temp_dataset_path} removed.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <rep>")
        sys.exit(1)
    
    rep = sys.argv[1]
    main(rep)
