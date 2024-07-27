import copy
import random
import pandas as pd
import sys
import pandas as pd
import os
import time
# import logging
import random
from pathlib import Path
from os import path
import numpy as np

from src.simulator import Simulator_Plebiscito
from src.config import ApplicationGraphType, DebugLevel, SchedulingAlgorithm, Utility
from src.dataset_builder import generate_dataset
# from kubernetes.kubernetes_scheduler import KubernetesScheduler

from src.dataset_loader import init_go_
# from Alibaba.simulator import Simulator
# from Alibaba.utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT
# from Alibaba.simulator import Simulator
# from Alibaba.utils import print_fn, ALLOC_POLICY_DICT, PREEMPT_POLICY_DICT


if __name__ == '__main__':
    NUM_JOBS = 35 #args.num_jobs
    NUM_NODES = 100
    n_failure = 0
    
    # # ------ START FROM ALIBABA -------
    
    DATE = "%02d%02d" % (time.localtime().tm_mon, time.localtime().tm_mday)

    # INPUT TRACE FILE
    CSV_FILE_PATH = Path(__file__).parent / 'traces/pai/'
    DESCRIBE_FILE = None
    # CSV_FILE = 'df_dataset.csv'
    CSV_FILE = 'pai_job_no_estimate_100K.csv'
    
    # rep = sys.argv[1]
    rep = 0
    
    ARRIVAL_RATE =0 # args.arrival_rate
    NUM_GPUS = 0 #args.num_gpus
    REPEAT =1 # args.repeat
    SORT_NODE_POLICY = 3
    MAX_TIME = int(1e9)
    VERBOSE = 0
    # LOG_LEVEL = logging.WARNING
    NUM_CPUS = round(23.22 * NUM_GPUS)  # 23.22 * num_gpus 156576/6742
    HETERO = True  # heterogeneous cluster
    PATTERN = 0  # Cluster capacity varying pattern
    GPU_TYPE_MATCHING = 1 # GPU type perfect match
    EXPORT_JOB_STATS = True
    EXPORT_CLUSTER_UTIL = True
    RANDOM_SEED = 42
    NUM_SPARE_NODE = 0
    SORT_BY_JCT = True

    # Logging in directory
    LOG_DIR = Path(__file__).parent / 'logs'

    comments = '%dg_%dn_h%d_%dp_%dsn_%dgt-%dar-%dj-%dx-%dr' % (NUM_GPUS, NUM_NODES, HETERO, PATTERN, SORT_NODE_POLICY, GPU_TYPE_MATCHING, ARRIVAL_RATE, NUM_JOBS, REPEAT, RANDOM_SEED)

    log_time = int(time.time() % 100000)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # log_file = LOG_DIR / ("%s-%s-%s-%s.log" % (DATE, CSV_FILE, log_time, comments))
    # logging.basicConfig(level=LOG_LEVEL, format="%(message)s", filename=log_file, filemode='a')
    # describe_file = CSV_FILE_PATH / DESCRIBE_FILE if DESCRIBE_FILE is not None else None
    
    # ------ END FROM ALIBABA -------
    # generate common dataset and adjust it for plebi
    dataset = init_go_(NUM_JOBS, CSV_FILE, rep)
    # dataset = sorted(dataset, key=lambda x: x['submit_time'])


    # df = pd.read_csv('/home/andrea/Desktop/PlebiscitoN/traces/pai/df_dataset.csv')

    # Convert the DataFrame to a list of dictionaries
    # dataset = df.to_dict(orient='records')
    # dataset = dataset[:5]
    # df = pd.DataFrame(dataset)

    # # Write to CSV file
    # df.to_csv('static_dataset.csv', index=False)



    
    dataset_plebi = pd.DataFrame(dataset[:NUM_JOBS])
    dataset_plebi.to_csv('MISC.csv')
    # dataset_plebi = dataset_plebi.sort_values(by=['num_pod', 'job_id'])
    # if (dataset_plebi['num_gpu']*dataset_plebi['num_pod']).sum()/100 > 8 * NUM_NODES:
    print((dataset_plebi['num_gpu']*dataset_plebi['num_pod']).sum()/100, 8 * NUM_NODES)
    print((dataset_plebi['num_cpu']*dataset_plebi['num_pod']).sum()/100, 96 * NUM_NODES)
    print(dataset_plebi['num_pod'].sum(), NUM_JOBS)
    
    # dataset = generate_dataset(entries_num=NUM_JOBS)
    # failures = generate_node_failures(n_nodes, n_failure, NUM_JOBS)
    
    # # ------ START ALIBABA SIMULATION -------
    
    # for alloc_policy in [0, 1, 2, 4, 8]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py)
    # for alloc_policy in [0, 8, 16]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py) 16exec time
    # # for alloc_policy in [16]:  # 0SDF, 1SJU, 2SJG, 4SJGG, 8FIFO (see utils.py)
    #     # for preempt_policy in [2]:  # 2LGF
    #     preempt_policy =2
    #     for sorting_policy in [1, 2, 3]:  
    #     # for sorting_policy in [3]:  
    #         print('INIT,', str(alloc_policy),', ', str(sorting_policy))

    #         key = (alloc_policy, preempt_policy)
    #         print_key = "(%-4s,%4s)" % (ALLOC_POLICY_DICT.get(key[0]), PREEMPT_POLICY_DICT.get(key[1]))

    #         # running
    #         start_time = time.time()
    #         print_fn("\n###### %s ######" % print_key)

    #         simulator = Simulator(
    #             csv_file=CSV_FILE_PATH / CSV_FILE,
    #             alloc_policy=alloc_policy,
    #             preempt_policy=preempt_policy,
    #             sort_node_policy=sorting_policy,
    #             num_nodes=NUM_NODES,
    #             random_seed=RANDOM_SEED,
    #             max_time=MAX_TIME,
    #             num_spare_node=NUM_SPARE_NODE,
    #             pattern=PATTERN,
    #             hetero=HETERO,
    #             num_gpus=NUM_GPUS,
    #             num_cpus=NUM_CPUS,
    #             describe_file=describe_file,
    #             log_file=log_file,
    #             export_job_stats=EXPORT_JOB_STATS,
    #             export_cluster_util=EXPORT_CLUSTER_UTIL,
    #             arrival_rate=ARRIVAL_RATE,
    #             num_jobs_limit=NUM_JOBS,
    #             gpu_type_matching=GPU_TYPE_MATCHING,
    #             verbose=VERBOSE,
    #             dataset=dataset,
    #             repetition=rep)
    #         results = simulator.simulator_go(repeat=REPEAT)
    #         print('done,', str(alloc_policy),', ', str(sorting_policy))
            
    # # ------ END ALIBABA SIMULATION -------
    
    # ------ START PLEBISCITO SIMULATION -------
    

    # utils = ['SPEEDUP', 'SPEEDUPV2', "LGF", "UTIL"]  
    # utils = ["UTIL", "SGF"]
    # sched = ['FIFO', 'SDF'] 
    utils = ['SGF', 'UTIL', 'LGF']

    sched = ['FIFO'] 
    # utils = ['SGF']

    split = [False]
    rebid = [False]
    # rebid = [True]
    # dec_factor = [0, .25, .5, .75, 1]
    dec_factor = [0.5]
    probability = [0.5 + i * 0.1 for i in range(int((1.0 - 0.5) / 0.1) + 1)]
    # probability = [1]
    # bw = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
    bw = [50000]

    for u in utils:
        utility = getattr(Utility, u, None)
        if utility is None:
            print(f"Warning: '{u}' is not a valid Utility member.")
            continue
        
        dec_factor = [0]
        # if u == "LGF":
        #     dec_factor = [0, 1]

        for s in sched:
            scheduling_algorithm = getattr(SchedulingAlgorithm, s, None)
            if scheduling_algorithm is None:
                print(f"Warning: '{s}' is not a valid SchedulingAlgorithm member.")
                continue

            for sp in split:
                for rb in rebid:
                    for b in bw:
                        for prob in probability:
                            simulator = Simulator_Plebiscito(filename=rep,
                                                n_nodes=NUM_NODES,
                                                n_jobs=NUM_JOBS,
                                                dataset=dataset_plebi,
                                                # failures=failures,
                                                # logical_topology="ring_graph",
                                                # logical_topology="compute_ring_graph",
                                                logical_topology="compute_probabilistic_graph",
                                                # logical_topology="probability_graph",
                                                scheduling_algorithm=scheduling_algorithm,
                                                utility=utility,
                                                debug_level=DebugLevel.TRACE,
                                                # enable_logging=True,
                                                split=sp,
                                                enable_post_allocation=rb,
                                                # decrement_factor=dc,
                                                # probability=1,
                                                probability=prob,
                                                max_bw=b
                                                )
                            simulator.run()
                            simulator.save_res('results.csv', rep)
    
    # nodes = simulator1.get_nodes()
    # adj = simulator1.get_adjacency_matrix()
    
    # simulator_kubernetes = KubernetesScheduler(nodes, dataset, "kubernetes", ApplicationGraphType.LINEAR, True, adj, failures)
    

    
    # simulator_kubernetes.run()
    
    # ------ END PLEBISCITO SIMULATION -------
    
    
    
    
    
    
    









