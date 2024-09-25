import copy
import csv
import datetime
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Event, Manager, JoinableQueue
import time
from matplotlib import pyplot as plt
import pandas as pd
# from src.topology import Topology
# from src.topology_tst import SpineLeafTopology
from src.topology_nx import SpineLeafTopology
pd.set_option('display.max_rows', 500)
import signal
import logging
import os
import sys
import numpy as np


# from src.network_topology import NetworkTopology
# from src.topology import topo as LogicalTopology
# from src.network_topology import  TopologyType
from src.utils import generate_gpu_types, GPUSupport
from src.node import node
from src.config import Utility, DebugLevel, SchedulingAlgorithm, ApplicationGraphType
import src.jobs_handler as job
import src.utils as utils
import src.plot as plot
from src.jobs_handler import message_data
from src.plot import generate_plots


from queue import Queue

class MyManager(SyncManager): pass

main_pid = ""
nodes_thread = []
TRACE = 5    
NUM_SPINE_SWITCHES = 5  # Number of spine switches
NUM_LEAF_SWITCHES = 10  # Number of leaf switches
NUM_NODES = 100  # Number of nodes
MAX_SPINE_CAPACITY = 500.0  # Maximum capacity for each spine switch
MAX_LEAF_CAPACITY = 300.0  # Maximum capacity for each leaf switch
MAX_NODE_BW = 100.0  # Maximum bandwidth capacity for each node

def sigterm_handler(signum, frame):
    """Handles the SIGTERM signal by performing cleanup actions and gracefully terminating all processes."""
    # Perform cleanup actions here
    # ...    
    global main_pid
    if os.getpid() == main_pid:
        print("SIGINT received. Performing cleanup...")
        for t in nodes_thread:
            t.terminate()
            t.join()    
            
        print("All processes have been gracefully teminated.")
        sys.exit(0)  # Exit gracefully    

class Simulator_Plebiscito:
    def __init__(self, 
                 filename: str, 
                 n_nodes: int, 
                 n_jobs: int, 
                 dataset = pd.DataFrame(), 
                 alpha = 1, 
                 utility = Utility.LGF, 
                 debug_level = DebugLevel.INFO, 
                 scheduling_algorithm = SchedulingAlgorithm.FIFO, 
                 decrement_factor = 1, 
                 split = True, 
                 app_type = ApplicationGraphType.LINEAR, 
                 enable_logging = False, 
                 use_net_topology = False, 
                 progress_flag = False, 
                 n_client = 0, 
                 node_bw = 0, 
                 failures = {}, 
                 logical_topology = "ring_graph", 
                 probability = 0, 
                 enable_post_allocation = False,
                 max_bw = 0,
                 with_bw = False) -> None:   
        
        if utility == Utility.FGD and split:
            print(f"FGD utility and split are not supported simultaneously. Exiting...")
            os._exit(-1)
        
        self.filename = str(filename) + "_" + utility.name + "_" + scheduling_algorithm.name + "_" + str(decrement_factor)
        if split:
            self.filename = self.filename + "_split"
        else:
            self.filename = self.filename + "_nosplit"
            
        if enable_post_allocation:
            self.filename = self.filename + "_rebid"
        else:
            self.filename = self.filename + "_norebid"
            
        self.n_nodes = n_nodes
        self.node_bw = node_bw
        self.n_jobs = n_jobs
        self.n_client = n_client
        self.enable_logging = enable_logging
        self.use_net_topology = use_net_topology
        self.progress_flag = progress_flag
        self.dataset = dataset
        self.debug_level = debug_level
        self.counter = 0
        self.alpha = alpha
        self.scheduling_algorithm = scheduling_algorithm
        self.decrement_factor = decrement_factor
        self.split = split
        self.app_type = app_type
        self.failures = failures
        self.enable_post_allocation = enable_post_allocation
        self.utility = utility
        
        
        self.job_count = {}
        self.probability = probability
        self.max_bw = max_bw
        self.tot_assigned_jobs = 0
        self.tot_allocated_cpu = 0
        self.tot_allocated_gpu = 0
        self.tot_allocated_bw = 0
        self.with_bw = with_bw
        # create a suitable network topology for multiprocessing 
        # MyManager.register('NetworkTopology', NetworkTopology)
        # MyManager.register('LogicalTopology', LogicalTopology)
        # self.physycal_network_manager = MyManager()
        # self.physycal_network_manager.start()
        # self.logical_network_manager = MyManager()
        # self.logical_network_manager.start()
        
        #Build Topolgy
        # self.t = self.logical_network_manager.LogicalTopology(func_name=logical_topology, max_bandwidth=node_bw, min_bandwidth=node_bw/2,num_clients=n_client, num_edges=n_nodes, probability=probability)
        # self.network_t = self.physycal_network_manager.NetworkTopology(n_nodes, node_bw, node_bw, group_number=4, seed=4, topology_type=TopologyType.FAT_TREE)
        # self.topology = Topology(logical_topology, max_bw, n_nodes, probability)

        self.nodes = []
        self.gpu_types = generate_gpu_types(n_nodes)
        self.topology = SpineLeafTopology(num_spine=NUM_SPINE_SWITCHES,
                                          num_leaf=NUM_LEAF_SWITCHES,
                                          num_hosts_per_leaf=10,
                                          spine_bw=MAX_SPINE_CAPACITY,
                                          leaf_bw=MAX_LEAF_CAPACITY,
                                          link_bw_leaf_to_node=MAX_NODE_BW,
                                          link_bw_leaf_to_spine=200)
        print(self.topology.adj)
        for i in range(n_nodes):
            self.nodes.append(node(id=i, 
                                   max_bw=MAX_NODE_BW,
                                #    gpu_type=self.gpu_types[i], 
                                   
                                   utility=utility, 
                                   alpha=alpha, 
                                   enable_logging=enable_logging, 
                                   logical_topology = self.topology, 
                                   tot_nodes = n_nodes, 
                                   progress_flag = progress_flag, 
                                   decrement_factor=decrement_factor,
                                   with_bw = self.with_bw))            
        # Set up the environment
        self.setup_environment()

        # self.topology.init_topology(
        #                     nodes= self.nodes,
        #                     num_spine_switches=NUM_SPINE_SWITCHES,
        #                     num_leaf_switches=NUM_LEAF_SWITCHES,
        #                     num_nodes=NUM_NODES,
        #                     max_spine_capacity=MAX_SPINE_CAPACITY,
        #                     max_leaf_capacity=MAX_LEAF_CAPACITY,
        #                     max_node_bw=MAX_NODE_BW)
        
    def get_nodes(self):
        return self.nodes

            
    def setup_environment(self):
        """
        Set up the environment for the program.

        Registers the SIGTERM signal handler, sets the main process ID, and initializes logging.
        """
        # signal.signal(signal.SIGINT, sigterm_handler)
        main_pid = os.getpid()

        # Reset logging configuration
        logging.getLogger().handlers = []
        logging.addLevelName(DebugLevel.TRACE, "TRACE")
        logging.basicConfig(filename='LOG_'+str(self.probability)+'_'+str(self.max_bw)+'.log', 
                            level=self.debug_level.value, 
                            format='%(message)s', 
                            filemode='w')

        logging.debug('Clients number: ' + str(self.n_client))
        logging.debug('Edges number: ' + str(self.n_nodes))
        logging.debug('Requests number: ' + str(self.n_jobs))
        
    # def setup_nodes(self, terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events):
    #     """
    #     Sets up the nodes for processing. Generates threads for each node and starts them.
        
    #     Args:
    #     terminate_processing_events (list): A list of events to terminate processing for each node.
    #     start_events (list): A list of events to start processing for each node.
    #     use_queue (list): A list of events to indicate if a queue is being used by a node.
    #     manager (multiprocessing.Manager): A multiprocessing manager object.
    #     return_val (list): A list of return values for each node.
    #     queues (list): A list of queues for each node.
    #     progress_bid_events (list): A list of events to indicate progress of bid processing for each node.
    #     """
    #     global nodes_thread
        
    #     for i in range(self.n_nodes):
    #         q = JoinableQueue()
    #         e = Event() 
            
    #         queues.append(q)
    #         use_queue.append(e)
            
    #         e.set()

    #     #Generate threads for each node
    #     for i in range(self.n_nodes):
    #         e = Event() 
    #         e2 = Event()
    #         e3 = Event()
    #         return_dict = manager.dict()
            
    #         self.nodes[i].set_queues(queues, use_queue)
            
    #         p = Process(target=self.nodes[i].work, args=(e, e2, e3, return_dict))
    #         nodes_thread.append(p)
    #         return_val.append(return_dict)
    #         terminate_processing_events.append(e)
    #         start_events.append(e2)
    #         e3.clear()
    #         progress_bid_events.append(e3)
            
    #         p.start()
            
    #     for e in start_events:
    #         e.wait()
    
    def collect_node_results(self, return_val, jobs: pd.DataFrame, exec_time, time_instant, save_on_file):
        """
        Collects the results from the nodes and updates the corresponding data structures.
        
        Args:
        - return_val: list of dictionaries containing the results from each node
        - jobs: list of job objects
        - exec_time: float representing the execution time of the jobs
        - time_instant: int representing the current time instant
        
        Returns:
        - float representing the utility value calculated based on the updated data structures
        """
        
        if time_instant != 0:
            for _, j in jobs.iterrows():
                self.job_count[j["job_id"]] = 0
                for v in return_val: 
                    nodeId = v["id"]
                
                    self.nodes[nodeId].bids[j["job_id"]] = v["bids"][j["job_id"]]                        
                    self.job_count[j["job_id"]] += v["counter"][j["job_id"]]

            for v in return_val: 
                nodeId = v["id"]
                self.nodes[nodeId].updated_cpu = v["updated_cpu"]
                self.nodes[nodeId].updated_gpu = v["updated_gpu"]
                self.nodes[nodeId].updated_bw = v["updated_bw"]
                self.nodes[nodeId].gpu_type = v["gpu_type"]
        
        return utils.calculate_utility(
            nodes=self.nodes, 
            num_edges=self.n_nodes, 
            jobs=jobs, 
            time_instant=time_instant, 
            filename=self.filename, 
            gpu_types=self.gpu_types, 
            save_on_file=save_on_file)    
    
    # def terminate_node_processing(self, events):
    #     global nodes_thread
        
    #     for e in events:
    #         e.set()
            
    #     # Block until all tasks are done.
    #     for nt in nodes_thread:
    #         nt.join()
            
    def clear_screen(self):
        # Function to clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_simulation_values(self, time_instant, processed_jobs, queued_jobs: pd.DataFrame, running_jobs, batch_size):
        print()
        print("Infrastructure info")
        print("Last refresh: " + str(datetime.datetime.now()))
        print(f"Number of nodes: {self.n_nodes}")
        
        for t in set(self.gpu_types):
            count = 0
            for i in self.gpu_types:
                if i == t:
                    count += 1
            print(f"Number of {t.name} GPU nodes: {count}")
        
        print()
        print("Performing simulation at time " + str(time_instant) + ".")
        print(f"# Jobs assigned: \t\t{processed_jobs}/{len(self.dataset)}")
        print(f"# Jobs currently in queue: \t{len(queued_jobs)}")
        print(f"# Jobs currently running: \t{running_jobs}")
        print(f"# Current batch size: \t\t{batch_size}")
        print()
        NODES_PER_LINE = 6
        count = 0
        print("Node GPU resource usage")
        for n in self.nodes:
            if count == NODES_PER_LINE:
                count = 0
                print()
            print("Node{0} ({1}):{2:3.0f} %CPU:{3:3.0f}%".format(
                n.id,
                n.gpu_type,
                (n.initial_gpu - n.updated_gpu) / n.initial_gpu * 100,
                (n.initial_cpu - n.updated_cpu) / n.initial_cpu * 100
            ), end=" |   ")
            count += 1
            #print(f"Node{n.id} ({n.gpu_type}):\t{(n.initial_gpu - n.updated_gpu)/n.initial_gpu*100}%   ", end=" | ")
        print()
        print()
        print("Jobs in queue stats for gpu type:")
        if len(queued_jobs) == 0:
            print("<no jobs in queue>")
        else:
            #print(queued_jobs["gpu_type"].value_counts().to_dict())
            print(queued_jobs[["gpu_type", "num_cpu", "num_gpu"]])
        print()

    def print_simulation_progress(self, time_instant, job_processed, queued_jobs, running_jobs, batch_size):
        # self.clear_screen()
        self.print_simulation_values(time_instant, job_processed, queued_jobs, running_jobs, batch_size) 
        
    def deallocate_jobs(self, progress_bid_events, queues, jobs_to_unallocate):
        if len(jobs_to_unallocate) > 0:
            
            for _, j in jobs_to_unallocate.iterrows():
                allocations = self.nodes[0].bids[j['job_id']]['auction_id'] 

                
                # Remove BW allocation
                if not float('-inf') in allocations:
                    self.topology.deallocate_ps_from_workers([allocations[0]], allocations[1:], int( self.nodes[0].bids[j['job_id']]['read_count']))
            
                data = message_data(
                            j,
                            deallocate=True,
                            split=self.split,
                            app_type=self.app_type
                        )
                for q in queues:
                    q.put(data)

            # for e in progress_bid_events:
            #     e.wait()
            #     e.clear()  '
            while not all(q.empty() for q in queues):
                for node in self.nodes:
                    node.work(0, 0)

            return True
        return False     

    def skip_deconfliction(self, jobs): # :)
        if jobs.empty:
            return True
        
        if self.split:
            node_gpu = {}
            node_cpu = {}
            largest_gpu = {}
            largest_cpu = {}
            
            for node in self.nodes:
                gpu_type = GPUSupport.get_gpu_type(node.gpu_type)
                if gpu_type not in node_gpu:
                    node_gpu[gpu_type] = 0
                    node_cpu[gpu_type] = 0
                    largest_cpu[gpu_type] = 0
                    largest_gpu[gpu_type] = 0
                    
                node_gpu[gpu_type] += node.get_avail_gpu()  # Consider caching these values if they don't change
                node_cpu[gpu_type] += node.get_avail_cpu()

                if node.get_avail_cpu() > largest_cpu[gpu_type]:
                    largest_cpu[gpu_type] = node.get_avail_cpu()
                if node.get_avail_gpu() > largest_gpu[gpu_type]:
                    largest_gpu[gpu_type] = node.get_avail_gpu()
        
        for _, row in jobs.iterrows():
            num_gpu = row['num_gpu']
            num_cpu = row['num_cpu']
            
            if self.split:
                # TODO: improve using the largest_cpu and the largest_gpu info
                gpu_type = GPUSupport.get_gpu_type(row["gpu_type"])
                for k in node_gpu:
                    if GPUSupport.can_host(k, gpu_type):
                        if node_cpu[gpu_type] >= num_cpu and node_gpu[gpu_type] >= num_gpu:
                            print(f"Job {row['job_id']} [{row['gpu_type']}] can be dispatched. Req: {row['num_cpu']} ({node_cpu[gpu_type]}) CPU. Req: {row['num_gpu']} ({node_gpu[gpu_type]}) GPU.")
                            return False
                        #else:
                        #    print(f"Job {row['job_id']} can't be dispatched. Req: {row['num_cpu']} ({node_cpu[gpu_type]}) CPU. Req: {row['num_gpu']} ({node_gpu[gpu_type]}) GPU.")
            else:
                for node in self.nodes:           
                    if GPUSupport.can_host(GPUSupport.get_gpu_type(node.gpu_type), GPUSupport.get_gpu_type(row["gpu_type"])) and node.get_avail_cpu() >= num_cpu and node.get_avail_gpu() >= num_gpu:
                        print(f"Job {row['job_id']} [{row['gpu_type']}] can be dispatched. Req: {row['num_cpu']} ({node.get_avail_cpu()}) CPU. Req: {row['num_gpu']} ({node.get_avail_gpu()}) GPU.")
                        return False
                        # dispatch.append(row)
                        # break
        return True
        # return pd.DataFrame(dispatch) if dispatch else None
        
    # def detach_node(self, nodeid):
    #     self.t.detach_node(nodeid)

    def get_node_snapshot(self):
        nodes_snapshot = {}
        for n in self.nodes:
            nodes_snapshot[n.id] = {
                "avail_cpu": n.get_avail_cpu(),
                "avail_gpu": n.get_avail_gpu()
            }
        return nodes_snapshot

    def run(self):
        # Set up nodes and related variables
        global nodes_thread
        terminate_processing_events = []
        start_events = []
        progress_bid_events = []
        use_queue = []
        manager = Manager()
        return_val = []
        queues = []
        # self.setup_nodes(terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events)
        for i in range(self.n_nodes):
            q = Queue()
            queues.append(q)

        for i in range(self.n_nodes):
            self.nodes[i].set_queues(queues)
        

        # Initialize job-related variables
        self.job_ids=[]
        jobs = pd.DataFrame()
        running_jobs = pd.DataFrame()
        processed_jobs = pd.DataFrame()

        # Collect node results
        start_time = time.time()
        self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, 0, save_on_file=True)
        
        time_instant = 1
        batch_size = 1
        jobs_to_unallocate = pd.DataFrame()
        unassigned_jobs = pd.DataFrame()
        tot_assigned_jobs = 0
        tot_allocated_cpu = 0
        tot_allocated_gpu = 0
        tot_allocated_bw = 0
        prev_job_list = pd.DataFrame()
        curr_job_list = pd.DataFrame()
        prev_running_jobs = pd.DataFrame()
        curr_running_jobs = pd.DataFrame()
        jobs_report = pd.DataFrame()
        job_allocation_time = []
        job_post_process_time = []
        done = False
        jobs_submitted = 0
        
        # for index, row in self.dataset.iterrows():
        #         print('\njob id',row['submit_time'],
        #               'pods',row['num_pod'],
        #               'BW', row['write_count'],
        #               'submit',row['submit_time'])


        while not done:
            start_time = time.time()
            print('\n--time_instant',time_instant)
            
            # Extract completed jobs
            if len(running_jobs) > 0:
                running_jobs["current_duration"] = running_jobs["current_duration"] + running_jobs["speedup"]
                prev_running_jobs = list(running_jobs["job_id"])
                
            jobs_to_unallocate, running_jobs = job.extract_completed_jobs(running_jobs, time_instant)
            # print(jobs_to_unallocate)
            
            jobs_report = pd.concat([jobs_report, jobs_to_unallocate])
            
            # Deallocate completed jobs
            if len(jobs_to_unallocate) > 0:
                self.deallocate_jobs(progress_bid_events, queues, jobs_to_unallocate)                
            
            
            if len(running_jobs) > 0:
                curr_running_jobs = list(running_jobs["job_id"])
            self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=False)
            nodes_snapshot = self.get_node_snapshot()
            
            # id = -1
            # if bool(self.failures):
            #     for i in range(len(self.failures["time"])):
            #         if time_instant == self.failures["time"][i]:
            #             id = self.failures["nodes"][i]
            #             break
            #     if id != -1:
            #         self.detach_node(id)
                    
            
            # Select jobs for the current time instant
            new_jobs = job.select_jobs(self.dataset, time_instant)

            # Add new jobs to the job queue
            if len(jobs) > 0:
                prev_job_list = list(jobs["job_id"])
                
            jobs = pd.concat([jobs, new_jobs], sort=False)
            
            # Schedule jobs
            jobs = job.schedule_jobs(jobs, self.scheduling_algorithm)
            
            if len(jobs) > 0:
                curr_job_list = list(jobs["job_id"])
            
            n_jobs = len(jobs)
            # if time_instant >1 and prev_job_list == jobs and prev_running_jobs == running_jobs:
            #     n_jobs = 0
            if sorted(prev_job_list) == sorted(curr_job_list) and sorted(prev_running_jobs) == sorted(curr_running_jobs):
                n_jobs = 0
            
            jobs_to_submit = job.create_job_batch(jobs, n_jobs)
            
            unassigned_jobs = pd.DataFrame()
            assigned_jobs = pd.DataFrame()
            
            # Dispatch jobs
            if len(jobs_to_submit) > 0: 
                # print('allocating', jobs_to_submit['job_id'], jobs_to_submit['num_pod'], jobs_to_submit['submit_time'] )
                start_id = 0

                while start_id < len(jobs_to_submit):
                    print('time_instant', time_instant, 'processed_jobs', len(processed_jobs), 'running jobs', len(running_jobs), len(jobs))

                    jobs_submitted += 1
                    subset = jobs_to_submit.iloc[start_id:start_id+batch_size]
                    row = subset.iloc[0]

                    print('** id:', row['job_id'],
                        'gpu:', row['num_gpu'],
                        'cpu:', row['num_cpu'],
                        'num_pod:', row['num_pod'],
                        'write_count:', row['write_count'],
                        'read_count:', row['read_count'])

                    # if self.enable_logging:
                    # logging.log(TRACE, '\n-------------------------------NEW JOB---------------------------------')
                    # logging.log(TRACE, subset)

                    t = time.time()

                    self.dispatch_jobs(progress_bid_events, queues, subset) 
                    time_now = 0 

                    
                    while not all(q.empty() for q in queues):
                        # if subset["job_id"].values[0] == 342 and time_now == 2:

                        # if subset["job_id"].values[0] == 20 and time_now == 2:
                        #         print('JOB CHECKER!!!!!!')

                        for node in self.nodes:
                            if int(subset['job_id'].iloc[0]) in node.bids:
                                prev_bid = copy.deepcopy(node.bids[int(subset['job_id'].iloc[0])]['auction_id'])
                            else:
                                prev_bid = []
                            while not queues[node.id].empty():
                                # if node.id ==6 and time_now == 0:
                                # # if subset["job_id"].values[0] == 234 and node.id ==0 and time_now == 1:
                                #         print(queues[node.id].qsize())
                                #         print('JOB CHECKER!!!!!!')
                                

                                rebroadcast = node.work(time_now, time_instant)

                                assert node.get_avail_cpu() >= 0
                                assert node.get_avail_cpu() <= node.initial_cpu
                                assert node.get_avail_gpu() >= 0
                                assert node.get_avail_gpu() <= node.initial_gpu

                                cur_bid = node.bids[int(subset['job_id'].iloc[0])]['auction_id']

                            if cur_bid!=prev_bid or rebroadcast:
                                node.forward_to_neighbohors()
                            
                        time_now += 1


                    print(cur_bid, node.get_avail_cpu(), node.get_avail_gpu())
                    job_allocation_time.append(time.time()-t)
                    # if self.enable_logging:
                    # logging.log(TRACE, 'All nodes completed the processing... bid processing time:' + str(time_now) +
                                # ' jobs allocated:' + str(tot_assigned_jobs))
                    exec_time = time.time() - start_time
                
                    t = time.time()

                    # Collect node results
                    a_jobs, u_jobs = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
                    job_post_process_time.append(time.time() - t)
                    assigned_jobs = pd.concat([assigned_jobs, pd.DataFrame(a_jobs)])
                    unassigned_jobs = pd.concat([unassigned_jobs, pd.DataFrame(u_jobs)])

                    # Deallocate unassigned jobs
                    self.deallocate_jobs(progress_bid_events, queues, pd.DataFrame(u_jobs))
                    self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=False)
                    
                    #subtract network resources
                    if a_jobs:


                        a_jobs_id = a_jobs[0]['job_id']
                        allocations = node.bids[a_jobs_id]['auction_id'] 
                        seen = set()
                        allocations_ids = []

                        for allocation in allocations:
                            if allocation not in seen:
                                seen.add(allocation)
                                allocations_ids.append(allocation)

                        break_outer_loop = False
                        allocated_bw = False
                        # tmp_topo = copy.deepcopy(self.topology.bandwidth_matrix_updated)
                        tmp_topo = copy.deepcopy(self.topology.adj)
                        
                        
                        # if a_jobs_id == 1022:
                        #     print('JOB CHECKER!!!!!!')
                            
                        # print('before')
                        # print(self.topology.get_total_percentage_bw_used())
                        with_bw = False
                        

                        if with_bw:
                            for allocation in list(allocations_ids)[1:]:
                                for _ in range(allocations.count(allocation)):
                                    
                                    allocated_bw = self.topology.allocate_ps_to_workers(allocations[0], allocation, int(subset['read_count'].iloc[0]), tmp_topo)
                                    if allocated_bw == False:
                                        # self.topology.restore_updated_topo(prev_topo)
                                        
                                        # if np.any(prev_topo != self.topology.bandwidth_matrix_updated):
                                        #     print('Rolling back the topology')
                                        # else:
                                        print(self.topology.get_total_percentage_bw_used())
                                        self.deallocate_jobs(progress_bid_events, queues, pd.DataFrame(a_jobs))
                                        break_outer_loop = True
                                        logging.log(TRACE, 'Bandwidth allocation failed!!!!!!!!!')
                                        break
                                if break_outer_loop:
                                    break
                        else:
                            print('Allocating BW!')
                            print(allocations)
                            allocated_bw = self.topology.allocate_ps_to_workers([allocations[0]], allocations[1:], int(subset['read_count'].iloc[0]))
                            if not allocated_bw:
                                print('ERROR, insufficient BW')
                                self.deallocate_jobs(progress_bid_events, queues, pd.DataFrame(a_jobs))
                                
                        

                        if len(allocations_ids)>0 and allocated_bw == True:
                            tot_allocated_bw += int(subset['read_count'].iloc[0])
                            # self.topology.restore_updated_topo(tmp_topo)
                            logging.log(TRACE, '\n-----Allocated JOB!!!!!!!!!!')
                            logging.log(TRACE, subset)
                            
                            
                            
                        
                        if len(allocations_ids)==1 or allocated_bw == True:
                            tot_assigned_jobs +=1
                            tot_allocated_gpu += int(subset['num_gpu'].iloc[0]) * int(subset['num_pod'].iloc[0])
                            tot_allocated_cpu += int(subset['num_cpu'].iloc[0]) * int(subset['num_pod'].iloc[0])


                    # for n in self.nodes:
                    #     if subset["job_id"].values[0] in n.bids:
                    #         if n.id in n.bids[subset['job_id'].values[0]]['auction_id']:
                    #             won_inst = n.bids[subset['job_id'].values[0]]['auction_id'].count(n.id)
                    #             allocated_gpu = won_inst * n.bids[subset['job_id'].values[0]]['NN_gpu'] 
                    #             allocated_cpu = won_inst * n.bids[subset['job_id'].values[0]]['NN_cpu'] 
                    #             # print(f"Node {n.id} won {won_inst} instances of job {subset['job_id'].values[0]} with {allocated_cpu} CPUs and {allocated_gpu} GPUs")
                    #             previous_cpu = nodes_snapshot[n.id]['avail_cpu']
                    #             previous_gpu = nodes_snapshot[n.id]['avail_gpu']
                    #             # print(f"Previous CPU: {previous_cpu} - Previous GPU: {previous_gpu}")
                    #             assert previous_cpu - allocated_cpu == n.get_avail_cpu(), (
                    #                 f"1Assertion failed: previous_cpu ({previous_cpu}) - allocated_cpu ({allocated_cpu}) "
                    #                 f"!= n.get_avail_cpu() ({n.get_avail_cpu()})"
                    #             )
                    #             assert previous_gpu - allocated_gpu == n.get_avail_gpu(), (
                    #                 f"1Assertion failed: previous_Gpu ({previous_gpu}) - allocated_Gpu ({allocated_gpu}) "
                    #                 f"!= n.get_avail_gpu() ({n.get_avail_gpu()})"
                    #             )
                    #         else:
                    #             # print(f"Node {n.id} didn't win any instance of job {subset['job_id'].values[0]}")
                    #             previous_cpu = nodes_snapshot[n.id]['avail_cpu']
                    #             previous_gpu = nodes_snapshot[n.id]['avail_gpu']
                    #             # print(f"Previous CPU: {previous_cpu} - Previous GPU: {previous_gpu}")
                    #             assert previous_cpu == n.get_avail_cpu(), (
                    #                 f"2Assertion failed: previous_cpu ({previous_cpu}) != n.get_avail_cpu() ({n.get_avail_cpu()})"
                    #             )
                    #             assert previous_gpu == n.get_avail_gpu(), (
                    #                 f"2Assertion failed: previous_Gpu ({previous_gpu}) != n.get_avail_gpu() ({n.get_avail_gpu()})"
                    #             )
                        
                    start_id += batch_size


                    
            # Assign start time to assigned jobs
            assigned_jobs = job.assign_job_start_time(assigned_jobs, time_instant)
            
            # Add unassigned jobs to the job queue
            jobs = pd.concat([jobs, unassigned_jobs], sort=False)  
            running_jobs = pd.concat([running_jobs, assigned_jobs], sort=False)
            processed_jobs = pd.concat([processed_jobs,assigned_jobs], sort=False)
            
            unassigned_jobs = pd.DataFrame()
            assigned_jobs = pd.DataFrame()

            # if self.enable_post_allocation:
            #     if time_instant%50 == 0:
            #         low_speedup_threshold = 1
            #         high_speedup_threshold = 1.3
                                
            #         jobs_to_reallocate, running_jobs = job.extract_rebid_job(running_jobs, low_thre=low_speedup_threshold, high_thre=high_speedup_threshold, duration_therehold=250)
                                
            #         if len(jobs_to_reallocate) > 0: 
            #             start_id = 0
            #             while start_id < len(jobs_to_reallocate):
            #                 subset = jobs_to_reallocate.iloc[start_id:start_id+batch_size]
            #                 # self.deallocate_jobs(progress_bid_events, queues, subset)
            #                 # print(f"Job deallocated {float(subset['speedup'])}")
            #                 self.dispatch_jobs(progress_bid_events, queues, subset, check_speedup=True, low_th=low_speedup_threshold, high_th=high_speedup_threshold) 
                            
            #                 a_jobs, u_jobs = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
            #                 assigned_jobs = pd.concat([assigned_jobs, pd.DataFrame(a_jobs)])
            #                 unassigned_jobs = pd.concat([unassigned_jobs, pd.DataFrame(u_jobs)])
            #                 # print(f"Job dispatched {float(pd.DataFrame(a_jobs)['speedup'])}")
            #                 start_id += batch_size
                            
            # append unassigned jobs !!!!!!!
            # jobs = pd.concat([jobs, unassigned_jobs], sort=False)  
            running_jobs = pd.concat([running_jobs, assigned_jobs], sort=False)
            
            self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=True)
            
            # self.print_simulation_progress(time_instant, len(processed_jobs), jobs, tot_assigned_jobs, batch_size)
            time_instant += 1

            # Check if all jobs have been processed
            if len(processed_jobs) == len(self.dataset) and len(running_jobs) == 0 and len(jobs) == 0: # add to include also the final deallocation
            # if len(processed_jobs) == len(self.dataset) and len(jobs) == 0: # add to include also the final deallocation
                print('!!!last allocated', time_instant, time_now, jobs_submitted, self.n_jobs)
                job.extract_allocated_jobs(processed_jobs, self.filename + "_allocations.csv")

                done=True
                break

            
            self.topology.plot_node_available_bandwidth()  # Call the plot method for spine utilization
            # self.topology.plot_congestion_metric()  # Call the plot method for spine utilization
            self.topology.plot_bandwidth_utilization()  # Call the plot method for spine utilization
            # self.topology.plot_utilization_metric()  # Call the plot method for spine utilization
            self.generate_plots_resources()
            
            # if jobs_submitted == self.n_jobs:
            # # if len(assigned_jobs) + len(unassigned_jobs) == self.n_jobs:
            #     done = True
        
        # Collect final node results
        # self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant+1, save_on_file=True)
        
        # self.print_simulation_progress(time_instant, len(processed_jobs), jobs, len(running_jobs), batch_size)
        
        # Terminate node processing
        # self.terminate_node_processing(terminate_processing_events)

        # Save processed jobs to CSV
        jobs_report.to_csv(self.filename + "_jobs_report.csv")
        self.tot_assigned_jobs = tot_assigned_jobs
        self.tot_allocated_cpu = tot_allocated_cpu
        self.tot_allocated_gpu = tot_allocated_gpu
        self.tot_allocated_bw = tot_allocated_bw
        
        
        
        # adjacency_matrix_available_bw = self.topology.calculate_host_to_host_adjacency_matrix()
        # self.topology.print_adjacency_matrix(adjacency_matrix_available_bw)


        # Plot results
        # if self.use_net_topology:
        #     self.network_t.dump_to_file(self.filename, self.alpha)
    import matplotlib.pyplot as plt
    import numpy as np

    def generate_plots_resources(self):
        # Data collection
        gpus_initial = []
        gpus_updated = []
        cpus_initial = []
        cpus_updated = []
        
        for n in self.nodes:
            gpus_initial.append(n.initial_gpu)
            gpus_updated.append(n.updated_gpu)
            cpus_initial.append(n.initial_cpu)  # Assuming initial_cpu is available
            cpus_updated.append(n.updated_cpu)  # Assuming updated_cpu is available
        
        # Calculate percentage of used resources for GPUs and CPUs
        gpu_used_percent = 100 - np.array(gpus_updated) / np.array(gpus_initial) * 100
        cpu_used_percent = 100 - np.array(cpus_updated) / np.array(cpus_initial) * 100

        # Labels for the nodes
        labels = [f'{i}' for i in range(len(self.nodes))]
        x = np.arange(len(labels))  # Label locations

        # First plot for GPU usage
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(x, gpu_used_percent, width=0.4, label='GPU Usage %', color='skyblue')
        ax1.set_xlabel('Nodes')
        ax1.set_ylabel('GPU Usage (%)')
        ax1.set_title('GPU Usage per Node')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        plt.tight_layout()
        
        # Display the first plot

        plt.savefig('gpu.png')

        # Second plot for CPU usage
        fig, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(x, cpu_used_percent, width=0.4, label='CPU Usage %', color='lightcoral')
        ax2.set_xlabel('Nodes')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_title('CPU Usage per Node')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        plt.tight_layout()
        

        plt.savefig('cpu.png')
        plt.close()

# Assuming you are running this function in a context where 'self.nodes' is available
# self.generate_plots_resources
        
    def rebid(self, progress_bid_events, return_val, queues, running_jobs, time_instant, batch_size, unassigned_jobs, assigned_jobs, exec_time):
        low_speedup_threshold = 1
        high_speedup_threshold = 1.2
                    
        jobs_to_reallocate, running_jobs = job.extract_rebid_job(running_jobs, low_thre=low_speedup_threshold, high_thre=high_speedup_threshold, duration_therehold=500)
                    
        if len(jobs_to_reallocate) > 0: 
            start_id = 0
            while start_id < len(jobs_to_reallocate):
                subset = jobs_to_reallocate.iloc[start_id:start_id+batch_size]
                # self.deallocate_jobs(progress_bid_events, queues, subset)
                print("Job deallocated")
                self.dispatch_jobs(progress_bid_events, queues, subset, check_speedup=True, low_th=low_speedup_threshold, high_th=high_speedup_threshold) 
                print("Job dispatched")
                a_jobs, u_jobs = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
                assigned_jobs = pd.concat([assigned_jobs, pd.DataFrame(a_jobs)])
                unassigned_jobs = pd.concat([unassigned_jobs, pd.DataFrame(u_jobs)])
                start_id += batch_size
        return running_jobs,unassigned_jobs,assigned_jobs

        #plot.plot_all(self.n_nodes, self.filename, self.job_count, "plot")

    def dispatch_jobs(self, progress_bid_events, queues, subset, check_speedup=False, low_th=1, high_th=1.2):
        job.dispatch_job(subset, queues, self.use_net_topology, self.split, check_speedup=check_speedup, low_th=low_th, high_th=high_th)

        # for e in progress_bid_events:
        #     e.wait()
        #     e.clear()

    def save_res(self, file_path, rep):
        
        msg_count = 0
        for node in self.nodes:
            msg_count += node.count_msgs
            # print(msg_count)
        


        init_cpu = self.nodes[0].initial_cpu * self.n_nodes
        allocated_cpu = self.tot_allocated_cpu
        init_gpu = self.nodes[0].initial_gpu * self.n_nodes
        allocated_gpu = self.tot_allocated_gpu

        # used_bw = self.topology.calculate_occupied_bandwidth()
        # self.topology.plot_bandwidth_matrices(self.probability, self.max_bw)
        # self.topology.plot_occupied_bandwidth()

        data_dict = {
            'utility': [self.utility],
            'rep':[rep],
            'num_nodes': [self.n_nodes],
            'link_prob': [self.probability],
            'link_bw' : [self.max_bw],
            # 'tot_init_bw': [self.topology.get_total_initial_bw()],
            # 'tot_updated_bw': [self.topology.get_total_remaining_bw()],
            # 'tot_allocated_bw': [self.topology.get_total_allocated_bw()],
            # 'tot_percentage_used_bw': [self.topology.get_total_percentage_bw_used()],
            # 'link_overhead_avg':[self.topology.calculate_average_link_utilization()],
            'tot_cpu': [init_cpu],
            'allocated_cpu': [allocated_cpu],
            'cpu':  [(100 - ((init_cpu - allocated_cpu) / init_cpu) * 100)],
            'tot_gpu': [init_gpu],
            'allocated_gpu': [allocated_gpu],
            'gpu': [(100 - ((init_gpu - allocated_gpu) / init_gpu) * 100)],
            'allocated_jobs': [self.tot_assigned_jobs],
            'rejected_jobs': [self.n_jobs - self.tot_assigned_jobs]
        }

        file_exists = os.path.isfile(file_path)
        
        # Open the CSV file for appending
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
            
            # Write the header only if the file does not exist
            if not file_exists:
                writer.writeheader()
            
            # Write the rows
            rows = zip(*data_dict.values())
            for row in rows:
                writer.writerow(dict(zip(data_dict.keys(), row)))