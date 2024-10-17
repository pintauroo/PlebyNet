import copy
import csv
import datetime
from multiprocessing.managers import SyncManager
import time
from matplotlib import pyplot as plt
import pandas as pd
from src.topology_nx import SpineLeafTopology
import logging
import os
import numpy as np
from queue import Queue, Empty  # Ensure you import Empty
import copy
from src.utils import generate_gpu_types, GPUSupport
from src.node import node
from src.config import Utility, DebugLevel, SchedulingAlgorithm
import src.jobs_handler as job
import src.utils as utils
import src.plot as plot
from src.jobs_handler import message_data
import traceback
from queue import Queue

TRACE = 5    

class MyManager(SyncManager): pass

main_pid = ""
nodes_thread = []


# def sigterm_handler(signum, frame):
#     """Handles the SIGTERM signal by performing cleanup actions and gracefully terminating all processes."""
#     # Perform cleanup actions here
#     # ...    
#     global main_pid
#     if os.getpid() == main_pid:
#         print("SIGINT received. Performing cleanup...")
#         for t in nodes_thread:
#             t.terminate()
#             t.join()    
            
#         print("All processes have been gracefully teminated.")
#         sys.exit(0)  # Exit gracefully    



class Simulator_Plebiscito:
    def __init__(self, 
                 filename: str, 
                 n_nodes: int, 
                 n_jobs: int, 
                 dataset = pd.DataFrame(), 
                 scheduling_algorithm = SchedulingAlgorithm.FIFO, 
                #  alpha = 1, 
                 utility = Utility.LGF, 
                 debug_level = DebugLevel.INFO, 
                 with_bw = False,
                 max_spine_capacity=300,
                 max_leaf_capacity=100,
                 max_node_bw=20,
                 max_leaf_to_spine_bw=100,
                 num_spine_switches=10,
                 num_leaf_switches=10,
                 num_hosts_per_leaf=10,
                 decrement_factor = 1, 
                 split = False, 
                 enable_logging = False, 
                 progress_flag = False, 
                 discard_job = False,
                 heterogeneous_nodes = False
                 ) -> None:   
        
        if utility == Utility.FGD and split:
            print(f"FGD utility and split are not supported simultaneously. Exiting...")
            os._exit(-1)
        
        logging.getLogger().handlers = []
        logging.addLevelName(DebugLevel.TRACE, "TRACE")
        logging.basicConfig(filename='LOG.log', 
                            level=debug_level.value, 
                            format='%(message)s', 
                            filemode='w')
        if with_bw:
            self.filename = "BW_" + str(filename) + "_" + utility.name + "_" + scheduling_algorithm.name
        else:
            self.filename = str(filename) + "_" + utility.name + "_" + scheduling_algorithm.name
            
        self.n_nodes = n_nodes
        self.n_jobs = n_jobs
        self.enable_logging = enable_logging
        self.progress_flag = progress_flag
        self.dataset = dataset
        self.debug_level = debug_level
        self.counter = 0
        # self.alpha = alpha
        self.scheduling_algorithm = scheduling_algorithm
        self.utility = utility
        self.discard_job = discard_job
        
        self.job_count = {}
        self.tot_assigned_jobs = 0
        self.tot_allocated_cpu = 0  
        self.tot_allocated_gpu = 0
        self.tot_allocated_bw = 0
        self.with_bw = with_bw
        self.heterogeneous_nodes = heterogeneous_nodes


        # Generate TOPOLOGY
        self.topology = SpineLeafTopology(num_spine=num_spine_switches,
                                          num_leaf=num_leaf_switches,
                                          num_hosts_per_leaf=num_hosts_per_leaf,
                                          spine_bw=max_spine_capacity,
                                          leaf_bw=max_leaf_capacity,
                                          link_bw_leaf_to_node=max_node_bw,
                                          link_bw_leaf_to_spine=max_leaf_to_spine_bw)

        # Generate Nodes
        self.nodes = []
        if self.heterogeneous_nodes:
            self.gpu_types = ['MISC'] * self.n_nodes
        else:
            self.gpu_types = generate_gpu_types(n_nodes)
        # print(self.gpu_types)
        for i in range(n_nodes):
            self.nodes.append(node(id=i, 
                                   max_bw=max_node_bw,
                                #    initial_cpu=int(i_cpu[i]),
                                #    initial_gpu=int(i_gpu[i]),
                                   initial_gpu=None,
                                   initial_cpu=None,
                                   gpu_type=self.gpu_types[i], 
                                   utility=utility, 
                                   alpha=1, 
                                   enable_logging=enable_logging, 
                                   logical_topology = self.topology, 
                                   tot_nodes = n_nodes, 
                                   progress_flag = progress_flag, 
                                   decrement_factor=decrement_factor,
                                   with_bw = self.with_bw))    


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
            gpu_types=None, 
            # gpu_types=self.gpu_types, 
            save_on_file=save_on_file)    
    
            
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
        
    def deallocate_jobs(self, progress_bid_events, queues, jobs_to_unallocate, failure = False):
        if len(jobs_to_unallocate) > 0:
            
            for _, j in jobs_to_unallocate.iterrows():
                for i, n in enumerate(self.nodes):

                    if j['job_id'] in n.bids and n.id in n.bids[j['job_id']]['auction_id']:

                        allocations = self.nodes[0].bids[j['job_id']]['auction_id'] 

                        
                        # Remove BW allocation
                        if not float('-inf') in allocations:
                            # self.topology.deallocate_ps_from_workers([allocations[0]], allocations[1:], int( self.nodes[0].bids[j['job_id']]['read_count']))
                            self.topology.deallocate_ps_from_workers(allocations, int(self.nodes[0].bids[j['job_id']]['read_count']))
                    
                        data = message_data(
                                    j,
                                    0,
                                    0,
                                    failure=failure,
                                    deallocate=True,
                                    # split=self.split,
                                    # app_type=self.app_type
                                )
                        # for q in queues:
                        queues[i].put(data)

            # for e in progress_bid_events:
            #     e.wait()
            #     e.clear()  '
            while not all(q.empty() for q in queues):
                for node in self.nodes:
                    node.work(0, 0)

            return True
        return False     



    def get_node_snapshot(self):
        nodes_snapshot = {}
        for n in self.nodes:
            nodes_snapshot[n.id] = {
                "avail_cpu": n.get_avail_cpu(),
                "avail_gpu": n.get_avail_gpu()
            }
        return nodes_snapshot

    def run(self):

        progress_bid_events = []
        return_val = []
        queues = []

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
        
        time_instant = 0
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
        job_speedup = {}
        unassigned_ids = []
        completed_jobs = 0
        final_allocations={
            "t_gpu": 0, # Dataset tot gpu
            "t_cpu": 0,
            "gpu_discarded": 0, # Dataset tot gpu
            "cpu_discarded": 0,
            "gpu": 0, # Allocated tot gpu
            "cpu": 0,
            "allocated": 0,
            "first_unassigned": 0, # How many jobs before the first failed allocation
            "first_unassigned_gpu": 0,
            "first_unassigned_cpu": 0,
            "tot_unassigned": 0, 
            "discarded_jobs":0,
            "utility": self.utility,
            "jct": 0
        }
        
        # all_jobs_ids = self.dataset[['job_id', 'submit_time']].to_dict(orient='records')
        # print(all_jobs_ids)

        all_jobs_ids = self.dataset['job_id'].tolist()
        # print(all_jobs_ids)
        all_new_jobs = []

         
        while not done and time_instant<10000000:
            start_time = time.time()
            if 'job_id' in jobs:
                queuing_job_ids = jobs['job_id'].tolist()
                assert all(job_id_ in all_jobs_ids for job_id_ in queuing_job_ids), f"Not all job_ids are present in all_jobs_ids"

            else:
                queuing_job_ids = []


            jobs_to_unallocate, running_jobs = job.extract_completed_jobs(running_jobs, time_instant)
            sim_stats = (
                f"[SIM] {'time instant:':<5} {time_instant:<5} | "
                f"{'running jobs:':<15} {len(running_jobs):>5} | "
                f"{'completed jobs:':<15} {completed_jobs:>5} | "
                f"{'discarded jobs:':<15} {final_allocations['discarded_jobs']:>5} | "
                f"{'queuing jobs:':<15} {len(jobs):>5} | "
                f"{queuing_job_ids if len(queuing_job_ids) else 'None'}"
            )
            
            print()
            print(sim_stats)

            jobs_report = pd.concat([jobs_report, jobs_to_unallocate])
            
            # Deallocate completed jobs
            if len(jobs_to_unallocate) > 0:
                print('\n[SIM] Deallocating completed jobs!', list(jobs_to_unallocate['job_id']))
                print(sim_stats)
                self.deallocate_jobs(progress_bid_events, queues, jobs_to_unallocate)                
                curr_running_jobs = list(running_jobs["job_id"])
                completed_jobs += len(jobs_to_unallocate)


            if len(running_jobs) > 0:
                prev_running_jobs = list(running_jobs["job_id"])

                for index, rj in running_jobs.iterrows():
                    job_id = rj['job_id']    
                    speedup_info = job_speedup[job_id]
                    if self.with_bw:
                        running_jobs.at[index, 'current_duration'] += speedup_info['alloc_bw']/speedup_info['read_count']
                        print(
                            f"[SIM] job_id {rj['job_id']:10} | "
                            f"speed_factor {speedup_info['alloc_bw']:12} | "
                            f"completed {int(running_jobs.at[index, 'current_duration'] / rj['duration'] * 100):3}%"
                        )

                    else:
                        running_jobs.at[index, 'current_duration'] += 1
                        print(
                            f"[SIM] job_id {rj['job_id']:10} | "
                            f"completed {int(running_jobs.at[index, 'current_duration'] / rj['duration'] * 100):3}%"
                            # f"completed {running_jobs.at[index, 'current_duration'], rj['duration']}%"
                        )
                        
                curr_running_jobs = list(running_jobs["job_id"])
                self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=False)
                utils.verify_tot_res(self.nodes, running_jobs)

            # nodes_snapshot = self.get_node_snapshot()
            
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
            if len(new_jobs):
                print('\n[SIM] --------------- Submitting job ---------------', new_jobs['job_id'].tolist())
                all_new_jobs.append(new_jobs['job_id'].tolist())
            
            # Add new jobs to the job queue
            if len(jobs) > 0:
                prev_job_list = list(jobs["job_id"])
                
            jobs = pd.concat([jobs, new_jobs], sort=False)
            
            # Schedule jobs
            jobs = job.schedule_jobs(jobs, self.scheduling_algorithm)
            
            if len(jobs) > 0:
                curr_job_list = list(jobs["job_id"])
            
            n_jobs = len(jobs)
            if sorted(prev_job_list) == sorted(curr_job_list) and sorted(prev_running_jobs) == sorted(curr_running_jobs):
                # print(time_instant, 'NO CHANGES Waiting for resources ...')
                n_jobs = 0
            
            jobs_to_submit = job.create_job_batch(jobs, n_jobs)
            
            unassigned_jobs = pd.DataFrame()
            assigned_jobs = pd.DataFrame()
            
            # Dispatch jobs
            if len(jobs_to_submit) > 0: 
                start_id = 0
                while start_id < len(jobs_to_submit):
                    jobs_submitted += 1
                    subset = jobs_to_submit.iloc[start_id:start_id+batch_size]
                    if self.enable_logging:
                        logging.log(TRACE, '\n-------------------------------NEW JOB---------------------------------')
                        logging.log(TRACE, subset.to_string())

                    row = subset.iloc[0]
                    job_id = int(subset['job_id'].iloc[0])
                    nmpds = row['num_pod']
                    num_gpu = row['num_gpu']
                    num_cpu = row['num_cpu']
                    write_count = row['write_count']
                    read_count = row['read_count']
                    
                    if job_id not in job_speedup:
                        job_speedup[job_id]={
                            'read_count':read_count,
                            'alloc_bw':read_count
                        }

                    output_string = ('** id: ' + str(job_id) +
                                    ', gpu: ' + str(num_gpu) +
                                    ', cpu: ' + str(num_cpu) +
                                    ', num_pod: ' + str(nmpds) +
                                    ', write_count: ' + str(write_count) +
                                    ', read_count: ' + str(read_count) )
                    print(output_string)

                    t = time.time()
                    time_now = 0 
                    cnt = 0
                    allctd = False
                    self.dispatch_jobs(progress_bid_events, queues, subset, nmpds, read_count) 
                    discard_flag = False

                    while not discard_flag and not allctd and cnt < 10:

                        cnt += 1
                        consume_queues = True
                        consume = True
                        while consume:
                            time_now += 1
                            consume_queues = False
                            
                            # Check for queues to drain
                            consume_queues = any(not q.empty() for q in queues)

                            if consume_queues:
                                for node in self.nodes:

                                    # Fetch previous bid if it exists
                                    if job_id in node.bids:
                                        prev_bid = copy.deepcopy(node.bids[job_id]['auction_id'])
                                    else:
                                        prev_bid = []

                                    node_queue = queues[node.id]

                                    cur_bid = prev_bid
                                    rebroadcast = False

                                    while not node_queue.empty():
                                        try:
                                            rebroadcast = node.work(time_now, time_instant)

                                            # Ensure node's resources are within expected limits
                                            assert node.get_avail_cpu() >= 0, "Available CPU less than 0"
                                            assert node.get_avail_cpu() <= node.initial_cpu, "Available CPU exceeds initial CPU"
                                            assert node.get_avail_gpu() >= 0, "Available GPU less than 0"
                                            assert node.get_avail_gpu() <= node.initial_gpu, "Available GPU exceeds initial GPU"

                                            # Update current bid after processing
                                            cur_bid = node.bids[job_id]['auction_id']

                                            # Mark the task as done
                                            node_queue.task_done()
                                        except Empty:
                                            break
                                        except Exception as e:
                                            print(f"[SIM] Error processing queue for node ID {node.id}: {e}")
                                            traceback.print_exc()
                                            break

                                    # Decide whether to forward to neighbors based on bid changes or rebroadcast flag
                                    if cur_bid != prev_bid or rebroadcast:
                                        if cur_bid.count(float('-inf')) != len(cur_bid):
                                            node.forward_to_neighbohors()
                                        
                            else:
                                # print('[SIM] All queues Drained!')
                                consume = False

                                if self.enable_logging:
                                    logging.log(TRACE, 'All nodes completed the processing... bid processing time:' + str(time_now) +
                                                ' jobs allocated:' + str(tot_assigned_jobs))

                                job_allocation_time.append(time.time()-t)

                                exec_time = time.time() - start_time
                            
                                t = time.time()

                                # Collect node results CONTROLLA!
                                a_jobs, u_jobs, _ = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
                                job_post_process_time.append(time.time() - t)

                                # Deallocate unassigned jobs

                                def u_job_handler(final_allocations, job_id, unassigned_ids, unassigned_jobs, u_jobs, all_jobs_ids, processed_jobs, tot_allocated_gpu, tot_allocated_cpu, discard_job=False, with_bw=False):
                                    # Check and update 'first_unassigned' metrics if unassigned
                                    if final_allocations['first_unassigned'] == 0:
                                        final_allocations['first_unassigned'] = final_allocations['allocated']
                                        final_allocations['first_unassigned_gpu'] = tot_allocated_gpu
                                        final_allocations['first_unassigned_cpu'] = tot_allocated_cpu

                                    # Add job_id to unassigned_ids and update unassigned count
                                    if job_id not in unassigned_ids:
                                        final_allocations['tot_unassigned'] += 1
                                        unassigned_ids.append(job_id)

                                    # Process jobs based on bandwidth conditions
                                    if not with_bw:
                                        u_df = pd.DataFrame(u_jobs)
                                        
                                        if discard_job:
                                            # Discard job logic
                                            all_jobs_ids.remove(job_id)
                                            final_allocations['discarded_jobs'] += 1
                                            final_allocations['gpu_discarded'] += (u_df['num_gpu'].iloc[0] * u_df['num_pod'].iloc[0]) / 100
                                            final_allocations['cpu_discarded'] += (u_df['num_cpu'].iloc[0] * u_df['num_pod'].iloc[0]) / 100
                                            processed_jobs = pd.concat([processed_jobs, u_df], sort=False)
                                            print('[MSG] Do not enqueue to unassigned jobs list', final_allocations['gpu_discarded'], final_allocations['cpu_discarded'])
                                        else:
                                            # Enqueue the job
                                            print('[MSG] Enqueue job', job_id)
                                            unassigned_jobs = pd.concat([unassigned_jobs, u_df])

                                    self.deallocate_jobs(progress_bid_events, queues, u_df, failure=True)

                                    return unassigned_ids, unassigned_jobs, processed_jobs
                                
                                if u_jobs:
                                    # for uj in u_jobs:
                                    #     print('[SIM] Unallocated job_id:', uj['job_id'], 'pods' , uj['num_pod'], 'num_cpu' , uj['num_cpu'], 'num_gpu' ,uj['num_gpu'], 'read_count' ,uj['read_count'])
                                    
                                    allocations = node.bids[job_id]['auction_id'] 
                                    print('\n[SIM] Unallocated job_id:', allocations, output_string)
                                    print(output_string)
                                    print(sim_stats)

                                    u_job_handler(final_allocations, job_id, unassigned_ids, unassigned_jobs, u_jobs, all_jobs_ids, processed_jobs, tot_allocated_gpu, tot_allocated_cpu, discard_job=False, with_bw=False)


                                def savemetrics(final_allocations, all_jobs_ids, job_id, nmpds, num_gpu, tot_assigned_jobs, tot_allocated_gpu,tot_allocated_cpu, subset):
                                        # Save metrics
                                        final_allocations['gpu'] += nmpds * num_gpu
                                        final_allocations['cpu'] += nmpds * num_cpu
                                        final_allocations['allocated'] += 1

                                        tot_assigned_jobs +=1
                                        tot_allocated_gpu += int(subset['num_gpu'].iloc[0]) * int(subset['num_pod'].iloc[0])
                                        tot_allocated_cpu += int(subset['num_cpu'].iloc[0]) * int(subset['num_pod'].iloc[0])

                                        # Construct and print the formatted log entry
                                        print(sim_stats)

                                        try:
                                            all_jobs_ids.remove(job_id)
                                            print(f"Job ID {job_id} has been removed. remaining {len(all_jobs_ids)}")
                                        except ValueError:
                                            print(f"Job ID {job_id} not found in the list.")
                                    
                                if a_jobs:
                                    assigned_jobs = pd.concat([assigned_jobs, pd.DataFrame(a_jobs)])
                                    allctd = utils.check_allocation(jobs=pd.DataFrame(a_jobs), nodes=self.nodes) 
                                    allocations = node.bids[job_id]['auction_id'] 

                                    # Allocate BW
                                    if self.with_bw and allctd:
                                        print(allocations, output_string)
                                        ps_on = set(allocations)
                                        
                                        if len(ps_on) == 1:
                                            savemetrics(final_allocations, all_jobs_ids, job_id, nmpds, num_gpu, tot_assigned_jobs, tot_allocated_gpu,tot_allocated_cpu, subset)
                                            print('\n[SIM] BW not used LLCTD')


                                        elif len(ps_on) > 1:
                                            print('[SIM] Allocating BW!')
                                            allocated_bw = False
                                            # tmp_bw = job_speedup[job_id]['alloc_bw'] / (len(ps_on)-1)
                                            # tmp_bw = round(tmp_bw, 2)
                                            cnt_bw = 0

                                            while job_speedup[job_id]['alloc_bw'] >= job_speedup[job_id]['read_countnot']/2 and not allocated_bw and cnt_bw<20:
                                                # print('allocationg', tmp_bw, cnt_bw)
                                                cnt_bw+=1
                                                allocated_bw = self.topology.allocate_ps_to_workers_single(
                                                    ps_node=[allocations[0]], 
                                                    worker_nodes=allocations[1:], 
                                                    required_bw=job_speedup[job_id]['alloc_bw'], 
                                                    allow_oversubscription=False)

                                                # allocated_bw = self.topology.allocate_ps_to_workers_balanced(allocations, tmp_bw)
                                                if not allocated_bw:
                                                    job_speedup[job_id]['alloc_bw'] = int(job_speedup[job_id]['alloc_bw'] * 0.9)

                                            if not allocated_bw:
                                                print('[SIM] DLLCT insufficient BW')
                                                u_job_handler(final_allocations, job_id, unassigned_ids, unassigned_jobs, a_jobs, all_jobs_ids, processed_jobs, tot_allocated_gpu, tot_allocated_cpu, discard_job=False, with_bw=False)
                                            else:
                                                savemetrics(final_allocations, all_jobs_ids, job_id, nmpds, num_gpu, tot_assigned_jobs, tot_allocated_gpu,tot_allocated_cpu, subset)
                                                print('[SIM] BW LLCTD , final', 'reduced: ', cnt_bw, 'bw:', job_speedup[job_id])


                                        consume = False
                                    
                                    elif allctd:
                                        print(allocations)
                                        savemetrics(final_allocations, all_jobs_ids, job_id, nmpds, num_gpu, tot_assigned_jobs, tot_allocated_gpu,tot_allocated_cpu, subset)
                                        print(f"{"[SIM] (NO BW) LLCTD:":<25} {job_id:<10}")    
                                        


                        # if not allctd and cnt > 2:
                        if not allctd:

                            if self.with_bw: # Here we try the allocation multiple times by reducing the bw which might be the bottleneck!
                                if cnt < 10 and job_speedup[job_id]['alloc_bw'] > 0.5:
                                    job_speedup[job_id]['alloc_bw'] = int(job_speedup[job_id]['alloc_bw']*0.8)
                                    subset.loc[subset['job_id'] == job_id, 'read_count'] = job_speedup[job_id]['alloc_bw']
                                    print('\n[SIM] RETRY with lower BW:', cnt, 'initial bw', job_speedup[job_id]['read_count'], 'reduced bw', job_speedup[job_id]['alloc_bw'])
                                    self.dispatch_jobs(progress_bid_events, queues, subset, nmpds, job_speedup[job_id]['alloc_bw']) 
                                
                                else:
                                    unassigned_jobs = pd.concat([unassigned_jobs, pd.DataFrame(u_jobs)])
                                    print('\n[SIM] (BW) NOT LLCTD', job_id)
                                    print(sim_stats)

                                    job_speedup[job_id]['alloc_bw'] = job_speedup[job_id]['read_count']
                                    discard_flag = True
                                    break
                            else:
                                print(f"{"[SIM] (NO BW) NOT LLCTD:":<25} {job_id:<10}")    
                                print(sim_stats)
                                break

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
                utils.verify_tot_res(self.nodes, running_jobs)


                done=True
                break

            # plttng = True
            plttng = False
            if plttng:
                self.topology.plot_node_available_bandwidth()  # Call the plot method for spine utilization
                self.topology.plot_bandwidth_utilization()  # Call the plot method for spine utilization
                self.generate_plots_resources()

        
        # Collect final node results
        # self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant+1, save_on_file=True)
        


        # Save processed jobs to CSV
        jobs_report.to_csv(self.filename + "_jobs_report.csv")
        self.tot_assigned_jobs = tot_assigned_jobs
        self.tot_allocated_cpu = tot_allocated_cpu
        self.tot_allocated_gpu = tot_allocated_gpu
        self.tot_allocated_bw = tot_allocated_bw



        t_gpu = sum(self.dataset['num_gpu'] * self.dataset['num_pod'])
        t_cpu = sum(self.dataset['num_cpu'] * self.dataset['num_pod'])
        final_allocations['t_gpu'] = t_gpu/100
        final_allocations['t_cpu'] = t_cpu/100
        final_allocations['gpu'] /= 100
        final_allocations['cpu'] /= 100
        
        # Assert section
        print(sim_stats)

        if len(all_jobs_ids) != 0:
            print(f"Missing jobs {all_jobs_ids}")
            for j in all_jobs_ids:
                print(self.dataset[self.dataset['job_id'] == j])
            assert False, f"Some jobs are missing: {all_jobs_ids} {len(all_new_jobs)} \n"

        assert int(final_allocations['allocated'] + final_allocations['discarded_jobs']) == int(len(self.dataset)), (
            f"{self.filename} Allocated items ({int(final_allocations['allocated'] + final_allocations['discarded_jobs'])}) do not match the dataset length ({len(self.dataset)})."
        )
        print(final_allocations['gpu'], final_allocations['gpu_discarded'])
        assert int(final_allocations['gpu'] + final_allocations['gpu_discarded']) == int(final_allocations['t_gpu']), (
            f"{self.filename} GPU allocations ({int(final_allocations['gpu'] + final_allocations['gpu_discarded'])}) do not match target GPU allocations ({final_allocations['t_gpu']})."
        )
        assert int(final_allocations['cpu'] + final_allocations['cpu_discarded']) == int(final_allocations['t_cpu']), (
            f"{self.filename} CPU allocations ({int(final_allocations['cpu'] + final_allocations['cpu_discarded']) }) do not match target CPU allocations ({final_allocations['t_cpu']})."
        )

        self.topology.assert_original_state()

        # Save data section

        final_allocations['first_unassigned_gpu'] /= 100
        final_allocations['first_unassigned_cpu'] /= 100
            
        final_allocations['utility'] = self.utility
        final_allocations['jct_tot'] = time_instant 
        
        final_allocations['jct_avg'] = ( jobs_report['complete_time'] - jobs_report['submit_time'] ) / jobs_report['duration'] 
        
        if self.with_bw:
            csv_file = 'BW_final_allocations.csv'
        else:
            # csv_file = 'fixed_duration_final_allocations.csv'
            csv_file = 'final_allocations.csv'
            # csv_file = '999final_allocations.csv'

        # Check if the file already exists
        file_exists = os.path.isfile(csv_file)

        # Writing the final_allocations to a CSV file, appending if file exists
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=final_allocations.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(final_allocations)




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
        
    # def rebid(self, progress_bid_events, return_val, queues, running_jobs, time_instant, batch_size, unassigned_jobs, assigned_jobs, exec_time):
    #     low_speedup_threshold = 1
    #     high_speedup_threshold = 1.2
                    
    #     jobs_to_reallocate, running_jobs = job.extract_rebid_job(running_jobs, low_thre=low_speedup_threshold, high_thre=high_speedup_threshold, duration_therehold=500)
                    
    #     if len(jobs_to_reallocate) > 0: 
    #         start_id = 0
    #         while start_id < len(jobs_to_reallocate):
    #             subset = jobs_to_reallocate.iloc[start_id:start_id+batch_size]
    #             # self.deallocate_jobs(progress_bid_events, queues, subset)
    #             print("Job deallocated")
    #             self.dispatch_jobs(progress_bid_events, queues, subset, check_speedup=True, low_th=low_speedup_threshold, high_th=high_speedup_threshold) 
    #             print("Job dispatched")
    #             a_jobs, u_jobs = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
    #             assigned_jobs = pd.concat([assigned_jobs, pd.DataFrame(a_jobs)])
    #             unassigned_jobs = pd.concat([unassigned_jobs, pd.DataFrame(u_jobs)])
    #             start_id += batch_size
    #     return running_jobs,unassigned_jobs,assigned_jobs

    #     #plot.plot_all(self.n_nodes, self.filename, self.job_count, "plot")

    def dispatch_jobs(self, progress_bid_events, queues, subset, nmpds, read_count, check_speedup=False, low_th=1, high_th=1.2):
        job.dispatch_job(dataset=subset, 
                         nmpds=nmpds, 
                         read_count=read_count, 
                         queues=queues,  
                         check_speedup=check_speedup, 
                         low_th=low_th, 
                         high_th=high_th)

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
            # 'link_prob': [self.probability],
            # 'link_bw' : [self.max_bw],
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