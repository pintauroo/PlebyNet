'''
This module impelments the behavior of a node
'''

from queue import Empty
import time

import numpy as np
from src.config import Utility, GPUType, GPUSupport
from src.network_topology import NetworkTopology
from src.node_performance import NodePerformance
from datetime import datetime, timedelta
import copy
import logging
import math 
import threading
from threading import Event
# import math5
# from src.topology import topo as LogicalTopology
# from FGD.src.utils import Quadrant
from typing import List, Dict, Any


TRACE = 5    

class BandwidthAllocationError(Exception):
    """Custom exception for bandwidth allocation issues."""
    pass

class InternalError(Exception):
    "Raised when the input value is less than 18"
    pass

class node:

    # def __init__(self, id, max_bw: float, gpu_type: GPUType, utility: Utility, alpha: float, enable_logging: bool, logical_topology, tot_nodes: int, progress_flag: bool, decrement_factor=0.00001, with_bw = False):
    def __init__(self, id, initial_gpu, initial_cpu, gpu_type: GPUType, max_bw: float, utility: Utility, alpha: float, enable_logging: bool, logical_topology, tot_nodes: int, progress_flag: bool, decrement_factor=0.00001, with_bw = False):
        self.id = id    # unique edge node id
        # self.gpu_type = gpu_type
        # self.gpu_type = GPUType.T4
        self.utility = utility
        self.alpha = alpha
        self.enable_logging = enable_logging
        self.logical_topology = logical_topology
        self.tot_nodes = tot_nodes
        self.progress_flag = progress_flag
        self.decrement_factor = decrement_factor

        if initial_cpu == None:
            self.initial_cpu, self.initial_gpu = GPUSupport.get_compute_resources(gpu_type)
            # print(self.initial_cpu, self.initial_gpu)
        else:
            self.initial_cpu = initial_cpu
            self.initial_gpu = initial_gpu

        self.gpu_type = GPUType.MISC

        self.initial_cpu *= 100
        self.initial_gpu *= 100
        self.updated_gpu = self.initial_gpu
        self.updated_cpu = self.initial_cpu
        self.updated_bw = 0
        self.initial_bw = 0
        self.timestamp_now = 0
        self.time_global = 0
        self.count_msgs = 0


        # self.performance = NodePerformance(self.initial_cpu, self.initial_gpu, self.id)
        
        if utility == Utility.FGD:
            self.individual_gpu = []
            self.allocated_on = {}
            for _ in range(self.initial_gpu):
                self.individual_gpu.append(1)

        self.available_cpu_per_task = {}
        self.available_gpu_per_task = {}
        self.available_bw_per_task = {}

        self.last_sent_msg = {}
        self.resource_remind = {}
        self.job_hosted = []

        self.cum_cpu_reserved = 0
        self.cum_gpu_reserved = 0
        self.cum_bw_reserved = 0
        self.with_bw = with_bw
        
    
        
        self.last_bid_timestamp = {}
        

        
        if self.initial_gpu != 0:
            #print(f"Node {self.id} CPU/GPU ratio: {self.initial_cpu/self.initial_gpu}")
            pass
        else:
            #print(f"Node {self.id} CPU/GPU ratio: <inf>")
            pass
        
        self.counter = {}
        
        self.user_requests = []
        self.item={}
        self.bids= {}
        self.layer_bid_already = {}
        self.node_id = id
        self.max_bw = max_bw
        self.allocated_bw = 0.0  # Currently allocated bandwidth
        self.communication_log = []  # To track communications with other nodes

    def allocate_bandwidth(self, bw: float):
        """
        Allocate bandwidth to the node.

        :param bw: Bandwidth to allocate.
        :raises BandwidthAllocationError: If allocation exceeds node capacity.
        """
        if self.allocated_bw + bw > self.max_bw:
            raise BandwidthAllocationError(
                f"Node {self.node_id}: Allocation of {bw} exceeds maximum bandwidth {self.max_bw}."
            )
        self.allocated_bw += bw
        # print(f"Node {self.node_id}: Allocated {bw} BW (Total Allocated: {self.allocated_bw}/{self.max_bw})")

    def __repr__(self):
        return f"Node(id={self.node_id}, allocated_bw={self.allocated_bw}/{self.max_bw})"

    def get_avail_gpu(self):
        return self.updated_gpu
    
    def get_avail_cpu(self):
        return self.updated_cpu
        
    def compute_curr_cpu_power_consumption(self):
        return self.power_function(self.initial_cpu - self.updated_cpu, "cpu")
    
    def compute_curr_gpu_power_consumption(self):
        return self.power_function(self.initial_gpu - self.updated_gpu, "gpu")
        
    # def set_queues_(self, q, use_queue):
    #     self.q = q
    #     self.empty_queue = use_queue
    #     self.empty_queue[self.id].set()

    def set_queues(self, q):
        self.q = q

    
    def init_null(self):
        # print(self.item['duration'])
        self.bids[self.item['job_id']]={
            "count":0,
            "consensus_count":0,
            "forward_count":0,
            "deconflictions":0,
            "job_id": self.item['job_id'], 
            # "user": int(), 
            "auction_id": list(), 
            "NN_gpu": self.item['NN_gpu'], 
            "NN_cpu": self.item['NN_cpu'], 
            "NN_data_size": self.item['NN_data_size'],
            "bid": list(), 
            "bid_gpu": list(), 
            "bid_cpu": list(), 
            "bid_bw": list(), 
            "timestamp": list(),
            "arrival_time":self.time_global,
            "start_time": 0, #datetime.now(),
            "progress_time": 0, #datetime.now(),
            "complete":False,
            "complete_timestamp":None,
            "N_layer_min": self.item["N_layer_min"],
            "N_layer_max": self.item["N_layer_max"],
            "edge_id": self.id, 
            "N_layer": self.item["N_layer"],
            'consensus':False,
            'clock':False,
            'rebid':False,
            "N_layer_bundle": self.item["N_layer_bundle"],
            "retry":0,
            "retry_ts":0,
            'write_count':int(self.item['write_count']),
            'read_count':int(self.item['read_count'])


            }
        
        self.layer_bid_already[self.item['job_id']] = [False] * self.item["N_layer"]

        self.available_gpu_per_task[self.item['job_id']] = [self.updated_gpu]
        self.available_cpu_per_task[self.item['job_id']] = [self.updated_cpu]


        # NN_len = len(self.item['NN_gpu'])
        
        for _ in range(0, self.item['N_layer']):
            self.bids[self.item['job_id']]['bid'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_gpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_cpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_bw'].append(float('-inf'))
            self.bids[self.item['job_id']]['auction_id'].append(float('-inf'))
            self.bids[self.item['job_id']]['timestamp'].append(-1)
            # self.bids[self.item['job_id']]['timestamp'].append(datetime.now() - timedelta(days=1))

    def util_rate(self):
        cpus_util = 1 - self.updated_cpu / self.initial_cpu
        if self.updated_gpu > 0:
            gpus_util = 1 - self.updated_gpu / self.initial_gpu
            util_rate = round((gpus_util + cpus_util) / 2)
        else:
            util_rate = 0 # round(cpus_util)
        return util_rate

    def utility_function(self, avail_bw, avail_cpu, avail_gpu):
        if self.item['job_id'] in self.job_hosted and GPUSupport.compute_speedup(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type']) ) == self.item['speedup']:
            return -999999999
        
        def f(x, alpha, beta):
            if beta == 0 and x == 0:
                return 1
            
            # shouldn't happen
            if beta == 0 and x != 0:
                return 0
            
            # if beta != 0 and x == 0 is not necessary
            return math.exp(-((alpha/100) * (x - beta))**2)
            #return math.exp(-(alpha/100)*(x-beta)**2)

        if (isinstance(avail_bw, float) and avail_bw == float('inf')):
            avail_bw = self.initial_bw
        
        # we assume that every job/node has always at least one CPU
        if self.utility == Utility.STEFANO:
            x = 0
            if self.item['NN_gpu'] == 0:
                x = 0
            else:
                x = self.item['NN_cpu']/self.item['NN_gpu']
                
            beta = 0
            if avail_gpu == 0:
                beta = 0
            else:
                beta = avail_cpu/avail_gpu
            if self.alpha == 0:
                return f(x, 0.01, beta)
            else:
                return f(x, self.alpha, beta)
        elif self.utility == Utility.LIKELIHOOD:
            x = self.item['NN_cpu']/self.item['NN_gpu']
            if self.updated_gpu > 0:
                local_ratio = self.updated_cpu /self.updated_gpu
                util_rate = (x / local_ratio )
            else:
                util_rate = 0
            return util_rate

        elif self.utility == Utility.ALPHA_GPU_CPU:
            return (self.alpha*(avail_bw/self.initial_bw))+((1-self.alpha)*(avail_cpu/self.initial_cpu)) #BW vs CPU
        elif self.utility == Utility.ALPHA_GPU_CPU:
            return (self.alpha*(avail_gpu/self.initial_gpu))+((1-self.alpha)*(avail_cpu/self.initial_cpu)) #GPU vs CPU
        elif self.utility == Utility.ALPHA_GPU_BW:
            return (self.alpha*(avail_gpu/self.initial_gpu))+((1-self.alpha)*(avail_bw/self.initial_bw)) # GPU vs BW
        elif self.utility == Utility.LGF:
            corrective_factor = GPUSupport.get_GPU_corrective_factor(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type']), decrement=self.decrement_factor)
            return avail_gpu * corrective_factor
        elif self.utility == Utility.SGF:
            corrective_factor = GPUSupport.get_GPU_corrective_factor(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type']), decrement=self.decrement_factor)
            # return (self.initial_gpu - avail_gpu) * corrective_factor
            return (800 - avail_gpu) * corrective_factor
        elif self.utility == Utility.UTIL:
            return self.util_rate()
        elif self.utility == Utility.SEQ:
            return 100-self.id
        elif self.utility == Utility.DRF:
                nn_gpu = self.item.get('NN_gpu', 0)
                nn_cpu = self.item.get('NN_cpu', 0)
                
                # Check for zero in denominators and return 0 if any are zero
                if self.updated_gpu == 0 or self.updated_cpu == 0:
                    return 0.0
                
                # Calculate ratios
                gpu_ratio = nn_gpu / self.updated_gpu
                cpu_ratio = nn_cpu / self.updated_cpu
                
                # Return the maximum ratio
                return max(gpu_ratio, cpu_ratio)

        elif self.utility == Utility.TETRIS:
            # Task resource demands normalized by total node capacity
            task_res_gpu = self.item['NN_gpu'] / self.initial_gpu  # Task's GPU requirement normalized
            task_res_cpu = self.item['NN_cpu'] / self.initial_cpu  # Task's CPU requirement normalized

            # Node's available resources normalized by total node capacity
            node_res_gpu = self.updated_gpu / self.initial_gpu  # Node's available GPU normalized
            node_res_cpu = self.updated_cpu / self.initial_cpu  # Node's available CPU normalized

            # Dot product-like computation
            return task_res_gpu * node_res_gpu + task_res_cpu * node_res_cpu
        



        elif self.utility == Utility.POWER:
            pass # we need to define here the utility function
        elif self.utility == Utility.SPEEDUP:
            return GPUSupport.compute_speedup(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type'])) * avail_gpu
        elif self.utility == Utility.SPEEDUPV2:
            return GPUSupport.compute_speedup(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type'])) * (avail_gpu/self.initial_gpu)
        elif self.utility == Utility.NET:
            speed = GPUSupport.compute_speedup(self.gpu_type, GPUSupport.get_gpu_type(self.item['gpu_type'])) * (avail_gpu/self.initial_gpu)
            return speed / (avail_bw/self.initial_bw)

    def forward_to_neighbohors(self, custom_dict=None, resend_bid=False, first_msg=False):            
        msg = {
            "job_id": self.item['job_id'], 
            # "user": self.item['user'],
            "edge_id": self.id, 
            "NN_gpu": self.item['NN_gpu'],
            "NN_cpu": self.item['NN_cpu'],
            "NN_data_size": self.item['NN_data_size'], 
            "N_layer": self.item["N_layer"],
            "N_layer_min": self.item["N_layer_min"],
            "N_layer_max": self.item["N_layer_max"],
            "N_layer_bundle": self.item["N_layer_bundle"],
            "gpu_type": self.item["gpu_type"],
            "speedup": self.item["speedup"],
            "increase": self.item["increase"],
            # "ps": self.item["ps"],
            "write_count":self.item['write_count'],
            "read_count":self.item['read_count']

        }
        
        
        # if first_msg:
        #     for i in range(self.tot_nodes):
        #         topology = self.logical_topology.calculate_host_to_host_adjacency_matrix()
        #         if topology[i][self.id] and self.id != i:
        #         # if topology[i][self.id] and self.id != i and i != self.item['edge_id']:
        #             self.q[i].put(msg)
        #     return
        
        if custom_dict == None and not resend_bid:
            msg["auction_id"] = copy.deepcopy(self.bids[self.item['job_id']]['auction_id'])
            msg["bid"] = copy.deepcopy(self.bids[self.item['job_id']]['bid'])
            msg["timestamp"] = copy.deepcopy(self.bids[self.item['job_id']]['timestamp'])
        elif custom_dict != None and not resend_bid:
            msg["auction_id"] = copy.deepcopy(custom_dict['auction_id'])
            msg["bid"] = copy.deepcopy(custom_dict['bid'])
            msg["timestamp"] = copy.deepcopy(custom_dict['timestamp'])
        elif resend_bid:
            if "auction_id" in self.item:
                msg["auction_id"] = copy.deepcopy(self.item['auction_id'])
                msg["bid"] = copy.deepcopy(self.item['bid'])
                msg["timestamp"] = copy.deepcopy(self.item['timestamp'])
            #msg['edge_id'] = self.item['edge_id']
                
        if self.item['job_id'] not in self.last_sent_msg:
            self.last_sent_msg[self.item['job_id']] = msg
        elif (self.last_sent_msg[self.item['job_id']]["auction_id"] == msg["auction_id"] and \
            self.last_sent_msg[self.item['job_id']]["timestamp"] == msg["timestamp"] and \
            self.last_sent_msg[self.item['job_id']]["bid"] == msg["bid"]):
            # msg already sent before
            return
        
        if self.enable_logging:
            self.print_node_state('[FORWARD]', bid = True, state= False, forward=True)
            # logging.log(TRACE, '[FORWARD]')


        # topology = self.logical_topology.calculate_host_to_host_adjacency_matrix()
        for i in range(self.tot_nodes):
            # if self.id != i and topology[i][self.id]:
            if self.id != i:
                self.q[i].put(msg)
                self.count_msgs+=1
                
        
        #self.last_sent_msg[self.item['job_id']] = msg

    def print_node_state(self, msg, bid=False, state =True, forward = False, type='debug'):
        logger_method = getattr(logging, type)
        #print(str(self.item.get('auction_id')) if bid and self.item.get('auction_id') is not None else "\n")
        logger_method(str(msg) +
                    (" job_id:" + str(self.item['job_id'])      if state else "") +
                    (" rcv:" + str(self.id)                  if state else "") +
                    (" from_edge:" + str(self.item['edge_id'])  if state else "") + 
                    (" initial GPU:" + str(self.initial_gpu)   if state else "") +
                    (" available GPU:" + str(self.updated_gpu) if state else "") + 
                    (" initial CPU:" + str(self.initial_cpu)   if state else "") +
                    (" available CPU:" + str(self.updated_cpu) if state else "") + 
                    #" initial BW:" + str(self.initial_bw) if hasattr(self, 'initial_bw') else str(0) +
                    #" available BW:" + str(self.updated_bw) if hasattr(self, 'updated_bw') else str(0)  +
                    # "\n" + str(self.layer_bid_already[self.item['job_id']]) +
                    (("\n"+str(self.bids[self.item['job_id']]['auction_id'])+" "+str(self.bids[self.item['job_id']]['bid'])+" "+str(self.bids[self.item['job_id']]['timestamp'])) if bid else "") +
                    (("\n" + str(self.item.get('auction_id'))+" "+ str(self.item.get('bid'))+" "+ str(self.item.get('timestamp')) if bid and self.item.get('auction_id') is not None and not forward  else "\n"))
                    )
    
    def update_tmp_val(self, tmp, index, id, bid, timestamp, count):
        tmp['job_id'] = self.item['job_id']
        tmp['auction_id'][index] = id
        tmp['bid'][index] = bid
        tmp['timestamp'][index] = timestamp
        # return index + 1

    def reset(self, index, dict, bid_time):
        dict['auction_id'][index] = float('-inf')
        dict['bid'][index]= float('-inf')
        dict['timestamp'][index] = bid_time # - timedelta(days=1)
        return index + 1
    
    # NOTE: inprove in future iterations
    def compute_layer_score(self, cpu, gpu, bw):
        return gpu
    
    def _compute_fragmentation(self, workload_cpus, workload_gpus, node_gpus):
        u = self.compute_u(node_gpus)
        f = 0
        
        for i in range(len(workload_cpus)):
            quadrant = self.compute_quadrant(workload_cpus[i], workload_gpus[i], u)
                
            if quadrant == Quadrant.Q124:
                for g in node_gpus:
                    f += g
            elif quadrant == Quadrant.Q3:
                for g in node_gpus:
                    if g < min(workload_gpus[i], 1):
                        f += g
            else:
                for g in node_gpus:
                    f += g
        
        return f
    
    def compute_quadrant(self, cpu, gpu, u):
        if gpu == 0:
            return Quadrant.OTHER
        
        if cpu > self.updated_cpu or gpu > u:
            return Quadrant.Q124
        else:
            return Quadrant.Q3
        
    def compute_u(self, node_gpus):
        fully_unallocated = 0
        maximum_partial = 0
                
        for i in range(len(node_gpus)):
            if node_gpus[i] == 1:
                fully_unallocated += 1
            elif node_gpus[i] > maximum_partial:
                maximum_partial = node_gpus[i]
                        
        u = fully_unallocated + maximum_partial
        return u

    def bid_FGD(self):        
        sum = 0
        for i in range(len(self.item["NN_cpu"])): 
            sum += self.item["NN_cpu"][i]
        if sum > self.updated_cpu:
            return False
        
        if True in self.layer_bid_already[self.item['job_id']]:
            return False
        
        for i in range(len(self.layer_bid_already[self.item['job_id']])):
            self.layer_bid_already[self.item['job_id']][i] = True
        
        node_gpus = copy.deepcopy(self.individual_gpu) 
        self.allocated_on[self.item["job_id"]] = []
        
        fragmentation = 0
        
        # for each task in the workload
        best_frag = float('inf')
        best_id = -1
                    
        for j in range(len(node_gpus)):
            if node_gpus[j] < self.item["NN_gpu"][i]:
                continue
            
            f_before = self._compute_fragmentation(self.item["NN_cpu"], self.item["NN_gpu"], node_gpus)
            node_gpus[j] -= self.item["NN_gpu"][0]
            f_after = self._compute_fragmentation(self.item["NN_cpu"], self.item["NN_gpu"], node_gpus)
            frag = f_after - f_before
            
            if frag < best_frag:
                best_frag = frag
                best_id = j
                
            node_gpus[j] += self.item["NN_gpu"][0]
            
        if best_frag == float('inf'):
            self.layer_bid_already[self.item['job_id']][0] = True
            return False
        
        self.allocated_on[self.item["job_id"]].append(best_id)
        fragmentation += best_frag * 1/len(self.item["NN_gpu"])
        fragmentation = -fragmentation
                
        success = False
        for i in range(len(self.bids[self.item['job_id']]['bid'])):
            if fragmentation > self.bids[self.item['job_id']]['bid'][i] or self.bids[self.item['job_id']]['bid'][i] == float('-inf'):
                self.bids[self.item['job_id']]['bid'][i] = fragmentation
                self.bids[self.item['job_id']]['auction_id'][i] = self.id
                # self.bids[self.item['job_id']]['timestamp'][i] = datetime.now()
                self.bids[self.item['job_id']]['timestamp'][i] = self.time_now
                self.updated_cpu -= self.item["NN_cpu"][i]
                self.updated_gpu -= self.item["NN_gpu"][i]
                success = True
            self.layer_bid_already[self.item['job_id']][i] = True
        
        if success:
            self.individual_gpu[best_id] -= self.item["NN_gpu"][0]
            return True
        
        return False
    
    def gang_schedule(self, count: int, start: int, bidtime: Any, connected_to: Any = None) -> None:
        """
        Schedules tasks in a gang scheduling manner, ensuring that tasks are allocated efficiently.
        """
        if self.with_bw:
            topology = self.logical_topology.adj.copy()  # Copy to avoid mutating shared state

        nn_cpu = self.item.get('NN_cpu')
        nn_gpu = self.item.get('NN_gpu')
        read_count = self.item.get('read_count')
        max_workers = self.item.get('N_layer_max')
        num_workers = self.item.get('N_layer')
        

        res_bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)

        # Local reference to bids for faster access
        job_id = self.item.get('job_id')
        if not job_id or job_id not in self.bids:
            return  # Early exit if job_id is not provided or not in bids

        job_bids = self.bids[job_id]

        for i in range(start, start + count):
            # Before updating resources, check to prevent negative values
            if self.updated_gpu < nn_gpu or self.updated_cpu < nn_cpu:
                # Insufficient resources, cannot proceed with this index
                break  # Or continue to skip this index and try the next one

            # Check if this index has already been bid on
            if self.layer_bid_already[job_id][i]:
                continue  # Skip if already bid

            if self.with_bw:
                # if connected_to is not None:
                #     # Ensure connected_to exists in the topology
                #     if (connected_to in topology[self.node_id] and
                #             topology[self.node_id][connected_to] >= read_count):
    
                #         # Directly connected to the previous node with sufficient bandwidth
                #         net_bid = topology[self.node_id][connected_to]
                #         topology[self.node_id][connected_to] -= read_count
                #         self.print_node_state('Connected to ' + str(net_bid) ,bid = True, state= False)

                #     else:
                #         # Not directly connected or insufficient bandwidth
                #         self.print_node_state('Not directly connected or insufficient bandwidth' + str(topology[self.node_id][connected_to]) +' ' + str(read_count ),bid = True, state= False)
                        
                #         continue  # Skip this index due to bandwidth constraints
                # else:
                #     # if start == 0:
                #         # First bidder or not directly connected
                neighbor_bandwidths = [
                    bw for neighbor_id, bw in enumerate(topology[self.node_id])
                    if neighbor_id != self.node_id
                ]

                net_bid = sum(neighbor_bandwidths) / len(neighbor_bandwidths)
                if net_bid < read_count * (num_workers / max_workers):
                    net_bid = 0
                # self.print_node_state('NET bidder ' + str(read_count) + ' ' + str(num_workers)  + ' ' + str(max_workers)  + ' ' + str(net_bid) ,bid = True, state= False)



                if net_bid > 0 and res_bid > 0:
                    bid = net_bid / res_bid
                else:
                    bid = 0
            else:
                bid = res_bid

            # Update resources
            self.updated_gpu -= nn_gpu
            self.updated_cpu -= nn_cpu

            # Mark that we've already bid on this layer
            self.layer_bid_already[job_id][i] = True

            # Assign bids using local reference
            job_bids['bid'][i] = bid
            job_bids['auction_id'][i] = self.node_id
            job_bids['timestamp'][i] = bidtime

            # Optionally, log the successful allocation
            # print(f"Node {self.node_id} successfully bid on index {i} with bid {bid}")



    def count_layers(self) -> int:
        """
        Counts the number of layers the node can bid on based on resource availability.
        """
        job_id = self.item.get('job_id')
        if not job_id:
            return 0  # Early exit if job_id is not provided

        auction = self.bids.get(job_id, {}).get('auction_id', [])
        if not auction:
            return 0  # Early exit if no auction data

        # Prevent division by zero by setting defaults if necessary
        nn_cpu: int = self.item.get('NN_cpu', 1)
        nn_gpu: int = self.item.get('NN_gpu', 1)

        # Calculate resource counts
        res_count_cpu = self.updated_cpu // nn_cpu
        res_count_gpu = self.updated_gpu // nn_gpu
        n_layer_max = self.bids[job_id].get('N_layer_max', len(auction))

        res_count = min(res_count_cpu, res_count_gpu, len(auction), n_layer_max)
        count: int = res_count

        if self.with_bw and 0 < res_count < len(auction):
            elements = self.logical_topology.adj[self.node_id]
            read_count = self.item.get('read_count', 1)

            if max(elements) >= read_count:
                total = sum(elements)
                num_elements = len(elements)
                average = total // num_elements  # Ensure average is an integer

                # Calculate the potential new count
                potential_count = int(min(res_count, average // read_count))

                if 1 < potential_count < len(auction):
                    # Find the largest divisor of len(auction) that is <= potential_count
                    largest = self.largest_divisor(len(auction), potential_count)
                    count = largest
            else:
                count = 0
        return int(count)

    def largest_divisor(self, n: int, max_divisor: int) -> int:
        """
        Finds the largest divisor of `n` that is less than or equal to `max_divisor`.
        """
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1





    # def count_layers(self):
    #     # tmp_cpu = copy.deepcopy(self.updated_cpu)
    #     # tmp_gpu = copy.deepcopy(self.updated_gpu)
    #     count = 0 
    #     auction = self.bids[self.item['job_id']]['auction_id']
    #     res_count =  int(min(self.updated_cpu/self.item['NN_cpu'], self.updated_gpu/self.item['NN_gpu'], len(auction) ))
    #     if res_count!= 0 and res_count < len(auction):
    #         topology = copy.deepcopy(self.logical_topology.adj)
    #         elements = topology[self.id]
    #         total = sum(float(element) for element in elements)
    #         average = int(float(total) / len(elements)) 
    #         count = min(res_count, average)
    #         print(self.id, 'net', average, count)
    #     else:        
    #         count = res_count
            # print(self.id, 'full', count)

        # print('final',res_count, average, min(res_count, average))    



        # first = True
        # # self.with_bw = False
        # if float('-inf') in self.bids[self.item['job_id']]['auction_id'] :
        #     start = self.bids[self.item['job_id']]['auction_id'].index(float('-inf'))

        #     for i in range(start, start + self.item['N_layer']):
        #         # if not self.layer_bid_already[self.item['job_id']][i] \
                    
        #         if self.with_bw:

        #             if self.item['NN_gpu'] <= tmp_gpu and\
        #                 self.item['NN_cpu'] <= tmp_cpu:
        #                 if start == 0:
        #                     tmp_cpu-=self.item['NN_cpu']
        #                     tmp_gpu-=self.item['NN_gpu']
        #                     count+=1

        #                 else:
        #                     if first:
        #                         # topoology = self.logical_topology.get_updated_bw_matrix().copy()
        #                         topoology = copy.deepcopy(self.logical_topology.adj)
                                
        #                         connected_to = self.bids[self.item['job_id']]['auction_id'][0]
        #                         first = False
        #                     if topoology[self.id][connected_to]>=self.item['read_count']:
        #                         count+=1
        #                         tmp_cpu-=self.item['NN_cpu']
        #                         tmp_gpu-=self.item['NN_gpu']
        #                         topoology[self.id][connected_to]-=self.item['read_count']
                
        #         else:
        #             if self.item['NN_gpu'] <= tmp_gpu and\
        #                 self.item['NN_cpu'] <= tmp_cpu:
        #                     tmp_cpu-=self.item['NN_cpu']
        #                     tmp_gpu-=self.item['NN_gpu']
        #                     count+=1
                        
        # return count


    def bid_index(self, count: int, bidtime: Any) -> None:
        """
        Handles bidding on layers other than the first layer.
        """
        topology = self.logical_topology.adj.copy()  # Corrected variable name and made a copy
        job_id = self.item.get('job_id')

        if not job_id or job_id not in self.bids:
            return  # Early exit if job_id is not provided or not in bids

        auction_id_list = self.bids[job_id].get('auction_id', [])

        if count > 0 and float('-inf') in auction_id_list:
            start = auction_id_list.index(float('-inf'))
            count = min(count, auction_id_list.count(float('-inf')))

            if start != 0:
                if self.with_bw:
                    single_ps = True
                    if single_ps:
                        connected_to = auction_id_list[0]  # Parameter Server (PS)
                        if (topology[self.node_id][connected_to] > 0 and
                                topology[connected_to][self.node_id] > 0):
                            # If directly connected, bid on the next layer
                            self.gang_schedule(count, start, bidtime, connected_to)
                else:
                    self.gang_schedule(count, start, bidtime)
            else:
                self.gang_schedule(count, start, bidtime)

    def bid(self):  
        job_id = self.item.get('job_id')
        if not job_id:
            return False  # Early exit if job_id is not provided
        
        job_GPU_type = GPUSupport.get_gpu_type(self.item['gpu_type'])
        
        if not GPUSupport.can_host(self.gpu_type, job_GPU_type):
            return False
        
        # if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) < self.item['speedup'] and self.item["increase"]:
        #     return False

        # if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) > self.item['speedup'] and not self.item["increase"]:
        #     return False
        
        # if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) == self.item['speedup'] and self.item['job_id'] not in self.job_hosted:
        #     return False

        bidtime = self.time_now

        count = self.count_layers()

        # If first time bidding (no received bid yet), then just bid
        if 'bid' not in self.item:
            self.gang_schedule(count, 0, bidtime)
        else:
            if not self.layer_bid_already[job_id][0]:
                # Can you outbid the first set?
                winner = self.item['auction_id'].count(self.item['auction_id'][0])
                if count >= winner:
                    self.gang_schedule(count, 0, bidtime)
            else:
                auction_ids = self.item.get('auction_id', [])
                if (self.node_id not in auction_ids and
                        self.node_id not in self.bids[job_id].get('auction_id', [])):
                    self.bid_index(count, bidtime)

        values = self.bids[job_id]['auction_id']
        assert all((v >= 0 or v == float('-inf')) for v in values), \
            "All values must be greater than 0 or equal to float('-inf')"

        return self.node_id in self.bids[job_id]['auction_id']
                        
    def update_bw(self, prev_bid, deallocate=False):
        bw = 0
                
        if prev_bid is not None:
            for i, b_id in enumerate(prev_bid):
                if b_id == self.id:
                    for j in range(len(self.item["NN_data_size"][i])):
                        if i == j:
                            continue
                        
                        if self.item["NN_data_size"][i][j] != 0 and prev_bid[j] != self.id:
                            bw += self.item["NN_data_size"][i][j]
                            
        if deallocate:
            self.updated_bw += bw
            return
        
        if self.item['job_id'] in self.bids:                
            for i, b_id in enumerate(self.bids[self.item['job_id']]['auction_id']):
                if b_id == self.id:
                    for j in range(len(self.item["NN_data_size"][i])):
                        if i == j:
                            continue
                        
                        if self.item["NN_data_size"][i][j] != 0 and self.bids[self.item['job_id']]['auction_id'][j] != self.id:
                            bw -= self.item["NN_data_size"][i][j]
                
            
        self.updated_bw += bw
    
    def deconfliction(self, index, count):
        rebroadcast = False
        k = self.item['edge_id'] # sender
        i = self.id # receiver
        self.bids[self.item['job_id']]['deconflictions']+=1
        release_to_client = False
        previous_winner_id = float('-inf')
        job_id = self.item["job_id"]
        
        tmp_local = copy.deepcopy(self.bids[self.item['job_id']])
        prev_bet = copy.deepcopy(self.bids[self.item['job_id']])
        # index = 0
        
        reset_flag = False
        reset_ids = []
        bid_time = self.time_now
        # bid_time = datetime.now()

        

        if index < self.item["N_layer"]:
            
            z_kj = self.item['auction_id'][index]
            z_ij = tmp_local['auction_id'][index]
            y_kj = self.item['bid'][index]
            y_ij = tmp_local['bid'][index]
            t_kj = self.item['timestamp'][index]
            t_ij = tmp_local['timestamp'][index]

            if self.enable_logging:
                logger_method = getattr(logging, 'debug')
                logger_method('DECONFLICTION:\n' +
                            ' rcv(i):' + str(i)  + ' snd(k):' + str(k) + '\n' +
                            '   z_ij:' + str(z_ij)    + '   z_kj:' + str(z_kj) + '\n' +
                            '   y_ij:' + str(y_ij)    + '   y_kj:' + str(y_kj) + '\n' + 
                            '   t_ij:' + str(t_ij)    + '   t_kj:' + str(t_kj))
            # chi mi manda il messaggio è il vincitore
            if z_kj==k : 
                # io penso di essere il vincitore
                if z_ij==i:
                    if y_kj>y_ij: 
                        rebroadcast = True
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #1')
                        # if index == 0:
                        #     release_to_client = True
                        # elif previous_winner_id == float('-inf'):
                        #     previous_winner_id = prev_bet['auction_id'][index-1]
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1

                    elif y_kj==y_ij and z_kj<z_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #3')
                        rebroadcast = True
                        # if index == 0:
                        #     release_to_client = True
                        # elif previous_winner_id == float('-inf'):
                        #     previous_winner_id = prev_bet['auction_id'][index-1]

                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1

                    else:# (y_kj<y_ij):
                        rebroadcast = True
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #2')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, tmp_local['auction_id'][index], tmp_local['bid'][index], bid_time, count)
                            index += 1
                            count -= 1
                    
                    # else:
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  ' #3else')
                    #     index+=1
                    #     rebroadcast = True

                elif z_ij==k:
                    if t_kj>t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#4')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True 
                    else:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #5 - 6')
                        index+=1
                
                elif z_ij == float('-inf'):
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #12')
                    while count != 0:
                        self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                        index += 1
                        count -= 1
                    rebroadcast = True

                elif z_ij!=i and z_ij!=k:
                    if y_kj>y_ij and t_kj>=t_ij: #KTM
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #7KTM')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True
                    elif y_kj<y_ij and t_kj<t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #8')
                        index += 1
                        rebroadcast = True

                    elif y_kj==y_ij and z_kj<z_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#9-new')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)    
                            index += 1
                            count -= 1              
                        rebroadcast = True
                    # else:
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  ' #9KTM')
                    #     rebroadcast = True
                    #     index+=1

                    elif y_kj<y_ij and t_kj>=t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #10reset')
                        index += 1
                        rebroadcast = True
                        # reset_ids.append(index)
                        # index += 1
                        # reset_flag = True
                        # rebroadcast = True  
                    elif y_kj>y_ij and t_kj<t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #11rest')
                        # index, reset_flag = self.reset(index, tmp_local)
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True  
                    else:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #11else')
                        index += 1  
                        rebroadcast = True  
                
                else:
                    index += 1   
                    if self.enable_logging:
                        logging.log(TRACE, "eccoci")    
            
            # chi mi manda il messaggio dice che vinco io
            elif z_kj==i:                                
                if z_ij==i:
                    if t_kj>t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #13Flavio')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True  
                    else:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #13elseFlavio')
                        index+=1
                        #rebroadcast = True

                elif z_ij==k:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #14reset')
                    reset_ids.append(index)
                    # index = self.reset(index, self.bids[self.item['job_id']])
                    index += 1
                    reset_flag = True
                    rebroadcast = True                        

                elif z_ij == float('-inf'):
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #16gay')
                    rebroadcast = True
                    #tmp_local['timestamp'][index] = bid_time
                    index+=1
                
                elif z_ij!=i and z_ij!=k:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #15')
                    rebroadcast = True
                    index+=1
                
                else:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #15else')
                    rebroadcast = True
                    index+=1                
            
            # chi mi manda il messaggio non mette un vincitore
            elif z_kj == float('-inf'):
                if z_ij==i:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #31')
                    rebroadcast = True
                    index+=1
                    
                elif z_ij==k:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #32')
                    while count != 0:
                        self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                        index += 1
                        count -= 1
                    rebroadcast = True
                    
                elif z_ij == float('-inf'):
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #34')
                    index+=1
                    
                elif z_ij!=i and z_ij!=k:
                    if t_kj>t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #33')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True
                    else: 
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #33else')
                        index+=1
                    
                else:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #33elseelse')
                    index+=1
                    rebroadcast = True

            # chi manda il messaggio dice che non vinco nè io nè lui
            elif z_kj!=i and z_kj!=k:   
                                    
                if z_ij==i:
                    if y_kj>y_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#16puttana')
                        rebroadcast = True
                        # if index == 0:
                        #     release_to_client = True
                        # elif previous_winner_id == float('-inf'):
                        #     previous_winner_id = prev_bet['auction_id'][index-1]
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                    elif y_kj==y_ij and z_kj<z_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#17')
                        rebroadcast = True
                        # index+=1

                        # if index == 0:
                        #     release_to_client = True
                        # elif previous_winner_id == float('-inf'):
                        #     previous_winner_id = prev_bet['auction_id'][index-1]
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                    else:# (y_kj<y_ij):
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#19')
                        rebroadcast = True
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, tmp_local['auction_id'][index], tmp_local['bid'][index], bid_time, count)
                            index += 1
                            count -= 1
                    # else:
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  ' #19else')
                    #     index+=1
                    #     rebroadcast = True

                # io penso che vinca lui
                elif z_ij==k:
                    # if y_kj>y_ij:
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  ' #20Flavio')
                    #     while count != 0:
                    #         self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                    #         index += 1
                    #         count -= 1
                    #     rebroadcast = True 
                    # elif (y_kj==y_ij and z_kj<z_ij):
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  ' #3stefano')
                    #     rebroadcast = True
                    #     while count != 0:
                        # self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], self.bids[self.item['job_id']])
                    # elif t_kj>t_ij:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  '#20KTM')
                    while count != 0:
                        self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                        index += 1
                        count -= 1
                    rebroadcast = True
                    # else:
                    #     if self.enable_logging:
                    #         logging.log(TRACE, 'rcv:'+str(self.id) +  '#21reset')
                    #     # for _ in range(count):
                    #     #     index = self.reset(index, tmp_local, bid_time)
                    #     index += 1
                    #     rebroadcast = True

                elif z_ij == z_kj:
                
                    if t_kj>t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#22')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                            index += 1
                            count -= 1
                        rebroadcast = True
                    else:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  ' #23 - 24')
                        index+=1
                
                elif z_ij == float('-inf'):
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  '#30')
                    while count != 0:
                        self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)
                        index += 1
                        count -= 1
                    rebroadcast = True

                elif z_ij!=i and z_ij!=k and z_ij!=z_kj:
                    if y_kj>y_ij and t_kj>=t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#25')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)    
                            index += 1
                            count -= 1               
                        rebroadcast = True
                    elif y_kj<y_ij and t_kj<t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#26')
                        rebroadcast = True
                        index+=1
                    elif y_kj==y_ij and z_kj<z_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#27')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], self.bids[self.item['job_id']])                   
                            index += 1
                            count -= 1   
                        rebroadcast = True
                    elif y_kj==y_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#27bis')
                        index+=1
                        rebroadcast = True
                    elif y_kj<y_ij and t_kj>t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#28')
                        while count != 0:
                            self.update_tmp_val(tmp_local, index, self.item['auction_id'][index], self.item['bid'][index], self.item['timestamp'][index], count)   
                            index += 1
                            count -= 1                
                        rebroadcast = True
                        # reset_ids.append(index)
                        # index += 1
                        # reset_flag = True
                        #rebroadcast = True
                    elif y_kj>y_ij and t_kj<t_ij:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#29')
                        # index, reset_flag = self.reset(index, tmp_local)
                        index += 1
                        rebroadcast = True
                    else:
                        if self.enable_logging:
                            logging.log(TRACE, 'rcv:'+str(self.id) +  '#29else')
                        index+=1
                        #rebroadcast = True
                
                else:
                    if self.enable_logging:
                        logging.log(TRACE, 'rcv:'+str(self.id) +  ' #29else2')
                    index+=1
            
            else:
                if self.enable_logging:
                    self.print_node_state('smth wrong?', type='error')

        if reset_flag:
            msg_to_resend = copy.deepcopy(tmp_local)
            #self.forward_to_neighbohors(tmp_local)
            for i in reset_ids:
                # _ = self.reset(i, tmp_local, bid_time - timedelta(days=1))
                _ = self.reset(i, tmp_local, bid_time - 1)
                msg_to_resend['auction_id'][i] = self.item['auction_id'][i]
                msg_to_resend['bid'][i] = self.item['bid'][i]
                msg_to_resend['timestamp'][i] = self.item['timestamp'][i]
                
            self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
            # self.forward_to_neighbohors(msg_to_resend)
            return False             

        cpu = 0
        gpu = 0
        #bw = 0

        first_1 = False
        first_2 = False
        for i in range(len(tmp_local["auction_id"])):
            # if tmp_local["auction_id"][i] == self.id and prev_bet["auction_id"][i] == self.id:
            #     if i != 0 and tmp_local["auction_id"][i-1] != prev_bet["auction_id"][i-1]: 
            #         if self.use_net_topology:
            #             print(f"Failure in node {self.id} job_bid {job_id}. Deconfliction failed. Exiting ...")
            #             raise InternalError
            # el
            if tmp_local["auction_id"][i] == self.id and prev_bet["auction_id"][i] != self.id:
                # self.release_reserved_resources(self.item['job_id'], i)
                cpu -= self.item['NN_cpu']
                gpu -= self.item['NN_gpu']
                if not first_1:
                    #bw -= self.item['NN_data_size'][i]
                    first_1 = True
            elif tmp_local["auction_id"][i] != self.id and prev_bet["auction_id"][i] == self.id:
                cpu += self.item['NN_cpu']
                gpu += self.item['NN_gpu']
                # if self.enable_logging:
                #     self.print_node_state(self.logical_topology.get_updated_bw_matrix(), bid=False, state=False, forward=False)
                
                
                if self.utility == Utility.FGD:
                    self.individual_gpu[self.allocated_on[self.item["job_id"]][i]] += self.item["NN_gpu"][i]
                        
                if not first_2:
                    #bw += self.item['NN_data_size'][i]
                    first_2 = True
                
        self.updated_cpu += cpu
        self.updated_gpu += gpu


        # self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
        

        return tmp_local 
    
    def find_next_contiguous_sequence(self, lst, start):
        """ Find the length of the next contiguous sequence starting from index `start` """
        n = len(lst)
        initial_value = lst[start]
        length = 0
        while start < n and lst[start] == initial_value:
            length += 1
            start += 1
        return length
    
    def count_instances(self, lst, value):
        """Helper function to count instances of a value in a list."""
        return lst.count(value) if value != float('-inf') else 0

    def process_bids(self):
        tmp_winner = {'bid': [float('-inf')] * self.item['N_layer'], 
                      'auction_id': [float('-inf')] * self.item['N_layer'], 
                      'timestamp': [float('-inf')] * self.item['N_layer']}
        index = 0
        rebroadcast = False
        condition = self.id == 8 and self.item['job_id']==18187 and self.time_now == 1
        # if condition:
        # print('check BEFORE!')
        # print(self.bids[self.item['job_id']]['auction_id'])
        # print(self.item['auction_id'])
        # print()


        while index < self.item['N_layer']:
            local = self.bids[self.item['job_id']]['auction_id'][index]
            received = self.item['auction_id'][index]

            if received != local:
                received_w = self.count_instances(self.item['auction_id'], received)
                local_w = self.count_instances(self.bids[self.item['job_id']]['auction_id'], local)

                if local_w == received_w:
                    new_item = self.deconfliction(index, received_w)

                    if index == 0:
                        # keep the first sequence 
                        overwrite = self.item['N_layer'] - index
                        

                        i = index
                        while i < index + overwrite:
                            if new_item['auction_id'][0] == self.bids[self.item['job_id']]['auction_id'][0]:
                                # you win use your sequence
                                tmp_winner['bid'][i] = self.bids[self.item['job_id']]['bid'][i]
                                tmp_winner['auction_id'][i] = self.bids[self.item['job_id']]['auction_id'][i]
                                tmp_winner['timestamp'][i] = self.bids[self.item['job_id']]['timestamp'][i]
                            else:
                                #you lose
                                tmp_winner['bid'][i] = self.item['bid'][i]
                                tmp_winner['auction_id'][i] = self.item['auction_id'][i]
                                tmp_winner['timestamp'][i] = self.item['timestamp'][i]

                                # check if you need to update your resources
                                local = self.bids[self.item['job_id']]['auction_id'][i]
                                received = self.item['auction_id'][i]
                                if (self.bids[self.item['job_id']]['auction_id'][0] != self.id and 
                                    local == self.id and
                                    received != self.id):
                                        self.updated_cpu += self.item['NN_cpu']
                                        self.updated_gpu += self.item['NN_gpu']
                            
                            i += 1
                        index+=overwrite


                    else:
                        # all the sequences are independent and of the same length here
                        i = index
                        while i < index + local_w:
                            # local = self.bids[self.item['job_id']]['auction_id'][i]
                            # if local == self.id:
                            #     self.updated_cpu += self.item['NN_cpu']
                            #     self.updated_gpu += self.item['NN_gpu']
                            tmp_winner['bid'][i]        = new_item['bid'][i]
                            tmp_winner['auction_id'][i] = new_item['auction_id'][i]
                            tmp_winner['timestamp'][i]  = new_item['timestamp'][i]
                            i += 1
                        index += local_w

                    
                elif local_w < received_w:
                    # if self.item['auction_id'][0] == received:
                    
                    # if float('-inf') in self.item['auction_id']:

                    #     if len(self.item['auction_id']) > received_w and  self.item['auction_id'][index + received_w] == float('-inf'):
                    #         #  need to overwrite everything local 
                    #         received_w = self.item['N_layer'] - index
                    
                    overwrite = self.item['N_layer'] - index
                    i = index
                    while i < index + overwrite:
                        local = self.bids[self.item['job_id']]['auction_id'][i]
                        if local == self.id:
                            self.updated_cpu += self.item['NN_cpu']
                            self.updated_gpu += self.item['NN_gpu']
                        tmp_winner['bid'][i] = self.item['bid'][i]
                        tmp_winner['auction_id'][i] = self.item['auction_id'][i]
                        tmp_winner['timestamp'][i] = self.item['timestamp'][i]
                        i += 1
                    index+=overwrite


                        
                else:
   
                    # if received_w == 0 :
                    overwrite = self.item['N_layer'] - index
                    

                    for i in range(index, index + overwrite):
                        tmp_winner['bid'][i] = self.bids[self.item['job_id']]['bid'][i]
                        tmp_winner['auction_id'][i] = self.bids[self.item['job_id']]['auction_id'][i]
                        tmp_winner['timestamp'][i] = self.bids[self.item['job_id']]['timestamp'][i]
                    
                    index += overwrite               
            else:
                tmp_winner['bid'][index] = self.bids[self.item['job_id']]['bid'][index]
                tmp_winner['auction_id'][index] = self.bids[self.item['job_id']]['auction_id'][index]
                tmp_winner['timestamp'][index]  = self.bids[self.item['job_id']]['timestamp'][index]
                index += 1
                    

        self.bids[self.item['job_id']]['bid'] = tmp_winner['bid']
        self.bids[self.item['job_id']]['auction_id'] = tmp_winner['auction_id']
        self.bids[self.item['job_id']]['timestamp'] = tmp_winner['timestamp']

        # if condition:
        # print('check AFTER!')
        # print(self.bids[self.item['job_id']]['auction_id'])
        # print(self.item['auction_id'])

        # if (self.bids[self.item['job_id']]['auction_id'][0] != float('-inf') and
        # self.bids[self.item['job_id']]['auction_id'] == self.item['auction_id']):
        #     rebroadcast = True

        # return rebroadcast
        return True
        
    def update_bid(self):
        if self.enable_logging:
            self.print_node_state('[BEFORE]',bid = True, state= False)
            # logging.log(TRACE, '[BEFORE]')

            
        if 'auction_id' in self.item:       
            # Consensus check
            if  self.bids[self.item['job_id']]['auction_id'] == self.item['auction_id'] and \
                self.bids[self.item['job_id']]['bid'] == self.item['bid'] and \
                self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp'] and \
                float('-inf') not in self.bids[self.item['job_id']]['auction_id']:
                
                if self.enable_logging:
                    self.print_node_state('Consensus -', bid=True)
                    self.bids[self.item['job_id']]['consensus_count']+=1
                    # pass        
            else:   

                rebroadcast = True
                # try to place your bid
                if self.utility == Utility.FGD:
                    success = self.bid_FGD()
                else:
                    success = self.bid()

                # if success or float('-inf') not in self.bids[self.item['job_id']]['auction_id']:
                rebroadcast = self.process_bids()

                # tmp_winner = {
                #     'bid': [],
                #     'auction_id': [],
                #     'timestamp':  []
                # }
                # checking deconfliction
                # index=0
                # n_elements = self.item['N_layer']
                # while index < n_elements:

                #     len_local =    self.find_next_contiguous_sequence(self.bids[self.item['job_id']]['auction_id'], index)
                #     len_received = self.find_next_contiguous_sequence(self.item['auction_id'], index)


                #     if len_local > len_received:
                #         tmp_winner['bid']        .extend(self.bids[self.item['job_id']]['bid']       [index: index+ len_local])
                #         tmp_winner['auction_id'] .extend(self.bids[self.item['job_id']]['auction_id'][index: index+ len_local])
                #         tmp_winner['timestamp']  .extend(self.bids[self.item['job_id']]['timestamp'] [index: index+ len_local])
                #     elif len_local < len_received:
                #         tmp_winner['bid']        .extend(self.item['bid']        [index: index+ len_received])
                #         tmp_winner['auction_id'] .extend(self.item['auction_id'] [index: index+ len_received])
                #         tmp_winner['timestamp']  .extend(self.item['timestamp']  [index: index+ len_received])
                #     else:
                #         self.deconfliction(index, len_received)
                #         tmp_winner['bid']        .extend(self.bids[self.item['job_id']]['bid']       [index: index+ len_local])
                #         tmp_winner['auction_id'] .extend(self.bids[self.item['job_id']]['auction_id'][index: index+ len_local])
                #         tmp_winner['timestamp']  .extend(self.bids[self.item['job_id']]['timestamp'] [index: index+ len_local])
                #     index += len_local  


                # self.bids[self.item['job_id']]['bid'] = tmp_winner['bid']
                # self.bids[self.item['job_id']]['auction_id'] = tmp_winner['auction_id']
                # self.bids[self.item['job_id']]['timestamp'] = tmp_winner['timestamp']

                    
                    
                    
                #     # compare the winners
                #     if received != local:

                #         # count how many instances to allocate
                #         received_w = self.item['auction_id'].count(received) if received != float('-inf') else 0
                #         local_w = self.bids[self.item['job_id']]['auction_id'].count(local) if local != float('-inf') else 0



                            

                #         if local_w < received_w:
                #             # received has the longest sequence
  
                #             i = index
                #             while i < index+received_w:          
                #                 local = self.bids[self.item['job_id']]['auction_id'][i]

                #                 if local == self.id:
                #                     local_w = self.bids[self.item['job_id']]['auction_id'].count(local)
                #                     for k in range(i, i + local_w):
                #                         self.updated_cpu += self.item['NN_cpu']
                #                         self.updated_gpu += self.item['NN_gpu']
                                        
                #                     i += local_w

                #                 else:
                #                     tmp_winner['bid'][i] = self.item['bid'][i]
                #                     tmp_winner['auction_id'][i]= self.item['auction_id'][i]
                #                     tmp_winner['timestamp'][i] = self.item['timestamp'][i]
                #                     i += 1
                                    

                #             # index += i
                #             rebroadcast = True

                #         elif local_w == received_w:
                        
                #             # deconfliction when the instances number is equal
                #             rebroadcast = self.deconfliction(index, received_w)
                #             for k in range(index, index + local_w):
                #                 tmp_winner['bid'][k] = self.bids[self.item['job_id']]['bid'][k]
                #                 tmp_winner['auction_id'][k] = self.bids[self.item['job_id']]['auction_id'][k]
                #                 tmp_winner['timestamp'][k] = self.bids[self.item['job_id']]['timestamp'][k]
                #             index += received_w

                #         # rebroadcast = self.deconfliction(index)
                #     else:
                #         # local has the longest sequence

                #         for i in range(index, index+local_w): 
                #             tmp_winner['bid'][i] = self.bids[self.item['job_id']]['bid'][i]
                #             tmp_winner['auction_id'][i] = self.bids[self.item['job_id']]['auction_id'][i]
                #             tmp_winner['timestamp'][i] = self.bids[self.item['job_id']]['timestamp'][i]

                #         index += local_w

                        
                # self.bids[self.item['job_id']]['bid'] = tmp_winner['bid']
                # self.bids[self.item['job_id']]['auction_id'] = tmp_winner['auction_id']
                # self.bids[self.item['job_id']]['timestamp'] = tmp_winner['timestamp']

                
                    
                # return success or rebroadcast
                return rebroadcast 
        else:
            if self.utility == Utility.FGD:
                self.bid_FGD()
            else:
                self.bid()
            return True

    def get_sequence_length(self, auction_list, start_index):
        current_value = auction_list[start_index]
        length = 0
        for i in range(start_index, len(auction_list)):
            if auction_list[i] == current_value:
                length += 1
            else:
                break
        return length

    def update_bids(self, start_index, received_w):
        i = start_index
        while i < start_index + received_w:
            local = self.bids[self.item['job_id']]['auction_id'][i]
            if local == self.id:
                local_w = self.get_sequence_length(self.bids[self.item['job_id']]['auction_id'], i)
                self.apply_bid_update(i, local_w)
                i += local_w
            else:
                self.apply_bid_update(i, 1)
                i += 1

    def apply_bid_update(self, start_index, length):
        for k in range(start_index, start_index + length):
            self.updated_cpu += self.item['NN_cpu']
            self.updated_gpu += self.item['NN_gpu']
            self.bids[self.item['job_id']]['bid'][k] = self.item['bid'][k]
            self.bids[self.item['job_id']]['auction_id'][k] = self.item['auction_id'][k]
            self.bids[self.item['job_id']]['timestamp'][k] = self.item['timestamp'][k]

    def check_if_hosting_job(self):
        if self.item['job_id'] in self.bids and self.id in self.bids[self.item['job_id']]['auction_id']:
            return True
        return False
    
    def release_resources(self):
        cpu = 0
        gpu = 0
        
        for i, id in enumerate(self.bids[self.item['job_id']]['auction_id']):
            if id == self.id:
                cpu += self.item['NN_cpu']
                gpu += self.item['NN_gpu']
                
        self.updated_cpu += cpu
        self.updated_gpu += gpu
        
        if self.utility == Utility.FGD:
            for n, id in enumerate(self.allocated_on[self.item["job_id"]]):
                self.individual_gpu[id] += self.item["NN_gpu"][n]

    def get_node_res_snapshot(self):
        self.res_snapshot = {}
        self.res_snapshot['gpu'] = self.updated_gpu
        self.res_snapshot['cpu'] = self.updated_cpu
            
    def work(self, time_now, time_global):
        if self.enable_logging:
            logger_method = getattr(logging, 'debug')
            logger_method('\n------------------------ \nID: ' + str(self.id) +
                        ' queue: ' +str(self.q[self.id].qsize()) + ' \n' +
                        'time: '+ str(time_now)+
                        ' CPU: ' + str(self.updated_cpu) +'/'+ str(self.initial_cpu)+
                        ' GPU: ' + str(self.updated_gpu) +'/'+ str(self.initial_gpu)
                        )
        prev_res_cpu = self.updated_cpu
        prev_res_gpu = self.updated_gpu
        
        self.time_now = time_now
        self.time_global = time_global
        # notify_start.set()
        # if self.use_net_topology:
        #     timeout = 15
        # else:
        timeout = 0.000000001

        ret_val = {}
        
        ret_val["id"] = self.id
        ret_val["bids"] = copy.deepcopy(self.bids)
        ret_val["counter"] = self.counter
        ret_val["updated_cpu"] = self.updated_cpu
        ret_val["updated_gpu"] = self.updated_gpu
        ret_val["updated_bw"] = self.updated_bw
        ret_val["gpu_type"] = self.gpu_type.name
        # ret_val["cpu_consumption"] = self.performance.compute_current_power_consumption_cpu(self.initial_cpu-self.updated_cpu)

        self.already_finished = True
        
        # while True:
        #     try: 

        ###################################33
        self.item = None
        items = self.extract_all_job_msg(timeout)  
        if items is None:
            return
        first_msg = False
        need_rebroadcast = False   
        success = False
        
        # self.updated_cpu = round(self.updated_cpu, 3) 
        # self.updated_gpu = round(self.updated_gpu, 3)                  
                            
        # self.empty_queue[self.id].clear() 
        
        for it in items:
            self.item = it

            # if the message is a "unallocate" message, the node must release the resources
            # if the node is hosting the job
            if "unallocate" in self.item:
                if self.check_if_hosting_job():
                    if  self.item['failure'] == True:
                        allocated_gpu = self.res_snapshot['gpu'] - self.updated_gpu
                        allocated_cpu = self.res_snapshot['cpu'] - self.updated_cpu
                    
                    self.release_resources()
                    
                    if  self.item['failure'] == True:
                        assert self.res_snapshot['gpu'] == self.updated_gpu, print('allocated:', allocated_cpu, allocated_gpu)
                        assert self.res_snapshot['cpu'] == self.updated_cpu, print('allocated:', allocated_cpu, allocated_gpu)
                    
                    
                    self.job_hosted.append(self.item['job_id'])
                
                #p_bid = copy.deepcopy(self.bids[self.item['job_id']]["auction_id"])
                
                # if the bidding process didn't complete, reset the bid (it will be submitted later)
                #if float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                del self.bids[self.item['job_id']]
                del self.counter[self.item['job_id']]
                
                #self.update_bw(prev_bid=p_bid, deallocate=True)

                assert self.updated_cpu >= 0
                assert self.updated_gpu >= 0
                    
                ret_val["id"] = self.id
                ret_val["bids"] = copy.deepcopy(self.bids)
                ret_val["counter"] = copy.deepcopy(self.counter)
                ret_val["updated_cpu"] = self.updated_cpu
                ret_val["updated_gpu"] = self.updated_gpu
                ret_val["updated_bw"] = self.updated_bw
                ret_val["gpu_type"] = self.gpu_type.name

            else:   

                
                if self.item['job_id'] in self.bids and \
                    'auction_id' in self.item and \
                    self.bids[self.item['job_id']]['auction_id'] == self.item['auction_id'] and \
                    self.bids[self.item['job_id']]['bid'] == self.item['bid'] and \
                    self.bids[self.item['job_id']]['retry_ts'] !=  time_now: 
                    # self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp']:
                        # print(self.id)


                        # print(time_now, time_global, self.id, it['job_id'], self.bids[self.item['job_id']]['retry'])

                        consume = True
                        while consume:
                            try:
                                if self.q[self.id].empty():
                                    consume = False
                                else:

                                    it = self.q[self.id].get(timeout=timeout)
                                    if self.bids[self.item['job_id']]['auction_id'] != it['auction_id']:
                                        consume = False
                                        self.item = it
                                        
                                        # if self.enable_logging:
                                        #     print('consume', self.id, self.item['job_id'], self.bids[self.item['job_id']]['retry'])

                            except Empty:
                                if len(items) == 0:
                                    # print('node', self.id, 'empty')
                                    return None
                        
                        if self.bids[self.item['job_id']]['retry'] > 10:
                            if self.enable_logging:
                                self.print_node_state('DISCARDED JOB!!!!!!!!!!!!!!!'+str(self.bids[self.item['job_id']]['retry']))
                            return      
                        else:                      
                            if self.enable_logging:
                                self.print_node_state('DISCARDED JOB??????????'+str(self.bids[self.item['job_id']]['retry']))

                
                # if self.item['job_id'] in self.bids:
                #     prev_bid = copy.deepcopy(self.bids[self.item['job_id']]["auction_id"])
                
                # if self.item['job_id'] not in self.counter:
                #     first_msg = True
                    
                if 'user' in self.item and  self.item['user'] == 1111:
                    self.get_node_res_snapshot()
                    self.init_null()
                    self.counter[self.item['job_id']] = 0
                else:
                    self.bids[self.item['job_id']]['retry'] += 1
                    self.bids[self.item['job_id']]['retry_ts'] = time_now
                    
                self.counter[self.item['job_id']] += 1      
                prev_bid = copy.deepcopy(self.bids[self.item['job_id']]['auction_id'])

                    
                if self.enable_logging:
                    self.print_node_state('START', bid = True, state= False)

                success = self.update_bid()

                tries = 0
                bid_ = self.bids[self.item['job_id']]['auction_id']
                bid_now = None
                while (tries < self.item['N_layer'] and
                    self.updated_cpu >= self.item['NN_cpu'] and
                    self.updated_gpu >= self.item['NN_gpu'] and
                    float('-inf') in self.bids[self.item['job_id']]['auction_id'] and
                    self.id not in self.bids[self.item['job_id']]['auction_id'] and
                    bid_ != bid_now):

                        # if self.item['job_id'] == 330:
                        #     print('REBIDDING', self.bids[self.item['job_id']]['auction_id'])
                            
                        success = self.update_bid()
                        bid_now = self.bids[self.item['job_id']]['auction_id']
                        tries += 1

                # Ensure updated CPU and GPU values are within valid ranges
                assert 0 <= self.updated_cpu <= self.initial_cpu, "Updated CPU out of valid range."
                assert 0 <= self.updated_gpu <= self.initial_gpu, "Updated GPU out of valid range."

                # Extract relevant data for clarity and to avoid repetitive lookups
                job_id = self.item['job_id']
                auction_ids = self.bids[job_id]['auction_id']
                nn_cpu = self.item['NN_cpu']
                nn_gpu = self.item['NN_gpu']
                auction_id_count = auction_ids.count(self.id)
                prev_bid_count = prev_bid.count(self.id)

                if self.id not in auction_ids:
                    if self.id in prev_bid:
                        # Calculate expected updated resources based on previous bids
                        expected_cpu = prev_res_cpu + (nn_cpu * prev_bid_count)
                        expected_gpu = prev_res_gpu + (nn_gpu * prev_bid_count)
                    else:
                        # No change in resources if the ID is not in previous bids
                        expected_cpu = prev_res_cpu
                        expected_gpu = prev_res_gpu

                    # Assert the updated resources match the expected values
                    assert self.updated_cpu == expected_cpu, (
                        f"CPU mismatch: expected {expected_cpu}, got {self.updated_cpu}"
                    )
                    assert self.updated_gpu == expected_gpu, (
                        f"GPU mismatch: expected {expected_gpu}, got {self.updated_gpu}"
                    )

                else:
                    if self.id in prev_bid:
                        if auction_id_count == prev_bid_count:
                            # No net change in resource allocation
                            expected_cpu = prev_res_cpu
                            expected_gpu = prev_res_gpu
                        else:
                            # Calculate the difference in resource allocation based on auction ID counts
                            cpu_difference = (nn_cpu * prev_bid_count) - (nn_cpu * auction_id_count)
                            gpu_difference = (nn_gpu * prev_bid_count) - (nn_gpu * auction_id_count)
                            expected_cpu = prev_res_cpu + cpu_difference
                            expected_gpu = prev_res_gpu + gpu_difference
                    else:
                        # Resource allocation decreases based on the current auction ID counts
                        cpu_difference = nn_cpu * auction_id_count
                        gpu_difference = nn_gpu * auction_id_count
                        expected_cpu = prev_res_cpu - cpu_difference
                        expected_gpu = prev_res_gpu - gpu_difference

                    # Assert the updated resources match the expected values
                    assert self.updated_cpu == expected_cpu, (
                        f"CPU mismatch: expected {expected_cpu}, got {self.updated_cpu}"
                    )
                    assert self.updated_gpu == expected_gpu, (
                        f"GPU mismatch: expected {expected_gpu}, got {self.updated_gpu}"
                    )

            
                need_rebroadcast = need_rebroadcast or success

                self.bids[self.item['job_id']]['start_time'] = 0                            
                self.bids[self.item['job_id']]['count'] += 1
                


        return success

            # except Empty:
            #     # the exception is raised if the timeout in the queue.get() expires.
            #     # the break statement must be executed only if the event has been set 
            #     # by the main thread (i.e., no more task will be submitted)

            #     self.empty_queue[self.id].set()
                
            #     all_finished = True
            #     for _, e in enumerate(self.empty_queue):
            #         if not e.is_set():
            #             all_finished = False
            #             break
            #             # print(f"Waiting for node {id} to finish")
                        
            #     if all_finished and not self.already_finished: 
                    
            #         self.already_finished = True   
                    
            #         ret_val["id"] = self.id
            #         ret_val["bids"] = copy.deepcopy(self.bids)
            #         ret_val["counter"] = copy.deepcopy(self.counter)
            #         ret_val["updated_cpu"] = self.updated_cpu
            #         ret_val["updated_gpu"] = self.updated_gpu
            #         ret_val["updated_bw"] = self.updated_bw
            #         ret_val["gpu_type"] = self.gpu_type.name
                        
            #         # for j_key in self.resource_remind:
            #         #     for id in self.resource_remind[j_key]["idx"]:
            #         #         self.release_reserved_resources(j_key, id)
                        
            #         # with self.last_bid_timestamp_lock:
            #         #     if self.use_net_topology:
            #         #         self.updated_bw = self.network_topology.get_node_direct_link_bw(self.id)
                        
            #         # notify the main process that the bidding process has completed and the result has been saved in the ret_val dictionary    
            #         progress_bid.set()

            #     if end_processing.is_set():    
            #         if int(self.updated_cpu) > int(self.initial_cpu):
            #             print(f"Node {self.id} -- Mannaggia updated={self.updated_cpu} initial={self.initial_cpu}", flush=True)
            #         break 

    def extract_one_job_msg(self, timeout):
        first = True
        job_id = None
        items = []
        _items = []
        # while True:
        try:
            print('Extracting', self.id)
            if self.q[self.id].empty():
                return None
            else:
                # print(self.time_global, self.time_now, 'node', self.id, 'qlen:', self.q[self.id].qsize())

                # it = self.q[self.id].get(timeout=timeout)
                it = self.q[self.id].get(timeout=1)

                self.already_finished = False
                if first:
                    first = False
                    job_id = it["job_id"]
                if job_id == it["job_id"]:
                    items.append(it)
                else:
                    _items.append(it)
            # raise Empty
        except Empty:
            if len(items) == 0:
                # print('node', self.id, 'empty')
                return None
                # raise Empty
                    
            # for i in _items:
            #     self.q[self.id].put(i)               
            # break  
            
        return items           
    
    def extract_all_job_msg(self, timeout):
        first = True
        job_id = None
        items = []
        _items = []
        # while True:
        try:
            if self.q[self.id].empty():
                return None
            else:
                # print(self.time_global, self.time_now, 'node', self.id, 'qlen:', self.q[self.id].qsize())

                it = self.q[self.id].get(timeout=timeout)

                self.already_finished = False
                if first:
                    first = False
                    job_id = it["job_id"]
                if job_id == it["job_id"]:
                    items.append(it)
                else:
                    _items.append(it)
            # raise Empty
        except Empty:
            if len(items) == 0:
                # print('node', self.id, 'empty')
                return None
                # raise Empty
                    
            # for i in _items:
            #     self.q[self.id].put(i)               
            # break  
            
        return items           

      
      
