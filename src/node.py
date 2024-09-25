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
import math
# from src.topology import topo as LogicalTopology
# from FGD.src.utils import Quadrant


TRACE = 5    

class BandwidthAllocationError(Exception):
    """Custom exception for bandwidth allocation issues."""
    pass

class InternalError(Exception):
    "Raised when the input value is less than 18"
    pass

class node:

    # def __init__(self, id, max_bw: float, gpu_type: GPUType, utility: Utility, alpha: float, enable_logging: bool, logical_topology, tot_nodes: int, progress_flag: bool, decrement_factor=0.00001, with_bw = False):
    def __init__(self, id, max_bw: float, utility: Utility, alpha: float, enable_logging: bool, logical_topology, tot_nodes: int, progress_flag: bool, decrement_factor=0.00001, with_bw = False):
        self.id = id    # unique edge node id
        # self.gpu_type = gpu_type
        self.gpu_type = GPUType.MISC
        # self.gpu_type = GPUType.T4
        self.utility = utility
        self.alpha = alpha
        self.enable_logging = enable_logging
        self.logical_topology = logical_topology
        self.tot_nodes = tot_nodes
        self.progress_flag = progress_flag
        self.decrement_factor = decrement_factor
        
        self.initial_cpu, self.initial_gpu = GPUSupport.get_compute_resources(self.gpu_type)
        # self.initial_cpu, self.initial_gpu = 200, 20
        self.initial_cpu *= 100
        self.initial_gpu *= 100
        self.updated_gpu = self.initial_gpu
        self.updated_cpu = self.initial_cpu
        self.updated_bw = 0
        self.initial_bw = 0
        self.timestamp_now = 0
        self.time_global = 0
        self.count_msgs = 0


        self.performance = NodePerformance(self.initial_cpu, self.initial_gpu, self.id)
        
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
            "user": int(), 
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
            return (self.initial_gpu - avail_gpu) * corrective_factor
        elif self.utility == Utility.UTIL:
            return self.util_rate()

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
            "ps": self.item["ps"],
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
        
    def gang_schedule(self, count, start, bidtime, connected_to = None):
            # self.with_bw = False
            # gang schedule 
            topoology = self.logical_topology.adj
            # print(topoology)
            # topoology = self.logical_topology.calculate_host_to_host_adjacency_matrix()
            # topoology = self.logical_topology.get_updatede_bw_matrix().copy()
            # if start != 0  and \
            #     topoology[self.id][start-1] and \
            #     topoology[start-1][self.id]:
            #         print("connected")

            if self.item['NN_cpu'] * count <= self.updated_cpu and self.item['NN_gpu'] * count <= self.updated_gpu:
                for i in range(start, start+count):
                    self.layer_bid_already[self.item['job_id']][i] = True

                    if connected_to is not None and topoology[self.id][connected_to] >= self.item['read_count']:
                        # you are directly connected to the previous node
                        res_bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
                        net_bid = topoology[self.id][connected_to]
                        topoology[self.id][connected_to] -= self.item['read_count']

                    else:
                        # you are the first bidder
                        res_bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
                        # net_bid = 0
                        # for k in topoology[self.id]:
                        #     net_bid += k
                        net_bid = max(topoology[self.id])
                        
                    if self.with_bw:
                        bid = int(res_bid * net_bid)
                        # print('bids', self.id, 'bid', bid, res_bid, net_bid)
                    else:
                        bid = int(res_bid)
                        

                    # if bid == 0:
                    #     return

                    

                    # if i == count-1:
                    #     if i == self.item['N_layer']-1:
                    #         # bid = self.updated_gpu
                    #         bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
                    #     else:
                    #         #consider networking towards the best node
                    #         array = self.logical_topology.to()[self.id]

                    #         highest_bw = np.max(array)
                    #         highest_bw_index = np.argmax(array)
                    #         bid = self.updated_gpu / highest_bw

                    # else:
                    #     #gang schedule the hihest number of layers
                    #     bid = self.updated_gpu
                    

                    self.updated_gpu -= self.item['NN_gpu']
                    self.updated_cpu -= self.item['NN_cpu']
                    # if connected_to is not None:
                    #     self.logical_topology.to()[self.id][connected_to] -= self.item['write_count']/1000 if self.item['write_count'] > 1000 else 1

                    self.bids[self.item['job_id']]['bid'][i] = bid
                    self.bids[self.item['job_id']]['auction_id'][i]= self.id
                    self.bids[self.item['job_id']]['timestamp'][i] = bidtime

    def count_layers(self):
        count = 0 
        tmp_cpu = copy.deepcopy(self.updated_cpu)
        tmp_gpu = copy.deepcopy(self.updated_gpu)
        first = True
        # self.with_bw = False
        if float('-inf') in self.bids[self.item['job_id']]['auction_id'] :
            start = self.bids[self.item['job_id']]['auction_id'].index(float('-inf'))

            for i in range(start, start + self.item['N_layer']):
                # if not self.layer_bid_already[self.item['job_id']][i] \
                    
                if self.with_bw:

                    if self.item['NN_gpu'] <= tmp_gpu and\
                        self.item['NN_cpu'] <= tmp_cpu:
                        if start == 0:
                            tmp_cpu-=self.item['NN_cpu']
                            tmp_gpu-=self.item['NN_gpu']
                            count+=1

                        else:
                            if first:
                                # topoology = self.logical_topology.get_updated_bw_matrix().copy()
                                topoology = copy.deepcopy(self.logical_topology.adj)
                                
                                connected_to = self.bids[self.item['job_id']]['auction_id'][0]
                                first = False
                            if topoology[self.id][connected_to]>=self.item['read_count']:
                                count+=1
                                tmp_cpu-=self.item['NN_cpu']
                                tmp_gpu-=self.item['NN_gpu']
                                topoology[self.id][connected_to]-=self.item['read_count']
                
                else:
                    if self.item['NN_gpu'] <= tmp_gpu and\
                        self.item['NN_cpu'] <= tmp_cpu:
                            tmp_cpu-=self.item['NN_cpu']
                            tmp_gpu-=self.item['NN_gpu']
                            count+=1
                        
        return count

    def bid_index(self, count, bidtime):
        # topoology = self.logical_topology.get_updated_bw_matrix()
        topoology = self.logical_topology.adj

        if count>0 and float('-inf') in self.bids[self.item['job_id']]['auction_id']:
            start = self.bids[self.item['job_id']]['auction_id'].index(float('-inf'))
            count = min(count, self.bids[self.item['job_id']]['auction_id'].count(float('-inf')))
            if start != 0:
                # connected_to=self.bids[self.item['job_id']]['auction_id'][start-1] # RING
                connected_to=self.bids[self.item['job_id']]['auction_id'][0] # PS
                if  topoology[self.id][connected_to]>0 and \
                    topoology[connected_to][self.id]>0:
                    # if you are directly connected, then you can bid on the next layer    
                    self.gang_schedule(count, start, bidtime, connected_to)
            else:
                    self.gang_schedule(count, start, bidtime)

    def bid(self):  
        job_GPU_type = GPUSupport.get_gpu_type(self.item['gpu_type'])                     
        # check if node GPU is capable of hosting the job
        if not GPUSupport.can_host(self.gpu_type, job_GPU_type):
            return False
        
        if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) < self.item['speedup'] and self.item["increase"]:
            return False

        if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) > self.item['speedup'] and not self.item["increase"]:
            return False
        
        if GPUSupport.compute_speedup(self.gpu_type, job_GPU_type) == self.item['speedup'] and self.item['job_id'] not in self.job_hosted:
            return False
              
        tmp_bid = copy.deepcopy(self.bids[self.item['job_id']])
        bidtime = self.time_now
        # bidtime = datetime.now()
        
        # create an array containing the indices of the layers that can be bid on
        possible_layer = []
        
        count = self.count_layers()

        # If first time bidding (no received bid yet) then just bid
        if 'bid' not in self.item:
            
            # count how many you can support
            # if count >= self.item['ps']:
                self.gang_schedule(count, 0, bidtime)

        else:
            # if you didn't bid on the first layer yet
            if not self.layer_bid_already[self.item['job_id']][0]:
                # can you outbid the first set?
                winner = self.item['auction_id'].count(self.item['auction_id'][0]) 
                # if count >= self.item['ps'] and count >= winner:
                if count >= winner:
                    self.gang_schedule(count, 0, bidtime)
                
            else:
                # if you did bid on the first set of layer, check if you are directly connected with the previous winner
                # if float('-inf') in self.item['auction_id']:

                #     to_bid = self.item['auction_id'].count(float('-inf'))
                #     if count >= to_bid:
                #         start_local = self.item['auction_id'].index(float('-inf'))
                #         start_bide = self.layer_bid_already[self.item['job_id']].index(False)
                #         if start_local == start_bide:
                #             self.bid_index(count, bidtime)
                #         else:
                #             self.bid_index(start_local, bidtime)    

                if self.id not in self.item['auction_id'] and self.id not in self.bids[self.item['job_id']]['auction_id']:
                    self.bid_index(count, bidtime)
                
        values = self.bids[self.item['job_id']]['auction_id']
        assert all((v >= 0 or v == float('-inf')) for v in values), "All values must be greater than 0 or equal to float('-inf')"

        if self.id in self.bids[self.item['job_id']]['auction_id']:
            return True
        else:
            return False
                    
                
                # if False in self.layer_bid_already[self.item['job_id']]:




        # if self.id not in self.bids[self.item['job_id']]['auction_id']:
        #     for i in range(len(self.layer_bid_already[self.item['job_id']])):
        #         # include only those layers that have not been bid on yet and that can be executed on the node (i.e., the node has enough resources)
        #         if not self.layer_bid_already[self.item['job_id']][i] \
        #             and self.item['NN_gpu'][i] <= self.updated_gpu \
        #                 and self.item['NN_cpu'][i] <= self.updated_cpu:# and self.item['NN_data_size'][i] <= self.updated_bw:
        #             possible_layer.append(i)
        # else:
        #     # if I already own at least one layer, I'm not allowed to bet anymore
        #     # otherwise I break the property of monotonicity
        #     return False
                        
        # # if there are no layers that can be bid on, return
        # # as the iteration goes on, the number of possible layers decreases (i.e., remove the ufeasible layers)
        # while len(possible_layer) > 0:
        #     best_placement = None
        #     best_score = None
            
        #     # iterate on the identify the preferable layer to bid on
        #     for l in possible_layer:
        #         score = self.compute_layer_score(self.item["NN_cpu"][l], self.item["NN_gpu"][l], self.item["NN_data_size"][l])
        #         if best_score == None or score > best_score:
        #             best_score = score
        #             best_placement = l
            
        #     # compute the bid for the current layer, and remove it from the list of possible layers (no matter if the bid is valid or not)
        #     bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
        #     #bid -= self.id * 0.000000001
        #     self.layer_bid_already[self.item['job_id']][best_placement] = True    
        #     possible_layer.remove(best_placement)       

        #     # if my bid is higher than the current bid, I can bid on the layer
        #     if bid > tmp_bid['bid'][best_placement] or (bid == tmp_bid['bid'][best_placement] and self.id < tmp_bid['auction_id'][best_placement]):
                                
        #         gpu_ = self.item['NN_gpu'][best_placement]
        #         cpu_ = self.item['NN_cpu'][best_placement]
        #         #bw_ = self.item["NN_data_size"][best_placement]
                
        #         n_layer = 1
                        
        #         layers = []
                
        #         tmp_bid['bid'][best_placement] = bid
        #         tmp_bid['auction_id'][best_placement] = self.id
        #         tmp_bid['timestamp'][best_placement] = bidtime
                
        #         left_bound = best_placement
        #         right_bound = best_placement
                
        #         # the success is checked at the end of the while loop. If it is true, it means that the bid is valid
        #         success = False 
                                
        #         while True:
        #             # if I already bid on the maximum number of layers, return with success
        #             if n_layer == self.item["N_layer_max"]:
        #                 success = True
        #                 break
                    
        #             left_bound = left_bound - 1                    
        #             right_bound = right_bound + 1
                    
        #             left_score = None
        #             right_score = None
                    
        #             if left_bound >= 0 and self.layer_bid_already[self.item['job_id']][left_bound] == False \
        #                 and self.item['NN_gpu'][left_bound] <= self.updated_gpu - gpu_ \
        #                     and self.item['NN_cpu'][left_bound] <= self.updated_cpu - cpu_:
        #                 left_score = self.compute_layer_score(self.item["NN_cpu"][left_bound], self.item["NN_gpu"][left_bound], self.item["NN_data_size"][left_bound])
                        
        #             if right_bound < len(self.item["NN_cpu"]) and self.layer_bid_already[self.item['job_id']][right_bound] == False \
        #                 and self.item['NN_gpu'][right_bound] <= self.updated_gpu - gpu_ \
        #                     and self.item['NN_cpu'][right_bound] <= self.updated_cpu - cpu_:
                                
        #                 right_score = self.compute_layer_score(self.item["NN_cpu"][right_bound], self.item["NN_gpu"][right_bound], self.item["NN_data_size"][right_bound])
                    
        #             target_layer = None
                    
        #             # proceed on the leyer on the left  
        #             if (left_score is not None and right_score is None) or (left_score is not None and right_score is not None and left_score >= right_score):
        #                 target_layer = left_bound
                        
        #                 # not betting on the right bound layer, so decrease
        #                 right_bound -= 1
                    
        #             # proceed on the layer on the right
        #             if (right_score is not None and left_score is None) or (left_score is not None and right_score is not None and left_score < right_score):
        #                 target_layer = right_bound
                         
        #                 # not betting on the right bound layer, so decrease
        #                 left_bound += 1
                    
        #             # if there is a layer that can be bid on, bid on it    
        #             if target_layer is not None:     
        #                 bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
        #                 #bid -= self.id * 0.000000001
                            
        #                 # if my bid is higher than the current bid, I can bid on the layer
        #                 if bid > tmp_bid['bid'][target_layer] or (bid == tmp_bid['bid'][target_layer] and self.id < tmp_bid['auction_id'][target_layer]):
        #                     tmp_bid['bid'][target_layer] = bid
        #                     tmp_bid['auction_id'][target_layer]=(self.id)
        #                     tmp_bid['timestamp'][target_layer] = bidtime
                        
        #                     n_layer += 1
        #                     layers.append(target_layer)
                            
        #                     cpu_ += self.item['NN_cpu'][target_layer]
        #                     gpu_ += self.item['NN_gpu'][target_layer]
        #                     #bw_ += self.item['NN_data_size'][target_layer] 
        #                 else: # try also on the other side
        #                     found = False
                            
        #                     # we tried on the left bound, let's try on the right one now
        #                     if target_layer == left_bound and right_score is not None:
        #                         target_layer = right_bound + 1
        #                         found = True
                                
        #                     # we tried on the right bound, let's try on the left one now    
        #                     if target_layer == right_bound and left_score is not None:
        #                         target_layer = left_bound - 1
        #                         found = True
                                
        #                     if found:
        #                         bid = self.utility_function(self.updated_bw, self.updated_cpu, self.updated_gpu)
        #                         bid -= self.id * 0.000000001
                                
        #                         # if my bid is higher than the current bid, I can bid on the layer
        #                         if bid > tmp_bid['bid'][target_layer] or (bid == tmp_bid['bid'][target_layer] and self.id < tmp_bid['auction_id'][target_layer]):
        #                             tmp_bid['bid'][target_layer] = bid
        #                             tmp_bid['auction_id'][target_layer]=(self.id)
        #                             tmp_bid['timestamp'][target_layer] = bidtime
                                
        #                             n_layer += 1
        #                             layers.append(target_layer)
                                    
        #                             cpu_ += self.item['NN_cpu'][target_layer]
        #                             gpu_ += self.item['NN_gpu'][target_layer]
        #                             #bw_ += self.item['NN_data_size'][target_layer]
        #                         else:
        #                             if n_layer >= self.item["N_layer_min"] and n_layer <= self.item["N_layer_max"]:
        #                                 success = True
        #                             break
        #                     else:
        #                         if n_layer >= self.item["N_layer_min"] and n_layer <= self.item["N_layer_max"]:
        #                             success = True
        #                         break           
        #             else: 
        #                 if n_layer >= self.item["N_layer_min"] and n_layer <= self.item["N_layer_max"]:
        #                     success = True
        #                 break

                # if success:
                #     self.updated_cpu -= cpu_
                #     self.updated_gpu -= gpu_
                #     #self.updated_bw -= bw_
                    
                #     self.bids[self.item['job_id']] = copy.deepcopy(tmp_bid)
                    
                #     for l in layers:
                #         self.layer_bid_already[self.item['job_id']][l] = True

                    # return True
                              
        # return False     
        # return True  
    
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
        # if self.enable_logging:
        #     self.print_node_state('[BEFORE]',bid = True, state= False)
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
        

    # def process_auctions(self):
    #     index = 0
    #     while index < self.item['N_layer']:
    #         local = self.bids[self.item['job_id']]['auction_id'][index]
    #         received = self.item['auction_id'][index]

    #         if received != local:
    #             received_w = self.get_sequence_length(self.item['auction_id'], index)
    #             local_w = self.get_sequence_length(self.bids[self.item['job_id']]['auction_id'], index)

    #             if local_w == received_w:
    #                 rebroadcast = self.deconfliction(index, received_w)
    #                 index += received_w
    #             elif local_w < received_w:
    #                 self.update_bids(index, received_w)
    #                 rebroadcast = True
    #             else:
    #                 index += local_w
    #                 rebroadcast = True
    #         else:
    #             index += 1
    #     return rebroadcast

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

    # def work(self, end_processing, notify_start, progress_bid, ret_val):

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

            if self.item['job_id'] in self.bids and \
                'auction_id' in self.item and \
                self.bids[self.item['job_id']]['auction_id'] == self.item['auction_id'] and \
                self.bids[self.item['job_id']]['bid'] == self.item['bid'] and \
                self.bids[self.item['job_id']]['retry_ts'] !=  time_now: 
                # self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp']:
                    # print(self.id)

                    self.bids[self.item['job_id']]['retry'] += 1
                    self.bids[self.item['job_id']]['retry_ts'] = time_now

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
                    
                    if self.bids[self.item['job_id']]['retry'] > 5:
                        if self.enable_logging:
                            self.print_node_state('DISCARDED JOB!!!!!!!!!!!!!!!')
                        return                            



            # if the message is a "unallocate" message, the node must release the resources
            # if the node is hosting the job
            if "unallocate" in self.item:
                if self.check_if_hosting_job():
                    self.release_resources()
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
                # prev_bid = None
                first_msg = False
                
                # if self.item['job_id'] in self.bids:
                #     prev_bid = copy.deepcopy(self.bids[self.item['job_id']]["auction_id"])
                
                if self.item['job_id'] not in self.counter:
                    self.init_null()
                    first_msg = True
                    self.counter[self.item['job_id']] = 0
                self.counter[self.item['job_id']] += 1      
                prev_bid = copy.deepcopy(self.bids[self.item['job_id']]['auction_id'])

                    
                if self.enable_logging:
                    self.print_node_state('START', bid = True, state= False)

                # if self.with_networking:
                #     if self.counter[self.item['job_id']] == 1:


                if self.item['job_id'] == 30:
                    print('check')
                success = self.update_bid()



                # if float('-inf') in self.bids[self.item['job_id']]['auction_id'] and \
                #     self.id not in self.bids[self.item['job_id']]['auction_id']:
                #     success = self.update_bid()

                tries = 0
                bid_ = self.bids[self.item['job_id']]['auction_id']
                bid_now = None
                while (tries < self.item['N_layer'] and
                    self.updated_cpu >= self.item['NN_cpu'] and
                    self.updated_gpu >= self.item['NN_gpu'] and
                    float('-inf') in self.bids[self.item['job_id']]['auction_id'] and
                    self.id not in self.bids[self.item['job_id']]['auction_id'] and
                    bid_ != bid_now):

                        success = self.update_bid()
                        bid_now = self.bids[self.item['job_id']]['auction_id']
                        tries += 1

                assert self.updated_cpu >= 0
                assert self.updated_gpu >= 0
                assert self.updated_cpu <= self.initial_cpu
                assert self.updated_gpu <= self.initial_gpu

                if not self.id in self.bids[self.item['job_id']]['auction_id']:
                    if self.id in prev_bid:
                        assert self.updated_cpu == prev_res_cpu + (self.item['NN_cpu'] * prev_bid.count(self.id))
                        assert self.updated_gpu == prev_res_gpu + (self.item['NN_gpu'] * prev_bid.count(self.id))
                    else:
                        assert self.updated_cpu == prev_res_cpu
                        assert self.updated_gpu == prev_res_gpu

                else:
                    if self.id in prev_bid:
                        if self.bids[self.item['job_id']]['auction_id'].count(self.id) == prev_bid.count(self.id):
                            assert prev_res_cpu == self.updated_cpu
                            assert prev_res_gpu == self.updated_gpu
                        else:
                            assert self.updated_cpu == prev_res_cpu +  (self.item['NN_cpu'] * prev_bid.count(self.id)) - (self.item['NN_cpu'] * self.bids[self.item['job_id']]['auction_id'].count(self.id))
                            assert self.updated_gpu == prev_res_gpu +  (self.item['NN_gpu'] * prev_bid.count(self.id)) - (self.item['NN_gpu'] * self.bids[self.item['job_id']]['auction_id'].count(self.id))
                    else:
                        assert self.updated_cpu == prev_res_cpu -  self.item['NN_cpu'] * self.bids[self.item['job_id']]['auction_id'].count(self.id)
                        assert self.updated_gpu == prev_res_gpu -  self.item['NN_gpu'] * self.bids[self.item['job_id']]['auction_id'].count(self.id)
            
                need_rebroadcast = need_rebroadcast or success

                self.bids[self.item['job_id']]['start_time'] = 0                            
                self.bids[self.item['job_id']]['count'] += 1
                
                #self.update_bw(prev_bid)
                
        # if need_rebroadcast:
        #     self.forward_to_neighbohors()
        # elif first_msg:
        #     self.forward_to_neighbohors(first_msg=True)

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

      
      
