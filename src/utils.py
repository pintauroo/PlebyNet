"""
This module contains utils functions to calculate all necessary stats
"""
from csv import DictWriter
import os
import logging
import sys
import time
import pandas as pd
import numpy as np
from src.config import *



import math

def generate_gpu_types(n_nodes):
    occurrencies = [0.3, 0.17, 0.47, 0.06]
    GPU_types = ["T4", "MISC", "P100", "V100"]
    np.random.seed(1)
    
    gpu_types = []
    for _ in range(n_nodes):
        # t_id = np.random.choice(np.arange(0, 4), p=occurrencies)
        # gpu_types.append(GPUSupport.get_gpu_type(GPU_types[t_id])) GPUType.MISC
        gpu_types.append(GPUType.MISC) 
        
    return gpu_types
    

def wrong_bids_calc(nodes, job, num_edges, use_net_topology):
    
    j = job['job_id']
    #print('\n[WRONG BID]' + str(j))
    wrong_bids=[] # used to not replicate same action over different nodes
    wrong_ids=[]
    equal_values=True
    for curr_node in range(0, num_edges):
        if nodes[curr_node].bids[j]['auction_id'] not in wrong_bids:
            for i in range(1, num_edges):
                if nodes[i].bids[j]['auction_id'] != nodes[i-1].bids[j]['auction_id']:
                    equal_values = False
                    break
            if not equal_values:
                pass


    for curr_node in range(0, num_edges):
        if nodes[curr_node].bids[j]['auction_id'] not in wrong_bids:
            if all(x == float('-inf') for x in nodes[curr_node].bids[j]['auction_id']):
                continue
            else:

                if curr_node in nodes[curr_node].bids[j]['auction_id'] and curr_node not in wrong_ids:
                    
                    wrong_ids.append(curr_node)
                    # first_time = True
                    # index=0
                    # while index<len(nodes[curr_node].bids[j]['auction_id']):
                    #     if nodes[curr_node].bids[j]['auction_id'][index] == curr_node and id != float('-inf'):
                    #         nodes[curr_node].updated_cpu += float(job['num_cpu']) / float(len(nodes[curr_node].bids[j]['auction_id']))
                    #         nodes[curr_node].updated_gpu += float(job['num_gpu']) / float(len(nodes[curr_node].bids[j]['auction_id']))
                    #         if first_time:
                    #             nodes[curr_node].updated_bw += float(job['bw']) / float(len(nodes[curr_node].bids[j]['auction_id']))
                    #             first_time = False
                    #     index += 1
        else:
            continue
            
    if use_net_topology:
        # release network resources between client and node        
        for curr_node in range(0, num_edges):
            for i, n_id in enumerate(nodes[curr_node].bids[j]['auction_id']):
                if i == 0 and n_id == curr_node:
                    network_t.release_bandwidth_node_and_client(curr_node, float(job['bw']) / float(len(nodes[curr_node].bids[j]['auction_id'])), j)
                    
        # release network resources between nodes        
        for curr_node in range(0, num_edges):
            prev_val = nodes[curr_node].bids[j]['auction_id'][0]
            for i, n_id in enumerate(nodes[curr_node].bids[j]['auction_id']):
                if i != 0:
                    if prev_val != n_id and n_id == curr_node:
                        network_t.release_bandwidth_between_nodes(curr_node, prev_val, float(job['bw']) / float(len(nodes[curr_node].bids[j]['auction_id'])), j)
                    prev_val = nodes[curr_node].bids[j]['auction_id'][i]

def allocation_to_gpu_type(allocation, gpu_types):
        ret = []
        for a in allocation:
            ret.append(gpu_types[a].name)
        return ret

def calculate_utility(nodes, num_edges, jobs, time_instant, filename, gpu_types, save_on_file):
    stats = {}
    stats['nodes'] = {}
    stats['tot_utility'] = 0
    
    #field_names = ['n_nodes', 'n_req', 'exec_time', 'alpha']
    #dictionary = {'n_nodes': num_edges, 'n_req' : n_req, 'exec_time': simulation_time, 'alpha': alpha}
    dictionary = {}
    field_names = []

    # ---------------------------------------------------------
    # calculate assigned jobs, update resources if job not assigned
    # ---------------------------------------------------------

    count_assigned = 0
    count_unassigned = 0
    assigned_sum_cpu = 0
    assigned_sum_gpu = 0
    assigned_sum_bw = 0
    unassigned_sum_cpu = 0
    unassigned_sum_gpu = 0
    unassigned_sum_bw = 0

    count = 0 
    assigned_jobs = []
    unassigned_jobs = []
    assigned_jobs_id = []
    unassigned_job_id = []
    valid_bids = {}
    count_success = 0
    
    for _, job in jobs.iterrows():
        count += 1
        flag = True
        j = job['job_id']
        node_with_bid = None
        n_layer = 0
        GPUs = []
        unmatch = False
        
        for n in nodes:
            if j in n.bids:
                n_layer = len(n.bids[j]['auction_id'])
                break
        
        # Check correctness of all bids
        for k in range(n_layer):
            unmatch = False
            alloc = None
            for i in range(0, num_edges):
                if j not in nodes[i].bids:
                    continue
                if alloc == None:
                    alloc = nodes[i].bids[j]['auction_id'][k]
                    node_with_bid = i
                else:
                    if alloc != nodes[i].bids[j]['auction_id'][k]:
                        unmatch = True
                        break
                    
            if unmatch:
                print('BROKEN BID id: ' + str(j))
                for n in nodes:
                    if j in n.bids:
                        print(f"Node: {n.id}: {n.bids[j]['auction_id']}")
                # something bad happened
                break
            
            if alloc != float('-inf'):
                GPUs.append(nodes[alloc].gpu_type)
                
        if node_with_bid != None and float('-inf') not in nodes[node_with_bid].bids[j]['auction_id'] and not unmatch:
            count_success += 1
            valid_bids[j] = nodes[node_with_bid].bids[j]['auction_id']
            logging.info(f"Job {j} assignment {nodes[node_with_bid].bids[j]['auction_id']}")
        else:
            flag = False 

        if flag:
            job["final_node_allocation"] = nodes[node_with_bid].bids[j]['auction_id']
            job["final_gpu_allocation"] = allocation_to_gpu_type(nodes[node_with_bid].bids[j]['auction_id'], gpu_types=gpu_types)
            
            lower_speedup = 10000
            for g in set(GPUs):
                s = GPUSupport.compute_speedup(GPUSupport.get_gpu_type(g), GPUSupport.get_gpu_type(job["gpu_type"]))
                if lower_speedup > s:
                    lower_speedup = s
            
            job["speedup"] = lower_speedup
            
            assigned_jobs.append(job)
            assigned_jobs_id.append(j)
            
            count_assigned += 1
            assigned_sum_cpu += float(job['num_cpu'])
            assigned_sum_gpu += float(job['num_gpu'])
            assigned_sum_bw += float(job['bw']) 
        else:
            unassigned_jobs.append(job)
            # for k in range(num_edges):
            #     print(nodes[k].bids[j]['auction_id'])
            unassigned_job_id.append(j)
            count_unassigned += 1
            unassigned_sum_cpu += float(job['num_cpu'])
            unassigned_sum_gpu += float(job['num_gpu'])
            unassigned_sum_bw += float(job['bw']) 
            
    # if use_net_topology:
    #     print()
    #     net_topology.check_network_consistency(valid_bids)
            
    #print(f"Count assigned {count_assigned} count unassigned {count_unassigned}")    
    #field_names.append('count_assigned')
    #field_names.append('count_unassigned')
    field_names.append('time_instant')
    #dictionary['count_assigned'] = round(count_assigned,2)
    #dictionary['count_unassigned'] = round(count_unassigned,2)
    dictionary['time_instant'] = time_instant

    tot_used_bw = 0
    tot_used_cpu = 0
    tot_used_gpu = 0
    tot_gpu_nodes = 0
    tot_cpu_nodes = 0
    tot_bw_nodes = 0

    # ---------------------------------------------------------
    # calculate node utility, assigned jobs and used res
    # ---------------------------------------------------------
    for i in range(num_edges):
        # field_names.append('node_'+str(i)+'_jobs')
        # field_names.append('node_'+str(i)+'_utility')
        field_names.append('node_'+str(i)+'_initial_gpu')
        #field_names.append('node_'+str(i)+'_updated_gpu')
        field_names.append('node_'+str(i)+'_used_gpu')
        field_names.append('node_'+str(i)+'_initial_cpu')
        #field_names.append('node_'+str(i)+'_updated_cpu')
        field_names.append('node_'+str(i)+'_used_cpu')
        field_names.append('node_'+str(i)+'_initial_bw')
        #field_names.append('node_'+str(i)+'_updated_bw')
        field_names.append('node_'+str(i)+'_used_bw')
        field_names.append('node_'+str(i)+'_gpu_type')
        #field_names.append('node_'+str(i)+'_cpu_consumption')
        #field_names.append('node_'+str(i)+'_gpu_consumption')

        tot_gpu_nodes += round(nodes[i].initial_gpu,2)
        dictionary['node_'+str(i)+'_initial_gpu'] = round(nodes[i].initial_gpu,2)
        #dictionary['node_'+str(i)+'_updated_gpu'] = round(nodes[i].updated_gpu,2)
        dictionary['node_'+str(i)+'_used_gpu'] = 0 if math.isclose(nodes[i].initial_gpu - nodes[i].updated_gpu, 0.0, abs_tol=1e-1) else round(nodes[i].initial_gpu - nodes[i].updated_gpu, 2)

        tot_used_gpu += dictionary['node_'+str(i)+'_used_gpu']

        tot_cpu_nodes += round(nodes[i].initial_cpu,2)
        dictionary['node_'+str(i)+'_initial_cpu'] = round(nodes[i].initial_cpu,2)
        #dictionary['node_'+str(i)+'_updated_cpu'] = round(nodes[i].updated_cpu,2)
        dictionary['node_'+str(i)+'_used_cpu'] = 0 if math.isclose(nodes[i].initial_cpu - nodes[i].updated_cpu, 0.0, abs_tol=1e-1) else round(nodes[i].initial_cpu - nodes[i].updated_cpu,2)

        tot_used_cpu += dictionary['node_'+str(i)+'_used_cpu']

        tot_bw_nodes += round(nodes[i].initial_bw,2)
        dictionary['node_'+str(i)+'_initial_bw'] = round(nodes[i].initial_bw,2)
        #dictionary['node_'+str(i)+'_updated_bw'] = round(nodes[i].updated_bw,2)
        dictionary['node_'+str(i)+'_used_bw'] = 0 if math.isclose(nodes[i].initial_bw - nodes[i].updated_bw, 0.0, abs_tol=1e-1) else round(nodes[i].initial_bw - nodes[i].updated_bw,2)

        tot_used_bw += dictionary['node_'+str(i)+'_used_bw']
        dictionary['node_'+str(i)+'_gpu_type'] = nodes[i].gpu_type
        
        #dictionary['node_'+str(i)+'_cpu_consumption'] = round(nodes[i].performance.compute_current_power_consumption_cpu(nodes[i].initial_cpu-nodes[i].updated_cpu), 2)
        #dictionary['node_'+str(i)+'_gpu_consumption'] = round(nodes[i].performance.compute_current_power_consumption_gpu(nodes[i].initial_gpu-nodes[i].updated_gpu), 2)

        #calculate node assigned count and utility
        # for j, job_id in enumerate(nodes[i].bids):
            
        #     if job_id in assigned_jobs_id and nodes[i].id in nodes[i].bids[job_id]['auction_id']:
        #         stats['nodes'][nodes[i].id]['assigned_count'] += 1

        #     # print(nodes[i].bids[job_id])
        #     for k, auctioner in enumerate(nodes[i].bids[job_id]['auction_id']):
        #         # print(nodes[i].id)
        #         if job_id in assigned_jobs_id and auctioner== nodes[i].id:
        #             # print(nodes[i].bids[job_id]['bid'])
        #             stats['nodes'][nodes[i].id]['utility'] += nodes[i].bids[job_id]['bid'][k]

        #print('node: ' + str(nodes[i].id) + ' utility: ' + str(stats['nodes'][nodes[i].id]['utility']))
        # dictionary['node_'+str(i)+'_utility'] = round(stats['nodes'][nodes[i].id]['utility'],2)
        # stats["tot_utility"] += stats['nodes'][nodes[i].id]['utility']

    # for i in stats['nodes']:

    #     #print('node: '+ str(i) + ' assigned jobs count: ' + str(stats['nodes'][i]['assigned_count']))
    #     dictionary['node_'+str(i)+'_jobs'] = round(stats['nodes'][i]['assigned_count'],2)

    if save_on_file:        
        write_data(field_names, dictionary, filename)
    
    return assigned_jobs, unassigned_jobs


def write_data(field_names, dictionary, filename):
    filename = str(filename)+'.csv'

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f: 
        writer = DictWriter(f, fieldnames=field_names)
    
        # Pass the dictionary as an argument to the Writerow()
        if not file_exists:
            writer.writeheader()  # write the column headers if the file doesn't exist
    
        writer.writerow(dictionary)    



def jaini_index(dictionary, num_nodes):
    data=[]
    for i in range(num_nodes):
        data.append(dictionary['node_'+str(i)+'_jobs'])

    sum_normal = 0
    sum_square = 0

    for arg in data:
        sum_normal += arg
        sum_square += arg**2

    if len(data) == 0 or sum_square == 0:
        return 1

    return sum_normal ** 2 / (len(data) * sum_square)

