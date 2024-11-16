import random
import sys
import time
import numpy as np
import pandas as pd
from src.config import SchedulingAlgorithm, ApplicationGraphType

def assign_job_start_time(dataset: pd.DataFrame, time_instant):
    dataset.replace(-1, time_instant, inplace=True)
    return dataset
        
def extract_completed_jobs(dataset: pd.DataFrame, time_instant):
    if len(dataset) == 0:
        return dataset, dataset
    
    condition = dataset.current_duration >= dataset.duration
    ret = dataset[condition].copy()
    ret["complete_time"] = time_instant
    
    if len(ret) > 0:
        dataset = dataset[~condition]
    
    return ret, dataset

def stop_job(dataset: pd.DataFrame, time_instant, job_id):
    if len(dataset) == 0:
        return dataset, dataset
    
    ret = dataset[dataset['job_id'] == job_id].copy()
    ret["complete_time"] = time_instant
    
    if len(ret) > 0:
        dataset = dataset[~(dataset['job_id'] == job_id)]
    
    return ret, dataset
 
def extract_allocated_jobs(dataset: pd.DataFrame, filename):

    if len(dataset) > 0:
        dataset.to_csv(filename)
        # Filter the DataFrame to get only rows where 'allocated_at' is >= 'submit_time'
        # filtered_dataset = dataset[dataset['allocated_at'] >= dataset['submit_time']]
        
        # # Print the 'job_id' for these filtered rows
        # for index, row in filtered_dataset.iterrows():
        #     print(row['job_id'])
        
    

def extract_rebid_job(dataset: pd.DataFrame, low_thre, high_thre, duration_therehold):
    return dataset, pd.DataFrame()
    # if len(dataset) == 0:
    #     return dataset, dataset
    
    # condition = (dataset['speedup'] > high_thre) & (dataset['duration'] - dataset['current_duration'] > duration_therehold)
    # ret = dataset[condition]
    
    # if len(ret) > 0:
    #     dataset = dataset[~condition]
    
    # condition = (dataset['speedup'] < low_thre) & (dataset['duration'] - dataset['current_duration'] > duration_therehold)
    # ret = pd.concat([ret, dataset[condition]])
    
    # if len(ret) > 0:
    #     dataset = dataset[~condition]
    
    # return ret, dataset

def select_jobs(dataset, time_instant):
    return dataset[dataset['submit_time'] == time_instant]

def create_job_batch(dataset, batch_size, time_instant):
    ret = dataset[dataset['submit_time'] <= time_instant]
    dataset.drop(index=dataset.index[:batch_size], axis=0, inplace=True)
    return ret

def schedule_jobs(jobs: pd.DataFrame, scheduling_algorithm: SchedulingAlgorithm):
    if scheduling_algorithm == SchedulingAlgorithm.FIFO:
        return jobs.sort_values(by=["submit_time"])
    elif scheduling_algorithm == SchedulingAlgorithm.SDF:
        return jobs.sort_values(by=["duration"])
    elif scheduling_algorithm == SchedulingAlgorithm.Tiresias:
        print("Tiresias scheduling algorithm !!!!!!!!")
        # if 'waiting_for' not in jobs.columns:
        #     jobs['waiting_for'] = 0

        # # Iterate over each row to check for NaN values individually (if needed for specific logic)
        # for index, job in jobs.iterrows():
        #     # Check if the value is NaN and fill with 0 if so
        #     if pd.isna(job['waiting_for']):
        #         jobs.at[index, 'waiting_for'] = 0
                
        #     if 'executed_for' not in job:
        #         jobs.loc[index, 'executed_for'] = 0
        # Create a new column for the difference between duration and executed_for
        jobs["remaining_duration"] = jobs["duration"] - jobs["executed_for"]

        # Sort by the new column, then by waiting_for, and finally by submit_time
        return jobs.sort_values(by=["remaining_duration", "waiting_for", "submit_time"])



        # return jobs.sort_values(by=["duration"-"executed_for", "waiting_for", "submit_time"])

def dispatch_job(dataset: pd.DataFrame, nmpds: int, read_count, queues, use_net_topology=False, split=False, app_type=ApplicationGraphType.LINEAR, check_speedup=False, low_th=1, high_th=1.2):        
    # if use_net_topology:
    #     timeout = 1 # don't change it
    # else:
    #     timeout = 0.05
 
    for _, job in dataset.iterrows():
        increase = True
        speedup = 0
        if check_speedup:
            speedup = job['speedup']
            if speedup > high_th:
                increase = False
        
        data = message_data(
                    job,
                    # int(dataset['max_pod']),
                    int(nmpds),
                    read_count,
                    deallocate=False,
                    split=split,
                    app_type=app_type,
                    speedup=speedup,
                    increase=increase
                )
        
        # random.seed(job['job_id'])
        # node_to_submit = random.randint(0, len(queues)-1)
        
        for q in queues:
            q.put(data)
        # queues[node_to_submit].put(data)

        #time.sleep(timeout)

def get_simulation_end_time_instant(dataset):
    return dataset['submit_time'].max() + dataset['duration'].max()

def generate_application_graph(layer_number, app_type, bandwidth):
    graph = np.zeros((layer_number, layer_number))
    
    for i in range(layer_number):
        for j in range(i):
            if app_type == ApplicationGraphType.LINEAR:
                if j == i-1:
                    #b = random.uniform(0.5, 1.5)*bandwidth
                    b = bandwidth
                    graph[i][j] = b
                    graph[j][i] = b
            else:
                prob = 0
                if app_type == ApplicationGraphType.GRAPH20:
                    prob = 0.2
                elif app_type == ApplicationGraphType.GRAPH40:
                    prob = 0.4
                elif app_type == ApplicationGraphType.GRAPH60:
                    prob = 0.6
                
                #b = np.random.choice([0, 1], p=[1-prob, prob])*random.uniform(0.5, 1.5)*bandwidth
                b = np.random.choice([0, 1], p=[1-prob, prob])*bandwidth
                graph[i][j] = b
                graph[j][i] = b
                
    return graph        


def message_data(job, nmpds, read_count, failure=False, deallocate=False, split=False, app_type=ApplicationGraphType.LINEAR, speedup=0, increase=True):
    
    random.seed(job['job_id'])
    np.random.seed(int(job['job_id']))

    data = {
        "job_id": job['job_id'],
        "user": 1111,
        "num_gpu": job['num_gpu'],
        "num_cpu": job['num_cpu'],
        "duration": job['duration'],
        "N_layer": int(job['num_pod']),
        "N_layer_min": 1,
        "N_layer_max": nmpds,
        "N_layer_bundle": 2,
        "edge_id": None,
        "NN_gpu": job['num_gpu'],
        "NN_cpu": job['num_cpu'],
        "NN_data_size": job['bw'],
        "gpu_type": job['gpu_type'],
        "increase": increase,
        # "ps": job['ps'],
        "failure": failure,  
        "read_count":  read_count,
        "write_count": read_count,
        "speedup": speedup
    }

    if deallocate:
        data["unallocate"] = True

    return data
