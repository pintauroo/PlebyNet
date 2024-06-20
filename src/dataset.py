"""
Module to extract a subset of jobs from the main dataset
"""

import csv
import pandas as pd
import copy
import random
import numpy as np
import sys

pd.options.display.max_columns = None
pd.options.display.max_rows = None

class JobList:
    def __init__(self, csv_file, num_jobs_limit, seed):
        if seed is not None:
            random.seed(seed)
        
        self.job_list = []
        self.describe_dict = None
        self.job_origin_list = self.add_job(csv_file, self.describe_dict, limit=num_jobs_limit * 100)
        self.csv_file = csv_file
        self.arrival_rate = 1000
        self.arrival_interval = 60
        self.arrival_shuffle = False
        self.num_jobs_limit = num_jobs_limit
        self.num_jobs = None


    def add_job(self, csv_file, describe_dict, limit=None):
        """
        limit: To avoid reading too many jobs when the sampled number << total number of jobs in trace file.
        """
        job_list = []
        with open(csv_file, 'r') as fd:
            reader = csv.DictReader(fd, delimiter=',')
            keys = reader.fieldnames
            for i, row in enumerate(reader):
                #if float(row['num_gpu']) != float(0):
                    # print(row['num_gpu'])
                self._add_job(job_list, row, describe_dict)
                if limit is not None and i >= limit:
                    break
        return job_list
        
    def _add_job(self, job_list, job_dict, describe_dict=None):
        # Add job (job_dict) into job_list
        for key, value in job_dict.items():
            if value is not None and value.isdigit() and key != 'user':
                if type(value) == str:
                    job_dict[key] = round(float(value))
                else:  # duration becomes an int
                    job_dict[key] = round(value)
            elif key in ['wait_time','user_dur','user_gpu_dur','group_dur','group_gpu_dur']:
                try:
                    job_dict[key] = float(value)
                except:
                    pass

        keys = ['num_cpu', 'num_gpu', 'submit_time', 'num_inst']
        for key in keys:
            if key not in job_dict or job_dict[key] == '':
                if key in ['num_cpu', 'num_gpu']:
                    job_dict[key] = 0
                else:  # key in ['submit_time', 'num_inst']
                    job_dict[key] = 1
            else:
                if key in ['num_cpu', 'num_gpu']:  # in %
                    job_dict[key] = round(100 * float(job_dict[key]))
                else:
                    job_dict[key] = round(float(job_dict[key]))

        # Add entries to be used in scheduling
        job_dict['duration'] = int(float(job_dict['duration']))
        if job_dict['duration'] <= 0:
            job_dict['duration'] = 1  # fix duration == 0 problem.
        job_dict['size'] = int((job_dict['num_gpu'] + job_dict['num_cpu']) * job_dict['duration']) # (gpu + cpu) x duration
        job_dict['on_time'] = 0
        job_dict['wasted'] = 0
        job_dict['jct'] = -1
        job_dict['resource'] = [job_dict['num_gpu'], job_dict['num_cpu']] # list of resources
        job_dict['node'] = None
        job_dict['execution_time'] = 0


        # Add duration estimation
        if describe_dict is not None:
            jd_user = describe_dict.get(job_dict['user'])
            if jd_user is not None:
                job_dict['dur_avg'] = float(jd_user['mean'])  # expectation
                job_dict['dur_std'] = float(jd_user['std'])  # standard deviation
                job_dict['dur_med'] = float(jd_user['50%'])  # median
                job_dict['dur_trim_mean'] = float(jd_user['trim_mean'])  # discard 10% top and 10% tail when calc. mean

        # Remove original unused entries
        for drop_col in ['fuxi_job_name','fuxi_task_name','inst_id','running_cluster','model_name','iterations','interval','vc','jobid','status']:
            if drop_col in job_dict: job_dict.pop(drop_col)

        job_list.append(job_dict)


    def select_jobs(self):
        self.cur_time = 0
        self.job_list = copy.deepcopy(self.job_origin_list)  # copy each obj in the list
        num_jobs = self.num_jobs if self.num_jobs is not None else self.num_jobs_limit
        if (num_jobs is not None) and num_jobs <= len(self.job_list):
            random.shuffle(self.job_list)
            self.job_list = self.job_list[:num_jobs]
        self.set_job_list_arrival_time(self.job_list, self.arrival_rate, self.arrival_interval, self.arrival_shuffle)


    def set_job_list_arrival_time(self, job_list, arrival_rate=None, interval=60, shuffle_order=False):
        """
        job_list: jobs to execute in this run
        arrival_rate: num of jobs to arrive at each time interval (-1 or None means no changes)
        interval: time interval (default: 60)
        shuffle_order: bool, whether each user's inherent job order are shuffled (default: False)
        """
        if arrival_rate is None or arrival_rate < 0:
            return 0  # respect the original submit time
        if shuffle_order is True:
            np.random.shuffle(job_list)
        else:
            job_list.sort(key=lambda e: (e.get('submit_time', float('inf')), e['job_id']))

        arrival_counter = 0
        id_map = {}
        next_id = 0
        for job in job_list:
            user_id = job["user"]
            if user_id not in id_map:
                id_map[user_id] = next_id
                next_id += 1
            arrival_time = (arrival_counter // arrival_rate) * interval
            job['submit_time'] = arrival_time
            arrival_counter += 1
        
        for job in job_list:
            user_id = job["user"]
            job["user"] = id_map[user_id]



def main():
    #Data analysis
    dataset='./df_dataset.csv'
    req_number = int(sys.argv[1]) #Total number of requests
    min_cpu_gpu_ratio=10
    max_cpu_gpu_ratio=15

    job_list_instance = JobList(dataset, num_jobs_limit=req_number, min_cpu_gpu_ratio=min_cpu_gpu_ratio, max_cpu_gpu_ratio=max_cpu_gpu_ratio)
    job_list_instance.select_jobs()
    print(job_list_instance.job_list)
    #job_dict = {job['job_id']: job for job in job_list_instance.job_list} # to find jobs by id
    print('jobs number = ' + str(len(job_list_instance.job_list)))

    filename = 'dataset_'+str(req_number)+'_jobs_ratio_'+str(min_cpu_gpu_ratio)+'_'+str(max_cpu_gpu_ratio)+'.csv'
    data = job_list_instance.job_list
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Use the main to customize your dataset
# main()