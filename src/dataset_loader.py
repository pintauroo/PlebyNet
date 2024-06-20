import random
import csv
import time
import numpy as np
import random
import os

def set_job_list_arrival_time(job_list, arrival_rate=None, interval=60, shuffle_order=False):
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
    for job in job_list:
        #arrival_time = (arrival_counter // arrival_rate) # * interval
        job['submit_time'] = arrival_time
        arrival_counter += 1
    
    return job_list


def poisson_arrivals(job_list):

    total_time = 5000
    total_jobs = 1000
    interval_length = 10
    num_intervals = total_time // interval_length

    # Mean arrival rate per interval
    lambda_per_interval = (total_jobs / total_time) * interval_length

    # Initialize job counter and list to store arrival times
    job_counter = 0
    arrival_times = []

    # Assign jobs to each interval
    for interval in range(num_intervals):
        # Expected number of arrivals in this interval
        expected_arrivals = np.random.poisson(lambda_per_interval)

        # Assign jobs to this interval
        for _ in range(expected_arrivals):
            if job_counter < total_jobs:
                # Assign an arrival time within the interval
                arrival_time = interval * interval_length + np.random.uniform(0, interval_length)
                arrival_times.append(arrival_time)
                job_counter += 1
            else:
                print('limit jobs reached')
                break

        if job_counter >= total_jobs:
            print('limit jobs found')

            break

    # If there are still unassigned jobs, distribute them in the remaining time
    while job_counter < total_jobs:
        arrival_time = np.random.uniform(0, total_time)
        arrival_times.append(arrival_time)
        job_counter += 1

    # Sort the arrival times
    arrival_times.sort()

    # Assign arrival times to jobs
    for i, job in enumerate(job_list):
        job['submit_time'] = int(arrival_times[i])

    
    return job_list


def _add_job(job_list, job_dict, describe_dict=None):
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
            elif key == 'submit_time':  
                job_dict[key] = int(job_dict[key]) - 3
            else:
                job_dict[key] = 1
        else:
            if key in ['num_cpu', 'num_gpu']:  # in %
                job_dict[key] = float(job_dict[key]) #round(100 * float(job_dict[key]))
            else:
                job_dict[key] = round(float(job_dict[key]))
        # if key == 'num_inst':
        #     job_dict[key] = 1


    # Add entries to be used in scheduling
    job_dict['duration'] = int(float(job_dict['duration'])) #if int(float(job_dict['duration']))< 1000 else 1000
    # job_dict['duration'] = 1
    # if job_dict['duration'] <= 0:
    #     job_dict['duration'] = 1  # fix duration == 0 problem.
    job_dict['size'] = int((job_dict['num_gpu'] + job_dict['num_cpu']) * job_dict['duration']) # (gpu + cpu) x duration
    job_dict['on_time'] = 0
    job_dict['wasted'] = 0
    job_dict['jct'] = -1
    job_dict['resource'] = [job_dict['num_gpu'], job_dict['num_cpu']] # list of resources
    job_dict['node'] = None
    job_dict['waiting_for'] = 0
    job_dict['executed_for'] = 0

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
    
    job_dict['allocated_at'] = 0

    if job_dict['num_gpu'] != 0:
        # if job_dict['gpu_type'] == 'MISC':
        #     if job_dict['num_gpu'] <= 8 and job_dict['num_cpu'] <96:
        #         job_list.append(job_dict)
            # else:
            #     print(job_dict)
        # if job_dict['gpu_type'] == 'P100':
        #     if job_dict['num_gpu'] <= 2 and job_dict['num_cpu'] <64:
        #         job_list.append(job_dict)
        #     # else:
        #     #     print(job_dict)
        # if job_dict['gpu_type'] == 'T4':
        #     if job_dict['num_gpu'] <= 2 and job_dict['num_cpu'] <96:
        #         job_list.append(job_dict)
        #     # else:
        #     #     print(job_dict)
        # if job_dict['gpu_type'] == 'V100':
        #     if job_dict['num_gpu'] <= 8 and job_dict['num_cpu'] <96:
        #         job_list.append(job_dict)
            # else:
            #     print(job_dict)
        job_list.append(job_dict)
    


def add_job(csv_file, describe_dict, limit=None):
    """
    limit: To avoid reading too many jobs when the sampled number << total number of jobs in trace file.
    """
    job_list = []
    with open(csv_file, 'r') as fd:
        reader = csv.DictReader(fd, delimiter=',')
        keys = reader.fieldnames
        for i, row in enumerate(reader):
            _add_job(job_list, row, describe_dict)
            # if limit is not None and i >= 1000:
            #     break
    return job_list




def init_go_(num_jobs, arrivals, seed):
    # random.seed(int(time.time()))
    random.seed(int(seed))
    np.random.seed(int(seed))
    current_directory = os.getcwd()
    csv_file=current_directory+'/traces/pai_job_no_estimate_100K.csv'
    # csv_file=str(current_directory)+'/traces/pai/pai_job_no_estimate_100K.csv'
    job_list = add_job(csv_file, None, limit=num_jobs)
    print('job_list size:')
    print(len(job_list))
    if (num_jobs is not None) and num_jobs <= len(job_list):
        random.shuffle(job_list)
        job_list = job_list[:num_jobs]
    # job_list = set_job_list_arrival_time(job_list, arrivals)
    job_list = poisson_arrivals(job_list)
    # print(job_list)
    # sys.exit()
    return job_list



# jobs = init_go()

# df = pd.DataFrame(jobs)

# print(df.describe())
