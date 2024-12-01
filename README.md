# PlebyNet
PlebyNet is a network simulation tool designed to allocate distributed jobs across datacenter nodes with customizable topologies. The system currently supports FatTree and Spine-Leaf topologies for modeling network environments. The main focus of the project is on efficient job allocation, bandwidth management, and topology visualization for distributed applications.

This version of PlebyNet extends the functionality to include support for FatTree topology and introduces several improvements to job allocation methods, bandwidth utilization, and visualization.

# Features

### 1. FatTree Topology
Implemented and customizable FatTree topology with adjustable pods, switches, and bandwidth.

### 2. Spine-Leaf Topology
Included an existing implementation for Spine-Leaf topology.

### 3. Bandwidth Allocation
Efficient allocation and deallocation of bandwidth between parameter servers and worker nodes.

### 4. Job Allocation Methods
### Balanced Allocation:
Allocates jobs across nodes in a balanced way.
### Single Node Allocation:
Allocates resources to specific workers.

### 5. Bandwidth Utilization
Tracks and calculates the total bandwidth usage, available bandwidth, and bandwidth utilization percentage across the entire network.

### 6. Visualization:
Visualizes network topology with different node types (core:red, aggregation:blue, edge:green, hosts:yellow) using matplotlib.

### 7. Unit Testing:
Comprehensive unit tests for the FatTree topology, allocation methods, and bandwidth handling.

# Requirements

### Python 3.x
### networkx
### matplotlib
### pytest
### pyyaml

# How to Run the Project

Follow the steps below to set up and run the project:

### 1. Clone the Repository
Fork and clone the repository to your local:

```sh
git clone https://github.com/username/PlebyNet.git
```

### 2. Install Dependencies
Ensure you have all the required Python packages by running:

```sh
pip install -r requirements.txt
```

This will install all necessary libraries such as "networkx" for graph handling and "matplotlib" for visualization.

### 3. Configure Settings

The settings are defined in the config.yaml file. We can customize the configuration by adjusting parameters such as:
### Number of jobs (num_jobs)
### Number of nodes (num_nodes)
### Topology type (topology_type) - Choose between "FatTree" and "SpineLeaf")
### Bandwidth settings (with_bw) - Set to true for bandwidth-limited simulations, or false for infinite bandwidth.
### Pod configuration for FatTree topology (FatTree) - Number of pods and connection details.

### 4. Execute the Script

To run the FatTree topology, execute the following command:

```sh
python3 fat_tree_topology.py
```
The script will initialize a FatTree network with the configuration defined in config.yaml. It will create the topology, allocate bandwidth for the parameter servers and worker nodes, and display statistics.

Run the main script with:

```sh
python3 main.py rep_1
```
This will execute the simulation with the configuration settings specified in the config.yaml file and display results accordingly.

We can run the FatTree topology using command-line arguments with argparse to define the number of pods and the bandwidth for the topology.

### Command-line Arguments:
### --num_pods
The number of pods in the Fat-Tree topology (default: 4).
### --bandwidth
The default bandwidth for connections (default: 10)

To run the script, execute the following command:

```sh
python3 fat_tree_topology.py --num_pods 4 --bandwidth 10
```
This will initialize a FatTree network with the number of pods and bandwidth as specified by the command-line arguments. If no arguments are passed, it will default to 4 pods and a bandwidth of 10.

The script will create the topology, allocate bandwidth for the parameter servers and worker nodes, and display statistics.

### 5. Visualizing the Topology

After the execution, the script will generate a "matplotlib" visualization of the FatTree topology, showing different node types (core, aggregation, edge switches, and hosts) and their connections.

### 6. Unit Tests

Unit tests are located in "test_fat_tree_topology.py" and ensure the correctness of the FatTree topology, allocation methods, and bandwidth management.

Run the tests with:
```sh
python3 -m pytest test_fat_tree_topology.py
```
### 





