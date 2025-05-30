#!/usr/bin/env python
# coding: utf-8

# In[13]:


import gurobipy as gp
from gurobipy import GRB
import sys
import random
import numpy as np
import time
import random
import pandas as pd
import ast




# In[14]:


def read_prob(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    num_nodes = None
    num_path = None
    edge_weights = None
    start_reading_edges = False
    
   
    for line in lines:
        line = line.strip()
        
        if 'DIMENSION' in line:
            num_nodes = int(line.split()[1])
            edge_weights = np.zeros((num_nodes, num_nodes))
        elif 'N_PATH' in line:
            num_path = int(line.split()[1])
            
        elif 'NODE_COORD_SECTION' in line:
            continue
            
        elif 'EDGE_WEIGHT_SECTION' in line:
            start_reading_edges = True
            continue
            
        elif start_reading_edges:
            if line == '' or line == 'EOF':  
                break
            
            try:
                id_node_1, id_node_2, weight = map(int, line.split())
                edge_weights[id_node_1 - 1, id_node_2 - 1] = weight
            except ValueError:
                print(f"error: {line}")
                continue
    
    return num_nodes, num_path, edge_weights


# In[20]:


def calculate_expected_costs_from_datasets(n, scenarios, num_nodes, num_paths):
    global random_scenarios_dataset


    scenario_data = np.zeros((scenarios, num_nodes, num_nodes, num_paths))
    
    with open("train_set_scenarios.txt", "r") as f:
        valid_range = [int(line.strip()) for line in f]
 

    random_indices = random.sample(valid_range, scenarios)
   
    
    new_entry = {'#': scenarios, 'Scenarios': random_indices}
    random_scenarios_dataset = pd.concat([random_scenarios_dataset, pd.DataFrame([new_entry])], ignore_index=True)



    for idx, s in enumerate(random_indices):
        scenario = filepath + f"Scenario{s}.dat" 
        with open(scenario, 'r') as f:
            lines = f.readlines()[1:] 
        
        index = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_paths):
                    if i < num_nodes and j < num_nodes:
                        scenario_data[idx, i, j, k] = float(lines[index].strip())
                    index += 1
    
    deltas = np.zeros_like(scenario_data) 
    for idx in range(scenarios):
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_paths):
                    deltas[idx, i, j, k] = scenario_data[idx, i, j, k] - edge_weights[i, j]
    
    expected_costs = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            min_deltas = np.min(deltas[:, i, j, :], axis=1)
            expected_costs[i, j] = np.mean(min_deltas)  

    return expected_costs


# In[16]:


def generate_random_solution(num_nodes):
    number = list(range(2, num_nodes + 1))
    random.shuffle(number)
    
    number = [1] + number
    
    
    random_solution = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes - 1):
        random_solution[number[i]-1][number[i + 1]-1] = 1
    random_solution[number[-1]-1][number[0]-1] = 1

    return random_solution, number


# In[17]:


scenarios = [100]
num_experiments = 30
nr_random_sol = 100000
filepath = f'data_prob/'
prob_path = 'prob.txt'
num_nodes, num_path, edge_weights = read_prob(filepath + prob_path)


# In[18]:


def calculate_expected_cost(random_solution, expected_costs):
    expected_cost = 0  

    for i in range(num_nodes):
        for j in range(num_nodes):
            if random_solution[i, j] == 1: 
                expected_cost += expected_costs[i, j] 

    return expected_cost


# In[21]:


for s in scenarios:  
    random_scenarios_dataset = pd.DataFrame(columns=['#', 'Scenarios'])
    
    for n in range(1, num_experiments + 1):
        start_time = time.perf_counter()
        expected_costs = calculate_expected_costs_from_datasets(n, s, num_nodes, num_path)

        training_path = 'dataset_training'

        with open(training_path, 'a') as f: 
            for i in range(nr_random_sol):
                random_sol, route = generate_random_solution(num_nodes)
                result = round(calculate_expected_cost(random_sol, expected_costs), 3)
                f.write(f"{route}; {result}\n")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

    random_scenarios_dataset.to_csv(f"Scenarios for training 105 nodes/scenarios_for_105x{s}.csv", index=False)
