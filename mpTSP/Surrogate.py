#!/usr/bin/env python
# coding: utf-8

# # Surrogate model

# In[15]:


import gurobipy as gp
import numpy as np
import sys
from gurobipy import GRB
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from gurobi_ml import add_predictor_constr
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import itertools
import random
import os
from joblib import dump, load
import time
import joblib
import resource
import re

from itertools import product
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers


# In[2]:


scenarios = [100]
num_experiments = 30


# In[3]:


def read_prob(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    num_nodes = None
    num_path = None
    edge_weights = None
    node_coordinates = {}
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


# In[4]:


def read_dataset_from_file(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence_part, cost_part = line.strip().split(";")
            sequence = list(map(int, sequence_part.strip("[]").split(",")))
            cost = float(cost_part.strip())
            dataset.append((sequence, cost))
    return dataset


def transform_to_binary_oriented(dataset):
    num_nodes = max(max(row[0]) for row in dataset)
    edges = [(i, j) for i in range(1, num_nodes + 1) for j in range(1, num_nodes + 1) if i != j]
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    
    X = []
    y = []
    for sequence, cost in dataset:
        sequence.append(1)
        
        binary_representation = [0] * len(edges)
        for i in range(len(sequence) - 1):
            edge = (sequence[i], sequence[i + 1])
            binary_representation[edge_to_index[edge]] = 1
        
        X.append(binary_representation)
        y.append(cost)
    
    return np.array(X), np.array(y)


# In[16]:


for s in scenarios:    
    output_file = f'Results/output_surrogate_105x{s}.txt'
    
    with open(output_file, 'a') as f:
        f.write("Key, Route, ObjVal, Eval_cost, Time_Sec, CPU_Time_Sec\n")
            
        for n in range(1, num_experiments +1):
            file_path = f"Training data 105 prova/105X{s}_{n}.txt"
    
            dataset = read_dataset_from_file(file_path)
            X, Y = transform_to_binary_oriented(dataset)
    




    
            def create_dense_model(input_dim):
                model = Sequential([
                    Dense(16, activation='relu', input_dim=input_dim),
                    Dense(8, activation='relu'),
                    #Dense(8, activation='relu'),
                    Dense(1)
                ])
                return model

       

        

            Y = Y.reshape(-1, 1)   
    
            num_samples = len(X)
            split_ratio = 0.8
            split_index = int(num_samples * split_ratio)
    
            X_train, X_test = X[:split_index], X[split_index:]
            Y_train, Y_test = Y[:split_index], Y[split_index:]
    
            input_dim = X.shape[1]
            NN_model = create_dense_model(input_dim)
            NN_model.compile(optimizer=Adam(learning_rate=0.01), loss='mae')
    
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
            num_epochs = 1000
    
            start_time = time.perf_counter()


            batch_size = 256
    
            NN_model.fit(X_train, Y_train, 
                         epochs=num_epochs, 
                         batch_size=batch_size,  
                         validation_data=(X_test, Y_test), 
                         callbacks=[early_stopping, reduce_lr],
                         verbose=1)
    
    
            training_time = time.perf_counter() - start_time
    
            y_pred = NN_model.predict(X_test)
            y_pred_train = NN_model.predict(X_train)


            r2_score_train = metrics.r2_score(Y_train, y_pred_train)
            r2_score = metrics.r2_score(Y_test, y_pred)
            mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
            mape_train = np.mean(np.abs((Y_train - y_pred_train) / Y_train)) * 100
            mae = metrics.mean_absolute_error(Y_test, y_pred)
    
            print(f"Scenario {s}, Experiment {n}:")
            print(f"R2 Score (Test): {r2_score:.2f}")
            print(f"MAPE (Test): {mape}%")
            print(f"MAE (Test): {mae}%")
            print(f"training time:{training_time}")
                    #print(f"R2 Score (Train): {r2_score_train:.2f}")
                    #print(f"MAPE (Train): {mape_train:.2f}%")
                    #print(f"Tempo di allenamento: {training_time:.2f} secondi")
            #NN_model.save(f"NN Models 105 nodes/NN_model_105x{s}_{n}.keras")




            with open(output_file, 'a') as f:
                    f.write("Key, Route, ObjVal, Eval_cost, Time_Sec, CPU_Time_Sec\n")

            nn_model = NN_model
            route, res, total_cost, resolution_time, cpu = solve_mpTSP_with_nn(edge_weights, num_nodes, num_path, nn_model)

            f.write(f"{n},{route},{res},{total_cost},{resolution_time},{cpu}\n")

            print(s, n, route, res, total_cost, resolution_time, cpu)




# In[5]:


def read_scenario(filepath, num_nodes, num_path):
    
    scenario_weights = np.zeros((num_nodes, num_nodes, num_path))
    
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:] 
    
    index = 0
    for i in range(num_nodes): 
        for j in range(num_nodes):
            for k in range(num_path):
                
                if i < num_nodes and j < num_nodes:
                    scenario_weights[i, j, k] = float(lines[index].strip())
                index += 1  
    
    return scenario_weights


# In[6]:


def calculate_deltas(scenario_weights):
    
    deltas = np.zeros_like(scenario_weights)
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_path):
                deltas[i, j, k] = scenario_weights[i, j, k] - edge_weights[i, j]
    
    #deltas = np.min(deltas, axis=2)
    return deltas


# In[7]:


def solve_recourse(solution_x, deltas, num_nodes, num_path):
    model = gp.Model("Second_Stage_Problem")

    
    y = model.addVars(num_nodes, num_nodes, num_path, vtype=GRB.BINARY, name="y")

    
    model.setObjective(
        gp.quicksum(deltas[i, j, k] * y[i, j, k] 
                      for i in range(num_nodes) for j in range(num_nodes) for k in range(num_path)),
        sense=GRB.MINIMIZE
    )

    
    model.addConstrs(
        (gp.quicksum(y[i, j, k] for k in range(num_path)) == solution_x[i][j] for i in range(num_nodes) for j in range(num_nodes)),
        name="select_path")
    
  
    model.setParam('OutputFlag', 0)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        res = model.objVal
        return res 


# In[8]:


with open("test_set_scenarios.txt", "r") as f:
        valid_range = [int(line.strip()) for line in f]


# In[9]:


def eval_first(first_stage_solution, edge_weights, num_nodes):
    return sum(edge_weights[i,j] * first_stage_solution[i][j] for i in range (num_nodes) for j in range(num_nodes))


# In[10]:


def eval_second(first_stage_solution, num_nodes, num_path):
    second = 0
    
    for i in valid_range:
        filepath = f"data_prob/Scenario{i}.dat"
        scenario_weights = read_scenario(filepath, num_nodes, num_path)
        deltas = calculate_deltas(scenario_weights)
        second += solve_recourse(first_stage_solution, deltas, num_nodes, num_path)
    return second/500       


# In[11]:


def sequence_to_adjacency_matrix(sequence):
    n = max(sequence)
    
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(len(sequence) - 1):
        adjacency_matrix[sequence[i] - 1][sequence[i + 1] - 1] = 1
    
    adjacency_matrix[sequence[-1] - 1][sequence[0] - 1] = 1
    
    return adjacency_matrix




# In[12]:


def solve_mpTSP_with_nn(edge_weights, num_nodes, num_path, model_nn):
    model = gp.Model("mpTSP_with_NN")

    x = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="x")
    phi = model.addVars(num_nodes, num_nodes, vtype=GRB.CONTINUOUS, name="phi")
    cost_surrogate = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="cost_surrogate")
    binary_sequence = model.addVars(num_nodes * (num_nodes - 1), vtype=GRB.BINARY, name="binary_sequence")

    model.setObjective(
        gp.quicksum(edge_weights[i, j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes) if j != i) +
        cost_surrogate,  
        GRB.MINIMIZE
    )

    model.addConstrs(
        gp.quicksum(x[i, j] for j in range(num_nodes) if j != i) == 1 for i in range(num_nodes)
    )
    
    model.addConstrs(
        gp.quicksum(x[i, j] for i in range(num_nodes) if i != j) == 1 for j in range(num_nodes)
    )
    
    model.addConstrs(x[i, i] == 0 for i in range(num_nodes))
    
    
    model.addConstrs(
        gp.quicksum(phi[i, j] for i in range(num_nodes) if i != j) -
        gp.quicksum(phi[j, m] for m in range(num_nodes) if m != j) == 1
        for j in range(1, num_nodes)
    )
    model.addConstr(
        gp.quicksum(phi[i, 0] for i in range(num_nodes) if i != 0) -
        gp.quicksum(phi[0, j] for j in range(num_nodes) if j != 0) == 1 - num_nodes
    )
    model.addConstr(
        gp.quicksum(phi[0, j] for j in range(num_nodes) if j != 0) == num_nodes
    )
    model.addConstrs(
        phi[i, j] <= num_nodes * x[i, j] for i in range(num_nodes) for j in range(num_nodes)
    )

 
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  
                if j > i:
                    k = i * (num_nodes - 1) + j - 1
                else:
                    k = i * (num_nodes - 1) + j
                model.addConstr(x[i, j] == binary_sequence[k])
          
                
                
    pred_constr = add_predictor_constr(model, model_nn, binary_sequence, cost_surrogate)
    
    
    solver_options = {
    "TimeLimit": 18000, 
    "MIPGap": 0, 
    #"Threads": 8,  
    "OutputFlag": 0,  
    #"LogFile": f"solver_log_{s}_{n}.txt"  
    }


    for param, value in solver_options.items():
        model.setParam(param, value)
    
    
    
    start_time = time.time()
    start_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    model.optimize()
    end_time = time.time()
    end_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    resolution_time = end_time - start_time
    cpu = end_cpu - start_cpu
    
    route = []
    X = {}
    BinarySequence = np.zeros(num_nodes * (num_nodes - 1))

    if model.status == GRB.OPTIMAL:
        res = model.objVal
        current_node = 0
        route.append(current_node + 1)

        while len(route) < num_nodes:
            for j in range(num_nodes):
                if x[current_node, j].x > 0.5:
                    route.append(j + 1)
                    current_node = j
                    break

        for i in range(num_nodes):
            for j in range(num_nodes):
                X[(i, j)] = x[i, j].x

        for k in range(num_nodes * (num_nodes - 1)):
            BinarySequence[k] = binary_sequence[k].x 
            
            
    solution = sequence_to_adjacency_matrix(route)
    first_stage_cost = eval_first(solution, edge_weights, num_nodes)
    second_stage_cost = eval_second(solution, num_nodes, num_path)
    total_cost = first_stage_cost + second_stage_cost

    return route, res, total_cost, resolution_time, cpu








