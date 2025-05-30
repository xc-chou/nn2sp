#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
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
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import time
import random
import resource
import re
import csv



# In[2]:


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




# In[3]:


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
                
    #scenario_weights = np.min(scenario_weights, axis=2)
    return scenario_weights


# In[4]:


def calculate_deltas(scenario_weights):
    
    deltas = np.zeros_like(scenario_weights)
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_path):
                deltas[i, j, k] = scenario_weights[i, j, k] - edge_weights[i, j]
    
    deltas = np.min(deltas, axis=2)
    return deltas


# In[5]:


def solve_recourse(solution_x, deltas, num_nodes, num_path):
    model = gp.Model("Second_Stage_Problem")

    
    
    y = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="y")
    


    
    model.setObjective(
        gp.quicksum(deltas[i, j] * y[i, j] 
                      for i in range(num_nodes) for j in range(num_nodes)),
        sense=GRB.MINIMIZE
    )
    
    

    
    model.addConstrs(
        (y[i, j]  == solution_x[i][j] for i in range(num_nodes) for j in range(num_nodes)),
        name="select_path")
    
  
    model.setParam('OutputFlag', 0)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        res = model.objVal
        return res 
    
with open("test_set_scenarios.txt", "r") as f:
        valid_range = [int(line.strip()) for line in f]    
    
def eval_first(first_stage_solution, edge_weights, num_nodes):
    return sum(edge_weights[i,j] * first_stage_solution[i][j] for i in range (num_nodes) for j in range(num_nodes))


def eval_second(first_stage_solution, num_nodes, num_path):
    second = 0
    for i in valid_range:
        filepath = f"data_prob/Scenario{i}.dat"
        scenario_weights = read_scenario(filepath, num_nodes, num_path)
        deltas = calculate_deltas(scenario_weights)
        second += solve_recourse(first_stage_solution, deltas, num_nodes, num_path)
    return second/500


def sequence_to_adjacency_matrix(sequence):
    n = max(sequence)
    
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(len(sequence) - 1):
        adjacency_matrix[sequence[i] - 1][sequence[i + 1] - 1] = 1
    
    adjacency_matrix[sequence[-1] - 1][sequence[0] - 1] = 1
    
    return adjacency_matrix




# In[6]:


def build_model(deltas):
    model = pyo.ConcreteModel()
    
    
    model.nodes = set(range(num_nodes))
    model.path = set(range(num_path))
    
    model.x = pyo.Var(model.nodes, model.nodes, domain=pyo.Binary)
    model.y = pyo.Var(model.nodes, model.nodes, domain=pyo.Binary)
    model.phi = pyo.Var(model.nodes, model.nodes, domain=pyo.NonNegativeReals)
    
    model.firstStageCost = sum(model.x[i, j] * edge_weights[i][j] for i in model.nodes for j in model.nodes if j != i)
    model.secondStageCost = sum(deltas[i, j] * model.y[i, j] for i in model.nodes for j in model.nodes)
    
    model.OBJ = pyo.Objective(
        expr=model.firstStageCost + model.secondStageCost,
        sense=pyo.minimize
    )

    model.CONSTR = pyo.ConstraintList()
    
    for i in model.nodes:
        model.CONSTR.add(sum(model.x[i, j] for j in model.nodes if j != i) == 1 
        )
  
    
    for j in model.nodes:
        model.CONSTR.add(sum(model.x[i, j] for i in model.nodes if j != i) == 1 
        )

        
    for i in model.nodes:
        model.CONSTR.add(model.x[i, i] == 0 
        )

            
            
    for i in model.nodes:
        for j in model.nodes:
            model.CONSTR.add(model.y[i, j] == model.x[i, j]
            )        

    for j in range(1, num_nodes):
        model.CONSTR.add(sum(model.phi[i, j] for i in model.nodes if i != j) - 
            sum(model.phi[j, m] for m in model.nodes if m != j) == 1
        )

    model.CONSTR.add(sum(model.phi[i, 0] for i in model.nodes if i != 0) - 
        sum(model.phi[0, j] for j in model.nodes if j != 0) == 1 - num_nodes
    )

    model.CONSTR.add(sum(model.phi[0, j] for j in model.nodes if j != 0) == num_nodes
    )

    for i in model.nodes:
        for j in model.nodes:
            model.CONSTR.add(model.phi[i, j] <= num_nodes * model.x[i, j])

    return model


# In[11]:


def scenario_creator(scenario_name):
    for i in scenario_set:
        if scenario_name == "Scenario"+str(i):
            filepath = f"data_prob/Scenario{i}.dat"
            scenario_weights = read_scenario(filepath, num_nodes, num_path)
            deltas = calculate_deltas(scenario_weights)
    model = build_model(deltas)

    sputils.attach_root_node(model, model.firstStageCost, [model.x])
    model._mpisppy_probability = 1.0 / s
    return model  




# In[8]:


def extract_route_from_solution(solution_dict):
    x = {}
    for key, value in solution_dict.items():
        if value > 0.5: 
            i, j = map(int, key.strip('x[]').split(','))
            x[(i, j)] = value

    current_node = 0
    route = [current_node + 1]
    
    while len(route) < num_nodes:
        for j in range(num_nodes):
            if (current_node, j) in x: 
                route.append(j + 1) 
                current_node = j
                break
    
    return route


# In[9]:


file_path = "data_prob/prob.txt"
num_nodes, num_path, edge_weights = read_prob(file_path)


# In[10]:


s = 100


# In[15]:


        
with open(output_file, 'w') as output:
    output.write("Key, Route, ObjVal, Eval_cost, Time_Sec, CPU_Time_Sec\n")

    
    all_scenario_names = [f"Scenario{i}" for i in scenario_set]
    start_time = time.time()
    start_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime

    options = {"solver": "gurobi"}

    solver_options = {
            "TimeLimit": 18000,  
            "MIPGap": 0,
            "OutputFlag": 0
            #"Threads": 8 ,
            #"LogFile": f"solver_log_det_{s}_{key}.txt"
        }

    ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
    results = ef.solve_extensive_form(solver_options=solver_options, tee=True)

    end_time = time.time()
    end_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime

    objval = ef.get_objective_value()
    soln = ef.get_root_solution()

    route = extract_route_from_solution(soln)

    solution = sequence_to_adjacency_matrix(route)
    first_stage_cost = eval_first(solution, edge_weights, num_nodes)
    second_stage_cost = eval_second(solution, num_nodes, num_path)
    total_cost = first_stage_cost + second_stage_cost

    output.write(f"{key}, {route}, {round(objval, 3)}, {round(total_cost, 3)}, {round(end_time - start_time, 2)}, {round(end_cpu - start_cpu, 2)}\n")

    print(route, round(objval, 3), round(total_cost, 3), round(end_time - start_time, 2), round(end_cpu - start_cpu, 2))



