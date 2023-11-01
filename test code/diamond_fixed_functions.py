# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:17:18 2023

@author: lioyu
"""


import numpy as np 
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import csv
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import copy
import time
import pickle
import sys
import multiprocessing
import datetime
import os.path



# Bounds for input layer
def calc_inputL_bounds(input_values, inputSize, input_epsilon):

    dict_bounds = {}
    for num in range(inputSize):

        # No need for LC and UC for input nodes
        dict_bounds['0_'+str(num)] = { 'LB' : max(0,input_values[num] - input_epsilon), 'UB' : min(1,input_values[num] + input_epsilon), 'type' : 'input'}
    
    return dict_bounds


def calc_relu_bounds(layer_index, node_index, weights, dict_bound):
    '''
    this : the current node for which values are being computed
    prev : previous node (whose values in turn were computed by affine)

    dict_bound is dict_bounds

    NOTE : Actually this and prev refer to same node, but DeepPoly treats them differently
    '''



    prev_layer_index = layer_index - 1
    weight_layer_index = int((layer_index-2)/2)

    prev_node_index = str(prev_layer_index)+'_'+str(node_index)     # this is affine node
    this_node_index = str(layer_index)+'_'+str(node_index)

    prev_LB = dict_bound[prev_node_index]['LB']
    prev_UB = dict_bound[prev_node_index]['UB']


    if prev_UB <= 0:
        this_LC_vec = np.zeros(weights[weight_layer_index].shape[0])  # CHECK
        this_UC_vec = np.zeros(weights[weight_layer_index].shape[0])  # Both this_UC_vec and this_LC_vec are the same here

        dict_bound[this_node_index] = {'LC' : {'vec' : this_LC_vec, 'const' : 0}, 'UC' : {'vec' : this_UC_vec, 'const' : 0}, 'LB' : 0, 'UB' : 0, 'type' : 'relu'}

    elif prev_LB >= 0:
        this_LC_vec = np.zeros(weights[weight_layer_index].shape[0])  #CHECK
        this_UC_vec = np.zeros(weights[weight_layer_index].shape[0])
        this_LC_vec[node_index] = 1
        this_UC_vec[node_index] = 1                                  # Both this_UC_vec and this_LC_vec are the same here


        dict_bound[this_node_index] = {'LC' : {'vec' : this_LC_vec, 'const' : 0}, 'UC' : {'vec' : this_UC_vec, 'const' : 0}, 
        'LB' : prev_LB, 'UB' : prev_UB, 'type' : 'relu'}


    elif prev_UB > 0 and prev_LB <0:
        if (prev_UB - prev_LB)*prev_UB >= prev_LB*(prev_LB - prev_UB):    # triangle abstraction 4(b)      
            

            this_LB = prev_LB           # This is Original Line
            this_UB = prev_UB
            arr_temp = np.zeros(weights[weight_layer_index].shape[0])
            arr_temp[node_index] = 1

            this_LC_const = 0
            this_LC_vec = arr_temp              

            scalar_mult = prev_UB / (prev_UB - prev_LB)    # Variable used to ease calculation
            this_UC_const = (-1*prev_LB) * scalar_mult     # CHECK
            this_UC_vec = arr_temp * scalar_mult           # CHECK

            dict_bound[this_node_index] = {'LC' : {'vec' : this_LC_vec, 'const' : this_LC_const}, 'UC' : {'vec' : this_UC_vec, 'const' : this_UC_const}, 
            'LB' : this_LB, 'UB' : this_UB , 'type' : 'relu'}

        else:   # triangle abstraction 4(c) 

            this_LB = 0
            this_UB = prev_UB
            this_LC_const = 0
            this_LC_vec = np.zeros(weights[weight_layer_index].shape[0])

            scalar_mult = prev_UB / (prev_UB - prev_LB)     # Variable used to ease calculation
            this_UC_const = (-1*prev_LB) * scalar_mult      # CHECK
            arr_temp = np.zeros(weights[weight_layer_index].shape[0])
            arr_temp[node_index] = 1
            this_UC_vec = arr_temp * scalar_mult                  # CHECK

            dict_bound[this_node_index] = {'LC' : {'vec' : this_LC_vec, 'const' : this_LC_const}, 'UC' : {'vec' : this_UC_vec, 'const' : this_UC_const}, 
            'LB' : this_LB, 'UB' : this_UB , 'type' : 'relu'}
            
    
    return dict_bound


def calc_affine_bounds(start_time,layer_index, node_index, weights, bias, dict_bounds):
    '''
    this : the current node for which values are being computed

    dict_bound is dict_bounds
    '''


    this_node_index = str(layer_index)+'_'+str(node_index)
    weight_layer_index = int((layer_index-1)/2)

    this_LB = 0
    this_UB = 0
    
    # First Vector and const of current node
    this_UC_vec = weights[weight_layer_index][node_index]
    this_UC_const = bias[weight_layer_index][node_index]    # CHECK   - what happens when bias is negative / need to switch signs?
    this_LC_vec = weights[weight_layer_index][node_index]
    this_LC_const = bias[weight_layer_index][node_index]    # CHECK   - what happens when bias is negative / need to switch signs?

    this_UC_vec_store = weights[weight_layer_index][node_index]
    this_UC_const_store = bias[weight_layer_index][node_index]

    this_LC_vec_store = weights[weight_layer_index][node_index]
    this_LC_const_store = bias[weight_layer_index][node_index]
    



    for ctr in range(layer_index-1, -1, -1):
        

        
        if ctr == 0:                                        # For input layer

            for num in range(this_UC_vec.shape[0]):         # this_UC_vec and this_LC_vec should have same shape
                if this_UC_vec[num] >= 0:   
                    this_UB = this_UB + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UB']
                else:
                    this_UB = this_UB + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LB']

                if this_LC_vec[num] >= 0:   
                    this_LB = this_LB + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LB']
                else:
                    this_LB = this_LB + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UB']
            
            this_LB += this_LC_const
            this_UB += this_UC_const
            
            dict_bounds[this_node_index] = {'LC' : {'vec' : this_LC_vec_store, 'const' : this_LC_const_store}, 'UC' : {'vec' : this_UC_vec_store, 'const' : this_UC_const_store}, 
            'LB' : this_LB, 'UB' : this_UB , 'type' : 'affine'}
            

        else:
                                               # For every layer after input layer 

            temp_UC_vec = np.zeros(dict_bounds[str(ctr)+'_'+'0']['UC']['vec'].shape[0])
            temp_LC_vec = np.zeros(dict_bounds[str(ctr)+'_'+'0']['LC']['vec'].shape[0])
            temp_UC_const = 0
            temp_LC_const = 0

            for num in range(this_UC_vec.shape[0]):         # this_UC_vec and this_LC_vec should have same shape

                if this_UC_vec[num] >= 0:   
                    temp_UC_vec = temp_UC_vec + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UC']['vec']
                    temp_UC_const = temp_UC_const + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UC']['const']
                else:
                    temp_UC_vec = temp_UC_vec + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LC']['vec']
                    temp_UC_const = temp_UC_const + this_UC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LC']['const']

                if this_LC_vec[num] >= 0:   
                    temp_LC_vec = temp_LC_vec + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LC']['vec']
                    temp_LC_const = temp_LC_const + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['LC']['const']
                else:
                    temp_LC_vec = temp_LC_vec + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UC']['vec']
                    temp_LC_const = temp_LC_const + this_LC_vec[num] * dict_bounds[str(ctr)+'_'+str(num)]['UC']['const']
            
            this_UC_vec = np.copy(temp_UC_vec)
            this_LC_vec = np.copy(temp_LC_vec)
            this_UC_const += temp_UC_const
            this_LC_const += temp_LC_const

    return 0

def certify_model(LastLayer_id, numNodes, dict_bounds):
    '''
    Output : 1 then certified, if not then 0
    '''


    cert = 1
    lb_li = []

    for numn in range(0, numNodes):

        lb_li.append(dict_bounds[str(LastLayer_id)+'_'+str(numn)]['LB'])
        
        if dict_bounds[str(LastLayer_id)+'_'+str(numn)]['LB'] < 0:
            cert = 0
            # break

    return cert, lb_li


def remake_last_weights(cor_label, weights):
    '''
    cor_label expected to be between 0 and whatever (ex. 10)

    NOTE: weights expected to be list of numpy arrays

    numLayers here does not include input and the last DeepPoly layer
    '''

    numNodes = weights[-1].shape[0] - 1
    li = []
    counter = 0

    for node in range(numNodes+1):
        arr = np.zeros(weights[-1].shape[0])
        arr[cor_label] = 1
            
        if counter == cor_label:
            counter += 1
            
        else:
            arr[counter] = -1
            li.append(arr)
            counter += 1
    
    last_weight = np.array(li, dtype=np.float32)
    last_bias = np.zeros(numNodes)

    return last_weight, last_bias





def make_last_weights(cor_label, weights):
    '''
    cor_label expected to be between 0 and whatever (ex. 10)

    NOTE: weights expected to be list of numpy arrays

    numLayers here does not include input and the last DeepPoly layer
    '''

    numNodes = weights[-1].shape[0] - 1
    li = []
    counter = 0

    for node in range(numNodes+1):
        arr = np.zeros(weights[-1].shape[0])
        arr[cor_label] = 1
            
        if counter == cor_label:
            counter += 1
            
        else:
            arr[counter] = -1
            li.append(arr)
            counter += 1
    
    last_weight = np.array(li, dtype=np.float32)
    last_bias = np.zeros(numNodes)

    return last_weight, last_bias
    
# Reading Data - MNIST

def extract_mnist_data(data_path):
    '''
    create dict : {id_num : {label : num, image : []}}
    '''
    data_dict = {}
    id_num = 0

    csvfile = open(data_path, 'r')
    mnist_tests = csv.reader(csvfile, delimiter=',')
    for row in mnist_tests:
        label = int(row[0])
        image = np.array(row[1:], dtype = np.float64).reshape([1, 1, 28, 28])

        data_dict[id_num] = {'label' : label, 'image' : image}
        id_num += 1
    
    return data_dict

# onnx



def make_onnx_prediction(model_path, image):

    sess = ort.InferenceSession(model_path,providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_arr = sess.run([label_name], {input_name: image.astype(np.float32)})[0]
    # pred_arr = sess.run([label_name], {input_name: image})[0]
    pred_label = np.argmax(pred_arr) 

    return pred_label, pred_arr



def extract_onnx_weights_biases_numNodes(model_path,network_name):

    model = onnx.load(model_path)
    weights_model = model.graph.initializer
    
    if network_name == 'mnist_relu_9_100':

        bias_0 = numpy_helper.to_array(weights_model[10])
        weights_0 = numpy_helper.to_array(weights_model[11])
        
    
        
        bias_1 = numpy_helper.to_array(weights_model[12])
        weights_1 = numpy_helper.to_array(weights_model[13])
        bias_2 = numpy_helper.to_array(weights_model[14])
        weights_2 = numpy_helper.to_array(weights_model[15])
    
        bias_3 = numpy_helper.to_array(weights_model[16])
        weights_3 = numpy_helper.to_array(weights_model[17])
        bias_4 = numpy_helper.to_array(weights_model[0])
        weights_4 = numpy_helper.to_array(weights_model[1])
        bias_5 = numpy_helper.to_array(weights_model[2])
        weights_5 = numpy_helper.to_array(weights_model[3])
        
        bias_6 = numpy_helper.to_array(weights_model[4])
        weights_6 = numpy_helper.to_array(weights_model[5])
        bias_7 = numpy_helper.to_array(weights_model[6])
        weights_7 = numpy_helper.to_array(weights_model[7])
        
        bias_8 = numpy_helper.to_array(weights_model[8])
        weights_8 = numpy_helper.to_array(weights_model[9])
    
    
        weights = [np.array(weights_0, dtype=np.float32), np.array(weights_1, dtype=np.float32), np.array(weights_2, dtype=np.float32),
                   np.array(weights_3, dtype=np.float32), np.array(weights_4, dtype=np.float32), np.array(weights_5, dtype=np.float32),
                   np.array(weights_6, dtype=np.float32), np.array(weights_7, dtype=np.float32), np.array(weights_8, dtype=np.float32)]
        bias = [np.array(bias_0, dtype=np.float32), np.array(bias_1, dtype=np.float32), np.array(bias_2, dtype=np.float32),
                np.array(bias_3, dtype=np.float32), np.array(bias_4, dtype=np.float32), np.array(bias_5, dtype=np.float32),
                np.array(bias_6, dtype=np.float32), np.array(bias_7, dtype=np.float32), np.array(bias_8, dtype=np.float32)]
        
    elif network_name == 'mnist_relu_6_100' or network_name == 'mnist_relu_6_200':
        
        bias_0 = numpy_helper.to_array(weights_model[4])
        weights_0 = numpy_helper.to_array(weights_model[5])
        bias_1 = numpy_helper.to_array(weights_model[6])
        weights_1 = numpy_helper.to_array(weights_model[7])
        bias_2 = numpy_helper.to_array(weights_model[8])
        weights_2 = numpy_helper.to_array(weights_model[9])

        bias_3 = numpy_helper.to_array(weights_model[10])
        weights_3 = numpy_helper.to_array(weights_model[11])
        bias_4 = numpy_helper.to_array(weights_model[0])
        weights_4 = numpy_helper.to_array(weights_model[1])
        bias_5 = numpy_helper.to_array(weights_model[2])
        weights_5 = numpy_helper.to_array(weights_model[3])

        weights = [np.array(weights_0, dtype=np.float32), np.array(weights_1, dtype=np.float32), np.array(weights_2, dtype=np.float32),
                   np.array(weights_3, dtype=np.float32), np.array(weights_4, dtype=np.float32), np.array(weights_5, dtype=np.float32)]
        bias = [np.array(bias_0, dtype=np.float32), np.array(bias_1, dtype=np.float32), np.array(bias_2, dtype=np.float32),
                np.array(bias_3, dtype=np.float32), np.array(bias_4, dtype=np.float32), np.array(bias_5, dtype=np.float32)]

    numNodesPerL = []

    for weight in weights:
        numNodesPerL.append(weight.shape[0])

    return weights, bias, numNodesPerL

def calc_affine_bounds_iteration(start_time,numl, nodeID_start,num_nodes_per_process, weights, bias, dict_bounds_DP):
    # sys.stdout = open(file_name,'at')
    
    dict_bounds = copy.deepcopy(dict_bounds_DP)
    

    
    for i in range(num_nodes_per_process):
        calc_affine_bounds(start_time,numl, nodeID_start+i, weights, bias, dict_bounds)
        
        # if nodeID_start == 0:
        #     print('layerID:',numl,time.time()-start_time,'\n')
        #     sys.stdout.flush()

    return dict_bounds



def calc_relu_bounds_iteration(numl, nodeID_start,num_nodes_per_process, weights, dict_bounds_DP):
    # sys.stdout = open(file_name,'at')
    
    dict_bounds = copy.deepcopy(dict_bounds_DP)
    

    
    for i in range(num_nodes_per_process):
        calc_relu_bounds(numl, nodeID_start+i, weights, dict_bounds)
        
        # if nodeID_start == 0:
        #     print('layerID:',numl,time.time()-start_time,'\n')
        #     sys.stdout.flush()

    return dict_bounds


def gurobi_initialize_model_layer_new(numLayers_Cof,input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()

    model = gp.Model()
    model.Params.LogToConsole = 0
    
    # SETTING Timeout
    # model.setParam('TimeLimit', timelimit)
    # model.setParam('TuneTimeLimit', 1)

    # SETTING MIPFocus
    # model.Params.MIPFo = 1
    model.setParam('MIPFocus', mipf)

    # Setting number of Threads
    if layerID == 1 or layerID == numLayers_Cof*2+1:
        model.params.Threads = 1
    else:
        model.params.Threads = 3
        

    counter_integer_variable_1 = 0
    counter_integer_variable_2 = 0


    # NOTE : Here 'linear' is used as if 'continuous'
    li_linear_nodes = []    # NOTE : all nodes- affine and relu of all kinds have a linear var that is from -ve to +ve infinity
    relu_nodes = []   # NOTE : all other relu nodes (that are in stepback range) have an additional linear var varying from 0 to 1

    li_integer_layers = []      # keeps track of relu layers that are encoded by integer variables (reluNum)


    



    numNodesPerL_li = numNodesPerL

    # FirstLayer from which only upper and lower bounds are taken
    FirstLayer = 0
    
    for i in range(FirstLayer, layerID+1):

        if i == 0: 
            for j in range(inputSize):
                tup = (0 , j)
                li_linear_nodes.append(tup)

        else:
            for j in range(0, numNodesPerL_li[i-1]):
                
                tup = (i , j)

                if i%2==0 and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:

                    relu_nodes.append(tup)

                li_linear_nodes.append(tup)
        
    # li_linear_nodes.append((layerID, nodeID))

    # Variables Initialization
    linear_vars = model.addVars(li_linear_nodes, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, ub=GRB.INFINITY, name = 'linear_vars')
    relu_vars = model.addVars(relu_nodes, vtype=GRB.CONTINUOUS, lb= 0, ub= 1, name = 'relu_vars')



    # Constraints for Inputs
    # MIGHT NEED TO CHECK whether input is np array or list or whatever
    if FirstLayer == 0:
        for i in range(inputSize):
            input_upper_temp = min(1,input[i] + epsilon)
            input_lower_temp = max(0,input[i] - epsilon)
            model.addConstr(linear_vars[0,i] <= input_upper_temp)
            model.addConstr(linear_vars[0,i] >= input_lower_temp)
    else:
        for i in range(numNodesPerL_li[FirstLayer-1]):
            model.addConstr(linear_vars[FirstLayer,i] <= dict_bounds[str(FirstLayer)+'_'+str(i)]['UB'])
            model.addConstr(linear_vars[FirstLayer,i] >= dict_bounds[str(FirstLayer)+'_'+str(i)]['LB'])

    FirstAffine = 0
    FirstRelu = 0

    if FirstLayer==0:

        FirstAffine = 1
        FirstRelu = 2
    else:
        FirstAffine = FirstLayer+2
        FirstRelu = FirstLayer+1

    # Constraints for affine nodes
    for i in range(FirstAffine, layerID+1, 2):
        weight_layer_index = int((i-1)/2)

        for j in range(numNodesPerL_li[i-1]):
            
            model.addConstr(linear_vars[i,j] <= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            model.addConstr(linear_vars[i,j] >= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            if i < layerID:
                model.addConstr(linear_vars[i,j] >= dict_bounds[str(i)+'_'+str(j)]['LB'])

                model.addConstr(linear_vars[i,j] <= dict_bounds[str(i)+'_'+str(j)]['UB'])


    # NOTE : Have not put absolute upper and lower bounds on affine nodes

    
    # Constraints for Relu 
    for i in range(FirstRelu, layerID, 2):

        # For the integer Relu 
        if i%2==0:

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)

    
        
        
        
    for i in range(reluNum):

        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)
            
            
    for i in range(2, layerID+1):

        for j in range(0, numNodesPerL_li[i-1]):
            
            tup = (i , j)

            if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:   

                counter_integer_variable_1 += 1
                relu_vars[i,j].vtype = GRB.BINARY
                
    model.update()


    time_initialize = time.time() - start_time_initialize
        

    return model, counter_integer_variable_1, time_initialize




def gurobi_initialize_model_layer_by_layer(model, numLayers_Cof,input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()

    model.setParam('MIPFocus', mipf)

    # Setting number of Threads
    if layerID == 1 or layerID == numLayers_Cof*2+1:
        model.params.Threads = 1
    else:
        model.params.Threads = 3
        

    counter_integer_variable_1 = 0
    counter_integer_variable_2 = 0


    # NOTE : Here 'linear' is used as if 'continuous'
    li_linear_nodes = []    # NOTE : all nodes- affine and relu of all kinds have a linear var that is from -ve to +ve infinity
    relu_nodes = []   # NOTE : all other relu nodes (that are in stepback range) have an additional linear var varying from 0 to 1

    li_integer_layers = []      # keeps track of relu layers that are encoded by integer variables (reluNum)


    



    numNodesPerL_li = numNodesPerL

    # FirstLayer from which only upper and lower bounds are taken
    FirstLayer = layerID - 2
    
    for i in range(FirstLayer+1, layerID+1):

        if i == 0: 
            for j in range(inputSize):
                tup = (0 , j)
                li_linear_nodes.append(tup)

        else:
            for j in range(0, numNodesPerL_li[i-1]):
                
                tup = (i , j)

                if i%2==0 and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:

                    relu_nodes.append(tup)

                li_linear_nodes.append(tup)
        
    # li_linear_nodes.append((layerID, nodeID))

    # Variables Initialization
    linear_vars = model.addVars(li_linear_nodes, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, ub=GRB.INFINITY, name = 'linear_vars')
    relu_vars = model.addVars(relu_nodes, vtype=GRB.CONTINUOUS, lb= 0, ub= 1, name = 'relu_vars')


    # FirstLayer = max(0, layerID - 2)
    
    # Constraints for Inputs
    # MIGHT NEED TO CHECK whether input is np array or list or whatever
    if FirstLayer <= 0:
        for i in range(inputSize):
            input_upper_temp = min(1,input[i] + epsilon)
            input_lower_temp = max(0,input[i] - epsilon)
            model.addConstr(linear_vars[0,i] <= input_upper_temp)
            model.addConstr(linear_vars[0,i] >= input_lower_temp)
    else:
        for i in range(numNodesPerL_li[FirstLayer-1]):
            model.addConstr(linear_vars[FirstLayer,i] <= dict_bounds[str(FirstLayer)+'_'+str(i)]['UB'])
            model.addConstr(linear_vars[FirstLayer,i] >= dict_bounds[str(FirstLayer)+'_'+str(i)]['LB'])

    FirstAffine = 0
    FirstRelu = 0

    if FirstLayer <= 0:

        FirstAffine = 1
        FirstRelu = 2
    else:
        FirstAffine = FirstLayer+2
        FirstRelu = FirstLayer+1

    # Constraints for affine nodes
    for i in range(FirstAffine, layerID+1, 2):
        weight_layer_index = int((i-1)/2)

        for j in range(numNodesPerL_li[i-1]):
            
            model.addConstr(linear_vars[i,j] <= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            model.addConstr(linear_vars[i,j] >= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            if i < layerID:
                model.addConstr(linear_vars[i,j] >= dict_bounds[str(i)+'_'+str(j)]['LB'])

                model.addConstr(linear_vars[i,j] <= dict_bounds[str(i)+'_'+str(j)]['UB'])


    # NOTE : Have not put absolute upper and lower bounds on affine nodes

    
    # Constraints for Relu 
    for i in range(FirstRelu, layerID, 2):

        # For the integer Relu 
        if i%2==0:

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)

    
        
        
        
    for i in range(reluNum):

        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)
            
            
    for i in range(2, layerID+1):

        for j in range(0, numNodesPerL_li[i-1]):
            
            tup = (i , j)

            if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:   

                counter_integer_variable_1 += 1
                relu_vars[i,j].vtype = GRB.BINARY
            else:
                relu_vars[i,j].vtype = GRB.CONTINUOUS
                
    model.update()


    time_initialize = time.time() - start_time_initialize
        

    return model, counter_integer_variable_1, time_initialize










def gurobi_initialize_model_layer(numLayers_Cof,input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()

    model = gp.Model()
    model.Params.LogToConsole = 0
    
    # SETTING Timeout
    # model.setParam('TimeLimit', timelimit)
    # model.setParam('TuneTimeLimit', 1)

    # SETTING MIPFocus
    # model.Params.MIPFo = 1
    model.setParam('MIPFocus', mipf)

    # Setting number of Threads
    if layerID == 1 or layerID == numLayers_Cof*2+1:
        model.params.Threads = 1
    else:
        model.params.Threads = 3
        

    counter_integer_variable_1 = 0
    counter_integer_variable_2 = 0


    # NOTE : Here 'linear' is used as if 'continuous'
    li_linear_nodes = []    # NOTE : all nodes- affine and relu of all kinds have a linear var that is from -ve to +ve infinity
    li_integer_nodes = []   # NOTE : only some relu nodes depending on whether they are in reluNUM/ k-param layers have integer vars
    li_linear_relu_nodes = []   # NOTE : all other relu nodes (that are in stepback range) have an additional linear var varying from 0 to 1

    li_integer_layers = []      # keeps track of relu layers that are encoded by integer variables (reluNum)


    for i in range(reluNum):

        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)



    numNodesPerL_li = numNodesPerL

    # FirstLayer from which only upper and lower bounds are taken
    FirstLayer = layerID -(2*reluNum) -(2*stepBack)
    if FirstLayer < 0:
        FirstLayer = 0
    
    for i in range(FirstLayer, layerID+1):

        if i == 0: 
            for j in range(inputSize):
                tup = (0 , j)
                li_linear_nodes.append(tup)

        else:
            for j in range(0, numNodesPerL_li[i-1]):
                
                tup = (i , j)

                if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:   

                    li_integer_nodes.append(tup)
                    counter_integer_variable_1 += 1

                if i not in li_integer_layers and i%2==0 and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:

                    li_linear_relu_nodes.append(tup)

                li_linear_nodes.append(tup)
        
    # li_linear_nodes.append((layerID, nodeID))

    # Variables Initialization
    linear_vars = model.addVars(li_linear_nodes, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, ub=GRB.INFINITY, name = 'linear_vars')
    integer_vars = model.addVars(li_integer_nodes, vtype=GRB.BINARY)
    linear_relu_vars = model.addVars(li_linear_relu_nodes, vtype=GRB.CONTINUOUS, lb= 0, ub= 1)



    # Constraints for Inputs
    # MIGHT NEED TO CHECK whether input is np array or list or whatever
    if FirstLayer == 0:
        for i in range(inputSize):
            input_upper_temp = min(1,input[i] + epsilon)
            input_lower_temp = max(0,input[i] - epsilon)
            model.addConstr(linear_vars[0,i] <= input_upper_temp)
            model.addConstr(linear_vars[0,i] >= input_lower_temp)
    else:
        for i in range(numNodesPerL_li[FirstLayer-1]):
            model.addConstr(linear_vars[FirstLayer,i] <= dict_bounds[str(FirstLayer)+'_'+str(i)]['UB'])
            model.addConstr(linear_vars[FirstLayer,i] >= dict_bounds[str(FirstLayer)+'_'+str(i)]['LB'])

    FirstAffine = 0
    FirstRelu = 0

    if FirstLayer==0:

        FirstAffine = 1
        FirstRelu = 2
    else:
        FirstAffine = FirstLayer+2
        FirstRelu = FirstLayer+1

    # Constraints for affine nodes
    for i in range(FirstAffine, layerID+1, 2):
        weight_layer_index = int((i-1)/2)

        for j in range(numNodesPerL_li[i-1]):
            
            model.addConstr(linear_vars[i,j] <= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            model.addConstr(linear_vars[i,j] >= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            if i < layerID:
                model.addConstr(linear_vars[i,j] >= dict_bounds[str(i)+'_'+str(j)]['LB'])

                model.addConstr(linear_vars[i,j] <= dict_bounds[str(i)+'_'+str(j)]['UB'])


    # NOTE : Have not put absolute upper and lower bounds on affine nodes

    
    # Constraints for Relu 
    for i in range(FirstRelu, layerID, 2):

        # For the integer Relu 
        if i in li_integer_layers:

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-integer_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * integer_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)

        # For the continuous Relu 
        else:    

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    # counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-linear_relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * linear_relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)



    time_initialize = time.time() - start_time_initialize

    if counter_integer_variable_2 != counter_integer_variable_1:
        print('Counter ERROR in Gurobi Model Initialization')
        

    return model, counter_integer_variable_1, time_initialize


def gurobi_max(model, layerID, nodeID):
        
    # model.setObjective(linear_vars[layerID, nodeID], GRB.MAXIMIZE)

    model.update()
    string = 'linear_vars['+str(layerID)+','+str(nodeID)+']'
    model.setObjective(model.getVarByName(string), GRB.MAXIMIZE)


    return model


def gurobi_min(model, layerID, nodeID):
        
    # model.setObjective(linear_vars[layerID, nodeID], GRB.MINIMIZE)

    model.update()
    string = 'linear_vars['+str(layerID)+','+str(nodeID)+']'
    model.setObjective(model.getVarByName(string), GRB.MINIMIZE)



    return model



def gurobi_initialize_model_node_diamond_new(numLayers_Cof, input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack,relu_nodes_open, print_stuff):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()
    
    
    # for node in relu_nodes_open:
    #     if layerID == 3 and not (dict_bounds[node]['UB'] > 0 and dict_bounds[node]['LB'] < 0):
    #         print(node)
    
    

    model = gp.Model()
    model.Params.LogToConsole = 0
    
    # SETTING Timeout
    # model.setParam('TimeLimit', timelimit)
    # model.setParam('TuneTimeLimit', 1)

    # SETTING MIPFocus
    # model.Params.MIPFo = 1
    model.setParam('MIPFocus', mipf)

    # Setting number of Threads
    if layerID == 1 or layerID == numLayers_Cof*2+1:
        model.params.Threads = 1
    else:
        model.params.Threads = 3

    counter_integer_variable_1 = 0
    counter_integer_variable_2 = 0


    # NOTE : Here 'linear' is used as if 'continuous'
    li_linear_nodes = []    # NOTE : all nodes- affine and relu of all kinds have a linear var that is from -ve to +ve infinity
    li_integer_nodes = []   # NOTE : only some relu nodes depending on whether they are in reluNUM/ k-param layers have integer vars
    li_linear_relu_nodes = []   # NOTE : all other relu nodes (that are in stepback range) have an additional linear var varying from 0 to 1

    li_integer_layers = []      # keeps track of relu layers that are encoded by integer variables (reluNum)


    for i in range(reluNum):
        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)

    numNodesPerL_li = numNodesPerL


    # FirstLayer from which only upper and lower bounds are taken
    FirstLayer = layerID -(2*reluNum) -(2*stepBack)
    if FirstLayer < 0:
        FirstLayer = 0
    
    for i in range(FirstLayer, layerID+1):

        if i == 0: 
            for j in range(inputSize):
                tup = (0 , j)
                li_linear_nodes.append(tup)

        else:
            for j in range(0, numNodesPerL_li[i-1]):
                
                tup = (i , j)

                if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and str(i-1)+'_'+str(j) in relu_nodes_open:   

                    li_integer_nodes.append(tup)

                    counter_integer_variable_1 += 1
                
                # if i in li_integer_layers and j in relu_nodes_open and (dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 or dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0):

                #     li_linear_relu_nodes.append(tup)
                
                if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and str(i-1)+'_'+str(j) not in relu_nodes_open:
                    
                    li_linear_relu_nodes.append(tup)

                if i not in li_integer_layers and i%2==0 and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:

                    li_linear_relu_nodes.append(tup)

                li_linear_nodes.append(tup)
        
    # li_linear_nodes.append((layerID, nodeID))

    # Variables Initialization
    linear_vars = model.addVars(li_linear_nodes, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, ub=GRB.INFINITY, name = 'linear_vars')
    integer_vars = model.addVars(li_integer_nodes, vtype=GRB.BINARY)
    linear_relu_vars = model.addVars(li_linear_relu_nodes, vtype=GRB.CONTINUOUS, lb= 0, ub= 1)

    # Constraints for Inputs
    # MIGHT NEED TO CHECK whether input is np array or list or whatever
    if FirstLayer == 0:
        for i in range(inputSize):
            input_upper_temp = min(1,input[i] + epsilon)
            input_lower_temp = max(0,input[i] - epsilon)
            model.addConstr(linear_vars[0,i] <= input_upper_temp)
            model.addConstr(linear_vars[0,i] >= input_lower_temp)
    else:
        for i in range(numNodesPerL_li[FirstLayer-1]):
            model.addConstr(linear_vars[FirstLayer,i] <= dict_bounds[str(FirstLayer)+'_'+str(i)]['UB'])
            model.addConstr(linear_vars[FirstLayer,i] >= dict_bounds[str(FirstLayer)+'_'+str(i)]['LB'])

    FirstAffine = 0
    FirstRelu = 0

    if FirstLayer==0:

        FirstAffine = 1
        FirstRelu = 2
    else:
        FirstAffine = FirstLayer+2
        FirstRelu = FirstLayer+1

    # Constraints for affine nodes
    for i in range(FirstAffine, layerID+1, 2):
        weight_layer_index = int((i-1)/2)

        for j in range(numNodesPerL_li[i-1]):
            
            model.addConstr(linear_vars[i,j] <= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            model.addConstr(linear_vars[i,j] >= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            if i < layerID:
                model.addConstr(linear_vars[i,j] >= dict_bounds[str(i)+'_'+str(j)]['LB'])

                model.addConstr(linear_vars[i,j] <= dict_bounds[str(i)+'_'+str(j)]['UB'])
    

    for i in range(FirstRelu, layerID, 2):

        # For the integer Relu 
        if i in li_integer_layers:

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and str(i-1)+'_'+str(j) in relu_nodes_open:

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-integer_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * integer_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)


                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and str(i-1)+'_'+str(j) not in relu_nodes_open:

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-linear_relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * linear_relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)

        # For the continuous Relu 
        else:    

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    # counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-linear_relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * linear_relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)



    time_initialize = time.time() - start_time_initialize
    
    sys.stdout.flush()

    if counter_integer_variable_2 != counter_integer_variable_1:
        print('Counter ERROR in Gurobi Model Initialization')
        
        
    # print_stuff += 'Time for initialization : '
    # # print_stuff += str(time_initialize_0d + time_taken) please recover here after testing, liao
    # print_stuff += '\n'
    # # print('Org num of abbc tuples : ', Org_num_of_abbc_tuples)
    # print_stuff += 'Number of tuples left after filtering : '
    # # print_stuff += str(num_left_tuples)  please recover here after testing, liao
    # print_stuff += '\n'
    print_stuff += 'Final number of int relu : '
    print_stuff += str(len(relu_nodes_open))
    print_stuff += '\n'
    print_stuff += 'Num of int relu in g model : '
    print_stuff += str(counter_integer_variable_1)
    print_stuff += '\n'

    return model, counter_integer_variable_1 , print_stuff








def gurobi_initialize_model_node_diamond_new2(model, numLayers_Cof, input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack,relu_nodes_open, print_stuff):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()
    
    binary_var_count = 0
    
    
    # for node in relu_nodes_open:
    #     if layerID == 3 and not (dict_bounds[node]['UB'] > 0 and dict_bounds[node]['LB'] < 0):
    #         print(node)
    
    # for v in model.getVars():
    #     print(f"{v.VarName}")
    
    li_integer_layers = []
    
    for i in range(reluNum):

        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)
    
    
        
    for i in li_integer_layers:
        
        if i < 2:
            continue

        for j in range(0, numNodesPerL[int(i/2)-1]):
            
            node_name = str(i-1)+'_'+str(j)
            
            var_name = 'relu_vars['+str(i)+','+str(j)+']'
            

            
            x = model.getVarByName(var_name)


            if dict_bounds[node_name]['UB'] > 0 and dict_bounds[node_name]['LB'] < 0:   
                if node_name in relu_nodes_open:
                    x.vtype = GRB.BINARY
                    binary_var_count += 1
                else:
                    x.vtype = GRB.CONTINUOUS
                
    model.update()
        
    
    

    time_initialize = time.time()-start_time_initialize
    

    return model, binary_var_count, print_stuff



def gurobi_initialize_model_node_recover_layer_model(model, numLayers_Cof, input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack,relu_nodes_open, print_stuff):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()
    
    binary_var_count = 0
    
    
    # for node in relu_nodes_open:
    #     if layerID == 3 and not (dict_bounds[node]['UB'] > 0 and dict_bounds[node]['LB'] < 0):
    #         print(node)
    
    # for v in model.getVars():
    #     print(f"{v.VarName}")
    
    li_integer_layers = []
    
    for i in range(reluNum):

        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)
    
    
        
    for i in li_integer_layers:
        
        if i < 2:
            continue

        for j in range(0, numNodesPerL[int(i/2)-1]):
            
            node_name = str(i-1)+'_'+str(j)
            
            var_name = 'relu_vars['+str(i)+','+str(j)+']'
            

            
            x = model.getVarByName(var_name)


            if dict_bounds[node_name]['UB'] > 0 and dict_bounds[node_name]['LB'] < 0:   
                x.vtype = GRB.BINARY
                binary_var_count += 1
                
    model.update()
        
    
    

    time_initialize = time.time()-start_time_initialize
    

    return model, binary_var_count, print_stuff









































        
    

def gurobi_initialize_model_node_diamond(numLayers_Cof,input, inputSize, numNodesPerL, epsilon, dict_bounds, layerID, reluNum, weights, bias, mipf, stepBack,relu_nodes_open):
    '''
    This function initializes a gurobi model for a particular layer id

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    stepBack : number of relu layers that are taken as linear relaxed behind the reluNum relu nodes. Note that the first layer with constraints is 
               either the input layer or an affine layer (the layer just before the relu layer last in stepBack)

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer
    '''

    start_time_initialize = time.time()

    model = gp.Model()
    model.Params.LogToConsole = 0
    
    # SETTING Timeout
    # model.setParam('TimeLimit', timelimit)
    # model.setParam('TuneTimeLimit', 1)

    # SETTING MIPFocus
    # model.Params.MIPFo = 1
    model.setParam('MIPFocus', mipf)

    # Setting number of Threads
    if layerID == 1 or layerID == numLayers_Cof*2+1:
        model.params.Threads = 1
    else:
        model.params.Threads = 3

    counter_integer_variable_1 = 0
    counter_integer_variable_2 = 0


    # NOTE : Here 'linear' is used as if 'continuous'
    li_linear_nodes = []    # NOTE : all nodes- affine and relu of all kinds have a linear var that is from -ve to +ve infinity
    li_integer_nodes = []   # NOTE : only some relu nodes depending on whether they are in reluNUM/ k-param layers have integer vars
    li_linear_relu_nodes = []   # NOTE : all other relu nodes (that are in stepback range) have an additional linear var varying from 0 to 1

    li_integer_layers = []      # keeps track of relu layers that are encoded by integer variables (reluNum)


    for i in range(reluNum):
        if i == 0:
            li_integer_layers.append(layerID-1)
        else :
            li_integer_layers.append(li_integer_layers[-1]-2)


    numNodesPerL_li = numNodesPerL


    # FirstLayer from which only upper and lower bounds are taken
    FirstLayer = layerID -(2*reluNum) -(2*stepBack)
    if FirstLayer < 0:
        FirstLayer = 0
    
    for i in range(FirstLayer, layerID+1):

        if i == 0: 
            for j in range(inputSize):
                tup = (0 , j)
                li_linear_nodes.append(tup)

        else:
            for j in range(0, numNodesPerL_li[i-1]):
                
                tup = (i , j)

                if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and j in relu_nodes_open:   

                    li_integer_nodes.append(tup)
                    counter_integer_variable_1 += 1
                
                # if i in li_integer_layers and j in relu_nodes_open and (dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 or dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0):

                #     li_linear_relu_nodes.append(tup)
                
                if i in li_integer_layers and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and j not in relu_nodes_open:
                    
                    li_linear_relu_nodes.append(tup)

                if i not in li_integer_layers and i%2==0 and dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0:

                    li_linear_relu_nodes.append(tup)

                li_linear_nodes.append(tup)
        
    # li_linear_nodes.append((layerID, nodeID))

    # Variables Initialization
    linear_vars = model.addVars(li_linear_nodes, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, ub=GRB.INFINITY, name = 'linear_vars')
    integer_vars = model.addVars(li_integer_nodes, vtype=GRB.BINARY)
    linear_relu_vars = model.addVars(li_linear_relu_nodes, vtype=GRB.CONTINUOUS, lb= 0, ub= 1)

    # Constraints for Inputs
    # MIGHT NEED TO CHECK whether input is np array or list or whatever
    if FirstLayer == 0:
        for i in range(inputSize):
            input_upper_temp = min(1,input[i] + epsilon)
            input_lower_temp = max(0,input[i] - epsilon)
            model.addConstr(linear_vars[0,i] <= input_upper_temp)
            model.addConstr(linear_vars[0,i] >= input_lower_temp)
    else:
        for i in range(numNodesPerL_li[FirstLayer-1]):
            model.addConstr(linear_vars[FirstLayer,i] <= dict_bounds[str(FirstLayer)+'_'+str(i)]['UB'])
            model.addConstr(linear_vars[FirstLayer,i] >= dict_bounds[str(FirstLayer)+'_'+str(i)]['LB'])

    FirstAffine = 0
    FirstRelu = 0

    if FirstLayer==0:

        FirstAffine = 1
        FirstRelu = 2
    else:
        FirstAffine = FirstLayer+2
        FirstRelu = FirstLayer+1

    # Constraints for affine nodes
    for i in range(FirstAffine, layerID+1, 2):
        weight_layer_index = int((i-1)/2)

        for j in range(numNodesPerL_li[i-1]):
            
            model.addConstr(linear_vars[i,j] <= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            model.addConstr(linear_vars[i,j] >= gp.quicksum(weights[weight_layer_index][j][k] * linear_vars[i-1, k] for k 
            in range(len(weights[weight_layer_index][j]))) + bias[weight_layer_index][j])

            if i < layerID:
                model.addConstr(linear_vars[i,j] >= dict_bounds[str(i)+'_'+str(j)]['LB'])

                model.addConstr(linear_vars[i,j] <= dict_bounds[str(i)+'_'+str(j)]['UB'])


    
    # Constraints for Relu 
    for i in range(FirstRelu, layerID, 2):

        # For the integer Relu 
        if i in li_integer_layers:

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and j in relu_nodes_open:

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-integer_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * integer_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)


                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 and j not in relu_nodes_open:

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-linear_relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * linear_relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)

        # For the continuous Relu 
        else:    

            for j in range(numNodesPerL_li[i-1]): 

                if  dict_bounds[str(i-1)+'_'+str(j)]['LB'] >= 0 : 

                    # print('HIT - LB >= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1,j])

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1,j])
                
                if  dict_bounds[str(i-1)+'_'+str(j)]['UB'] <= 0 :

                    # print('HIT - UB <= 0 in MAX')

                    model.addConstr(linear_vars[i,j] <= 0)

                    model.addConstr(linear_vars[i,j] >= 0)

                if dict_bounds[str(i-1)+'_'+str(j)]['UB'] > 0 and dict_bounds[str(i-1)+'_'+str(j)]['LB'] < 0 :

                    # print('HIT - UB > 0  and LB < 0 in MAX')
                    # counter_integer_variable_2 += 1

                    model.addConstr(linear_vars[i,j] <= linear_vars[i-1, j] - dict_bounds[str(i-1)+'_'+str(j)]['LB'] * (1-linear_relu_vars[i,j]))

                    model.addConstr(linear_vars[i,j] >= linear_vars[i-1, j])

                    model.addConstr(linear_vars[i,j] <= dict_bounds[str(i-1)+'_'+str(j)]['UB'] * linear_relu_vars[i,j])

                    model.addConstr(linear_vars[i,j] >= 0)



    time_initialize = time.time() - start_time_initialize
    
    sys.stdout.flush()

    if counter_integer_variable_2 != counter_integer_variable_1:
        print('Counter ERROR in Gurobi Model Initialization')

    return model, counter_integer_variable_1, time_initialize



def gurobi_master_per_node_max(layerID, nodeID, model_0, counter_integer_variable_1):
    '''
    This function initializes and optimizes a gurobi model for one node based on diamonds strategy for max (UB) only. It returns the model and the UB
    It expects model as input as well

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer

    numLayers : Does not include input layer and last weird layer

    bia : bias

    weight : weights
    '''


    if counter_integer_variable_1 >0:

        #################################################
        start_time_max = time.time()
        # num_times_in_while_loop_max = 0
        

        model_0 = gurobi_max(model_0, layerID, nodeID)     # setting objective
        

        model_0.Params.SolutionLimit = 1
        
        
        sys.stdout.flush()
        
        
        model_0.optimize()
        sys.stdout.flush()

        if model_0.MIPGap <= 0.01 :
            max_nodeID = model_0.ObjBoundC
            # print('MIPGap for max : ', model_0.MIPGap)
            # model_0.reset(0)
        else :

            
            model_0.Params.SolutionLimit = 2000000000   # 2 bil is the max limit

            model_0.Params.TimeLimit = 3
            model_0.setParam('MIPGap', 0.01)
            sys.stdout.flush()
            model_0.optimize()
            sys.stdout.flush()
            


            model_0.Params.TimeLimit = 10
            model_0.setParam('MIPGap', 0.1)
            sys.stdout.flush()
            model_0.optimize()
            sys.stdout.flush()

                            
            max_nodeID = model_0.ObjBoundC
            # print('MIPGap for max : ', model_0.MIPGap)
            # model_0.reset(0)
        
        # print('Num of times in loop for max : ', num_times_in_while_loop_max)
        time_max_node = time.time() - start_time_max
        #################################################

    
    else :
        start_time_max = time.time()
        # num_times_in_while_loop_max = 0
        model_0 = gurobi_max(model_0, layerID, nodeID)     # setting objective
        sys.stdout.flush()
        model_0.optimize()
        sys.stdout.flush()
        max_nodeID = model_0.getObjective().getValue()
        # model_0.reset(0)
        time_max_node = time.time() - start_time_max
    
    sys.stdout.flush()

    # model_0.reset(0)

    return model_0, time_max_node, max_nodeID




def gurobi_master_per_node_min(layerID, nodeID, model_0, counter_integer_variable_1):
    '''
    This function initializes and optimizes a gurobi model for one node based on diamonds strategy for min (LB) only. It expects model to be input.
    It returns the LB and model

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer

    numLayers : Does not include input layer and last weird layer

    bia : bias

    weight : weights
    '''

    if counter_integer_variable_1 >0:

        #################################################
        start_time_min = time.time()    
        # num_times_in_while_loop_min = 0

        model_0 = gurobi_min(model_0, layerID, nodeID)     # setting objective

        model_0.Params.SolutionLimit = 1
        sys.stdout.flush()
        
        
        model_0.optimize()
        sys.stdout.flush()

        if model_0.MIPGap <= 0.01 :
            min_nodeID = model_0.ObjBoundC
            # print('MIPGap for min : ', model_0.MIPGap)
            # model_0.reset(0)
        else :

            
            model_0.Params.SolutionLimit = 2000000000   # 2 bil is the max limit

            model_0.Params.TimeLimit = 3
            model_0.setParam('MIPGap', 0.01)
            sys.stdout.flush()
            model_0.optimize()
            sys.stdout.flush()
            
            
            model_0.Params.TimeLimit = 10
            model_0.setParam('MIPGap', 0.1)
            sys.stdout.flush()
            model_0.optimize()
            sys.stdout.flush()

                                
            min_nodeID = model_0.ObjBoundC
            # print('MIPGap for min : ', model_0.MIPGap)
            # model_0.reset(0)
        
        # print('Num of times in loop for min : ', num_times_in_while_loop_min)
        time_min_node = time.time() - start_time_min
        #################################################
    
    else :


        start_time_min = time.time()
        # num_times_in_while_loop_min = 0
        model_0 = gurobi_min(model_0, layerID, nodeID)     # setting objective
        sys.stdout.flush()
        model_0.optimize()
        sys.stdout.flush()
        min_nodeID = model_0.getObjective().getValue()
        # model_0.reset(0)
        time_min_node = time.time() - start_time_min

    sys.stdout.flush()

    return model_0, time_min_node, min_nodeID


















