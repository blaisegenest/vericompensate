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
from diamond_fixed_functions import calc_inputL_bounds,calc_relu_bounds,calc_affine_bounds,certify_model,make_last_weights,extract_mnist_data,make_onnx_prediction,extract_onnx_weights_biases_numNodes,calc_affine_bounds_iteration,calc_relu_bounds_iteration
from diamond_fixed_functions import gurobi_initialize_model_layer_new, gurobi_max, gurobi_min, gurobi_initialize_model_node_diamond_new,gurobi_master_per_node_max,gurobi_master_per_node_min
from diamond_fixed_functions import gurobi_initialize_model_node_diamond_new2, gurobi_initialize_model_node_recover_layer_model

# DeepPoly Master

def DeepPoly(cor_label, input_values, inputSize, input_epsilon, numLayers, numNodesPerL, weight, bia):
    # bia is bias 
    # weight is weights 

    weights = copy.deepcopy(weight)
    bias = copy.deepcopy(bia)

    start_time = time.time()

    dict_bounds_tbc = calc_inputL_bounds(input_values, inputSize, input_epsilon)
    

    
    numDeepPolyLayers = numLayers*2
    counter_numNodes = 0
    
    
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()
    dict_bounds_DP = manager.dict()
    # dict_bounds_DP = copy.deepcopy(dict_bounds_tbc)
    
    
    for key in dict_bounds_tbc.keys():
        
        dict_bounds_DP[key] = dict_bounds_tbc[key]

    num_nodes_per_process = 20

    for numl in range(1, numDeepPolyLayers+1):
        

        
        if numl >= numDeepPolyLayers+1:
            num_nodes_per_process = 5

        if numl % 2 == 0:

                # dict_bounds_DP = calc_relu_bounds(numl, numn, weights, dict_bounds_DP)

            return_dict = pool.starmap(calc_relu_bounds_iteration, [(numl, nodeID_start, num_nodes_per_process, weights, dict_bounds_DP) for nodeID_start in range(0,numNodesPerL[counter_numNodes],num_nodes_per_process)])

            for nodeID_start in range(0,numNodesPerL[counter_numNodes],num_nodes_per_process):
                for i in range(num_nodes_per_process):
                    dict_bounds_DP[str(numl)+'_'+str(nodeID_start+i)] = return_dict[int(nodeID_start/num_nodes_per_process)][str(numl)+'_'+str(nodeID_start+i)]
                    
            counter_numNodes += 1

        
        else:
            # for numn in range(num_nodes_per_process):
            #     # dict_bounds_DP = calc_relu_bounds(numl, numn, weights, dict_bounds_DP)

                
            return_dict = pool.starmap(calc_affine_bounds_iteration, [(start_time,numl, nodeID_start,num_nodes_per_process, weights, bias, dict_bounds_DP) for nodeID_start in range(0,numNodesPerL[counter_numNodes],num_nodes_per_process)])

            
            for nodeID_start in range(0,numNodesPerL[counter_numNodes],num_nodes_per_process):
                for i in range(num_nodes_per_process):
                    dict_bounds_DP[str(numl)+'_'+str(nodeID_start+i)] = return_dict[int(nodeID_start/num_nodes_per_process)][str(numl)+'_'+str(nodeID_start+i)]
                    

            
    numl = numDeepPolyLayers+1
    
    dict_bounds = copy.deepcopy(dict_bounds_DP)
    

    for numn in range(weights[len(weights)-1].shape[0]):

        calc_affine_bounds(start_time,numl, numn, weights, bias, dict_bounds)
        
        

    
    time_DP = time.time() - start_time

    LastLayer_id = numDeepPolyLayers+1
    PenultimateLayer_id = numDeepPolyLayers
    Last3Layer = numDeepPolyLayers - 1 

    numNodes_penultimateLayer = weights[len(weights)-2].shape[0] 
    numNodes_lastLayer = weights[len(weights)-1].shape[0] 
    numNodes_3_Layer = weights[len(weights)-3].shape[0] 


    cert = certify_model(LastLayer_id, numNodes_lastLayer, dict_bounds)


    return cert, dict_bounds, time_DP

# GUROBI

def analysis_LB_active_path(path,dict_bounds):

    nodes = path.split(',')
    
    nodes = nodes[1:-1]
    
    active = False
    
    for node in nodes:
        if dict_bounds[node]['LB'] < 0:
            active = True
            break
    return active


def analysis_LB_active_nodes(nodes,dict_bounds):
    
    active = False
    
    for node in nodes[1:-1]:
        if dict_bounds[node]['LB'] < 0:
            active = True
            break
    return active


def analysis_LB_active_nodes_all(nodes,dict_bounds):
    
    active = False
    
    for node in nodes:
        if dict_bounds[node]['LB'] < 0:
            active = True
            break
    return active




def analysis_common_node_path(path1,path2):

    nodes1 = path1.split(',')
    nodes2 = path2.split(',')
    

    active = False   
    
    for i in range(1,len(nodes1)-1):
        if nodes1[i] == nodes2[i]:
            active = True
            break
    return active


def bootstrap_decide_binary_nodes_new(weights, numNodesPerL,layerID,nodeID,source_layer_ID, source_node_ID):
    
    
    '''

    source layer node is the begining node of the Diamond path
    
    layerID and nodeID are the end of Diamond path
    
    '''
    
    
    # if layerID == 3:
    #     record_at_most_number = 100

    if layerID- source_layer_ID <= 2:
        print('error')
        return 0

    positive_paths = {}
    
    negative_paths = {}
    
    source_layer_ID_begin = source_layer_ID
    
    if source_layer_ID_begin == 0:
        source_layer_ID_begin = -1

    for inter_layerID in range(source_layer_ID_begin+2,layerID+1,2):
        
        positive_paths_temp = {}
        negative_paths_temp = {}
        
        inter_layerID_w = int((inter_layerID-1)/2)

            # the following are the layer exact before source layer
        if inter_layerID == source_layer_ID_begin+2:
            
            for inter_node_ID in range(numNodesPerL[inter_layerID_w]):
                
                positive_paths[inter_node_ID] = {}
                
                negative_paths[inter_node_ID] = {}
                
                weight = weights[inter_layerID_w][inter_node_ID][source_node_ID]

                if weight > 0:

                    positive_paths[inter_node_ID] = {str(source_layer_ID)+'_'+str(source_node_ID)+','+str(inter_layerID)+'_'+str(inter_node_ID)+',': weights[inter_layerID_w][inter_node_ID][source_node_ID]}
                
                elif weight < 0:
            
    
                    negative_paths[inter_node_ID] = {str(source_layer_ID)+'_'+str(source_node_ID)+','+str(inter_layerID)+'_'+str(inter_node_ID)+',': weights[inter_layerID_w][inter_node_ID][source_node_ID]}
        
        elif inter_layerID == layerID:
            
            inter_node_ID = nodeID
            
            positive_paths_temp = {}
            negative_paths_temp = {}

            for inter_inter_nodeID in range(numNodesPerL[inter_layerID_w-1]):
                
                weight_temp = weights[inter_layerID_w][inter_node_ID][inter_inter_nodeID]

                for key in positive_paths[inter_inter_nodeID].keys():

                    path_value = positive_paths[inter_inter_nodeID][key]*weight_temp
                    path_name = key+str(inter_layerID)+'_'+str(inter_node_ID)
                    

                    
                    if path_value > 0:
                        positive_paths_temp[path_name] = path_value
                    elif path_value < 0:
                        negative_paths_temp[path_name] = path_value
                        
                for key in negative_paths[inter_inter_nodeID].keys():

                    path_value = negative_paths[inter_inter_nodeID][key]*weight_temp
                    path_name = key+str(inter_layerID)+'_'+str(inter_node_ID)

                    if path_value > 0:
                        positive_paths_temp[path_name] = path_value
                    elif path_value < 0:
                        negative_paths_temp[path_name] = path_value
    
            positive_paths_temp_temp = sorted(positive_paths_temp.items(), key=lambda x:-x[1])
            negative_paths_temp_temp = sorted(negative_paths_temp.items(), key=lambda x:x[1])
            

            positive_paths_temp = {}
            negative_paths_temp = {}
            
            
            count = 0
            for i in range(len(positive_paths_temp_temp)):
                path = positive_paths_temp_temp[i][0] 
                value = positive_paths_temp_temp[i][1]
                positive_paths_temp[path] = value
                count += 1
                if count >= record_at_most_number:
                    break
                
            count = 0
            for i in range(len(negative_paths_temp_temp)):
                path = negative_paths_temp_temp[i][0] 
                value = negative_paths_temp_temp[i][1]
                negative_paths_temp[path] = value
                count += 1
                if count >= record_at_most_number:
                    break
            positive_paths = copy.deepcopy(positive_paths_temp)
            negative_paths = copy.deepcopy(negative_paths_temp)
            

        else:

            for inter_node_ID in range(numNodesPerL[inter_layerID_w]):

            # the following are layers after the layer exactly before the source layer
            


                positive_paths_temp[inter_node_ID] = {}
                negative_paths_temp[inter_node_ID] = {}
                
                for inter_inter_nodeID in range(numNodesPerL[inter_layerID_w-1]):
                    
                    weight_temp = weights[inter_layerID_w][inter_node_ID][inter_inter_nodeID]
                    
                    
                    for key in positive_paths[inter_inter_nodeID].keys():

                        path_value = positive_paths[inter_inter_nodeID][key]*weight_temp
                        path_name = key+str(inter_layerID)+'_'+str(inter_node_ID)+','
                        
                        if path_value > 0:
                            positive_paths_temp[inter_node_ID][path_name] = path_value
                        elif path_value < 0:
                            negative_paths_temp[inter_node_ID][path_name] = path_value
                            
                positive_paths_temp_temp = sorted(positive_paths_temp[inter_node_ID].items(), key=lambda x:-x[1])
                negative_paths_temp_temp = sorted(negative_paths_temp[inter_node_ID].items(), key=lambda x:x[1])
                
                positive_paths_temp[inter_node_ID] = {}
                negative_paths_temp[inter_node_ID] = {}
                
                
                count = 0
                for i in range(len(positive_paths_temp_temp)):
                    path = positive_paths_temp_temp[i][0] 
                    value = positive_paths_temp_temp[i][1]
                    positive_paths_temp[inter_node_ID][path] = value
                    count += 1
                    if count >= record_at_most_number:
                        break
                    
                count = 0
                for i in range(len(negative_paths_temp_temp)):
                    path = negative_paths_temp_temp[i][0] 
                    value = negative_paths_temp_temp[i][1]
                    negative_paths_temp[inter_node_ID][path] = value
                    count += 1
                    if count >= record_at_most_number:
                        break
                    

                    
            positive_paths = copy.deepcopy(positive_paths_temp)
            negative_paths = copy.deepcopy(negative_paths_temp)

    return positive_paths, negative_paths


def random_decide_binary_nodes_new(dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize):
    
    open_relu = []

    return open_relu


def Node_to_Id(source_node):
    
    layer_node = source_node.split('_')
    
    layerID_source = int(layer_node[0])
    nodeID_source = int(layer_node[1])
    
    return nodeID_source



def decide_binary_nodes_new_5l(Loop_para_process, dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize,Bootstrap_data_2):
    
    open_relu = []

    
    global total_len3
    
    local_open_limit = Loop_para_process['open_limit']
    
    
    
    if layerID >= 5:
        
        
        # first, collect paths of len 4
    
        
        Bootstrap_section = Bootstrap_data_2[str(layerID)+'_'+str(nodeID)]

        source_layer_ID = max(0, layerID - 6)
        
        
        if source_layer_ID < 0:
            node_num = inputSize
        else:
            node_num = numNodesPerL[int((source_layer_ID-1)/2)]
        
        
        bbprime_positive = {}
        
        
        for source_node_ID in range(node_num):
            
    
            
            bound_source_node_ID = dict_bounds[str(source_layer_ID)+'_'+str(source_node_ID)]['UB']
            
            for key1 in Bootstrap_section[source_node_ID][0]:
                
                if not analysis_active_path(key1,dict_bounds):
                    continue
                nodes_1 = analysis_path(key1)
            
                
            
                for key2 in Bootstrap_section[source_node_ID][1]:
                    if not analysis_active_path(key2,dict_bounds):
                        continue
                    nodes_2 = analysis_path(key2)
                    
                    key_paths = nodes_1[1]+','+nodes_1[2]+';'+nodes_2[1]+','+nodes_2[2]
                    
                    

                    temp_value_1 = bound_source_node_ID*Bootstrap_section[source_node_ID][0][key1]
                    temp_value_2 = -bound_source_node_ID*Bootstrap_section[source_node_ID][1][key2]
                    
                    if key_paths not in bbprime_positive.keys():
                        bbprime_positive[key_paths] = 0
                    
                    bbprime_positive[key_paths] += min(temp_value_1,temp_value_2)
        
        
        
        for key in bbprime_positive.keys():
            

            
            
            nodes_1,nodes_2 = analysis_two_path(key)
            
            if len(nodes_1) == 2:
            
                node_b = nodes_1[0]
                
                node_bp = nodes_2[0]
                
                node_c = nodes_1[1]
                
                node_cp = nodes_2[1]
    
                
                bv = max(0,dict_bounds[node_b]['UB'])*abs(weights[1][Node_to_Id(node_c)][Node_to_Id(node_b)])*abs(weights[2][nodeID][Node_to_Id(node_c)])
      
                bpv = max(0,dict_bounds[node_bp]['UB'])*abs(weights[1][Node_to_Id(node_cp)][Node_to_Id(node_bp)])*abs(weights[2][nodeID][Node_to_Id(node_cp)])
                
                cv = max(0,dict_bounds[node_c]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_c)])
                
                cpv = max(0,dict_bounds[node_cp]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_cp)])
                
                bbprime_positive[key] = min(bbprime_positive[key],bv,bpv,cv,cpv)
                
                
        bbprime_final = {}
        
        for key in bbprime_positive.keys():
            nodes_1,nodes_2 = analysis_two_path(key)
            
            new_key = nodes_1[0]+';'+nodes_2[0]
            
            if new_key in bbprime_final.keys():
                bbprime_final[new_key] = max(bbprime_final[new_key],bbprime_positive[key])
            else:
                bbprime_final[new_key] = bbprime_positive[key]

        
        
                    

        
        
        summary_len_4 = sorted(bbprime_final.items(), key=lambda x:x[1])          
       
                    
       
        # then deal with paths of len 3
       
        
        ccprime_positive = {}
        Bootstrap_section = Bootstrap_data[str(layerID)+'_'+str(nodeID)]

        source_layer_ID = layerID - 4


        for source_node_ID in range(node_num):
            
    
            
            bound_source_node_ID = dict_bounds[str(source_layer_ID)+'_'+str(source_node_ID)]['UB']
            
            for key1 in Bootstrap_section[source_node_ID][0]:
                
                if not analysis_active_path(key1,dict_bounds):
                    continue
                nodes_1 = analysis_path(key1)
            
                
            
                for key2 in Bootstrap_section[source_node_ID][1]:
                    if not analysis_active_path(key2,dict_bounds):
                        continue
                    nodes_2 = analysis_path(key2)
                    
                    key_paths = nodes_1[1]+';'+nodes_2[1]
                    
                    

                    temp_value_1 = bound_source_node_ID*Bootstrap_section[source_node_ID][0][key1]
                    temp_value_2 = -bound_source_node_ID*Bootstrap_section[source_node_ID][1][key2]
                    
                    if key_paths not in ccprime_positive.keys():
                        ccprime_positive[key_paths] = 0
                    
                    ccprime_positive[key_paths] += min(temp_value_1,temp_value_2)
        
        

    
    
   
        
        for key in ccprime_positive.keys():
            
    
            nodes_1,nodes_2 = analysis_two_path(key)
            
            if len(nodes_1) == 2:
            
                node_b = nodes_1[0]
                
                
                
            elif len(nodes_1) == 1:
                
                
                node_c = nodes_1[0]
                
                node_cp = nodes_2[0]

    
                
                cv = max(0,dict_bounds[node_c]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_c)])
      
                cpv = max(0,dict_bounds[node_cp]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_cp)])

                
                ccprime_positive[key] = min(ccprime_positive[key],cv,cpv)
                
                # if ccprime_positive[key] > len3_record[0]:
                #     len3_record[0] = ccprime_positive[key]
                #     len3_record[1] = key
                
        summary_len_3 = sorted(ccprime_positive.items(), key=lambda x:x[1]) 


        # with  open(output_name,'at') as f:
        #     f.write('Node: '+str(nodeID)+', Len3 max value: '+str(len3_record[0])+', Len4 max value: '+str(len4_record[0])+','+len4_record[1]+', the Ratio: '+str(len4_record[0]/len3_record[0])+'\n')
    


        count= 0
        len3_count = 0
        
        while len(summary_len_3)+len(summary_len_4)>0:
            
            in_loop_count = 0

            
            if summary_len_3 != []:
                value_1 = summary_len_3[-1][1]
            else:
                value_1 = -1
            
            
            if summary_len_4 != []:
                value_2 = summary_len_4[-1][1]*len3_count
            else:
                value_2 = -1
            
            

            
            
            if value_1 > value_2:
                nodes_choice = summary_len_3[-1][0]
                len3_cof = 1
                summary_len_3.pop()
            else:
                nodes_choice = summary_len_4[-1][0]
                # print(nodes_choice)
                len3_cof = 0
                summary_len_4.pop()
            
            
            
            nodes_1,nodes_2 = analysis_two_path(nodes_choice)
    
            
            # if dict_bounds[nodes_1[0]]['LB'] < 0  and dict_bounds[nodes_2[0]]['LB'] < 0:
                
            for node in nodes_1+nodes_2:
                
                if count >= local_open_limit:

                    
                    # with  open(output_name,'at') as f:
                    #     f.write('Node: '+str(nodeID)+', Last chosen nodes value: '+str(summary_temp[i][1])+', Len 4 max value: '+str(len4_record[0])+', the ratio: '+str(len4_record[0]/summary_temp[i][1])+'\n')
                    #     f.write('\n')
                    
                    
                    
                    # total_len3 += len3_count
                    # print(value_1,value_2)
                    # print(open_relu)
                    # print('\n')
                    return open_relu
                    
    
                if node not in open_relu and dict_bounds[node]['LB'] < 0:
                    open_relu.append(node)
                    count += 1
                    in_loop_count += 1
            len3_count += len3_cof*in_loop_count
            


    
    return open_relu

def decide_binary_nodes_new_3l(Loop_para_process, dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize,Bootstrap_data_2):
    
    
    
    open_relu = []
    
    
    Bootstrap_section = Bootstrap_data[str(layerID)+'_'+str(nodeID)]
    
    local_open_limit = Loop_para_process['open_limit']
    
    ccprime_positive = {}
    Bootstrap_section = Bootstrap_data[str(layerID)+'_'+str(nodeID)]

    source_layer_ID = layerID - 4
    
    if source_layer_ID < 0:
        node_num = inputSize
        source_layer_ID = 0
    else:
        node_num = numNodesPerL[int((source_layer_ID-1)/2)]


    for source_node_ID in range(node_num):
        

        
        bound_source_node_ID = dict_bounds[str(source_layer_ID)+'_'+str(source_node_ID)]['UB']
        
        for key1 in Bootstrap_section[source_node_ID][0]:
            
            if not analysis_active_path(key1,dict_bounds):
                continue
            nodes_1 = analysis_path(key1)
        
            
        
            for key2 in Bootstrap_section[source_node_ID][1]:
                if not analysis_active_path(key2,dict_bounds):
                    continue
                nodes_2 = analysis_path(key2)
                
                key_paths = nodes_1[1]+';'+nodes_2[1]
                
                

                temp_value_1 = bound_source_node_ID*Bootstrap_section[source_node_ID][0][key1]
                temp_value_2 = -bound_source_node_ID*Bootstrap_section[source_node_ID][1][key2]
                
                if key_paths not in ccprime_positive.keys():
                    ccprime_positive[key_paths] = 0
                
                ccprime_positive[key_paths] += min(temp_value_1,temp_value_2)

    for key in ccprime_positive.keys():
        

        nodes_1,nodes_2 = analysis_two_path(key)

        node_c = nodes_1[0]
        
        node_cp = nodes_2[0]


        
        cv = max(0,dict_bounds[node_c]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_c)])
  
        cpv = max(0,dict_bounds[node_cp]['UB'])*abs(weights[2][nodeID][Node_to_Id(node_cp)])

        ccprime_positive[key] = min(ccprime_positive[key],cv,cpv)

            
    summary_len_3 = sorted(ccprime_positive.items(), key=lambda x:-x[1]) 

    count= 0
    
    for i in range(len(summary_len_3)):

        nodes_1,nodes_2 = analysis_two_path(summary_len_3[i][0])

            
        for node in nodes_1+nodes_2:
            
            if count >= local_open_limit:

                return open_relu

            if node not in open_relu and dict_bounds[node]['LB'] < 0:
                open_relu.append(node)
                count += 1

    return open_relu


def decide_binary_nodes_new(Loop_para_process, dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize,Bootstrap_data_2):

    if layerID == 3:
        
        open_relu = decide_binary_nodes_new_3l(Loop_para_process, dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize, Bootstrap_data_2)
    
    elif layerID <= numLayers_Cof*2+1:
     
        open_relu = decide_binary_nodes_new_5l(Loop_para_process, dict_bounds, layerID, nodeID, Bootstrap_data, numNodesPerL, inputSize, Bootstrap_data_2)
    
    # elif layerID == numLayers_Cof*2+1:
        
    #     open_relu = [str(numLayers_Cof*2-1)+'_'+str(i) for i in range(10)]+[str(numLayers_Cof*2-3)+'_'+str(i) for i in range(100)]
        
    return open_relu





def analysis_path(path):
    
    nodes = path.split(',')
    
    
    return nodes

def analysis_active_path(path,dict_bounds):

    nodes = path.split(',')
    
    nodes = nodes[:-1]
    
    active = True
    
    for node in nodes:
        if dict_bounds[node]['UB'] <= 0:
            active = False
            break
    return active


def analysis_two_path(path):
    
    paths = path.split(';')
    
    nodes_1 = paths[0].split(',')
    nodes_2 = paths[1].split(',')
    
    
    return nodes_1,nodes_2



class gurobi_multiprocess_dict:

    def __setitem__(self, idx, value):
    
        self.data[idx] = value
        
  
        
def Do_break_loop(ObjVal, ObjBoundC, max_or_min):
    
    break_loop = True
    
    if max_or_min == 'Max':
        if ObjVal < 0 and ObjBoundC > 0 and abs(ObjVal) > 0.3*abs(ObjBoundC):
            break_loop = False
    
    if max_or_min == 'Min':
        if ObjVal > 0 and ObjBoundC < 0 and abs(ObjVal) > 0.3*abs(ObjBoundC):
            break_loop = False

    
    return break_loop
    
  
    
  
    
  
    
  
    
  
    
        
        
        
def gurobi_r2_loop(model_r2, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias):
    
    
    r2_fail_max = 0

    start_time_max = time.time()

    model_r2 = gurobi_max(model_r2, layerID, nodeID)     # setting objective

    sys.stdout.flush()
    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
    model_r2.Params.TimeLimit = Loop_para_process['Fist_timeout_para_r2']
    model_r2.optimize()
    this_time = time.time()
    sys.stdout.flush()

    if model_r2.SolCount > 0:
        # if model_0.SolCount, it means model_0 fail to provide information, we turn to Diamond model
        
        old_gap = model_r2.MIPGap
        
        print_stuff += '1st Max r_2 time: '
        print_stuff += str(this_time-start_time_max)
        print_stuff += '\n'
        print_stuff += '1st Max r_2 Gap: '
        print_stuff += str(old_gap)
        print_stuff += '\n'

        if old_gap > Loop_para_process['Second_Gap_para_r2'] and model_r2.ObjBoundC > 0: # check if MIP.Gap is good enough. If not, we do iteration to improve the MIP.Gap 

            try_times = 2
            
            print_stuff += 'enter loop'
            print_stuff += '\n'
            
            break_loop_gap = 0
            break_loop = True
            time_rest = 5   

            while try_times <= 99:
                
                model_r2.Params.TimeLimit = Loop_para_process['Timeout_step_para_r2']
                if try_times == 2:
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                else:
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                sys.stdout.flush()
                model_r2.optimize()
                this_time = time.time()
                sys.stdout.flush()
                
                new_gap = model_r2.MIPGap
                
                print_stuff += str(try_times)+'th Max r_2 total time: '
                print_stuff += str(this_time-start_time_max)
                print_stuff += '\n'
                print_stuff += str(try_times)+'th Max r_2 Gap: '
                print_stuff += str(new_gap)
                print_stuff += '\n'
                
                try_times += 1   # Otherwise, there is hope to improve the Gap futher. 
                
                
                break_loop = Do_break_loop(model_r2.ObjVal,model_r2.ObjBoundC,'Max')
                
                # Now check whether we need to break the loop by the rule of Gap
                
                
                if model_r2.ObjBoundC < 0 or new_gap == 0:
                    break_loop_gap = 1
                
                elif new_gap <= Loop_para_process['Second_Gap_para_r2']:
                    break_loop_gap = 2
                
                elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para_r2']):
                    break_loop_gap = 3
                
                
                # Do the operation of breaking the loop
                
                old_gap = new_gap
                
                if break_loop_gap == 1:
                    break
                
                elif break_loop_gap >= 2:
                    
                    if break_loop == True:
                        break
                    
                    elif break_loop == False:
                        time_rest += -1
                        if time_rest <= 0:
                            break
                
                
                
                
                # if model_r2.ObjBoundC<0 or new_gap==0:
                #     break
                
                # if new_gap <= Loop_para_process['Second_Gap_para_r2'] and break_loop == True: # if so, the new Gap is good enough.
                #     break                                
                # elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para_r2']) and break_loop == True: # if so, the improvement is too small and we do not try again.
                #     break
                
                

            if new_gap > Loop_para_process['Third_Gap_para_r2'] and model_r2.ObjBoundC > 0: 
                
                # If so, we try to improve the Gap to <= 0.1.
                
                print_stuff += 'enter loop r_2'
                print_stuff += '\n'

                # while timeout_set <= 99:
                    
                    
                break_loop_gap = 0
                break_loop = True
                time_rest = 5   
                
                while try_times <= 99:
                    
                    model_r2.Params.TimeLimit = Loop_para_process['Timeout_step_para_r2']
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                    sys.stdout.flush()
                    model_r2.optimize()
                    this_time = time.time()
                    sys.stdout.flush()
                    
                    new_gap = model_r2.MIPGap
                    
                    print_stuff += str(try_times)+'th Max r_2 total time: '
                    print_stuff += str(this_time-start_time_max)
                    print_stuff += '\n'
                    print_stuff += str(try_times)+'th Max r_2 Gap: '
                    print_stuff += str(new_gap)
                    print_stuff += '\n'
                    
                    try_times += 1
                    
                    break_loop = Do_break_loop(model_r2.ObjVal,model_r2.ObjBoundC,'Max')
                    
                    
                    # Now check whether we need to break the loop by the rule of Gap
                    
                    
                    if model_r2.ObjBoundC < 0 or new_gap == 0:
                        break_loop_gap = 1
                    
                    elif new_gap <= Loop_para_process['Third_Gap_para_r2']:
                        break_loop_gap = 2
                    
                    elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para_r2']):
                        break_loop_gap = 3
                    
                    
                    # Do the operation of breaking the loop
                    
                    old_gap = new_gap
                    
                    if break_loop_gap == 1:
                        break
                    
                    elif break_loop_gap == 2:
                        
                        if break_loop == True:
                            break
                        
                        elif break_loop == False:
                            time_rest += -1
                            if time_rest <= 0:
                                break
                    
                    elif break_loop_gap == 3:
                        
                        if break_loop == True:
                            r2_fail_max = 1
                            break
                        
                        elif break_loop == False:
                            time_rest += -1
                            if time_rest <= 0:
                                r2_fail_max = 1
                                break
                    
                    # if model_r2.ObjBoundC<0 or new_gap==0:
                    #     break
                    
                    # if new_gap <= Loop_para_process['Third_Gap_para_r2'] and break_loop == True: # if so, the new Gap is good enough.
                    #     break
                    
                    # elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para_r2']) and break_loop == True: # if so, the improvement is quite small and we do not try again.
                    #     r2_fail_max = 1
                    #     break 
                    
                    
                    
        max_bound_r2 = model_r2.ObjBoundC
                
    else:
        r2_fail_max = 1
        max_bound_r2 = model_r2.ObjBoundC
        print_stuff += 'Have not found solution for layer model Max'
        print_stuff += '\n'
        print_stuff += '\n'

    if r2_fail_max != 1:
        
        print_stuff += 'Used layer model for Max r_2'
        print_stuff += '\n'
        print_stuff += 'MIPGap for max r_2: '
        print_stuff += str(model_r2.MIPGap)
        print_stuff += '\n'
        print_stuff += '\n'
    
        
        model_r2.reset()
        
    r2_max_time = time.time()-start_time_max
        
######################################################################################################

          # start r=2 min part

######################################################################################################    

    r2_fail_min = 0
    


    start_time_min = time.time()

    model_r2 = gurobi_min(model_r2, layerID, nodeID)     # setting objective

    sys.stdout.flush()
    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
    model_r2.Params.TimeLimit = Loop_para_process['Fist_timeout_para_r2']
    model_r2.optimize()
    this_time = time.time()
    sys.stdout.flush()

    if model_r2.SolCount > 0:
        # if model_0.SolCount, it means model_0 fail to provide information, we turn to Diamond model                 
        old_gap = model_r2.MIPGap
        
        print_stuff += '1st min time r_2: '
        print_stuff += str(this_time-start_time_min)
        print_stuff += '\n'
        print_stuff += '1st min Gap r_2: '
        print_stuff += str(old_gap)
        print_stuff += '\n'
   

        if old_gap > Loop_para_process['Second_Gap_para_r2']: # check if MIP.Gap is good enough. If not, we do iteration to improve the MIP.Gap 
 
            try_times = 2       
            print_stuff += 'enter loop r_2'
            print_stuff += '\n'
            
            break_loop_gap = 0
            break_loop = True
            time_rest = 5   

            while try_times <= 99:
                
                model_r2.Params.TimeLimit = Loop_para_process['Timeout_step_para_r2']
                if try_times == 2:
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                else:
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                sys.stdout.flush()
                model_r2.optimize()
                this_time = time.time()
                sys.stdout.flush()
                
                new_gap = model_r2.MIPGap
                
                print_stuff += str(try_times)+'th min r_2 total time: '
                print_stuff += str(this_time-start_time_min)
                print_stuff += '\n'
                print_stuff += str(try_times)+'th min r_2 Gap: '
                print_stuff += str(new_gap)
                print_stuff += '\n'
                
                try_times += 1   # Otherwise, there is hope to improve the Gap futher. 
                
                
                break_loop = Do_break_loop(model_r2.ObjVal, model_r2.ObjBoundC,'Min')
                
                # Now check whether we need to break the loop by the rule of Gap
                
                
                if new_gap == 0:
                    break_loop_gap = 1
                
                elif new_gap <= Loop_para_process['Second_Gap_para_r2']:
                    break_loop_gap = 2
                
                elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para_r2']):
                    break_loop_gap = 3
                
                
                # Do the operation of breaking the loop
                
                old_gap = new_gap
                
                if break_loop_gap == 1:
                    break
                
                elif break_loop_gap >= 2:
                    
                    if break_loop == True:
                        break
                    
                    elif break_loop == False:
                        time_rest += -1
                        if time_rest <= 0:
                            break
                
                # if new_gap == 0:
                #     break
                
                # if new_gap <= Loop_para_process['Second_Gap_para_r2'] and break_loop == True: # if so, the new Gap is good enough.
                #     break                                
                # elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para_r2']) and break_loop == True: # if so, the improvement is too small and we do not try again.
                #     break
                
                

            if new_gap > Loop_para_process['Third_Gap_para_r2']: 
                
                # If so, we try to improve the Gap to <= 0.1.
                
                print_stuff += 'enter loop r_2'
                print_stuff += '\n'

                # while timeout_set <= 99:
                    
                    
                break_loop_gap = 0
                break_loop = True
                time_rest = 5   
                while try_times <= 99:
                    
                    model_r2.Params.TimeLimit = Loop_para_process['Timeout_step_para_r2']
                    model_r2.setParam('MIPGap', Loop_para_process['First_Gap_para_r2'])
                    sys.stdout.flush()
                    model_r2.optimize()
                    this_time = time.time()
                    sys.stdout.flush()
                    
                    new_gap = model_r2.MIPGap
                    
                    print_stuff += str(try_times)+'th min r_2 total time: '
                    print_stuff += str(this_time-start_time_min)
                    print_stuff += '\n'
                    print_stuff += str(try_times)+'th min r_2 Gap: '
                    print_stuff += str(new_gap)
                    print_stuff += '\n'
                    
                    try_times += 1
                    
                    break_loop = Do_break_loop(model_r2.ObjVal,model_r2.ObjBoundC,'Min')
                    
                    # Now check whether we need to break the loop by the rule of Gap
                    
                    
                    if  new_gap == 0:
                        break_loop_gap = 1
                    
                    elif new_gap <= Loop_para_process['Third_Gap_para_r2']:
                        break_loop_gap = 2
                    
                    elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para_r2']):
                        break_loop_gap = 3
                    
                    
                    # Do the operation of breaking the loop
                    
                    old_gap = new_gap
                    
                    if break_loop_gap == 1:
                        break
                    
                    elif break_loop_gap == 2:
                        
                        if break_loop == True:
                            break
                        
                        elif break_loop == False:
                            time_rest += -1
                            if time_rest <= 0:
                                break
                    
                    elif break_loop_gap == 3:
                        
                        if break_loop == True:
                            r2_fail_min = 1
                            break
                        
                        elif break_loop == False:
                            time_rest += -1
                            if time_rest <= 0:
                                r2_fail_min = 1
                                break
                    
                    # if new_gap == 0:
                    #     break
                    
                    # if new_gap <= Loop_para_process['Third_Gap_para_r2'] and break_loop == True: # if so, the new Gap is good enough.
                    #     break
                    
                    # elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para_r2']) and break_loop == True: # if so, the improvement is quite small and we do not try again.
                    #     r2_fail_min = 1
                    #     break 
                    
                    
                    
        min_bound_r2 = model_r2.ObjBoundC
        
        if model_r2.ObjVal > 0:
            re_do_min_last_layer = True
        else:
            re_do_min_last_layer = False

                
    else:
        r2_fail_min = 1
        min_bound_r2 = model_r2.ObjBoundC
        print_stuff += 'Have not found solution for layer model Min'
        print_stuff += '\n'
        print_stuff += '\n'
        re_do_min_last_layer = True

    if r2_fail_min != 1:
        
        print_stuff += 'Used layer model for min r_2'
        print_stuff += '\n'
        print_stuff += 'MIPGap for min r_2: '
        print_stuff += str(model_r2.MIPGap)
        print_stuff += '\n'
        print_stuff += '\n'

        
        if layerID < numLayers_Cof*2+1:
            model_r2.reset()
        
    r2_min_time = time.time()-start_time_min

        
    return print_stuff, r2_max_time, r2_min_time, r2_fail_max, r2_fail_min, min_bound_r2, max_bound_r2, re_do_min_last_layer



def gurobi_loop_max(model_0, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias):
    
    
    
    start_time_max = time.time()
    print_stuff += 'enter Diamond model Max\n'

    model_0 = gurobi_max(model_0, layerID, nodeID)     # setting objective

    sys.stdout.flush()
    model_0.setParam('MIPGap', Loop_para_process['First_Gap_para'])
    model_0.Params.TimeLimit = Loop_para_process['Fist_timeout_para']
    model_0.optimize()
    this_time = time.time()
    sys.stdout.flush()

    flag_need_diamond_max = 0

    if model_0.SolCount > 0:

        # if model_0.SolCount, it means model_0 fail to provide information, we turn to Diamond model
        
        old_gap = model_0.MIPGap
        
        print_stuff += '1st Max time: '
        print_stuff += str(this_time-start_time_max)
        print_stuff += '\n'
        print_stuff += '1st Max Gap: '
        print_stuff += str(old_gap)
        print_stuff += '\n'

        if old_gap > Loop_para_process['Second_Gap_para'] and model_0.ObjBoundC > 0: # check if MIP.Gap is good enough. If not, we do iteration to improve the MIP.Gap 

            try_times = 2
            
            print_stuff += 'enter loop'
            print_stuff += '\n'
            
            break_loop_gap = 0
            break_loop = True
            time_rest = 5   
   
            while try_times <= 99:
                
                model_0.Params.TimeLimit = Loop_para_process['Timeout_step_para']
                if try_times == 2:
                    model_0.setParam('MIPGap', Loop_para_process['First_Gap_para'])
                else:
                    model_0.setParam('MIPGap', Loop_para_process['Second_Gap_para'])
                sys.stdout.flush()
                model_0.optimize()
                this_time = time.time()
                sys.stdout.flush()
                
                new_gap = model_0.MIPGap
                
                print_stuff += str(try_times)+'th Max total time: '
                print_stuff += str(this_time-start_time_max)
                print_stuff += '\n'
                print_stuff += str(try_times)+'th Max Gap: '
                print_stuff += str(new_gap)
                print_stuff += '\n'
                
                try_times += 1   # Otherwise, there is hope to improve the Gap futher. 
                
                break_loop = True
                
                break_loop = Do_break_loop(model_0.ObjVal,model_0.ObjBoundC,'Max')
                
                
                if model_0.ObjBoundC < 0 or new_gap==0:
                    break
                
                if new_gap <= Loop_para_process['Second_Gap_para'] and break_loop == True: # if so, the new Gap is good enough.
                    break
                
                elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para']) and break_loop == True: # if so, the improvement is too small and we do not try again.
                    break

                old_gap = new_gap
   
                
            if new_gap > Loop_para_process['Third_Gap_para'] and model_0.ObjBoundC > 0: 
                
                # If so, we try to improve the Gap to <= 0.1.
                
                print_stuff += 'enter loop'
                print_stuff += '\n'

                # while timeout_set <= 99:
                    
                break_loop_gap = 0
                break_loop = True
                time_rest = 5   
                
                while try_times <= 99:
                    
                    model_0.Params.TimeLimit = Loop_para_process['Timeout_step_para']
                    model_0.setParam('MIPGap', Loop_para_process['Third_Gap_para'])
                    sys.stdout.flush()
                    model_0.optimize()
                    this_time = time.time()
                    sys.stdout.flush()
                    
                    new_gap = model_0.MIPGap
                    
                    print_stuff += str(try_times)+'th Max total time: '
                    print_stuff += str(this_time-start_time_max)
                    print_stuff += '\n'
                    print_stuff += str(try_times)+'th Max Gap: '
                    print_stuff += str(new_gap)
                    print_stuff += '\n'
                    
                    try_times += 1
                    
                    break_loop = True
                    
                    break_loop = Do_break_loop(model_0.ObjVal,model_0.ObjBoundC,'Max')
                    
                    if model_0.ObjBoundC<0 or new_gap==0:
                        break
                    
                    if new_gap <= Loop_para_process['Third_Gap_para'] and break_loop == True: # if so, the new Gap is good enough.
                        break
                    
                    elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para']) and break_loop == True: # if so, the improvement is quite small and we do not try again.
                        
                        if model_0.MIPGap > Loop_para_process['Call_Diamond_para']:  # if so, the Gap is too bad and we must call Diamond model
                            flag_need_diamond_max = 1
                        break
   
                    old_gap = new_gap

        # flag_need_diamond_max means if call Diamond model or not. If it is 0, then we get a relatively good MIP.Gap
        print_stuff += '\n'
                            
    else:
        flag_need_diamond_max = 1
        print_stuff += 'Have not found solution for Diamond model Max'
        print_stuff += '\n'
        print_stuff += '\n'

    return print_stuff, flag_need_diamond_max








def gurobi_loop_min(model_0, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias):
    
    
    print_stuff += 'enter Diamond model Min\n'
    start_time_min = time.time()    
    
    model_0 = gurobi_min(model_0, layerID, nodeID)     # setting objective
    
    
    sys.stdout.flush()
    model_0.setParam('MIPGap', Loop_para_process['First_Gap_para'])
    model_0.Params.TimeLimit = Loop_para_process['Fist_timeout_para']
    model_0.optimize()
    this_time_min = time.time()
    sys.stdout.flush()
    
    flag_need_diamond_min = 0
    
    if model_0.SolCount > 0:
        
        # if model_0.SolCount, it means model_0 fail to provide information, we turn to Diamond model
    
        old_gap = model_0.MIPGap
        
        print_stuff += '1st Min time: '
        print_stuff += str(this_time_min-start_time_min)
        print_stuff += '\n'
        print_stuff += '1st Min Gap: '
        print_stuff += str(old_gap)
        print_stuff += '\n'
    
        if old_gap > Loop_para_process['Second_Gap_para']: # check if MIP.Gap is good enough. If not, we do iteration to improve the MIP.Gap
    
            try_times = 2
            
            print_stuff += 'enter loop'
            print_stuff += '\n'
            
            break_loop_gap = 0
            break_loop = True
            time_rest = 5   
    
            while try_times <= 99:
                
                model_0.Params.TimeLimit = Loop_para_process['Timeout_step_para']
                if try_times == 2:
                    model_0.setParam('MIPGap', Loop_para_process['First_Gap_para'])
                else:
                    model_0.setParam('MIPGap', Loop_para_process['Second_Gap_para'])
                sys.stdout.flush()
                model_0.optimize()
                this_time = time.time()
                sys.stdout.flush()
                
                new_gap = model_0.MIPGap
                
                print_stuff += str(try_times)+'th Min total time: '
                print_stuff += str(this_time-start_time_min)
                print_stuff += '\n'
                print_stuff += str(try_times)+'th Min Gap: '
                print_stuff += str(new_gap)
                print_stuff += '\n'
                
                try_times += 1
                if new_gap == 0:
                    break
                
                break_loop = True
                
                break_loop = Do_break_loop(model_0.ObjVal,model_0.ObjBoundC,'Min')
                
                if new_gap <= Loop_para_process['Second_Gap_para'] and break_loop == True: # if so, the new Gap is good enough.
                    break
                
                elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Second_Gap_para']) and break_loop == True: # if so, the improvement is too small and we do not try again.
                    break
    
                   # Otherwise, there is hope to improve the Gap futher. 
                
                old_gap = new_gap
    
            if new_gap > Loop_para_process['Third_Gap_para']:   # If so, we try to improve the Gap to <= 0.1.
            
                print_stuff += 'enter loop'
                print_stuff += '\n'
    
                # while timeout_set <= 99:
                    
                break_loop_gap = 0
                break_loop = True
                time_rest = 5   
                
                while try_times <= 99:
                    
                    model_0.Params.TimeLimit = Loop_para_process['Timeout_step_para']
                    model_0.setParam('MIPGap', Loop_para_process['Third_Gap_para'])
                    sys.stdout.flush()
                    model_0.optimize()
                    this_time = time.time()
                    sys.stdout.flush()
                    
                    new_gap = model_0.MIPGap
                    
                    print_stuff += str(try_times)+'th Min total time: '
                    print_stuff += str(this_time-start_time_min)
                    print_stuff += '\n'
                    print_stuff += str(try_times)+'th Min Gap: '
                    print_stuff += str(new_gap)
                    print_stuff += '\n'
                    
                    try_times += 1
                    
                    break_loop = True
                    
                    break_loop = Do_break_loop(model_0.ObjVal,model_0.ObjBoundC,'Min')
                    
                    if new_gap == 0:
                        break
                    
                    if new_gap <= Loop_para_process['Third_Gap_para'] and break_loop == True: # if so, the new Gap is good enough.
                        break
                    
                    elif old_gap-new_gap < Loop_para_process['Strict_para']*(new_gap-Loop_para_process['Third_Gap_para']) and break_loop == True: # if so, the improvement is quite small and we do not try again.
                        
                        if model_0.MIPGap > Loop_para_process['Call_Diamond_para']:  # if so, the Gap is too bad and we must call Diamond model
                            flag_need_diamond_min = 1
                        break
    
                    old_gap = new_gap
    
        # flag_need_diamond_max means if call Diamond model or not. If it is 0, then we get a relatively good MIP.Gap
        print_stuff += '\n'

    else:
        flag_need_diamond_min = 1
        print_stuff += 'Have not found solution for Diamond model Min'
        print_stuff += '\n'
        print_stuff += '\n'
        
    
    return print_stuff, flag_need_diamond_min








def gurobi_master_single_node(imgID, Loop_para_process, input, layerID, nodeID_start, num_nodes_per_process, reluNum, stepBack, dict_bounds_gurobi, dict_bounds_gurobi_manager, 
                              numNodesPerL, bootstrap_ctype, inputSize, weights, bias, mipf, epsilon):
    
    process_start_time = time.time()
    
    
    last_node = nodeID_start+num_nodes_per_process
    
    
    # [Bootstrap_data_2, Bootstrap_data_3 ] = pickle.loads(bootstrap_ctype)
    [Bootstrap_data_2, Bootstrap_data_3 ] = bootstrap_ctype

    

    if numNodesPerL[layerID-1] < last_node:
        last_node = numNodesPerL[layerID-1]
        
    time_env = time.time()
    

    # with gp.Env() as env, gp.Model(env=env) as model:
    with gp.Env() as env, gp.Model(env=env) as model_0:

        model_r2, counter_integer_variable_r2, time_initialize = gurobi_initialize_model_layer_new(numLayers_Cof,input, inputSize, numNodesPerL, epsilon, 
                                                                            dict_bounds_gurobi, layerID, reluNum+1, weights, bias, mipf, stepBack)
        
        print_stuff = ''
        print_stuff_on_model = ''

        print_stuff_on_model += '\n'
        print_stuff_on_model += '---------------------------'
        print_stuff_on_model += '\n'
        print_stuff_on_model += 'Time taken to initialize Gurobi model for layerID: '
        print_stuff_on_model += str(layerID)
        print_stuff_on_model += '\n'
        print_stuff_on_model += str(time_initialize)
        print_stuff_on_model += '\n'
        print_stuff_on_model += '---------------------------'
        print_stuff_on_model += '\n'

        print_stuff_on_model += '\n'
        print_stuff_on_model += '---------------------------'
        print_stuff_on_model += '\n'
        print_stuff_on_model += 'Time taken to initialize model + env for layerID: '
        print_stuff_on_model += str(layerID)
        print_stuff_on_model += '\n'
        print_stuff_on_model += str(time.time() - time_env)
        print_stuff_on_model += '\n'
        print_stuff_on_model += '---------------------------'
        print_stuff_on_model += '\n'


        # for nodeID in range(nodeID_start, nodeID_start+num_nodes_per_process,1):

        if layerID >= 3:
            
            
            num_binary_nodes = dict_bounds_gurobi['layer_binary'+str(layerID-2)]
            
            if layerID >= 5:
                num_binary_nodes += dict_bounds_gurobi['layer_binary'+str(layerID-4)]
            
            # num_binary_nodes = 0
            # for j in range(numNodesPerL[layerID-2-1]):
            #     if dict_bounds_gurobi[str(layerID-2)+'_'+str(j)]['UB'] > 0 and dict_bounds_gurobi[str(layerID-2)+'_'+str(j)]['LB'] < 0:
            #         num_binary_nodes +=1
            
            print_stuff_on_model += '\n'
            print_stuff_on_model += '---------------------------'
            print_stuff_on_model += '\n'
            print_stuff_on_model += 'Number of Binary nodes for layerID : '
            print_stuff_on_model += str(layerID)
            print_stuff_on_model += '\n'
            print_stuff_on_model += str(num_binary_nodes)
            print_stuff_on_model += '\n'
            print_stuff_on_model += '---------------------------'
            print_stuff_on_model += '\n'

        if layerID == 1:
            
            model_0 = model_r2

            # model_0, counter_integer_variable_1, time_initialize = gurobi_initialize_model_layer(input, inputSize, numNodesPerL, epsilon, 
            #                                                                     dict_bounds_gurobi, layerID, reluNum, weights, bias, mipf, stepBack)

            for nodeID in range(nodeID_start, last_node, 1):
            
                sys.stdout.flush()

                print_stuff += '\n'
                print_stuff += 'reluNum : '
                print_stuff += str(reluNum)
                print_stuff += '\n'
                print_stuff += 'stepBack: '
                print_stuff += str(stepBack)
                print_stuff += '\n'
                print_stuff += 'layerID : '
                print_stuff += str(layerID)
                print_stuff += '\n'
                print_stuff += 'nodeID: '
                print_stuff += str(nodeID)
                print_stuff += '\n'

                
                start_time_max = time.time()
                # num_times_in_while_loop_max = 0
                model_0 = gurobi_max(model_0, layerID, nodeID)     # setting objective
                sys.stdout.flush()
                model_0.optimize()
                sys.stdout.flush()
                max_nodeID = model_0.getObjective().getValue()
                model_0.reset(0)
                time_max_node = time.time() - start_time_max

                start_time_min = time.time()
                # num_times_in_while_loop_min = 0
                model_0 = gurobi_min(model_0, layerID, nodeID)     # setting objective
                sys.stdout.flush()
                model_0.optimize()
                sys.stdout.flush()
                min_nodeID = model_0.getObjective().getValue()
                model_0.reset(0)
                time_min_node = time.time() - start_time_max

                
                sys.stdout.flush()

                print_stuff += 'time for max total = '
                print_stuff += str(time_max_node)
                print_stuff += '\n'
                print_stuff += 'time for min total = '
                print_stuff += str(time_min_node)
                print_stuff += '\n'
                print_stuff += 'max for node = '
                print_stuff += str(max_nodeID)
                print_stuff += '\n'
                print_stuff += 'min for node = '
                print_stuff += str(min_nodeID)
                print_stuff += '\n'
                print_stuff += str('DONEEEEEEE')
                print_stuff += '\n'


                dict_bounds_gurobi_manager[str(layerID)+'_'+str(nodeID)] = {'LB' : min_nodeID, 'UB' : max_nodeID, 'time_max_total' : time_max_node,
                    'time_min_total' : time_min_node, 'print' : print_stuff, 'counter_integer_variable' : counter_integer_variable_r2, 
                    'print_model' : print_stuff_on_model}
                print_stuff = ''
                print('Done with layerID ' + str(layerID)+' - node ID '+ str(nodeID))
                sys.stdout.flush()

                
            model_0.dispose()


            sys.stdout.flush()


            dict_bounds_gurobi_manager.__setitem__(str(layerID)+'_'+str(nodeID_start) + '_'+ 'process_start_time', process_start_time)
            dict_bounds_gurobi_manager.__setitem__(str(layerID)+'_'+str(nodeID_start) + '_'+ 'process_end_time', time.time() )
            
            # return dict_bounds_gurobi_manager

        if layerID >= 3:

            for nodeID in range(nodeID_start, last_node, 1):
                

                flag_node_diamond_init = 0
                
                print_stuff += '\n'
                print_stuff += 'Image ID: '+str(imgID)
                print_stuff += '\n'
                print_stuff += 'reluNum : '
                print_stuff += str(reluNum)
                print_stuff += '\n'
                print_stuff += 'layerID : '
                print_stuff += str(layerID)
                print_stuff += '\n'
                print_stuff += 'nodeID: '
                print_stuff += str(nodeID)
                print_stuff += '\n'
                print_stuff += '\n'
                
                
                
            
                open_relu_nodes = decide_binary_nodes_new(Loop_para_process, dict_bounds_gurobi, layerID, nodeID, Bootstrap_data_2, numNodesPerL, inputSize,Bootstrap_data_3)

       
                max_bounds = [99999]
                min_bounds = [-99999]

                if counter_integer_variable_r2 > 0:
                    
                    
                    if Loop_para_process['Open_r2'] == 0:
                        r2_fail_max = 1
                        r2_fail_min = 1
                        
                        r2_max_time = 0
                        r2_min_time = 0
                    else:
                        return_loop_r2 = gurobi_r2_loop(model_r2, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias)
                        [print_stuff, r2_max_time, r2_min_time, r2_fail_max, r2_fail_min, min_bound_r2, max_bound_r2, re_do_min_last_layer] = return_loop_r2

                        
                        print_stuff += 'max for k=2:  '
                        print_stuff += str(max_bound_r2)
                        
                        print_stuff += '\n'
                        
                        print_stuff += 'min for k=2:  '
                        print_stuff += str(min_bound_r2)
                        print_stuff += '\n'
                        print_stuff += '\n'
                        
                        
                        max_bounds.append(max_bound_r2)
                        min_bounds.append(min_bound_r2)
                        #################################################
                        
                    r1_max_time = 0
                    r1_min_time = 0
                    
                    
      
                    
                    if r2_fail_max == 1 or r2_fail_min == 1:
                        
                        print_stuff += 'Call Diamond model'
                        print_stuff += '\n'
                        
                        
                        
                        if num_binary_nodes >= 50 or Loop_para_process['Open_r2'] == 1 or 1 == 1:  #  please testing here, liao
                            model_0, counter_integer_variable_0_diamond, print_stuff = gurobi_initialize_model_node_diamond_new2(model_r2, numLayers_Cof, input, inputSize, numNodesPerL, epsilon, 
                                                                                              dict_bounds_gurobi, layerID, reluNum+1, weights, bias, mipf, stepBack, open_relu_nodes, print_stuff)
                        elif Loop_para_process['Open_r2'] == 0:
                            model_r2.reset(0)
                            model_0, counter_integer_variable_0_diamond, print_stuff = model_r2, counter_integer_variable_r2, print_stuff
                            
                            
                        start_time_max = time.time()
                        if r2_fail_max == 1:
                            
                            
                            
                            
                            return_loop = gurobi_loop_max(model_0, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias)
                            [print_stuff, flag_need_diamond_max] = return_loop
                            


                            print_stuff += 'max for Diamond model:'
                            print_stuff += str(model_0.ObjBoundC)
                            print_stuff += '\n'
                            
                            if flag_need_diamond_max == 0:
                                
                                max_bounds.append(model_0.ObjBoundC)
                            
                                print_stuff += 'MIPGap for max : '
                                print_stuff += str(model_0.MIPGap)
                                print_stuff += '\n'
                                print_stuff += '\n'
                            
                                
                            
                            if flag_need_diamond_max == 1:
                                print_stuff += '\n'
                                print_stuff += 'Initialized less node model for Max'
                                print_stuff += '\n'
                                
                                flag_node_diamond_init = 1
                                
                                open_relu_nodes_used = open_relu_nodes
                                if len(open_relu_nodes) > 50:
                                    open_relu_nodes_used = open_relu_nodes[0:40]
                                elif len(open_relu_nodes) > 10:
                                    open_relu_nodes_used = open_relu_nodes[0:len(open_relu_nodes)-10]
    
    
    
                                model_1, counter_integer_variable_1_diamond, print_stuff = gurobi_initialize_model_node_diamond_new2(model_r2, numLayers_Cof,input, inputSize, numNodesPerL, epsilon, 
                                                                                        dict_bounds_gurobi, layerID, reluNum+1, weights, bias, mipf, stepBack, open_relu_nodes_used, print_stuff)

                                model_1, time_max_n, max_nodeID_diamond = gurobi_master_per_node_max(layerID, nodeID, model_1, counter_integer_variable_1_diamond)
                                
                                
                                
                                print_stuff += 'max for less nodes Diamond:'
                                print_stuff += str(max_nodeID_diamond)
                                print_stuff += '\n'
                                
                                if model_0.solCount >0:
    
                                    if model_0.ObjBoundC <= max_nodeID_diamond:
                                        print_stuff += 'MIPGap for max : '
                                        print_stuff += str(model_0.MIPGap)
                                        print_stuff += '\n'
                                    else:
                                        print_stuff += 'MIPGap for max : '
                                        print_stuff += str(model_1.MIPGap)
                                        print_stuff += '\n'
                                else:
                                    print_stuff += 'MIPGap for max : '
                                    print_stuff += str(model_1.MIPGap)
                                    print_stuff += '\n'
                                
                                max_bounds.append(max_nodeID_diamond)
                                model_1.reset(0)   
                                print_stuff += '\n'
    
                            model_0.reset(0)
                            time_max_node = time.time() - start_time_max
                            
    
                        r1_max_time = time.time() - start_time_max
                        #################################################
    
                        #################################################
                        start_time_min = time.time()
                        if r2_fail_min == 1: # testing
                        
                            re_do_min_last_layer = False
                        
                            
                        
                            return_loop = gurobi_loop_min(model_0, print_stuff, Loop_para_process, layerID, nodeID, dict_bounds_gurobi, dict_bounds_gurobi_manager, inputSize, weights, bias)
                            [print_stuff, flag_need_diamond_min] = return_loop
                            
                            print_stuff += 'min for Diamond model:'
                            print_stuff += str(model_0.ObjBoundC)
                            print_stuff += '\n'
    
                            if flag_need_diamond_min == 0:
                                
                                if model_0.ObjVal > 0:
                                    re_do_min_last_layer = True
                            
                                print_stuff += 'MIPGap for min : '
                                print_stuff += str(model_0.MIPGap)
                                print_stuff += '\n'
                                print_stuff += '\n'
                            
                                min_bounds.append(model_0.ObjBoundC)
                
                            if flag_need_diamond_min == 1:
                                print_stuff += '\n'
                                print_stuff += 'Considered less nodes for Min'
                                print_stuff += '\n'
        
                                if flag_node_diamond_init == 0:
        
                                    print_stuff += 'Initialized less node model for Min'
                                    print_stuff += '\n'
        
                                    flag_node_diamond_init = 1
                                    open_relu_nodes_used = open_relu_nodes
                                    
                                    if len(open_relu_nodes) > 50:
                                        open_relu_nodes_used = open_relu_nodes[0:40]
                                    elif len(open_relu_nodes) > 10:
                                        open_relu_nodes_used = open_relu_nodes[0:len(open_relu_nodes)-10]
                                    
        
                                    model_1, counter_integer_variable_1_diamond, print_stuff = gurobi_initialize_model_node_diamond_new2(model_r2, numLayers_Cof,input, inputSize, numNodesPerL, epsilon, 
                                                                                            dict_bounds_gurobi, layerID, reluNum+1, weights, bias, mipf, stepBack, open_relu_nodes_used, print_stuff)
        
    
                                model_1, time_min_n, min_nodeID_diamond = gurobi_master_per_node_min(layerID, nodeID, model_1, counter_integer_variable_1_diamond)
                                print_stuff += 'min for less nodes Diamond:'
                                print_stuff += str(min_nodeID_diamond)
                                print_stuff += '\n'
                                
                                if model_0.solCount > 0:
                                    
                                    if model_0.ObjVal > 0:
                                        re_do_min_last_layer = True
        
                                    if model_0.ObjBoundC >= min_nodeID_diamond:
                                        print_stuff += 'MIPGap for min : '
                                        print_stuff += str(model_0.MIPGap)
                                        print_stuff += '\n'
                                    else:
                                        print_stuff += 'MIPGap for min : '
                                        print_stuff += str(model_1.MIPGap)
                                        print_stuff += '\n'
                                else:
                                    re_do_min_last_layer = True
                                    print_stuff += 'MIPGap for min : '
                                    print_stuff += str(model_1.MIPGap)
                                    print_stuff += '\n'
                                min_bounds.append(min_nodeID_diamond)
                                
                                print_stuff += '\n'
        
                                model_1.reset(0)
        
                            model_0.reset(0)
                            time_min_node = time.time() - start_time_min
                            
                            
                            if model_0 != model_r2:
                                model_0.dispose()
                            
                        r1_min_time = time.time() - start_time_min
                    
                            
                    max_nodeID = min(max_bounds)
                    min_nodeID = max(min_bounds)
                    time_max_node = r2_max_time+r1_max_time
                    time_min_node = r2_min_time+r1_min_time
                    
                    
                    
                    
                    
                    
                    if layerID == numLayers_Cof*2+1 and min_nodeID < 0 and re_do_min_last_layer == True:
                        
                        print_stuff += 'enter redo\n'
                        
                        try_times_redo = 0
                        
                        while try_times_redo <= 3 and re_do_min_last_layer == True :
                            model_r2.setParam('MIPGap', 0)
                            model_r2.Params.TimeLimit = 5
                            model_r2.optimize()
                            try_times_redo += 1
                            print_stuff += 'redo bound = '
                            print_stuff += str(model_r2.ObjBoundC)
                            print_stuff += '\n'
                            print_stuff += 'redo Gap = '
                            print_stuff += str(model_r2.MIPGap)
                            print_stuff += '\n'
                            
                            if model_r2.SolCount > 0:
                                if model_r2.ObjVal < 0:
                                    re_do_min_last_layer = False
                                    
                            if model_r2.ObjBoundC > 0:
                                re_do_min_last_layer = False
                                
                        if model_r2.SolCount > 0:
                            
                            if model_r2.ObjVal > 0:
                                
                                try_times_redo = 0
                                
                                while try_times_redo <= 19 and re_do_min_last_layer == True :
                                    model_r2.setParam('MIPGap', 0)
                                    model_r2.Params.TimeLimit = 5
                                    model_r2.optimize()
                                    try_times_redo += 1
                                    print_stuff += 'redo bound = '
                                    print_stuff += str(model_r2.ObjBoundC)
                                    print_stuff += '\n'
                                    print_stuff += 'redo Gap = '
                                    print_stuff += str(model_r2.MIPGap)
                                    print_stuff += '\n'
                                    
                                    if model_r2.ObjVal < 0:
                                            re_do_min_last_layer = False
                                            
                                    if model_r2.ObjBoundC > 0:
                                        re_do_min_last_layer = False
                                
                        # print_stuff += 'redo bound = '
                        # print_stuff += str(model_r2.ObjBoundC)
                        # print_stuff += '\n'
                        # print_stuff += 'redo Gap = '
                        # print_stuff += str(model_r2.MIPGap)
                        # print_stuff += '\n\n'
                        
                        print_stuff += '\n'
                        
                        min_nodeID = max(model_r2.ObjBoundC,min_nodeID)
                    
                else :
                    start_time_max = time.time()
                    model_r2 = gurobi_max(model_r2, layerID, nodeID)     # setting objective
                    model_r2.optimize()
                    max_nodeID = model_r2.getObjective().getValue()
                    model_r2.reset(0)
                    time_max_node = time.time() - start_time_max

                    start_time_min = time.time()
                    model_r2 = gurobi_min(model_r2, layerID, nodeID)     # setting objective
                    model_r2.optimize()
                    min_nodeID = model_r2.getObjective().getValue()
                    model_r2.reset(0)
                    time_min_node = time.time() - start_time_min
                
                model_r2.reset(0)   
                model_r_2, counter_integer_variable, print_stuff = gurobi_initialize_model_node_recover_layer_model(model_r2, numLayers_Cof, input, inputSize, numNodesPerL, epsilon, 
                                                                                          dict_bounds_gurobi, layerID, reluNum+1, weights, bias, mipf, stepBack, open_relu_nodes, print_stuff)
                
                   
                    

                        
                

                print_stuff += 'time for max total = '
                print_stuff += str(time_max_node)
                print_stuff += '\n'
                print_stuff += 'time for min total = '
                print_stuff += str(time_min_node)
                print_stuff += '\n'
                print_stuff += 'max for node = '
                print_stuff += str(max_nodeID)
                print_stuff += '\n'
                print_stuff += 'min for node = '
                print_stuff += str(min_nodeID)
                print_stuff += '\n'
                print_stuff += str('DONEEEEEEE')
                print_stuff += '\n'
                print_stuff += '\n'
                print_stuff += '\n'


                dict_bounds_gurobi_manager[str(layerID)+'_'+str(nodeID)] = {'LB' : min_nodeID, 'UB' : max_nodeID, 'time_max_total' : time_max_node,
                    'time_min_total' : time_min_node, 'print' : print_stuff, 'counter_integer_variable' : counter_integer_variable_r2, 
                    'print_model' : print_stuff_on_model}
                print_stuff = ''
                print('Done with layerID ' + str(layerID)+' - node ID '+ str(nodeID))
                sys.stdout.flush()


                

            
            model_r2.dispose()
  
            sys.stdout.flush()

            dict_bounds_gurobi_manager.__setitem__(str(layerID)+'_'+str(nodeID_start) + '_'+ 'process_start_time', process_start_time)
            dict_bounds_gurobi_manager.__setitem__(str(layerID)+'_'+str(nodeID_start) + '_'+ 'process_end_time', time.time() )






def gurobi_master(id, Loop_para, input, inputSize, numNodesPerL, epsilon, dict_bounds, reluNum, weight, bia, numLayers, layerID, stepBack, bootstrap_data, 
                  num_nodes_per_process):
    '''
    This function initializes and optimized a gurobi model for one entire layer

    layerID : Must be affine layer. Layer ID, in DeepPoly terms that indicates node's layer. 

    reluNum : number of relu layers behind which layerID that should be denoted by integer variables. If 0 then model is LP.

    numNodesPerL : expected to be a list. Does not include input size. Starts from first layer

    numLayers : Does not include input layer and last weird layer

    bia : bias

    weight : weights
    '''



    if layerID == 1:
        num_nodes_per_process = 100


    if layerID >= numLayers_Cof*2+1 : 
        num_nodes_per_process = Last2layer_NodeNum_perProcess
        

    
    mipf = 0

    dict_bounds_gurobi = copy.deepcopy(dict_bounds)

    weight_layer_index = int((layerID-1)/2)

    weights = copy.deepcopy(weight)
    bias = copy.deepcopy(bia)

    numNodesPerL = list(np.repeat(numNodesPerL,2))
    # Adding number of nodes in last weird layer  
    numNodesPerL.append(numNodesPerL[-1] -1)


    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()
    dict_bounds_gurobi_manager = manager.dict()
    Loop_para_process = manager.dict()
    
    Loop_para_process = copy.deepcopy(Loop_para)
    
    if layerID > 3 and layerID <= numLayers_Cof*2+1:
        bootstrap_data_cut = {}

        for nodeID_start in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
            bootstrap_data_cut[nodeID_start] = {}
            bootstrap_data_cut_temp_2 = {}
            bootstrap_data_cut_temp_3 = {}
            
            for nodeID_1 in range(nodeID_start,nodeID_start+num_nodes_per_process):
                bootstrap_data_cut_temp_2[str(layerID)+'_'+str(nodeID_1)] = bootstrap_data[0][str(layerID)+'_'+str(nodeID_1)]
                bootstrap_data_cut_temp_3[str(layerID)+'_'+str(nodeID_1)] = bootstrap_data[1][str(layerID)+'_'+str(nodeID_1)]
            bootstrap_data_cut[nodeID_start] = [bootstrap_data_cut_temp_2,bootstrap_data_cut_temp_3]
            
    elif layerID == 3:
        
        bootstrap_data_cut = {}

        for nodeID_start in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
            bootstrap_data_cut[nodeID_start] = {}
            bootstrap_data_cut_temp_2 = {}
            bootstrap_data_cut_temp_3 = {}
            
            for nodeID_1 in range(nodeID_start,nodeID_start+num_nodes_per_process):
                bootstrap_data_cut_temp_2[str(layerID)+'_'+str(nodeID_1)] = bootstrap_data[0][str(layerID)+'_'+str(nodeID_1)]
                bootstrap_data_cut_temp_3[str(layerID)+'_'+str(nodeID_1)] = {}
            bootstrap_data_cut[nodeID_start] = [bootstrap_data_cut_temp_2,bootstrap_data_cut_temp_3]
    
    else:
        bootstrap_data_cut = {}
        for nodeID_start in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
            bootstrap_data_cut[nodeID_start] = [{},{}]

    if layerID >= 3:
        
        num_binary_nodes = 0
        for j in range(numNodesPerL[layerID-2-1]):
            if dict_bounds_gurobi[str(layerID-2)+'_'+str(j)]['UB'] > 0 and dict_bounds_gurobi[str(layerID-2)+'_'+str(j)]['LB'] < 0:
                num_binary_nodes +=1
    
        dict_bounds_gurobi['layer_binary'+str(layerID-2)] = num_binary_nodes    
    

    pool.starmap(gurobi_master_single_node, [(id, Loop_para_process, input, layerID, nodeID_start, num_nodes_per_process, reluNum, stepBack, dict_bounds_gurobi, dict_bounds_gurobi_manager, 
        numNodesPerL, bootstrap_data_cut[nodeID_start], inputSize, weights, bias, mipf, epsilon) for nodeID_start in range(0,numNodesPerL[layerID-1],num_nodes_per_process)])
    sys.stdout.flush()
    print()
    print('*******************************')
    print()

    print('For LayerID : ', layerID)
    print()
    
    print('Time details for processes in layer : ', layerID)
    print()
    print('Processes listed with difference from start time of first process : ')
    print('first nodeID in process : difference in time')
    print()
    
    

    
    
    
    dic_start_times = {}

    for i in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
        vartime = dict_bounds_gurobi_manager[str(layerID)+'_'+str(i) +'_'+ 'process_start_time']

        dic_start_times[i] = vartime

    dic_start_times = {k: v for k, v in sorted(dic_start_times.items(), key=lambda item: item[1])}

    num = min(dic_start_times.values())

    for key in dic_start_times.keys():
        print(str(key)+  ' : ' + str(dic_start_times[key]-num))
    
    print()

    sys.stdout.flush()
    print('Processes listed with difference from end time of the first process that ended : ')
    print('first nodeID in process : difference in time')
    print()

    dic_start_times = {}

    for i in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
        vartime = dict_bounds_gurobi_manager[str(layerID)+'_'+str(i) +'_'+ 'process_end_time']

        dic_start_times[i] = vartime

    dic_start_times = {k: v for k, v in sorted(dic_start_times.items(), key=lambda item: item[1])}

    num = min(dic_start_times.values())

    for key in dic_start_times.keys():
        print(str(key)+  ' : ' + str(dic_start_times[key]-num))

    print()

    print()
    sys.stdout.flush()

    print('Processes run time : ')
    print('first nodeID in process : time taken to run')
    print()

    dic_times = {}

    for i in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
        vartime = dict_bounds_gurobi_manager[str(layerID)+'_'+str(i) +'_'+ 'process_end_time'] - dict_bounds_gurobi_manager[str(layerID)+'_'+str(i) +'_'+ 'process_start_time']

        dic_times[i] = vartime


    for key in dic_times.keys():
        print(str(key)+  ' : ' + str(dic_times[key]))

    print()
    
    sys.stdout.flush()
    print()
    print('*******************************')
    print()

    for i in range(0,numNodesPerL[layerID-1],num_nodes_per_process):
        print(dict_bounds_gurobi_manager[str(layerID)+'_'+str(i)]['print_model'])
        sys.stdout.flush()

    sys.stdout.flush()
    print()
    print()

    for nodeID in range(numNodesPerL[layerID-1]):

        dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)] = dict_bounds_gurobi_manager[str(layerID)+'_'+str(nodeID)]

        print()

        print(dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['print'])
        # f.write(dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['print'])
        sys.stdout.flush()
        
        


    sys.stdout.flush()

    counter_integer_variable_1 = dict_bounds_gurobi_manager[str(layerID)+'_'+str(0)]['counter_integer_variable']

    cert = -1

    if layerID == (numLayers*2 +1) :  # If layerID is the last weird layer

        # NOTE : After calling gurobi master need to call calc_relu_bounds() for the next layer 

        numNodes_lastLayer = weights[weight_layer_index].shape[0] 

        cert, lb_li = certify_model(layerID, numNodes_lastLayer, dict_bounds_gurobi)

    print()
    print('****************************')
    print('DONE with IMAGE : ', id)
    print('****************************')
    print()
    
    
    if position == 'LapTop':
        f = open(full_output_name,'at')
        for nodeID in range(numNodesPerL[layerID-1]):
            f.write(dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['print'])
        f.write('****************************')
        f.write('\n')
        f.write('DONE with IMAGE : ')
        f.write(str(id))
        f.write('\n')
        f.write('****************************')
        f.write('\n')

        f.close()


    return cert, counter_integer_variable_1, dict_bounds_gurobi
    




# MAIN
def main():
    '''
    Assumes all layers have Relu activation
    '''
    print(datetime.datetime.now())
    
    start_time_global =  time.time()

    # REAL INPUT 
    


    # model_path = "../pretrained_model/mnist_relu_6_100.onnx"
    # data_path = '../data/test.csv'
    
    # Preliminaries
    numLayers = numLayers_Cof       # not counting input layer and last weird layer
    inputSize = 784
    data_dict = extract_mnist_data(data_path)
    weights, bias, numNodesPerL = extract_onnx_weights_biases_numNodes(model_path,network_name)


    

    id_list = range(250)

    input_epsilon = 0.026
    
    key = "generate"
    
    k = 1
    
    
    Fist_timeout_para = 2.5
    
    First_Gap_para = 0.001*k
    
    Timeout_step_para = 2.5
    
    Second_Gap_para = 0.01*k
    
    Third_Gap_para = 0.1*k
    
    Call_Diamond_para = 0.2
    
    Strict_para = 0.25
    
    Open_r2 = 1
    
    timeout_setting = 1000
    # a = 5
    
    # if timeout_setting == 2000:
    #     a = 5
    # elif timeout_setting == 900:
    #     a = 2
    
    
    Fist_timeout_para_r2 = 5
    
    First_Gap_para_r2 = 0.001*k
    
    Timeout_step_para_r2 = 5
    
    Second_Gap_para_r2 = 0.01*k
    
    Third_Gap_para_r2 = 0.1*k
    
    
    
    Loop_para = {'Open_r2':  Open_r2, 'Fist_timeout_para_r2':Fist_timeout_para_r2, 'First_Gap_para_r2':First_Gap_para_r2,'Timeout_step_para_r2':Timeout_step_para_r2,
                 'Second_Gap_para_r2':Second_Gap_para_r2, 'Third_Gap_para_r2':Third_Gap_para_r2,'Fist_timeout_para':Fist_timeout_para, 'First_Gap_para':First_Gap_para,'Timeout_step_para':Timeout_step_para,
                 'Second_Gap_para':Second_Gap_para, 'Third_Gap_para':Third_Gap_para, 'Call_Diamond_para':Call_Diamond_para,'Strict_para':Strict_para,'open_limit':50}
    
    
    running_information = {}
    
    
    ReferenceNew = {}
    ReferenceNew[1] = [{},{}]
    ReferenceNew[numLayers_Cof*2+1] = [{},{}]

    
    if key == "generate":
        pool = multiprocessing.Pool(20)

        for layerID in range(3,numLayers_Cof*2+1,2):
        
            if os.path.isfile(ReferenceData_name+str(layerID)+'.b') == True:
                file2 = open(ReferenceData_name+str(layerID)+'.b', 'rb')
                Bootstrap_datas = pickle.load(file2)
                ReferenceNew[layerID] = Bootstrap_datas
                
                
                file2.close()                    
            else:
                Bootstrap_data = {}
                
                source_layer_ID = max(0, layerID - 4)
                source_layer_ID_w = int((source_layer_ID-1)/2)
                
                if source_layer_ID == 0:
                    node_num = inputSize
                else:
                    node_num = numNodesPerL[source_layer_ID_w]
                
                
        
                for nodeID in range(numNodesPerL[source_layer_ID_w+2]):

                    
                    Bootstrap_data[str(layerID)+'_'+str(nodeID)] = {}
                    
                    Bootstrap_data[str(layerID)+'_'+str(nodeID)] = pool.starmap(bootstrap_decide_binary_nodes_new, [(weights, numNodesPerL,layerID,nodeID,source_layer_ID, source_node_ID) for source_node_ID in range(0,node_num)])
                
                Bootstrap_data2 = {}
                
                if layerID >= 5:
                    
                    source_layer_ID = max(0, layerID - 6)
                    source_layer_ID_w = int((source_layer_ID-1)/2)
                    
                    if source_layer_ID == 0:
                        node_num = inputSize
                    else:
                        node_num = numNodesPerL[source_layer_ID_w]
                    
                    
            
                    for nodeID in range(numNodesPerL[source_layer_ID_w+3]):
                        print(nodeID,datetime.datetime.now())
                        
                        Bootstrap_data2[str(layerID)+'_'+str(nodeID)] = {}
                        
                        Bootstrap_data2[str(layerID)+'_'+str(nodeID)] = pool.starmap(bootstrap_decide_binary_nodes_new, [(weights, numNodesPerL,layerID,nodeID,source_layer_ID, source_node_ID) for source_node_ID in range(0,node_num)])
                
                ReferenceNew[layerID] = [Bootstrap_data,Bootstrap_data2]
                file3 = open(ReferenceData_name+str(layerID)+'.b', 'wb')
                pickle.dump(ReferenceNew[layerID], file3)
                file3.close() 
        pool.close()          


    
 


    print_li = []
    
    for turns in [0]:
        
        
        Loop_para['Open_r2'] = min(turns,1)
        
        next_id_list = []
        
        number_verified = 0
        
        for id in id_list:
            
            
        
            
            running_information[id] = {}
            
            running_information[id]['DeepPoly_cert'] = -1
            running_information[id]['DeepPoly_accuracy'] = -1
            running_information[id]['DeepPoly_time'] = -1
            running_information[id]['MILP_cert'] = -1
            running_information[id]['MILP_accuracy'] = -1
            running_information[id]['MILP_time'] = -1
            running_information[id]['Min_of_Min'] = []
    
            start_time_total = time.time()
    
            cor_label = data_dict[id]['label']
            image = data_dict[id]['image']    
    
            # For multiprocessing - number of nodes processed sequentially in one process 
            # num_nodes_per_process = 4
            num_nodes_per_process = Basic_NodeNum_perProcess
    
            # Normalization
            image = image/np.float32(255).copy()
    
            input_values = image.reshape(784).copy()
            
            # +- error on each input dimension
            
    
            # Pred and Correct label
            weights, bias, numNodesPerL = extract_onnx_weights_biases_numNodes(model_path,network_name)
            pred_label, pred_arr = make_onnx_prediction(model_path, image)
            numNodesPerL[-1] = numNodesPerL[-1] - 1
            
    
    
            if cor_label == pred_label:
                
                last_weight, last_bias = make_last_weights(cor_label, weights)
                
                real_last_weight = np.matmul(last_weight,weights[-1])
                real_last_bias = np.matmul(last_weight,bias[-1])

                weights.pop()
                bias.pop()
                weights.append(real_last_weight)
                bias.append(real_last_bias)
                
                
                
                pool = multiprocessing.Pool(20)
                for layerID in [numLayers_Cof*2+1]:
                
                    if os.path.isfile(ReferenceData_name+str(layerID)+'_lable'+str(cor_label)+'.b') == True:
                        file2 = open(ReferenceData_name+str(layerID)+'_lable'+str(cor_label)+'.b', 'rb')
                        Bootstrap_datas = pickle.load(file2)
                        ReferenceNew[layerID] = Bootstrap_datas
                        file2.close()                    
                    else:
                        Bootstrap_data = {}
                        
                        source_layer_ID = max(0, layerID - 4)
                        source_layer_ID_w = int((source_layer_ID-1)/2)
                        
                        if source_layer_ID == 0:
                            node_num = inputSize
                        else:
                            node_num = numNodesPerL[source_layer_ID_w]
                        
                        
                
                        for nodeID in range(9):

                            
                            Bootstrap_data[str(layerID)+'_'+str(nodeID)] = {}
                            
                            Bootstrap_data[str(layerID)+'_'+str(nodeID)] = pool.starmap(bootstrap_decide_binary_nodes_new, [(weights, numNodesPerL,layerID,nodeID,source_layer_ID, source_node_ID) for source_node_ID in range(0,node_num)])
                        
                        Bootstrap_data2 = {}
                        
                        if layerID >= 5:
                            
                            source_layer_ID = max(0, layerID - 6)
                            source_layer_ID_w = int((source_layer_ID-1)/2)
                            
                            
                    
                            for nodeID in range(9):
                                print(nodeID,datetime.datetime.now())
                                
                                Bootstrap_data2[str(layerID)+'_'+str(nodeID)] = {}
                                
                                Bootstrap_data2[str(layerID)+'_'+str(nodeID)] = pool.starmap(bootstrap_decide_binary_nodes_new, [(weights, numNodesPerL,layerID,nodeID,source_layer_ID, source_node_ID) for source_node_ID in range(0,node_num)])
                        
                        ReferenceNew[layerID] = [Bootstrap_data,Bootstrap_data2]
                        file3 = open(ReferenceData_name+str(layerID)+'_lable'+str(cor_label)+'.b', 'wb')
                        pickle.dump(ReferenceNew[layerID], file3)
                        file3.close() 
                pool.close()
                

                cert, dict_bounds_DP, time_DP = DeepPoly(cor_label, input_values, inputSize, input_epsilon, numLayers, numNodesPerL, weights, bias)
                cert_DP = cert[0]
    
                lastLayerID = numLayers * 2 + 1
                total_uncertainty_DP = 0
    
                weight_layer_index = int((lastLayerID-1)/2)
                
                for nodeID in range(weights[weight_layer_index].shape[0]):
                            total_uncertainty_DP += dict_bounds_DP[str(lastLayerID)+'_'+str(nodeID)]['UB'] - dict_bounds_DP[str(lastLayerID)+'_'+str(nodeID)]['LB']
                            
    
                            
                avg_uncertainty = (total_uncertainty_DP/(weights[weight_layer_index].shape[0]))
                
                DeepPoly_end = time.time()
                
                
                running_information[id]['DeepPoly_cert'] = cert_DP
                running_information[id]['DeepPoly_accuracy'] = avg_uncertainty
                running_information[id]['DeepPoly_time'] = DeepPoly_end-start_time_total
    
                
    
                
        
    
                if cert_DP == 1: # testing, please recover after use, liao
                    line_str = str(id)+' : '+ str(cert_DP)+', '+str(avg_uncertainty)+', '+str(time.time()-start_time_total)+', DeepPoly'
                    print_li.append(line_str) 
                    print(line_str)
                    number_verified += 1
    
                    continue
                
                elif avg_uncertainty > DeepPoly_bound: 
    
                    line_str = str(id)+' : '+ str(cert_DP)+', '+str(avg_uncertainty)+', '+str(time.time()-start_time_total)+', DeepPoly'
                    print_li.append(line_str)  
                    print(line_str)
    
                    continue

                sys.stdout.flush()
    
                k_param = [1]
                stepBack = [20]
    

                for reluNum in k_param:
                    for sb in stepBack:
    
                        # start_time_total = time.time()
                        
                        dict_bounds_gurobi_tbc = calc_inputL_bounds(input_values, inputSize, input_epsilon) # this line does not use DeepPoly bound
    
                     #   dict_bounds_gurobi_tbc = dict_bounds_DP
                        dict_bounds_gurobi =  copy.deepcopy(dict_bounds_gurobi_tbc)
    
                        integer_nodes_opened = []
                        
                        end_layer = numLayers*2+1

    
                        for layerID in range(1, end_layer+1, 2):
                            
                            # if layerID == (end_layer) and test_key == 'save':
                            #     with open('Save_bounds'+str(id)+'.b','wb') as file:
                            #         pickle.dump(dict_bounds_gurobi, file)
                            
                            # if test_key == 'read':
                            #     if layerID <= 16:
                            #         continue
                            #     if layerID == 17:
                            #         with open('Save_bounds'+str(id)+'.b','rb') as file:
                            #             dict_bounds_gurobi = pickle.load(file)
                            
                            
                            
                            if time.time()-DeepPoly_end > 0.4*timeout_setting+0.1*timeout_setting*(max(0,layerID/2-2.5)) and layerID < numLayers*2 + 1:
                                line_str = str(id)+' : '+ str(cert_DP)+', '+str(avg_uncertainty)+', '+str(time.time()-DeepPoly_end)+', Diamond timeout'
                                print_li.append(line_str)
                                print(line_str)
                                running_information[id]['MILP_time'] = time.time()-DeepPoly_end
                                running_information[id]['MILP_cert'] = 0
                                
                                break
    
                            sys.stdout.flush()
    
                            weight_layer_index = int((layerID-1)/2)
                            cert, counter_integer_variable_1, dict_bounds_gurobi = gurobi_master(id,Loop_para,input_values, inputSize, numNodesPerL,
                                            input_epsilon, dict_bounds_gurobi, reluNum, weights, bias, numLayers, layerID, sb, ReferenceNew[layerID], num_nodes_per_process)
                            
                            integer_nodes_opened.append(counter_integer_variable_1)
                    
                            if layerID == (end_layer):
    
                                # cert_MILP[str(reluNum)+'_'+str(sb)] = cert
                                total_uncertainty = 0
                                
                                min_of_min_text = 0
    
    
                                for nodeID in range(weights[weight_layer_index].shape[0]):
                                    running_information[id]['Min_of_Min'].append(dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['LB'])
                                    total_uncertainty += dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['UB'] - dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['LB']
                                    min_of_min_text = min(min_of_min_text,dict_bounds_gurobi[str(layerID)+'_'+str(nodeID)]['LB'])

                                
                                line_str = str(id)+' : '+ str(cert)+', '+str((total_uncertainty/(weights[weight_layer_index].shape[0])))+', '+str(time.time()-start_time_total)+', Diamond'+'Min of Min: '+str(min_of_min_text)
                                print_li.append(line_str)
    
                                print(line_str)
                                
                                if cert == 1:
                                    number_verified += 1
                                
                                sys.stdout.flush()
                                
                                if cert == 0 and min_of_min_text>-2.5:
                                    next_id_list.append(id)
                                
                                
                                running_information[id]['MILP_cert'] = cert
                                running_information[id]['MILP_accuracy'] = (total_uncertainty/(weights[weight_layer_index].shape[0]))
                                running_information[id]['MILP_time'] = time.time()-DeepPoly_end
            else:
                line_str = str(id)+' : '+ str(-1)+', '+'False predicate'
                print_li.append(line_str)
        # id_list = next_id_list

    
    
    file_3 = open(Record_name+'_image'+str(id_list[0])+'to'+str(id_list[-1])+'_'+str(current_time)+'.b','wb')    
    pickle.dump(running_information, file_3)    
    file_3.close()


    print(datetime.datetime.now())

    print()
    print()
    print('Epsilon : ', input_epsilon)
    print('ID : cert_DP, avg_uncertainty, time (including bootstrapping)')
    print('-----------------------------------------------')
    

    
    print()

    for line in print_li:
        print(line)
        # f1.write(line)

    print()
    print()
    
    if position == 'LapTop':
        f1 = open(full_output_name,'at')
        f1.write('\n')
        f1.write(str(datetime.datetime.now()))
        f1.write('\n')
        f1.write('ID : cert_DP, avg_uncertainty, time (including bootstrapping)')
        f1.write('\n')
        f1.write('-----------------------------------------------')
        f1.write('\n')
        
        for line in print_li:
            f1.write(line)
            f1.write('\n')
        
        f1.close()

    
    total_number_images = len(id_list)
    total_time = time.time() - start_time_global
    
    rate_percentage = "{:.1%}".format(number_verified/total_number_images)
    
    last_line = 'Total time: '+str(total_time)+', Average time: '+str(total_time/total_number_images)+', '
    last_line += 'Total verified number: '+str(number_verified)+', Rate: '+rate_percentage+'.'
    
    print('\n')
    print(last_line)




# Prepare work before running the main code. 


model_path = "../pretrained_model/mnist_relu_6_100.onnx"
data_path = '../data/test.csv'


threads_num = os.cpu_count()

if threads_num <= 20:
    position = 'LapTop'
    
elif threads_num >= 30:
    position = 'Server'


position = 'Server'




if position == 'LapTop':
    
    Basic_NodeNum_perProcess = 20
    Last2layer_NodeNum_perProcess = 2
    
    record_at_most_number = 10
    
elif position == 'Server':
  
    Basic_NodeNum_perProcess = 5
    Last2layer_NodeNum_perProcess = 1
    
    record_at_most_number = 30






net_work_file = model_path

if net_work_file == "../pretrained_model/mnist_relu_9_100.onnx":
    
    network_name = 'mnist_relu_9_100'
    numLayers_Cof = 8
    DeepPoly_bound = 1000
    
    
    ReferenceData_name = 'Bootstrap_9_100_'
    Record_name = 'Diamond_record'
    base_name = 'D_9_'
    
 
if net_work_file == "../pretrained_model/mnist_relu_6_100.onnx":
    
    network_name = 'mnist_relu_6_100'
    numLayers_Cof = 5
    DeepPoly_bound = 100

    
    ReferenceData_name = 'Bootstrap_6_100_'
    Record_name = 'Diamond_record'
    base_name = 'D_6_'

if net_work_file == "../pretrained_model/mnist_relu_6_200.onnx":
    
    network_name = 'mnist_relu_6_200'
    numLayers_Cof = 5
    DeepPoly_bound = 100
    
    Basic_NodeNum_perProcess = 10

    
    ReferenceData_name = 'Bootstrap_data/Bootstrap_6_200_'
    Record_name = 'Diamond_record'
    base_name = 'D_6_200'
    



now = datetime.datetime.now()

current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

full_output_name = base_name+current_time+'.txt'


# NOTE : CHECK - float 32 or 64, does it matter?

# time.sleep(120)

weights, bias, numNodesPerL = extract_onnx_weights_biases_numNodes(model_path,network_name)
  
# last_weight, last_bias = make_last_weights(0, weights)





# print(real_last_bias)

# print(real_last_weight.shape)

# print(bias[-1])

# print(weights[-1].shape)


if __name__ == '__main__':
    main()






