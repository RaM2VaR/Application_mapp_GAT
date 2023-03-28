#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import torch
from torch_scatter import scatter
from torch import nn
import random
import subprocess as sp
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import sys

#%%
@torch.no_grad()
def communication_cost(edge_index, edge_attr, batch, distance_matrix, predicted_mappings):
    batch_size = predicted_mappings.size(0) 
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    costs = distance_matrix[reverse_mappings_flattened[edge_index[0]], reverse_mappings_flattened[edge_index[1]]]\
        .unsqueeze(-1)
    # print(f"costs.device: {costs.device}, edge_attr.device: {edge_attr.device}")
    comm_cost = costs * edge_attr
    comm_cost.squeeze_(-1)
    comm_cost_each = scatter(comm_cost, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_cost_each

#%%
@torch.no_grad()
def communication_energy(edge_index, edge_attr, batch, distance_matrix, predicted_mappings):
    batch_size = predicted_mappings.size(0) 
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    costs = distance_matrix[reverse_mappings_flattened[edge_index[0]], reverse_mappings_flattened[edge_index[1]]]\
        .unsqueeze(-1)
    # print(f"costs.device: {costs.device}, edge_attr.device: {edge_attr.device}")
    comm_energy = ((costs *  6.675)+0.56)* edge_attr
    comm_energy.squeeze_(-1)
    comm_energy_each = scatter(comm_energy, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_energy_each

@torch.no_grad()
def communication_cost_multiple_samples(edge_index:torch.Tensor, edge_attr, batch, distance_matrix, predicted_mappings, num_samples, calculate_baseline = False):
    """
    Samples should be present in the graph wise fashion. This means that the sample corresponding to the first graph is present at the indices 0, 0 + batch_size, 0 + 2 * batch_size etc. The sample corresponding to the second graph is present at indices 1, 1 + batch_size, 1 + 2 * batch_size etc.
    """
    graph_size = predicted_mappings.size(1)
    batch_size = predicted_mappings.size(0) // num_samples
    device = edge_index.device
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat(1, num_samples)
    edge_index_adjust = \
        torch.arange(num_samples, device = device).repeat_interleave(edge_index.size(1)) * graph_size * batch_size
    edge_index_adjusted = edge_index_repeated + edge_index_adjust
    edge_attr_repeated = edge_attr.repeat(num_samples, 1)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_adjusted[0]], reverse_mappings_flattened[edge_index_adjusted[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    batch_repeated = batch.repeat(num_samples)
    batch_adjust = torch.arange(num_samples, device = device).repeat_interleave(batch.size(0)) * batch_size
    batch_adjusted = batch_repeated + batch_adjust
    comm_cost_each = \
        scatter(comm_cost, batch_adjusted[edge_index_adjusted[0]], dim=0, dim_size = num_samples * batch_size, reduce='sum').squeeze(-1)
    if calculate_baseline:
        indices = torch.arange(batch_size, device = device).repeat(num_samples)
        baseline = scatter(comm_cost_each, indices, dim=0, dim_size = batch_size, reduce='sum') / num_samples
        return comm_cost_each, baseline
    else:
        return comm_cost_each

@torch.no_grad()
def communication_energy_multiple_samples(edge_index:torch.Tensor, edge_attr, batch, distance_matrix, predicted_mappings, num_samples, calculate_baseline = False):
    """
    Samples should be present in the graph wise fashion. This means that the sample corresponding to 
    the first graph is present at the indices 0, 0 + batch_size, 0 + 2 * batch_size etc. 
    The sample corresponding to the second graph is present at indices 1, 1 + batch_size, 
    1 + 2 * batch_size etc.
    """
    graph_size = predicted_mappings.size(1)
    batch_size = predicted_mappings.size(0) // num_samples
    device = edge_index.device
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat(1, num_samples)
    edge_index_adjust = \
        torch.arange(num_samples, device = device).repeat_interleave(edge_index.size(1)) * graph_size * batch_size
    edge_index_adjusted = edge_index_repeated + edge_index_adjust
    edge_attr_repeated = edge_attr.repeat(num_samples, 1)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_adjusted[0]], reverse_mappings_flattened[edge_index_adjusted[1]]].unsqueeze(-1)
    comm_cost = ((costs *  6.675)+0.56) * edge_attr_repeated 
    batch_repeated = batch.repeat(num_samples)
    batch_adjust = torch.arange(num_samples, device = device).repeat_interleave(batch.size(0)) * batch_size
    batch_adjusted = batch_repeated + batch_adjust
    comm_cost_each = \
        scatter(comm_cost, batch_adjusted[edge_index_adjusted[0]], dim=0, dim_size = num_samples * batch_size, reduce='sum').squeeze(-1)
    if calculate_baseline:
        indices = torch.arange(batch_size, device = device).repeat(num_samples)
        baseline = scatter(comm_cost_each, indices, dim=0, dim_size = batch_size, reduce='sum') / num_samples
        return comm_cost_each, baseline
    else:
        return comm_cost_each

#%%

#%%
@torch.no_grad()
def get_reverse_mapping(predicted_mappings):
    device = predicted_mappings.device
    mask = predicted_mappings == -1
    reverse_mappings = torch.zeros_like(predicted_mappings)
    indices = torch.arange(predicted_mappings.size(1)).expand(predicted_mappings.size(0), -1)\
        .clone().to(device)
    # since -1 is not a valid index and predicted mappings are used as indices in scatter function, we need to replace -1 with the valid indices
    predicted_mappings[mask] = indices[mask]
    indices.masked_fill_(mask, -1)
    reverse_mappings.scatter_(1, predicted_mappings, indices)
    return reverse_mappings

#%%
@torch.no_grad()
def norm(x,stat):
    return (x - stat['mean']) / (stat['std']+0.0000001)
@torch.no_grad()
def mesh4x4_pred(mappings,mesh_size,inject_rate,traffic_type):
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/')
    model = load_model('mode2.h5') 
    train_stats = pd.read_pickle('mode2_stat.pkl')
    file2 = open("latency_model_2010_pso/files/vopd_mapp.txt","w")
    for i in range(mappings.shape[0]):
        file2.write('%d\t %s \t %f\n'%(i,str(mappings[i].to('cpu').numpy()+1),inject_rate))
    file2.close()
    # ************* begin modify file  files/config.txt for diff syn traffic pattern *************#
    with open('latency_model_2010_pso/files/config.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[3] = 'traffic file#\tfiles/'+traffic_type[23:-3]+'.txt'+'\n'
    # and write everything back
    with open('latency_model_2010_pso/files/config.txt', 'w') as file:
        file.writelines( data )
    #************** end modifying the file **************************************************#
    open('latency_model_2010_pso/files/vopd_mapp_features.txt','w').close()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/latency_model_2010_pso/')
    os.system('./latency_mode_2010_updated')
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/')
    with open('latency_model_2010_pso/files/vopd_mapp_features.txt','r') as my_file:
        text = my_file.read()
        text = text.replace("    ", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace(" ",",")
        text = text.replace(",,,",",")
        text = text.replace(",,",",")
        text = text.replace(",,",",")
        text = text.replace(",\n,\n,\n,",'\n')
        text = text.replace('\n,',',')
    my_file.close()
    open('latency_model_2010_pso/files/vopd_mapp_features.csv', 'w').close()
    with open('latency_model_2010_pso/files/vopd_mapp_features.csv','w') as analy_file:
        analy_file.write(text)
    X = pd.read_csv("latency_model_2010_pso/files/vopd_mapp_features.csv",header=None,
                          error_bad_lines=False)
    X.drop(X.columns[-1],inplace=True,axis=1)
    X.drop(X.columns[-1],inplace=True,axis=1)
    X.drop(X.columns[0],inplace=True,axis=1)
    X_norm = norm(X,train_stats)
    # X_norm.drop(X_norm.columns[0],inplace=True,axis=1)
    Y_pred = model.predict(X_norm).flatten()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/')
    return Y_pred

#%%
def mesh4x4_sim(mappings,mesh_size,inject_rate,traffic_type):
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/booksim')
        # ************* begin modify file  files/config.txt for diff syn traffic pattern *************#
    file1 = open("examples/vopd_mapp.txt","w")
    for i in range(mappings.shape[0]):
        file1.write('%d\t %s \t %f\n'%(i,str(mappings[i].to('cpu').numpy()+1),inject_rate))
    file1.close()
    
    with open('examples/anynet_config', 'r') as file:
        # read a list of lines into data
        trff = file.readlines()


    # now change the 2nd line, note that you have to add a newline
    trff[73] = 'traffic_file = examples/'+traffic_type[-19:-3]+'.txt'+';\n'

    # and write everything back
    with open('examples/anynet_config', 'w') as file:
        file.writelines(trff)
    #************** end modifying the file **************************************************#
    
    open('examples/vopd_mapp_out.txt', 'w').close()
    sp.call('./pgm_within_pgm > out_check.txt',shell=True)

    with open('examples/vopd_mapp_out.txt','r') as my_file:
        text = my_file.read()
        text = text.replace(" \t ", ",")
        text = text.replace("\t\t", ",")
        text = text.replace("\t", ",")
        text = text.replace("[", "")
        text = text.replace("]", " ")
        text = text.replace(" ","")
    my_file.close()
    open('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/booksim_out.csv', 'w').close()
    with open('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_4x4/booksim_out.csv','w') as myfile:
        myfile.write(text)
    my_file.close()
    book_lat = pd.read_csv('../booksim_out.csv',header=None, error_bad_lines=False)
    Y_sim = book_lat.iloc[:,-4].to_numpy()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/')
    return Y_sim

#%%
@torch.no_grad()
def mesh8x8_sim(mappings,mesh_size,inject_rate,traffic_type):
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/booksim')
        # ************* begin modify file  files/config.txt for diff syn traffic pattern *************#
    file1 = open("examples/vopd_mapp.txt","w")
    for i in range(mappings.shape[0]):
        file1.write('%d\t %s \t %f\n'%(i,str((mappings[i]+1).tolist()),inject_rate))
    file1.close()
    
    with open('examples/anynet_config', 'r') as file:
        # read a list of lines into data
        trff = file.readlines()


    # now change the 2nd line, note that you have to add a newline
    trff[73] = 'traffic_file = examples/'+traffic_type[-19:-3]+'.txt'+';\n'

    # and write everything back
    with open('examples/anynet_config', 'w') as file:
        file.writelines(trff)
    #************** end modifying the file **************************************************#
    
    open('examples/vopd_mapp_out.txt', 'w').close()
    sp.call('./pgm_within_pgm > out_check.txt',shell=True)

    with open('examples/vopd_mapp_out.txt','r') as my_file:
        text = my_file.read()
        text = text.replace(" \t ", ",")
        text = text.replace("\t\t", ",")
        text = text.replace("\t", ",")
        text = text.replace("[", "")
        text = text.replace("]", " ")
        text = text.replace(" ","")
    my_file.close()
    open('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/booksim_out.csv', 'w').close()
    with open('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/booksim_out.csv','w') as myfile:
        myfile.write(text)
    my_file.close()
    book_lat = pd.read_csv('../booksim_out.csv',header=None, error_bad_lines=False)
    Y_sim = book_lat.iloc[:,-1].to_numpy()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/')
    return Y_sim
#%%
@torch.no_grad()
def mesh8x8_pred(mappings,mesh_size,inject_rate,traffic_type):
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/')
    model = load_model('mesg8x8_12012021.h5') 
    train_stats = pd.read_pickle('mesh8x8_1201stat.pkl')
    file2 = open("latency_model_old/files/mapping_file.txt","w")
    for i in range(mappings.shape[0]):
#       mapp_vec = random.sample(range(1,NSIZE+1),NSIZE)
        file2.write('%d\t %s \t %f\n'%(i,str((mappings[i]+1).to('cpu').tolist()),inject_rate))
    file2.close()    
    # ************* begin modify file  files/config.txt for diff syn traffic pattern *************#
    with open('latency_model_old/files/config.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[3] = 'traffic file#\tfiles/'+traffic_type[23:-3]+'.txt'+'\n'
    # and write everything back
    with open('latency_model_old/files/config.txt', 'w') as file:
        file.writelines( data )
    #************** end modifying the file **************************************************#
    open('latency_model_old/files/mapping_file_features.txt','w').close()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/latency_model_old/')
    os.system('./latency_mode_2010_updated')
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/2d_noc_LPNet/mesh2D_8x8/')

    with open('latency_model_old/files/mapping_file_features.txt','r') as my_file:
        text = my_file.read()
        text = text.replace("    ", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace(" ",",")
        text = text.replace(",,,",",")
        text = text.replace(",,",",")
        text = text.replace(",,",",")
        text = text.replace(",\n,\n,\n,",'\n')
        text = text.replace('\n,',',')
    my_file.close()
    
    open('latency_model_old/files/mesh64_map_features.csv', 'w').close()
    with open('latency_model_old/files/mesh64_map_features.csv','w') as analy_file:
        analy_file.write(text)

    X = pd.read_csv("latency_model_old/files/mesh64_map_features.csv",header=None,
                          error_bad_lines=False)
   
    X.drop(X.columns[-1],inplace=True,axis=1)
#     X.drop(X.columns[-2:-66],inplace=True,axis=1)
#     X.drop(X.iloc[:,-66:-2],inplace=True,axis=1)
    X.drop(X.columns[0],inplace=True,axis=1)
    X_norm = norm(X,train_stats)
    Y_pred = model.predict(X_norm).flatten()
    os.chdir('/home/ram_lak/Ramesh_work/RL_work/MPNN-Ptr-master_apr26/')
    return Y_pred
#%%
@torch.no_grad()
def LPNet_pred(mappings,mesh_size,inject_rate,traffic_type):
    if mesh_size == 16:
        Y_pred = mesh4x4_pred(mappings,mesh_size,inject_rate,traffic_type)        
    elif mesh_size == 64:
        Y_pred = mesh8x8_pred(mappings,mesh_size,inject_rate,traffic_type)
    else:
        print(f'trained model for {mesh_size} not available')
        sys.exit("Error message")
    return Y_pred

@torch.no_grad()
def booksim_latency(mappings,mesh_size,inject_rate,traffic_type):
    if mesh_size == 16:
        Y_pred = mesh4x4_sim(mappings,mesh_size,inject_rate,traffic_type)        
    elif mesh_size == 64:
        Y_pred = mesh8x8_sim(mappings,mesh_size,inject_rate,traffic_type)
    else:
        print(f'trained model for {mesh_size} not available')
        sys.exit("Error message")
    return Y_pred
#%%
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def cr_trDic(traffic_file, max_val):
    # # prepare a dictionary of links with their BW,src and dst
    traff_dict = {}
    for idx, line in enumerate(traffic_file):
        for pos, word in enumerate(line.split()):
            if(word.isnumeric() and pos != idx):
                traff_dict['l%st%s' % (str(idx), str(pos))] = {
                    'val': float(word)/max_val, 'src': idx, 'dst': pos}

    return traff_dict


def mesh_design(dimX, dimY, dimZ):
    mesh_cords = []
    for z in range(dimZ):
        for y in range(dimY):
            for x in range(dimX):
                mesh_cords.append([x, y, z])
    return torch.tensor(mesh_cords)


def create_mesh(dimX, dimY):
    nw_dict = {}
    cords = []
    for y in range(dimY):
        for x in range(dimX):
            cords.append((x, y))
    for idx, line in enumerate(cords):
        nw_dict[str(idx)] = {'cor': cords[idx]}
    return nw_dict

def id2coor_batch(id,dimX,dimY):
    coord = torch.zeros(id.shape[0],2).int().to(device='cuda')
    coord[:,1] = id / dimX  # z cord
    # coord[:,1] = (id-dimX*dimY) /dimX
    coord[:,0] = (id-coord[:,1]*dimX)
    return coord

def test_cost2(mapp,traff_dict,batch,nw_dict):
    cost=torch.zeros(batch).to(device='cuda')
    for i in traff_dict:
        p1 = (mapp == traff_dict[i]['src']).nonzero(as_tuple=True)[1]
        p2 = (mapp == traff_dict[i]['dst']).nonzero(as_tuple=True)[1]
        p1_coor = id2coor_batch(p1,4,4)
        p2_coor = id2coor_batch(p2,4,4)
        hops = torch.abs(p1_coor[:,0]-torch.abs(p2_coor[:,0]))+torch.abs(p1_coor[:,1]-torch.abs(p2_coor[:,1]))
        cost += traff_dict[i]['val']* hops
    return cost
#%%
if __name__ == '__main__':
    # create a dataloader
    # predicted_mappings[0] = torch.tensor([13,6,3,10,2,12,0,5,4,7,8,9,15,14,11,1]).to('cuda:0')
    dimX = 4
    dimY = 4
    traffic_file = open('traffic_benchmark/vopd_traffic.txt', 'r+')
    tmp = open('traffic_benchmark/vopd_traffic.txt', 'r+')
    # traff_data = torch.load('data/traffic_32.pt')
    num_list = [float(num) for num in tmp.read().split() if num.isnumeric()]
    max_val = max(num_list)
    traff_dict = cr_trDic(traffic_file, max_val)   # traffic dictionary
    nw_dict = create_mesh(dimX, dimY)  # create 2-D mesh
    mapp = torch.tensor([13,6,3,10,2,12,0,5,4,7,8,9,15,14,11,1]).to('cuda:0')
    mapp = mapp.repeat(10,1)
    test_cost2(mapp,traff_dict,10,nw_dict) 


    import math
    batch_size = 4
    graph_size = 6
    from utils.datagenerate import generate_graph_data_list, generate_distance_matrix
    from torch_geometric.loader import DataLoader
    data_list = generate_graph_data_list(graph_size=graph_size, num_graphs=batch_size)
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    num_samples = 3
    predicted_mappings = torch.zeros(num_samples * batch_size, graph_size)
    for i in range(num_samples*batch_size):
        predicted_mappings[i,:] = torch.randperm(graph_size)
    predicted_mappings = predicted_mappings.long()
    data = next(iter(data_loader))
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(graph_size/n)
    distance_matrix = generate_distance_matrix(n,m)
    baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)

