import torch
from utils.utils import communication_cost_multiple_samples,communication_energy_multiple_samples,LPNet_pred,booksim_latency
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
def validate_dataloader(model, dataloader:DataLoader, distance_matrix_dict, beam_width,fit_fun):
    comm_cost = 0
    for data in dataloader:
        data = data.to(model.device)

        graph_size = data.num_nodes // data.num_graphs
        distance_matrix = distance_matrix_dict[graph_size].to(model.device)
        model.eval()
        model.decoding_type = 'greedy'
        _, comm_cost_batch = beam_search_data(model, data, distance_matrix, beam_width,fit_fun,64,0.001,data)
        comm_cost += float(comm_cost_batch.sum())
    return comm_cost 
def beam_search_data(model, data, distance_matrix, beam_width,fit_fun,graph_size,inj_rate,dataset):
    """
    don't forget to set model.decoding_type = 'greedy' and model.eval() before calling this function
    """
    # print(distance_matrix.shape)
    # print(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mappings, _ = model(data, beam_width)
        # print(mappings)
        if fit_fun == 'LPNet':
            # None
            penalty = LPNet_pred(mappings,graph_size,inj_rate,dataset)
            penalty = torch.tensor(penalty).to(device)
        if fit_fun == 'sim_lat':
            # None
            penalty = booksim_latency(mappings,graph_size,inj_rate,dataset)
            penalty = torch.tensor(penalty).to(device)
        elif fit_fun == 'comm_cost':
            penalty = communication_cost_multiple_samples(data.edge_index, data.edge_attr, 
                    data.batch, distance_matrix, mappings, beam_width)
        elif fit_fun == 'comm_energy': 
            penalty = communication_energy_multiple_samples(data.edge_index, data.edge_attr, 
                    data.batch, distance_matrix, mappings, beam_width)
        else:
            raise ValueError('penalty function not defined')
        # comm_cost = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, beam_width)
        penalty, min_indices = penalty.view(beam_width, -1).min(dim=0)
        choosen_mappings = mappings.view(beam_width, data.num_graphs, -1)[min_indices, torch.arange(data.num_graphs)]
    return choosen_mappings, penalty


def beam_search_data_3D(model, data, beam_width,fit_fun,graph_size,inj_rate,dataset,env,pred_TSVs):
    """
    don't forget to set model.decoding_type = 'greedy' and model.eval() before calling this function
    """
    # print(distance_matrix.shape)
    # print(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pred_mappings, _ = model(data, beam_width)
        if fit_fun == 'sim_lat':
            # None
            penalty = booksim_latency(pred_mappings,graph_size,inj_rate,dataset)
            penalty = torch.tensor(penalty).to(device)
        elif fit_fun == 'comm_cost':
            penalty = env.test_cost2(pred_mappings, pred_TSVs[0].repeat(beam_width,1))
        elif fit_fun == 'comm_energy': 
            penalty = env.test_cost2(pred_mappings, pred_TSVs)
        else:
            raise ValueError('penalty function not defined')
        # comm_cost = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, beam_width)
        penalty, min_indices = penalty.view(beam_width, -1).min(dim=0)
        choosen_mappings = pred_mappings.view(beam_width, data.num_graphs, -1)[min_indices, torch.arange(data.num_graphs)]
        # choosen_TSVs = pred_TSVs.view(beam_width, data.num_graphs, -1)[min_indices, torch.arange(data.num_graphs)]
    return choosen_mappings, penalty#,choosen_TSVs