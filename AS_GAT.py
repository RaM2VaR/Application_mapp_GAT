#%%
from utils.datagenerate import generate_distance_matrix, generate_distance_matrix_3D, get_mesh_dimensions_newer
import math
import torch
from torch_geometric.loader import DataLoader
from model import GAT
from model import PointerNet
from utils.utils import  communication_cost_multiple_samples,LPNet_pred,booksim_latency,communication_energy_multiple_samples
from torch import nn
import matplotlib.pyplot as plt
from train.validation import beam_search_data
from timeit import default_timer as timer
import argparse
# from graphdataset import get_transform
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path to dataset', type=str,default ='data_TGFF/data_single_TGFF1_norm_121.pt')
parser.add_argument('--max_iter', help='max iterations', type=int, default=1000)
parser.add_argument('--num_samples', help='number of unique solutions to be sampled in each iteration', type=int, default=128)
parser.add_argument('--pretrained_model_path', help='path to pretrained model', type=str, default=None)
parser.add_argument('--lr', help='learning rate', type=float, default=0.002)
parser.add_argument('--three_D', help='use a fully connected 3D NoC with 2 layers in the Z direction', action='store_true',default=None)
parser.add_argument('--decoding_type', help='type of decoding', type=str, default='sampling')
parser.add_argument('--transformer_version', help='version of transformer', type=str, default='v2')
parser.add_argument('--model', help='type of model', type=str, default='lstm')
parser.add_argument('--obj_fun',help='objective function (comm_cost,LPNet, sim_lat or comm_energy)',type=str,default='comm_cost')
parser.add_argument('--inj_rate',help='injection rate for LPNet model',type=float,default=0.001)
parser.add_argument('--n_layers',help='LSTM layer count',type=int,default=1)
parser.add_argument('--hidd_dim',help='LSTM hidd_layer dimension',type=int,default=0)

# GAT parameters
parser.add_argument('--ENC',help='Do you want encoder YES or No',type=str,default=True)
parser.add_argument('--num_GAT_layers',help='No. of GAT laers',type=int,default=1)
parser.add_argument('--num_heads_per_layer',help='multi attention head',type=int,default=[2])
parser.add_argument('--num_features_per_GATlayer',help='output feature vec size',type=int,default=[121,256])
parser.add_argument('--GAT_dropout',help='GAT dropout',type=float,default=0.2)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
def load_and_process_data(dataset_path, device = torch.device("cpu")):
    data = torch.load(dataset_path, map_location=device)
    # data = data[0]  # comment this line for other than bench mark traffic Ramesh

    dataloader = DataLoader([data], batch_size=1)
    data = next(iter(dataloader))
    return data
#%%
def load_model(graph_size=121, device = torch.device('cpu'), feature_scale = 1, pretrained_model_path = None, model='lstm', transformer_version='v2'):
    if model == 'lstm':        
        mpnn_ptr =PointerNet(num_GAT_layers=args.num_GAT_layers, num_heads_per_layer = args.num_heads_per_layer,
                 num_features_per_GATlayer = args.num_features_per_GATlayer, GAT_dropout = args.GAT_dropout,
                 ENC = args.ENC,input_dim=graph_size, embb_dim=args.num_features_per_GATlayer[-1], hidd_dim=args.num_features_per_GATlayer[-1], 
                 dec_layers=args.n_layers, dropout=0.4, mh_attn = 4, device=device, logit_clipping=False)
    elif model == 'transformer':
        mpnn_ptr = GAT_Transformer(input_dim=graph_size, embedding_dim=graph_size + 8, hidden_dim=graph_size + 16, K=3, n_layers=1, p_dropout=0, device=device, logit_clipping=True, version=transformer_version)
    if pretrained_model_path is not None:
        mpnn_ptr.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    mpnn_ptr.to(device)
    return mpnn_ptr

#%%
max_graph_size = 121
num_samples = args.num_samples
# transform = get_transform(max_graph_size, device)
data = load_and_process_data(args.dataset, device)
# data.x = torch.ones(data.num_nodes, max_graph_size, device=device)
# print('pretrained model prediction only')
# data = data[0]  # comment this line for other than bench mark traffic Ramesh
graph_size = data.num_nodes
if args.obj_fun == 'LPNet':
    print('-----------fitness function is packet latency (LPNet)------')
elif args.obj_fun == 'comm_cost':
    print('-----------fitness function is communication cost------')
elif args.obj_fun == 'comm_energy':
    print('-----------fitness function is communication energy------')
elif args.obj_fun == 'sim_lat':
    print('*********** fitness function is latency(simulator) ************')
if args.three_D:
    n = math.ceil(math.sqrt(graph_size/2))
    m = math.ceil(graph_size/ (n * 2))
    l = 2
    distance_matrix = generate_distance_matrix_3D(n, m, l).to(device)
else:
    if args.model == 'transformer' and args.transformer_version == 'v2':
        n, m = get_mesh_dimensions_newer(graph_size)
        distance_matrix = generate_distance_matrix(n, m, numbering="new").to(device)
    else:
        n, m = get_mesh_dimensions_newer(graph_size)
        distance_matrix = generate_distance_matrix(n, m, numbering="new").to(device)
        # n = math.floor(math.sqrt(graph_size))
        # m = math.ceil(graph_size / n)
if args.pretrained_model_path is None:
    feature_scale = data.edge_attr.max()
else:
    feature_scale = 1
mpnn_ptr = load_model(graph_size, device, feature_scale=feature_scale, pretrained_model_path=args.pretrained_model_path, model=args.model, transformer_version=args.transformer_version)
mpnn_ptr.train()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=args.lr)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.93)
best_mapping = None
best_cost = float('inf')
baseline = torch.tensor(0.0)
loss_list = []
num_epochs = args.max_iter
count_not_decrease = 0
# start measuring time
start = timer()
# predicted_mappings = torch.zeros(num_samples,graph_size,dtype=torch.int64).to(device)
# log_probs = torch.zeros(num_samples,dtype=torch.float32).to(device)
for epoch in range(num_epochs):
    mpnn_ptr.train()
    mpnn_ptr.decoding_type = args.decoding_type
    # if(graph_size > 64):
    #     iter = int(num_samples/256)
    #     for i in range(iter):
    #         predicted_mappings[256*i:(256*i+256)], log_probs[256*i:(256*i+256)] = mpnn_ptr(data, 256)[:2]
    # else:
    predicted_mappings, log_probs = mpnn_ptr(data, num_samples)[:2]
    if args.obj_fun == 'LPNet':
        penalty = LPNet_pred(predicted_mappings,graph_size,args.inj_rate,args.dataset)
        penalty = torch.tensor(penalty).to(device)
    elif args.obj_fun == 'sim_lat':
        penalty = booksim_latency(predicted_mappings,graph_size,args.inj_rate,args.dataset)
        penalty = torch.tensor(penalty).to(device)
    elif args.obj_fun == 'comm_cost':
        penalty = communication_cost_multiple_samples(data.edge_index, 
        data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)
    elif args.obj_fun == 'comm_energy': 
        penalty = communication_energy_multiple_samples(data.edge_index, 
        data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)
    else:
        raise ValueError('penalty function not defined')
    min_penalty = torch.argmin(penalty)
    if penalty[min_penalty] < best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = predicted_mappings[min_penalty]
        count_not_decrease = 0
    baseline = penalty.mean()
    loss = (1/(num_samples-1)) * torch.sum((penalty.detach() - baseline.detach())*log_probs)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
    optim.step()
    if epoch % 20 == 0:
        # mpnn_ptr.eval()
        # mpnn_ptr.decoding_type = 'greedy'
        # mapping, cost = beam_search_data(mpnn_ptr, data, distance_matrix, 128,args.obj_fun,
        #                                     graph_size,args.inj_rate,args.dataset) #1024
        # if cost < best_cost:
        #     best_cost = float(cost)
        #     best_mapping = mapping[0]
        #     count_not_decrease = 0
        print(f'Epoch: {epoch + 1:4}/{num_epochs} Min Comm Cost: {best_cost:8.2f}   Avg Comm Cost: {penalty.mean():8.2f}')
        # print(f'{best_mapping}')
    # break the training loop if min_penalty is not decreasing for consecutive 10000 epochs
    # if cost >= best_cost:
    count_not_decrease += 1
    # else:
        # count_not_decrease = 0
    if count_not_decrease > 500:
        print('Early stopping at epoch {}'.format(epoch))
        break
    loss_list.append(penalty[min_penalty].item())
    # lr_scheduler.step()
# stop measuring time
# use the model with the best cost to do greedy beam search
mpnn_ptr.eval()
mpnn_ptr.decoding_type = 'greedy'
# mapping, cost = beam_search_data(mpnn_ptr, data, distance_matrix, 512,args.obj_fun,
#                                     graph_size,args.inj_rate,args.dataset) #3072
# if cost < best_cost:
#     best_cost = float(cost)
#     best_mapping = mapping[0]
end = timer()
# torch.save(mpnn_ptr.state_dict(), f'./models_data/model_single_uniform_{graph_size}.pt')
print(f'Best cost: {best_cost}, time taken: {end - start}')
# print(f'Best mapping: {best_mapping}')
bst_mapp = best_mapping.to('cpu').tolist()
# torch.save(mpnn_ptr.state_dict(), f'saved_models_for_revision/model_for_inference'+args.dataset[-16:-3]+'.pt')
# file1 = open("results_for_revision/ATSR_hyp_tuning"+'.csv','a')
# file1.write(f'{args.dataset[-16:-3]},{args.lr},{args.n_layers},{args.hidd_dim},{args.num_samples},{best_cost},{end - start},{epoch},{bst_mapp}\n')

# # plot loss vs epoch
# fig, ax = plt.subplots()  # Create a figure and an axes.
# ax.plot(loss_list)  # Plot some data on the axes.
# ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
# ax.set_ylabel('communication cost')  # Add a y-label to the axes.
# ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
# fig.savefig(f'./plots/loss_single_uniform_{graph_size}_3.png')  # Save the figure.

# command to run with pretrained model:
# python3 active_search.py data_tgff/single/traffic_32.pt --lr 0.002 --pretrained_model_path models_data_final/model_16_01-10.pt --max_iter 5000 --num_samples 2048 --three_D
# command to run without pretrained model:
# python3 active_search.py data_tgff/single/traffic_72.pt --lr 0.001 --max_iter 10000 --num_samples 1024 --model transformer
# command to run with pretrained model and 3D:
# python3 active_search.py data_tgff/single/traffic_32.pt --lr 0.001 --max_iter 5000 --num_samples 1024 --three_D

## The final command for running with pretrained model
# python3 active_search.py data_tgff/single/traffic_72.pt --lr 0.002 --pretrained_model_path models_data_multiple/small/models_data/model_pretrain_04-24_17-37.pt --max_iter 5000

## AS using LPNet model
#python3 active_search.py graphs_for_publication/random_TGFF_16_2.pt --lr 0.001 --pretrained_model_path models_data_final/model_pretrain_04-24_17-37.pt --max_iter 500 --num_samples 1024 --obj_fun LPNet --inj_rate 0.001