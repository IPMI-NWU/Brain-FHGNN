import torch
from src.models import GAT, GCN, BrainNN, MLP
from src.models.hgnn import HGNN
from dhg.models import HGNNP
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes):
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes, num_classes=2),
                      MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                      GAT(num_features, args, num_nodes, num_classes=2),
                      MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    elif model_name == 'hgnn':
        model = HGNN(in_ch=num_features,
                     n_class=200,
                     n_hid=200,
                     dropout=0.5).to(device)
    elif model_name == 'hgnnp':
        model = HGNNP(in_channels=num_features,
                      hid_channels=128,
                      num_classes=2,
                      use_bn=True)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model
