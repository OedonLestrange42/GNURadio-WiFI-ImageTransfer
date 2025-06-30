import torch
import math

from torch import Tensor
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, node_features, edge_index):
        # node_features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(node_features, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


if __name__ == '__main__':
    """from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='.', name='Cora')
    data = dataset[0]"""

    batch_size = 8
    user_num = 6
    in_channel = 3
    model = GNN(in_channel, 16, 64)

    G = torch.randn(batch_size, user_num, in_channel)  # batch_size:8 node_num:6 in_channel:3
    edge_num = math.comb(user_num, 2)
    edge_index = torch.randint(0, 5, (2, edge_num))

    pred = model(G, edge_index)

    print(pred)
