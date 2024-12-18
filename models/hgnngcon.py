import dhg
import torch
import torch.nn as nn
from dhg.structure.hypergraphs import Hypergraph
import torch.nn.functional as F
# from datasets.visual_data import return_adj


class HGNNGConv(nn.Module):
    """
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
            n_head=4,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.drop = nn.Dropout(drop_rate)
        # self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_channels // 2, out_channels),
            # nn.Sigmoid(),
        )
        self.low_cat = nn.Linear(2*out_channels, out_channels, bias=bias)
        self.out = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels),
            nn.Sigmoid(),
        )
        # Attention layer
        self.branch_SE = SEblock(channel=1)
        # Adaptive w
        # self.w = nn.Parameter(torch.tensor([0.1, 0.9]))
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor, G: dhg.Hypergraph, graph: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # normalization weight
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        if not self.is_last:
            x = self.theta(x)
        else:
            x = self.out(x)
        x_hg = G.v2v(x, aggr="mean")
        # x_g = graph.smoothing_with_GCN(x)
        x_g = graph.v2v(x, aggr='mean')
        x_fuse = (x_g + x_hg) / 2
        # x_cat = torch.cat([x_hg, x_g], dim=1)
        # x = self.branch_SE(x)
        # x_cat = self.low_cat(x_cat)
        x = w1 * x_fuse + w2 * x
        # x = self.mha(x)
        # print(w1,w2,w3)
        if self.bn is not None:
            x = self.bn(x)
        # x_fuse = F.relu(x_fuse, inplace=True)
        # x_fuse = F.dropout(x_fuse, self.dropout)
        if not self.is_last:
            x = self.drop(self.act(x))
            # x = F.elu(self.out_att(x, adj))
        return x


class SEblock(nn.Module):  # 注意力机制模块
    def __init__(self, channel):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        # branch = branch.view(branch.size(0), -1)
        # print("brain", branch.size())
        # 全连接层得到权重
        weight = self.fc(branch)

        # print("weight", weight.size())
        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w))
        # 乘积获得结果
        scale = weight * x
        return scale