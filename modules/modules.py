from torch import nn
import torch
from torch.distributions import Normal
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
from modules.sparsemax import *

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers = 1):
        super(ProjectionHead, self).__init__()

        self.head = nn.Sequential()
        for i in range(nlayers - 1): 
            """
             TODO: 利用nn.Sequential的add_module函数，参考下一行输出层，实现mlp，
                   对每一个i,添加一个名为linear_{i}的线性层和relu_{i}的非线性层,其中线性层的输入输出维度均为in_dim
            """
            self.head.add_module(f"linear_{i}", nn.Linear(in_dim, in_dim))
            self.head.add_module(f"relu_{i}", nn.ReLU())
        # 实现输出层
        self.head.add_module(f"linear_final", nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.head(x)

class MultiHeadLinearProjection(nn.Module):
    def __init__(self, in_dim, output_size, nlayers=1):
        super(MultiHeadLinearProjection, self).__init__()
        self.linears = nn.ModuleList()
        for i in output_size:
            self.linears.append(ProjectionHead(in_dim, i, nlayers))
        
    def forward(self, features):
        out = []
        for head in self.linears:
            out += [head(features) / (features.shape[1]**0.5)]
        return out
    
class HyperNet(nn.Module):
    def __init__(self, cfg):
        super(HyperNet, self).__init__()
        self.num_joints = cfg.num_joints
        encoder_hidden_dim = cfg.encoder_hidden_dim
        encoder_embedding_dim = cfg.encoder_embedding_dim

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_embedding_dim)
        )

        self.rpy_encoder = nn.Sequential(
            nn.Linear(3, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_embedding_dim)
        )

        concat_dim = encoder_embedding_dim * 2
        self.hidden_layer_sizes = [cfg.hypernet_hidden_size] * cfg.hypernet_num_hidden_layers
        dims = [concat_dim] + self.hidden_layer_sizes

        self.feature_net = nn.Sequential()
        for i in range(len(dims)-1):
            """
            TODO:
            在每次循环中：
            参考ProjectionHead中self.head，创建一个新的 nn.Sequential()子容器
            向子容器中添加以下三个组件（按顺序）：
            nn.BatchNorm1d(dims[i])- 批归一化层，输入维度为当前层的维度
            nn.Linear(dims[i], dims[i+1])- 全连接层，输入维度是当前层，输出维度是下一层
            nn.ReLU()- ReLU激活函数
            将这个子容器添加到 self.feature_net中
            添加方法参考下一行在循环结束后，单独添加最后的输出层
            """
            layer = nn.Sequential()
            layer.add_module(f"batchnorm_{i}", nn.BatchNorm1d(dims[i]))
            layer.add_module(f"linear_{i}", nn.Linear(dims[i], dims[i+1]))
            layer.add_module(f"relu_{i}", nn.ReLU())
            self.feature_net.add_module(f"layer_{i}", layer)
            
        self.feature_net.add_module(f"linear_final", nn.Linear(dims[-1], cfg.embedding_dim))

        num_parameters_list = []
        if cfg.jointnet_num_layers < 2:
            raise ValueError("jointnet_num_layers must be at least 2 (Input -> Output)")

        for i in range(1, self.num_joints + 1):
            # 输入层 (Input: i, Output: hidden_size)
            num_parameters_list += [cfg.jointnet_hidden_size * i,  # W1
                                    cfg.jointnet_hidden_size]      # b1
            
            # 中间隐藏层 (Input: hidden_size, Output: hidden_size)
            for _ in range(cfg.jointnet_num_layers - 2):
                num_parameters_list += [cfg.jointnet_hidden_size * cfg.jointnet_hidden_size, # W_mid
                                        cfg.jointnet_hidden_size]                            # b_mid
            
            # 输出层 (Input: hidden_size, Output: output_dim)
            num_parameters_list += [cfg.jointnet_hidden_size * cfg.jointnet_output_dim, # W_out
                                    cfg.jointnet_output_dim]
            
        self.projection = MultiHeadLinearProjection(cfg.embedding_dim, num_parameters_list, 1)

    def forward(self, inputs):
        pos_input = inputs[:, :3]
        rpy_input = inputs[:, 3:]
        
        pos_feat = self.pos_encoder(pos_input)
        rpy_feat = self.rpy_encoder(rpy_input)

        concat_feat = torch.cat((pos_feat, rpy_feat), dim=1)

        features = self.feature_net(concat_feat)

        weights = self.projection(features)
        return weights

class JointNetTemplate(nn.Module):
    def __init__(self, cfg):
        super(JointNetTemplate, self).__init__()
        self.hidden_layer_size = cfg.jointnet_hidden_size
        self.output_dim = cfg.jointnet_output_dim
        self.num_layers = cfg.jointnet_num_layers

    def forward(self, input, weights):
        out = input
        
        for i in range(self.num_layers):
            w = weights[i*2]
            b = weights[i*2 + 1]
            
            if i == 0:
                in_dim = input.shape[1]
                hidden_dim = self.hidden_layer_size
                if in_dim == 1:
                    out = out * w + b
                else:
                    out = torch.bmm(out.unsqueeze(1), w.reshape(w.shape[0], in_dim, hidden_dim)).squeeze(1) + b
            
            elif i == self.num_layers - 1:
                in_dim = self.hidden_layer_size
                hidden_dim = self.output_dim
                out = torch.bmm(out.unsqueeze(1), w.reshape(w.shape[0], in_dim, hidden_dim)).squeeze(1) + b
            
            else:
                in_dim = self.hidden_layer_size
                hidden_dim = self.hidden_layer_size
                out = torch.bmm(out.unsqueeze(1), w.reshape(w.shape[0], in_dim, hidden_dim)).squeeze(1) + b

            if i < self.num_layers - 1:
                out = torch.relu(out)
                
        return out
    
class MainNet(nn.Module):
    def __init__(self, cfg):
        super(MainNet, self).__init__()
        self.num_joints = cfg.num_joints
        self.num_gaussians = cfg.num_gaussians
        self.joint_template = JointNetTemplate(cfg)
        self.params_per_joint = cfg.jointnet_num_layers * 2

    def _create_distribution(self, out):
        """
        TODO:
        创建混合高斯分布
        1. 理解输入输出
        输入参数 out: 神经网络的输出张量，形状为 (batch_size, num_gaussians*3)
        前 num_gaussians个值表示各个高斯分布的均值
        接下来的 num_gaussians个值表示各个高斯分布的对数方差(需要取指数得到方差)
        最后 num_gaussians个值表示混合权重(当 num_gaussians > 1时)
        """
        means = out[:, :self.num_gaussians].unsqueeze(2)
        log_vars = out[:, self.num_gaussians:self.num_gaussians*2].unsqueeze(2)
        vars = log_vars.exp() + 1e-7
        """
        2. 计算混合权重
        如果 self.num_gaussians == 1:
        创建一个全1的张量作为权重,形状为 (batch_size, 1)
        使用 torch.ones(out.shape[0], 1, device=out.device)
        如果 self.num_gaussians > 1:
        从 out中提取权重部分:out[:, self.num_gaussians*2:]
        使用 Sparsemax()激活函数处理权重(确保权重和为1)
        调用 forward方法:Sparsemax().forward(...)
        """
        if self.num_gaussians == 1:
            selection_weights = torch.ones(out.shape[0], 1, device=out.device)
        elif self.num_gaussians > 1:
            selection_weights = Sparsemax().forward(out[:, self.num_gaussians*2:])
        """
        3. 创建混合分布
        使用 D.Categorical创建混合权重分布:
        参数:selection_weights(上一步计算的权重)
        """
        mix = D.Categorical(probs=selection_weights)
        """
        4. 创建高斯分量
        从 out中提取均值部分(参考从 out中提取权重部分:out[:, self.num_gaussians*2:])
        添加一个维度：.unsqueeze(2)使其形状变为 (batch_size, num_gaussians, 1)
        从 out中提取对数方差部分同上
        添加一个维度：.unsqueeze(2)
        取指数得到方差：.exp()
        添加一个小常数避免数值不稳定：+ 1e-7

        使用 D.Normal创建高斯分布:
        参数1:均值张量
        参数2:方差张量

        使用 D.Independent包装高斯分布:
        comp = D.Independent(上一步创建的高斯分布, 1) 这里1表示将最后一个维度视为事件维度
        """
        comp = D.Independent(Normal(loc=means, scale=vars.sqrt()), 1)
        """
        5. 组合分布
        使用 MixtureSameFamily将混合权重分布和高斯分量组合:
        参数1:mix(混合权重分布)
        参数2:comp(高斯分量)
        6. 返回结果
        混合高斯分布
        混合权重
        """
        dist = MixtureSameFamily(mix, comp) 
        return dist, selection_weights

    def forward(self, x, weights):
        x = x.unsqueeze(2)
        distributions, selection = [], []
        for i in range(x.shape[1] - 1):
            start_idx = self.params_per_joint * i
            end_idx = self.params_per_joint * (i + 1)
            out = self.joint_template(x[:, :i + 1].squeeze(2), weights[start_idx : end_idx])
            dist, sel = self._create_distribution(out)
            distributions.append(dist)
            selection.append(sel)

        return distributions, selection
    
    def validate(self, x, weights, lower, upper, init_joint_angles = None):
        samples, distributions = [], []

        curr_input = x[:, 0].unsqueeze(1)
        for i in range(self.num_joints):
            start_idx = self.params_per_joint * i
            end_idx = self.params_per_joint * (i + 1)
            out = self.joint_template(curr_input, weights[start_idx : end_idx])
            dist, _ = self._create_distribution(out)
            distributions.append(dist)

            sample = dist.sample().clip(lower[i], upper[i])

            if init_joint_angles is not None:
                """
                TODO :
                在有初始关节角度的情况下实现拒绝采样
                即：在一定步数内, 如果采样值与初始角度距离过远则直接拒绝, 再次采样
                注意这里每次抽样值都只是第i个关节角度
                """
                max_attempts = 10
                attempts = 0
                while torch.any(torch.abs(sample - init_joint_angles[i]) > 0.5) and attempts < max_attempts:
                    sample = dist.sample().clip(lower[i], upper[i])
                    attempts += 1
                    
            curr_input = torch.cat((curr_input, sample), dim = 1)
            samples.append(sample)

        return samples, distributions






