import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# overall model: MPNN+CNN+clf
class MPNN_CNN(nn.Sequential):
    def __init__(self, **config):
        """
        MPNN : output the hidden representation of drug
        CNN : output the hidden representation of protein including attention mechanism
        Classifier : MLP for simple concatenation
        """
        super(MPNN_CNN, self).__init__()
        self.config = config
        self.model_drug = MPNN(self.config['hidden_dim_drug'], self.config['mpnn_depth'])
        self.model_protein = CNN(**config)
        self.classifier = Classifier(**config)

    def forward(self, v_D, v_P):
        v_D = self.model_drug(v_D)
        v_P = self.model_protein(v_P,v_D)
        v_f = self.classifier(v_P,v_D)
        return v_f


class Classifier(nn.Sequential):  # self-defined
    def __init__(self, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.input_dim_protein = config['hidden_dim_protein']
        self.hidden_dims = config['cls_hidden_dims']
        self.visual_attention=config['visual_attention']
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [2]
        if config['attention']:
            if config['concatenation']:
                dims[0]+=config['cnn_target_filters'][-1]
            else:
                dims[0]=self.input_dim_drug+config['cnn_target_filters'][-1]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(self.hidden_dims)+1)])
        self.dropout = nn.Dropout(0.25)
        self._initialize()


    def _initialize(self):
        """
        Use Kaiming Normalization
        """
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)


    def forward(self, v_P, v_D):
        #simple concatenation
        v_f = torch.cat((v_P, v_D), 1)

        #automatically filter out nan value to prevent from thrashing backpropagation
        fault_idx = torch.logical_or(torch.any(torch.isnan(v_f), dim=1), torch.any(torch.isinf(v_f), dim=1))  # Batch size
        v_f[fault_idx] = 0

        #pass out the multi-layer perceptron with dropout
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.leaky_relu_(self.dropout(l(v_f)))
        return v_f


class Attention(nn.Sequential):
    def __init__(self,**config):
        """

        """
        super(Attention,self).__init__()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=config['cnn_target_filters'][-1])
        self.hidden_dim_drug = config['hidden_dim_drug']
        self.visual_attention=config['visual_attention']
        self.W_attention = nn.Linear(config['cnn_target_filters'][-1], config['cnn_target_filters'][-1])
        self.drug_reduce_dim = nn.Linear(self.hidden_dim_drug, config['cnn_target_filters'][-1])

    def forward(self, v_P, v_D):
        #(64,48,859)
        p_raw = F.adaptive_max_pool1d(v_P, output_size=100)
        fault_idx = torch.logical_or(torch.any(torch.isnan(v_D), dim=1), torch.any(torch.isinf(v_D), dim=1))  # Batch size
        v_D[fault_idx] = 0
        #(64,48,100)
        reduced_drug = self.drug_reduce_dim(v_D)
        #(64,48)
        #residual connection
        query_drug = torch.unsqueeze(torch.relu(self.W_attention(reduced_drug)),1)
        #(64,1,48)
        attention_p = torch.transpose(torch.relu(self.W_attention(torch.transpose(p_raw,1,2))),1,2)
        #(64,48,100)
        #(1)100 keys (2)48 dimensions for every key (3)48 dimensions for every query
        weights = torch.softmax(torch.einsum('ijk,ikl->ijl', query_drug, attention_p)/np.sqrt(48),dim=2)
        weights = torch.unsqueeze(torch.squeeze(weights,1),-1)
        if self.visual_attention:
            np.save("attention_weight.npy",torch.squeeze(weights,-1).detach().to('cpu').numpy(),allow_pickle=True)
        #(64,100,1)
        ys = torch.einsum('ijk,ilj->ilk', weights, attention_p)
        #(64,48,1)
        v_P_attention=torch.squeeze(ys,-1)
        v_P_attention = self.layernorm(v_P_attention)
        return v_P_attention


class CNN(nn.Sequential):
    def __init__(self, **config):
        super(CNN, self).__init__()
        in_channel = [26] + config['cnn_target_filters']
        kernels = config['cnn_target_kernels']
        self.layer_size = len(config['cnn_target_filters'])
        self.visual_attention=config['visual_attention']
        self.concatenation=config['concatenation']
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channel[i],
                                              out_channels=in_channel[i + 1],
                                              kernel_size=kernels[i]) for i in range(self.layer_size)])
        self.convs = self.convs.float()
        self.attention = config['attention']
        protein_size = self.simulate_output((26, 1000))
        self.fc = nn.Linear(protein_size, config['hidden_dim_protein'])
        self.Attention=Attention(**config)


    def simulate_output(self, shape):
        eg = 1
        input = Variable(torch.rand(eg, *shape))
        output_feat= self.conv_op(input.float())
        output_feat= self.pooling(output_feat)
        n_size = output_feat.data.view(eg, -1).size(1)
        return n_size


    def conv_op(self, x):
        for i,l in enumerate(self.convs):
            x = F.relu(l(x))
            if i == self.layer_size - 1:
             np.save("conv_out.npy",x.detach().cpu().numpy(),allow_pickle=True)
        return x


    def pooling(self, x):
       x = F.adaptive_max_pool1d(x, output_size=1)
       return x


    def forward(self, v_P, v_D):
        v_P = self.conv_op(v_P.float())
        v_P_attention=None
        if self.attention:
            v_P_attention=self.Attention(v_P,v_D)
        v_P = self.pooling(v_P)
        v_P = v_P.view(v_P.size(0), -1)
        v_P = self.fc(v_P.float())
        if self.attention:
            if self.concatenation:
                v_P=torch.cat((v_P_attention,v_P),1)
            else:
                v_P=v_P_attention
        return v_P


class MPNN(nn.Sequential):
    def __init__(self, hid_size, depth):
        super(MPNN, self).__init__()
        self.hid_size = hid_size
        self.depth = depth
        self.input_layer = nn.Linear(ATOM_FDIM + BOND_FDIM, self.hid_size, bias=False)
        self.output_layer = nn.Linear(ATOM_FDIM + self.hid_size, self.hid_size)
        self.graph_layer = nn.Linear(self.hid_size, self.hid_size, bias=False)

    def forward(self, x):
        #parameter definition
        feature_atom, feature_bond, graph_atom, graph_bond, N_ab = x
        N_ab = torch.squeeze(N_ab, dim=1)
        atom_dict = []
        N_atom = 0
        N_bond = 0
        f_atom, f_bond, g_atom, g_bond = [], [], [], []

        # add features and graph
        for one_record in range(N_ab.shape[0]):
            a_num = int(N_ab[one_record][0].item())
            b_num = int(N_ab[one_record][1].item())
            f_atom.append(feature_atom[one_record, :a_num, :])
            f_bond.append(feature_bond[one_record, :b_num, :])
            g_atom.append(graph_atom[one_record, :a_num, :] + N_atom)
            g_bond.append(graph_bond[one_record, :b_num, :] + N_bond)
            atom_dict.append((N_atom, a_num))
            N_atom += a_num
            N_bond += b_num
        f_atom = create_var(torch.cat(f_atom, 0)).to(device)
        f_bond = create_var(torch.cat(f_bond, 0)).to(device)
        g_atom = create_var(torch.cat(g_atom, 0).long()).to(device)
        g_bond = create_var(torch.cat(g_bond, 0).long()).to(device)
        emb_input = self.input_layer(f_bond)
        mes = F.relu(emb_input)

        # build gnn
        for layer in range(self.depth - 1):
            nei_mes = index_select_ND(mes, 0, g_bond)
            nei_mes = nei_mes.sum(dim=1)
            nei_mes = self.graph_layer(nei_mes)
            mes = F.relu(emb_input + nei_mes)
        nei_mes = index_select_ND(mes, 0, g_atom).sum(dim=1)
        hid = torch.cat([f_atom, nei_mes], dim=1)
        hid = F.relu(self.output_layer(hid))
        out = [torch.mean(hid.narrow(0, a, numm), 0) for a, numm in atom_dict]
        out = torch.stack(out, 0)
        return out
