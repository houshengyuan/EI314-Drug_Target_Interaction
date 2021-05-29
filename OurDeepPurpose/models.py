from torch import nn
import torch.nn.functional as F

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Sequential):   # self-defined
    def __init__(self, model_drug, model_protein, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.input_dim_protein = config['hidden_dim_protein']
        self.model_drug = model_drug
        self.model_protein = model_protein
        self.dropout = nn.Dropout(0.25)
        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [2]
        dims[0] = self.input_dim_drug+100
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(dims[i + 1])for i in range(layer_size)])
        self._initialize()

        # attention to protein using drug embeddings
        self.hidden_dim_protein = config['hidden_dim_protein']
        self.hidden_dim_drug = config['hidden_dim_drug']
        self.W_attention = nn.Linear(100, 100)
        self.drug_reduce_dim = nn.Linear(self.hidden_dim_drug, 100)

    def _initialize(self):
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)

    def forward(self, v_D, v_P):
        v_D = self.model_drug(v_D)
        v_P, p_raw = self.model_protein(v_P)
        p_raw = F.adaptive_max_pool1d(p_raw, output_size=100)
        # attention mechinism
        filter_size = p_raw.shape[-2]
        p_raw = torch.reshape(p_raw, (-1, 100))
        reduced_drug = self.drug_reduce_dim(v_D)
        query_drug = torch.relu(self.W_attention(reduced_drug))
        query_drug = torch.unsqueeze(query_drug, 1)
        attention_p = torch.relu(self.W_attention(p_raw))
        attention_p = torch.reshape(attention_p, (-1, filter_size, 100))
        weights = torch.tanh(torch.einsum('ijk,ilk->ilj', query_drug, attention_p))
        ys = weights * attention_p
        v_P_attention = torch.mean(ys, 1)
        # end of attention
        v_f = torch.cat((v_D, v_P_attention), 1)
        fault_idx = torch.logical_or(torch.any(torch.isnan(v_f), dim=1), torch.any(
            torch.isinf(v_f), dim=1))  # Batch size
        v_f[fault_idx] = 0
        for i, (l, bn) in enumerate(zip(self.predictor, self.bn)):
            if i == (len(self.predictor)-1):
                v_f = l(v_f)
            else:
                v_f = F.leaky_relu_(self.dropout(bn(l(v_f))))
        return v_f

class CNN(nn.Sequential):
    def __init__(self, _, **config):
        super(CNN, self).__init__()
        in_channel = [26] + config['cnn_target_filters']
        kernels = config['cnn_target_kernels']
        layer_size = len(config['cnn_target_filters'])
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channel[i],
                                              out_channels=in_channel[i+1],
                                              kernel_size=kernels[i]) for i in range(layer_size)])
        self.convs = self.convs.float()
        protein_size = self.simulate_output((26, 1000))
        self.fc = nn.Linear(protein_size, config['hidden_dim_protein'])

    def simulate_output(self, shape):
        eg = 1
        input = Variable(torch.rand(eg, *shape))
        output_feat, _ = self.conv_op(input.float())
        n_size = output_feat.data.view(eg, -1).size(1)
        return n_size

    def conv_op(self, x):
        for l in self.convs:
            x = F.relu(l(x))
        pool_out = F.adaptive_max_pool1d(x, output_size=1)
        return pool_out, x

    def forward(self, v):
        v, conv_out = self.conv_op(v.float())
        v = v.view(v.size(0), -1)
        v = self.fc(v.float())
        return v, conv_out


class MPNN(nn.Sequential):
    def __init__(self, hid_size, depth):
        super(MPNN, self).__init__()
        self.hid_size = hid_size
        self.depth = depth
        self.input_layer = nn.Linear(ATOM_FDIM + BOND_FDIM, self.hid_size, bias=False)
        self.output_layer = nn.Linear(ATOM_FDIM+self.hid_size, self.hid_size)
        self.graph_layer = nn.Linear(self.hid_size, self.hid_size, bias=False)

    def forward(self, x):
        feature_atom, feature_bond, graph_atom, graph_bond, N_ab = x
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
            g_atom.append(graph_atom[one_record, :a_num, :]+N_atom)
            g_bond.append(graph_bond[one_record, :b_num, :]+N_bond)
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
        for layer in range(self.depth-1):
            nei_mes = index_select_ND(mes, 0, g_bond)
            nei_mes = nei_mes.sum(dim=1)
            nei_mes = self.graph_layer(nei_mes)
            mes = F.relu(emb_input+nei_mes)
        nei_mes = index_select_ND(mes, 0, g_atom).sum(dim=1)
        hid = torch.cat([f_atom, nei_mes], dim=1)
        hid = F.relu(self.output_layer(hid))
        out = [torch.mean(hid.narrow(0, a, numm), 0) for a, numm in atom_dict]
        out = torch.stack(out, 0)
        return out
