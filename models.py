""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class GraphSAGEConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GraphSAGEConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # Define the linear transformation layers
        self.W = nn.Linear(in_features * 2, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W.weight.data)

    def forward(self, x, adj):
        # Aggregate neighbor features
        aggregated = torch.matmul(adj, x)  # Aggregate neighbors
        combined = torch.cat([x, aggregated], dim=1)  # Concatenate self and neighbor features
        h = self.W(combined)  # Linear transformation
        h = F.relu(h)  # Non-linear activation
        h = F.dropout(h, self.dropout, training=self.training)  # Apply dropout
        return h

class GraphSAGE_E(nn.Module):
    def __init__(self, in_dim, hgat_dim=[128,256,512], dropout=0.5):
        super(GraphSAGE_E, self).__init__()
        # Initialize GraphSAGE layers
        self.sage1 = GraphSAGEConvolution(in_dim, hgat_dim[0], dropout)
        self.sage2 = GraphSAGEConvolution(hgat_dim[0], hgat_dim[1], dropout)
        self.sage3 = GraphSAGEConvolution(hgat_dim[1], hgat_dim[2], dropout)

    def forward(self, x, adj):
        # Forward pass through GraphSAGE layers
        x = self.sage1(x, adj)
        x = self.sage2(x, adj)
        x = self.sage3(x, adj)
        return x

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GraphSAGE_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict