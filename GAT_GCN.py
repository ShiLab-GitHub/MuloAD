class GATConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W.weight.data)
        nn.init.xavier_normal_(self.a.weight.data)

    def forward(self, x, adj):
        h = self.W(x)
        N = h.size(0)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(self.a(a_input), negative_slope=self.alpha).squeeze(2)

        zero_vec = -9e15 * torch.ones_like(e)
        adj=adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)
        h_prime = F.elu(h_prime)

        return h_prime


class GCNConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W.weight.data)

    def forward(self, x, adj):
        h = self.W(x)
        N = h.size(0)

        h = torch.matmul(adj, h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)

        return h


class GAT_GCN_E(nn.Module):
    def __init__(self, in_dim, hgat_dim=[256, 512, 1024], hgcnn_dim=[256, 512, 1024], dropout=0.5):
        super().__init__()
        self.gat1 = GATConvolution(in_dim, 64, dropout)
        self.gat2 = GATConvolution(64, 128, dropout)
        self.gat3 = GATConvolution(128, 50, dropout)

        self.gcn1 = GCNConvolution(in_dim, 64, dropout)
        self.gcn2 = GCNConvolution(64, 128, dropout)
        self.gcn3 = GCNConvolution(128, 50, dropout)

    def forward(self, x, adj):
        h_gat = self.gat1(x, adj)
        h_gat = self.gat2(h_gat, adj)
        h_gat = self.gat3(h_gat, adj)

        h_gcn = self.gcn1(x, adj)
        h_gcn = self.gcn2(h_gcn, adj)
        h_gcn = self.gcn3(h_gcn, adj)

        h = torch.cat((h_gat, h_gcn), dim=1)
        return h