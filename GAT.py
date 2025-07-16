class GATConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W.weight.data)
        init.xavier_uniform_(self.a.weight.data)

    def forward(self, x, adj):
        h = self.W(x)
        N = h.size(0)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(self.a(a_input).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT_E(nn.Module):
    def __init__(self, in_dim, hgat_dim=[256, 512, 1024], dropout=0.7):
        super().__init__()
        self.gat1 = GATConvolution(in_dim, hgat_dim[0], dropout)
        self.gat2 = GATConvolution(hgat_dim[0], hgat_dim[1], dropout)
        self.gat3 = GATConvolution(hgat_dim[1], hgat_dim[2], dropout)

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.relu(x)
        x = self.gat2(x, adj)
        x = F.relu(x)
        x = self.gat3(x, adj)
        x = F.relu(x)
        return x