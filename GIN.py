class GINConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        x = self.mlp(x)
        return x

class GIN_E(nn.Module):
    def __init__(self, in_dim, hgin_dim, dropout):
        super().__init__()
        self.gin1 = GINConvolution(in_dim, hgin_dim[0], dropout)
        self.gin2 = GINConvolution(hgin_dim[0], hgin_dim[1], dropout)
        self.gin3 = GINConvolution(hgin_dim[1], hgin_dim[2], dropout)

    def forward(self, x, adj):
        x = self.gin1(x, adj)
        x = F.relu(x)
        x = self.gin2(x, adj)
        x = F.relu(x)
        x = self.gin3(x, adj)
        x = F.relu(x)
        return x