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