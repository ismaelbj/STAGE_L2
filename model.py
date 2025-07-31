import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, TransformerConv, SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import degree
from torch_scatter import scatter


class DMPNN(nn.Module):
    def __init__(self, n_feats, n_iter):
        super(DMPNN, self).__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(n_feats, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):
        edge_index = data.edge_index

        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr

        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]],
                          data.line_graph_edge_index[1],
                          dim_size=edge_attr.size(0),
                          dim=0, reduce='add')
            out = edge_attr + out

            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(torch.tanh(self.lin_gout(gout)))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)

        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)
        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super(LinearBlock, self).__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)
        return x


class DrugEncoder(nn.Module):
    def __init__(self, hidden_dim, n_iter):
        super(DrugEncoder, self).__init__()
        self.line_graph = DMPNN(hidden_dim, n_iter)

    def forward(self, data):
        x = self.line_graph(data)
        return x


class MPNN_Block(nn.Module):
    def __init__(self, hidden_dim, n_iter):
        super(MPNN_Block, self).__init__()
        self.drug_encoder = DrugEncoder(hidden_dim, n_iter)
        self.readout = GlobalAttentionPool(hidden_dim)

    def forward(self, data):
        data.x = self.drug_encoder(data)
        global_graph_emb = self.readout(data.x, data.edge_index, data.batch)
        return data, global_graph_emb


class MPNN_DDI(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, n_iter, kge_dim, rel_total):
        super().__init__()
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = 3
        self.lin_edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.blocks = []
        for i in range(self.n_blocks):
            block = MPNN_Block(hidden_dim, n_iter=n_iter)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels = triples
        h_data.x = self.mlp(h_data.x)
        t_data.x = self.mlp(t_data.x)

        h_data.edge_attr = self.lin_edge(h_data.edge_attr)
        t_data.edge_attr = self.lin_edge(t_data.edge_attr)

        repr_h, repr_t = [], []
        for block in self.blocks:
            out1, out2 = block(h_data), block(t_data)
            h_data = out1[0]
            t_data = out2[0]
            repr_h.append(out1[1])
            repr_t.append(out2[1])

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores
    
    

class GAT_DDI(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, n_iter=3, kge_dim=64, rel_total=86, heads=4, dropout=0.1):
        super(GAT_DDI, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_iter = n_iter
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(node_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(1, n_iter):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        self.kge_embedding = nn.Embedding(rel_total, kge_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * heads * 2 + kge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        head_pairs, tail_pairs, rel = inputs

        for gat in self.gat_layers:
            head_pairs.x = F.elu(gat(head_pairs.x, head_pairs.edge_index))
            tail_pairs.x = F.elu(gat(tail_pairs.x, tail_pairs.edge_index))

        head_embed = global_add_pool(head_pairs.x, head_pairs.batch)
        tail_embed = global_add_pool(tail_pairs.x, tail_pairs.batch)

        rel_embed = self.kge_embedding(rel)

        out = torch.cat([head_embed, tail_embed, rel_embed], dim=-1)
        out = self.mlp(out)

        return out.view(-1)


class GIN_DDI(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_iter, kge_dim, rel_total, dropout=0.2):
        super(GIN_DDI, self).__init__()
        self.n_iter = n_iter
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_iter):
            mlp = nn.Sequential(
                nn.Linear(node_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.kge = nn.Embedding(rel_total, kge_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + kge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        head, tail, rel = inputs

        x1, edge_index1, batch1 = head.x, head.edge_index, head.batch
        x2, edge_index2, batch2 = tail.x, tail.edge_index, tail.batch

        for conv, bn in zip(self.convs, self.bns):
            x1 = conv(x1, edge_index1)
            x1 = bn(x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

            x2 = conv(x2, edge_index2)
            x2 = bn(x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x1 = global_add_pool(x1, batch1)
        x2 = global_add_pool(x2, batch2)

        r = self.kge(rel).squeeze(1)

        x = torch.cat([x1, x2, r], dim=-1)
        out = self.fc(x)

        return out.view(-1)
    
class LocalTransformer_DDI(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_iter, num_heads, kge_dim, rel_total, dropout):
        super(LocalTransformer_DDI, self).__init__()

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_iter)

        self.kge = nn.Embedding(rel_total, kge_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + kge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        head, tail, rel = inputs

        x1, batch1 = head.x, head.batch
        x2, batch2 = tail.x, tail.batch

        x1_proj = self.node_proj(x1)
        x2_proj = self.node_proj(x2)

        x1_out = []
        for i in batch1.unique():
            idx = (batch1 == i)
            x1_sub = x1_proj[idx].unsqueeze(0) 
            x1_enc = self.encoder(x1_sub).squeeze(0).mean(dim=0)  
            x1_out.append(x1_enc)

        x2_out = []
        for i in batch2.unique():
            idx = (batch2 == i)
            x2_sub = x2_proj[idx].unsqueeze(0)
            x2_enc = self.encoder(x2_sub).squeeze(0).mean(dim=0)
            x2_out.append(x2_enc)

        x1_pool = torch.stack(x1_out, dim=0)
        x2_pool = torch.stack(x2_out, dim=0)

        r = self.kge(rel).squeeze(1) 

        x = torch.cat([x1_pool, x2_pool, r], dim=-1)
        out = self.fc(x)

        return out.view(-1)
    

class GraphSAGE_DDI(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_iter, kge_dim, rel_total, dropout=0.2):
        super(GraphSAGE_DDI, self).__init__()
        self.n_iter = n_iter
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_iter):
            in_dim = node_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.kge = nn.Embedding(rel_total, kge_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + kge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        head, tail, rel = inputs

        x1, edge_index1, batch1 = head.x, head.edge_index, head.batch
        x2, edge_index2, batch2 = tail.x, tail.edge_index, tail.batch

        for conv, bn in zip(self.convs, self.bns):
            x1 = conv(x1, edge_index1)
            x1 = bn(x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

            x2 = conv(x2, edge_index2)
            x2 = bn(x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x1 = global_add_pool(x1, batch1)
        x2 = global_add_pool(x2, batch2)

        r = self.kge(rel).squeeze(1)

        x = torch.cat([x1, x2, r], dim=-1)
        out = self.fc(x)

        return out.view(-1)