from models.gcl import E_GCL, unsorted_segment_sum
from .sort import embed_vec_sort, calc_ug, prep_vec_for_embed, count_num_vecs_cloud
import torch
from torch import nn


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, device='cpu', act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, sep_nf=None, exp=False, two_dim=False, n_nodes=25):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, sep_nf=sep_nf)

        del self.coord_mlp
        self.act_fn = act_fn
        self.exp = exp
        self.two_dim= two_dim
        self.device = device
        if not self.two_dim:
            self.embed = embed_vec_sort(3, 29, sep_nf, varying_set_size=True, learnable_weights=True)
        if self.two_dim:
            self.embed = embed_vec_sort(2, 29, sep_nf, varying_set_size=True, learnable_weights=True)
        self.to(device)
    def coord2gram(self, edge_index, coord): # needs testing!
        """"
        edge index: tuple of vectors specifying which edge indices are "linked"
        coord: concatenated point clouds in bach mode (contiguous)

        return: local frame per pair, shape (edge_index.size(0), d^2 + 2nd + 1)
        """
        row, col = edge_index
        #prep for global features
        final_index_list = count_num_vecs_cloud(edge_index, coord)
        vec_for_embed = prep_vec_for_embed(coord, final_index_list, edge_index)
        ug_vec = calc_ug(vec_for_embed, edge_index)
        (gram, projections) = torch.split(ug_vec, [3, ug_vec.size(2)-3], dim=2) #shape b x d  x (n+1)
        #pad with zeros for global size
        assert(projections.size(2) <= 29)
        pad_proj = torch.zeros(projections.size(0), projections.size(1), 29 - projections.size(2), device=self.device)
        projections = torch.cat([projections, pad_proj], dim=2)
        #refit projections without the upper most row
        d_temp, n_temp = projections.size(-2), projections.size(-1)
        non_empty_mask = projections.abs().sum(dim=1).bool().int().reshape(projections.size(0), 1, n_temp) # list of 0/1 for when plgging into Tal's embedding
        fit = torch.cat([non_empty_mask, projections], dim=1)
        if self.two_dim:
            projections = projections[:, 1:3, :]
        if self.exp:
            projections = torch.exp((-0.5)*projections)
        embedding= self.embed(fit)

        return torch.flatten(gram, start_dim=1), embedding #flatten
    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr, node_attr, n_nodes):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        _, embedding = self.coord2gram(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, embedding, edge_attr)

        edge_feat = edge_feat * edge_mask



        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr




class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1, sep_nf=1, exp=False, two_dim=False, n_nodes=25):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embed_dec = embed_vec_sort( hidden_nf, 30, hidden_nf, varying_set_size=True, learnable_weights=True )

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, device=device,act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention, sep_nf=sep_nf, exp=exp, two_dim=two_dim, n_nodes=n_nodes))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = h.transpose(dim0=-1, dim1=-2) #h : ( b , hidden_nf, n_nodes)
        #pad h so number dim 2 is 29 (or 30) rather than self.n_nodes
        pad = torch.zeros(h.size(0), self.hidden_nf, 30 - n_nodes, device=self.device)
        h_pad = torch.cat([h, pad], dim=2)
        non_empty_mask = h_pad.abs().sum(dim=1).bool().int().reshape(h_pad.size(0),1, 30)
        h = torch.cat([non_empty_mask, h_pad], dim=1)
        #apply embedding
        h = self.embed_dec(h)
        pred = self.graph_dec(h)
        return pred.squeeze(1)




