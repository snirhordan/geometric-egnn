import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F# This version supports varying-size-sets as inputs
def tr12(input):
   return torch.transpose(input, 1, 2)

    
class embed_vec_sort(nn.Module):
   # Calculates a permutation-invariant embedding of a set of vectors in R^d, using Nadav
   # and Ravina's sort embedding. It has two modes of operation:
   #
   # Constant set size: (varying_set_size = False)
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1
   #
   # Varying set size: (varying_set_size = True)
   # Here the input can contain any number of vectors. 
   # n-1 is taken to be the number of linear regions in the weight function, and should be
   # larger or equal to the maximal input set size. 
   # Input size: b x (d+1) x n, with b = batch size, d = input feature dimension
   # Output size: b x d_out. Default d_out is 2*d*n+1
   # The first row of the input should contain 1 or 0, corresponding to existing or absent input vectors.
   #
   # The weight interpolation was taken as in the paper:
   # Learning Set Representations with Featurewise Sort Pooling, by Zhang et al., 2020   
   #
   # learnable_weights - Treats the weights as learnable parameters
   def __init__(self, d, n, d_out = None, varying_set_size = False, learnable_weights = False):
      super().__init__()
      if torch.cuda.is_available():
            self.device = torch.device('cuda')
      else:
            self.device = torch.device('cpu')
      if d_out is None:
         d_out = 2*d*n+1

      self.d = d
      self.n = n
      self.d_out = d_out
      self.varying_set_size = varying_set_size

      if learnable_weights:
         torch.manual_seed(42)
         self.A = nn.Parameter(torch.randn([d, d_out], requires_grad=True, device=self.device)) 
         torch.manual_seed(42)
         self.w = nn.Parameter(torch.randn([1, n, d_out], requires_grad=True, device=self.device))
      else:
         torch.manual_seed(42)
         self.A = torch.randn([d, d_out], requires_grad=False, device=self.device)
         torch.manual_seed(42)
         self.w = torch.randn([1, n, d_out], requires_grad=False, device=self.device)
         self.register_buffer(name="sort-embedding matrix A", tensor=self.A)
         self.register_buffer(name="sort-embedding vector w", tensor=self.w)

   # Input: A vector k of size b, containing the set size at each batch
   # Output: A weight tensor w_out of size b x n x d_out, which is an interpolated
   #         version of w, such that w_out[r, :, t] corresponds to the weight vector self.w[0, :, t]
   #         after interpolation to the set size k[r].
   def interpolate_weights(self, k):
      b = torch.numel(k)

      # i corresponds to the sorted sample number in the output weight w_out
      i = (torch.arange(1, 1+self.n, device=self.device)).unsqueeze(0).unsqueeze(2)

      # j corresponds to the sorted sample number in the input weight self.w
      j = (torch.arange(1, 1+self.n, device=self.device)).unsqueeze(0).unsqueeze(0)

      # k contains the set size of each batch
      # Note that the roles of k and n here are replaced compared to those in the paper.
      k = k.unsqueeze(1).unsqueeze(1)
      
      interp_matrix = torch.clamp(1-torch.abs( (i-1)*(self.n-1)/(k-1) - (j-1) ), min=0)     
      w_out = torch.bmm(interp_matrix, self.w.repeat(b,1,1))

      return w_out

   def forward(self, input): #TODO add exponential
      if self.varying_set_size:
         assert input.shape[1] == self.d+1, 'Dimension 1 of input should be d+1'
         q = input[:,0,:]
         q = q.unsqueeze(1)
         X = input[:,1:,:]
      else:
         assert input.shape[1] == self.d, 'Dimension 1 of input should be d'
         X = input

      assert input.shape[2] == self.n, 'Dimension 2 of input should be n'

      prod = tr12( torch.tensordot( tr12(X), self.A, [[2], [0]] ) ) 

      if self.varying_set_size:
         prod[q.repeat(1,self.d_out,1) == 0] = np.inf
         k = torch.sum(q, dim=2).flatten()
         w = self.interpolate_weights(k)
      else:
         w = self.w   
      
      [prod_sort, inds_sort] = torch.sort(prod, dim=2)
      prod_sort[prod_sort == np.inf] = 0

      out = torch.sum( prod_sort * tr12(w), dim=2)

      return out


def count_num_vecs_cloud(edge_index, coord):
    """"
    For each local frame, count and organize vectors in cloud not in local frame
    return index list of indices of batched point cloud for embedding
    """
    row, col = edge_index
    col_tmp = col
    bincount = torch.bincount(row)#each index is number of times this int is present in row
    #bincount_col = torch.bincount(col) # should be identical
    #local frame index list
    init_index_list = torch.cat([row.unsqueeze(-1), col.unsqueeze(-1)], dim=1)
    #resize to fit n-1 incdices to prep for embedding
    empty = init_index_list.new_full(torch.Size([init_index_list.size(0), init_index_list.size(1) + bincount[0] -1 ]), 0) 
    #create tensor with lists of cloud indices to take to embed
    col = col.reshape(-1,bincount[0])
    cat_prep = col[row] 
    #filter out values present in col vector
    cat_prep = cat_prep[cat_prep != col_tmp.unsqueeze(-1).expand(-1,cat_prep.size(1))].reshape(cat_prep.size(0),-1)
    final_index_list = torch.cat([init_index_list, cat_prep], dim=1) #checked

    return final_index_list

def prep_vec_for_embed(coord, final_index_list, edge_index):
    """"
    Returns big vector of candidate for projections along batches
    """
    row, col = edge_index
    #populate big vector with indexed point cloud
    populate = coord[final_index_list]
    #add the perp vector
    perp_vec = torch.linalg.cross( coord[row] , coord[col] ).unsqueeze(-2)
    #normalize perp vector
    row_norm = torch.linalg.norm(coord[row], dim=1)
    col_norm = torch.linalg.norm(coord[col], dim=1)
    #candidates is for differentiable maximum
    candidates = torch.cat([row_norm.unsqueeze(1), col_norm.unsqueeze(1)], dim=1)
    #calculate differentiable maximum for each 2-tuple choice
    rlsoftmax = torch.log(torch.sum(torch.exp(candidates), dim=1)).reshape(perp_vec.size(0),1,1)
    perp_vec = F.normalize(perp_vec, dim=2) #checked
    #after normalization multiply by maximum
    perp_vec = torch.mul(perp_vec.transpose(dim0=-1,dim1=-2), rlsoftmax).transpose(dim0=-1,dim1=-2)
    #return point cloud with "maximum"-normed cross product vec
    cat = torch.cat([ perp_vec, populate], dim=1) #checked
    return cat

def calc_ug(vec_for_embed, edge_index):
    """"
    returns vector with upper gram matrix of distances+norms, with superfluous projection
    """
    #get 3 first rows
    cat = vec_for_embed[:,:3,:] #checked
    #here change to vec without the cross product vec
    cdist = torch.cdist(cat, vec_for_embed, p=2)
    
    norms = torch.linalg.norm(cat, ord=None, axis=2, keepdims=False) #checked
    #make diagonal matrix with norms on diagonal
    norms_diag = torch.diag_embed(norms)
    #add to dist upper gram matrix:
    cdist[:,:,:3] += norms_diag
    return cdist
