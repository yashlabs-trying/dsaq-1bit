"""Reference compressed linear execution; deliberately not a fast CUDA kernel.

Packed weights/corrections stay on GPU. Reconstruct one linear weight matrix
temporarily per call, never a persistent full dense transformer checkpoint.
"""
import torch
from model_experiment import unpack


class CompressedLinear(torch.nn.Module):
    def __init__(self, archive, bias=None, device='cuda'):
        super().__init__()
        self.shape=tuple(archive['shape'])
        for prefix in ('q2','pca','sparse'):
            for key,value in archive[prefix].items():
                self.register_buffer(prefix+'_'+key,value.to(device))
        self.register_buffer('bias',None if bias is None else bias.detach().clone().to(device))

    def reconstructed_weight(self,dtype):
        base=unpack({k:getattr(self,'q2_'+k) for k in ('packed','scale','offset')},2,self.q2_packed.device)
        base += self.pca_codes.float() @ self.pca_basis.float() + self.pca_mean.float()
        base.flatten()[self.sparse_index.long()] += self.sparse_value.float()
        return base.reshape(self.shape).to(dtype)

    def forward(self,x):
        return torch.nn.functional.linear(x,self.reconstructed_weight(x.dtype),self.bias)
