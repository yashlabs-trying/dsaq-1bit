import torch

def _features(x,normalize=False,extra=None):
    x=x.float()
    if normalize:x=x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)
    if extra is not None:x=torch.cat([x,extra.float()],1)
    return x

def kmeans(x,k,normalize=False,extra=None,iters=12,seed=20260922):
    """Deterministic full-feature Lloyd k-means; normalization gives spherical/cosine clusters."""
    x=_features(x,normalize,extra);n=len(x);k=min(k,n)
    g=torch.Generator(device=x.device).manual_seed(seed)
    # Deterministic k-means++ initialization.
    ids=[int(torch.randint(n,(1,),generator=g,device=x.device))]
    dist=((x-x[ids[0]])**2).sum(1)
    for _ in range(1,k):
        nxt=int(torch.multinomial(dist.clamp_min(1e-20),1,generator=g));ids.append(nxt)
        dist=torch.minimum(dist,((x-x[nxt])**2).sum(1))
    cent=x[torch.tensor(ids,device=x.device)].clone()
    labels=torch.zeros(n,dtype=torch.long,device=x.device)
    for _ in range(iters):
        # squared distance without allocating n*k*d
        new=(-2*x@cent.T+x.square().sum(1,keepdim=True)+cent.square().sum(1)[None]).argmin(1)
        if torch.equal(new,labels):break
        labels=new
        sums=torch.zeros_like(cent).index_add_(0,labels,x)
        counts=torch.bincount(labels,minlength=k).clamp_min(1)[:,None]
        updated=sums/counts
        empty=(counts[:,0]==1)&(torch.bincount(labels,minlength=k)==0)
        if empty.any():updated[empty]=x[torch.randperm(n,generator=g,device=x.device)[:int(empty.sum())]]
        cent=updated
    return labels,cent,x

def ordered_permutation(v,k,normalize=False,within='centroid',extra=None,seed=20260922):
    labels,cent,x=kmeans(v,k,normalize,extra,seed=seed)
    pieces=[]
    for cluster in range(cent.shape[0]):
        ids=(labels==cluster).nonzero().flatten()
        if len(ids)==0:continue
        z=v[ids].float()
        if within=='centroid':score=((x[ids]-cent[cluster])**2).sum(1)
        elif within=='norm':score=z.norm(dim=1)
        elif within=='pca':
            q=z-z.mean(0);direction=torch.pca_lowrank(q,q=1,center=False,niter=2)[2][:,0];score=q@direction
        else:raise ValueError(within)
        pieces.append(ids[score.argsort()])
    p=torch.cat(pieces);assert torch.equal(p.sort().values,torch.arange(len(v),device=v.device))
    return p

def local_permutation(v,k,macro,normalize=True,within='pca',extra=None,seed=20260922):
    pieces=[]
    for start in range(0,len(v),macro):
        stop=min(len(v),start+macro);e=extra[start:stop] if extra is not None else None
        pieces.append(ordered_permutation(v[start:stop],min(k,stop-start),normalize,within,e,seed+start)+start)
    return torch.cat(pieces)

def magnitude_features(v,scale=1.0):
    z=v.float();stats=torch.stack([z.norm(dim=1),z.abs().mean(1),z.var(1),z.abs().amax(1),(z.abs()>3*z.std(1,keepdim=True)).float().sum(1)],1)
    stats=(stats-stats.mean(0))/stats.std(0).clamp_min(1e-8)
    return scale*stats
