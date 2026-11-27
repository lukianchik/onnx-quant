"""FP64 Torch-бэкенд PTQ. Оригиналы NumPy неизменны; результат INT8/FP32."""
import numpy as np
from algorithms import activation_int8 as ref, spectral_rounding as sr


def _tensor(a, device):
    import torch
    return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.float64)


def _scale(t):
    import torch
    lo=float(np.nextafter(np.float32(0),np.float32(1)))
    hi=np.float32(float(np.finfo(np.float32).max)/127)
    if float(hi)*127>float(np.finfo(np.float32).max):hi=np.nextafter(hi,np.float32(0))
    return t.clamp(lo,float(hi)).float().double()


def _baseline(w):
    import torch
    peak=w.abs().amax(0)
    s=_scale(torch.where(peak==0,1.,peak/127))
    return (w/s).round().clamp(-127,127),s


def _loss(x,w,q,s):
    return (x@(w-(q.float()*s.float()).double())).square().sum(0)


def _result(q,s):
    return q.to(dtype=__import__('torch').int8).cpu().numpy(),s.float().cpu().numpy()


def v1_torch(x,w,paired=True,sweeps=3,device='cuda',chunk=128):
    """Те же координатные шаги, масштаб и защитная проверка; каналы пакетно."""
    import torch
    x0=ref._matrix(x);w0=ref._matrix(w)
    if x0.shape[1]!=w0.shape[0]:raise ValueError('shape')
    pairs=ref._pairs(x0) if paired else []  # исходное детерминированное правило пар
    X=_tensor(x0,device);W=_tensor(w0,device)
    Q,S=_baseline(W);q0=Q.clone();s0=S.clone()
    # Нулевые направления исключаем один раз, а не синхронизируем CUDA в каждом шаге.
    directions=[(i,None,1,X[:,i],float(x0[:,i]@x0[:,i])) for i in range(x0.shape[1])]
    for i,j,sign in pairs:
        a=x0[:,i]+sign*x0[:,j];directions.append((i,j,sign,X[:,i]+sign*X[:,j],float(a@a)))
    for begin in range(0,W.shape[1],chunk):
        sl=slice(begin,begin+chunk);w=W[:,sl];best=Q[:,sl].clone();stored=S[sl].clone();best_loss=_loss(X,w,best,stored)
        # Оригинал останавливается для блоков 32; сохраняем это правило независимо от ширины GPU-пакета.
        alive=torch.ones_like(stored,dtype=torch.bool)
        for _ in range(sweeps):
            scale=stored.clone();trial=best.clone();res=X@(w-trial*scale)
            for i,j,sign,a,energy in directions:
                if not energy:continue
                low=-127-trial[i];high=127-trial[i]
                if j is not None:
                    low=torch.maximum(low,-127-sign*trial[j]);high=torch.minimum(high,127-sign*trial[j])
                step=(torch.minimum(torch.maximum((a@res)/(energy*scale),low),high)).round()
                step=torch.where(alive,step,0.)
                trial[i]+=step
                if j is not None:trial[j]+=sign*step
                res-=a[:,None]*(scale*step)
            res=X@(w-trial*scale);response=X@trial;energy=response.square().sum(0);cross=(response*res).sum(0)
            fitted=_scale(scale+torch.where(energy>0,cross/energy,0.));improved=torch.zeros_like(alive)
            for candidate in (stored.clone(),fitted):
                loss=_loss(X,w,trial,candidate);keep=(loss<best_loss)&alive
                best=torch.where(keep[None,:],trial,best);stored=torch.where(keep,candidate,stored);best_loss=torch.where(keep,loss,best_loss);improved|=keep
            # Остановка блока исходной реализации, не отдельных каналов.
            for i in range(0,len(alive),32):alive[i:i+32]&=improved[i:i+32].any()
            if not bool(alive.any()):break
        Q[:,sl]=best;S[sl]=stored
    keep=_loss(X,W,Q,S)<=_loss(X,W,q0,s0)
    return _result(torch.where(keep[None,:],Q,q0),torch.where(keep,S,s0))


def _round_batched(start,lo,hi,gram,grad,coupled,tol):
    import torch
    v=start.clone();b=v.shape[1];eps=torch.finfo(v.dtype).eps
    for _ in range(b):
        active=(lo<v)&(v<hi);any_active=active.any(1)
        first=active.long().argmax(1);unit=torch.nn.functional.one_hot(first,b).to(v.dtype)*any_active[:,None]
        u=unit
        if coupled:
            C=gram[None,:,:]*(active[:,:,None]&active[:,None,:]) + torch.diag_embed((~active).to(v.dtype)*(b+2))
            vals,vecs=torch.linalg.eigh(C);eig=vecs[:,:,0]*active
            eig=torch.where(eig.abs()<16*eps,0.,eig)
            sign=eig.gather(1,eig.abs().argmax(1,keepdim=True));eig=torch.where(sign<0,-eig,eig)
            curvature=(eig*(eig@gram-eig)).sum(1)
            use=(vals[:,0]<1-tol)&(active.sum(1)>1)&(curvature<=0)
            u=torch.where(use[:,None],eig,unit)
        lin=2*grad+(hi-v)+(lo-v)
        def endpoint(direction):
            edge=torch.where(direction>0,hi,lo);step=torch.where(direction!=0,(edge-v)/direction,torch.inf)
            a,hit=step.min(1);a=torch.where(any_active,a,0.)
            point=torch.minimum(torch.maximum(v+a[:,None]*direction,lo),hi)
            point.scatter_(1,hit[:,None],edge.gather(1,hit[:,None]));point=torch.where(any_active[:,None],point,v)
            delta=point-v;cost=(lin*delta+delta*(delta@gram-delta)).sum(1)
            return point,cost
        p,c=endpoint(u);q,d=endpoint(-u);new=torch.where((d<c)[:,None],q,p)
        grad=grad+(new-v)@gram;v=new
    if bool(((lo<v)&(v<hi)).any()):raise ArithmeticError('SDR unfinished rounding')
    return v


def sdr_torch(x,w,config=None,device='cuda',chunk=128):
    """Пакетный полёт SDR по активным граням, FP64 и исходный MinMax fallback."""
    import torch
    cfg=sr.Config() if config is None else config
    q0,s0=sr._minmax(w);plan=sr._plan(len(x),*w.shape,cfg)
    if plan is None:return (q0,s0),{'reason':'budget_or_width','accepted':0}
    b,work,memory=plan;X=_tensor(x,device);W=_tensor(w,device);Q=_tensor(q0,device);S=_tensor(s0,device)
    blocks=[];ranks=[];norm=torch.linalg.vector_norm(X,dim=0)
    for i in range(0,W.shape[0],b):
        sl=slice(i,i+b);part=X[:,sl];h=norm[sl];A=part/torch.where(h>0,h,1.);G=A.T@A;G.fill_diagonal_(1.)
        tol=64*np.finfo(float).eps*max(len(x),len(h));r=int((torch.linalg.eigvalsh(G)>1+tol).sum());ranks.append(r);blocks.append((sl,A,G,h,r,tol))
    info={'block':b,'work_estimate':work,'workspace_estimate':memory,'protected_ranks':ranks,'accepted':0}
    if not any(ranks):return (q0,s0),{**info,'reason':'flat_blocks'}
    for begin in range(0,W.shape[1],chunk):
        slc=slice(begin,begin+chunk);weight=W[:,slc];s=S[slc];base=Q[:,slc].clone()
        low=(weight/s).floor().clamp(-127,127);high=(low+1).clamp(max=127)
        lo=(low.float()*s.float()).double();hi=(high.float()*s.float()).double();center=torch.minimum(torch.maximum(weight,lo),hi)
        residual=X@(center-weight);candidate=base.clone()
        for sl,A,G,h,r,tol in blocks:
            a=((lo[sl]-weight[sl])*h[:,None]).T;z=((hi[sl]-weight[sl])*h[:,None]).T
            start=torch.minimum(torch.maximum(torch.zeros_like(a),a),z)
            end=_round_batched(start,a,z,G,(A.T@residual).T,r,tol).T
            candidate[sl]=torch.where(h[:,None]==0,base[sl],torch.where(end==z.T,high[sl],low[sl]))
            decoded=(candidate[sl].float()*s.float()).double();residual+=X[:,sl]@(decoded-center[sl])
        old=_loss(X,weight,base,s);new=_loss(X,weight,candidate,s);keep=torch.isfinite(new)&(new<old*(1-64*np.finfo(float).eps))
        Q[:,slc]=torch.where(keep[None,:],candidate,base);info['accepted']+=int(keep.sum())
    return _result(Q,S),{**info,'reason':'accepted' if info['accepted'] else 'no_gain'}
