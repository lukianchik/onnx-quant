"""Новые предобученные модели; CIFAR-10 для fidelity (без ложных ImageNet labels), AG News для embeddings."""
from pathlib import Path
import gc
import numpy as np
from models import Case
from pretrained import indices,export_model,check_export,sha
VISION_NEW=('efficientnet_b0','vit_tiny')
NLP_NEW=('minilm_embeddings',)

def build(name,seed,cfg,root):
    import torch,onnx
    torch.set_num_threads(cfg.threads)
    cache=Path(root)/'pretrained_cache';cache.mkdir(parents=True,exist_ok=True);path=cache/(name+'.onnx')
    if name in VISION_NEW:
        from torchvision.datasets import CIFAR10
        if name=='efficientnet_b0':
            from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1;model=efficientnet_b0(weights=weights).eval();transform=weights.transforms();repo=weights.url
        else:
            import timm
            repo='vit_tiny_patch16_224.augreg_in21k_ft_in1k';model=timm.create_model(repo,pretrained=True).eval()
            for mod in model.modules():
                if hasattr(mod,'fused_attn'):mod.fused_attn=False
            transform=timm.data.create_transform(**timm.data.resolve_model_data_config(model),is_training=False)
        names=['input'];args=(torch.randn(1,3,224,224,generator=torch.Generator().manual_seed(731)),)
        p=onnx.load(path) if path.exists() else export_model(model,args,path,names,{'input':{0:'batch'},'logits':{0:'batch'}})
        diff=check_export(model,args,p,names,cfg);del model;gc.collect()
        tr=CIFAR10(str(cache),train=True,download=True,transform=transform);te=CIFAR10(str(cache),train=False,download=True,transform=transform)
        ti=indices(len(tr),cfg.fit+cfg.dev,seed);vi=indices(len(te),cfg.test,731)
        def fetch(ds,ids):return np.stack([ds[int(i)][0].numpy() for i in ids]).astype(np.float32)
        x=fetch(tr,ti);y=fetch(te,vi);shift=y*.8+.15
        case=Case(name,p,{'input':x[:cfg.fit]},{'input':x[cfg.fit:]},{'input':y},'classification',None,{'input':shift},None,'ImageNet checkpoint, CIFAR-10 images; only FP32 fidelity and prediction agreement, not CIFAR accuracy.')
        meta=dict(checkpoint=repo,dataset='torchvision CIFAR10',fit_ids=ti[:cfg.fit].tolist(),dev_ids=ti[cfg.fit:].tolist(),test_ids=vi.tolist(),shift='affine normalized pixels')
    elif name in NLP_NEW:
        from transformers import AutoModel,AutoTokenizer
        from datasets import load_dataset
        repo='sentence-transformers/all-MiniLM-L6-v2';tok=AutoTokenizer.from_pretrained(repo);base=AutoModel.from_pretrained(repo,attn_implementation='eager').eval()
        rev=getattr(base.config,'_commit_hash',None)
        class Wrapper(torch.nn.Module):
            def __init__(self,model):super().__init__();self.model=model
            def forward(self,input_ids,attention_mask):
                h=self.model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
                mask=attention_mask.unsqueeze(-1).to(h.dtype);z=(h*mask).sum(1)/mask.sum(1).clamp_min(1.)
                return torch.nn.functional.normalize(z,p=2,dim=1)
        model=Wrapper(base).eval();names=['input_ids','attention_mask'];args=(torch.ones((1,128),dtype=torch.long),torch.ones((1,128),dtype=torch.long))
        p=onnx.load(path) if path.exists() else export_model(model,args,path,names,{k:{0:'batch'} for k in names+['logits']})
        diff=check_export(model,args,p,names,cfg);del model,base;gc.collect()
        ds=load_dataset('fancyzhx/ag_news');ti=indices(len(ds['train']),cfg.fit+cfg.dev,seed);vi=indices(len(ds['test']),cfg.test,731)
        def enc(split,ids):
            e=tok(list(ds[split].select(ids.tolist())['text']),padding='max_length',truncation=True,max_length=128,return_tensors='np');return {k:e[k].astype(np.int64) for k in names}
        tr=enc('train',ti);te=enc('test',vi);shift={k:v.copy() for k,v in te.items()};shift['input_ids'][:,64:]=tok.pad_token_id;shift['attention_mask'][:,64:]=0
        case=Case(name,p,{k:v[:cfg.fit] for k,v in tr.items()},{k:v[cfg.fit:] for k,v in tr.items()},te,'generic',None,shift,None,'Normalized mean-pooled MiniLM embeddings on AG News; no task accuracy claimed.')
        meta=dict(checkpoint=repo,checkpoint_revision=rev,dataset='fancyzhx/ag_news',fit_ids=ti[:cfg.fit].tolist(),dev_ids=ti[cfg.fit:].tolist(),test_ids=vi.tolist(),shift='truncate to 64 tokens')
    else:raise ValueError(name)
    meta.update(model_sha256=sha(path),export_max_abs=diff);case.provenance=meta;return case


def lstm_case(seed,cfg,root):
    """Реальная обученная LSTM; recurrent W/R остаются FP32, проверяем границы адаптеров."""
    import torch,onnx
    from sklearn.datasets import load_digits
    from models import split_indices
    torch.set_num_threads(cfg.threads);torch.manual_seed(731)
    data=load_digits();x=(data.data/16).astype(np.float32).reshape(-1,8,8);y=data.target.astype(np.int64)
    tr,fi,de,te=split_indices(len(x),seed,cfg.fit,cfg.dev,cfg.test)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rnn=torch.nn.LSTM(8,32,batch_first=True);self.fc=torch.nn.Linear(32,10)
        def forward(self,x):
            h=x.new_zeros((1,x.shape[0],32));c=x.new_zeros((1,x.shape[0],32))
            z,_=self.rnn(x,(h,c));return self.fc(z[:,-1])
    net=Net();opt=torch.optim.Adam(net.parameters(),lr=.01);xt=torch.from_numpy(x[tr]);yt=torch.from_numpy(y[tr]);gen=torch.Generator().manual_seed(731)
    for _ in range(20):
        for idx in torch.randperm(len(xt),generator=gen).split(64):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(xt[idx]),yt[idx]);loss.backward();opt.step()
    net.eval();cache=Path(root)/'pretrained_cache';cache.mkdir(exist_ok=True,parents=True);path=cache/'lstm_head.onnx';args=(torch.from_numpy(x[:1]),)
    p=onnx.load(path) if path.exists() else export_model(net,args,path,['input'],{'input':{0:'batch'},'logits':{0:'batch'}})
    diff=check_export(net,args,p,['input'],cfg);shift=np.clip(x[te]+np.random.default_rng(731).normal(0,.18,x[te].shape).astype(np.float32),0,1)
    case=Case('lstm_head',p,{'input':x[fi]},{'input':x[de]},{'input':x[te]},'classification',y[te],{'input':shift},y[te],'Digits trained LSTM; fused LSTM operator is unsupported, only supported output weights quantized.')
    case.provenance=dict(train_ids=tr.tolist(),fit_ids=fi.tolist(),dev_ids=de.tolist(),test_ids=te.tolist(),model_sha256=sha(path),export_max_abs=diff)
    return case
