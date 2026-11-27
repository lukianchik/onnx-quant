"""Реальные предобученные модели; выборки фиксируются независимо от методов."""
from pathlib import Path
import io, tarfile, urllib.request, json, hashlib, gc
import numpy as np
from models import Case

VISION=('resnet18','mobilenet_v2','squeezenet1_1')
NLP=('distilbert_sst2',)
IMAGENETTE={'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,'n03445777':574,'n03888257':701}
URL='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

def indices(n, k, seed):
    return np.random.default_rng(seed).permutation(n)[:min(k,n)]

def download(path):
    if not path.exists():
        tmp=path.with_suffix('.partial')
        print('Download:',URL,flush=True)
        urllib.request.urlretrieve(URL,tmp);tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def vision_data(cache, transform, cfg, seed):
    from PIL import Image
    path=cache/'imagenette2-320.tgz';download(path)
    # Только чтение выбранных файлов, без распаковки путей из архива.
    with tarfile.open(path,'r:gz') as tar:
        items={'train':[],'val':[]}
        for m in tar.getmembers():
            p=Path(m.name).parts
            if m.isfile() and len(p)==4 and p[1] in items and p[2] in IMAGENETTE and p[-1].lower().endswith(('.jpeg','.jpg','.png')):items[p[1]].append(m)
        for a in items.values():a.sort(key=lambda m:m.name)
        tr=indices(len(items['train']),cfg.fit+cfg.dev,seed);te=indices(len(items['val']),cfg.test,731)
        if len(tr)!=cfg.fit+cfg.dev or len(te)!=cfg.test:raise ValueError('Not enough Imagenette examples')
        chosen=[items['train'][i] for i in tr]+[items['val'][i] for i in te]
        x=np.empty((len(chosen),3,224,224),np.float32);y=np.empty(len(chosen),np.int64)
        # Читаем в порядке смещений архива, не перематывая gzip для каждого изображения.
        for j in sorted(range(len(chosen)),key=lambda j:chosen[j].offset_data):
            m=chosen[j]
            with tar.extractfile(m) as f:raw=f.read()
            with Image.open(io.BytesIO(raw)) as img:x[j]=transform(img.convert('RGB')).numpy()
            y[j]=IMAGENETTE[Path(m.name).parts[2]]
    a,b=cfg.fit,cfg.fit+cfg.dev
    test=x[b:]; shift=test.copy();shift*=np.float32(.8);shift+=np.float32(.15)
    return {'input':x[:a]},{'input':x[a:b]},{'input':test},y[b:],{'input':shift},y[b:],{'fit_ids':[m.name for m in chosen[:a]],'dev_ids':[m.name for m in chosen[a:b]],'test_ids':[m.name for m in chosen[b:]],'dataset':URL,'dataset_sha256':sha(path)}

def export_model(model, args, path, names, axes):
    import torch,onnx
    tmp=path.with_suffix('.tmp.onnx')
    with torch.no_grad():
        torch.onnx.export(model.eval(),args,str(tmp),input_names=names,output_names=['logits'],dynamic_axes=axes,opset_version=17,dynamo=False,do_constant_folding=True)
    p=onnx.load(tmp);onnx.checker.check_model(p);tmp.replace(path)
    return p

def check_export(model, args, p, names, cfg):
    import torch
    from benchmark import session
    with torch.no_grad():ref=model(*args).detach().cpu().numpy()
    s=session(p,cfg.threads);out=s.run(None,{k:v.cpu().numpy() for k,v in zip(names,args)})[0]
    np.testing.assert_allclose(out,ref,rtol=2e-3,atol=5e-4)
    return float(np.max(abs(out-ref)))

def make_pretrained(name,seed,cfg,root):
    import torch,onnx
    torch.set_num_threads(cfg.threads)
    cache=Path(root)/'pretrained_cache';cache.mkdir(parents=True,exist_ok=True)
    path=cache/(name+'.onnx')
    if name in VISION:
        import torchvision.models as tv
        factory,weights={'resnet18':(tv.resnet18,tv.ResNet18_Weights.IMAGENET1K_V1),'mobilenet_v2':(tv.mobilenet_v2,tv.MobileNet_V2_Weights.IMAGENET1K_V1),'squeezenet1_1':(tv.squeezenet1_1,tv.SqueezeNet1_1_Weights.IMAGENET1K_V1)}[name]
        # Явно фиксируем checkpoint, не DEFAULT.
        model=factory(weights=weights).eval(); args=(torch.randn(1,3,224,224,generator=torch.Generator().manual_seed(731)),);names=['input']
        p=onnx.load(path) if path.exists() else export_model(model,args,path,names,{'input':{0:'batch'},'logits':{0:'batch'}})
        diff=check_export(model,args,p,names,cfg);del model;gc.collect()
        fit,dev,test,labels,shift,sy,meta=vision_data(cache,weights.transforms(),cfg,seed)
        meta.update(checkpoint=weights.url,export_max_abs=diff,model_sha256=sha(path),shift='affine change of normalized pixels; labels unchanged')
        case=Case(name,p,fit,dev,test,'classification',labels,shift,sy,'ImageNet pretrained; Imagenette train/val separated for PTQ, not necessarily unseen during ImageNet pretraining; 1000-class output, original class IDs.')
    elif name in NLP:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        from datasets import load_dataset
        repo='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        tok=AutoTokenizer.from_pretrained(repo);base=AutoModelForSequenceClassification.from_pretrained(repo,attn_implementation='eager').eval()
        class Wrapper(torch.nn.Module):
            def __init__(self,base):super().__init__();self.base=base
            def forward(self,input_ids,attention_mask):return self.base(input_ids=input_ids,attention_mask=attention_mask).logits
        model=Wrapper(base).eval();names=['input_ids','attention_mask'];args=(torch.ones((1,128),dtype=torch.long),torch.ones((1,128),dtype=torch.long))
        p=onnx.load(path) if path.exists() else export_model(model,args,path,names,{k:{0:'batch'} for k in names+['logits']})
        diff=check_export(model,args,p,names,cfg);rev=getattr(base.config,'_commit_hash',None);del model,base;gc.collect()
        ds=load_dataset('stanfordnlp/imdb');tr=indices(len(ds['train']),cfg.fit+cfg.dev,seed);te=indices(len(ds['test']),cfg.test,731)
        train=ds['train'].select(tr.tolist());test=ds['test'].select(te.tolist())
        def enc(texts,max_len=128):
            e=tok(list(texts),padding='max_length',truncation=True,max_length=max_len,return_tensors='np');return {k:e[k].astype(np.int64) for k in names}
        fit=enc(train[:cfg.fit]['text']);dev=enc(train[cfg.fit:]['text']);testfeeds=enc(test['text']);shift=enc(test['text']);shift['attention_mask'][:,64:]=0;shift['input_ids'][:,64:]=tok.pad_token_id
        labels=np.asarray(test['label'],np.int64)
        meta={'checkpoint':repo,'checkpoint_revision':rev,'model_sha256':sha(path),'export_max_abs':diff,'dataset':'stanfordnlp/imdb','dataset_fingerprints':{k:v._fingerprint for k,v in ds.items()},'fit_ids':tr[:cfg.fit].tolist(),'dev_ids':tr[cfg.fit:].tolist(),'test_ids':te.tolist(),'shift':'truncate to 64 tokens; fixed shape 128'}
        case=Case(name,p,fit,dev,testfeeds,'classification',labels,shift,labels,'SST-2 pretrained checkpoint; IMDb train for PTQ, official IMDb test; cross-domain sentiment.')
    else:raise ValueError(name)
    case.provenance=meta;return case
