"""Небольшие воспроизводимые модели и раздельные наборы данных; без загрузок из сети."""
from dataclasses import dataclass
from pathlib import Path
import json,zipfile
import numpy as np

@dataclass
class Case:
    name: str
    model: object
    fit: dict
    dev: dict
    test: dict
    task: str
    labels: np.ndarray | None
    shift: dict | None = None
    shift_labels: np.ndarray | None = None
    note: str = ''
    historical: bool = False

class Builder:
    def __init__(self):
        self.nodes=[]; self.weights=[]
    def const(self,name,a):
        from onnx import numpy_helper
        self.weights.append(numpy_helper.from_array(np.ascontiguousarray(a),name)); return name
    def node(self,op,ins,out,**attrs):
        from onnx import helper
        self.nodes.append(helper.make_node(op,ins,[out],name=out,**attrs)); return out
    def dense(self,x,w,b,out,trans=False):
        return self.node('Gemm',[x,self.const(out+'_W',np.asarray(w.T if trans else w,np.float32)),self.const(out+'_b',np.asarray(b,np.float32))],out,transB=int(trans))
    def model(self,name,inputs,outputs):
        import onnx
        from onnx import helper,TensorProto
        make=lambda item:helper.make_tensor_value_info(item[0],TensorProto.FLOAT,item[1])
        g=helper.make_graph(self.nodes,name,[make(i) for i in inputs],[make(i) for i in outputs],self.weights)
        p=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)],producer_name='quant-benchmark',ir_version=8)
        onnx.checker.check_model(p); return p


def split_indices(n,seed,fit_count,dev_count,test_count):
    # Test фиксирован между seed, обучение не пересекается с calibration/dev/test.
    perm=np.random.default_rng(731).permutation(n)
    train=perm[:int(.6*n)]; pool=perm[int(.6*n):int(.8*n)]; test=perm[int(.8*n):]
    pool=np.random.default_rng(seed).permutation(pool)
    f=min(fit_count,max(1,len(pool)*2//3)); d=min(dev_count,len(pool)-f)
    return train,pool[:f],pool[f:f+d],test[:test_count]


def ridge(x,y,lam=1e-2):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    a=x-x.mean(0); b=y-y.mean(0); w=np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@b)
    return w.astype(np.float32),(y.mean(0)-x.mean(0)@w).astype(np.float32)


def conv_numpy(x,w,groups=1):
    # Только для построения небольших демонстрационных моделей 3x3, stride 1, pad 1.
    k=w.shape[-1]; p=k//2
    padded=np.pad(x,((0,0),(0,0),(p,p),(p,p)))
    patches=np.lib.stride_tricks.sliding_window_view(padded,(k,k),axis=(2,3))
    parts=[]
    for a,b in zip(np.split(patches,groups,axis=1),np.split(w,groups,axis=0)):
        parts.append(np.einsum('nchwij,ocij->nohw',a,b,optimize=True))
    return np.concatenate(parts,axis=1).astype(np.float32)


def make_case(name,seed,cfg,root,archive=None):
    from sklearn.datasets import load_digits,load_diabetes
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    if name=='bert_tiny': return bert_case(seed,cfg,archive)
    rng=np.random.default_rng(1207); b=Builder()
    if name=='regression':
        xx,yy=load_diabetes(return_X_y=True); x=xx.astype(np.float32); y=yy.astype(np.float32)[:,None]
        tr,fit,dev,test=split_indices(len(x),seed,cfg.fit,cfg.dev,cfg.test)
        mean=x[tr].mean(0); sd=x[tr].std(0); x=(x-mean)/np.maximum(sd,1e-6)
        w1=rng.normal(0,.35,(x.shape[1],32)).astype(np.float32); b1=rng.normal(0,.1,32).astype(np.float32)
        h=np.tanh(x@w1+b1); w2,b2=ridge(h[tr],y[tr],.2)
        b.dense('x',w1,b1,'h',trans=True);b.node('Tanh',['h'],'a');b.dense('a',w2,b2,'prediction')
        model=b.model(name,[('x',[None,x.shape[1]])],[('prediction',[None,1])]); task='regression'; label=y
        shift=x[test]+np.float32(.35)*rng.normal(size=x[test].shape).astype(np.float32)
        # Сдвиг входов без известных истинных меток; оцениваем только fidelity.
        return Case(name,model,{'x':x[fit]},{'x':x[dev]},{'x':x[test]},task,label[test],{'x':shift},None,'Diabetes; фиксированные случайные признаки, обученный ridge-выход.')
    data=load_digits(); x=(data.data/16).astype(np.float32); y=data.target.astype(np.int64)
    tr,fit,dev,test=split_indices(len(x),seed,cfg.fit,cfg.dev,cfg.test)
    task='classification'; label=y; out=[('logits',[None,10])]; shape=[None,64]; note='Digits; независимые train/calibration/dev/test.'
    if name=='mlp':
        model_fit=MLPClassifier(hidden_layer_sizes=(32,),max_iter=200,random_state=731,early_stopping=True).fit(x[tr],y[tr])
        b.dense('x',model_fit.coefs_[0],model_fit.intercepts_[0],'h',trans=True);b.node('Relu',['h'],'a');b.dense('a',model_fit.coefs_[1],model_fit.intercepts_[1],'logits')
    elif name in ('cnn','depthwise_cnn'):
        x=x.reshape(-1,1,8,8); shape=[None,1,8,8]; f=rng.normal(0,.35,(4,1,3,3)).astype(np.float32)
        z=b.node('Conv',['x',b.const('conv0_W',f)],'conv0',pads=[1]*4); z=b.node('Relu',[z],'relu0')
        h=np.maximum(conv_numpy(x[tr],f),0)
        if name=='depthwise_cnn':
            f2=rng.normal(0,.35,(4,1,3,3)).astype(np.float32);f3=rng.normal(0,.35,(6,2,1,1)).astype(np.float32)
            z=b.node('Conv',[z,b.const('depth_W',f2)],'depth',group=4,auto_pad='SAME_UPPER');z=b.node('Relu',[z],'relu1');h=np.maximum(conv_numpy(h,f2,4),0)
            z=b.node('Conv',[z,b.const('group_W',f3)],'group',group=2);z=b.node('Relu',[z],'relu2');h=np.maximum(conv_numpy(h,f3,2),0)
        z=b.node('AveragePool',[z],'pool',kernel_shape=[2,2],strides=[2,2]);b.node('Flatten',[z],'features',axis=1)
        feat=h.reshape(len(h),h.shape[1],4,2,4,2).mean((3,5)).reshape(len(h),-1)
        w,bb=ridge(feat,np.eye(10)[y[tr]],.1);b.dense('features',w,bb,'logits',trans=True)
        note+=' Фиксированные случайные свёртки, обученная ridge-голова; не pretrained CNN.'
    elif name=='autoencoder':
        p=PCA(n_components=16,random_state=731).fit(x[tr]); w=p.components_.T.astype(np.float32); mu=p.mean_.astype(np.float32)
        b.dense('x',w,-mu@w,'latent');b.dense('latent',w.T,mu,'reconstruction',trans=True);out=[('reconstruction',[None,64])];task='reconstruction';label=x
        note+=' PCA-автоэнкодер, обученный на train.'
    elif name=='multi_output':
        w=rng.normal(0,.4,(64,24)).astype(np.float32); bb=np.zeros(24,np.float32); h=np.maximum(x@w,0)
        b.dense('x',w,bb,'h');b.node('Relu',['h'],'a')
        w1,b1=ridge(h[tr],np.eye(10)[y[tr]],.1);w2,b2=ridge(h[tr],x[tr],.1)
        b.dense('a',w1,b1,'logits');b.dense('a',w2,b2,'reconstruction');out=[('logits',[None,10]),('reconstruction',[None,64])];task='multi';label=None
        note+=' Две головы: классификация и реконструкция; V2 пропускается.'
    elif name=='rnn':
        # Явная развёртка рекуррентной ячейки на 8 шагов, общие W/R не дублируются.
        x=x.reshape(-1,8,8);shape=[None,8,8]; w=rng.normal(0,.35,(8,16)).astype(np.float32);r=rng.normal(0,.15,(16,16)).astype(np.float32)
        b.const('W_shared',w);b.const('R_shared',r)
        hidden=np.zeros((len(tr),16),np.float32); prev=None
        for i in range(8):
            idx=b.const(f'idx{i}',np.array([i],np.int64));g=b.node('Gather',['x',idx],f'g{i}',axis=1)
            g=b.node('Flatten',[g],f'f{i}',axis=1);a=b.node('MatMul',[g,'W_shared'],f'u{i}')
            if prev is not None:
                rpart=b.node('MatMul',[prev,'R_shared'],f'r{i}');a=b.node('Add',[a,rpart],f'add{i}')
            prev=b.node('Tanh',[a],f'h{i}');hidden=np.tanh(x[tr,i]@w+hidden@r)
        w2,b2=ridge(hidden,np.eye(10)[y[tr]],.1);b.dense(prev,w2,b2,'logits');note+=' Развёрнутый RNN; общие W/R намеренно пропускаются, квантуется выходная голова.'
    else: raise ValueError(name)
    model=b.model(name,[('x',shape)],out)
    shift=np.clip(x[test]+rng.normal(0,.18,x[test].shape).astype(np.float32),0,1)
    sl=label[test] if label is not None else None
    if task=='reconstruction': shift_label=shift
    else: shift_label=sl
    return Case(name,model,{'x':x[fit]},{'x':x[dev]},{'x':x[test]},task,sl,{'x':shift},shift_label,note)


def bert_case(seed,cfg,archive):
    import onnx
    from tokenizers import Tokenizer
    if archive is None: raise FileNotFoundError('Поместите onnx-quant-lab-handoff-v0.5.zip рядом со стендом или передайте --archive.')
    path=Path(archive); prefix='onnx-quant-lab/demo-assets/'
    if path.is_dir():
        d=path/'demo-assets' if (path/'demo-assets').exists() else path
        read=lambda n:(d/n).read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            blobs={n:z.read(prefix+n) for n in ['bert_tiny_fp32.onnx','calibration_samples.json','evaluation_samples.json','tokenizer/tokenizer.json']}
        read=blobs.__getitem__
    model=onnx.load_model_from_string(read('bert_tiny_fp32.onnx')); tok=Tokenizer.from_str(read('tokenizer/tokenizer.json').decode())
    tok.enable_truncation(max_length=64);tok.enable_padding(length=64,pad_id=tok.token_to_id('[PAD]') or 0,pad_token='[PAD]')
    cal=json.loads(read('calibration_samples.json')); eva=json.loads(read('evaluation_samples.json'))
    perm=np.random.default_rng(seed).permutation(len(cal)); f=min(cfg.fit,192);d=min(cfg.dev,len(cal)-f)
    inputs={i.name for i in model.graph.input}
    def encode(rows):
        enc=tok.encode_batch([r['text'] for r in rows]); allfeeds={'input_ids':np.array([e.ids for e in enc],np.int64),'attention_mask':np.array([e.attention_mask for e in enc],np.int64),'token_type_ids':np.array([e.type_ids for e in enc],np.int64)}
        return {k:v for k,v in allfeeds.items() if k in inputs}
    # Фиксированная подвыборка исторического test между seed.
    chosen=[eva[i] for i in np.random.default_rng(731).permutation(len(eva))[:cfg.test]]
    return Case('bert_tiny',model,encode([cal[i] for i in perm[:f]]),encode([cal[i] for i in perm[f:f+d]]),encode(chosen),'classification',np.array([r['label'] for r in chosen]),note='Архив SST-2, исторический test повторно использован; не независимое подтверждение.',historical=True)


def custom_case(model_path,fit_path,dev_path,test_path,task='generic',shift_path=None):
    import onnx
    def load(path):
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files if k!='__labels__'},z['__labels__'] if '__labels__' in z.files else None
    a,_=load(fit_path);b,_=load(dev_path);c,y=load(test_path);s,sy=load(shift_path) if shift_path else (None,None)
    return Case(Path(model_path).stem,onnx.load(model_path),a,b,c,task,y,s,sy,'Пользовательская ONNX; непересечение выборок обеспечивается пользователем.')
