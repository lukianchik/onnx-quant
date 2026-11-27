"""Проверяем MatMul и реальную форму Conv из сбоя ResNet на конкретной GPU."""
import os,json,tempfile
from pathlib import Path
import numpy as np


def probe(kind,threads=2):
    import onnx,torch
    from onnx import helper as h,TensorProto as T,numpy_helper as nh
    from benchmark import session
    rng=np.random.default_rng(731)
    if kind=='MatMul':
        x=rng.normal(size=(16,16)).astype('f');w=np.eye(16,dtype='f');nodes=[h.make_node('MatMul',['x','w'],['y'])];shape=x.shape;ref=x
    else:
        x=rng.normal(size=(4,128,28,28)).astype('f');w=rng.normal(0,.02,size=(128,128,3,3)).astype('f')
        # Тестируем также DQ -> Conv, а не только FP32.
        if kind=='DQConv':
            q=np.clip(np.rint(w/.001),-127,127).astype(np.int8);s=np.full(128,.001,np.float32);w=q*s[:,None,None,None]
            nodes=[h.make_node('DequantizeLinear',['q','s'],['w'],axis=0),h.make_node('Conv',['x','w'],['y'],pads=[1,1,1,1])]
        else:nodes=[h.make_node('Conv',['x','w'],['y'],pads=[1,1,1,1])]
        shape=x.shape
        ref=torch.nn.functional.conv2d(torch.from_numpy(x),torch.from_numpy(w),padding=1).numpy()
    inits=[nh.from_array(q,'q'),nh.from_array(s,'s')] if kind=='DQConv' else [nh.from_array(w,'w')]
    p=h.make_model(h.make_graph(nodes,'probe',[h.make_tensor_value_info('x',T.FLOAT,list(x.shape))],[h.make_tensor_value_info('y',T.FLOAT,list(shape))],inits),opset_imports=[h.make_opsetid('',17)],ir_version=8)
    sess=session(p,threads,profiling=True)
    try:
        got=sess.run(None,{'x':x})[0];np.testing.assert_allclose(got,ref,atol=3e-4,rtol=2e-3)
    finally:
        f=sess.end_profiling();events=json.loads(Path(f).read_text());Path(f).unlink(missing_ok=True)
    matching=[e for e in events if e.get('args',{}).get('provider')=='CUDAExecutionProvider' and ('MatMul' in e.get('args',{}).get('op_name','') if kind=='MatMul' else 'Conv' in e.get('args',{}).get('op_name',''))]
    if not matching:raise RuntimeError(f'{kind} did not execute on CUDA')
    return {'max_abs':float(np.max(abs(got-ref))),'cuda_events':len(matching)}


def configure(model,allow_cpu=False):
    import torch,onnxruntime as ort
    os.environ['QUANT_INFER_PROVIDER']='cuda'
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('PyTorch CUDA missing')
    major=str(torch.version.cuda).split('.')[0]
    # Включаем явное имя runtime для динамического загрузчика cuDNN frontend, если библиотека найдена.
    import site
    for base in site.getsitepackages():
        paths=list(Path(base).glob(f'nvidia/**/libcudart.so.{major}'))
        if paths:
            os.environ.setdefault('CUDNN_FRONTEND_CUDART_LIB_NAME',str(paths[0]));break
    cfg=os.environ.get('QUANT_GPU_CONFIG')
    if cfg:
        result=json.loads(cfg)
    else:
        result={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'ort':ort.__version__,'cudnn':torch.backends.cudnn.version(),'matmul':probe('MatMul'),'conv':None,'attempts':[]}
        if model in ('resnet18','mobilenet_v2','squeezenet1_1','cnn','depthwise_cnn','efficientnet_b0','vit_tiny','custom','all'):
            for algo,nhwc in [('HEURISTIC','0'),('DEFAULT','0'),('HEURISTIC','1'),('DEFAULT','1')]:
                os.environ.update(QUANT_CONV_ALGO=algo,QUANT_NHWC=nhwc)
                try:
                    result['conv']={'fp32':probe('Conv'),'dq':probe('DQConv'),'algo':algo,'nhwc':nhwc};break
                except Exception as exc:result['attempts'].append({'algo':algo,'nhwc':nhwc,'error':str(exc)[:500]})
    print('GPU_PROBE',json.dumps(result),flush=True)
    if result.get('conv'):
        os.environ.update(QUANT_CONV_ALGO=result['conv']['algo'],QUANT_NHWC=result['conv']['nhwc'])
    elif model in ('resnet18','mobilenet_v2','squeezenet1_1','cnn','depthwise_cnn','efficientnet_b0','vit_tiny','custom'):
        if not allow_cpu:raise RuntimeError('Conv CUDA probe failed; use --conv-cpu-fallback explicitly or change environment')
        os.environ['QUANT_INFER_PROVIDER']='cpu';print('EXPLICIT FALLBACK: CNN ONNX inference CPU; math remains CUDA',flush=True)
    return result

if __name__=='__main__':
    import subprocess,sys
    result=configure('all')
    # Проверяем исходные и пакетные алгоритмы на этой физической GPU.
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_cuda_math.py','-v'],cwd=Path(__file__).parent,check=True)
