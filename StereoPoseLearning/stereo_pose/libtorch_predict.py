"""C++ LibTorch inference, CUDA preprocessing and an optional TensorRT frontend."""
import importlib
import json
import math
from pathlib import Path
import sys
import warnings
import numpy as np
import torch
from .predict import StereoPredictor, _display_api
from .data import resize_intrinsics


class LibTorchModel(torch.nn.Module):
    def __init__(self, state, config, visual_engine=None, execution_mode="reference", cuda_fusions=()):
        super().__init__()
        if config.get('backbone') != 'convnextv2_nano.fcmae_ft_in22k_in1k' or config.get('c_translation_head') != 'landmark_ray_depth_v1':
            raise ValueError('This C++ definition implements the current ConvNeXtV2 Nano / unified-C model')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native/libtorch/build'))
        if visual_engine is not None:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native/libtorch/build/tensorrt'))
        if execution_mode != 'reference':
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native/libtorch/build/optimized'))
        self.native_module='stereo_libtorch_opt' if execution_mode != 'reference' else ('stereo_libtorch_trt' if visual_engine is not None else 'stereo_libtorch')
        if cuda_fusions:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native/libtorch/build/fused'))
            self.native_module='stereo_libtorch_fused'
        native=importlib.import_module(self.native_module)
        self.execution_mode=execution_mode
        self.actual_execution=execution_mode
        self._graph=None
        self.graph_error=None
        numeric={k:float(v) for k,v in config.items() if isinstance(v,(int,float,bool))}
        # Transfer the original state_dict tensors directly once at startup.
        # The default stays eager. An explicitly selected TensorRT engine only
        # replaces the visual frontend in the separately compiled native module.
        numeric.update({'fusion_'+name:1.0 for name in cuda_fusions})
        self.engine=native.StereoNet(dict(state),numeric)
        if visual_engine is not None:self.engine.load_visual_engine(str(visual_engine))

    def forward(self, images, K, D, hidden=None, hires_images=None):
        if hidden is None: hidden=torch.zeros((3,2,256),device=images.device,dtype=torch.float32)
        if self.execution_mode!='optimized_graph' or self.graph_error is not None:
            values,state,_=self.engine.forward(images,K,D,hidden,hires_images,False)
            return values,state
        if self._graph is None:
            try:
                self._static=[x.clone() for x in (images,K,D,hidden,hires_images)]
                # Warm with scratch history; the real history is copied only for replay.
                self._static[3].zero_()
                for _ in range(3):self.engine.forward(*self._static,False)
                torch.cuda.current_stream().synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
                    self._graph_values,self._graph_next,_=self.engine.forward(*self._static,False)
                self._graph=graph
            except RuntimeError as exc:
                self.graph_error=str(exc);self.actual_execution='optimized_eager'
                self._graph=None
                warnings.warn('CUDA Graph unavailable; explicit optimized_eager fallback: '+str(exc),RuntimeWarning)
                values,state,_=self.engine.forward(images,K,D,hidden,hires_images,False)
                return values,state
        for target,source in zip(self._static,(images,K,D,hidden,hires_images)):
            target.copy_(source)
        self._graph.replay()
        return self._graph_values,self._graph_next

    @staticmethod
    def export_values(values):
        return values


class LibTorchPredictor(StereoPredictor):
    def __init__(self, checkpoint, *, fps=60.0, preprocessing='gpu', visual_engine=None, execution_mode='reference', cuda_fusions=()):
        if execution_mode not in ('reference','optimized_eager','optimized_graph'):
            raise ValueError('Unsupported execution_mode')
        if execution_mode!='reference' and preprocessing!='gpu':
            raise ValueError('Optimized execution requires GPU preprocessing')
        self.execution_mode=execution_mode
        self.cuda_fusions=tuple(cuda_fusions)
        if len(set(cuda_fusions))!=len(cuda_fusions) or set(cuda_fusions)-{'preprocess','hires_crop','landmark_reduce'}:
            raise ValueError('Invalid cuda_fusions')
        if cuda_fusions and (execution_mode=='reference' or preprocessing!='gpu' or visual_engine is None):
            raise ValueError('CUDA fusions require optimized execution, GPU input and TensorRT')
        if preprocessing not in ('cpu','gpu'): raise ValueError('preprocessing must be cpu or gpu')
        self.checkpoint_path=Path(checkpoint).expanduser().resolve()
        data=torch.load(self.checkpoint_path,map_location='cpu',weights_only=True)
        if data['diagnostic']: raise ValueError('Expected a trained checkpoint')
        if not math.isfinite(fps) or fps<=0: raise ValueError('Invalid fps')
        self.diagnostic=False
        self.training_config=data['config']; self.instrument_model=data['instrument_model']
        self.epoch=int(data['epoch']); self.image_size=tuple(self.training_config['image_size'])
        self.period_ns=1e9/fps; self.device=torch.device('cuda:0')
        self.visual_engine=None
        if visual_engine is not None:
            import hashlib
            visual_engine=Path(visual_engine).expanduser().resolve()
            metadata=json.loads(visual_engine.with_suffix('.json').read_text())
            def sha(path):
                with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
            if sha(self.checkpoint_path)!=metadata['checkpoint_sha256'] or sha(visual_engine)!=metadata['engine_sha256']:
                raise ValueError('TensorRT engine/checkpoint provenance mismatch')
            self.visual_engine=metadata
        self.model=LibTorchModel(data['model'],self.training_config['model'],visual_engine,execution_mode,cuda_fusions)
        self.preprocessing=preprocessing
        if preprocessing=='gpu':
            if self.image_size!=(360,640): raise ValueError('GPU preprocessing expects a 640x360 network input')
            # A tiny constant table retains the original NumPy FP32 rounding;
            # resizing, lookup normalization and layout conversion run on GPU.
            mean=np.asarray([.485,.456,.406],np.float32)
            std=np.asarray([.229,.224,.225],np.float32)
            lut=(np.arange(256,dtype=np.float32)[:,None]/255.0-mean)/std
            self.preprocess_module=self.model.native_module if 'preprocess' in cuda_fusions else 'stereo_libtorch'
            factory=importlib.import_module(self.preprocess_module).StereoPreprocess
            self.preprocessor=factory(torch.from_numpy(lut),True) if 'preprocess' in cuda_fusions else factory(torch.from_numpy(lut))
            self._host_rgb=self.preprocessor.host_input().numpy()
            self._input_done=torch.cuda.Event()
            self._input_pending=False
            self._calibration_key=None
        self.pipeline=_display_api(); self.reset_count=0; self.reset()
        self._stream=None
        if execution_mode=='optimized_graph':
            self._stream=torch.cuda.Stream(device=self.device)
            self._stream.wait_stream(torch.cuda.current_stream(self.device))

    def __call__(self, frame, calibration):
        if self._stream is None:return super().__call__(frame,calibration)
        with torch.cuda.stream(self._stream):
            return super().__call__(frame,calibration)

    def _preprocess(self, frame, calibration):
        if self.preprocessing=='cpu': return super()._preprocess(frame,calibration)
        if (calibration['image']['height'],calibration['image']['width'])!=(720,1280):
            raise ValueError('GPU preprocessing expects original 1280x720 RGB')
        images=[np.asarray(getattr(frame,e+'_rgb')) for e in ('left','right')]
        if any(x.shape!=(720,1280,3) or x.dtype!=np.uint8 for x in images):
            raise ValueError('Frame RGB dimensions/type disagree with its calibration')
        # Usually already complete after the previous values17 D2H. The event
        # also protects pinned memory if the preceding call raised an error.
        if self._input_pending and not self._input_done.query(): self._input_done.synchronize()
        for i,image in enumerate(images): np.copyto(self._host_rgb[i],image)
        try:
            small,self._hires,self.resized_rgb=self.preprocessor.run()
        finally:
            self._input_done.record()
            self._input_pending=True
        key=json.dumps(calibration,sort_keys=True,separators=(',',':'))
        if key!=self._calibration_key:
            K=np.stack([resize_intrinsics(np.asarray(calibration[e]['K']),(720,1280),self.image_size)
                        for e in ('left','right')])
            D=np.asarray([calibration[e]['D'] for e in ('left','right')])
            self._K=torch.as_tensor(K,dtype=torch.float32,device=self.device)[None,None]
            self._D=torch.as_tensor(D,dtype=torch.float32,device=self.device)[None,None]
            if self.execution_mode!='reference':self.model.engine.set_calibration(self._K,self._D)
            self._calibration_key=key
        return small,self._K,self._D

    def _prepare_inputs(self, frame, cameras):
        if self.preprocessing=='cpu': return super()._prepare_inputs(frame,cameras)
        inputs=self._preprocess(frame,cameras)
        return inputs,{'hires_images':self._hires}


def build_predictor(config):
    path=Path(config['checkpoint']).expanduser()
    if not path.is_absolute(): path=Path(config.get('_config_dir','.'))/path
    engine=config.get('visual_engine')
    if engine is not None:
        engine=Path(engine).expanduser()
        if not engine.is_absolute():engine=Path(config.get('_config_dir','.'))/engine
    return LibTorchPredictor(path,fps=float(config.get('fps',60)),preprocessing=config.get('preprocessing','gpu'),visual_engine=engine,execution_mode=config.get('execution_mode','reference'),cuda_fusions=config.get('cuda_fusions',()))
