"""Compile only the trained ConvNeXt/FPN/mask/landmark frontend for C++.

The engine has one static [2,3,360,640] FP32 input and three FP32 outputs.
No training, image labels, camera extrinsics or recurrent state enter the build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
from torch import nn
import torch.nn.functional as F


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class VisualFrontend(nn.Module):
    def __init__(self, model):
        super().__init__()
        for name in ('backbone','laterals','spatial','mask_head','heatmap_head'):
            setattr(self,name,getattr(model,name))

    def forward(self,images):
        maps=self.backbone(images)
        pyramid=None
        for feature,lateral in reversed(list(zip(maps,self.laterals))):
            projected=lateral(feature)
            pyramid=projected if pyramid is None else projected+F.interpolate(
                pyramid.float(),size=projected.shape[-2:],mode='bilinear',align_corners=False).to(projected.dtype)
        spatial=self.spatial(pyramid)
        return spatial,self.mask_head(spatial),self.heatmap_head(spatial)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=('fp32','fp16'),default='fp16')
    p.add_argument('--builder-level',type=int,choices=range(6),default=3)
    p.add_argument('--activation-policy',choices=('conservative','conv_gelu'),default='conservative')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from stereo_pose.model import StereoPoseNet,joint_bounds_from_model
    import tensorrt as trt
    import onnx
    torch.set_num_threads(4)
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=StereoPoseNet(data['config']['model'],joint_bounds_from_model(data['instrument_model']),pretrained=False)
    model.load_state_dict(data['model'],strict=True);model.eval()
    frontend=VisualFrontend(model).eval()
    onnx_path=a.output.with_suffix('.onnx')
    with torch.inference_mode():
        torch.onnx.export(frontend,torch.zeros(2,3,360,640),str(onnx_path),
            input_names=['images'],output_names=['spatial','mask','heat'],opset_version=17,
            dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(onnx_path))
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(0)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    config.builder_optimization_level=a.builder_level
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    # Keep reduction, normalization and GRN arithmetic in FP32. Convolutions
    # and learned linear maps use FP16; boundaries stay FP32 for the unchanged
    # eager downstream model. No blanket FP16 conversion of sensitive math.
    # Only GELU arithmetic whose nonconstant external input comes directly from
    # one Conv is eligible. Never include LayerNorm/GRN/reduction subgraphs.
    graph=onnx.load(str(onnx_path)).graph
    producers={out:node for node in graph.node for out in node.output}
    groups={}
    for node in graph.node:
        if '/act/' in node.name:
            groups.setdefault(node.name.split('/act/')[0]+'/act/',[]).append(node)
    approved=set()
    for nodes in groups.values():
        outputs={out for node in nodes for out in node.output}
        external={inp for node in nodes for inp in node.input if inp not in outputs and inp in producers and producers[inp].op_type!='Constant'}
        if (len(external)==1 and all(producers[x].op_type=='Conv' for x in external)
                and any(n.op_type=='Erf' for n in nodes)
                and all(n.op_type in ('Constant','Div','Erf','Add','Mul') for n in nodes)):
            approved.update(n.name for n in nodes if n.op_type!='Constant')
    half_layers=[];activation_layers=[];precision_layers=[]
    if a.precision=='fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer=network.get_layer(i)
            if layer.type in (trt.LayerType.CONVOLUTION,trt.LayerType.MATRIX_MULTIPLY):
                layer.precision=trt.float16;half_layers.append(layer.name)
            elif a.activation_policy=='conv_gelu' and layer.name in approved:
                layer.precision=trt.float16;half_layers.append(layer.name);activation_layers.append(layer.name)
            elif layer.type not in (trt.LayerType.CONSTANT,trt.LayerType.SHAPE):
                if any(layer.get_input(j) is not None and layer.get_input(j).dtype==trt.float32 for j in range(layer.num_inputs)):
                    layer.precision=trt.float32
    for i in range(network.num_layers):
        layer=network.get_layer(i)
        precision_layers.append({'name':layer.name,'type':str(layer.type),
                                'requested_precision':str(layer.precision) if layer.precision_is_set else 'builder'})
    if a.activation_policy=='conv_gelu' and not activation_layers:
        raise RuntimeError('No eligible Conv-adjacent GELU layers; candidate would be unchanged')
    for i in range(network.num_outputs):network.get_output(i).dtype=trt.float32
    print(json.dumps({'building':str(a.output),'precision':a.precision,'layers':network.num_layers,
                      'fp16_layers':len(half_layers),'trt':trt.__version__}),flush=True)
    start=time.perf_counter()
    blob=builder.build_serialized_network(network,config)
    if blob is None:raise RuntimeError('TensorRT engine build failed')
    a.output.write_bytes(bytes(blob))
    report={'schema':'stereo_visual_trt_v1','checkpoint_sha256':sha(a.checkpoint),
        'engine_sha256':sha(a.output),'onnx_sha256':sha(onnx_path),'precision':a.precision,
        'build_seconds':time.perf_counter()-start,'engine_bytes':a.output.stat().st_size,
        'tensorrt':trt.__version__,'torch':torch.__version__,'onnx':onnx.__version__,
        'gpu':torch.cuda.get_device_name(0),'input_shape':[2,3,360,640],
        'outputs':{'spatial':[2,128,90,160],'mask':[2,1,90,160],'heat':[2,5,90,160]},
        'boundary_dtype':'float32','tf32':False,'fp16_layers':half_layers,
        'builder_optimization_level':a.builder_level,'activation_policy':a.activation_policy,
        'activation_fp16_layers':activation_layers,'layer_precision_requests':precision_layers,
        'source_sha256':sha(__file__),'workspace_limit_bytes':1<<30}
    with trt.Runtime(logger) as runtime:
        engine=runtime.deserialize_cuda_engine(blob)
        inspector=engine.create_engine_inspector()
        a.output.with_suffix('.layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON)+'\n')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    onnx_path.unlink()  # Rebuildable intermediate; retain hash, not duplicate weights.
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
