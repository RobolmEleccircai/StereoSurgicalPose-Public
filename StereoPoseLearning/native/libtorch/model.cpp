#ifdef STEREO_FUSIONS
#include "fusion_cuda.h"
#endif
#include "model.h"
#include "profile_ranges.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <limits>
#ifdef STEREO_OPTIMIZED
#include "position_cuda.h"
#endif

using T = at::Tensor;
using V = std::vector<T>;
using namespace at::indexing;
namespace {
T normed(const T& x, double eps=1e-6) {
  return x / at::linalg_vector_norm(x, 2, {-1}, true).clamp_min(eps);
}
T resize(const T& x, int64_t height, int64_t width) {
  return at::upsample_bilinear2d(x, {height,width}, false, std::nullopt, std::nullopt);
}
T fresh(const T& x, std::initializer_list<float> values) {
  return at::tensor(values, x.options());
}
T rotation(const T& raw) {
  auto a=raw.slice(-1,0,3), b=raw.slice(-1,3,6);
  auto fallback=at::zeros_like(a); fallback.select(-1,0).fill_(1);
  auto x=at::where(at::linalg_vector_norm(a,2,{-1},true)>1e-6,normed(a),fallback);
  auto y0=b-(x*b).sum(-1,true)*x;
  auto basis=at::one_hot(x.abs().argmin(-1),3).to(x.scalar_type());
  auto ortho=basis-(basis*x).sum(-1,true)*x;
  y0=at::where(at::linalg_vector_norm(y0,2,{-1},true)>1e-6,y0,ortho);
  auto y=normed(y0), z=at::linalg_cross(x,y,-1);
  return at::stack({x,y,z},-1);
}
T quaternion(const T& m) {
  auto m00=m.index({Ellipsis,0,0}),m01=m.index({Ellipsis,0,1}),m02=m.index({Ellipsis,0,2});
  auto m10=m.index({Ellipsis,1,0}),m11=m.index({Ellipsis,1,1}),m12=m.index({Ellipsis,1,2});
  auto m20=m.index({Ellipsis,2,0}),m21=m.index({Ellipsis,2,1}),m22=m.index({Ellipsis,2,2});
  auto sq=at::stack({1+m00+m11+m22,1+m00-m11-m22,1-m00+m11-m22,1-m00-m11+m22},-1);
  auto magnitude=sq.clamp_min(std::numeric_limits<float>::epsilon()).sqrt();
  auto candidates=at::stack({at::stack({sq.select(-1,0),m21-m12,m02-m20,m10-m01},-1),
      at::stack({m21-m12,sq.select(-1,1),m10+m01,m02+m20},-1),
      at::stack({m02-m20,m10+m01,sq.select(-1,2),m12+m21},-1),
      at::stack({m10-m01,m20+m02,m21+m12,sq.select(-1,3)},-1)},-2);
  candidates=candidates/(2*magnitude.clamp_min(.1).unsqueeze(-1));
  auto index=magnitude.argmax(-1).unsqueeze(-1).unsqueeze(-1).expand({1,1,2,1,4});
  auto q=normed(candidates.gather(-2,index).squeeze(-2),1e-12);
  q=at::where(q.slice(-1,0,1)<0,-q,q);
  return at::cat({q.slice(-1,1,4),q.slice(-1,0,1)},-1);
}
T position(const T& uv,const T& z,const T& K,const T& D) {
  auto pixels=uv.to(at::kDouble), calibration=K.to(at::kDouble), distortion=D.to(at::kDouble);
  auto inverse=std::get<0>(at::linalg_inv_ex(calibration,false));
  auto homogeneous=at::cat({pixels,at::ones_like(pixels.slice(-1,0,1))},-1);
  auto distorted=at::matmul(inverse,homogeneous.unsqueeze(-1)).squeeze(-1);
  auto xy=(distorted.slice(-1,0,2)/distorted.slice(-1,2,3)).unbind(-1);
  auto xd=xy[0],yd=xy[1],x=xd,y=yd;
  auto ds=distortion.unbind(-1); auto k1=ds[0],k2=ds[1],p1=ds[2],p2=ds[3],k3=ds[4];
  for(int i=0;i<10;++i) {
    auto r2=x*x+y*y;
    auto radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2;
    auto tx=2*p1*x*y+p2*(r2+2*x*x),ty=p1*(r2+2*y*y)+2*p2*x*y;
    auto g=k1+2*k2*r2+3*k3*r2*r2,rx=2*x*g,ry=2*y*g;
    auto j11=radial+x*rx+2*p1*y+6*p2*x,j12=x*ry+2*p1*x+2*p2*y;
    auto j21=y*rx+2*p1*x+2*p2*y,j22=radial+y*ry+6*p1*y+2*p2*x;
    auto ex=x*radial+tx-xd,ey=y*radial+ty-yd,det=j11*j22-j12*j21;
    double eps=std::numeric_limits<double>::epsilon()*8;
    det=at::where(det.abs()>=eps,det,at::where(det>=0,eps,-eps));
    auto sx=(j22*ex-j12*ey)/det,sy=(-j21*ex+j11*ey)/det;
    x=x-sx; y=y-sy;
  }
  auto rayxy=at::stack({x,y},-1);
  return (at::cat({rayxy,at::ones_like(rayxy.slice(-1,0,1))},-1)*z.to(at::kDouble)).to(at::kFloat);
}
}

T stereo_position_reference(const T& uv,const T& z,const T& K,const T& D) {return position(uv,z,K,D);}
T StereoNet::constant(const std::string& name,const std::function<T()>& create) const {
#ifdef STEREO_OPTIMIZED
  auto found=constants_.find(name);
  if(found!=constants_.end())return found->second;
  auto value=create();constants_[name]=value;return value;
#else
  return create();
#endif
}
void StereoNet::set_calibration(const T& K,const T& D) {
#ifdef STEREO_OPTIMIZED
  auto inv=std::get<0>(at::linalg_inv_ex(K.to(at::kDouble),false)).contiguous();
  auto distortion=D.to(at::kDouble).contiguous();
  if(inverse_.defined()) {inverse_.copy_(inv);distortion_.copy_(distortion);}
  else {inverse_=inv;distortion_=distortion;}
#endif
}

const T& StereoNet::w(const std::string& name) const {return weights_.at(name);}
#ifdef STEREO_WITH_TENSORRT
void StereoNet::load_visual_engine(const std::string& path) {
  visual_engine_=std::make_unique<TrtVisual>(path);
}
#endif
double StereoNet::cfg(const std::string& name,double fallback) const {
  auto p=config_.find(name); return p==config_.end()?fallback:p->second;
}
StereoNet::StereoNet(const TensorMap& state,const NumericConfig& config):config_(config) {
  c10::InferenceMode guard;
  for(const auto& item:state) weights_[item.first]=item.second.detach().to(c10::Device(c10::kCUDA,0));
  TORCH_CHECK(cfg("feature_dim",128)==128 && cfg("hidden_dim",256)==256 &&
      cfg("channels_last",0)==0 && cfg("roi_pool_size",4)==4 && cfg("roi_local_encoder",1)==1,
      "LibTorch definition must match this checkpoint architecture");
  for(const auto& prefix: {std::string("temporal"),std::string("foreground_parts_net.wrist_branch.temporal"),
                          std::string("foreground_parts_net.jaw_branch.temporal")}) {
    auto gru=torch::nn::GRU(torch::nn::GRUOptions(128,256).batch_first(true));
    gru->to(c10::Device(c10::kCUDA,0));
    for(auto& p:gru->named_parameters()) p.value().copy_(w(prefix+"."+p.key()));
    gru->eval(); gru->flatten_parameters(); recurrent_.emplace(prefix,gru);
  }
}
T StereoNet::conv(const T& x,const std::string& p,int stride,int pad,int groups) const {
  auto b=weights_.find(p+".bias");
  return at::conv2d(x,w(p+".weight"),b==weights_.end()?std::optional<T>{}:b->second,
                    {stride,stride},{pad,pad},{1,1},groups);
}
T StereoNet::linear(const T& x,const std::string& p) const {return at::linear(x,w(p+".weight"),w(p+".bias"));}
T StereoNet::ln(const T& x,const std::string& p,double eps) const {
  return at::layer_norm(x,{w(p+".weight").numel()},w(p+".weight"),w(p+".bias"),eps,true);
}
T StereoNet::ln2(const T& x,const std::string& p) const {return ln(x.permute({0,2,3,1}),p,1e-6).permute({0,3,1,2});}
T StereoNet::gn(const T& x,const std::string& p) const {return at::group_norm(x,8,w(p+".weight"),w(p+".bias"),1e-5,true);}
T StereoNet::mlp(const T& x,const std::string& p) const {return linear(at::gelu(linear(x,p+".0")),p+".2");}

T StereoNet::attention(const T& qin,const T& kvin,const std::string& p) const {
  // Same packed projection and SDPA layout used by nn.MultiheadAttention when
  // key==value, query differs, batch_first=True, need_weights=False, eval().
  auto q0=qin.transpose(1,0),kv0=kvin.transpose(1,0);
  auto q=at::linear(q0,w(p+".in_proj_weight").slice(0,0,128),w(p+".in_proj_bias").slice(0,0,128));
  auto kv=at::linear(kv0,w(p+".in_proj_weight").slice(0,128,384),w(p+".in_proj_bias").slice(0,128,384));
  kv=kv.unflatten(-1,{2,128}).unsqueeze(0).transpose(0,-2).squeeze(-2).contiguous();
  auto k=kv[0],v=kv[1];
  auto heads=static_cast<int64_t>(cfg("attention_heads",4)); auto hd=128/heads;
  q=q.view({6,heads,hd}).transpose(0,1).view({1,heads,6,hd});
  k=k.view({6,heads,hd}).transpose(0,1).view({1,heads,6,hd});
  v=v.view({6,heads,hd}).transpose(0,1).view({1,heads,6,hd});
  auto out=at::scaled_dot_product_attention(q,k,v,std::nullopt,0,false);
  out=out.permute({2,0,1,3}).contiguous().view({6,128});
  return linear(out,p+".out_proj").view({6,1,128}).transpose(1,0);
}

std::tuple<T,T,T> StereoNet::guided(const T& spatial,const T& mask_logits,const T& heat) const {
  const std::string p="guided_regions";
  int64_t h=spatial.size(2),ww=spatial.size(3),patch=static_cast<int64_t>(cfg("roi_patch_size",8));
  auto probs=heat.to(at::kFloat).sigmoid();
  auto options=probs.options().dtype(at::kLong);
  auto grid=constant("guided_grid",[&]{return at::stack(at::meshgrid({(at::arange(h,options)+.5)*2/h-1,(at::arange(ww,options)+.5)*2/ww-1},"ij"));});
  auto yy=grid[0],xx=grid[1],weights=probs.pow(cfg("landmark_power",4))+1e-12;
  weights=weights/weights.sum({-2,-1},true).clamp_min(1e-12);
  auto xy=at::stack({(weights*xx).sum({-2,-1}),(weights*yy).sum({-2,-1})},-1);
  auto confidence=probs.amax({-2,-1});
  auto radius=constant("guided_radius",[&]{return fresh(xy,{static_cast<float>(2*cfg("roi_radius_px",24)/640),static_cast<float>(2*cfg("roi_radius_px",24)/360)});});
  auto offsets=constant("guided_offsets",[&]{auto t=(at::arange(patch,options)+.5)*2/patch-1;
    auto gg=at::meshgrid({t,t},"ij");return at::stack({gg[1],gg[0]},-1)*radius;});
  auto sample=xy.unsqueeze(2).unsqueeze(2)+offsets.unsqueeze(0).unsqueeze(0);
  auto patches=at::grid_sampler(spatial.to(at::kFloat),sample.reshape({2,5*patch,patch,2}),0,1,false);
  patches=patches.reshape({2,128,5,patch,patch}).permute({0,2,1,3,4}).reshape({10,128,patch,patch});
  auto local_conv=conv(at::gelu(gn(conv(patches,p+".local_encoder.0",1,1,128),p+".local_encoder.1")),p+".local_encoder.3");
  patches=patches+local_conv;
  auto pool=static_cast<int64_t>(cfg("roi_pool_size",4));
  patches=at::adaptive_avg_pool2d(patches,{pool,pool}).flatten(1).reshape({2,5,128*pool*pool});
  auto local=at::gelu(ln(linear(patches,p+".patch_project.0"),p+".patch_project.1"));
  auto mask=mask_logits.to(at::kFloat).sigmoid();
  auto wholeweights=mask.square()+0.0; // foreground_context_floor is forced to zero by StereoPoseNet.
  wholeweights=wholeweights/wholeweights.sum({-2,-1},true).clamp_min(1e-12);
  auto whole=(spatial*wholeweights).sum({-2,-1});
  auto centre=at::stack({(wholeweights.select(1,0)*xx).sum({-2,-1}),(wholeweights.select(1,0)*yy).sum({-2,-1})},-1);
  auto ex=xx-centre.select(1,0).unsqueeze(1).unsqueeze(1),ey=yy-centre.select(1,1).unsqueeze(1).unsqueeze(1);
  auto extent=at::stack({(ex.square()*wholeweights.select(1,0)).sum({-2,-1}).sqrt(),
                       (ey.square()*wholeweights.select(1,0)).sum({-2,-1}).sqrt()},-1);
  auto wholemeta=at::cat({centre,extent,mask.mean({-2,-1})},-1);
  auto localmeta=at::cat({xy,radius.expand({2,5,2}),confidence.unsqueeze(-1)},-1);
  auto meta=at::cat({wholemeta.unsqueeze(1),localmeta},1);
  auto tokens=at::cat({whole.unsqueeze(1),local},1);
  tokens=tokens+mlp(meta,p+".metadata")+w(p+".region_embedding").unsqueeze(0);
  auto uv=(xy+1)*constant("size",[&]{return fresh(xy,{640,360});})/2-.5;
  return {tokens,uv,confidence};
}

std::tuple<T,T,T> StereoNet::local_branch(const T& crop,const T& meta,const T& hidden,const std::string& p,bool debug) {
  auto x=crop;
  for(int i=0;i<3;++i) {
    auto stem=p+".encoder.stem."+std::to_string(i);
    x=at::gelu(gn(conv(x,stem+".0",i<2?2:1,1),stem+".1"));
  }
  auto refine=conv(at::gelu(gn(conv(x,p+".encoder.refine.0",1,1,64),p+".encoder.refine.1")),p+".encoder.refine.3");
  x=x+refine;
  T heat;
#ifdef STEREO_OPTIMIZED
  if(debug)
#endif
    heat=conv(x,p+".encoder.heatmap");
  auto desc=at::gelu(ln(linear(at::adaptive_avg_pool2d(x,{4,4}).flatten(1),p+".encoder.project.0"),p+".encoder.project.1"));
  desc=desc+mlp(meta,p+".metadata"); auto paired=desc.reshape({1,2,128});
  auto joined=at::cat({paired,paired.flip({1})},-1);
  auto cross=linear(at::gelu(linear(ln(joined,p+".cross_view.0"),p+".cross_view.1")),p+".cross_view.3");
  auto fused=paired+cross;
  auto seq=fused.reshape({1,1,2,128}).permute({0,2,1,3}).reshape({2,1,128});
  auto rr=recurrent_.at(p+".temporal")->forward(seq,hidden);
  auto state=std::get<0>(rr).reshape({1,2,1,256}).permute({0,2,1,3});
  return {mlp(state.flatten(-2),p+".head"),heat,std::get<1>(rr)};
}

std::tuple<T,T,TensorMap> StereoNet::forward(const T& images,const T& K,const T& D,
                                          const T& hidden,const T& hires,bool debug) {
  c10::InferenceMode guard;
  TORCH_CHECK(images.sizes()==at::IntArrayRef({1,1,2,3,360,640}) && images.scalar_type()==at::kFloat && images.is_cuda(),"Expected FP32 single stereo pair on CUDA");
  TORCH_CHECK(K.sizes()==at::IntArrayRef({1,1,2,3,3}) && D.sizes()==at::IntArrayRef({1,1,2,5}),"Invalid K/D shape");
  TORCH_CHECK(hires.sizes()==at::IntArrayRef({1,1,2,3,720,1280}) && hires.scalar_type()==at::kByte,"Expected uint8 native 720p input");
  TORCH_CHECK(hidden.sizes()==at::IntArrayRef({3,2,256}),"Expected three recurrent states");
  TensorMap detail;
  ProfileStages profile;
  T spatial,mask,heat;
#ifdef STEREO_WITH_TENSORRT
  if(visual_engine_) {
    profile.next("stage::tensorrt_visual");
    std::tie(spatial,mask,heat)=visual_engine_->forward(images.reshape({2,3,360,640}).contiguous());
  } else
#endif
  {
  profile.next("stage::backbone_stem");
  auto x=ln2(conv(images.reshape({2,3,360,640}),"backbone.stem_0",4),"backbone.stem_1");
  V maps; const int depths[]={2,2,8,2};
  for(int stage=0;stage<4;++stage) {
    const char* names[]={"stage::backbone_0","stage::backbone_1","stage::backbone_2","stage::backbone_3"};
    profile.next(names[stage]);
    auto p="backbone.stages_"+std::to_string(stage);
    if(stage) x=conv(ln2(x,p+".downsample.0"),p+".downsample.1",2);
    for(int block=0;block<depths[stage];++block) {
      auto b=p+".blocks."+std::to_string(block); auto shortcut=x;
      x=at::gelu(conv(ln2(conv(x,b+".conv_dw",1,3,x.size(1)),b+".norm"),b+".mlp.fc1"));
      auto g=at::linalg_vector_norm(x,2,{2,3},true);
      auto normal=g/(g.mean(1,true)+1e-6);
      x=x+at::addcmul(w(b+".mlp.grn.bias").view({1,-1,1,1}),w(b+".mlp.grn.weight").view({1,-1,1,1}),x*normal);
      x=conv(x,b+".mlp.fc2")+shortcut;
    }
    maps.push_back(x); if(debug) detail["backbone_"+std::to_string(stage)]=x;
  }
  profile.next("stage::fpn_mask_keypoints");
  T pyramid;
  for(int i=3;i>=0;--i) {
    auto projected=conv(maps[i],"laterals."+std::to_string(i));
    pyramid=i==3?projected:projected+resize(pyramid.to(at::kFloat),projected.size(2),projected.size(3));
  }
  spatial=at::gelu(gn(conv(pyramid,"spatial.0",1,1),"spatial.1"));
  mask=conv(spatial,"mask_head");heat=conv(spatial,"heatmap_head");
  }
  profile.next("stage::guided_roi_features");
  auto region=guided(spatial,mask,heat); auto tokens=std::get<0>(region),uv=std::get<1>(region),confidence=std::get<2>(region);
  profile.next("stage::cross_view_attention");
  auto camera=at::cat({at::stack({K.index({Ellipsis,0,0})/640,K.index({Ellipsis,1,1})/360,
       K.index({Ellipsis,0,2})/640,K.index({Ellipsis,1,2})/360},-1),D},-1);
  tokens=tokens.reshape({1,2,6,128})+mlp(camera.reshape({1,2,9}),"camera").unsqueeze(2);
  tokens=tokens+w("view_embedding").unsqueeze(0).unsqueeze(2);
  for(int i=0;i<static_cast<int>(cfg("fusion_layers",2));++i) {
    auto p="fusion."+std::to_string(i);
    auto left=tokens.select(1,0),right=tokens.select(1,1),lnorm=ln(left,p+".norm"),rnorm=ln(right,p+".norm");
    left=left+attention(lnorm,rnorm,p+".attention");
    right=right+attention(rnorm,lnorm,p+".attention");
    auto ffn=[&](const T& v){return linear(at::gelu(linear(ln(v,p+".ffn.0"),p+".ffn.1")),p+".ffn.3");};
    tokens=at::stack({left+ffn(left),right+ffn(right)},1);
  }
  profile.next("stage::body_gru_pose_heads");
  auto pooled=at::gelu(ln(linear(tokens.flatten(-2),"guided_pose_pool.0"),"guided_pose_pool.1")).reshape({1,1,2,128});
  auto Ctoken=tokens.select(2,1).reshape({1,1,2,128});
  auto body=recurrent_.at("temporal")->forward(pooled.permute({0,2,1,3}).reshape({2,1,128}),hidden.slice(0,0,1));
  auto states=std::get<0>(body).reshape({1,2,1,256}).permute({0,2,1,3});
  V rots,depths_raw; auto depth_features=at::cat({states,Ctoken},-1);
  for(int i=0;i<2;++i) {
    rots.push_back(mlp(states.select(2,i),"rotation_heads."+std::to_string(i)));
    depths_raw.push_back(mlp(depth_features.select(2,i),"depth_heads."+std::to_string(i)));
  }
  auto R=rotation(at::stack(rots,2));
  auto z=at::softplus(at::stack(depths_raw,2))*cfg("depth_scale_mm",100)+.001;
  profile.next("stage::C_ray_undistortion");
  auto alluv=uv.reshape({1,1,2,5,2}),Cuv=alluv.select(3,0);
#ifdef STEREO_OPTIMIZED
  TORCH_CHECK(inverse_.defined(),"Call set_calibration before optimized forward");
  auto C=stereo_position_cuda(Cuv,z,inverse_,distortion_);
#else
  auto C=position(Cuv,z,K,D);
#endif
  profile.next("stage::hires_mask_crop");
  // Original two foreground-masked local crops, with unchanged interpolation.
  auto hinge=uv.slice(1,1,3).mean(1,true);
  V selected={at::cat({uv.slice(1,0,1),hinge,uv.slice(1,3,5)},1),at::cat({hinge,uv.slice(1,3,5)},1)};
  V centres,extents;
  for(int i=0;i<2;++i) {
    std::string name=i==0?"wrist":"jaw";
    auto lo=selected[i].amin(1),hi=selected[i].amax(1);
    centres.push_back((lo+hi)*.5);
    auto radius=((hi-lo)*.5).amax(-1,true)+cfg(name+"_roi_margin_px",10);
    extents.push_back(radius.clamp(cfg(name+"_roi_min_half_px",i==0?24:20),cfg(name+"_roi_max_half_px",i==0?128:96)).expand({-1,2}));
  }
  auto centre=at::stack(centres,1),extent=at::stack(extents,1);
  int64_t size=static_cast<int64_t>(cfg("part_crop_size",160));
  auto offsets=constant("crop_offsets",[&]{auto axis=(at::arange(size,images.options())+.5)*2/size-1;
    auto grids=at::meshgrid({axis,axis},"ij");return at::stack({grids[1],grids[0]},-1);});
  auto pixel_grid=centre.unsqueeze(2).unsqueeze(2)+extent.unsqueeze(2).unsqueeze(2)*offsets;
  auto grid=(pixel_grid+.5)*constant("crop_scale",[&]{return fresh(pixel_grid,{2.f/640,2.f/360});})-1;
  T crop;
#ifdef STEREO_FUSIONS
  if(cfg("fusion_hires_crop",0))crop=stereo_crop_cuda(hires,mask,grid.reshape({2,2*size,size,2}),cfg("part_mask_cutoff",.2),static_cast<int>(cfg("part_mask_dilation_px",1)));
  else
#endif
  {
  auto mean=constant("mean",[&]{return fresh(grid,{.485f,.456f,.406f});}).view({1,3,1,1}),sd=constant("std",[&]{return fresh(grid,{.229f,.224f,.225f});}).view({1,3,1,1});
  auto probability=resize(mask.sigmoid(),720,1280); double cutoff=cfg("part_mask_cutoff",.2);
  probability=((probability-cutoff)/(1-cutoff)).clamp(0,1);
  int dilation=static_cast<int>(cfg("part_mask_dilation_px",1));
  if(dilation) probability=at::max_pool2d(probability,{2*dilation+1,2*dilation+1},{1,1},{dilation,dilation},{1,1},false);
  auto image=(hires.reshape({2,3,720,1280}).to(at::kFloat)/255-mean)/sd;
  auto masked=at::cat({image*probability,probability},1);
  crop=at::grid_sampler(masked,grid.reshape({2,2*size,size,2}),0,0,false);
  }
  crop=crop.reshape({2,4,2,size,size}).permute({0,2,1,3,4});
  auto observed=crop.select(2,3).mean({-2,-1});
  auto divisor=constant("size",[&]{return fresh(centre,{640,360});});
  auto boxmeta=at::cat({(centre+.5)/divisor,extent/divisor},-1);
  auto conf=confidence.mean(-1); V logits,heats,nextlocal;
  for(int i=0;i<2;++i) {
    profile.next(i==0?"stage::wrist_branch":"stage::jaw_branch");
    auto meta=at::cat({camera.reshape({2,9}),boxmeta.select(1,i),observed.select(1,i).unsqueeze(-1),conf.unsqueeze(-1)},-1);
    auto p=std::string("foreground_parts_net.")+(i==0?"wrist_branch":"jaw_branch");
    auto branch=local_branch(crop.select(1,i).contiguous(),meta,hidden.slice(0,i+1,i+2),p,debug);
    logits.push_back(std::get<0>(branch));heats.push_back(std::get<1>(branch));nextlocal.push_back(std::get<2>(branch));
  }
  profile.next("stage::pack_values17");
  auto joint=w("joint_lower")+(w("joint_upper")-w("joint_lower"))*at::cat(logits,-1).sigmoid();
  auto q=quaternion(R); auto values=at::cat({at::cat({C,q},-1).flatten(-2),joint},-1);
  auto next=at::cat({std::get<1>(body),at::cat(nextlocal,0)},0);
  T part_heat;
#ifdef STEREO_OPTIMIZED
  if(debug)
#endif
    part_heat=at::stack(heats,1).reshape({1,1,2,2,3,heats[0].size(2),heats[0].size(3)});
  if(debug) {
    detail["spatial"]=spatial;detail["tokens"]=tokens;detail["C"]=C;detail["R"]=R;detail["joints"]=joint;
    detail["landmark_uv"]=alluv;detail["landmark_confidence"]=confidence.reshape({1,1,2,5});
    detail["mask_logits"]=mask.reshape({1,1,2,1,mask.size(2),mask.size(3)});
    detail["heatmap_logits"]=heat.reshape({1,1,2,5,heat.size(2),heat.size(3)});
    detail["part_heatmap_logits"]=part_heat;detail["part_roi_centres"]=centre.reshape({1,1,2,2,2});
    detail["part_roi_half_extents"]=extent.reshape({1,1,2,2,2});
  }
  return {values,next,detail};
}

#ifdef STEREO_FUSIONS
at::Tensor stereo_crop_reference(const at::Tensor& hires,const at::Tensor& mask,const at::Tensor& grid,double cutoff,int dilation) {
  auto mean=at::tensor({.485f,.456f,.406f},at::kFloat).to(hires.device()).view({1,3,1,1});
  auto sd=at::tensor({.229f,.224f,.225f},at::kFloat).to(hires.device()).view({1,3,1,1});
  auto probability=at::upsample_bilinear2d(mask.sigmoid(),{720,1280},false,std::nullopt,std::nullopt);
  probability=((probability-cutoff)/(1-cutoff)).clamp(0,1);
  if(dilation)probability=at::max_pool2d(probability,{2*dilation+1,2*dilation+1},{1,1},{dilation,dilation},{1,1},false);
  auto image=(hires.reshape({2,3,720,1280}).to(at::kFloat)/255-mean)/sd;
  return at::grid_sampler(at::cat({image*probability,probability},1),grid,0,0,false);
}
#endif
