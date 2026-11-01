"""Preserved approved distortion and appearance processing, scaled to output size."""
import numpy as np
import cv2
EYES=("left","right")

def maps(profile,focal_length_mm=4.):
    h,w=profile['image']['height'],profile['image']['width']
    y,x=np.indices((h,w),dtype=np.float64); pixels=np.stack([x,y],-1).reshape(-1,1,2)
    rays={e:cv2.undistortPointsIter(pixels,np.array(profile[e]['K']),np.array(profile[e]['D']),None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(h,w,2) for e in EYES}
    f=1600.*w/1920
    cx=float(np.ceil((max(abs(r[:,:,0]).max() for r in rays.values())*f+16)/32)*32)
    cy=float(np.ceil((max(abs(r[:,:,1]).max() for r in rays.values())*f+16)/32)*32)
    mapped={e:(r*f+[cx,cy]).astype(np.float32) for e,r in rays.items()}
    return mapped,dict(width=int(cx*2),height=int(cy*2),f=f,cx=cx,cy=cy,
                      focal_length_mm=focal_length_mm,horizontal_aperture_mm=2*cx/f*focal_length_mm,vertical_aperture_mm=2*cy/f*focal_length_mm)

def project(points,profile,eye):
    p=np.asarray(points,float).reshape(-1,3)
    uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),np.array(profile[eye]['K']),np.array(profile[eye]['D']))
    return uv.reshape(-1,2)

def image_effects(linear,eye,profile,initial,state,seed,config):
    h,w=linear.shape[:2]; sx=w/1920; sy=h/1080
    y,x=np.indices((h,w),dtype=np.float32)
    cx,cy=profile[eye]['K'][0][2],profile[eye]['K'][1][2]
    far=max(np.hypot(px-cx,py-cy) for px,py in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)])
    r=np.clip(np.hypot(x-cx,y-cy)/far,0,1)
    sigma=np.sqrt(.3**2+(.8*r*r)**2)*sx*state['blur_multiplier']
    levels=np.linspace(float(sigma.min()),float(sigma.max()),17); step=levels[1]-levels[0]
    blurred=np.zeros_like(linear,dtype=np.float32)
    for s in levels:
        weight=np.maximum(1-abs(sigma-s)/step,0)
        blurred+=weight[:,:,None]*cv2.GaussianBlur(linear,(0,0),float(s),borderType=cv2.BORDER_REFLECT_101)
    # Preserve the accepted ellipse in normalized image coordinates.
    r=np.sqrt(((x-(959.5+.5)*sx+.5)/(840*sx))**2+((y-(539.5+.5)*sy+.5)/(470*sy))**2)
    t=np.clip((r-.3)/.8,0,1); gain=1-(1-config['vignette_minimum_linear_gain'])*t*t*(3-2*t)
    rgb=blurred*np.asarray(initial['illumination']['combined_rgb_scales'][eye])*gain[:,:,None]*config['global_linear_multiplier']
    rgb=np.maximum(rgb,0); rgb=np.clip(np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055),0,1)
    rng=np.random.default_rng(seed); luminance=rgb@np.array([.2126,.7152,.0722]); factor=state['noise_multiplier']
    noise=rng.standard_normal(rgb.shape)*np.sqrt(1+6*luminance)[:,:,None]*factor
    row=rng.normal(0,.3*factor,(h,1,1))
    return np.uint8(np.clip(np.rint(rgb*255+noise+row),0,255))
