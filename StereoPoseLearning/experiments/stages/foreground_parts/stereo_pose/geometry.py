"""Differentiable geometry for the frozen PSM model, in camera optical frames.

Lengths are millimetres, angles radians, and exported quaternions xyzw. No
camera extrinsic, baseline, or per-frame truth is an argument to the decoder.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

POINT_NAMES = ("C", "H1", "H2", "J1", "J2", "T")
JOINT_NAMES = ("psm_tool_pitch_joint", "psm_tool_yaw_joint", "psm_tool_gripper2_joint")


def rotation_6d_to_matrix(value: Tensor, eps: float = 1e-6) -> Tensor:
    """First three values and last three values are the first two columns.

    Degenerate inputs get a deterministic orthogonal completion. The ordinary
    nondegenerate path is Gram--Schmidt and fully differentiable.
    """
    if value.shape[-1] != 6:
        raise ValueError("Rotation input must end in six values")
    a, b = value[..., :3], value[..., 3:]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    x = torch.where(a.norm(dim=-1, keepdim=True) > eps, F.normalize(a, dim=-1, eps=eps), fallback)
    y0 = b - (x * b).sum(-1, keepdim=True) * x
    basis = F.one_hot(x.abs().argmin(-1), 3).to(x)
    orthogonal = basis - (basis * x).sum(-1, keepdim=True) * x
    y0 = torch.where(y0.norm(dim=-1, keepdim=True) > eps, y0, orthogonal)
    y = F.normalize(y0, dim=-1, eps=eps)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def quaternion_xyzw_to_matrix(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("Quaternion input must end in xyzw")
    if not torch.isfinite(quaternion).all() or (quaternion.norm(dim=-1) < 1e-12).any():
        raise ValueError("Quaternion must be finite and nonzero")
    x, y, z, w = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(matrix: Tensor) -> Tensor:
    """Stable largest-component conversion, including 180-degree rotations.

    Export uses w >= 0. Rotation losses operate on matrices, never on this
    arbitrary quaternion sign convention.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrix must end in 3 x 3")
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    square = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                          1-m00+m11-m22, 1-m00-m11+m22), -1)
    magnitude = torch.sqrt(square.clamp_min(torch.finfo(matrix.dtype).eps))
    candidates = torch.stack((
        torch.stack((square[..., 0], m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, square[..., 1], m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, square[..., 2], m12+m21), -1),
        torch.stack((m10-m01, m20+m02, m21+m12, square[..., 3]), -1),
    ), -2) / (2 * magnitude.clamp_min(0.1)[..., :, None])
    index = magnitude.argmax(-1)[..., None, None].expand(matrix.shape[:-2] + (1, 4))
    wxyz = F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)
    wxyz = torch.where(wxyz[..., :1] < 0, -wxyz, wxyz)
    return torch.cat((wxyz[..., 1:], wxyz[..., :1]), -1)


def values17(C: Tensor, R: Tensor, joints: Tensor) -> Tensor:
    """Export [..., 17], with left eye first and one shared joint triplet."""
    if C.shape[-2:] != (2, 3) or R.shape[-3:] != (2, 3, 3) or joints.shape[-1] != 3:
        raise ValueError("Expected two camera poses and three shared joint angles")
    per_eye = torch.cat((C, matrix_to_quaternion_xyzw(R)), -1)
    return torch.cat((per_eye.flatten(-2), joints), -1)


def _distortion_terms(x: Tensor, y: Tensor, distortion: Tensor):
    if distortion.shape[-1] != 5:
        raise ValueError("Brown distortion order is k1,k2,p1,p2,k3")
    parameters = list(distortion.unbind(-1))
    while parameters[0].ndim < x.ndim:
        parameters = [v.unsqueeze(-1) for v in parameters]
    k1, k2, p1, p2, k3 = parameters
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    return radial, dx, dy, (k1, k2, p1, p2, k3), r2


def project_brown(points: Tensor, K: Tensor, D: Tensor, eps: float = 1e-8) -> Tensor:
    """Project [..., N, 3] using K[...,3,3], D[...,5], returning [...,N,2].

    A single point per camera ([...,3]) is also accepted. Geometry validity
    (positive depth / in-frame / visibility) is separate from projection.
    """
    single = points.ndim == K.ndim - 1
    if single:
        points = points.unsqueeze(-2)
    z = points[..., 2]
    safe_z = torch.where(z.abs() >= eps, z, torch.where(z >= 0, eps, -eps))
    x, y = points[..., 0]/safe_z, points[..., 1]/safe_z
    radial, dx, dy, _, _ = _distortion_terms(x, y, D)
    distorted = torch.stack((x*radial+dx, y*radial+dy, torch.ones_like(x)), -1)
    uvw = torch.matmul(K.unsqueeze(-3), distorted.unsqueeze(-1)).squeeze(-1)
    result = uvw[..., :2] / uvw[..., 2:]
    return result.squeeze(-2) if single else result


def undistort_brown(pixels: Tensor, K: Tensor, D: Tensor, iterations: int = 10) -> Tensor:
    """Fixed-iteration differentiable Newton inversion to normalized x,y rays.

    This does not produce a rectified stereo pair, and does not use baseline.
    The returned ray is (x,y,1), with no length normalization.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    single = pixels.ndim == K.ndim - 1
    if single:
        pixels = pixels.unsqueeze(-2)
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    distorted = torch.matmul(torch.linalg.inv(K).unsqueeze(-3), homogeneous.unsqueeze(-1)).squeeze(-1)
    xd, yd = (distorted[..., :2] / distorted[..., 2:]).unbind(-1)
    x, y = xd, yd
    for _ in range(iterations):
        radial, tx, ty, (k1, k2, p1, p2, k3), r2 = _distortion_terms(x, y, D)
        g = k1 + 2*k2*r2 + 3*k3*r2*r2
        rx, ry = 2*x*g, 2*y*g
        j11 = radial + x*rx + 2*p1*y + 6*p2*x
        j12 = x*ry + 2*p1*x + 2*p2*y
        j21 = y*rx + 2*p1*x + 2*p2*y
        j22 = radial + y*ry + 6*p1*y + 2*p2*x
        residual_x, residual_y = x*radial+tx-xd, y*radial+ty-yd
        det = j11*j22 - j12*j21
        eps = torch.finfo(det.dtype).eps * 8
        det = torch.where(det.abs() > eps, det, torch.where(det >= 0, eps, -eps))
        step_x = (j22*residual_x-j12*residual_y)/det
        step_y = (-j21*residual_x+j11*residual_y)/det
        x, y = x-step_x, y-step_y
    result = torch.stack((x, y), -1)
    return result.squeeze(-2) if single else result


def _transform(rotation: Tensor, translation: Tensor) -> Tensor:
    upper = torch.cat((rotation, translation[..., None]), -1)
    bottom = translation.new_zeros(translation.shape[:-1] + (1, 4))
    bottom[..., 0, 3] = 1
    return torch.cat((upper, bottom), -2)


def _axis_rotation(angle: Tensor, axis: Tensor) -> Tensor:
    x, y, z = axis.unbind(-1)
    zero = x*0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    aat = axis[:, None] * axis[None, :]
    cos, sin = angle.cos()[..., None, None], angle.sin()[..., None, None]
    return cos*eye + (1-cos)*aat + sin*skew


class FixedKinematics(nn.Module):
    """The same frozen A Rot(axis,q-q0) A^-1 B chain as pose_display.decode."""

    def __init__(self, model: str | Path | dict):
        super().__init__()
        if not isinstance(model, dict):
            model = json.loads(Path(model).read_text())
        if model.get("unit") != "mm":
            raise ValueError("Instrument model must use mm")
        self.model = model
        self.shaft_body = model["shaft_body"]
        self.chain = []
        found = {self.shaft_body}
        parameter_map = {JOINT_NAMES[0]: (0, 1), JOINT_NAMES[1]: (1, 1),
                         "psm_tool_gripper1_joint": (2, -1), JOINT_NAMES[2]: (2, 1)}
        for i, joint in enumerate(model["joints"]):
            if joint["parent"] not in found or joint["name"] not in parameter_map:
                raise ValueError("Unsupported or out-of-order PSM joint chain")
            a = torch.tensor(joint["A_parent_joint_mm"], dtype=torch.float64)
            b = torch.tensor(joint["B_parent_child_bind_mm"], dtype=torch.float64)
            axis = torch.eye(3, dtype=torch.float64)["XYZ".index(joint["axis"])]
            for name, value in (("A", a), ("A_inv", torch.linalg.inv(a)), ("B", b), ("axis", axis)):
                self.register_buffer(f"{name}_{i}", value)
            self.register_buffer(f"bind_{i}", torch.tensor(joint["bind_angle_rad"], dtype=torch.float64))
            self.chain.append((joint["parent"], joint["child"], *parameter_map[joint["name"]]))
            found.add(joint["child"])
        definitions = {j["name"]: j for j in model["joints"]}
        self.register_buffer("joint_lower", torch.tensor([definitions[n]["lower_rad"] for n in JOINT_NAMES], dtype=torch.float64))
        self.register_buffer("joint_upper", torch.tensor([definitions[n]["upper_rad"] for n in JOINT_NAMES], dtype=torch.float64))
        self.jaw_bodies = []
        for i in (1, 2):
            jaw = model["jaws"][f"jaw{i}"]
            self.jaw_bodies.append(jaw["body"])
            self.register_buffer(f"hinge_{i}", torch.tensor(jaw["hinge_local_mm"], dtype=torch.float64))
            self.register_buffer(f"tip_{i}", torch.tensor(jaw["tip_local_mm"], dtype=torch.float64))

    def _constant(self, name: str, like: Tensor) -> Tensor:
        return getattr(self, name).to(dtype=like.dtype, device=like.device)

    def forward(self, C: Tensor, R: Tensor, joints: Tensor) -> dict[str, Tensor]:
        if C.shape[-2:] != (2, 3) or R.shape != C.shape[:-1] + (3, 3):
            raise ValueError("C and R must describe two optical-frame poses")
        if joints.shape != C.shape[:-2] + (3,):
            raise ValueError("One shared joint triplet is required per stereo pair")
        poses = {self.shaft_body: _transform(R, C)}
        for i, (parent, child, parameter, sign) in enumerate(self.chain):
            angle = sign*joints[..., parameter, None] - self._constant(f"bind_{i}", C)
            rotation = _axis_rotation(angle, self._constant(f"axis_{i}", C))
            motion = _transform(rotation, torch.zeros_like(rotation[..., :, 0]))
            fixed = (self._constant(f"A_{i}", C) @ motion @ self._constant(f"A_inv_{i}", C)
                     @ self._constant(f"B_{i}", C))
            poses[child] = poses[parent] @ fixed
        hinges, tips, rotations = [], [], []
        for i, body in enumerate(self.jaw_bodies, 1):
            pose = poses[body]
            rotation, translation = pose[..., :3, :3], pose[..., :3, 3]
            hinges.append((rotation @ self._constant(f"hinge_{i}", C)[..., None]).squeeze(-1) + translation)
            tips.append((rotation @ self._constant(f"tip_{i}", C)[..., None]).squeeze(-1) + translation)
            rotations.append(rotation)
        points = torch.stack((C, hinges[0], hinges[1], tips[0], tips[1], (tips[0]+tips[1])/2), -2)
        return {"points": points, "d": R[..., :, 2], "R_jaws": torch.stack(rotations, -3)}
