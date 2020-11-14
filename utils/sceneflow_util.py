from __future__ import absolute_import, division, print_function
import torch
from torch import nn
import torch.nn.functional as tf


def post_processing(l_disp, r_disp):
    """?

    Args:
        l_disp ([type]): [description]
        r_disp ([type]): [description]

    Returns:
        [type]: [description]
    """

    b, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    grid_l = (
        torch.linspace(0.0, 1.0, w)
        .view(1, 1, 1, w)
        .expand(1, 1, h, w)
        .float()
        .requires_grad_(False)
        .cuda()
    )
    l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), 0, 1)
    r_mask = torch.flip(l_mask, [3])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def flow_horizontal_flip(flow_input):

    flow_flip = torch.flip(flow_input, [3])
    flow_flip[:, 0:1, :, :] *= -1

    return flow_flip.contiguous()


def disp2depth_kitti(pred_disp, k_value):

    pred_depth = (
        k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-8)
    )
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)

    return pred_depth


def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = (
        torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()
    )

    return pixelgrid


def pixel2pts(intrinsics, depth, rotation=None, C=None, forward_flow=None):
    """convert camera coordinates into world coordinates(heterogenous coordinates, 3D). DK^-1P
    Args:
        intrinsics ([type]): camera intrinsic matrix
        depth ([type]): depth matrix

    Returns:
        [type]: [description]
    """
    b, _, h, w = depth.size()

    pixelgrid = get_pixelgrid(b, h, w)
    pixel_mat = pixelgrid.view(b, 3, -1)  # b*3*pixel_number. In image world
    if forward_flow is not None:
        # pixelgrid = reconstructImg(pixelgrid-forward_flow, )
        pixel_mat[:, 0:2, :] += forward_flow  # TODO

    depth_mat = depth.view(b, 1, -1)
    pts_mat = (
        torch.inverse(intrinsics.cpu()).cuda() @ pixel_mat
    )  # image world => camera world, heterogeneous coordinates
    if rotation != None and C != None:  # FIXME
        pts_mat = depth_mat * pts_mat @ rotation + C
    else:
        pts_mat *= depth_mat

    pts = pts_mat.view(b, -1, h, w)  #  2*3*8*26. #* camera world points

    return pts, pixelgrid


def pts2pixel(pts, intrinsics):
    """K @ Pts and normalize by dividing channel w. output 2D coordinates in camera world"""
    """[summary]

    Returns:
        torch.Tensor: 2D coordinates of pixel world
    """
    b, _, h, w = pts.size()
    proj_pts = torch.matmul(intrinsics, pts.view(b, 3, -1))
    pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]  # devide w

    return pixels_mat.view(b, 2, h, w)


def intrinsic_scale(intrinsic, scale_y, scale_x):
    b, h, w = intrinsic.size()
    fx = intrinsic[:, 0, 0] * scale_x
    fy = intrinsic[:, 1, 1] * scale_y
    cx = intrinsic[:, 0, 2] * scale_x
    cy = intrinsic[:, 1, 2] * scale_y

    zeros = torch.zeros_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim=1)
    r2 = torch.stack([zeros, fy, cy], dim=1)
    r3 = (
        torch.tensor([0.0, 0.0, 1.0], requires_grad=False)
        .cuda()
        .unsqueeze(0)
        .expand(b, -1)
    )
    intrinsic_s = torch.stack([r1, r2, r3], dim=1)

    return intrinsic_s


def pixel2pts_world(intrinsic, depth, rel_scale, rotation, C, flow):  # TODO
    intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:, 0], rel_scale[:, 1])
    h, w = depth.shape[-2:]
    pts, _ = pixel2pts(intrinsic_dp_s, depth, rotation=rotation, C=C, forward_flow=flow)
    return pts


def pixel2pts_ms(intrinsic, output_disp, rel_scale):  # TODO
    """Convert pixel to pts via rescaled K and KITTI depth(from disp space)

    Args:
        intrinsic ([type]): [description]
        output_disp ([type]): [description]
        rel_scale ([type]): [description]

    Returns:
        pts, rescaled_K
    """
    # pixel2pts
    intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:, 0], rel_scale[:, 1])
    output_depth = disp2depth_kitti(output_disp, intrinsic_dp_s[:, 0, 0])
    pts, _ = pixel2pts(intrinsic_dp_s, output_depth)

    return pts, intrinsic_dp_s


def pts2pixel_ms(intrinsic, pts, output_sf, disp_size):

    # +sceneflow and reprojection
    sf_s = tf.interpolate(output_sf, disp_size, mode="bilinear", align_corners=True)
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    # * normalize grid into [-1,1]
    norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)

    return sf_s, pts_tform, norm_coord


def reconstructImg(coord, img):  # TODO
    """remove outliers after warping

    Args:
        coord (Tensor): warped coordinates
        img (Tensor): Original Image

    Returns:
        Tensor: warped image with outlier pixels removed
    """
    grid = coord.transpose(1, 2).transpose(2, 3)
    img_warp = tf.grid_sample(img, grid)

    mask = torch.ones_like(img, requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return img_warp * mask


def reconstructPts(coord, pts):  # TODO
    grid = coord.transpose(1, 2).transpose(2, 3)
    import pdb

    pdb.set_trace()
    pts_warp = tf.grid_sample(pts, grid)

    mask = torch.ones_like(pts, requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return pts_warp * mask


def projectSceneFlow2Flow(intrinsic, sceneflow, disp):  # TODO

    _, _, h, w = disp.size()

    output_depth = disp2depth_kitti(disp, intrinsic[:, 0, 0])
    pts, pixelgrid = pixel2pts(intrinsic, output_depth)

    sf_s = tf.interpolate(sceneflow, [h, w], mode="bilinear", align_corners=True)
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    flow = coord - pixelgrid[:, 0:2, :, :]

    return flow
