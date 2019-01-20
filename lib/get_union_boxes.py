"""
credits to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/network.py#L91
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
import numpy as np
from torch.nn.modules.module import Module
from torch import nn
from config import BATCHNORM_MOMENTUM, ModelConfig

conf = ModelConfig()

class NewUnionBoxesAndFeats(Module):
    def __init__(self, pooling_size=7, stride=16, dim=256):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(NewUnionBoxesAndFeats, self).__init__()
        
        self.pooling_size = pooling_size
        self.stride = stride
        self.dim = dim

        self.conv = nn.Sequential(
            nn.Conv2d(2, dim //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM),
        )

    def forward(self, fmap, rois, union_inds, use_feats=True):

        union_S_pools, union_O_pools = new_union_boxes(fmap, rois, union_inds, pooling_size=self.pooling_size, stride=self.stride)

        if use_feats:
            pair_rois = torch.cat((rois[union_inds[:, 0]], rois[union_inds[:, 1]]),1).data.cpu().numpy()
            # rects_np = get_rect_features(pair_rois, self.pooling_size*2-1) - 0.5
            rects_np = draw_union_boxes(pair_rois, self.pooling_size*4-1) - 0.5
            rects = Variable(torch.FloatTensor(rects_np).cuda(fmap.get_device()), volatile=fmap.volatile)

            return union_S_pools, union_O_pools, self.conv(rects)
        else:
            return union_S_pools, union_O_pools, None

def new_union_boxes(fmap, rois, union_inds, pooling_size=14, stride=16):

    assert rois.shape[1] == 4
    assert union_inds.shape[1] == 2
    union_rois = torch.cat((
        torch.min(rois[:, :2][union_inds[:, 0]], rois[:, :2][union_inds[:, 1]]),
        torch.max(rois[:, 2:][union_inds[:, 0]], rois[:, 2:][union_inds[:, 1]])
    ),1)

    S_union_rois = torch.cat((Variable(union_inds[:, :1].float()), union_rois), dim=1)
    O_union_rois = torch.cat((Variable(union_inds[:, 1:].float()), union_rois), dim=1)

    union_S_pools = RoIAlignFunction(pooling_size, pooling_size,
                                   spatial_scale=1/stride)(fmap, S_union_rois)
    union_O_pools = RoIAlignFunction(pooling_size, pooling_size,
                                   spatial_scale=1/stride)(fmap, O_union_rois)

    return union_S_pools, union_O_pools



class UnionBoxesAndFeats(Module):
    def __init__(self, pooling_size=7, stride=16, dim=256, concat=False, use_feats=True):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(UnionBoxesAndFeats, self).__init__()
        
        self.pooling_size = pooling_size
        self.stride = stride

        self.dim = dim
        self.use_feats = use_feats

        self.conv = nn.Sequential(
            nn.Conv2d(2, dim //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM),
        )
        self.concat = concat

    def forward(self, fmap, rois, union_inds):

        union_pools = union_boxes(fmap, rois, union_inds, pooling_size=self.pooling_size, stride=self.stride)
        if not self.use_feats:
            return union_pools.detach()

        pair_rois = torch.cat((rois[:, 1:][union_inds[:, 0]], rois[:, 1:][union_inds[:, 1]]),1).data.cpu().numpy()
        # rects_np = get_rect_features(pair_rois, self.pooling_size*2-1) - 0.5
        rects_np = draw_union_boxes(pair_rois, self.pooling_size*4-1) - 0.5
        rects = Variable(torch.FloatTensor(rects_np).cuda(fmap.get_device()), volatile=fmap.volatile)
        if self.concat:
            return torch.cat((union_pools, self.conv(rects)), 1)

        if conf.rl_offdropout:
            self.conv[2].training = False
            self.conv[6].training = False

        return union_pools + self.conv(rects)

def union_boxes(fmap, rois, union_inds, pooling_size=14, stride=16):
    """
    :param fmap: (batch_size, d, IM_SIZE/stride, IM_SIZE/stride)
    :param rois: (num_rois, 5) with [im_ind, x1, y1, x2, y2]
    :param union_inds: (num_urois, 2) with [roi_ind1, roi_ind2]
    :param pooling_size: we'll resize to this
    :param stride:
    :return:
    """
    assert union_inds.size(1) == 2
    im_inds = rois[:,0][union_inds[:,0]]
    assert (im_inds.data == rois.data[:,0][union_inds[:,1]]).sum() == union_inds.size(0)
    union_rois = torch.cat((
        im_inds[:,None],
        torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
        torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
    ),1)

    # (num_rois, d, pooling_size, pooling_size)
    union_pools = RoIAlignFunction(pooling_size, pooling_size,
                                   spatial_scale=1/stride)(fmap, union_rois)
    return union_pools
 
