
import numpy as np
import torch
torch.manual_seed(2019)
np.random.seed(2019)

import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from config import BATCHNORM_MOMENTUM, ModelConfig

from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_preds, nms_overlaps
from lib.get_union_boxes import NewUnionBoxesAndFeats, UnionBoxesAndFeats
from lib.object_detector import load_vgg
from lib.pytorch_misc import Flattener
from lib.sparse_targets import FrequencyBias
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import time

MODES = ('sgdet', 'sgcls', 'predcls')
conf = ModelConfig()

def myNNLinear(input_dim, output_dim, bias=True):
    ret_layer = nn.Linear(input_dim, output_dim, bias=bias)
    ret_layer.weight = torch.nn.init.xavier_normal(ret_layer.weight, gain=1.0)
    return ret_layer


class RelInteractionDebug1(nn.Module):
    def __init__(self, num_rels, num_classes):
        super(RelInteractionDebug1, self).__init__()

        self.num_rels = num_rels
        self.pooling_size = 7
        self.pooling_dim = 4096
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                             dim=512,
                                             use_feats=False)

        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        # self.rel_compress = myNNLinear(self.pooling_dim, self.num_rels)

        #### add #####
        self.hidden_dim = conf.hidden_dim

        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels)
        self.post_lstm = myNNLinear(self.hidden_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)


    def forward(self, fmaps, rois, rel_inds, previous_state):

        SO_fmaps_pointwise = self.union_boxes(fmaps, rois, rel_inds[:, 1:])
        rel_input = self.roi_fmap(SO_fmaps_pointwise)

        ##### add #####
        subobj_rep = self.post_lstm(previous_state)
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]

        last_rel_input = self.reduce_rel_input(rel_input)
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        rel_logits = self.rel_compress(triple_rep)
        # rel_logits = self.rel_compress(rel_input)

        return rel_logits

class RelInteractionDebug2(nn.Module):
    def __init__(self, num_rels, num_classes):
        super(RelInteractionDebug2, self).__init__()

        self.num_rels = num_rels
        self.pooling_size = 7
        self.pooling_dim = 4096
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                             dim=512,
                                             use_feats=True) ######## difference compared to RelInteractionDebug1

        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        # self.rel_compress = myNNLinear(self.pooling_dim, self.num_rels)

        #### add #####
        self.hidden_dim = conf.hidden_dim

        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels)
        self.post_lstm = myNNLinear(self.hidden_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)


    def forward(self, fmaps, rois, rel_inds, previous_state):

        # import pdb; pdb.set_trace()
        SO_fmaps_pointwise = self.union_boxes(fmaps, rois, rel_inds[:, 1:])
        rel_input = self.roi_fmap(SO_fmaps_pointwise)

        ##### add #####
        subobj_rep = self.post_lstm(previous_state)
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]

        last_rel_input = self.reduce_rel_input(rel_input)
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        rel_logits = self.rel_compress(triple_rep)
        # rel_logits = self.rel_compress(rel_input)

        return rel_logits

class RelInteractionBaseline(nn.Module):
    def __init__(self, num_rels, num_classes):
        super(RelInteractionBaseline, self).__init__()

        self.num_rels = num_rels
        self.pooling_size = 7
        self.pooling_dim = 4096
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                             dim=512,
                                             use_feats=False)
        
        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)


    def forward(self, fmaps, rois, rel_inds):
        SO_fmaps_pointwise = self.union_boxes(fmaps, rois, rel_inds[:, 1:])
        rel_input = self.roi_fmap(SO_fmaps_pointwise)
        rel_logits = self.rel_compress(rel_input)

        return rel_logits

class RelInteraction(nn.Module):

    def __init__(self, num_rels, num_classes):
        super(RelInteraction, self).__init__()

        self.use_bias = True
        self.pooling_dim = 4096 
        self.pooling_size = 7
        self.num_rels = num_rels
        self.num_classes = num_classes

        self.post_obj = nn.Linear(self.pooling_dim, self.pooling_dim*2)
        self.mapping_x = nn.Linear(self.pooling_dim*2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)

        # if conf.debug_type in ['test5_8', 'test5_8_bn']:
        #     self.obj_compress = nn.Linear(512, self.num_classes, bias=True)
        #     self.obj_compress.weight = torch.nn.init.xavier_normal(self.obj_compress.weight, gain=1.0)
        # else:
        #     self.obj_compress = nn.Linear(self.pooling_dim, self.num_classes, bias=True)
        #     self.obj_compress.weight = torch.nn.init.xavier_normal(self.obj_compress.weight, gain=1.0)

        self.post_obj.weight = torch.nn.init.xavier_normal(self.post_obj.weight, gain=1.0)
        self.mapping_x.weight = torch.nn.init.xavier_normal(self.mapping_x.weight, gain=1.0)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        self.new_union_boxes = NewUnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16, dim=512)

        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        if self.use_bias:
            self.freq_bias = FrequencyBias()

        # # dynamic filter
        if conf.debug_type in ['test5_5', 'test5_5_res', 'test5_5_bn', 'test5_8', 'test5_8_bn']:
            self.mid_dim = 512
        else:
            raise ValueError

        if conf.debug_type in ['test5_5', 'test5_5_res', 'test5_5_bn', 'test5_8', 'test5_8_bn']:
            self.reduce_fmaps = nn.Conv2d(512, self.mid_dim, kernel_size=1)
            self.recover_fmaps = nn.Conv2d(self.mid_dim*2, 512, kernel_size=1)

            self.obj_filter_depthwise_gen = nn.Linear(512, self.mid_dim*3*3, bias=True)
            self.obj_filter_pointwise_gen = nn.Linear(512, self.mid_dim*self.mid_dim, bias=True)
            self.obj_filter_depthwise_gen.weight = torch.nn.init.xavier_normal(self.obj_filter_depthwise_gen.weight, gain=1.0)
            self.obj_filter_pointwise_gen.weight = torch.nn.init.xavier_normal(self.obj_filter_pointwise_gen.weight, gain=1.0)
        else:
            raise ValueError

        if conf.debug_type in ['test5_5_bn', 'test5_8_bn']:
            self.depthwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)
            self.pointwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)

    def forward(self, fmaps, rois, rel_inds, reduce_obj_feats):
        # import pdb; pdb.set_trace()
        # print('chenlong')

        im_inds = rois[:, 0].long()
        num_objs = rois.shape[0]

        obj_filter_depthwise = self.obj_filter_depthwise_gen(reduce_obj_feats)
        obj_filter_pointwise = self.obj_filter_pointwise_gen(reduce_obj_feats)
        obj_filter_depthwise = obj_filter_depthwise.view(num_objs, self.mid_dim, 3, 3)
        obj_filter_pointwise = obj_filter_pointwise.view(num_objs, self.mid_dim, self.mid_dim, 1, 1)

        reduce_fmaps = self.reduce_fmaps(fmaps)
        reduce_fmaps = nn.ReLU(inplace=True)(reduce_fmaps) # add after test5_5
        extend_fmaps = reduce_fmaps[im_inds]

        fmaps_h, fmaps_w = fmaps.shape[2], fmaps.shape[3]
        fmaps_depthwise = F.conv2d(extend_fmaps.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_depthwise.view(num_objs*self.mid_dim, 1, 3, 3), groups=num_objs*self.mid_dim, padding=1)
        fmaps_depthwise = fmaps_depthwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        if conf.debug_type in ['test5_5_bn', 'test5_8_bn']:
            fmaps_depthwise = self.depthwise_bn(fmaps_depthwise)

        fmaps_pointwise = F.conv2d(fmaps_depthwise.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_pointwise.view(num_objs*self.mid_dim, self.mid_dim, 1, 1), groups=num_objs)
        fmaps_pointwise = fmaps_pointwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        if conf.debug_type in ['test5_5_bn', 'test5_8_bn']:
            fmaps_pointwise = self.pointwise_bn(fmaps_pointwise)
        fmaps_pointwise = nn.ReLU(inplace=True)(fmaps_pointwise)

        if conf.debug_type in ['test5_5_res']:
            fmaps_pointwise = fmaps_pointwise + extend_fmaps

        S_fmaps_pointwise, O_fmaps_pointwise, spatial_fmaps = self.new_union_boxes(fmaps_pointwise, rois[:, 1:], rel_inds[:, 1:], use_feats=False)

        SO_fmaps_pointwise = torch.cat((S_fmaps_pointwise, O_fmaps_pointwise), dim=1)
        SO_fmaps_pointwise = self.recover_fmaps(SO_fmaps_pointwise)

        # rel_input = self.roi_fmap(SO_fmaps_pointwise + spatial_fmaps)
        rel_input = self.roi_fmap(SO_fmaps_pointwise)


        # get obj_input
        # subobj_rep = self.post_obj(obj_feats)
        # sub_rep = subobj_rep[:, :self.pooling_dim][rel_inds[:, 1]]
        # obj_rep = subobj_rep[:, self.pooling_dim:][rel_inds[:, 2]]
        # obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))

        # triple_rep = nn.ReLU(inplace=True)(obj_input + rel_input) - (obj_input - rel_input).pow(2)

        # rel_logits = self.rel_compress(triple_rep)
        rel_logits = self.rel_compress(rel_input)

        # if self.use_bias:
        #     if self.mode in ['sgcls', 'sgdet']:
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 pred_obj_cls[rel_inds[:, 1]],
        #                 pred_obj_cls[rel_inds[:, 2]],
        #                 ), 1))
        #     elif self.mode == 'predcls':
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 obj_labels[rel_inds[:, 1]],
        #                 obj_labels[rel_inds[:, 2]],
        #                 ), 1))
        #     else:
        #         raise NotImplementedError

        return rel_logits


class RelInteractionDebug3(nn.Module):

    def __init__(self, num_rels, num_classes):
        super(RelInteractionDebug3, self).__init__()

        self.use_bias = True
        self.pooling_dim = 4096 
        self.pooling_size = 7
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.hidden_dim = 512

        self.post_obj = myNNLinear(self.hidden_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels, bias=True)

        # if conf.debug_type in ['test5_8_bn', 'test6_2', 'test6_2_spatial']:
        #     self.obj_compress = myNNLinear(512, self.num_classes, bias=True)
        # else:
        #     self.obj_compress = myNNLinear(self.pooling_dim, self.num_classes, bias=True)

        self.new_union_boxes = NewUnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16, dim=512)

        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        if self.use_bias:
            self.freq_bias = FrequencyBias()

        # # dynamic filter
        if conf.debug_type in ['test5_8_bn', 'test6_2', 'test6_2_spatial']:
            self.mid_dim = 512
        else:
            raise ValueError

        if conf.debug_type in ['test5_8_bn', 'test6_2', 'test6_2_spatial']:
            self.reduce_fmaps = nn.Conv2d(512, self.mid_dim, kernel_size=1)
            self.recover_fmaps = nn.Conv2d(self.mid_dim*2, 512, kernel_size=1)

            self.obj_filter_depthwise_gen = myNNLinear(512, self.mid_dim*3*3, bias=True)
            self.obj_filter_pointwise_gen = myNNLinear(512, self.mid_dim*self.mid_dim, bias=True)
        else:
            raise ValueError

        if conf.debug_type in ['test6_2', 'test6_2_spatial']:
            self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)


        self.depthwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)
        self.pointwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)

    def forward(self, fmaps, rois, rel_inds, reduce_obj_feats):
        # import pdb; pdb.set_trace()
        # print('chenlong')

        im_inds = rois[:, 0].long()
        num_objs = rois.shape[0]

        obj_filter_depthwise = self.obj_filter_depthwise_gen(reduce_obj_feats)
        obj_filter_pointwise = self.obj_filter_pointwise_gen(reduce_obj_feats)
        obj_filter_depthwise = obj_filter_depthwise.view(num_objs, self.mid_dim, 3, 3)
        obj_filter_pointwise = obj_filter_pointwise.view(num_objs, self.mid_dim, self.mid_dim, 1, 1)

        reduce_fmaps = self.reduce_fmaps(fmaps)
        reduce_fmaps = nn.ReLU(inplace=True)(reduce_fmaps) # add after test5_5
        extend_fmaps = reduce_fmaps[im_inds]

        fmaps_h, fmaps_w = fmaps.shape[2], fmaps.shape[3]
        fmaps_depthwise = F.conv2d(extend_fmaps.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_depthwise.view(num_objs*self.mid_dim, 1, 3, 3), groups=num_objs*self.mid_dim, padding=1)
        fmaps_depthwise = fmaps_depthwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        fmaps_depthwise = self.depthwise_bn(fmaps_depthwise)

        fmaps_pointwise = F.conv2d(fmaps_depthwise.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_pointwise.view(num_objs*self.mid_dim, self.mid_dim, 1, 1), groups=num_objs)
        fmaps_pointwise = fmaps_pointwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        fmaps_pointwise = self.pointwise_bn(fmaps_pointwise)
        fmaps_pointwise = nn.ReLU(inplace=True)(fmaps_pointwise)

        if conf.debug_type in ['test6_2']:
            S_fmaps_pointwise, O_fmaps_pointwise, spatial_fmaps = self.new_union_boxes(fmaps_pointwise, rois[:, 1:], rel_inds[:, 1:], use_feats=False)
        elif conf.debug_type in ['test6_2_spatial']:
            S_fmaps_pointwise, O_fmaps_pointwise, spatial_fmaps = self.new_union_boxes(fmaps_pointwise, rois[:, 1:], rel_inds[:, 1:], use_feats=True)
        else:
            raise ValueError

        SO_fmaps_pointwise = torch.cat((S_fmaps_pointwise, O_fmaps_pointwise), dim=1)
        SO_fmaps_pointwise = self.recover_fmaps(SO_fmaps_pointwise)

        if conf.debug_type in ['test6_2']:
            rel_input = self.roi_fmap(SO_fmaps_pointwise)
        elif conf.debug_type in ['test6_2_spatial']:
            rel_input = self.roi_fmap(SO_fmaps_pointwise + spatial_fmaps)
        else:
            raise Variable

        # get obj_input
        subobj_rep = self.post_obj(reduce_obj_feats)
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        last_rel_input = self.reduce_rel_input(rel_input)
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        rel_logits = self.rel_compress(triple_rep)
        # rel_logits = self.rel_compress(rel_input)


        # if self.use_bias:
        #     if self.mode in ['sgcls', 'sgdet']:
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 pred_obj_cls[rel_inds[:, 1]],
        #                 pred_obj_cls[rel_inds[:, 2]],
        #                 ), 1))
        #     elif self.mode == 'predcls':
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 obj_labels[rel_inds[:, 1]],
        #                 obj_labels[rel_inds[:, 2]],
        #                 ), 1))
        #     else:
        #         raise NotImplementedError

        return rel_logits

class RelInteractionDebug4(nn.Module):

    def __init__(self, num_rels, num_classes):
        super(RelInteractionDebug4, self).__init__()

        self.use_bias = True
        self.pooling_dim = 4096 
        self.pooling_size = 7
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.hidden_dim = 512

        self.post_obj = myNNLinear(self.hidden_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels, bias=True)

        self.new_union_boxes = NewUnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16, dim=512)

        roi_fmap = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        if self.pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)

        if self.use_bias:
            self.freq_bias = FrequencyBias()

        # # dynamic filter
        self.mid_dim = 512

        self.reduce_fmaps = nn.Conv2d(512, self.mid_dim, kernel_size=1)
        self.recover_fmaps = nn.Conv2d(self.mid_dim*2, 512, kernel_size=1)

        self.obj_filter_depthwise_gen = myNNLinear(512, self.mid_dim*3*3, bias=True)
        self.obj_filter_pointwise_gen = myNNLinear(512, self.mid_dim*self.mid_dim, bias=True)

        self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)

        self.depthwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)
        self.pointwise_bn = nn.BatchNorm2d(512, eps=1e-5, momentum=BATCHNORM_MOMENTUM)

    def forward(self, fmaps, rois, rel_inds, reduce_obj_feats, last_obj_feats):
        # import pdb; pdb.set_trace()
        # print('chenlong')

        im_inds = rois[:, 0].long()
        num_objs = rois.shape[0]

        obj_filter_depthwise = self.obj_filter_depthwise_gen(reduce_obj_feats)
        obj_filter_pointwise = self.obj_filter_pointwise_gen(reduce_obj_feats)
        obj_filter_depthwise = obj_filter_depthwise.view(num_objs, self.mid_dim, 3, 3)
        obj_filter_pointwise = obj_filter_pointwise.view(num_objs, self.mid_dim, self.mid_dim, 1, 1)

        reduce_fmaps = self.reduce_fmaps(fmaps)
        reduce_fmaps = nn.ReLU(inplace=True)(reduce_fmaps) # add after test5_5
        extend_fmaps = reduce_fmaps[im_inds]

        fmaps_h, fmaps_w = fmaps.shape[2], fmaps.shape[3]
        fmaps_depthwise = F.conv2d(extend_fmaps.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_depthwise.view(num_objs*self.mid_dim, 1, 3, 3), groups=num_objs*self.mid_dim, padding=1)
        fmaps_depthwise = fmaps_depthwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        fmaps_depthwise = self.depthwise_bn(fmaps_depthwise)

        fmaps_pointwise = F.conv2d(fmaps_depthwise.view(1, num_objs*self.mid_dim, fmaps_h, fmaps_w), obj_filter_pointwise.view(num_objs*self.mid_dim, self.mid_dim, 1, 1), groups=num_objs)
        fmaps_pointwise = fmaps_pointwise.view(num_objs, self.mid_dim, fmaps_h, fmaps_w)
        fmaps_pointwise = self.pointwise_bn(fmaps_pointwise)
        fmaps_pointwise = nn.ReLU(inplace=True)(fmaps_pointwise)

        if conf.debug_type in ['test6_2', 'test6_3', 'test6_4', 'test6_5', 'test6_6']:
            S_fmaps_pointwise, O_fmaps_pointwise, spatial_fmaps = self.new_union_boxes(fmaps_pointwise, rois[:, 1:], rel_inds[:, 1:], use_feats=False)
        elif conf.debug_type in ['test6_2_spatial']:
            S_fmaps_pointwise, O_fmaps_pointwise, spatial_fmaps = self.new_union_boxes(fmaps_pointwise, rois[:, 1:], rel_inds[:, 1:], use_feats=True)
        else:
            raise ValueError

        SO_fmaps_pointwise = torch.cat((S_fmaps_pointwise, O_fmaps_pointwise), dim=1)
        SO_fmaps_pointwise = self.recover_fmaps(SO_fmaps_pointwise)

        if conf.debug_type in ['test6_2', 'test6_3', 'test6_4', 'test6_5', 'test6_6']:
            rel_input = self.roi_fmap(SO_fmaps_pointwise)
        elif conf.debug_type in ['test6_2_spatial']:
            rel_input = self.roi_fmap(SO_fmaps_pointwise + spatial_fmaps)
        else:
            raise Variable

        # get obj_input
        subobj_rep = self.post_obj(last_obj_feats)
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        last_rel_input = self.reduce_rel_input(rel_input)
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        rel_logits = self.rel_compress(triple_rep)
        # rel_logits = self.rel_compress(rel_input)


        # if self.use_bias:
        #     if self.mode in ['sgcls', 'sgdet']:
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 pred_obj_cls[rel_inds[:, 1]],
        #                 pred_obj_cls[rel_inds[:, 2]],
        #                 ), 1))
        #     elif self.mode == 'predcls':
        #         rel_logits = rel_logits + self.freq_bias.index_with_labels(
        #             torch.stack((
        #                 obj_labels[rel_inds[:, 1]],
        #                 obj_labels[rel_inds[:, 2]],
        #                 ), 1))
        #     else:
        #         raise NotImplementedError

        return rel_logits