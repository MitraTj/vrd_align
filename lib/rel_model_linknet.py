
import numpy as np
import torch
torch.manual_seed(2018)
np.random.seed(2018)

import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from config import BATCHNORM_MOMENTUM, BOX_SCALE, IM_SCALE, ModelConfig
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_preds, nms_overlaps
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.new_rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, Result
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_dropout_mask
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import time

MODES = ('sgdet', 'sgcls', 'predcls')
conf = ModelConfig()

class SceneDynamicContext(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', use_vision=True,
                 embed_dim=200, hidden_dim=256, obj_dim=2048, pooling_dim=2048,
                 pooling_size=7, dropout_rate=0.2, use_bias=True, use_tanh=True, 
                 limit_vision=True, sl_pretrain=False, num_iter=-1, use_resnet=False,
                 reduce_input=False, debug_type=None, post_nms_thresh=0.5, step_obj_dim=100):
        super(SceneDynamicContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.use_vision = use_vision 
        self.use_bias = use_bias
        self.use_tanh = use_tanh
        self.use_highway = True
        self.limit_vision = limit_vision
        # self.sl_pretrain = sl_pretrain
        self.num_iter = num_iter

        # self.embed_dim = embed_dim
        # self.obj_embed_dim = 100
        self.obj_embed_dim = 200
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.pooling_dim = pooling_dim 
        self.pooling_size = pooling_size
        self.dropout_rate = dropout_rate
        self.nms_thresh = post_nms_thresh
        # self.reduce = reduce_input
        self.debug_type = debug_type

        self.reduce_dim = 512
        self.step_obj_dim = step_obj_dim

        self.input_size = self.obj_dim + self.step_obj_dim + 512

        # EMBEDDINGS for relation model
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.obj_embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.obj_embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        # for each time step
        step_embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.step_obj_dim)
        self.step_obj_embed = nn.Embedding(self.num_classes, self.step_obj_dim)
        self.step_obj_embed.weight.data = step_embed_vecs.clone()

        if conf.msg_rm_head:
            self.reduce_rel_input = nn.Sequential(
                    nn.Linear(self.reduce_dim, self.reduce_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.reduce_dim, self.reduce_dim)
                )
        else:
            self.reduce_rel_input = nn.Linear(self.pooling_dim, self.reduce_dim)
        self.reduce_last_rel_input = nn.Linear(self.pooling_dim, self.hidden_dim*3)
        self.mapping_x = nn.Linear(self.hidden_dim*2, self.hidden_dim*3)

        if conf.use_obj_embed:
            self.post_lstm = nn.Linear(self.hidden_dim + self.obj_embed_dim, self.hidden_dim*2)
        else:
            self.post_lstm = nn.Linear(self.hidden_dim, self.hidden_dim*2)

        # self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress = nn.Linear(self.hidden_dim*3, self.num_rels, bias=True)

        # object-object message passing
        self.obj_obj_alpha = nn.Sequential(
                    nn.Linear(self.hidden_dim*2, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 1))
        # object-predicate message passing
        self.sub_rel_alpha = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))
        self.obj_rel_alpha = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        # update object feature
        self.W_o = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.W_sr = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.W_or = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.W_rs = nn.Linear(self.hidden_dim, self.reduce_dim)
        self.W_ro = nn.Linear(self.hidden_dim, self.reduce_dim)    

        self.post_lstm.weight = torch.nn.init.xavier_normal(self.post_lstm.weight, gain=1.0)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512,
                                              use_feats=True)

        self.union_boxes_rel = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                      dim=1024 if use_resnet else 512,
                                      use_feats=True)

        if not conf.msg_rm_head:
            roi_fmap = [Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)

        roi_fmap_rel = [Flattener(),
            load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,]
        if pooling_dim != 4096:
            roi_fmap_rel.append(nn.Linear(4096, pooling_dim))
        self.roi_fmap_rel = nn.Sequential(*roi_fmap_rel)

        self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        if conf.rl_offdropout:
            # dropout influence the predicted poisition of bbox
            # remove dropout layer in RL setting
            del self.roi_fmap_obj._modules['5']
            del self.roi_fmap_obj._modules['2']
            del self.roi_fmap_rel[1]._modules['2']
            if not conf.msg_rm_head:
                del self.roi_fmap[1]._modules['2']

        self.score_fc = nn.Linear(self.hidden_dim, self.num_classes, bias=True)
        self.bbox_fc = nn.Linear(self.hidden_dim, self.num_classes * 4, bias=True)
        self.score_fc.weight = torch.nn.init.xavier_normal(self.score_fc.weight, gain=1.0)
        self.bbox_fc.weight = torch.nn.init.xavier_normal(self.bbox_fc.weight, gain=1.0)

        # RNN setting
        if self.use_highway:
            self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_dim, bias=True)
            self.state_linearity = torch.nn.Linear(self.hidden_dim, 5 * self.hidden_dim, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.input_size, 4 * self.hidden_dim, bias=True)
            self.state_linearity = torch.nn.Linear(self.hidden_dim, 4 * self.hidden_dim, bias=True)

        self.reset_parameters()
        if self.use_bias:
            self.freq_bias = FrequencyBias()

        # try linknet
        self.mid_dim = 256

        self.W_W1 = torch.nn.Linear(self.input_size, int(self.input_size/2))
        self.W_U1 = torch.nn.Linear(self.input_size, int(self.input_size/2))
        self.W_H1 = torch.nn.Linear(self.input_size, int(self.input_size/2))
        self.W_W2 = torch.nn.Linear(self.mid_dim, int(self.mid_dim/2))
        self.W_U2 = torch.nn.Linear(self.mid_dim, int(self.mid_dim/2))
        self.W_H2 = torch.nn.Linear(self.mid_dim, int(self.mid_dim/2))

        if conf.debug_type == 'try3':
            self.fc0 = torch.nn.Linear(int(self.input_size/2), self.input_size)
            self.fc1 = torch.nn.Linear(self.input_size, self.mid_dim)
            self.fc2 = torch.nn.Linear(int(self.mid_dim/2), self.mid_dim)
            self.fc3 = torch.nn.Linear(self.mid_dim, len(self.classes))
        elif conf.debug_type == 'try6':
            self.fc0 = nn.Sequential(
                nn.Linear(int(self.input_size/2), self.input_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.input_size, self.input_size))
            self.fc1 = nn.Sequential(
                nn.Linear(self.input_size, self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mid_dim, self.mid_dim))
            self.fc2 = nn.Sequential(
                nn.Linear(int(self.mid_dim/2), self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mid_dim, self.mid_dim))
            self.fc3 = nn.Sequential(
                nn.Linear(self.mid_dim, len(self.classes)),
                nn.ReLU(inplace=True),
                nn.Linear(len(self.classes), len(self.classes)))
        elif conf.debug_type in ['try7']:
            self.fc0 = nn.Sequential(
                nn.Linear(int(self.input_size/2), self.input_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(self.input_size, self.input_size))
            self.fc1 = nn.Sequential(
                nn.Linear(self.input_size, self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(self.mid_dim, self.mid_dim))
            self.fc2 = nn.Sequential(
                nn.Linear(int(self.mid_dim/2), self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(self.mid_dim, self.mid_dim))
            self.fc3 = nn.Sequential(
                nn.Linear(self.mid_dim, len(self.classes)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(len(self.classes), len(self.classes)))            
        else:
            raise NotImplementedError

        # try4
        # self.post_rel = nn.Linear(self.mid_dim + self.obj_embed_dim, self.pooling_dim*2)
        self.post_rel = nn.Linear(self.mid_dim + self.obj_embed_dim, self.hidden_dim*2)


    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_dim, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_dim, self.hidden_dim])

        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def obj_feature_map(self, features, rois):
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    @property
    def num_classes(self):
        return len(self.classes)
    @property
    def num_rels(self):
        return len(self.rel_classes)

    @property
    def is_sgdet(self):
        return self.mode == 'sgdet'
    
    @property
    def is_sgcls(self):
        return self.mode == 'sgcls'

    def object_object_weights(self, obj_feats, rel_inds):
        num_objs = obj_feats.shape[0]
        oo_alpha_init = Variable(torch.zeros((num_objs*num_objs, ))).cuda(obj_feats.get_device())
        sub_obj_feats = torch.cat((obj_feats[rel_inds[:, 0]], obj_feats[rel_inds[:, 1]]), 1)
        u_o = self.obj_obj_alpha(sub_obj_feats)
        twod_inds = rel_inds[:, 0] * num_objs + rel_inds[:, 1]
        oo_alpha = oo_alpha_init.scatter_(0, Variable(twod_inds), u_o[:, 0]).view((num_objs, num_objs))
        oo_alpha_mask = (oo_alpha != 0).float()
        oo_alpha_exp = torch.exp(oo_alpha) * oo_alpha_mask
        norm_oo_alpha = oo_alpha_exp / (oo_alpha_exp.sum(1).unsqueeze(1) + 1e-8)
        norm_oo_alpha += Variable(torch.eye(num_objs)).cuda(obj_feats.get_device())

        # norm_oo_alpha = norm_oo_alpha / norm_oo_alpha.sum(1)
        return norm_oo_alpha

    def object_predicate_weights(self, obj_feats, pred_feats, single_rel_inds, type_name):
        num_objs = obj_feats.shape[0]
        obj_pred_feats = torch.cat((obj_feats[single_rel_inds], pred_feats), 1)
        if type_name == 'sub_rel':
            u = self.sub_rel_alpha(obj_pred_feats)
        elif type_name == 'obj_rel':
            u = self.obj_rel_alpha(obj_pred_feats)
        else:
            raise NotImplementedError
        u_exp_sum = Variable(torch.zeros((num_objs, 1))).cuda(obj_feats.get_device())
        u_exp = torch.exp(u)
        u_exp_sum.index_add_(0, Variable(single_rel_inds), u_exp)
        alpha = u_exp / (u_exp_sum[single_rel_inds] + 1e-8)
        return alpha

    # try linknet
    def forward(self, fmaps, obj_logits, im_inds, rel_inds, msg_rel_inds, reward_rel_inds, im_sizes, boxes_priors=None, \
                boxes_deltas=None, boxes_per_cls=None, obj_labels=None):
        # print(reward_rel_inds.shape[0])
        # import pdb; pdb.set_trace()
        num_objs = obj_logits.shape[0]
        rois = torch.cat((im_inds[:, None].float(), boxes_priors), 1)
        obj_fmaps = self.obj_feature_map(fmaps, rois)

        obj_feats = obj_fmaps
        batch_size = obj_feats.size(0)

        init_logits = obj_logits
        if self.mode == 'predcls':
            init_obj_embed = self.step_obj_embed(obj_labels)
        elif self.mode == 'sgcls':
            init_scores = F.softmax(init_logits, 1)
            init_obj_embed = init_scores @ self.step_obj_embed.weight
        else:
            init_scores = F.softmax(init_logits, 1)
            norm_init_scores = init_scores[:, 1:] / init_scores[:, 1:].sum(1)[:, None]
            init_obj_embed = norm_init_scores @ self.step_obj_embed.weight[1:, :]

        global_feats = fmaps.mean(3).mean(2)
        timestep_obj_input = torch.cat((obj_feats, init_obj_embed, global_feats[im_inds]), 1)

        out_logits_list = []
        out_label_list = []
        out_bbox_list = []
        out_offset_list = []
        for iter_i in range(self.num_iter):
            # import pdb; pdb.set_trace()
            R1_i = torch.mm(self.W_W1(timestep_obj_input), self.W_U1(timestep_obj_input).transpose(1, 0))
            R1_i = F.softmax(R1_i, 1)
            res_timestep_obj_input = self.fc0(torch.mm(R1_i, self.W_H1(timestep_obj_input)))
            timestep_obj_input = timestep_obj_input + res_timestep_obj_input
            O2 = self.fc1(timestep_obj_input)

            R2_i = torch.mm(self.W_W2(O2), self.W_U2(O2).transpose(1, 0))
            R2_i = F.softmax(R2_i, 1)
            O3 = O2 + self.fc2(torch.mm(R2_i, self.W_H2(O2)))
            pred_logits = self.fc3(O3)

            pred_logits = pred_logits + init_logits


            out_logits_list.append(pred_logits)
            pred_dists = F.softmax(pred_logits, dim=1)
            pred_cls = pred_dists[:, 1:].max(1)[1] + 1
            # not add faster rcnn output logits
            out_label_list.append(pred_cls)

            init_logits = pred_logits

        if self.mode == 'sgdet':
            unionbox = self.union_boxes(fmaps, rois, msg_rel_inds[:, 1:])
        else:
            unionbox = self.union_boxes(fmaps, rois, rel_inds[:, 1:])

        # # import pdb; pdb.set_trace()
        # if conf.msg_rm_head:
        #     timestep_rel_input = unionbox.mean(2).mean(2)
        #     timestep_rel_input = self.reduce_rel_input(timestep_rel_input)
        # else:
        #     timestep_rel_input = self.roi_fmap(unionbox)
        #     timestep_rel_input = self.reduce_rel_input(timestep_rel_input)

        unionbox_rel = self.union_boxes_rel(fmaps, rois, rel_inds[:, 1:])
        last_rel_input = self.roi_fmap_rel(unionbox_rel)

        # try4 need add comments
        last_rel_input = self.reduce_last_rel_input(last_rel_input)


        # Do NMS here as a post-processing step
        # if (boxes_per_cls is not None) and (not self.training or conf.rl_train) and (conf.use_postprocess):
        #     print('need debug...')
        #     import pdb; pdb.set_trace()
        #     assert conf.mode == 'sgdet'
        #     if self.training:
        #         assert conf.rl_train
        #     is_overlap = nms_overlaps(boxes_per_cls.data).view(
        #         num_objs, num_objs, self.num_classes).cpu().numpy() > self.nms_thresh

        #     # last time step pred_dists
        #     pred_dists_np = pred_dists.data.cpu().numpy()
        #     pred_dists_np[:, 0] = 0
        #     pred_cls = pred_cls.data.new(len(pred_cls)).fill_(0)
        #     for i in range(pred_cls.size(0)):
        #         box_ind, cls_ind = np.unravel_index(pred_dists_np.argmax(), pred_dists.shape)
        #         pred_cls[int(box_ind)] = int(cls_ind)
        #         pred_dists_np[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        #         pred_dists_np[box_ind] = -1.0 # This way we won't re-sample

        #     pred_cls = Variable(pred_cls)

        # calculate relation logits
        assert self.mode == 'sgcls'
        E0 = torch.cat((O3, self.obj_embed(pred_cls)), 1)
        subobj_rep = self.post_rel(E0)

        # try4
        # sub_rep = subobj_rep[:, :self.pooling_dim][rel_inds[:, 1]]
        # obj_rep = subobj_rep[:, self.pooling_dim:][rel_inds[:, 2]]
        # triple_rep = sub_rep * obj_rep * last_rel_input
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        if self.use_tanh:
            triple_rep = F.tanh(triple_rep)

        rel_logits = self.rel_compress(triple_rep)

        # all_rel_logits is used for calculate reward
        all_rel_logits = None
        if self.use_bias:
            if conf.rl_train and self.training:
                if self.mode == 'sgdet':
                    rel_logits = rel_logits + self.freq_bias.index_with_labels(
                        torch.stack((
                            pred_cls[rel_inds[:, 1]],
                            pred_cls[rel_inds[:, 2]],
                            ), 1))

                    # two union boxes
                    all_unionbox_rel = self.union_boxes_rel(fmaps, rois, reward_rel_inds[:, 1:])
                    all_last_rel_input = self.roi_fmap_rel(all_unionbox_rel)
                    all_last_rel_input = self.reduce_last_rel_input(all_last_rel_input)
                    all_sub_rep = subobj_rep[:, :self.hidden_dim][reward_rel_inds[:, 1]]
                    all_obj_rep = subobj_rep[:, self.hidden_dim:][reward_rel_inds[:, 2]]
                    all_last_obj_input = self.mapping_x(torch.cat((all_sub_rep, all_obj_rep), 1))
                    all_triple_rep = nn.ReLU(inplace=True)(all_last_obj_input + all_last_rel_input) - (all_last_obj_input - all_last_rel_input).pow(2)
                    all_rel_logits = self.rel_compress(all_triple_rep)
                    all_rel_logits = all_rel_logits.detach()
                    # all_rel_logits = all_rel_logits + self.freq_bias.index_with_labels(
                    #     torch.stack((pred_cls[all_rel_inds[:, 1]], pred_cls[all_rel_inds[:, 2]]), 1))
                elif self.mode == 'sgcls':
                    all_rel_logits = rel_logits.detach().clone()
                    rel_logits = rel_logits + self.freq_bias.index_with_labels(
                        torch.stack((
                            pred_cls[rel_inds[:, 1]],
                            pred_cls[rel_inds[:, 2]],
                            ), 1))
                elif self.mode == 'predcls':
                    all_rel_logits = rel_logits.detach().clone()
                    rel_logits = rel_logits + self.freq_bias.index_with_labels(
                        torch.stack((
                            obj_labels[rel_inds[:, 1]],
                            obj_labels[rel_inds[:, 2]],
                            ), 1))

                else:
                    raise ValueError
            else:
                if self.mode in ['sgcls', 'sgdet']:
                    rel_logits = rel_logits + self.freq_bias.index_with_labels(
                        torch.stack((
                            pred_cls[rel_inds[:, 1]],
                            pred_cls[rel_inds[:, 2]],
                            ), 1))
                elif self.mode == 'predcls':
                    rel_logits = rel_logits + self.freq_bias.index_with_labels(
                        torch.stack((
                            obj_labels[rel_inds[:, 1]],
                            obj_labels[rel_inds[:, 2]],
                            ), 1))
                else:
                    raise NotImplementedError

        return out_logits_list, out_label_list, None, out_bbox_list, out_offset_list, rel_logits, \
                pred_cls, boxes_per_cls, all_rel_logits

    # original CVPR submit version
    # def forward(self, fmaps, obj_logits, im_inds, rel_inds, msg_rel_inds, reward_rel_inds, im_sizes, boxes_priors=None, \
    #             boxes_deltas=None, boxes_per_cls=None, obj_labels=None):
    #     # print(reward_rel_inds.shape[0])
    #     import pdb; pdb.set_trace()
    #     num_objs = obj_logits.shape[0]
    #     rois = torch.cat((im_inds[:, None].float(), boxes_priors), 1)
    #     obj_fmaps = self.obj_feature_map(fmaps, rois)

    #     if self.mode == 'sgdet':
    #         unionbox = self.union_boxes(fmaps, rois, msg_rel_inds[:, 1:])
    #     else:
    #         unionbox = self.union_boxes(fmaps, rois, rel_inds[:, 1:])

    #     # import pdb; pdb.set_trace()
    #     if conf.msg_rm_head:
    #         timestep_rel_input = unionbox.mean(2).mean(2)
    #         timestep_rel_input = self.reduce_rel_input(timestep_rel_input)
    #     else:
    #         timestep_rel_input = self.roi_fmap(unionbox)
    #         timestep_rel_input = self.reduce_rel_input(timestep_rel_input)

    #     unionbox_rel = self.union_boxes_rel(fmaps, rois, rel_inds[:, 1:])
    #     last_rel_input = self.roi_fmap_rel(unionbox_rel)
    #     last_rel_input = self.reduce_last_rel_input(last_rel_input)

    #     obj_feats = obj_fmaps
    #     batch_size = obj_feats.size(0)
    #     # initialize lstm
    #     previous_memory = Variable(obj_feats.data.new().resize_(batch_size, self.hidden_dim).fill_(0))
    #     previous_state = Variable(obj_feats.data.new().resize_(batch_size, self.hidden_dim).fill_(0))
    #     if self.dropout_rate > 0.0:
    #         dropout_mask = get_dropout_mask(self.dropout_rate, previous_memory)
    #     else:
    #         dropout_mask = None

    #     # Only accumulating label predictions here, discarding everything else
    #     out_logits_list = []
    #     out_label_list = []
    #     out_bbox_list = []
    #     out_offset_list = []
    #     bbox_input = boxes_priors
    #     init_logits = obj_logits

    #     if self.mode == 'predcls':
    #         init_obj_embed = self.step_obj_embed(obj_labels)
    #     elif self.mode == 'sgcls':
    #         init_scores = F.softmax(init_logits, 1)
    #         init_obj_embed = init_scores @ self.step_obj_embed.weight
    #     else:
    #         init_scores = F.softmax(init_logits, 1)
    #         norm_init_scores = init_scores[:, 1:] / init_scores[:, 1:].sum(1)[:, None]
    #         init_obj_embed = norm_init_scores @ self.step_obj_embed.weight[1:, :]

    #     # import pdb; pdb.set_trace()
    #     timestep_obj_input = torch.cat((obj_feats, init_obj_embed), 1)
    #     for iter_i in range(self.num_iter):

    #         previous_state, previous_memory = self.lstm_equations(timestep_obj_input, previous_state,
    #                                                               previous_memory, dropout_mask=dropout_mask)
    #         pred_logits = self.score_fc(previous_state)
    #         pred_logits = pred_logits + init_logits
    #         out_logits_list.append(pred_logits)
    #         pred_dists = F.softmax(pred_logits, dim=1)
    #         pred_cls = pred_dists[:, 1:].max(1)[1] + 1

    #         out_label_list.append(pred_cls)

    #         if self.mode in ['sgcls', 'predcls']:
    #             oo_alpha = self.object_object_weights(previous_state, rel_inds[:, 1:])
    #             sr_alpha = self.object_predicate_weights(previous_state, timestep_rel_input, rel_inds[:, 1], 'sub_rel')
    #             or_alpha = self.object_predicate_weights(previous_state, timestep_rel_input, rel_inds[:, 2], 'obj_rel')
    #         elif self.mode in ['sgdet']:
    #             oo_alpha = self.object_object_weights(previous_state, reward_rel_inds[:, 1:])

    #             sr_alpha = self.object_predicate_weights(previous_state, timestep_rel_input, msg_rel_inds[:, 1], 'sub_rel')
    #             or_alpha = self.object_predicate_weights(previous_state, timestep_rel_input, msg_rel_inds[:, 2], 'obj_rel')
    #         else:
    #             raise ValueError

    #         timestep_obj_input = torch.mm(oo_alpha, self.W_o(previous_state))

    #         if self.mode in ['sgcls', 'predcls']:
    #             sr_message = sr_alpha * self.W_sr(timestep_rel_input)
    #             or_message = or_alpha * self.W_or(timestep_rel_input)
    #             timestep_obj_input.index_add_(0, Variable(rel_inds[:, 1]), sr_message)
    #             timestep_obj_input.index_add_(0, Variable(rel_inds[:, 2]), or_message)
    #         elif self.mode in ['sgdet']:
    #             sr_message = sr_alpha * self.W_sr(timestep_rel_input)
    #             or_message = or_alpha * self.W_or(timestep_rel_input)
    #             timestep_obj_input.index_add_(0, Variable(msg_rel_inds[:, 1]), sr_message)
    #             timestep_obj_input.index_add_(0, Variable(msg_rel_inds[:, 2]), or_message)
    #         else:
    #             raise ValueError

    #         timestep_obj_input = nn.ReLU(inplace=True)(timestep_obj_input)

    #         if self.mode in ['sgcls', 'predcls']:
    #             # update relation feature
    #             timestep_rel_input = nn.ReLU(inplace=True)( timestep_rel_input + \
    #                 self.W_rs(previous_state[rel_inds[:, 1]]) + self.W_ro(previous_state[rel_inds[:, 2]])
    #             )
    #         elif self.mode in ['sgdet']:
    #             timestep_rel_input = nn.ReLU(inplace=True)( timestep_rel_input + \
    #                 self.W_rs(previous_state[msg_rel_inds[:, 1]]) + self.W_ro(previous_state[msg_rel_inds[:, 2]])
    #             )
    #         else:
    #             raise ValueError

    #         init_logits = pred_logits
    #         if self.mode == 'predcls':
    #             step_obj_embed = init_obj_embed
    #         else:
    #             pred_scores = F.softmax(pred_logits, 1)

    #             if self.mode == 'sgdet':
    #                 norm_pred_scores = pred_scores[:, 1:] / pred_scores[:, 1:].sum(1)[:, None]
    #                 step_obj_embed = norm_pred_scores @ self.step_obj_embed.weight[1:, :]
    #             else:
    #                 step_obj_embed = pred_scores @ self.step_obj_embed.weight

    #         timestep_obj_input = torch.cat((timestep_obj_input, step_obj_embed), 1)

    #     # import pdb; pdb.set_trace()
    #     # Do NMS here as a post-processing step
    #     if (boxes_per_cls is not None) and (not self.training or conf.rl_train) and (conf.use_postprocess):
    #         assert conf.mode == 'sgdet'
    #         if self.training:
    #             assert conf.rl_train
    #         is_overlap = nms_overlaps(boxes_per_cls.data).view(
    #             num_objs, num_objs, self.num_classes).cpu().numpy() > self.nms_thresh

    #         # last time step pred_dists
    #         pred_dists_np = pred_dists.data.cpu().numpy()
    #         pred_dists_np[:, 0] = 0
    #         pred_cls = pred_cls.data.new(len(pred_cls)).fill_(0)
    #         for i in range(pred_cls.size(0)):
    #             box_ind, cls_ind = np.unravel_index(pred_dists_np.argmax(), pred_dists.shape)
    #             pred_cls[int(box_ind)] = int(cls_ind)
    #             pred_dists_np[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
    #             pred_dists_np[box_ind] = -1.0 # This way we won't re-sample

    #         pred_cls = Variable(pred_cls)

    #     # calculate relation logits
    #     if conf.use_obj_embed:
    #         if self.mode == 'sgcls':
    #             previous_state_aug = torch.cat((previous_state, self.obj_embed(pred_cls)), 1)
    #         elif self.mode == 'predcls':
    #             previous_state_aug = torch.cat((previous_state, self.obj_embed(obj_labels)), 1)
    #         else:
    #             raise ValueError
    #         subobj_rep = self.post_lstm(previous_state_aug)
    #     else:
    #         subobj_rep = self.post_lstm(previous_state)

    #     sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
    #     obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]

    #     last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
    #     triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

    #     if self.use_tanh:
    #         triple_rep = F.tanh(triple_rep)

    #     rel_logits = self.rel_compress(triple_rep)

    #     # all_rel_logits is used for calculate reward
    #     all_rel_logits = None
    #     if self.use_bias:
    #         if conf.rl_train and self.training:
    #             if self.mode == 'sgdet':
    #                 rel_logits = rel_logits + self.freq_bias.index_with_labels(
    #                     torch.stack((
    #                         pred_cls[rel_inds[:, 1]],
    #                         pred_cls[rel_inds[:, 2]],
    #                         ), 1))

    #                 # two union boxes
    #                 all_unionbox_rel = self.union_boxes_rel(fmaps, rois, reward_rel_inds[:, 1:])
    #                 all_last_rel_input = self.roi_fmap_rel(all_unionbox_rel)
    #                 all_last_rel_input = self.reduce_last_rel_input(all_last_rel_input)
    #                 all_sub_rep = subobj_rep[:, :self.hidden_dim][reward_rel_inds[:, 1]]
    #                 all_obj_rep = subobj_rep[:, self.hidden_dim:][reward_rel_inds[:, 2]]
    #                 all_last_obj_input = self.mapping_x(torch.cat((all_sub_rep, all_obj_rep), 1))
    #                 all_triple_rep = nn.ReLU(inplace=True)(all_last_obj_input + all_last_rel_input) - (all_last_obj_input - all_last_rel_input).pow(2)
    #                 all_rel_logits = self.rel_compress(all_triple_rep)
    #                 all_rel_logits = all_rel_logits.detach()
    #                 # all_rel_logits = all_rel_logits + self.freq_bias.index_with_labels(
    #                 #     torch.stack((pred_cls[all_rel_inds[:, 1]], pred_cls[all_rel_inds[:, 2]]), 1))
    #             elif self.mode == 'sgcls':
    #                 all_rel_logits = rel_logits.detach().clone()
    #                 rel_logits = rel_logits + self.freq_bias.index_with_labels(
    #                     torch.stack((
    #                         pred_cls[rel_inds[:, 1]],
    #                         pred_cls[rel_inds[:, 2]],
    #                         ), 1))
    #             elif self.mode == 'predcls':
    #                 all_rel_logits = rel_logits.detach().clone()
    #                 rel_logits = rel_logits + self.freq_bias.index_with_labels(
    #                     torch.stack((
    #                         obj_labels[rel_inds[:, 1]],
    #                         obj_labels[rel_inds[:, 2]],
    #                         ), 1))

    #             else:
    #                 raise ValueError
    #         else:
    #             if self.mode in ['sgcls', 'sgdet']:
    #                 rel_logits = rel_logits + self.freq_bias.index_with_labels(
    #                     torch.stack((
    #                         pred_cls[rel_inds[:, 1]],
    #                         pred_cls[rel_inds[:, 2]],
    #                         ), 1))
    #             elif self.mode == 'predcls':
    #                 rel_logits = rel_logits + self.freq_bias.index_with_labels(
    #                     torch.stack((
    #                         obj_labels[rel_inds[:, 1]],
    #                         obj_labels[rel_inds[:, 2]],
    #                         ), 1))
    #             else:
    #                 raise NotImplementedError

    #     return out_logits_list, out_label_list, None, out_bbox_list, out_offset_list, rel_logits, \
    #             pred_cls, boxes_per_cls, all_rel_logits

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_dim:1 * self.hidden_dim] +
                                   projected_state[:, 0 * self.hidden_dim:1 * self.hidden_dim])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_dim:2 * self.hidden_dim] +
                                    projected_state[:, 1 * self.hidden_dim:2 * self.hidden_dim])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_dim:3 * self.hidden_dim] +
                                 projected_state[:, 2 * self.hidden_dim:3 * self.hidden_dim])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_dim:4 * self.hidden_dim] +
                                    projected_state[:, 3 * self.hidden_dim:4 * self.hidden_dim])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_dim:5 * self.hidden_dim] +
                                         projected_state[:, 4 * self.hidden_dim:5 * self.hidden_dim])
            highway_input_projection = projected_input[:, 5 * self.hidden_dim:6 * self.hidden_dim]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

class RelModelCMAT(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048, use_resnet=False, thresh=0.01,
                 use_proposals=False, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True, sl_pretrain=False, eval_rel_objs=False, num_iter=-1, reduce_input=False, 
                 debug_type=None, post_nms_thresh=0.5, step_obj_dim=100):
        super(RelModelCMAT, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.num_iter = num_iter
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.sl_pretrain = sl_pretrain

        if not conf.load_det_res:
            self.detector = ObjectDetector(
                classes=classes,
                mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
                use_resnet=use_resnet,
                thresh=thresh,
                max_per_img=64,
            )

        self.context = SceneDynamicContext(self.classes, self.rel_classes, mode=self.mode,
                                            use_vision=self.use_vision, embed_dim=self.embed_dim, 
                                            hidden_dim=self.hidden_dim, obj_dim=self.obj_dim, 
                                            pooling_dim=self.pooling_dim, pooling_size=self.pooling_size, 
                                            dropout_rate=rec_dropout, 
                                            use_bias=self.use_bias, use_tanh=self.use_tanh,
                                            limit_vision=self.limit_vision,
                                            sl_pretrain = self.sl_pretrain,
                                            num_iter=self.num_iter,
                                            use_resnet=use_resnet,
                                            reduce_input=reduce_input,
                                            debug_type=debug_type,
                                            post_nms_thresh=post_nms_thresh,
                                            step_obj_dim=step_obj_dim)
    @property
    def num_classes(self):
        return len(self.classes)
    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_reward_rel_inds(self, im_inds, box_priors, box_score):

        rel_cands = im_inds.data[:, None] == im_inds.data[None]
        rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

        if self.require_overlap:
            rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                   box_priors.data) > 0)
        rel_cands = rel_cands.nonzero()
        if rel_cands.dim() == 0:
            rel_cands = im_inds.data.new(1, 2).fill_(0)

        rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def get_msg_rel_inds(self, im_inds, box_priors, box_score):

        rel_cands = im_inds.data[:, None] == im_inds.data[None]
        rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

        if self.require_overlap:
            rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                   box_priors.data) > conf.overlap_thresh)
        rel_cands = rel_cands.nonzero()
        if rel_cands.dim() == 0:
            rel_cands = im_inds.data.new(1, 2).fill_(0)

        rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def get_rel_inds(self, rel_labels, im_inds, box_priors, box_score):

        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            # Require overlap in the test stage
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)
            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def plain_forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):

        # import pdb; pdb.set_trace()
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        # boxes = result.boxes_assigned
        boxes_deltas = result.rm_box_deltas # sgcls is None
        boxes_all = result.boxes_all # sgcls is None

        if (self.training) and (result.rel_labels is None):
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data, result.rm_obj_dists.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes, result.rm_obj_dists.data)

        reward_rel_inds = None
        if self.mode == 'sgdet':
            msg_rel_inds = self.get_msg_rel_inds(im_inds, boxes, result.rm_obj_dists.data)
            reward_rel_inds = self.get_reward_rel_inds(im_inds, boxes, result.rm_obj_dists.data)

            # For 1080Ti 11G memory constrain
            if self.training and conf.rl_train and (reward_rel_inds.shape[0] > 2000): # overlap 1.0
                print('Skip')
                return None


        if self.mode == 'sgdet':
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists, result.obj_preds, result.boxes_all, result.all_rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(), im_inds, rel_inds, msg_rel_inds, 
                                            reward_rel_inds, im_sizes, boxes.detach(), boxes_deltas.detach(), boxes_all.detach(),
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)

        elif self.mode in ['sgcls', 'predcls']:
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists, result.obj_preds, result.boxes_all, result.all_rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(),
                                            im_inds, rel_inds, None, None, im_sizes, boxes.detach(), None, None,
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        else:
            raise NotImplementedError

        # comments to origin
        # use nms pred result
        # result.obj_preds = result.obj_preds_list[-1]
        result.rm_obj_dists = result.rm_obj_dists_list[-1]

        if conf.rl_train:
            result.rel_inds = rel_inds
            if self.mode == 'sgdet':
                result.all_rel_inds = reward_rel_inds
            elif self.mode in ['sgcls', 'predcls']:
                result.all_rel_inds = rel_inds
            else:
                raise ValueError

        if self.training:
            return result

        # import pdb; pdb.set_trace()
        if self.mode == 'predcls':
            result.obj_preds = result.rm_obj_labels
            result.obj_scores = Variable(torch.from_numpy(np.ones(result.obj_preds.shape[0],)).float().cuda())
        else:
            twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
            result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # # Bbox regression
        if self.mode == 'sgdet':
            if conf.use_postprocess:
                bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
            else:
                bboxes = result.rm_box_priors
        else:
            bboxes = result.rm_box_priors

        rel_scores = F.sigmoid(result.rel_dists)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_scores)


    def forward(self, *args, **kwargs):
        results = self.plain_forward(*args, **kwargs)
        return results

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs