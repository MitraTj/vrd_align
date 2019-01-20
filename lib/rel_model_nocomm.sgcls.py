"""
https://github.com/zjuchenlong/neural-motifs/blob/master/lib/rel_model_scenedynamic_nocomm.py
"""
import numpy as np
import torch
torch.manual_seed(2018)
np.random.seed(2018)

import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM, BOX_SCALE, IM_SCALE, ModelConfig, RELS_PER_IMG
from lib.fpn.nms.functions.nms import apply_nms

# from lib.lstm.scenedynamic_nocomm_rnn import DecoderRNN
from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_preds
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.new_rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_dropout_mask
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal

MODES = ('sgdet', 'sgcls', 'predcls')
conf = ModelConfig()

class SceneDynamicContext(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', use_vision=True,
                 embed_dim=200, hidden_dim=256, obj_dim=2048, pooling_dim=2048,
                 pooling_size=7, dropout_rate=0.2, use_bias=True, use_tanh=True, 
                 limit_vision=True, sl_pretrain=False, num_iter=-1, use_resnet=False):
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

        self.embed_dim = embed_dim
        self.rel_embed_dim = 100
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.pooling_dim = pooling_dim 
        self.pooling_size = pooling_size
        self.dropout_rate = dropout_rate

        # EMBEDDINGS
        # embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        # self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        # self.obj_embed.weight.data = embed_vecs.clone()

        # embed_vecs2 = obj_edge_vectors(['__START__'] + self.classes, wv_dim=self.embed_dim)
        # self.obj_embed2 = nn.Embedding(len(self.classes), self.embed_dim)
        # self.obj_embed2.weight.data = embed_vecs2.clone()

        # embed_rels = obj_edge_vectors(self.rel_classes, wv_dim=self.rel_embed_dim)
        # self.rel_embed = nn.Embedding(len(self.rel_classes), self.rel_embed_dim) 
        # self.rel_embed.weight.data = embed_rels.clone()

        # self.pos_embed = nn.Sequential(*[
        #     nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
        #     nn.Linear(4, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        # ])

        # self.input_size = self.obj_dim+self.embed_dim+128+self.embed_dim
        self.input_size = self.obj_dim
        # self.input_size = self.obj_dim + self.embed_dim + 128

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)
        if use_resnet:
            self.roi_fmap = nn.Sequential(resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),)
        else:
            roi_fmap = [Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

            if conf.rl_train and conf.mode == 'sgdet':
                # dropout influence the predicted poisition of bbox
                # remove dropout layer
                del self.roi_fmap_obj._modules['5']
                del self.roi_fmap_obj._modules['2']
                del self.roi_fmap[1]._modules['2']

        self.score_fc = nn.Linear(self.hidden_dim, self.num_classes, bias=True)
        self.bbox_fc = nn.Linear(self.hidden_dim, self.num_classes * 4, bias=True)
        # self.score_fc = nn.Linear(self.obj_dim, self.num_classes, bias=True)
        # self.bbox_fc = nn.Linear(self.obj_dim, self.num_classes * 4, bias=True)
        # self.out_lstm = nn.Sequential(*[
        #         nn.Linear(hidden_dim, self.obj_dim, bias=True),
        #         nn.ReLU(inplace=True)
        #     ])
        self.post_lstm = nn.Linear(self.hidden_dim, self.pooling_dim*2)
        self.rel_compress = nn.Linear(self.pooling_dim*3, self.num_rels, bias=True)
        self.post_lstm.weight = torch.nn.init.xavier_normal(self.post_lstm.weight, gain=1.0)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

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

            # simple relation model
            from dataloaders.visual_genome import VG
            from lib.get_dataset_counts import get_counts, box_filter
            fg_matrix, bg_matrix = get_counts(train_data=VG.splits(num_val_im=5000, 
                                                                   filter_non_overlap=True,
                                                                   filter_duplicate_rels=True,
                                                                   use_proposals=False)[0], must_overlap=True)
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix
            # TRYING SOMETHING NEW.
            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            prob_matrix[:,:,0] = 0 # Zero out BG
            self.prob_matrix = prob_matrix

        # object-object message passing
        self.obj_obj_alpha = nn.Sequential(
                nn.Linear(self.pooling_dim*2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1)
            )
        # object-predicate message passing
        self.sub_rel_alpha = nn.Sequential(
                nn.Linear(self.pooling_dim*2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1)
            )
        self.obj_rel_alpha = nn.Sequential(
                nn.Linear(self.pooling_dim*2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1)
            )
        # update object feature
        self.W_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        self.W_sr = nn.Linear(self.pooling_dim, self.pooling_dim)
        self.W_or = nn.Linear(self.pooling_dim, self.pooling_dim)
        # update relation feature
        self.W_rs = nn.Linear(self.pooling_dim, self.pooling_dim)
        self.W_ro = nn.Linear(self.pooling_dim, self.pooling_dim)

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_dim, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_dim, self.hidden_dim])

        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def obj_feature_map(self, features, rois):
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def visual_rep(self, features, rois, pair_inds):
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

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

    def forward(self, fmaps, obj_logits, im_inds, rel_inds, im_sizes, boxes_priors=None, boxes_deltas=None, \
                boxes_per_cls=None, obj_labels=None):

        rois = torch.cat((im_inds[:, None].float(), boxes_priors), 1)
        obj_fmaps = self.obj_feature_map(fmaps, rois)
        # if self.use_vision:
        assert self.use_vision
        timestep_rel_input = self.visual_rep(fmaps, rois, rel_inds[:, 1:])
        # test2_novision
        # timestep_rel_input = Variable(torch.from_numpy(np.zeros((rel_inds.shape[0], self.pooling_dim), dtype=np.float32)).cuda())


        # obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        # pos_embed = self.pos_embed(Variable(center_size(boxes_priors.data)))
        # obj_feats = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)
        obj_feats = obj_fmaps

        batch_size = obj_feats.size(0)
        # initialize lstm
        previous_memory = Variable(obj_feats.data.new().resize_(batch_size, self.hidden_dim).fill_(0))
        previous_state = Variable(obj_feats.data.new().resize_(batch_size, self.hidden_dim).fill_(0))
        # embedding for start
        # previous_embed = self.obj_embed2.weight[0, None].expand(batch_size, self.embed_dim)
        # import pdb; pdb.set_trace()
        if self.dropout_rate > 0.0:
            dropout_mask = get_dropout_mask(self.dropout_rate, previous_memory)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_logits_list = []
        out_label_list = []
        # out_sample_list = []
        out_bbox_list = []
        out_offset_list = []
        # out_rel_list = []
        bbox_input = boxes_priors
        init_logits = obj_logits
        if self.is_sgdet:
            init_offset = boxes_deltas.view(batch_size, -1)

        timestep_obj_input = obj_feats
        for iter_i in range(self.num_iter):
            # timestep_obj_input = torch.cat((obj_feats, previous_embed), 1)
            # timestep_obj_input = obj_feats
            previous_state, previous_memory = self.lstm_equations(timestep_obj_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)
            pred_logits = self.score_fc(previous_state)
            pred_logits = pred_logits + init_logits
            out_logits_list.append(pred_logits)
            pred_dists = F.softmax(pred_logits, dim=1)
            if conf.mode in ['sgcls', 'predcls']:
                pred_cls = pred_dists[:, 1:].max(1)[1] + 1
            elif conf.mode == 'sgdet':
                pred_cls = pred_dists.max(1)[1]
            else:
                raise ValueError

            out_label_list.append(pred_cls)

            if self.is_sgdet:
                pass
                # pred_offset = self.bbox_fc(previous_state)
                # pred_offset = pred_offset + init_offset

                # iter_twod_inds = (torch.arange(batch_size) * self.num_classes).long().cuda() + pred_cls.data
                # pred_cls_offset = pred_offset.view(-1, 4)[iter_twod_inds]
                # refine_bbox = bbox_preds(bbox_input.contiguous(), pred_cls_offset.view(-1, 4))

                # for im_i, s, e in enumerate_by_image(im_inds.data):
                #     h, w = im_sizes[im_i, :2]
                #     refine_bbox[s:e, 0].data.clamp_(min=0, max=w - 1)
                #     refine_bbox[s:e, 1].data.clamp_(min=0, max=h - 1)
                #     refine_bbox[s:e, 2].data.clamp_(min=0, max=w - 1)
                #     refine_bbox[s:e, 3].data.clamp_(min=0, max=h - 1)

                # out_offset_list.append(pred_cls_offset)
                # out_bbox_list.append(refine_bbox)
                # refine_rois = torch.cat((im_inds[:, None].float(), refine_bbox), 1)
                # timestep_obj_input = self.obj_feature_map(fmaps, refine_rois)
        
                # bbox_input = refine_bbox
                # init_offset = pred_offset

            oo_alpha = self.object_object_weights(timestep_obj_input, rel_inds[:, 1:])
            sr_alpha = self.object_predicate_weights(timestep_obj_input, timestep_rel_input, rel_inds[:, 1], 'sub_rel')
            or_alpha = self.object_predicate_weights(timestep_obj_input, timestep_rel_input, rel_inds[:, 2], 'obj_rel')

            # update object feature
            timestep_obj_input = torch.mm(oo_alpha, self.W_o(timestep_obj_input))
            timestep_obj_input.index_add_(0, Variable(rel_inds[:, 1]), sr_alpha * self.W_sr(timestep_rel_input))
            timestep_obj_input.index_add_(0, Variable(rel_inds[:, 2]), or_alpha * self.W_or(timestep_rel_input))
            timestep_obj_input = nn.ReLU(inplace=True)(timestep_obj_input)

            # update relation feature
            timestep_rel_input = nn.ReLU(inplace=True)( timestep_rel_input + \
                self.W_rs(timestep_obj_input[rel_inds[:, 1]]) + self.W_ro(timestep_obj_input[rel_inds[:, 2]])
            )
            init_logits = pred_logits

        # calculate relation logits
        subobj_rep = self.post_lstm(previous_state)
        sub_rep = subobj_rep[rel_inds[:, 1]][:, :self.pooling_dim]
        obj_rep = subobj_rep[rel_inds[:, 2]][:, self.pooling_dim:]
        triple_rep = torch.cat((sub_rep, obj_rep, timestep_rel_input), 1)

        if self.use_tanh:
            triple_rep = F.tanh(triple_rep)
        rel_logits = self.rel_compress(triple_rep)
        if self.use_bias:
            if self.mode in ['sgcls', 'sgdet']:
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        out_label_list[-1][rel_inds[:, 1]],
                        out_label_list[-1][rel_inds[:, 2]],
                        ), 1))
            elif self.mode == 'predcls':
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        obj_labels[rel_inds[:, 1]],
                        obj_labels[rel_inds[:, 2]],
                        ), 1))
            else:
                raise NotImplementedError

        return out_logits_list, out_label_list, None, out_bbox_list, out_offset_list, rel_logits

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

    def get_simple_rel_dist(self, obj_preds, rel_inds):

        obj_preds_np = obj_preds.cpu().numpy()
        rel_inds_np = rel_inds.cpu().numpy()

        rel_dists_list = []
        o1o2 = obj_preds_np[rel_inds_np][:, 1:]
        for o1, o2 in o1o2:
            rel_dists_list.append(self.prob_matrix[o1, o2])

        assert len(rel_dists_list) == len(rel_inds)
        return Variable(torch.from_numpy(np.array(rel_dists_list)).cuda(obj_preds.get_device())) # there is no gradient for this type of code


class RelModelSceneDynamicNocomm(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048, use_resnet=False, thresh=0.01,
                 use_proposals=False, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True, sl_pretrain=False, eval_rel_objs=False, num_iter=-1):

        super(RelModelSceneDynamicNocomm, self).__init__()
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
                                            use_resnet=use_resnet)
    @property
    def num_classes(self):
        return len(self.classes)
    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors, box_score):
        # if self.training or self.eval_rel_objs:
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
            # remove this code can improve the result
            # test1_4
            # if self.mode == 'sgdet':
            #     box_score_np = box_score.cpu().numpy()
            #     norm_box_score_np = np.exp(box_score_np) / (np.exp(box_score_np + 1e-8).sum(1)[:, None])
            #     each_roi_predlabel = np.argmax(norm_box_score_np[:, 1:], 1) + 1
            #     each_roi_predscore = norm_box_score_np[np.arange(norm_box_score_np.shape[0]), each_roi_predlabel]
            #     sub_predscore = each_roi_predscore[rel_cands[:, 0]]
            #     obj_predscore = each_roi_predscore[rel_cands[:, 1]]
            #     rel_cands_idx = np.argsort(sub_predscore*obj_predscore)[-RELS_PER_IMG:]
            #     rel_cands = rel_cands[torch.from_numpy(rel_cands_idx).cuda(rel_cands.get_device())]

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        ########### test2 ######### 
        # filter some detected box with high bg probability
        # filter_rm_obj_dists_np = F.softmax(result.rm_obj_dists.detach(), dim=1).data.cpu().numpy()
        # filter_obj_idx = torch.from_numpy((filter_rm_obj_dists_np[:, 0] < 2/3).nonzero()[0]).cuda()

        # result.im_inds = result.im_inds[filter_obj_idx]
        # result.rm_box_priors = result.rm_box_priors[filter_obj_idx]
        # result.rm_box_deltas = result.rm_box_deltas[filter_obj_idx]
        # result.boxes_all = result.boxes_all[filter_obj_idx]
        # result.rm_obj_dists = result.rm_obj_dists[filter_obj_idx]
        # if self.training:
        #     result.rm_obj_labels = result.rm_obj_labels[filter_obj_idx]
        ###########################

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        # boxes = result.boxes_assigned
        boxes_deltas = result.rm_box_deltas # sgcls is None
        boxes_all = result.boxes_all # sgcls is None

        # if (self.training or self.eval_rel_objs) and (result.rel_labels is None):
        if (self.training) and (result.rel_labels is None):
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data, result.rm_obj_dists.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes, result.rm_obj_dists.data)

        if self.mode == 'sgdet':
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(),
                                            im_inds, rel_inds, im_sizes, boxes.detach(), boxes_deltas.detach(), boxes_all,
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        elif self.mode in ['sgcls', 'predcls']:
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(),
                                            im_inds, rel_inds, im_sizes, boxes.detach(), None, None,
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        else:
            raise NotImplementedError

        # comments to origin
        result.obj_preds = result.obj_preds_list[-1]
        result.rm_obj_dists = result.rm_obj_dists_list[-1]

        if conf.rl_train:
            result.rel_inds = rel_inds 
        # if self.training or self.eval_rel_objs:
        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
        # mode in ['sgcls', 'predcls']
        #     # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors
        # bboxes = result.bbox_list[-1]

        if self.mode == 'predcls':
            result.obj_preds = result.rm_obj_labels
            result.obj_scores = Variable(torch.from_numpy(np.ones(result.obj_preds.shape[0],)).float().cuda())

        # debug
        # rel_scores = result.rel_dists
        # rel_scores = F.softmax(result.rel_dists, dim=1)
        rel_scores = F.sigmoid(result.rel_dists)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_scores)

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