import numpy as np
import torch
torch.manual_seed(2019)
np.random.seed(2019)

import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.functional as F
from config import BATCHNORM_MOMENTUM, BOX_SCALE, IM_SCALE, ModelConfig
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_preds, nms_overlaps
from lib.get_union_boxes import NewUnionBoxesAndFeats, UnionBoxesAndFeats
from lib.fpn.proposal_assignments.new_rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, Result
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_dropout_mask
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
# from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import time
from lib.ggnn import GGNNObj, GGNNRel

MODES = ('sgdet', 'sgcls', 'predcls')
conf = ModelConfig()

def myNNLinear(input_dim, output_dim, bias=True):
    ret_layer = nn.Linear(input_dim, output_dim, bias=bias)
    ret_layer.weight = torch.nn.init.xavier_normal(ret_layer.weight, gain=1.0)
    return ret_layer
#################################################
class DynamicFilterContext(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', use_vision=True,
                 embed_dim=200, hidden_dim=512, obj_dim=2048, pooling_dim=2048,
                 pooling_size=7, dropout_rate=0.2, use_bias=True, use_tanh=True, 
                 limit_vision=True, sl_pretrain=False, num_iter=-1, use_resnet=False,
                 reduce_input=False, debug_type=None, post_nms_thresh=0.5, num_obj_cls=151, 
                 time_step_num=3, output_dim=512, use_knowledge=True, knowledge_matrix=''):
        
        super(DynamicFilterContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        ###########
        self.num_obj_cls = num_obj_cls
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        
		self.ggnn_obj = GGNNObj(num_obj_cls=num_obj_cls, time_step_num=time_step_num, hidden_dim=hidden_dim, 
                                output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)
        ##############################################################      
        assert mode in MODES
        self.mode = mode

        self.output_dict = {}
        
        self.use_vision = use_vision 
        self.use_bias = use_bias
        self.use_tanh = use_tanh
        self.use_highway = True
        self.limit_vision = limit_vision
        
        self.gpu = False 
        self.pooling_dim = pooling_dim 
        self.pooling_size = pooling_size
        self.nms_thresh = post_nms_thresh
        
        self.obj_compress = myNNLinear(self.pooling_dim, self.num_classes, bias=True)

        # self.roi_fmap_obj = load_vgg(pretrained=False).classifier
        
        if not use_resnet:
             roi_fmap_obj = [myNNLinear(512*self.pooling_size*self.pooling_size, 4096, bias=True),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5),
                             myNNLinear(4096, 4096, bias=True),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5)]
             self.roi_fmap_obj = nn.Sequential(*roi_fmap_obj)
        ##### added
        else:
             roi_fmap_obj = [myNNLinear(256*self.pooling_size*self.pooling_size, 2048, bias=True),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5),
                             myNNLinear(2048, 2048, bias=True),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5)]
             self.roi_fmap_obj = nn.Sequential(*roi_fmap_obj)

        if self.use_bias:
            self.freq_bias = FrequencyBias()

        self.reduce_dim = 256
        self.reduce_obj_fmaps = nn.Conv2d(512, self.reduce_dim, kernel_size=1)
        
        similar_fun = [myNNLinear(self.reduce_dim*2, self.reduce_dim),
                       nn.ReLU(inplace=True),
                       myNNLinear(self.reduce_dim, 1)]
        self.similar_fun = nn.Sequential(*similar_fun)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                nn.Linear(self.reduce_dim*self.pooling_size*self.pooling_size, 2048, bias=True),  #1024
                nn.SELU(inplace=True),
                nn.AlphaDropout(p=0.05),
                nn.Linear(2048, 2048),
                nn.SELU(inplace=True),
                nn.AlphaDropout(p=0.05),
            )
        # roi_fmap = [Flattener(),
        #     load_vgg(use_dropout=False, use_relu=False, use_linear=self.pooling_dim == 4096, pretrained=False).classifier,]
        # if self.pooling_dim != 4096:
        #     roi_fmap.append(nn.Linear(4096, self.pooling_dim))
        # self.roi_fmap = nn.Sequential(*roi_fmap)
        else:
            roi_fmap = [Flattener(),
                        nn.Linear(self.reduce_dim*2*self.pooling_size*self.pooling_size, 4096, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096, 4096, bias=True)]
            self.roi_fmap = nn.Sequential(*roi_fmap)

        self.hidden_dim = hidden_dim
        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels)
        self.post_obj = myNNLinear(self.pooling_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)

    ###
    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False
   ###
    def obj_feature_map(self, features, rois):
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return feature_pool
        # return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    @property
    def num_classes(self):
        return len(self.rel_classes)

    @property
    def is_sgdet(self):
        return self.mode == 'sgdet'
    
    @property
    def is_sgcls(self):
        return self.mode == 'sgcls'

    def forward(self, *args, **kwargs):
        
        results = self.base_forward(*args, **kwargs)
        return results
 
    ###############################
    def forward(self, im_inds, obj_fmaps, obj_labels):
        """
        Reason object classes using knowledge of object cooccurrence
        """

        if self.mode == 'predcls':
            # in task 'predcls', there is no need to run GGNN_obj
            obj_dists = Variable(to_onehot(obj_labels.data, self.classes))
            return obj_dists
        else:
            input_ggnn = self.obj_proj(obj_fmaps)

            lengths = []
            for i, s, e in enumerate_by_image(im_inds.data):
                lengths.append(e - s)
            obj_cum_add = np.cumsum([0] + lengths)
            obj_dists = torch.cat([self.ggnn_obj(input_ggnn[obj_cum_add[i] : obj_cum_add[i+1]]) for i in range(len(lengths))], 0)
            return obj_dists

    ###############################
    def base_forward(self, fmaps, obj_logits, im_inds, rel_inds, msg_rel_inds, reward_rel_inds, im_sizes, boxes_priors=None, boxes_deltas=None, boxes_per_cls=None, obj_labels=None):
        assert self.mode == 'sgcls'
        
        num_objs = obj_logits.shape[0]
        num_rels = rel_inds.shape[0]
        temperature = 1    ##0.6
        rois = torch.cat((im_inds[:, None].float(), boxes_priors), 1)
        obj_fmaps = self.obj_feature_map(fmaps, rois)
        reduce_obj_fmaps = self.reduce_obj_fmaps(obj_fmaps)

        S_fmaps = reduce_obj_fmaps[rel_inds[:, 1]]
        O_fmaps = reduce_obj_fmaps[rel_inds[:, 2]]
        
        S_fmaps = F.normalize(S_fmaps, p=2, dim=-1)
        O_fmaps = F.normalize(O_fmaps, p=2, dim=-1)

        if conf.debug_type in ['test1_0']:
            last_SO_fmaps = torch.cat((S_fmaps, O_fmaps), dim=1)
        
        elif conf.debug_type in ['test1_1']:

            S_fmaps_trans = S_fmaps.view(num_rels, self.reduce_dim, self.pooling_size*self.pooling_size).transpose(2, 1)
            O_fmaps_trans = O_fmaps.view(num_rels, self.reduce_dim, self.pooling_size*self.pooling_size).transpose(2, 1)

            pooling_size_sq = self.pooling_size*self.pooling_size
            
             ################################# AdaHan ##########################
            SO_fmaps_extend = torch.cat((S_fmaps_trans.unsqueeze(1).expand(-1, pooling_size_sq, -1, -1), O_fmaps_trans.unsqueeze(2).expand(-1, -1, pooling_size_sq, -1)), dim=-1).view(num_rels, pooling_size_sq*pooling_size_sq, self.redu$            ##[506, 512, 25, 25]
            presence_vector =  self.similar_fun(SO_fmaps_extend)
            presence_vector = presence_vector.view(num_rels, pooling_size_sq, pooling_size_sq)     ##[506, 625, 1]
 
            m_vector = torch.mean(SO_fmaps_extend.transpose(2,1), dim=1).view(-1)    #[506, 25,25]  ##[259072]
            
            latent_mask = torch.where(F.softmax(presence_vector.view(-1), dim=0) >= (1/len(m_vector)), torch.ones_like(presence_vector.view(-1)), torch.zeros_like(presence_vector.view(-1)))
            attended_vector =  m_vector * latent_mask  # zeros out the activations at masked spatial locations
            attended_vector = attended_vector.view(num_rels,pooling_size_sq, pooling_size_sq)
            SO_fmaps_scores = F.softmax(attended_vector, dim=1)


            weighted_S_fmaps = torch.matmul(SO_fmaps_scores.transpose(2, 1), S_fmaps_trans) # (num_rels, 49, 49) x (num_rels, 49, self.reduce_dim)  ##[506, 25, 256]
            last_SO_fmaps = torch.cat((weighted_S_fmaps, O_fmaps_trans), dim=2)   #[506, 25, 512]
            last_SO_fmaps = last_SO_fmaps.transpose(2, 1).contiguous().view(num_rels, self.reduce_dim*2, self.pooling_size, self.pooling_size) ##[506, 512, 5, 5]
#            print('last_SO_fmaps.shape', last_SO_fmaps.shape)   ##[506, 512, 5, 5]
            #######################
            #batch_size = im_inds[-1] + 1
            assert batch_size == 1
            img_level_fmaps = last_SO_fmaps
            img_id = obj_labels  # id_list shape [batch_size]  ##fn is label
            img_obj_bbox = boxes_per_cls # obj bbox  #rois
            self.output_dict = {img_id: {'obj_fmaps':last_SO_fmaps, 'obj_bbox':obj_bbox, 'relation_pair':rel_inds}}
            pickle_out = open("dict.pickle", "wb")
            pickle.dump(self.output_dict, pickle_out)
            pickle_out.close()
            #######################
       # else:
         #   raise ValueError          

        # for object classification
        obj_feats = self.roi_fmap_obj(obj_fmaps.view(rois.size(0), -1))
        #################################modified
        if self.mode == 'predcls':
            obj_logits = self.obj_compress(obj_feats)
            obj_dists = F.softmax(obj_logits, dim=1)
            
        else:
            input_ggnn = self.obj_compress(obj_feats)

            lengths = []
            for i, s, e in enumerate_by_image(im_inds.data):
                lengths.append(e - s)
            obj_cum_add = np.cumsum([0] + lengths)
            obj_dists = torch.cat([self.ggnn_obj(input_ggnn[obj_cum_add[i] : obj_cum_add[i+1]]) for i in range(len(lengths))], 0)
        ##################################
        pred_obj_cls = obj_dists[:, 1:].max(1)[1] + 1

        # for relationship classification
        rel_input = self.roi_fmap(last_SO_fmaps)
        subobj_rep = self.post_obj(obj_feats)
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]

        last_rel_input = self.reduce_rel_input(rel_input)
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2)

        rel_logits = self.rel_compress(triple_rep)

        # follow neural-motifs paper
        if self.use_bias:
            if self.mode in ['sgcls', 'sgdet']:
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        pred_obj_cls[rel_inds[:, 1]],
                        pred_obj_cls[rel_inds[:, 2]],
                        ), 1))
            elif self.mode == 'predcls':
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        obj_labels[rel_inds[:, 1]],
                        obj_labels[rel_inds[:, 2]],
                        ), 1))
            else:
                raise NotImplementedError

      #  return pred_obj_cls, obj_logits, rel_logits, self.gumbel_softmax(logits, temperature=temperature, training=self.training)
        return pred_obj_cls, obj_logits, rel_logits


class RelModelAlign(nn.Module):

     def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048, use_resnet=False, thresh=0.01,
                 use_proposals=False, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True, sl_pretrain=False, eval_rel_objs=False, num_iter=-1, reduce_input=False, 
                 post_nms_thresh=0.5):
        super(RelModelAlign, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        # self.pooling_size = 7
        self.pooling_size = conf.pooling_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim
        ##################
        self.use_ggnn_obj=use_ggnn_obj
        ###################
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

        self.context = DynamicFilterContext(self.classes, self.rel_classes, mode=self.mode,
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
                                            post_nms_thresh=post_nms_thresh,
                                            ########################
                                            obj_dim=self.obj_dim,
                                            time_step_num=ggnn_obj_time_step_num,
                                            hidden_dim=ggnn_obj_hidden_dim,
                                            output_dim=ggnn_obj_output_dim,
                                            use_knowledge=use_obj_knowledge,
                                            knowledge_matrix=obj_knowledge)
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

    

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        
#        if temperature == None:
 #           temperature = self.temperature
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        ######################################
        self.save.dictionary = np.array{'key':fn, 'value'=os.path.join(data_dir), '}
       
        #######################################
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        # boxes = result.boxes_assigned
        boxes_deltas = result.rm_box_deltas # sgcls is None
        boxes_all = result.boxes_all # sgcls is None

        if (self.training) and (result.rel_labels is None):
            import pdb; pdb.set_trace()
            print('debug')
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data, result.rm_.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes, result.rm_obj_dists.data)

        reward_rel_inds = None
        ##########################added
        if self.use_ggnn_obj:          
                result.rm_obj_dists = self.ggnn_obj_reason(im_inds, 
                                                           result.obj_fmap,
                                                           result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        ###################################
        
        if self.mode == 'sgdet':
            msg_rel_inds = self.get_msg_rel_inds(im_inds, boxes, result.rm_obj_dists.data)
            reward_rel_inds = self.get_reward_rel_inds(im_inds, boxes, result.rm_obj_dists.data)
            if self.mode == 'sgdet':
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists, result.obj_preds, result.boxes_all, result.all_rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(), im_inds, rel_inds, msg_rel_inds, 
                                            reward_rel_inds, im_sizes, boxes.detach(), boxes_deltas.detach(), boxes_all.detach(),
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)

        elif self.mode in ['sgcls', 'predcls']:
            result.obj_preds, result.rm_obj_logits, result.rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(),
                                            im_inds, rel_inds, None, None, im_sizes, boxes.detach(), None, None,
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        else:
            raise NotImplementedError

        # result.rm_obj_dists = result.rm_obj_dists_list[-1]

        if self.training:
            return result 
       
        if self.mode == 'predcls':
            import pdb; pdb.set_trace()
            print('debug..')
            result.obj_preds = result.rm_obj_labels
            result.obj_scores = Variable(torch.from_numpy(np.ones(result.obj_preds.shape[0],)).float().cuda())
        else:
            twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
            result.obj_scores = F.softmax(result.rm_obj_logits, dim=1).view(-1)[twod_inds]

        # # Bbox regression
        if self.mode == 'sgdet':
            if conf.use_postprocess:
                bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
            else:
                bboxes = result.rm_box_priors
        else:
            bboxes = result.rm_box_priors
        rel_scores = F.sigmoid(result.rel_logits)

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
            return gather_res(outputs, 0, dim=0), gumbel_softmax(logits, temperature=temperature)
            # return gather_res(outputs, 0, dim=0)
        return outputs