"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = os.path.join(DATA_PATH, 'stanford_filtered')
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

# COCO_PATH = '/home/rowan/datasets/mscoco'
FASTER_RCNN_PATH = 'checkpoints/pretrained/vg-faster-rcnn.tar'
# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
# 256 for sgcls
RELS_PER_IMG_SGCLS = 256
RELS_PER_IMG_SGDET_SL = 128
RELS_PER_IMG_SGDET_RL = 128 # 5000 # a large number
# RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.coco = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = False
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.use_bias = False
        self.limit_vision=False
        self.num_epochs=None
        self.old_feats=False
        self.order=None
        self.det_ckpt=None
        # self.nl_edge=None
        # self.nl_obj=None
        self.hidden_dim=None
        # self.pass_in_obj_feats_to_decoder = None
        # self.pass_in_obj_feats_to_edge = None
        self.pooling_dim = None
        self.rec_dropout = None
        # extra-add
        self.num_iter = None
        self.tensorboard_interval = None
        # self.eval_rel_objs = None
        self.sl_train = None
        self.rl_train = None
        self.sl_rl_test = None
        self.rl_offdropout = None
        self.reduce_input = None
        self.debug_type = None
        self.filte_large = None
        self.use_postprocess = None
        # self.store_det_res = None
        self.load_det_res = None
        self.msg_rm_head = None
        self.use_obj_embed = None
        self.post_nms_thresh = None
        self.overlap_thresh = None
        self.step_obj_dim = None
        self.reward = None
        self.reward_type = None
        # self.baseline_type = None
        self.save_detection_results = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=4)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str,
                            default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='motifnet')
        parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='confidence')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        # parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
        # parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        # parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        # parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-limit_vision', dest='limit_vision',  action='store_true')
        # extra add
        parser.add_argument('-num_iter', dest='num_iter', type=int, default=-1)
        parser.add_argument('-tensorboard_interval', dest='tensorboard_interval', help='print during training', type=int, default=50)
        # parser.add_argument('-eval_rel_objs', dest='eval_rel_objs', action='store_true')
        parser.add_argument('-sl_train', dest='sl_train', action='store_true')
        parser.add_argument('-rl_train', dest='rl_train', action='store_true')
        parser.add_argument('-sl_rl_test', dest='sl_rl_test', action='store_true')
        parser.add_argument('-rl_offdropout', dest='rl_offdropout', action='store_true')
        parser.add_argument('-reduce', dest='reduce_input', action='store_true')
        parser.add_argument('-debug_type', dest='debug_type', type=str)
        parser.add_argument('-filte_large', dest='filte_large', action='store_true', help='filte image with many boxes in RL training set')
        parser.add_argument('-use_postprocess', dest='use_postprocess', action='store_true')
        # parser.add_argument('-store_det_res', dest='store_det_res', action='store_true')
        parser.add_argument('-load_det_res', dest='load_det_res', action='store_true')
        parser.add_argument('-msg_rm_head', dest='msg_rm_head', action='store_true')
        parser.add_argument('-use_obj_embed', dest='use_obj_embed', action='store_true')
        parser.add_argument('-post_nms_thresh', dest='post_nms_thresh', type=float, default=0.5)
        parser.add_argument('-overlap_thresh', dest='overlap_thresh', type=float, default=0.0)
        parser.add_argument('-step_obj_dim', dest='step_obj_dim', type=int, default=100)
        parser.add_argument('-reward', dest='reward', type=str, default='recall')
        parser.add_argument('-reward_type', dest='reward_type', type=int, default=50)
        # parser.add_argument('-baseline_type', dest='baseline_type', type=str, default='counterfactual')
        parser.add_argument('-save_detection_results', dest='save_detection_results', action='store_true')
        return parser
