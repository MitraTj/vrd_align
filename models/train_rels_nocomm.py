
import _init_paths
from dataloaders.new_visual_genome import VGDataLoader, VG, dist2idx
import numpy as np
from torch import optim
import torch
torch.manual_seed(2018)
np.random.seed(2018)

import pandas as pd
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from config import ModelConfig, BOX_SCALE, IM_SCALE # ,  FASTER_RCNN_PATH
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, enumerate_by_image, arange
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.fpn.box_utils import bbox_loss, bbox_preds, bbox_overlaps
from collections import defaultdict
from lib.surgery import filter_dets_np
import _pickle as pkl

conf = ModelConfig()

alpha_obj_loss = 1.0
alpha_rel_loss = 1.0
alpha_neg_entropy = 0.005
reward_type = conf.reward_type

constant_baseline = 0.5
baseline_decay = 0.99

if conf.model == 'sd_nocomm':
    from lib.rel_model_nocomm import RelModelSceneDynamicNocomm as RelModel
elif conf.model == 'motifnet':
    from lib.rel_model import RelModel
else:
    raise ValueError()

if conf.load_det_res:
    assert conf.mode == 'sgdet'
    from dataloaders.sgdet_det_res import SGDET_VG, SGDET_VG_LOADER
    from torch.utils.data import DataLoader

    train, val, test = SGDET_VG.splits(dataset_path='./data/sgdet_det_res')
    if conf.test:
        val = test

    train_loader, val_loader = SGDET_VG_LOADER.splits(train, val, batch_size=conf.batch_size, num_workers=conf.num_workers)

else:
    train, val, test = VG.splits(num_val_im=conf.val_size, 
                              filter_duplicate_rels=False, # default is True
                              use_proposals=conf.use_proposals,
                              filter_non_overlap=conf.mode == 'sgdet', # only filter overlap relation in sgdet mode;
                              )
    if conf.test:
        val = test
    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus,
                                                   return_fn=False)

# # save val gt for evaluation
# # set return_fn=True
# import pdb; pdb.set_trace()
# val_for_eval = {}
# for idx_i in tqdm(range(len(val))):
#     key_part = val.filenames[idx_i].split('/')
#     entry_key = '{}_{}'.format(key_part[-2], key_part[-1].split('.')[0])
#     gt_entry = {
#         'gt_classes': val.gt_classes[idx_i].copy(),
#         'gt_relations': val.relationships[idx_i].copy(),
#         'gt_boxes': val.gt_boxes[idx_i].copy(),
#     }
#     val_for_eval[entry_key] = gt_entry
# save_path = os.path.join('data', 'val_for_eval', 'val_gt.pkl')
# if not os.path.exists(os.path.join('data', 'val_for_eval')):
#     os.makedirs(os.path.join('data', 'val_for_eval',))
# pkl.dump(val_for_eval, open(save_path, 'wb'))


detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet,
                    hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pooling_dim=conf.pooling_dim,
                    # rec_dropout=0, # conf.rec_dropout,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision,
                    sl_pretrain=True,
                    # eval_rel_objs=conf.eval_rel_objs,
                    num_iter=conf.num_iter,
                    reduce_input=conf.reduce_input,
                    debug_type=conf.debug_type,
                    post_nms_thresh=conf.post_nms_thresh,
                    step_obj_dim=conf.step_obj_dim,
                    )

assert conf.save_dir.split('/')[-4] == 'checkpoints'
save_path = os.path.join(conf.save_dir, 'iter{}'.format(conf.num_iter))
tensorboard_path = os.path.join('tfboard', '/'.join(conf.save_dir.split('/')[-3:]), 'iter{}'.format(conf.num_iter))
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
summary_writer = SummaryWriter(tensorboard_path)

if not conf.load_det_res:
    # Freeze the detector
    for n, param in detector.detector.named_parameters():
        param.requires_grad = False

# for both sgdet/sgcls/predcls
if conf.rl_train:
    for n, param in detector.context.union_boxes.named_parameters():
        param.requires_grad = False
    if not conf.msg_rm_head:
        for n, param in detector.context.roi_fmap.named_parameters():
            param.requires_grad = False
        for n, param in detector.context.reduce_rel_input.named_parameters():
            param.requires_grad = False

print(print_para(detector), flush=True)

def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.3,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

def np_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

# import pdb; pdb.set_trace()
ckpt = torch.load(conf.ckpt)
if 'debug' in conf.ckpt:
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1

elif conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
elif conf.ckpt.split('-')[-2].split('/')[-1] == 'sgcls':
    # assert conf.mode == 'sgcls' or (conf.mode == 'sgdet' and conf.sgcls_pretrain == True)
    ##############

    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
elif conf.ckpt.split('-')[-2].split('/')[-1] == 'sgdet':
    assert conf.mode == 'sgdet'
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
    # faster_rcnn_model = torch.load(conf.FASTER_RCNN_PATH)
    # optimistic_restore(detector.detector, faster_rcnn_model['state_dict'])


elif conf.ckpt.split('-')[-2].split('/')[-1] == 'predcls':
    # assert conf.mode == 'predcls'
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
else:
    start_epoch = -1
    if not conf.load_det_res:
        optimistic_restore(detector.detector, ckpt['state_dict'])

    if not conf.msg_rm_head:
        detector.context.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.context.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.context.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.context.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.context.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.context.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.context.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.context.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.context.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.context.roi_fmap_rel[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.context.roi_fmap_rel[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.context.roi_fmap_rel[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.context.roi_fmap_rel[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

detector.cuda()

def train_epoch(epoch_num):

    detector.train()
    tr = []

    start = time.time()
    for b, batch in enumerate(train_loader):

        num_iter = epoch_num * len(train_loader) + b
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        # detector.eval()
        # mAp = val_epoch()
        # detector.train()

        # add some number to summarypath
        if b % conf.tensorboard_interval == 0:
            summary_writer.add_scalar('rel_loss', tr[-1]['rel_loss'], num_iter)
            summary_writer.add_scalar('class_loss', tr[-1]['class_loss'], num_iter)

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)

def train_batch(b, verbose=False):
    # import pdb; pdb.set_trace()
    result = detector[b]
    if result is None:
        return pd.Series({'class_loss': 0.0, 'rel_loss': 0.0, 'total': 0.0})   

    losses = {}

    losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    for iter_i in range(conf.num_iter-1):
        losses['class_loss'] += F.cross_entropy(result.rm_obj_dists_list[iter_i], result.rm_obj_labels)
    losses['class_loss'] /= conf.num_iter

    losses['rel_loss'] = F.binary_cross_entropy_with_logits(result.rel_dists, result.rel_labels[:, 3:].float())
    losses['rel_loss'] *= result.rel_labels[:, 3:].size(1)

    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})

    return res

def train_epoch_rl(epoch_num):

    freq_bias_weights = detector.context.freq_bias.obj_baseline.weight.data.cpu().numpy()
    evaluator = BasicSceneGraphEvaluator.all_modes(spice=(conf.reward == 'spice'))

    detector.train()
    tr = []
    start = time.time()
    
    for b, batch in enumerate(train_loader):
        # # for tracking result
        if conf.mode == 'sgdet':
            if b % 20000 == 0 and b > 0:
            # if b % 5000 == 0 and b > 0:
                torch.save({
                    'epoch': epoch,
                    'iter': b,
                    'state_dict': detector.state_dict(), 
                    #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
                    # 'optimizer': optimizer.state_dict(),
                }, os.path.join(save_path, '{}-{}_{}.tar'.format(conf.mode, epoch, b)))

                detector.eval()
                mAp = val_epoch()
                detector.train()

        num_iter = epoch_num * len(train_loader) + b
        tr.append(train_batch_rl(batch, verbose=b % (conf.print_interval*10) == 0, \
                    evaluator=evaluator, freq_bias_weights=freq_bias_weights)) #b == 0))

        # add some number to summarypath
        if b % conf.tensorboard_interval == 0 and b > 0:
            summary_writer.add_scalar('train_recall', pd.concat(tr[-conf.tensorboard_interval:], axis=0).mean(0), num_iter)

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=0).mean(0)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)

def train_batch_rl(b, verbose=False, evaluator=None, freq_bias_weights=None):
    # import pdb; pdb.set_trace()
    result = detector[b]

    if result is None:
        return pd.Series([0.0])
    losses = {}

    # post-process from calculate recall
    num_classes = len(train.ind_to_classes)
    num_relations = len(train.ind_to_predicates)
    rm_obj_dists = F.softmax(result.rm_obj_dists, dim=1)
    # select top N # N=2
    top_N_obj_dists, top_N_obj_preds = rm_obj_dists[:, 1:].sort(1, descending=True)
    top_N_obj_preds = top_N_obj_preds + 1

    if conf.mode in ['predcls']:
        sample_obj_preds = result.rm_obj_labels
    else:
        sample_obj_preds = (rm_obj_dists + 1e-8)[:, 1:].multinomial(1).squeeze(1) + 1

    sample_obj_preds_np = sample_obj_preds.data.cpu().numpy()
    all_rel_inds_np = result.all_rel_inds.cpu().numpy()
    all_rel_logits_np = result.all_rel_logits.data.cpu().numpy()
    top_N_obj_dists_np = top_N_obj_dists.data.cpu().numpy()
    top_N_obj_preds_np = top_N_obj_preds.data.cpu().numpy()
    rm_obj_dists_np = rm_obj_dists.data.cpu().numpy()

    sample_twod_inds = arange(sample_obj_preds.data) * num_classes + sample_obj_preds.data
    sample_obj_scores = rm_obj_dists.view(-1)[sample_twod_inds]
    sample_twod_inds_np = sample_twod_inds.cpu().numpy()
    sample_obj_scores_np = sample_obj_scores.data.cpu().numpy()

    if conf.mode in ['sgcls', 'predcls']:
        bboxes_np = result.rm_box_priors.data.cpu().numpy()
    elif conf.mode == 'sgdet':
        # import pdb; pdb.set_trace()
        # boxes_all_np  = result.boxes_all.data.cpu().numpy()
        # sample_bboxes_np = boxes_all_np.reshape(-1, 4)[sample_twod_inds_np].reshape(boxes_all_np.shape[0], 4)
        sample_bboxes_np = result.rm_box_priors.data.cpu().numpy()
    else:
        raise ValueError

    if conf.load_det_res:
        batch_img_inds = b[7][:, 0]
    else:
        batch_img_inds = b.gt_classes[:, 0].data
    recall_list = []
    image_recall = []
    select_pred_scores = []

    sample_freq_bias_idx = sample_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + sample_obj_preds_np[all_rel_inds_np[:, 2]]
    sample_rel_scores = result.all_rel_logits + Variable(torch.from_numpy(freq_bias_weights[sample_freq_bias_idx]).cuda(result.all_rel_logits.get_device()))
    sample_rel_scores_np = np_sigmoid(sample_rel_scores.data.cpu().numpy())
    sample_rel_scores_np[:, 0] = 0.0

    for i, gt_obj_s, gt_obj_e in enumerate_by_image(batch_img_inds):
        # clean recall cache
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][20] = []
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][50] = []
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][100] = []
        if conf.reward == 'spice':
            evaluator[conf.mode].result_dict[conf.mode+'_spice'][20] = []
            evaluator[conf.mode].result_dict[conf.mode+'_spice'][50] = []
            evaluator[conf.mode].result_dict[conf.mode+'_spice'][100] = []

        if conf.load_det_res:
            gt_rel_s = (b[8][:, 0] == i).nonzero().min()
            gt_rel_e = (b[8][:, 0] == i).nonzero().max()+1 # the last one is not included      
            pred_obj_s = (result.im_inds == i).nonzero().min().data.cpu().numpy().item()
            pred_obj_e = (result.im_inds == i).nonzero().max().data.cpu().numpy().item()+1 
            pred_rel_s = (result.all_rel_inds[:, 0] == i).nonzero().min()
            pred_rel_e = (result.all_rel_inds[:, 0] == i).nonzero().max()+1

            gt_entry = {
                'gt_classes': b[7][gt_obj_s:gt_obj_e, 1].cpu().numpy().copy(),
                'gt_relations': dist2idx(b[8][gt_rel_s:gt_rel_e, 1:].cpu().numpy().copy()),
                'gt_boxes': b[6][gt_obj_s:gt_obj_e].cpu().numpy().copy() * BOX_SCALE/IM_SCALE ,}            
        else:
            gt_rel_s = (b.gt_rels[:, 0] == i).nonzero().min().data.cpu().numpy().item()
            gt_rel_e = (b.gt_rels[:, 0] == i).nonzero().max().data.cpu().numpy().item()+1 # the last one is not included      
            pred_obj_s = (result.im_inds == i).nonzero().min().data.cpu().numpy().item()
            pred_obj_e = (result.im_inds == i).nonzero().max().data.cpu().numpy().item()+1 
            pred_rel_s = (result.all_rel_inds[:, 0] == i).nonzero().min()
            pred_rel_e = (result.all_rel_inds[:, 0] == i).nonzero().max()+1

            gt_entry = {
                'gt_classes': b.gt_classes[gt_obj_s:gt_obj_e, 1].data.cpu().numpy().copy(),
                'gt_relations': dist2idx(b.gt_rels[gt_rel_s:gt_rel_e, 1:].data.cpu().numpy().copy()),
                'gt_boxes': b.gt_boxes[gt_obj_s:gt_obj_e].data.cpu().numpy().copy() * BOX_SCALE/IM_SCALE ,}
        if conf.mode in ['sgcls', 'predcls']:
            boxes_sample, objs_sample, obj_scores_sample, rels_sample, pred_scores_sample = \
                                                                                filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                sample_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                sample_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                sample_rel_scores_np[pred_rel_s: pred_rel_e])
        elif conf.mode == 'sgdet':
            boxes_sample, objs_sample, obj_scores_sample, rels_sample, pred_scores_sample = \
                                                                                filter_dets_np(sample_bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                sample_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                sample_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                sample_rel_scores_np[pred_rel_s: pred_rel_e])
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError

        pred_sample_entry = {
            'pred_boxes': boxes_sample * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_sample, 'pred_rel_inds': rels_sample,
            'obj_scores': obj_scores_sample, 'rel_scores': pred_scores_sample,}

        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_sample_entry)
        if conf.reward == 'recall':
            recall_sample_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
        elif conf.reward == 'spice':
            recall_sample_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
        else:
            raise ValueError
        image_recall.append(recall_sample_100)

        # counterfactual baseline
        if conf.baseline_type == 'counterfactual':
            if conf.mode == 'sgdet':
                for cf_obj_i  in range(pred_obj_s, pred_obj_e):
                    cf_i_obj_preds_np = sample_obj_preds_np.copy()
                    cf_i_bboxes_np = sample_bboxes_np.copy()

                    ##### first-top ###
                    if top_N_obj_preds_np[cf_obj_i, 0] == cf_i_obj_preds_np[cf_obj_i]:
                        recall_i_0_100 = recall_sample_100
                    else:
                        cf_i_obj_preds_np[cf_obj_i] = top_N_obj_preds_np[cf_obj_i, 0]

                        # cf_i_bboxes_np[cf_obj_i] = boxes_all_np[cf_obj_i][cf_i_obj_preds_np[cf_obj_i]]

                        cf_i_obj_scores_np = sample_obj_scores_np.copy()
                        cf_i_obj_scores_np[cf_obj_i] = top_N_obj_dists_np[cf_obj_i, 0]

                        cf_i_freq_bias_idx = cf_i_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + cf_i_obj_preds_np[all_rel_inds_np[:, 2]]
                        cf_i_rel_scores_np = np_sigmoid(all_rel_logits_np + freq_bias_weights[cf_i_freq_bias_idx])
                        cf_i_rel_scores_np[:, 0] = 0

                        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets_np(cf_i_bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                cf_i_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                cf_i_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                cf_i_rel_scores_np[pred_rel_s: pred_rel_e])
                        pred_i_entry = {
                            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                            'pred_classes': objs_i, 'pred_rel_inds': rels_i,
                            'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
                        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
                        if conf.reward == 'recall':
                            recall_i_0_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
                        elif conf.reward == 'spice':
                            recall_i_0_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
                        else:
                            raise ValueError

                    ### second-top ###
                    if top_N_obj_preds_np[cf_obj_i, 1] == cf_i_obj_preds_np[cf_obj_i]:
                        recall_i_1_100 = recall_sample_100
                    else:
                        cf_i_obj_preds_np[cf_obj_i] = top_N_obj_preds_np[cf_obj_i, 1]

                        # cf_i_bboxes_np[cf_obj_i] = boxes_all_np[cf_obj_i][cf_i_obj_preds_np[cf_obj_i]]

                        cf_i_obj_scores_np = sample_obj_scores_np.copy()
                        cf_i_obj_scores_np[cf_obj_i] = top_N_obj_dists_np[cf_obj_i, 1]

                        cf_i_freq_bias_idx = cf_i_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + cf_i_obj_preds_np[all_rel_inds_np[:, 2]]
                        cf_i_rel_scores_np = np_sigmoid(all_rel_logits_np + freq_bias_weights[cf_i_freq_bias_idx])
                        cf_i_rel_scores_np[:, 0] = 0

                        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets_np(cf_i_bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                cf_i_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                cf_i_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                cf_i_rel_scores_np[pred_rel_s: pred_rel_e])
                        pred_i_entry = {
                            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                            'pred_classes': objs_i, 'pred_rel_inds': rels_i,
                            'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
                        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
                        if conf.reward == 'recall':
                            recall_i_1_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
                        elif conf.reward == 'spice':
                            recall_i_1_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
                        else:
                            raise ValueError

                    ########## bg ###################
                    recall_i_bg_100 = min(recall_i_0_100, recall_i_1_100)

                    # recall_list.append((recall_sample_100, recall_i_0_100, recall_i_1_100))
                    recall_list.append((recall_sample_100, recall_i_0_100, recall_i_1_100, recall_i_bg_100))

            elif conf.mode in ['sgcls', 'predcls']:

                for cf_obj_i  in range(pred_obj_s, pred_obj_e):
                    cf_i_obj_preds_np = sample_obj_preds_np.copy()
                    
                    ##### first-top ###
                    if top_N_obj_preds_np[cf_obj_i, 0] == cf_i_obj_preds_np[cf_obj_i]:
                        recall_i_0_100 = recall_sample_100
                    else:
                        cf_i_obj_preds_np[cf_obj_i] = top_N_obj_preds_np[cf_obj_i, 0]

                        cf_i_obj_scores_np = sample_obj_scores_np.copy()
                        cf_i_obj_scores_np[cf_obj_i] = top_N_obj_dists_np[cf_obj_i, 0]

                        cf_i_freq_bias_idx = cf_i_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + cf_i_obj_preds_np[all_rel_inds_np[:, 2]]
                        cf_i_rel_scores_np = np_sigmoid(all_rel_logits_np + freq_bias_weights[cf_i_freq_bias_idx])
                        cf_i_rel_scores_np[:, 0] = 0

                        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                cf_i_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                cf_i_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                cf_i_rel_scores_np[pred_rel_s: pred_rel_e])
                        pred_i_entry = {
                            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                            'pred_classes': objs_i, 'pred_rel_inds': rels_i,
                            'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
                        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
                        if conf.reward == 'recall':
                            recall_i_0_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
                        elif conf.reward == 'spice':
                            recall_i_0_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
                        else:
                            raise ValueError

                    ### second-top ###
                    if top_N_obj_preds_np[cf_obj_i, 1] == cf_i_obj_preds_np[cf_obj_i]:
                        recall_i_1_100 = recall_sample_100
                    else:
                        cf_i_obj_preds_np[cf_obj_i] = top_N_obj_preds_np[cf_obj_i, 1]

                        cf_i_obj_scores_np = sample_obj_scores_np.copy()
                        cf_i_obj_scores_np[cf_obj_i] = top_N_obj_dists_np[cf_obj_i, 1]

                        cf_i_freq_bias_idx = cf_i_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + cf_i_obj_preds_np[all_rel_inds_np[:, 2]]
                        cf_i_rel_scores_np = np_sigmoid(all_rel_logits_np + freq_bias_weights[cf_i_freq_bias_idx])
                        cf_i_rel_scores_np[:, 0] = 0

                        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                cf_i_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                cf_i_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                cf_i_rel_scores_np[pred_rel_s: pred_rel_e])
                        pred_i_entry = {
                            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                            'pred_classes': objs_i, 'pred_rel_inds': rels_i,
                            'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
                        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
                        if conf.reward == 'recall':
                            recall_i_1_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
                        elif conf.reward == 'spice':
                            recall_i_1_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
                        else:
                            raise ValueError

                    recall_list.append((recall_sample_100, recall_i_0_100, recall_i_1_100))
            else:
                raise ValueError


        elif conf.baseline_type ==  'self_critic':

            greedy_obj_scores_np = top_N_obj_dists_np[:, 0]
            greedy_obj_preds_np = top_N_obj_preds_np[:, 0]

            greedy_freq_bias_idx = greedy_obj_preds_np[all_rel_inds_np[:, 1]] * num_classes + greedy_obj_preds_np[all_rel_inds_np[:, 2]]
            greedy_rel_scores_np = np_sigmoid(all_rel_logits_np + freq_bias_weights[greedy_freq_bias_idx])
            greedy_rel_scores_np[:, 0] = 0.0

            if conf.mode in ['sgcls', 'predcls']:
                boxes_greedy, objs_greedy, obj_scores_greedy, rels_greedy, pred_scores_greedy = \
                                                                                    filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                    greedy_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                    greedy_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                    all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                    greedy_rel_scores_np[pred_rel_s: pred_rel_e])
            elif conf.mode == 'sgdet':
                boxes_greedy, objs_greedy, obj_scores_greedy, rels_greedy, pred_scores_greedy = \
                                                                                    filter_dets_np(sample_bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                    greedy_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                    greedy_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                    all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                    greedy_rel_scores_np[pred_rel_s: pred_rel_e])
            else:
                import pdb; pdb.set_trace()
                raise NotImplementedError


            pred_greedy_entry = {
                'pred_boxes': boxes_greedy * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_greedy, 'pred_rel_inds': rels_greedy,
                'obj_scores': obj_scores_greedy, 'rel_scores': pred_scores_greedy,}

            evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_greedy_entry)
            if conf.reward == 'recall':
                recall_greedy_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][reward_type][-1]
            elif conf.reward == 'spice':
                recall_greedy_100 = evaluator[conf.mode].result_dict[conf.mode+'_spice'][reward_type][-1]
            else:
                raise ValueError

            recall_list += [[recall_sample_100, recall_greedy_100] for i in range(pred_obj_e - pred_obj_s)]

        elif conf.baseline_type ==  'constant':
            global constant_baseline
            constant_baseline = constant_baseline + (1-baseline_decay) * (recall_sample_100-constant_baseline)
            recall_list += [[recall_sample_100, constant_baseline] for i in range(pred_obj_e - pred_obj_s)]

    recall_list_np = np.array(recall_list)

    if conf.baseline_type == 'counterfactual':
        if conf.mode == 'sgdet':
            top_N_label_weights = top_N_obj_dists[:, :2].cpu().data.numpy()
            # add bg
            top_N_label_weights = np.concatenate((top_N_label_weights, rm_obj_dists_np[:, :1]), 1)
            top_N_label_weights_norm = top_N_label_weights / top_N_label_weights.sum(1, keepdims=True)
            reward = recall_list_np[:, 0] - (top_N_label_weights_norm * recall_list_np[:, 1:]).sum(1)
        elif conf.mode in ['sgcls', 'predcls']:
            top_N_label_weights = top_N_obj_dists[:, :2].cpu().data.numpy()
            top_N_label_weights_norm = top_N_label_weights / top_N_label_weights.sum(1, keepdims=True)
            reward = recall_list_np[:, 0] - (top_N_label_weights_norm * recall_list_np[:, 1:]).sum(1)
        else:
            raise ValueError
    elif conf.baseline_type in ['self_critic', 'constant']:
        reward = recall_list_np[:, 0] - recall_list_np[:, 1]


    losses['cf_loss'] = - torch.log(sample_obj_scores + 1e-5) * Variable(torch.from_numpy(reward).cuda().float())

    # rel_loss
    losses['rel_loss'] = F.binary_cross_entropy_with_logits(result.rel_dists, result.rel_labels[:, 3:].float())
    losses['rel_loss'] = alpha_rel_loss * result.rel_labels[:, 3:].size(1) * losses['rel_loss']

    if conf.mode == 'sgdet':
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        losses['class_loss'] *= alpha_obj_loss

    # entropy regularization
    losses['neg_entropy'] = alpha_neg_entropy * (torch.log(rm_obj_dists + 1e-5) * rm_obj_dists).sum(1)

    loss = sum(losses['cf_loss'])
    loss += losses['rel_loss']
    loss += sum(losses['neg_entropy'])
    if conf.mode == 'sgdet':
        loss += losses['class_loss']

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    optimizer.step()

    return pd.Series(image_recall)


start_epoch = -1

def val_epoch():
    # import pdb; pdb.set_trace()
    detector.eval()

    if conf.load_det_res:
        with open('./data/val_for_eval/val_gt.pkl', 'rb') as f:
            val_gt = pkl.load(f)

    if conf.save_detection_results:
        all_save_res = []
    # evaluator = BasicSceneGraphEvaluator.all_modes()
    evaluator = BasicSceneGraphEvaluator.all_modes(spice=(conf.reward == 'spice'))

    for val_b, batch in enumerate(tqdm(val_loader)):
        if conf.load_det_res:
            val_batch(conf.num_gpus * val_b, batch, evaluator, val_gt)
        else:
            if conf.save_detection_results:
                save_res = {}
                gt_entry, pred_entry = val_batch(conf.num_gpus * val_b, batch, evaluator)
                # the relationship in pred_entry actually sort by scores already
                # all_pred_rels_scores = pred_entry['rel_scores'][:, 1:].max(1)
                # all_rels_subj_scores = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 0]]
                # all_rels_obj_scores = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 1]]
                # all_triplets_scores = all_pred_rels_scores * all_rels_subj_scores * all_rels_obj_scores
                all_pred_rel_type = pred_entry['rel_scores'][:, 1:].argmax(1) + 1
                save_res['save_pred_rel_type'] = all_pred_rel_type[:20]
                save_res['save_pred_rel_inds'] = pred_entry['pred_rel_inds'][:20]
                save_res['save_pred_boxes'] = pred_entry['pred_boxes']
                save_res['save_pred_classes'] = pred_entry['pred_classes']
                save_res['save_gt_classes'] = gt_entry['gt_classes']
                save_res['save_gt_relations'] = gt_entry['gt_relations']
                save_res['save_gt_boxes'] = gt_entry['gt_boxes']
                save_res['img_size'] = val[val_b]['img_size']
                save_res['filename'] = val[val_b]['fn']

                all_save_res.append(save_res)

            else:
                val_batch(conf.num_gpus * val_b, batch, evaluator)

    if conf.save_detection_results:
        all_recall20 = evaluator[conf.mode].result_dict[conf.mode + '_recall'][20]
        pkl.dump({'all_save_res': all_save_res, 'all_recall20': all_recall20}, \
                    open('visualization_detect_results.pkl', 'wb'))
        print('Finish Save Results!')

    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][50])


def val_batch(batch_num, b, evaluator, val_gt=None):

    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        if conf.load_det_res:
            fn_list = b[-1]
            assert val_gt is not None
            # exists some duplicated relations in GT
            # gt_entry = {
            #         'gt_classes': (val[batch_num + i]['gt_classes'][:, 1]).copy(),
            #         'gt_relations': dist2idx(val[batch_num + i]['gt_rels'][:, 1:]).copy(),
            #         'gt_boxes': val[batch_num + i]['gt_boxes'].copy(),
            #     }
            # pred_entry = {
            #     'pred_boxes': boxes_i,
            #     'pred_classes': objs_i,
            #     'pred_rel_inds': rels_i,
            #     'obj_scores': obj_scores_i,
            #     'rel_scores': pred_scores_i,  # hack for now.
            # }
            gt_entry = val_gt[fn_list[i]]
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,  # hack for now.
            }
        else:
            gt_entry = {
                'gt_classes': val.gt_classes[batch_num + i].copy(),
                'gt_relations': val.relationships[batch_num + i].copy(),
                'gt_boxes': val.gt_boxes[batch_num + i].copy(),
            }
            # assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,  # hack for now.
            }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

        if conf.save_detection_results:
            return gt_entry, pred_entry


def select_example_epoch():

    freq_bias_weights = detector.context.freq_bias.obj_baseline.weight.data.cpu().numpy()
    evaluator = BasicSceneGraphEvaluator.all_modes()

    detector.eval()
    all_res = []
    start = time.time()
    
    ind_to_classes = val.ind_to_classes
    ind_to_predicates = val.ind_to_predicates
    # import pdb; pdb.set_trace()
    for b, batch in enumerate(tqdm(val_loader)):

        res = select_example_batch(batch, verbose=b % (conf.print_interval*10) == 0, \
                    evaluator=evaluator, freq_bias_weights=freq_bias_weights)
        res['scale'] = val[b]['scale']
        res['img_size'] = val[b]['img_size']
        res['filename'] = val[b]['fn']
        all_res.append(res)
        # if b == 10:
        #     break

    pkl.dump(all_res, open('val_detect_results.pkl', 'wb'))
    print('Finish!')
    # tr_np = np.array(tr)
    # np.savez(open('examples.npy', 'wb'), example=tr_np)
    # import pdb; pdb.set_trace()


def select_example_batch(b, verbose=False, evaluator=None, freq_bias_weights=None):
    """
    need select the different prediction for different steps
    """
    import pdb; pdb.set_trace()
    # assert conf.mode == 'sgcls'
    result = detector[b]

    if result is None:
        return pd.Series([0.0])


    num_classes = len(train.ind_to_classes)
    num_relations = len(train.ind_to_predicates)

    all_obj_scores_list = []
    all_obj_preds_list = []
    for each_obj_dists in result.rm_obj_dists_list:
        each_obj_dists = F.softmax(each_obj_dists, 1)
        each_top_N_obj_dists, each_top_N_obj_preds = each_obj_dists[:, 1:].sort(1, descending=True)
        each_top_N_obj_preds = each_top_N_obj_preds[:, 0] + 1

        each_top_N_obj_preds_np = each_top_N_obj_preds.data.cpu().numpy()
        each_top_N_obj_dists_np = each_top_N_obj_dists.data.cpu().numpy()

        each_obj_scores_np = each_top_N_obj_dists_np[:, 0]
        all_obj_scores_list.append(each_obj_scores_np)
        all_obj_preds_list.append(each_top_N_obj_preds_np)

    bboxes_np = result.rm_box_priors.data.cpu().numpy()
    batch_img_inds = b.gt_classes[:, 0].data


    all_rel_inds_np = result.all_rel_inds.cpu().numpy()
    all_rel_logits_np = result.all_rel_logits.data.cpu().numpy()

    import pdb; pdb.set_trace()
    image_recall = []
    pred_sample_entry_list = []
    for each_obj_preds, each_obj_scores in zip(all_obj_preds_list, all_obj_scores_list):
        sample_freq_bias_idx = each_obj_preds[all_rel_inds_np[:, 1]] * num_classes + each_obj_preds[all_rel_inds_np[:, 2]]
        sample_rel_scores = all_rel_logits_np + freq_bias_weights[sample_freq_bias_idx]
        sample_rel_scores_np = np_sigmoid(sample_rel_scores)
        sample_rel_scores_np[:, 0] = 0.0

        for i, gt_obj_s, gt_obj_e in enumerate_by_image(batch_img_inds):
            # clean recall cache
            evaluator[conf.mode].result_dict[conf.mode+'_recall'][20] = []
            evaluator[conf.mode].result_dict[conf.mode+'_recall'][50] = []
            evaluator[conf.mode].result_dict[conf.mode+'_recall'][100] = []

            gt_rel_s = (b.gt_rels[:, 0] == i).nonzero().min().data.cpu().numpy().item()
            gt_rel_e = (b.gt_rels[:, 0] == i).nonzero().max().data.cpu().numpy().item()+1 # the last one is not included      
            pred_obj_s = (result.im_inds == i).nonzero().min().data.cpu().numpy().item()
            pred_obj_e = (result.im_inds == i).nonzero().max().data.cpu().numpy().item()+1 
            pred_rel_s = (result.all_rel_inds[:, 0] == i).nonzero().min()
            pred_rel_e = (result.all_rel_inds[:, 0] == i).nonzero().max()+1

            gt_entry = {
                'gt_classes': b.gt_classes[gt_obj_s:gt_obj_e, 1].data.cpu().numpy().copy(),
                'gt_relations': dist2idx(b.gt_rels[gt_rel_s:gt_rel_e, 1:].data.cpu().numpy().copy()),
                'gt_boxes': b.gt_boxes[gt_obj_s:gt_obj_e].data.cpu().numpy().copy() * BOX_SCALE/IM_SCALE ,}

            boxes_sample, objs_sample, obj_scores_sample, rels_sample, pred_scores_sample = \
                                                                                filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                each_obj_scores[pred_obj_s: pred_obj_e],
                                                                                each_obj_preds[pred_obj_s: pred_obj_e], 
                                                                                all_rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                sample_rel_scores_np[pred_rel_s: pred_rel_e])

            pred_sample_entry = {
                'pred_boxes': boxes_sample * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_sample, 'pred_rel_inds': rels_sample,
                'obj_scores': obj_scores_sample, 'rel_scores': pred_scores_sample,}

            evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_sample_entry)
            recall_sample_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][20][-1]
            image_recall.append(recall_sample_100)

            save_pred_sample_entry = {
                'pred_boxes': boxes_sample * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_sample, 'pred_rel_inds': rels_sample,
                'pred_predicates': pred_scores_sample.argmax(1)}
            pred_sample_entry_list.append(
                save_pred_sample_entry
                )
        import pdb; pdb.set_trace()

    return {'image_recall': image_recall, 'pred_entry': pred_sample_entry_list,
            'gt_entry': gt_entry}

print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

# import pdb; pdb.set_trace()
# mAp = val_epoch()
# import pdb; pdb.set_trace()

# select_example_epoch()
# import pdb; pdb.set_trace()

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    if conf.sl_train:
        rez = train_epoch(epoch)
    elif conf.rl_train:
        rez = train_epoch_rl(epoch)
    elif conf.sl_rl_test:
        mAp = val_epoch()
        import pdb; pdb.set_trace()  
    else:
        print("You want testing?")
        raise NotImplementedError

    torch.save({
        'epoch': epoch,
        'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
        # 'optimizer': optimizer.state_dict(),
    }, os.path.join(save_path, '{}-{}.tar'.format(conf.mode, epoch)))

    mAp = val_epoch()
    scheduler.step(mAp)
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
        print("exiting training early", flush=True)
        break
