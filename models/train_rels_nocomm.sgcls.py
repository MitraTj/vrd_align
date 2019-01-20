"""
https://github.com/zjuchenlong/neural-motifs/blob/master/models/train_rels_nocomm.py
"""
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
from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, enumerate_by_image, arange
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
# from lib.evaluation.sg_eval_for_train import get_recall_reward
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.fpn.box_utils import bbox_loss, bbox_preds, bbox_overlaps
from collections import defaultdict
from lib.surgery import filter_dets_np

conf = ModelConfig()
if conf.model == 'sd_nocomm':
    from lib.rel_model_nocomm import RelModelSceneDynamicNocomm as RelModel
else:
    raise ValueError()

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
                                               num_gpus=conf.num_gpus)


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
                    num_iter=conf.num_iter
                    )

assert conf.save_dir.split('/')[-4] == 'checkpoints'
save_path = os.path.join(conf.save_dir, 'iter{}'.format(conf.num_iter))
tensorboard_path = os.path.join('tfboard', '/'.join(conf.save_dir.split('/')[-3:]), 'iter{}'.format(conf.num_iter))
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
summary_writer = SummaryWriter(tensorboard_path)

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)

def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]
    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
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
elif conf.ckpt.split('-')[-2].split('/')[-1] == 'predcls':
    assert conf.mode == 'predcls'
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.context.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.context.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.context.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.context.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.context.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.context.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.context.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.context.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

detector.cuda()


def train_epoch(epoch_num):
    # import pdb; pdb.set_trace()
    detector.train()
    tr = []

    start = time.time()
    for b, batch in enumerate(train_loader):

        # if b % 5000 == 0 and b > 0:
        #     detector.eval()
        #     mAp = val_epoch()
        #     detector.train()

        num_iter = epoch_num * len(train_loader) + b
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        # add some number to summarypath
        if b % conf.tensorboard_interval == 0:
            summary_writer.add_scalar('rel_loss', tr[-1]['rel_loss'], num_iter)
            # summary_writer.add_scalar('bbox_loss', tr[-1]['bbox_loss'], num_iter)
            if conf.mode in ['sgdet', 'sgcls']:
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
    result = detector[b]
    losses = {}

    if conf.mode in ['sgdet', 'sgcls']:
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        for iter_i in range(conf.num_iter-1):
            losses['class_loss'] += F.cross_entropy(result.rm_obj_dists_list[iter_i], result.rm_obj_labels)
        losses['class_loss'] /= conf.num_iter

    # train_valid_inds = (result.rm_obj_labels.data > 0).nonzero().squeeze(1)
    # prior_boxes = result.rm_box_priors[train_valid_inds]
    # boxes_deltas = result.offset_list[-1][train_valid_inds]
    # gt_boxes = result.rm_gt_boxes[train_valid_inds]

    # fg_cnt = train_valid_inds.size(0)
    # bg_cnt = result.rm_obj_labels.data.size(0) - fg_cnt

    # losses['bbox_loss'] = bbox_loss(prior_boxes, boxes_deltas, gt_boxes) * 8 * (fg_cnt / (fg_cnt + bg_cnt + 1e-8))

    losses['rel_loss'] = F.binary_cross_entropy_with_logits(result.rel_dists, result.rel_labels[:, 3:].float())
    losses['rel_loss'] *= result.rel_labels[:, 3:].size(1)

    # hard negative
    # follow the function of F.binary_cross_entropy_with_logits
    # rel_input = result.rel_dists.detach()
    # rel_target = result.rel_labels[:, 3:].float().detach()
    # max_val = (-rel_input).clamp(min=0)
    # temp_loss = rel_input - rel_input * rel_target + max_val + ((-max_val).exp() + (-rel_input - max_val).exp()).log()
    # # the hardest 128 relation
    # hard_idx = (temp_loss.sum(1)).sort(0)[1][-128:]

    # losses['rel_loss'] = F.binary_cross_entropy_with_logits(result.rel_dists[hard_idx], result.rel_labels[hard_idx][:, 3:].float())
    # losses['rel_loss'] *= result.rel_labels[:, 3:].size(1)

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
    evaluator = BasicSceneGraphEvaluator.all_modes()

    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        # for tracking result
        # if b % 15000 == 0:
        #     detector.eval()
        #     mAp = val_epoch()
        #     detector.train()

        num_iter = epoch_num * len(train_loader) + b
        tr.append(train_batch_rl(batch, verbose=b % (conf.print_interval*10) == 0, evaluator=evaluator)) #b == 0))

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

def train_batch_rl(b, verbose=False, evaluator=None):

    result = detector[b]

    # post-process from calculate recall
    num_classes = len(train.ind_to_classes)
    rm_obj_dists = F.softmax(result.rm_obj_dists, dim=1)

    # select top N # N=2
    top_N_obj_dists, top_N_obj_preds = rm_obj_dists[:, 1:].sort(1, descending=True)
    top_N_obj_preds = top_N_obj_preds + 1

    greedy_obj_preds = result.obj_preds
    greedy_twod_inds = arange(greedy_obj_preds.data) * num_classes + greedy_obj_preds.data
    greedy_obj_scores = rm_obj_dists.view(-1)[greedy_twod_inds]

    if conf.mode in ['sgcls', 'predcls']:
        bboxes = result.rm_box_priors
    elif conf.mode == 'sgdet':
        bboxes = result.boxes_all.view(-1, 4)[greedy_twod_inds].view(result.boxes_all.size(0), 4)
    else:
        raise ValueError

    rel_scores = F.sigmoid(result.rel_dists)

    # change to np
    bboxes_np = bboxes.data.cpu().numpy()
    greedy_obj_scores_np = greedy_obj_scores.data.cpu().numpy()
    greedy_obj_preds_np = greedy_obj_preds.data.cpu().numpy()
    rel_inds_np = result.rel_inds.cpu().numpy()
    rel_scores_np = rel_scores.data.cpu().numpy()    

    batch_img_inds = b.gt_classes[:, 0].data
    recall_list = []
    image_recall = []
    for i, gt_obj_s, gt_obj_e in enumerate_by_image(batch_img_inds):
        # clean recall cache
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][20] = []
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][50] = []
        evaluator[conf.mode].result_dict[conf.mode+'_recall'][100] = []
        gt_rel_s = (b.gt_rels[:, 0] == i).nonzero().min().data.cpu().numpy().item()
        gt_rel_e = (b.gt_rels[:, 0] == i).nonzero().max().data.cpu().numpy().item()+1 # the last one is not included      
        pred_obj_s = (result.im_inds == i).nonzero().min().data.cpu().numpy().item()
        pred_obj_e = (result.im_inds == i).nonzero().max().data.cpu().numpy().item()+1 
        pred_rel_s = (result.rel_inds[:, 0] == i).nonzero().min()
        pred_rel_e = (result.rel_inds[:, 0] == i).nonzero().max()+1

        gt_entry = {
            'gt_classes': b.gt_classes[gt_obj_s:gt_obj_e, 1].data.cpu().numpy().copy(),
            'gt_relations': dist2idx(b.gt_rels[gt_rel_s:gt_rel_e, 1:].data.cpu().numpy().copy()),
            'gt_boxes': b.gt_boxes[gt_obj_s:gt_obj_e].data.cpu().numpy().copy() * BOX_SCALE/IM_SCALE ,}
        if conf.mode in ['sgcls', 'sgdet']:
            boxes_greedy, objs_greedy, obj_scores_greedy, rels_greedy, pred_scores_greedy = \
                                                                                filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                                greedy_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                                greedy_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                                rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                                rel_scores_np[pred_rel_s: pred_rel_e])
        elif conf.mode == 'predcls':
            import pdb; pdb.set_trace()
            pass
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError

        pred_greedy_entry = {
            'pred_boxes': boxes_greedy * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_greedy, 'pred_rel_inds': rels_greedy,
            'obj_scores': obj_scores_greedy, 'rel_scores': pred_scores_greedy,}
        evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_greedy_entry)
        recall_greedy_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][100][-1]
        image_recall.append(recall_greedy_100)
        # counterfactual baseline
        for cf_obj_i  in range(pred_obj_s, pred_obj_e):
            cf_i_obj_preds = greedy_obj_preds.detach().clone()
            cf_i_obj_preds[cf_obj_i] = top_N_obj_preds[cf_obj_i, 1]

            cf_i_obj_scores = greedy_obj_scores.detach().clone()
            cf_i_obj_scores[cf_obj_i] = top_N_obj_dists[cf_obj_i, 1]

            cf_i_obj_preds_np = cf_i_obj_preds.data.cpu().numpy()
            cf_i_obj_scores_np = cf_i_obj_scores.data.cpu().numpy()
            boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets_np(bboxes_np[pred_obj_s: pred_obj_e], 
                                                                    cf_i_obj_scores_np[pred_obj_s: pred_obj_e],
                                                                    cf_i_obj_preds_np[pred_obj_s: pred_obj_e], 
                                                                    rel_inds_np[pred_rel_s: pred_rel_e, 1:] - pred_obj_s, 
                                                                    rel_scores_np[pred_rel_s: pred_rel_e])
            pred_i_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_i, 'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
            evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
            recall_i_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][100][-1]
            recall_list.append((recall_greedy_100, recall_i_100))

    recall_list_np = np.array(recall_list)
    top_N_label_weights = top_N_obj_dists[:, :2].cpu().data.numpy()
    top_N_label_weights_norm = top_N_label_weights / top_N_label_weights.sum(1, keepdims=True)
    reward = recall_list_np[:, 0] - (top_N_label_weights_norm * recall_list_np).sum(1)

    losses = - torch.log(greedy_obj_scores + 1e-8) * Variable(torch.from_numpy(reward).cuda().float())

    # test3_2
    # preivous time step loss
    # if conf.num_iter > 1:
    #     for iter_i in range(conf.num_iter-1):
    #         prev_rm_obj_dists = F.softmax(result.rm_obj_dists_list[iter_i], dim=1)
    #         prev_greedy_obj_preds = prev_rm_obj_dists[:, 1:].max(1)[1] + 1
    #         prev_greedy_twod_inds = arange(prev_greedy_obj_preds.data) * num_classes + prev_greedy_obj_preds.data
    #         prev_greedy_obj_scores = prev_rm_obj_dists.view(-1)[prev_greedy_twod_inds]
    #         losses -= torch.log(prev_greedy_obj_scores + 1e-8) * Variable(torch.from_numpy(reward).cuda().float())

    loss = sum(losses)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    optimizer.step()

    return pd.Series(image_recall)

# for sgcls task
# def train_batch_rl(b, verbose=False, evaluator=None):
#     # import pdb; pdb.set_trace()
#     result = detector[b]

#     # post-process from calculate recall
#     num_classes = len(train.ind_to_classes)
#     rm_obj_dists = F.softmax(result.rm_obj_dists, dim=1)

#     # select top N # N=2
#     top_N_obj_dists, top_N_obj_preds = rm_obj_dists[:, 1:].sort(1, descending=True)
#     top_N_obj_preds = top_N_obj_preds + 1

#     greedy_obj_preds = result.obj_preds
#     greedy_twod_inds = arange(greedy_obj_preds.data) * num_classes + greedy_obj_preds.data
#     greedy_obj_scores = rm_obj_dists.view(-1)[greedy_twod_inds]

#     if conf.mode in ['sgcls', 'predcls']:
#         bboxes = result.rm_box_priors
#     elif conf.mode == 'sgdet':
#         bboxes = result.boxes_all.view(-1, 4)[greedy_twod_inds].view(result.boxes_all.size(0), 4)
#     else:
#         raise ValueError
#     import pdb; pdb.set_trace()
#     rel_scores = F.sigmoid(result.rel_dists)

#     batch_img_inds = b.gt_classes[:, 0].data
#     recall_list = []
#     image_recall = []
#     for i, gt_obj_s, gt_obj_e in enumerate_by_image(batch_img_inds):
#         # clean recall cache
#         evaluator[conf.mode].result_dict[conf.mode+'_recall'][20] = []
#         evaluator[conf.mode].result_dict[conf.mode+'_recall'][50] = []
#         evaluator[conf.mode].result_dict[conf.mode+'_recall'][100] = []
#         gt_rel_s = (b.gt_rels[:, 0] == i).nonzero().min().data.cpu().numpy().item()
#         gt_rel_e = (b.gt_rels[:, 0] == i).nonzero().max().data.cpu().numpy().item()+1 # the last one is not included      
#         pred_obj_s = (result.im_inds == i).nonzero().min().data.cpu().numpy().item()
#         pred_obj_e = (result.im_inds == i).nonzero().max().data.cpu().numpy().item()+1 
#         pred_rel_s = (result.rel_inds[:, 0] == i).nonzero().min()
#         pred_rel_e = (result.rel_inds[:, 0] == i).nonzero().max()+1

#         gt_entry = {
#             'gt_classes': b.gt_classes[gt_obj_s:gt_obj_e, 1].data.cpu().numpy().copy(),
#             'gt_relations': dist2idx(b.gt_rels[gt_rel_s:gt_rel_e, 1:].data.cpu().numpy().copy()),
#             'gt_boxes': b.gt_boxes[gt_obj_s:gt_obj_e].data.cpu().numpy().copy(),}
#         if conf.mode in ['sgcls', 'sgdet']:
#             boxes_greedy, objs_greedy, obj_scores_greedy, rels_greedy, pred_scores_greedy = \
#                                                                                 filter_dets(bboxes[pred_obj_s: pred_obj_e], 
#                                                                                 greedy_obj_scores[pred_obj_s: pred_obj_e],
#                                                                                 greedy_obj_preds[pred_obj_s: pred_obj_e], 
#                                                                                 result.rel_inds[pred_rel_s: pred_rel_e, 1:] - gt_obj_s, 
#                                                                                 rel_scores[pred_rel_s: pred_rel_e])
#         elif conf.mode == 'predcls':
#             import pdb; pdb.set_trace()
#             pass
#         else:
#             import pdb; pdb.set_trace()
#             raise NotImplementedError

#         pred_greedy_entry = {
#             'pred_boxes': boxes_greedy * BOX_SCALE/IM_SCALE,
#             'pred_classes': objs_greedy, 'pred_rel_inds': rels_greedy,
#             'obj_scores': obj_scores_greedy, 'rel_scores': pred_scores_greedy,}
#         evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_greedy_entry)
#         recall_greedy_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][100][-1]
#         image_recall.append(recall_greedy_100)
#         # counterfactual baseline
#         for cf_obj_i  in range(pred_obj_s, pred_obj_e):
#             cf_i_obj_preds = greedy_obj_preds.detach().clone()
#             cf_i_obj_preds[cf_obj_i] = top_N_obj_preds[cf_obj_i, 1]

#             cf_i_obj_scores = greedy_obj_scores.detach().clone()
#             cf_i_obj_scores[cf_obj_i] = top_N_obj_dists[cf_obj_i, 1]

#             boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = filter_dets(bboxes[pred_obj_s: pred_obj_e], 
#                                                                     cf_i_obj_scores[pred_obj_s: pred_obj_e],
#                                                                     cf_i_obj_preds[pred_obj_s: pred_obj_e], 
#                                                                     result.rel_inds[pred_rel_s: pred_rel_e, 1:] - gt_obj_s, 
#                                                                     rel_scores[pred_rel_s: pred_rel_e])
#             pred_i_entry = {
#                 'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
#                 'pred_classes': objs_i, 'pred_rel_inds': rels_i,
#                 'obj_scores': obj_scores_i, 'rel_scores': pred_scores_i,}
#             evaluator[conf.mode].evaluate_scene_graph_entry(gt_entry, pred_i_entry)
#             recall_i_100 = evaluator[conf.mode].result_dict[conf.mode+'_recall'][100][-1]
#             recall_list.append((recall_greedy_100, recall_i_100))

#     recall_list_np = np.array(recall_list)
#     top_N_label_weights = top_N_obj_dists[:, :2].cpu().data.numpy()
#     top_N_label_weights_norm = top_N_label_weights / top_N_label_weights.sum(1, keepdims=True)
#     reward = recall_list_np[:, 0] - (top_N_label_weights_norm * recall_list_np).sum(1)

#     losses = - torch.log(greedy_obj_scores + 1e-8) * Variable(torch.from_numpy(reward).cuda().float())

#     # test3_2
#     # preivous time step loss
#     # if conf.num_iter > 1:
#     #     for iter_i in range(conf.num_iter-1):
#     #         prev_rm_obj_dists = F.softmax(result.rm_obj_dists_list[iter_i], dim=1)
#     #         prev_greedy_obj_preds = prev_rm_obj_dists[:, 1:].max(1)[1] + 1
#     #         prev_greedy_twod_inds = arange(prev_greedy_obj_preds.data) * num_classes + prev_greedy_obj_preds.data
#     #         prev_greedy_obj_scores = prev_rm_obj_dists.view(-1)[prev_greedy_twod_inds]
#     #         losses -= torch.log(prev_greedy_obj_scores + 1e-8) * Variable(torch.from_numpy(reward).cuda().float())

#     loss = sum(losses)

#     optimizer.zero_grad()
#     loss.backward()
#     clip_grad_norm(
#         [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
#         max_norm=conf.clip, verbose=verbose, clip=True)
#     optimizer.step()

#     return pd.Series(image_recall)


start_epoch = -1

def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
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

print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

# import pdb; pdb.set_trace()
# mAp = val_epoch()
# import pdb; pdb.set_trace()

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    if conf.sl_train:
        rez = train_epoch(epoch)
    elif conf.rl_train:
        rez = train_epoch_rl(epoch)
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