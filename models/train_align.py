
import _init_paths
from dataloaders.new_visual_genome import VGDataLoader, VG, dist2idx
import numpy as np
from torch import optim
import torch
torch.manual_seed(2019)
np.random.seed(2019)

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
# alpha_neg_entropy = 0.005
# reward_type = conf.reward_type

constant_baseline = 0.5
baseline_decay = 0.99

if conf.model == 'align':
    from lib.rel_model_align import RelModelAlign as RelModel
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
                                               num_gpus=conf.num_gpus,
                                               return_fn=False)


detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet,
                    use_proposals=conf.use_proposals,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    limit_vision=conf.limit_vision,
                    sl_pretrain=True,
                    num_iter=conf.num_iter,
                    hidden_dim=conf.hidden_dim,
                    reduce_input=conf.reduce_input,
                    post_nms_thresh=conf.post_nms_thresh,
                    )

assert conf.save_dir.split('/')[-4] == 'checkpoints'
save_path = conf.save_dir
tensorboard_path = os.path.join('tfboard', '/'.join(conf.save_dir.split('/')[-3:]))
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
    pretrained_params = [p for n,p in detector.named_parameters() if ('roi_fmap_obj' in n or 'obj_compress' in n) and p.requires_grad]
    non_pretrained_params = [p for n,p in detector.named_parameters() if not ('roi_fmap_obj' in n or 'obj_compress' in n) and p.requires_grad]

    params = [{'params': pretrained_params, 'lr': lr / 10.0}, {'params': non_pretrained_params}]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.3,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    # new_scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)

    return optimizer, scheduler

def np_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

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
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.context.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.context.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.context.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.context.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.context.obj_compress.weight.data.copy_(ckpt['state_dict']['score_fc.weight'])
    detector.context.obj_compress.bias.data.copy_(ckpt['state_dict']['score_fc.bias'])


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

    result = detector[b]
    if result is None:
        return pd.Series({'class_loss': 0.0, 'rel_loss': 0.0, 'total': 0.0})   

    losses = {}

    losses['class_loss'] = F.cross_entropy(result.rm_obj_logits, result.rm_obj_labels)
    losses['rel_loss'] = F.binary_cross_entropy_with_logits(result.rel_logits, result.rel_labels[:, 3:].float())
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


def val_epoch():
    # import pdb; pdb.set_trace()
    detector.eval()

    if conf.save_detection_results:
        all_save_res = []
    # evaluator = BasicSceneGraphEvaluator.all_modes()
    evaluator = BasicSceneGraphEvaluator.all_modes(spice=False)

    for val_b, batch in enumerate(tqdm(val_loader)):

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
