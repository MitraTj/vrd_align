# --------------------------------------------------------
# Goal: assign ROIs to targets
# --------------------------------------------------------


import numpy as np
import numpy.random as npr
from config import BG_THRESH_HI, BG_THRESH_LO, REL_FG_FRACTION, RELS_PER_IMG_REFINE, RELS_PER_IMG_SGDET_SL, RELS_PER_IMG_SGDET_RL, ModelConfig
from lib.fpn.box_utils import bbox_overlaps
from lib.pytorch_misc import to_variable, nonintersecting_2d_inds
from collections import defaultdict
import torch

conf = ModelConfig()

def rel_assign_dist2idx(rel_dists):
    rel_idx, rel_type = rel_dists[:, 3:].nonzero()
    return np.concatenate((rel_dists[:, :3][rel_idx], rel_type[:, None]), 1)

@to_variable
def rel_assignments(im_inds, rpn_rois, roi_gtlabels, roi_predscore, gt_boxes, gt_classes, gt_rels, image_offset,
                    fg_thresh=0.5, num_sample_per_gt=4, filter_non_overlap=True):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    if conf.sl_train:
        rels_per_image = RELS_PER_IMG_SGDET_SL
    elif conf.rl_train:
        rels_per_image = RELS_PER_IMG_SGDET_RL
    else:
        raise ValueError

    fg_rels_per_image = int(np.round(REL_FG_FRACTION * rels_per_image))

    pred_inds_np = im_inds.cpu().numpy()
    pred_boxes_np = rpn_rois.cpu().numpy()
    pred_boxlabels_np = roi_gtlabels.cpu().numpy()
    gt_boxes_np = gt_boxes.cpu().numpy()
    gt_classes_np = gt_classes.cpu().numpy()
    gt_rels_np = gt_rels.cpu().numpy()

    # test1_4 change roi_pred_score to each object score
    roi_predscore_np = roi_predscore.cpu().numpy()
    norm_roi_predscore_np = np.exp(roi_predscore_np) / (np.exp(roi_predscore_np + 1e-8).sum(1)[:, None])
    each_roi_predlabel = np.argmax(norm_roi_predscore_np[:, 1:], 1) + 1
    each_roi_predscore = norm_roi_predscore_np[np.arange(norm_roi_predscore_np.shape[0]), each_roi_predlabel]

    # extra add for new_visual_genome.py
    # gt_rels_np = rel_assign_dist2idx(gt_rels_np)
    gt_classes_np[:, 0] -= image_offset
    gt_rels_np[:, 0] -= image_offset

    num_im = gt_classes_np[:, 0].max()+1

    # print("Pred inds {} pred boxes {} pred box labels {} gt classes {} gt rels {}".format(
    #     pred_inds_np, pred_boxes_np, pred_boxlabels_np, gt_classes_np, gt_rels_np
    # ))
    rel_labels = []
    num_box_seen = 0
    for im_ind in range(num_im):
        pred_ind = np.where(pred_inds_np == im_ind)[0]

        gt_ind = np.where(gt_classes_np[:, 0] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[gt_rels_np[:, 0] == im_ind, 1:]

        # [num_pred, num_gt]
        pred_boxes_i = pred_boxes_np[pred_ind]
        pred_boxlabels_i = pred_boxlabels_np[pred_ind]

        # test1_4
        pred_score_i = each_roi_predscore[pred_ind]

        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
        is_match = (pred_boxlabels_i[:,None] == gt_classes_i[None]) & (ious >= fg_thresh)

        # FOR BG. Limit ourselves to only IOUs that overlap, but are not the exact same box
        pbi_iou = bbox_overlaps(pred_boxes_i, pred_boxes_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_boxes_i.shape[0], pred_boxes_i.shape[0]),
                                        dtype=np.int64) - np.eye(pred_boxes_i.shape[0],
                                                                 dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)

        # extra set to comments
        # ONLY select relations between ground truth because otherwise we get useless data
        # rel_possibilities[pred_boxlabels_i == 0] = 0
        # rel_possibilities[:, pred_boxlabels_i == 0] = 0

        # Sample the GT relationships.
        fg_rels = []
        p_size = []
        for i, each_gt_rels_i in enumerate(gt_rels_i):
            from_gtind, to_gtind, rel_id = each_gt_rels_i[0], each_gt_rels_i[1], each_gt_rels_i[2:]
            fg_rels_i = []
            fg_scores_i = []

            for from_ind in np.where(is_match[:, from_gtind])[0]:
                for to_ind in np.where(is_match[:, to_gtind])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append(np.concatenate((np.array([from_ind, to_ind]), rel_id), 0))
                        fg_scores_i.append((ious[from_ind, from_gtind] * ious[to_ind, to_gtind]))
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue

            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])

        if len(fg_rels) > 0:
            fg_rels = np.vstack(fg_rels)
            if fg_rels.shape[0] > fg_rels_per_image:
                fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        else:
            fg_rels = np.zeros((0, 53), dtype=np.int64)

        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.ones(bg_rels.shape[0], dtype=np.int64), 
                                    np.zeros((bg_rels.shape[0], 50), dtype=np.int64)))

        num_bg_rel = min(rels_per_image - fg_rels.shape[0], bg_rels.shape[0])

        if bg_rels.size > 0:
            # test1_2, test1_3
            # origin
            # bg_rels = bg_rels[
            #     np.random.choice(bg_rels.shape[0],
            #                      #p=p,
            #                      size=num_bg_rel, replace=False)]
            # test1_4
            sub_pred_score_i = pred_score_i[bg_rels[:, 0]]
            obj_pred_score_i = pred_score_i[bg_rels[:, 1]]
            bg_rels_idx_i = np.argsort(sub_pred_score_i*obj_pred_score_i)[-num_bg_rel:] # larget in the tail
            bg_rels = bg_rels[bg_rels_idx_i]
        else:
            bg_rels = np.zeros((0, 53), dtype=np.int64)

        if fg_rels.size == 0 and bg_rels.size == 0:
            # Just put something here
            bg_rels = np.zeros((0, 53), dtype=np.int64)

        # print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:,0:2] += num_box_seen

        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:,1], all_rels_i[:,0]))]

        rel_labels.append(np.column_stack((
            im_ind*np.ones(all_rels_i.shape[0], dtype=np.int64),
            all_rels_i,
        )))

        num_box_seen += pred_boxes_i.shape[0]

    rel_labels_np = np.concatenate(rel_labels, 0)

    # extra add for new_visual_genome.py
    # num_rel_labels = rel_labels_np.shape[0]
    # rel_labels_tail = np.zeros((num_rel_labels, 51))
    # rel_labels_tail[range(num_rel_labels), rel_labels_np[:, -1]] = 1
    # rel_labels_head = rel_labels_np[:, :3]
    # rel_labels_np = np.concatenate((rel_labels_head, rel_labels_tail), 1)

    rel_labels = torch.LongTensor(rel_labels_np).cuda(rpn_rois.get_device(),
                                                                      async=True)
    return rel_labels
