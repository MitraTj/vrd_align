import os
import glob
import _pickle as pkl

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import VG_SGG_DICT_FN
from .new_visual_genome import load_info

class SGDET_VG(Dataset):
    def __init__(self, split, dataset_path, dict_file=VG_SGG_DICT_FN):
        super(SGDET_VG, self).__init__()
        assert split in ['train', 'val', 'test']
        self.split_path = os.path.join(dataset_path, split)
        self.all_items = sorted(glob.glob(os.path.join(self.split_path, '*.pkl')))
        self.dict_file = dict_file
        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)

    def __len__(self):
        return len(self.all_items) 

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):
        # entry_path = os.path.join(self.split_path, 'sgdet_{}.pkl'.format(index))
        entry_path = self.all_items[index]
        assert os.path.exists(entry_path)
        with open(entry_path, 'rb') as f:
            entry = pkl.load(f)

        return entry

def vg_collate(data):
    rm_box_priors_list = [] 
    rm_box_deltas_list = [] 
    boxes_all_list = [] 
    rm_obj_labels_list = [] 
    rm_obj_dists_list = []
    # rel_labels_list = []
    gt_boxes_list = [] 
    gt_classes_list = [] 
    gt_rels_list = [] 
    im_sizes_list = [] 
    fmap_list = []
    im_inds_list = []
    fn_list = []
    for idx, entry in enumerate(data):
        rm_box_priors_list.append(entry['rm_box_priors']) 
        rm_box_deltas_list.append(entry['rm_box_deltas']) 
        boxes_all_list.append(entry['boxes_all']) 
        rm_obj_labels_list.append(entry['rm_obj_labels']) 
        rm_obj_dists_list.append(entry['rm_obj_dists'])
        # rel_labels_list.append(entry['rel_labels'])
        gt_boxes_list.append(entry['gt_boxes'])
        gt_classes = entry['gt_classes']
        gt_rels = entry['gt_rels']
        gt_classes[:, 0] = idx
        gt_rels[:, 0] = idx
        gt_classes_list.append(gt_classes)
        gt_rels_list.append(gt_rels)
        im_sizes_list.append(entry['im_sizes']) 
        fmap_list.append(entry['fmap'])
        im_inds_list.append(np.array([idx]*len(entry['rm_box_priors'])))
        fn_list.append(entry['filename'])

    rm_box_priors_list = torch.from_numpy(np.concatenate(rm_box_priors_list))
    rm_box_deltas_list = torch.from_numpy(np.concatenate(rm_box_deltas_list))
    boxes_all_list = torch.from_numpy(np.concatenate(boxes_all_list))
    rm_obj_labels_list = torch.from_numpy(np.concatenate(rm_obj_labels_list))
    rm_obj_dists_list = torch.from_numpy(np.concatenate(rm_obj_dists_list))
    # rel_labels_list = rel_labels_list
    gt_boxes_list = torch.from_numpy(np.concatenate(gt_boxes_list))
    gt_classes_list = torch.from_numpy(np.concatenate(gt_classes_list))
    gt_rels_list = torch.from_numpy(np.concatenate(gt_rels_list))
    im_sizes_list = torch.from_numpy(np.concatenate(im_sizes_list))
    fmap_list = torch.from_numpy(np.concatenate(fmap_list))
    im_inds_list = torch.from_numpy(np.concatenate(im_inds_list))
    return rm_box_priors_list, rm_box_deltas_list, boxes_all_list, rm_obj_labels_list, rm_obj_dists_list, \
                None, gt_boxes_list, gt_classes_list, gt_rels_list, im_sizes_list, fmap_list, im_inds_list, fn_list

class SGDET_VG_LOADER(torch.utils.data.DataLoader):

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, **kwargs):
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x,),
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x),
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )
        return train_load, val_load