from dataloaders.visual_genome import VGDataLoader, VG
from lib.rel_model import RelModel
mport torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict

import numpy as np
from PIL import Image
from IPython.display import Image, display
import json
from matplotlib.pyplot import imshow
import json

path1 = '/data/VG_100k'
path2 = '/data/stanford_filtered/VG-SGG-dicts.json'

