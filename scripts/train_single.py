from argparse import ArgumentParser
import sys
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects import point_rend
from detectron2.data.datasets import register_coco_instances


# fold = sys.argv[1]
# start = int(sys.argv[2])
# finish = int(sys.argv[3])

parser = ArgumentParser()
parser.add_argument('--label', required=True, type=str)
parser.add_argument('--fold', required=True, type=int)
parser.add_argument('--start', required=True, type=int)
parser.add_argument('--finish', required=True, type=int)

args = parser.parse_args()

dset = args.label
fold = args.fold
start = args.start
finish = args.finish

print(f'Training fold: {fold} of 4')

register_coco_instances(name = f"datatrain_{dset}_{fold}", metadata = {}, 
                        json_file = f"/data/literature_images/cement_coco_train_{dset}_fold_{fold}.json",
                        image_root = f"/data/literature_images")

train_meta = DatasetCatalog.get(f'datatrain_{dset}_{fold}')

coco_metadata = MetadataCatalog.get("coco_2017_val")

cfg = get_cfg()

cfg.OUTPUT_DIR = f'./detectron_out/fold_{dset}_{fold}'

point_rend.add_pointrend_config(cfg)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (f"datatrain_{dset}_{fold}",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
# if training from starting, uncomment below line
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training 

# if training from checkpoint, uncomment below line
cfg.MODEL.WEIGHTS = f'./detectron_out/fold_{dset}_{fold}/model_final.pth'

# The for loop logs the model result every 10 epochs
epochs = list(np.arange(start,finish,10)) #1010
print('Logging at intervals:',epochs)
for epoch in epochs:
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 2e-5  
    cfg.SOLVER.MAX_ITER = int(epoch)    
    cfg.SOLVER.STEPS = []     
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.DEVICE='cpu'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=True) # if training from checkpoint
    trainer.resume_or_load(resume=False) # if training from starting
    trainer.train()
