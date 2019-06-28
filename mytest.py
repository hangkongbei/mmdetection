import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import os
import sys
from tqdm import tqdm
import os.path as osp
import os

# cfg = mmcv.Config.fromfile('configs/retinanet_x101_64x4d_fpn_1x.py')
# cfg.model.pretrained = None

# # construct the model and load checkpoint
# model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, './checkpoints/retinanet_x101_64x4d_fpn_2x_20181218-5e88d045.pth')

# test a single image
# img_dir = "./mydata/coco/"
# img_path = img_dir + '000000000001.jpg' 
# img = mmcv.imread(img_path)
# result = inference_detector(model, img, cfg)
# show_result(img, result)

# test a list of images
# img_dir = "./mydata/cityscapes/"
# folders = sorted([f for f in os.listdir(img_dir) if not f.startswith(".") and
#                       not f.endswith("py") and not f.endswith("npz")])
# imgs = []
# for folder in folders:
#     imgs.append(img_dir+folder)

# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     show_result(imgs[i], result, out_file="./mydata/cityscapes_del/"+folders[i])

def visual_image_reulsts(images_files, cfg_file, model_file, result_dir, device="cuda:0"):
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    if not osp.exists(cfg_file) or not osp.exists(model_file):
        print("Error: no such file: %s or %s"%(cfg_file, model_file))
        return None
    cfg = mmcv.Config.fromfile(cfg_file)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, model_file)
    #set specified score thresh for better results visualization
    cfg.test_cfg["score_thr"] = 0.35
    for i, result in enumerate(tqdm(inference_detector(model, images_files, cfg, device=device))):
        show_result(images_files[i], result, dataset="hangkongbei", out_file=osp.join(result_dir, osp.basename(images_files[i])))

def main():
    cfg_file = "./configs/hangkongbei/retinanet_x101_32x4d_fpn_1x.py"
    dataet_root = "../dataset/hangkongbei/VOC2007/"
    images_dir = "JPEGImages/"
    dataset_dir = "ImageSets/Main/"
    result_dir = "./mydata/hangkongbei_val_vis_resx101"
    model_file = "./work_dirs/retinanet_x101_32x4d_fpn_1x_800x800_cls2/latest.pth"

    val_set = osp.join(dataet_root, dataset_dir, "val.txt")
    print("val_set: %s"%(val_set))
    img_ids = mmcv.list_from_file(val_set)
    images_files = [osp.join(dataet_root, images_dir, img_id+".jpg") for img_id in img_ids]
    print("val set length: %d"%(len(images_files)))

    visual_image_reulsts(images_files, cfg_file, model_file, result_dir)

if __name__ == "__main__":
    main()
        