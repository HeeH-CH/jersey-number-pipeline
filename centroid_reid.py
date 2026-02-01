from pathlib import Path
import sys
import os
import argparse
import distutils.version 
print("EXEC:", sys.executable)
ROOT = './reid/centroids-reid/'
sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image

from config import cfg
from train_ctl_model import CTLModel

from datasets.transforms import ReidTransforms



# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT+'/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT+'/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/market1501_resnet50_256_128_epoch_120.ckpt')
ver_to_specs["res50_duke"]   = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt')


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights

def generate_features(input_folder, output_folder, model_version='res50_market', batch_size=32):
    # load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)

    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)

    # print("Loading from " + MODEL_FILE)
    if use_cuda:
        model.to('cuda')
        print("using GPU")
    model.eval()

    tracks = os.listdir(input_folder)
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)

    for track in tqdm(tracks):
        track_path = os.path.join(input_folder, track)
        images = sorted(os.listdir(track_path))
        output_file = os.path.join(output_folder, f"{track}_features.npy")

        # Batch processing: load and transform all images first
        img_tensors = []
        for img_path in images:
            img = cv2.imread(os.path.join(track_path, img_path))
            if img is not None:
                input_img = Image.fromarray(img)
                img_tensors.append(val_transforms(input_img))

        if not img_tensors:
            continue

        # Process in batches
        all_features = []
        for i in range(0, len(img_tensors), batch_size):
            batch = img_tensors[i:i + batch_size]
            batch_tensor = torch.stack(batch)

            with torch.no_grad():
                if use_cuda:
                    batch_tensor = batch_tensor.cuda()
                _, global_feat = model.backbone(batch_tensor)
                global_feat = model.bn(global_feat)
                all_features.append(global_feat.cpu())

        # Concatenate all features
        features = torch.cat(all_features, dim=0).numpy()

        with open(output_file, 'wb') as f:
            np.save(f, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    generate_features(args.tracklets_folder, args.output_folder, batch_size=args.batch_size)



