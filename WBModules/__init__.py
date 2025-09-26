import torch.nn as nn
from abc import ABC, abstractmethod
from WBModules.CAL.test import extract_img_feature
from WBModules.CAL.models.img_resnet import ResNet50
from WBModules.CAL.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from WBModules.BigGait.modeling.models.BigGait import BigGait__Dinov2_Gaitbase as GaitModel
from WBModules.AGRL.torchreid.utils.reidtools import calc_splits

from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import torch
import sys


# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

            
# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


if __name__ == '__main__':
    
    HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'
    path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_kprpe_webface4m')
    repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
    model = load_model_by_repo_id(repo_id, path, HF_TOKEN)

    # input is a rgb image normalized.
    from torchvision.transforms import Compose, ToTensor, Normalize
    from PIL import Image
    img = Image.open('path/to/image.jpg')
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(img).unsqueeze(0)  # torch.randn(1, 3, 112, 112)
    
    # KPRPE also takes keypoints locations as input
    aligner = load_model_by_repo_id('minchul/cvlface_DFA_mobilenet', path, HF_TOKEN)
    aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(input)
    keypoints = orig_ldmks  # torch.randn(1, 5, 2)
    out = model(input, keypoints)
