<div align='center'>

# A Quality-Guided Mixture of Score-fusion Experts Framework for Human Recognition
[![arXiv](https://img.shields.io/badge/arXiv-2508.00053-b31b1b)](https://arxiv.org/abs/2508.00053)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-red)](https://github.com/jiezhu23/QME_ICCV25)
</div>

## 🗞️ News  
- Training code, checkpoints and scripts have been released.
- Preprocessed datasets and evaluation protocol have been released.
- The paper is available on [arXiv](https://arxiv.org/abs/2508.00053).
- Our paper has been accepted to **ICCV 2025**! 🎉

## 📂 Project Structure
```
QME_ICCV25/
├── checkpoints/         # Pre-trained model weights
├── configs/             # Configuration files for models and training
├── data/                # Data loading and preprocessing scripts
│   ├── datasets/        # Dataset-specific loaders
│   └── ...
├── test_feats/          # Pre-computed features and score matrices for evaluation
├── tools/               # Utility scripts for evaluation, preprocessing, etc.
├── demo.ipynb           # Jupyter notebook for evaluation and demonstration
├── environment.yml      # Conda environment file
├── model.py             # Main model definition
└── README.md
```

## 🚀 Getting Started
Stay tuned for the update!

### 1. Environment Setup

To get started, clone the repository and set up the Conda environment:

```bash
git clone https://github.com/jiezhu23/QME_ICCV25.git
cd QME_ICCV25
conda env create -f environment.yml
conda activate qme
```

### 2. Download Preprocessed Datasets

For convenience, we provide preprocessed datasets and pre-computed score matrices. You can download them from the following link:

- **[[Google Drive]](https://drive.google.com/drive/folders/1TBt4HrJlm-Y-IO5SA7IAamZlWvj1vHQU?usp=sharing)**

The preprocessed datasets (`Dataset/`) are provided as `.h5` files. For example, the CCVID dataset includes `CCVID.h5` (for body images) and `ccvid_face.h5` (for face images), maintaining the original folder structure within the HDF5 file.

- `ccvid.h5`:
  - key: `'session1/001_01/00001'` contains the body image array.
- `ccvid_face.h5`:
  - key: `'session1/001_01/00001/face_0'` contains the face image array. Note that a single body image may contain zero or multiple detected faces.

The `test_feats/` directory contains pre-computed score matrices for each dataset and model. For example:
- `scoreamts_ccvid.h5`:
  - `adaface/score_mat`: Score matrix of shape `(num_probes, num_gallery_templates)` for Rank-1 and mAP evaluation.
  - `adaface/merge_score_mat`: Score matrix of shape `(num_probes, num_gallery)` for TAR and FNIR evaluation.
  - `q_pids`: Probe labels for the score matrix.
  - `g_pids`: Gallery labels for the score matrix.
  - ... 

### 3. Download Pre-trained Models

You can download the pre-trained models from the following link:

- **[[Google Drive]](https://drive.google.com/drive/folders/1TBt4HrJlm-Y-IO5SA7IAamZlWvj1vHQU?usp=sharing)**

The pre-trained models should be stored in the `./checkpoints/` directory. e.g., `checkpoints/CAL/...`; `checkpoints/BigGait/...`; `checkpoints/AIM/...`.

For `BigGait`, place `pretrained_LVMs` and `torchhub` folder in `./WBModules/BigGait/`. e.g., `./WBModules/BigGait/pretrained_LVMs/dinov2_vitl14_pretrain.pth`.

For `AdaFace`, we download the pre-trained model via HuggingFace. Change the cache path `adaface_cache_path` in `./configs/backbone_cfg-{dataset}.yaml` to your own path (absolute path) and `HF_TOKEN`. Please refer to `build_face_backbone()` in `model.py` for more details.

### 4. Precompute Center Features for Training Set

To precompute the center features, e.g., for `CAL` on `CCVID` dataset, run the following command:

```bash
python precompute_center.py --mode cal-ccvid --dataset ccvid --backbone_cfg ./configs/backbone_cfg-ccvid.yaml
```

Model mode is defined in `./configs/backbone_cfg-{dataset}.yaml`. By default, the center features are saved in the `./mod_center_feat/` directory. It will generate 
1. `./mod_center_feat/cal-ccvid_center_n100_ccvid_CC.h5`: `100` frames features (by default) for each video, stored with video key.
2. `./mod_center_feat/cal-ccvid_center_ccvid_CC.h5`: center features for each video aggregated by `100` frames, stored with video key.


### 5. Precompute Score Matrices for Test Set

We provide pre-computed score matrices for all datasets reported in the paper in the previous link: `test_feats/scoremats_{dataset}.h5`. To extract by yourself, e.g., score matrices for `CCVID` dataset, run the following command:

```bash
# for test query features
python test.py --mode cal-ccvid --dataset ccvid --eval_mode feat
python test.py --mode adface --dataset ccvid --eval_mode feat
python test.py --mode biggait --dataset ccvid --eval_mode feat

# Gather score matrices for CCVID
python test.py --mode adaface,biggait,cal-ccvid --dataset ccvid --eval_mode gather
```

The result maybe slightly different from the paper.

### 6. Training QME

#### 6.1 Training Quality Estimater

By default, we train quality estimator for face. For example, for `LTCC` dataset, run the following command:
```bash
python main.py \
  --mode mod_qe \
  --dataset ltcc \
  --backbone_cfg ./configs/backbone_cfg-ltcc.yaml \
  --train_cfg ./configs/train_cfg-ltcc.yaml \
  --rank_threshold 3 \
  --epoch 5 \
  --max_step 6000 \
  --test_per_step 3000
```
If you want to train quality estimator for body, please check the commented code in `model.py`.

#### 6.2 Training QME

You can start from using the provided quality estimator checkpoint in the link. For example, for `LTCC` dataset, run the following command:
```bash
python main.py \
  --mode score_loss \
  --dataset ltcc \
  --backbone_cfg ./configs/backbone_cfg-ltcc.yaml \
  --train_cfg ./configs/train_cfg-ltcc.yaml \
  --test_per_step 20
```

More details can be found in the `main.py` file.



## 📊 Evaluation

We provide evaluation code and usage examples in the `demo.ipynb` Jupyter notebook. Please follow the instructions in the notebook to run the evaluation on the provided datasets and models.

## 📄 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jie2025qme,
  title     = {A Quality-Guided Mixture of Score-fusion Experts Framework for Human Recognition},
  author    = {Jie Zhu and Yiyang Su and Minchul Kim and Anil Jain and Xiaoming Liu},
  booktitle = {In Proceeding of International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


