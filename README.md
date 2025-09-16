<div align='center'>

# A Quality-Guided Mixture of Score-fusion Experts Framework for Human Recognition
[![arXiv](https://img.shields.io/badge/arXiv-2508.00053-b31b1b)](https://arxiv.org/abs/2508.00053)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-red)](https://github.com/jiezhu23/QME_ICCV25)
</div>

## 🗞️ News  
- Preprocessed datasets and evaluation protocol have been released.
- The paper is available on [arXiv](https://arxiv.org/abs/2508.00053).
- Our paper has been accepted to **ICCV 2025**! 🎉

## 📋 TODO List
- [ ] Release training code and scripts.
- [X] Release preprocessed datasets.
- [X] Release evaluation code and model checkpoints.
- [X] Prepare camera-ready and arXiv version.

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

