<div align='center'>
  
# A Quality-Guided Mixture of Score-fusion Experts Framework for Human Recognition
[![arXiv](https://img.shields.io/badge/arXiv-2508.00053-b31b1b)](https://arxiv.org/abs/2508.00053)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-red)](https://github.com/jiezhu23/QME_ICCV25)
</div>

## 🗞️ News  
- Preprocessed datasets and evaluation protocol are released.
- The arXiv version has been released.
- 🎉 Our paper has been accepted to **ICCV 2025**!  

## 📋 TODO List
- [ ] Release training code.
- [x] Release preprocessed dataset.
- [x] Release evaluation code and model checkpoints.
- [x] Prepare camera-ready and arXiv version.

## 🚀 Getting Started  

Stay tuned for the update!

### Download the preprocessed dataset

For convenience, you can download the preprocessed data from the following link: [[Google Drive]](https://drive.google.com/drive/folders/1TBt4HrJlm-Y-IO5SA7IAamZlWvj1vHQU?usp=sharing)

We warp up the preprocessed dataset into a `Dataset/*.h5` file. e.g., In CCVID dataset, there are two files: `CCVID.h5` (for body images) and `ccvid_face.h5` (for face image) with the original folder structure:
- `ccvid.h5`: 
    - key: `'session1/001_01/00001'` is the body image array
- `ccvid_face.h5`: 
    - key: `'session1/001_01/00001/face_0'` is face image array

NOTE: one body image may contain 0 or multiple detected face.
`test_feats/` contains the preprocessed score matrices for each dataset and models. e.g.,:

- `scoreamts_ccvid.h5`: 
    - key: 'adaface' contains the score matrices of AdaFace model
        - key: `score_mat` is the score matrix of shape (num_probes, num_gallery_templates) for Rank1 and mAP evaluation.
        - key: `merge_score_mat` is the score matrix of shape (num_probes, num_gallery) for TAR and FNIR evaluation.
    - key: 'q_pids' is the probe labels of score matrix
    - key: 'g_pids' is the gallery labels of score matrix
    - ...


## Environment Setup

```
conda env create -f environment.yml
conda activate qme
```

## Evaluation

We provide the code of evaluation code in `demo.ipynb` and some usage examples in it.


## 📄 Citation
If you find this project useful in your research, please consider citing:
```
@inproceedings{jie2025qme,
  title     = {A Quality-Guided Mixture of Score-fusion Experts Framework for Human Recognition},
  author    = {Jie Zhu and Yiyang Su and Minchul Kim and Anil Jain and Xiaoming Liu},
  booktitle = {In Proceeding of International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

