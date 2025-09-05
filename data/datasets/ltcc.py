import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json


class LTCC(object):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC_ReID'
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes
        
        # reformat the data
        train = {'dataset': train, 'dataset_name': 'ltcc'}
        query = {'dataset': query}
        gallery = {'dataset': gallery}
        if hasattr(self, 'body_data'):
            train['body_data'] = self.body_data
            query['body_data'] = self.body_data
            gallery['body_data'] = self.body_data
        if hasattr(self, 'face_data'):
            train['face_data'] = self.face_data
            query['face_data'] = self.face_data
            gallery['face_data'] = self.face_data
            # generate face id2keys for training data
            train_face_dict = {}
            for img_name in self.face_data['train'].keys():
                id, clothes_id, cam_id, _  = img_name.split('_')
                if id not in train_face_dict:
                    train_face_dict[id] = []
                train_face_dict[id].append(img_name)
            train['id2face'] = train_face_dict

        logger = logging.getLogger('reid.dataset')
        logger.info("=> LTCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if osp.exists(osp.join(self.dataset_dir, 'ltcc.h5')):
            self.body_data = h5py.File(osp.join(self.dataset_dir, 'ltcc.h5'), 'r')
            print('h5 format file detected, using h5 format data')
            if not osp.exists(osp.join(self.dataset_dir, 'ltcc_face.h5')):
                print('face data is not available, using body data only')
            else:
                self.face_data = h5py.File(osp.join(self.dataset_dir, 'ltcc_face.h5'), 'r')
        else:
            if not osp.exists(self.train_dir):
                raise RuntimeError("'{}' is not available".format(self.train_dir))
            if not osp.exists(self.query_dir):
                raise RuntimeError("'{}' is not available".format(self.query_dir))
            if not osp.exists(self.gallery_dir):
                raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path):
        if hasattr(self, 'body_data'):
            folder = osp.split(dir_path)[-1]
            img_paths = [osp.join(folder, idx) for idx in self.body_data[folder].keys()] # h5 file format
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.png')) # original
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            dataset.append((img_path, pid, camid, clothes_id))
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        if hasattr(self, 'body_data'):
            query_folder = osp.split(query_path)[-1]
            gallery_folder = osp.split(gallery_path)[-1]
            query_img_paths = [osp.join(query_folder, idx) for idx in self.body_data[query_folder].keys()] # h5 file format
            gallery_img_paths = [osp.join(gallery_folder, idx) for idx in self.body_data[gallery_folder].keys()] # h5 file format
        else:
            query_img_paths = glob.glob(osp.join(query_path, '*.png'))
            gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            query_dataset.append((img_path, pid, camid, clothes_id))

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            gallery_dataset.append((img_path, pid, camid, clothes_id))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

