import os
import sys
import shutil
import errno
import json
import os.path as osp
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
from PIL import Image
from retinaface import RetinaFace


def visualize_face_bbox(img_path):
    try:
        cropped_faces = DeepFace.extract_faces(img_path, detector_backend = 'retinaface', align=False)
    except ValueError:
        cropped_faces = []

    img = cv2.imread(img_path)

    if len(cropped_faces) == 0:
        print('No face detected in the image')
    else:
        for face in cropped_faces:
            # Extract the facial area coordinates
            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']

            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()
    
    
def crop_face(img_path, save_path):
    faces = RetinaFace.extract_faces(img_path, align = False)

    if len(faces) == 0:
        print(f'No face detected in the {img_path}')
    else:
        for face in faces:
            face = Image.fromarray(face)
            face = face.resize((112, 112))
            face.save(save_path)


def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

'''
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
'''


def get_logger(fpath, local_rank=0, name=''):
    # Creat logger
    logger = logging.getLogger(name)
    level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logger.setLevel(level=level)

    # Output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Output to file
    if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
    file_handler = logging.FileHandler(fpath, mode='w')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger
