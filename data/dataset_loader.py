import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import random
from model import MODEL_MAPPING_DICT


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, is_training=False):
        self.dataset = dataset
        self.transform = transform
        self.is_training = is_training
        if 'jpg' in self.dataset['dataset'][0][0] or 'png' in self.dataset['dataset'][0][0]:
            self.dataset_type = 'img'
        else:
            self.dataset_type = 'h5'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset['dataset'][index]
        # get data based on key value of transform
        clips = {}
        for mode in self.transform.keys():
            if self.dataset_type == 'img':
                img = read_image(img_path)
                clip = [img]
            else:
                if MODEL_MAPPING_DICT[mode] == 'body_data':
                    clip = [Image.fromarray(self.dataset[MODEL_MAPPING_DICT[mode]][img_path][:])]
                else:
                    clip = self.face_sampler(img_path)
            if self.transform is not None:
                clip = [self.transform[mode](img) for img in clip]
            # trans (C x H x W) to (C x T=1 x H x W)
            if len(clip) == 0:
                clip = torch.zeros((0, 0, 0, 0))
            else:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips[mode] = clip

        return clips, pid, camid, clothes_id

    def face_sampler(self, img_path):
        clip = []
        if img_path in self.dataset['face_data']:
            face_idx = list(self.dataset['face_data'][img_path].keys())
            clip.append(Image.fromarray(self.dataset['face_data'][img_path][random.choice(face_idx)][:]))
        if self.is_training and len(clip) < 1:
            # Random padding from whole tracklet/video to make the length of face data equal to img_paths
            if self.dataset['dataset_name'] == 'ltcc':
                folder, img_name = osp.split(img_path)
                id, clothes_id, cam_id, _  = img_name.split('_')
                # randomly sample facial images from the same subject
                random_f = random.choice(self.dataset['id2face'][id])
                face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
        return clip


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (dict):
                Key: 'dataset' (list) - List with items (img_paths, pid, camid)
                Key: 'body_data' (h5 file pointer) - h5 file pointer to body data
                Key: 'face_data' (h5 file pointer) - h5 file pointer to face data
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True,
                 is_training=False):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader =get_loader()
        self.cloth_changing = cloth_changing
        self.is_training = is_training
        if 'jpg' in self.dataset['dataset'][0][0][0]:
            self.dataset_type = 'img'
        else:
            self.dataset_type = 'h5'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset['dataset'][index]
        else:
            img_paths, pid, camid = self.dataset['dataset'][index]

        # get data based on key value of transform
        clips = {}
        for mode in self.spatial_transform.keys():
            if self.temporal_transform[mode] is not None:
                mode_img_paths = self.temporal_transform[mode](img_paths)
            else:
                mode_img_paths = img_paths
            if self.dataset_type == 'img':
                clip = self.loader(mode_img_paths)
            else:
                if MODEL_MAPPING_DICT[mode] == 'body_data':
                    clip = [Image.fromarray(self.dataset[MODEL_MAPPING_DICT[mode]][p][:]) for p in mode_img_paths]
                else:
                    clip = self.face_sampler(mode_img_paths)

            if self.spatial_transform[mode] is not None:
                if hasattr(self.spatial_transform[mode], 'randomize_parameters'):
                    self.spatial_transform[mode].randomize_parameters()
                clip = [self.spatial_transform[mode](img) for img in clip]

            # trans T x C x H x W to C x T x H x W
            if len(clip) == 0:
                if self.is_training:
                    # create a zero tensor if face data is not available for subjects
                    clip = torch.zeros((3, len(mode_img_paths), 112, 112))
                else:
                    clip = torch.zeros((0, 0, 0, 0))
            else:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips[mode] = clip
            
        if self.cloth_changing:
            return clips, pid, camid, clothes_id
        else:
            return clips, pid, camid
        
    def face_sampler(self, img_paths):
        clip = []
        for p in img_paths:
            if p in self.dataset['face_data']:
                # one img may have multiple faces, sample one face randomly
                face_keys = list(self.dataset['face_data'][p].keys())
                clip.append(Image.fromarray(self.dataset['face_data'][p][random.choice(face_keys)][:]))
        if self.is_training:
            # Random padding from whole tracklet/video to make the length of face data equal to img_paths
            while len(clip) < len(img_paths):
                if self.dataset['dataset_name'] == 'ccvid':
                    folder, img_name = osp.split(img_paths[0])
                    if folder not in self.dataset['face_data']:
                        # some video may not have any face detected, randomly sample from other training videos with the same pid
                        session, vid = folder.split('/')
                        pid_tmp, _ = vid.split('_')
                        vid_pools = [vid for vid in self.dataset['face_data'][session].keys() if vid.split('_')[0] == pid_tmp]
                        folder = osp.join(session,random.choice(vid_pools))
                    random_f = random.choice(list(self.dataset['face_data'][folder].keys()))
                    face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                    clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
                elif self.dataset['dataset_name'] == 'mevid':
                    folder, img_name = osp.split(img_paths[0])
                    if folder not in self.dataset['face_data']:
                        # this subejct has no face detected, return an empty list
                        return clip
                    random_f = random.choice(list(self.dataset['face_data'][folder].keys()))
                    face_keys = list(self.dataset['face_data'][folder][random_f].keys())
                    clip.append(Image.fromarray(self.dataset['face_data'][folder][random_f][random.choice(face_keys)][:]))
        return clip