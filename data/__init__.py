import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from data.transforms import BaseRgbCuttingTransform, BasePILCuttingTransform

from torch.utils.data import DataLoader
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.last import LaST
from data.datasets.ccvid import CCVID
from data.datasets.mevid import MEVID
from data.datasets.deepchange import DeepChange
from data.datasets.vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'vcclothes': VCClothes,
    'vcclothes_sc': VCClothesSameClothes,
    'vcclothes_cc': VCClothesClothesChanging,
    'last': LaST,
    'ccvid': CCVID,
    'mevid': MEVID,
    'deepchange': DeepChange,
}
# register new video datasets here
VID_DATASET = ['ccvid', 'mevid']


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    if config.DATA.DATASET in VID_DATASET:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, 
                                                 sampling_step=config.DATA.SAMPLING_STEP,
                                                 seq_len=config.AUG.SEQ_LEN, 
                                                 stride=config.AUG.SAMPLING_STRIDE)
    else:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_vid_transforms(config):
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.HEIGHT, width=config.DATA.WIDTH, probability=config.AUG.RE_PROB)
    ])
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                         stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test


def get_transforms(mode='kprpe', config=None, dataset_type='vid'):
    # Define train and test transforms based on the mode
    if 'kprpe' in mode or 'adaface' in mode or 'arcface' in mode:
        # For face feature
        train_transform = T.Compose([
            T.Resize((112, 112)),
            T.BlurAugmenter(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if dataset_type == 'vid':
            if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
                temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
            elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
                temporal_transform_train = TT.TemporalRandomCrop(size=config.DATA.SAMPLING_STEP, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
            else:
                raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        else:
            temporal_transform_train = None
            
        temporal_transform_test = None
        
    elif 'vit' in mode or 'evl' in mode:
        # For body feature
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomCroping(p=0.2),
            T.ToTensor(),
            T.RandomErasing(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        temporal_transform_train, temporal_transform_test = None, None

    elif 'biggait' in mode:
        train_transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomCroping(p=0.2),
            T.ToTensor(),
            BasePILCuttingTransform(cutting=0),
            T.RandomErasing()
        ])
        test_transform = T.Compose([T.Resize((256, 128)),
                                    BasePILCuttingTransform(cutting=0)])
        if dataset_type == 'vid':
            if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
                temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
            elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
                temporal_transform_train = TT.TemporalRandomCrop(size=config.DATA.SAMPLING_STEP, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
            else:
                raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        else:
            temporal_transform_train = None
            
        temporal_transform_test = None

    elif 'cal' in mode:
        # override the height and width for pre-trained model
        if 'mevid' in mode:
            config.defrost()
            config.DATA.HEIGHT=256
            config.DATA.WIDTH=128
            config.AUG.SEQ_LEN=config.DATA.SAMPLING_STEP
            config.freeze()
        else:
            config.defrost()
            config.DATA.HEIGHT=384
            config.DATA.WIDTH=192
            if dataset_type == 'vid':
                config.AUG.SEQ_LEN=config.DATA.SAMPLING_STEP
            config.freeze()
        if dataset_type == 'img':
            train_transform, test_transform = build_img_transforms(config)
            temporal_transform_train, temporal_transform_test = None, None
        else:
            train_transform, test_transform, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)
    
    elif 'agrl' in mode:
        # from WBModules.AGRL.torchreid import transforms as AT
        train_transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
            temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
        elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
            temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                            stride=config.AUG.SAMPLING_STRIDE)
        else:
            raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))
        temporal_transform_test = None

    elif 'aim' in mode:
        config.defrost()
        config.DATA.HEIGHT=384
        config.DATA.WIDTH=192
        if dataset_type == 'img':
            train_transform, test_transform = build_img_transforms(config)
            temporal_transform_train, temporal_transform_test = None, None
        else:
            train_transform, test_transform, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)
    else:
        raise ValueError('Not a correct mode')

    return train_transform, test_transform, temporal_transform_train, temporal_transform_test


def build_dataloader(mode, config, is_training=True):
    dataset = build_dataset(config)
    # video dataset
    if config.DATA.DATASET in VID_DATASET:
        
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = {}, {}, {}, {}
        for m in mode:
            spatial_transform_train_m, spatial_transform_test_m, temporal_transform_train_m, temporal_transform_test_m = get_transforms(m, config, dataset_type='vid')
            spatial_transform_train[m] = spatial_transform_train_m
            spatial_transform_test[m] = spatial_transform_test_m
            temporal_transform_train[m] = temporal_transform_train_m
            temporal_transform_test[m] = temporal_transform_test_m
        
        # spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)

        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train, is_training=is_training),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_train, temporal_transform_train, is_training=is_training),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        
        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test, is_training=False),
            sampler=DistributedInferenceSampler(dataset.recombined_query['dataset']),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test, is_training=False),
            sampler=DistributedInferenceSampler(dataset.recombined_gallery['dataset']),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler
    # image dataset
    else:
        transform_train, transform_test = {}, {}
        for m in mode:
            transform_train_m, transform_test_m, _, _ = get_transforms(m, config, dataset_type='img')
            transform_train[m] = transform_train_m
            transform_test[m] = transform_test_m
        # transform_train, transform_test = build_img_transforms(config)
        train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                         num_instances=config.DATA.NUM_INSTANCES, 
                                                         seed=config.SEED)
        trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train, is_training=is_training),
                                 sampler=train_sampler,
                                 batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=True)

        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test, is_training=False),
                                   sampler=DistributedInferenceSampler(dataset.gallery['dataset']),
                                   batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=True, drop_last=False, shuffle=False)

        if config.DATA.DATASET == 'prcc':
            queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query_same),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)
            queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query_diff),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
        else:
            queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test, is_training=False),
                                     sampler=DistributedInferenceSampler(dataset.query['dataset']),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader, galleryloader, dataset, train_sampler
