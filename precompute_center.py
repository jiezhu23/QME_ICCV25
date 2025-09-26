import yaml
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from model import QME
from test import *
from loss import *
from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataset, get_transforms, VID_DATASET
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler

torch.backends.cudnn.enabled = False


def aggregate_mod_center(read_path, save_path):
    f = h5py.File(read_path, 'r')
    f2 = h5py.File(save_path, 'w')
    # gid is the idx from train loader
    gid = sorted(list(f.keys()), key=int)
    def get_all_feats(group):
        center_feat = []
        def recursive_collect_feats(sub_group):
            for key in sub_group.keys():
                item = sub_group[key]
                if isinstance(item, h5py.Group):
                    recursive_collect_feats(item)
                elif isinstance(item, h5py.Dataset):
                    if item[:].ndim == 1:
                        feats = item[:].reshape(1,-1)
                    else:
                        feats = item[:]
                    center_feat.append(feats)
        
        recursive_collect_feats(group)
        return center_feat
    
    for i in gid:
        center_feat = get_all_feats(f[i])
        if center_feat:
            f2[i] = np.concatenate(center_feat).mean(axis=0).reshape(1, -1)  # (1, d)
            # NOTE: for furture experiment
            # if 'biggait' in read_path or 'openset' in read_path:
            #     # biggait should use max pooling
            #     f2[i] = np.concatenate(center_feat).max(axis=0).reshape(1, -1)
            # else:
            #     f2[i] = np.concatenate(center_feat).mean(axis=0).reshape(1, -1)  # (1, d)
    f2.close()


def aggregate_mod_center_cc(read_path, save_path):
    f = h5py.File(read_path, 'r')
    f2 = h5py.File(save_path, 'w')
    # gid is the idx from train loader
    gids = sorted(list(f.keys()), key=int)
    for gid in gids:
        clothes_ids = sorted(list(f[gid].keys()), key=int)
        for c_id in clothes_ids:
            center_feat = f[gid][c_id][:]
            f2.create_dataset(f'{gid}/{c_id}', data=center_feat.mean(axis=0, keepdims=True))  # (1, d)
    f2.close()


def compute_mod_center():
    """
    Compute center feature for each person in dataset
    """
    config = parse_option()
    logging.info(f'Start computing {config.mode} center feature for {config.DATA.DATASET} datset')
    backbone_cfg = yaml.safe_load(open(config.backbone_cfg, 'r'))
    backbone_cfg['model_list'] = [config.mode] # overwrite
    model = QME(backbone_cfg)
    model = model.cuda()
    
    if config.DATA.DATASET == 'ccvid' or config.DATA.DATASET == 'mevid':
        dataset = build_dataset(config)
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = {}, {}, {}, {}
        for m in backbone_cfg['model_list']:
            spatial_transform_train_m, spatial_transform_test_m, temporal_transform_train_m, temporal_transform_test_m = get_transforms(m, config)
            spatial_transform_train[m] = spatial_transform_train_m
            spatial_transform_test[m] = spatial_transform_test_m
            temporal_transform_train[m] = temporal_transform_train_m
            temporal_transform_test[m] = temporal_transform_test_m
        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_test, temporal_transform_test, is_training=False),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_test, temporal_transform_test, is_training=False),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)

        file_path = os.path.join(config.save_path, f'{config.mode}_center_n{config.DATA.SAMPLING_STEP}_{config.DATA.DATASET}.h5')
        print(f'load {config.DATA.DATASET} dataset')
    else:
        raise ValueError('dataset not supported!')
    
    f = h5py.File(file_path, 'w')
    # get frame-level feature
    for batch_idx, (inputs, pids, camids, clothes_ids) in tqdm(enumerate(trainloader), total=len(trainloader), desc=""):
        print(inputs[config.mode].shape, pids, clothes_ids)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        feat_list, _ = model.get_feats(inputs, aggregated=False)
        if feat_list[config.mode] is None:
            continue
        feat_list = feat_list[config.mode].cpu().numpy()
        for pid, feat in zip(pids, feat_list):
            f.create_dataset(f'{pid}/{batch_idx}', data=feat)
            f.flush()
    f.close()
    # aggregated into subject-level center feat
    # aggregate_mod_center(file_path, file_path.replace(f'_n{config.DATA.SAMPLING_STEP}', ''))


def compute_mod_center_cc():
    """
    Compute center feature based on clothes ids for each person in dataset
    
    """
    config = parse_option()
    logging.info(f'Start computing {config.mode} center feature for {config.DATA.DATASET} datset')
    backbone_cfg = yaml.safe_load(open(config.backbone_cfg, 'r'))
    backbone_cfg['model_list'] = [config.mode] # overwrite
    model = QME(backbone_cfg)
    model = model.cuda()
    
    if config.DATA.DATASET == 'ccvid' or config.DATA.DATASET == 'mevid':
        dataset = build_dataset(config)
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = {}, {}, {}, {}
        for m in backbone_cfg['model_list']:
            spatial_transform_train_m, spatial_transform_test_m, temporal_transform_train_m, temporal_transform_test_m = get_transforms(m, config,dataset_type='vid')
            spatial_transform_train[m] = spatial_transform_train_m
            spatial_transform_test[m] = spatial_transform_test_m
            temporal_transform_train[m] = temporal_transform_train_m
            temporal_transform_test[m] = temporal_transform_test_m
        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            # use temporal_transform_train to make sure length of each clip is the same
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_test, temporal_transform_train, is_training=False),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_test, temporal_transform_test, is_training=False),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)

        print(f'load {config.DATA.DATASET} dataset')
    elif 'ltcc' in config.DATA.DATASET:
        dataset = build_dataset(config)
        transform_train, transform_test = {}, {}
        for m in backbone_cfg['model_list']:
            transform_train_m, transform_test_m, _, _ = get_transforms(m, config, dataset_type='img')
            transform_train[m] = transform_train_m
            transform_test[m] = transform_test_m
        train_sampler = DistributedRandomIdentitySampler(dataset.train['dataset'], 
                                                         num_instances=config.DATA.NUM_INSTANCES, 
                                                         seed=config.SEED)
        trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train, is_training=False),
                                 sampler=train_sampler,
                                 batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=True)     
    else:
        raise ValueError('dataset not supported!')
    
    file_path = os.path.join(config.save_path, f'{config.mode}_center_n{config.DATA.SAMPLING_STEP}_{config.DATA.DATASET}_CC.h5')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    f = h5py.File(file_path, 'w')
    # get frame-level feature
    for batch_idx, (inputs, pids, camids, clothes_ids) in tqdm(enumerate(trainloader), total=len(trainloader), desc=""):
        # print(inputs[config.mode].shape, pids, camids, clothes_ids)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        feat_list, _ = model.get_feats(inputs, aggregated=False)
        if feat_list[config.mode] is None:
            continue
        feat_list = feat_list[config.mode].cpu().numpy()  # (bs, T, d)
        for pid, clothes_id, camid, feat in zip(pids, clothes_ids, camids, feat_list):
            group_name = f'{pid}/{camid}_{clothes_id}'
            if group_name in f:
                # If the group exists, append the new feature
                existing_data = f[group_name][:]
                new_data = np.concatenate((existing_data, feat), axis=0)
                del f[group_name]  # Remove the old dataset
                f.create_dataset(group_name, data=new_data)  # Create a new dataset with merged data
            else:
                # If the group does not exist, create a new dataset
                f.create_dataset(group_name, data=feat)
            f.flush()
    f.close()
    
    # aggregated into subject-level center feat
    aggregate_mod_center_cc(file_path, file_path.replace(f'_n{config.DATA.SAMPLING_STEP}', ''))


def parse_option():
    parser = argparse.ArgumentParser(description='')
    # precompute settings
    parser.add_argument('--save_path', type=str, default='./mod_center_feat/')
    parser.add_argument('--train_batch', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=100)
    # model settings
    parser.add_argument('--backbone_cfg', type=str, default='./configs/backbone_cfg-ccvid.yaml')
    # parser.add_argument('--train_cfg', type=str, default='./configs/train_cfg.yaml')
    parser.add_argument('--mode', type=str, default='adaface', help='model version name')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory", default='/data')
    parser.add_argument('--dataset', type=str, default='mevid', help="ccvid, mevid, or ltcc", choices=['ccvid', 'mevid', 'ltcc'])

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)
    return config



if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29610'
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s : %(message)s', 
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    config = parse_option()
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    compute_mod_center_cc()