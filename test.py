import h5py
import yaml
import time
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as osp
import torch
import torch.nn.functional as F
from torch import distributed as dist
from pytorch_lightning import seed_everything
from model import sim_fn, QME, MODEL_DIM_KEYS
from data import build_dataloader, VID_DATASET
from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from tools.eval_metrics import *


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(model, dataloader):
    assert len(model.model_list) == 1, "Only allow ONE MODEL in model_list for inference"
    logger = logging.getLogger('reid.test')
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
        if (batch_idx + 1) % 100==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        imgs = {k: v.cuda() for k, v in imgs.items()}
        batch_features, _ = model.get_feats(imgs)  # (B, d)
        batch_features = batch_features[model.model_list[0]]
        if batch_features is None:
            # NaN feature if no face is detected
            batch_features = torch.full((1, MODEL_DIM_KEYS[model.model_list[0]]), float('nan'))
        # flip_imgs = torch.flip(imgs, [3])
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        # batch_features = model(imgs)
        # batch_features_flip = model(flip_imgs)
        # batch_features += batch_features_flip
        # batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    assert len(model.model_list) == 1, "Only allow ONE MODEL in model_list for inference"
    logger = logging.getLogger('reid.test')
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
        # if (batch_idx + 1) % 100==0:
        #     logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = {k: v.cuda() for k, v in vids.items()}
        batch_features, _ = model.get_feats(vids)  # (B, d)
        batch_features = batch_features[model.model_list[0]]
        if batch_features is None:
            # NaN feature if no face is detected
            batch_features = torch.full((1, MODEL_DIM_KEYS[model.model_list[0]]), float('nan'))
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        valid_mask = ~torch.isnan(clip_features[idx[0] : idx[1], :])
        if valid_mask.any():
            valid_feats = clip_features[idx[0] : idx[1], :][valid_mask].view(-1, clip_features.size(1))
            features[i] = valid_feats.mean(0)
        else:
            # NaN feature if no face is detected for all clips
            features[i] = torch.full((clip_features.size(1),), float('nan'))
            
        # features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        # features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def extract_test_feats(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query['dataset']))
        torch.cuda.empty_cache()
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery['dataset']))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query['dataset']))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery['dataset']))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    # save extracted features
    fname = osp.join('./test_feats', f"{'_'.join(model.model_list)}_{config.DATA.DATASET}_test.h5")
    f = h5py.File(fname, 'w')
    f['qf'] = qf.cpu()
    f['q_pids'] = q_pids.cpu()
    f['q_camids'] = q_camids.cpu()
    f['q_clothes_ids'] = q_clothes_ids.cpu()
    f['gf'] = gf.cpu()
    f['g_pids'] = g_pids.cpu()
    f['g_camids'] = g_camids.cpu()
    f['g_clothes_ids'] = g_clothes_ids.cpu()
    f.close()
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return fname


def get_scoremat_from_feats(h5_file_path, mode='biggait', dataset='ccvid', save_scoremat=False):
    # read extracted features
    f = h5py.File(h5_file_path, 'r')
    qf, q_pids, q_camids, q_clothes_ids = f['qf'][:], f['q_pids'][:], f['q_camids'][:], f['q_clothes_ids'][:]
    gf, g_pids, g_camids, g_clothes_ids = f['gf'][:], f['g_pids'][:], f['g_camids'][:], f['g_clothes_ids'][:]
    f.close()
    
    # Get tracklet-level similarity score matrix
    qf = torch.tensor(qf)
    gf = torch.tensor(gf)
    m, n = qf.size(0), gf.size(0)
    score_mat = torch.zeros((m,n))
    for i in range(m):
        score_mat[i] = sim_fn(qf[None, i], gf, mode)
    
    # NaN feature will be replaced with 0
    score_mat = np.nan_to_num(score_mat.numpy())
    
    # Get subject-level similarity score matrix
    # find the same g_pid in g_pids then merge gallery features
    unique_g_pids = np.unique(g_pids)
    unique_gf = np.zeros((len(unique_g_pids), gf.shape[1]))
    for i, pid in enumerate(unique_g_pids):
        idx = np.where(g_pids == pid)[0]
        # merge gallery feature using average pooling
        unique_gf[i] = np.nanmean(gf[idx], axis=0, keepdims=True)

    unique_gf = torch.from_numpy(unique_gf)
    merge_score_mat = torch.zeros((m, len(unique_g_pids)))
    for i in range(m):
        merge_score_mat[i] = sim_fn(qf[None, i], unique_gf, mode)
    merge_score_mat = merge_score_mat.numpy()
    merge_score_mat = np.nan_to_num(merge_score_mat)
    
    if save_scoremat:
        # save score_mat and merge_score_mat
        f = h5py.File(f'./test_feats/scoremat_{mode}_{dataset}_test.h5', 'w')
        f.create_dataset(f'score_mat', data=score_mat)
        f.create_dataset(f'merge_score_mat', data=merge_score_mat)
        f.create_dataset(f'q_pids', data=q_pids)
        f.create_dataset(f'q_camids', data=q_camids)
        f.create_dataset(f'q_clothes_ids', data=q_clothes_ids)
        f.create_dataset(f'g_pids', data=g_pids)
        f.create_dataset(f'g_camids', data=g_camids)
        f.create_dataset(f'g_clothes_ids', data=g_clothes_ids)
        f.create_dataset(f'unique_g_pids', data=unique_g_pids)
        f.flush()
        f.close()
    
    return score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids)
    

def test_score(score_mat, merge_score_mat, 
               q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, 
               dataset='ccvid'):
    
    seed_everything(45)
    result = {}
    logger = logging.getLogger('reid.test')  
    logger.info("Computing CMC and mAP for general")
    cmc, mAP = evaluate(score_mat, q_pids, g_pids, q_camids, g_camids)
    logger.info("----------------- General Results --------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("------------------------------------------------------------")
    gen_res = pd.Series({'GR_top1': cmc[0], 'GR_top5': cmc[4], 'GR_top10': cmc[9], 'GR_top20': cmc[19], 'GR_mAP': mAP})
    if dataset in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(score_mat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    logger.info("-------------------- SC Results ----------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("------------------------------------------------------------")
    sc_res = pd.Series({'SC_top1': cmc[0], 'SC_top5': cmc[4], 'SC_top10': cmc[9], 'SC_top20': cmc[19], 'SC_mAP': mAP})
    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(score_mat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("-------------------- CC Results ----------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("------------------------------------------------------------")
    cc_res = pd.Series({'CC_top1': cmc[0], 'CC_top5': cmc[4], 'CC_top10': cmc[9], 'CC_top20': cmc[19], 'CC_mAP': mAP})
    # Ours evaluation metric
    logger.info("Computing Biometric Performance on EVP1.0.0")

    # Compute TAR@FAR
    verification_result = compute_tar_at_far(score_mat, q_pids, g_pids)
    # Compute FNIR@FPIR (for openset evaluation, gallery should be unique)
    opensearch_result = pd.DataFrame()
    for i in range(10): # sample 10 times
        # if dataset == 'ccvid':
        #     openset_pids = np.sort(np.random.choice(unique_g_pids, int(len(unique_g_pids) * 0.8), replace=False))
        # elif dataset == 'mevid':
        openset_pids = np.sort(np.random.choice(unique_g_pids, int(len(unique_g_pids) * 0.8), replace=False))
        tmp = compute_fnir_at_fpir(merge_score_mat, q_pids, unique_g_pids, openset_pids)
        opensearch_result = pd.concat([opensearch_result, tmp], axis=1)
    # TODO: add std
    opensearch_result_mean = opensearch_result.T.mean(axis=0)
    opensearch_result_std = opensearch_result.T.std(axis=0)[[r'TPIR@0.01%FPIR', r'TPIR@0.10%FPIR', r'TPIR@1.00%FPIR', r'FNIR@0.01%FPIR', r'FNIR@0.10%FPIR', r'FNIR@1.00%FPIR']]
    opensearch_result_std.index = [f'{i}_std' for i in opensearch_result_std.index]
    result = pd.concat([gen_res, sc_res, cc_res, verification_result, opensearch_result_mean, opensearch_result_std], axis=0)
    logger.info("--------------- Evaluation Protocol V1.0.0 -----------------")
    logger.info('Number of Probes: {}, Number of Galleries subjects: {}'.format(int(result['num_probe_templates']), int(result['num_gallery_subjects'])))
    # logger.info('Rank1: {:.1%} Rank10: {:.1%} Rank20: {:.1%} mAP: {:.1%}'.format(result['Rank1'], result['Rank10'], result['Rank20'], result['mAP']))
    logger.info('Tar@1.00%FAR:{:.1%} Rank1:{:.1%} FNIR@1%FPIR:{:.1%}±{:.1%}'.format(result[r'TAR@1.00%FAR'], result['GR_top1'], result[r'FNIR@1.00%FPIR'], result[r'FNIR@1.00%FPIR_std']))
    logger.info("------------------------------------------------------------")
    return result


def collect_scoremat(model_list=['biggait', 'cal', 'adaface'], dataset='ccvid'):
    f = h5py.File(f'./test_feats/scoremats_{dataset}.h5', 'w')
    meta_written = False

    for model_name in model_list:
        score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(f'./test_feats/{model_name}_{dataset}_test.h5', model_name)
        f.create_dataset(f'{model_name}/score_mat', data=score_mat)
        f.create_dataset(f'{model_name}/merge_score_mat', data=merge_score_mat)
        if not meta_written:
            f.create_dataset('q_pids', data=q_pids)
            f.create_dataset('q_camids', data=q_camids)
            f.create_dataset('q_clothes_ids', data=q_clothes_ids)
            f.create_dataset('g_pids', data=g_pids)
            f.create_dataset('g_camids', data=g_camids)
            f.create_dataset('g_clothes_ids', data=g_clothes_ids)
            f.create_dataset('unique_g_pids', data=unique_g_pids)
            meta_written = True
        f.flush()
    f.close()    


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--mode', type=str, default='aim', help='model type/version name')

    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory", default='/data')
    parser.add_argument('--dataset', type=str, default='mevid', help="ccvid, mevid", choices=['ccvid', 'mevid', 'ltcc'])
    parser.add_argument('--eval_mode', type=str, default='score', choices=['score', 'feat', 'gather'], help='Evaluation type: [score, feat, gather]. feat: extract features for test set, score: compute score matrices for test set, gather: collect score matrices for all models')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29620'
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s : %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    config = parse_option()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    
    # collect_scoremat(['adaface', 'arcface', 'cal-mevid', 'agrl'], dataset='mevid')
    # collect_scoremat(['adaface', 'arcface', 'cal-ccvid', 'biggait'], dataset='ccvid')

    if config.eval_mode == 'feat':

        # /---------------Feature Inference Func-------------------/
        backbone_cfg = yaml.safe_load(open(f'./configs/backbone_cfg-{config.DATA.DATASET}.yaml', 'r'))
        backbone_cfg['model_list'] = [config.mode] # Overwrite model list, only use one model to extract test features
        print(config.DATA.DATASET, config.mode)
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(backbone_cfg['model_list'], config)
        model = QME(backbone_cfg)
        model = model.cuda()
        fname = extract_test_feats(config, model, queryloader, galleryloader, dataset)
        score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(fname, mode=model.model_list[0], dataset=config.DATA.DATASET)
        res = test_score(score_mat, merge_score_mat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, config.DATA.DATASET)
    elif config.eval_mode == 'score':
        # /---------------Score Evaluate Func-------------------/
        saved_feat = f'./test_feats/{config.mode}_{config.DATA.DATASET}_test.h5'
        score_mat, merge_score_mat, (q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids) = get_scoremat_from_feats(saved_feat, mode=config.mode, dataset=config.DATA.DATASET)
        res = test_score(score_mat, merge_score_mat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, config.DATA.DATASET)
    elif config.eval_mode == 'gather':
        collect_scoremat(config.mode.split(','), dataset=config.DATA.DATASET)
    else:
        raise ValueError(f'Invalid evaluation type: {config.eval_mode}')