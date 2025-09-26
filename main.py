import yaml
import argparse
import logging
import wandb
import copy
import torch.distributed as dist
import torch.optim as optim
from data import build_dataloader, VID_DATASET
from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from model import QME, MODEL_DIM_KEYS, Face_Quality_Estimator, Quality_Estimator
from test import *
from loss import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_or_none(value):
    if value == 'None':
        return None
    return value

def load_center_feat(h5_file, labels, model_name):
    # load subject-level center feat
    f = h5py.File(h5_file, 'r')
    center_feat = []
    for id in labels:
        if id in f:
            center_feat.append(f[id][:])
        else:
            # subject 102 in MEVID does not have face imgs
            center_feat.append(np.zeros((1, MODEL_DIM_KEYS[model_name]), dtype=np.float32))
    center_feat = np.array(center_feat).squeeze()
    return center_feat

def load_center_feat_cc(h5_file, labels, model_name):
    # load clothing-level center feat 
    f = h5py.File(h5_file, 'r')
    center_feat, center_pids,  = [], []
    center_keys = []
    for id in labels:
        if id in f:
            camids_clothesids = sorted(list(f[id].keys()))
            for cid in camids_clothesids:
                center_feat.append(f[id][cid][:])
                center_pids.append(int(id))
                center_keys.append(f'{id}_{cid}')

        # else:
        #     # subject 102 in MEVID does not have face imgs
        #     center_feat.append(np.zeros((1, MODEL_DIM_KEYS[model_name]), dtype=np.float32))
        #     center_pids.append(int(id))
        #     center_camids.append(int(cid.split('_')[0]))
        #     center_clothes_ids.append(int(cid.split('_')[1]))
            
    center_feat = np.array(center_feat).squeeze()
    center_pids = np.array(center_pids)
    # center_camids = np.array(center_camids)
    # center_clothes_ids = np.array(center_clothes_ids)
    center_keys = np.array(center_keys)
    return center_feat, center_pids, center_keys

@torch.no_grad()
def test(model, queryloader, vid2clip_index, data_length, dataset_name):
    result = {}
    if 'mod_qe' in model.mode:
        if dataset_name in VID_DATASET:
            clip_weights, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
            print('start inference query weights...')
            for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(queryloader):
                bs = len(batch_pids)
                vids = {k: v.cuda() for k, v in vids.items()}
                _, interm_feat_list = model.get_feats(vids)  # (B, d)
                for model_name in model.model_list:
                    if 'kprpe' in model_name or 'adaface' in model_name:
                        face_interm_feat = interm_feat_list[model_name]
                        if face_interm_feat is None:
                            # NaN if no face is detected
                            clip_weight = torch.full((bs, 1), float('nan')) # (B, 1)
                        else:
                            clip_weight = model.fgb(face_interm_feat)['face_weights'] # (B, 1)
                    elif 'cal' in model_name:
                        body_interm_feat = interm_feat_list[model_name]
                        clip_weight = model.fgb(body_interm_feat) # (B, 1)
                # if face_interm_feat is None:
                #     # NaN if no face is detected
                #     clip_weight = torch.full((bs, 1), float('nan')) # (B, 1)
                # else:
                #     clip_weight = model.fgb(face_interm_feat)['face_weights'] # (B, 1)
                clip_weights.append(clip_weight.cpu())
                clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
                clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
                clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
            clip_weights = torch.cat(clip_weights, 0)

            # Gather samples from different GPUs
            clip_weights, clip_pids, clip_camids, clip_clothes_ids = \
                concat_all_gather([clip_weights, clip_pids, clip_camids, clip_clothes_ids], data_length)

            # Use the averaged feature of all clips split from a video as the representation of this original full-length video
            features = torch.zeros(len(vid2clip_index), clip_weights.size(1)).cuda()
            clip_weights = clip_weights.cuda()
            pids = torch.zeros(len(vid2clip_index))
            camids = torch.zeros(len(vid2clip_index))
            clothes_ids = torch.zeros(len(vid2clip_index))
            for i, idx in enumerate(vid2clip_index):
                valid_mask = ~torch.isnan(clip_weights[idx[0] : idx[1], :])
                if valid_mask.any():
                    valid_feats = clip_weights[idx[0] : idx[1], :][valid_mask].view(-1, clip_weights.size(1))
                    features[i] = valid_feats.mean(0)
                else:
                    # zero weight if no face is detected for the whole video
                    features[i] = torch.full((clip_weights.size(1),), float(0))
                pids[i] = clip_pids[idx[0]]
                camids[i] = clip_camids[idx[0]]
                clothes_ids[i] = clip_clothes_ids[idx[0]]
            features = features.cpu()
        else:
            clip_weights, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
            print('start inference query weights...')
            for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(queryloader):
                bs = len(batch_pids)
                vids = {k: v.cuda() for k, v in vids.items()}
                _, interm_feat_list = model.get_feats(vids)  # (B, d)
                if isinstance(model.fgb, Face_Quality_Estimator):
                    for model_name in model.model_list:
                        if 'kprpe' in model_name or 'adaface' in model_name:
                            face_interm_feat = interm_feat_list[model_name]
                            if face_interm_feat is None:
                                # zero weight if no face is detected
                                clip_weight = torch.full((bs, 1), float(0)) # (B, 1)
                            else:
                                clip_weight = model.fgb(face_interm_feat)['face_weights'] # (B, 1)
                            break
                elif isinstance(model.fgb, Quality_Estimator):
                    for model_name in model.model_list:
                        if 'cal' in model_name:
                            body_interm_feat = interm_feat_list[model_name]
                            clip_weight = model.fgb(body_interm_feat) # (B, 1)
                            break
                # if face_interm_feat is None:
                #     # zero weight if no face is detected
                #     clip_weight = torch.full((bs, 1), float(0)) # (B, 1)
                # else:
                #     clip_weight = model.fgb(face_interm_feat)['face_weights'] # (B, 1)
                clip_weights.append(clip_weight.cpu())
                clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
                clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
                clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
            features = torch.cat(clip_weights, 0)
        result['overall'] = 0
        result['face_weights'] = features
        
    elif 'score' in model.mode:
        f = h5py.File(f'./test_feats/scoremats_{dataset_name}.h5', 'r')
        score_mats, merge_score_mats = {}, {}
        q_pids = f['q_pids'][:]
        q_camids = f['q_camids'][:]
        q_clothes_ids = f['q_clothes_ids'][:]
        g_pids = f['g_pids'][:]
        g_camids = f['g_camids'][:]
        g_clothes_ids = f['g_clothes_ids'][:]
        unique_g_pids = f['unique_g_pids'][:]
        for model_name in model.model_list:
            score_mats[model_name] = torch.from_numpy(f[model_name]['score_mat'][:])
            merge_score_mats[model_name] = torch.from_numpy(f[model_name]['merge_score_mat'][:])
        f.close()
        if dataset_name == 'ccvid':
            # f = h5py.File('./test_feats/mod_norm_adaface_ccvid.h5', 'r') # feature norm
            f = h5py.File('./test_feats/mod_qe_adaface-t1r3_ccvid_3.h5', 'r') # Adaface-QE
            # f = h5py.File('./test_feats/mod_qe_cal_t1r3_ccvid_6000.h5', 'r') # CAL-QE
        elif dataset_name == 'mevid':
            # f = h5py.File('./test_feats/mod_norm_adaface_mevid.h5', 'r') # feature norm
            f = h5py.File('./test_feats/mod_qe_adaface_t1r3_mevid_mevid_6000.h5', 'r') # Adaface-QE
            # f = h5py.File('./test_feats/mod_qe_cal_t1r3_mevid_6000.h5', 'r') # CAL-QE
        elif dataset_name == 'ltcc':
            f = h5py.File('./test_feats/mod_qe_adaface_t1r3_ltcc_6000.h5', 'r')
        face_weights = torch.from_numpy(f['face_weights'][:]) # (num_q, 1)
        fuse_score_mats = model.fgb.inference(score_mats, face_weights, None, None)['fuse_scores'].cpu().numpy()
        fuse_merge_score_mats = model.fgb.inference(merge_score_mats, face_weights, None, None)['fuse_scores'].cpu().numpy()        
        
        result = test_score(fuse_score_mats, fuse_merge_score_mats, 
               q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids, unique_g_pids, 
               dataset=dataset_name)
    return result

def train(config, train_cfg, backbone_cfg):
    # config = parse_option()
    # train_cfg = yaml.safe_load(open(config.train_cfg, 'r'))
    if not config.debug:
        wandb.init(project=config.log_prefix, config=train_cfg, dir='/mnt/scratch/zhujie4', name=config.run_name)
        if train_cfg is None:
            train_cfg = wandb.config
    else:
        wandb.init(mode='disabled')
    wandb.run.log_code(root='./', include_fn=lambda path: path.endswith(".py") or path.endswith('.yaml') )  # log .py
    print(f'{config.description}')
    print(
        f'epoch:{config.epoch}, test per step:{config.test_per_step}, '
        f'batch size:{config.DATA.TRAIN_BATCH}, num_sample:{config.DATA.SAMPLING_STEP} lr:{train_cfg["lr_rate"]}, norm_method:{train_cfg["norm_method"]}')
    seed_everything(train_cfg['seed'])
    
    # init dataloader
    trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(backbone_cfg['model_list'], config)
    
    # init model
    model = QME(backbone_cfg, num_experts=train_cfg['num_experts'], mlp_ratio=train_cfg['mlp_ratio'], 
                 out_dim=train_cfg['out_dim'], dropout_rate=train_cfg['dropout_rate'], mode=config.mode, use_qe=config.use_qe)
    model = model.to(config.device)
    
    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2)
    # if config.resume_from_ckpt:
    #     model.load_ckpt(config.resume_from_ckpt)
    #     print(f'resume from ckpt {config.resume_from_ckpt}')

    # init loss function
    if 'score' in model.mode or 'mod_qe' in model.mode:
        # load center feat
        center_feats = {}
        center_pids = {}
        center_keys = {}
        for model_name in backbone_cfg['model_list']:
            _center_feats, _center_pids, _center_keys= load_center_feat_cc(f'./mod_center_feat/{model_name}_center_{config.DATA.DATASET}_CC.h5', 
                                                        [str(i) for i in range(dataset.num_train_pids)], model_name)
            center_feats[model_name] = torch.tensor(_center_feats).to(config.device)
            center_pids[model_name] = _center_pids
            center_keys[model_name] = _center_keys
        # make sure center_labels are the same
        if not all(len(center_pids[model_name]) == len(center_pids[backbone_cfg['model_list'][0]]) for model_name in backbone_cfg['model_list']):
            ref_model_idx = np.array([len(center_pids[model_name]) for model_name in backbone_cfg['model_list']]).argmax()
            ref_model_name = backbone_cfg['model_list'][ref_model_idx]
            for model_name in backbone_cfg['model_list']:
                if center_feats[model_name].shape[0] != center_feats[ref_model_name].shape[0]:
                    _center_feats = torch.zeros((center_feats[ref_model_name].shape[0], center_feats[model_name].shape[1]), device=config.device)
                    _center_pids = np.empty_like(center_pids[ref_model_name])
                    _center_keys = np.empty_like(center_keys[ref_model_name])
                    for i, (pids, keys) in enumerate(zip(center_pids[ref_model_name], center_keys[ref_model_name])):
                        if keys in center_keys[model_name]:
                            _center_feats[i] = center_feats[model_name][center_keys[model_name] == keys]
                        _center_pids[i] = pids
                        _center_keys[i] = keys
                    center_feats[model_name] = _center_feats
                    center_pids[model_name] = _center_pids
                    center_keys[model_name] = _center_keys
        center_pids = torch.tensor(center_pids[backbone_cfg['model_list'][0]]).to(config.device)
        center_camids = torch.tensor([int(k.split('_')[1]) for k in center_keys[backbone_cfg['model_list'][0]]]).to(config.device)
        center_clothes_ids = torch.tensor([int(k.split('_')[2]) for k in center_keys[backbone_cfg['model_list'][0]]]).to(config.device)
        criterion = ScoreTripletLoss(margin=train_cfg['triplet_margin'], s=1.0, rank_threshold=config.rank_threshold)
    else:
        raise NotImplementedError('loss function not implemented')
    best_overall = -float('inf')
    total_step = 0
    for epoch in range(config.epoch):
        print(f'Epoch {epoch+1}/{config.epoch}')
        model.train()
        acc, total = 0, 0
        for step, data in enumerate(trainloader):
            total_step += 1
            optimizer.zero_grad()
            inputs, q_pids, q_camids, q_clothes_ids = data
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            log_dict = {}
            
            res = model(inputs, center_feats, labels=q_pids, norm_method=train_cfg['norm_method'])
                
            # Loss calculation
            if 'score' in model.mode:
                loss_dict = criterion(res, q_pids, center_pids)
                log_dict['score_loss'] = loss_dict['score_loss']
                loss = loss_dict['total_loss'] 
                total += len(q_pids)
                acc += 0
            elif 'mod_qe' in model.mode:
                loss = .0
                # calculate score loss and rank loss
                if 'face_weights' in res.keys():
                    print('face:')
                    # f_rank_loss = criterion.cal_rank_loss(res['face_weights'], res['face_scores'], q_pids)
                    f_rank_loss = criterion.cal_rank_loss_cc(res['face_weights'], res['face_scores'], q_pids, center_pids)
                    loss += f_rank_loss
                if 'gait_weights' in res.keys():
                    print('gait:')
                    g_rank_loss = criterion.cal_rank_loss_cc(res['gait_weights'], res['gait_scores'], q_pids)
                    loss += g_rank_loss
                if 'body_weights' in res.keys():
                    print('body:')
                    b_rank_loss = criterion.cal_rank_loss_cc(res['body_weights'], res['body_scores'], q_pids, center_pids)
                    loss += b_rank_loss
                total += len(q_pids)
                acc += 0
                
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + step / len(trainloader))
            torch.cuda.empty_cache()
            log_dict['train_step_loss'] = loss.item()
            # if config.debug:
            #     break
            
            # Evaluate
            if (total_step + 1) % config.test_per_step == 0:
                model.eval()
                if 'mod_qe' in model.mode:
                    # inference face weights for test set
                    if config.DATA.DATASET in VID_DATASET:
                        test_res = test(model, queryloader, dataset.query_vid2clip_index, len(dataset.recombined_query['dataset']), config.DATA.DATASET)
                    else:
                        test_res = test(model, queryloader, None, None, config.DATA.DATASET)
                    # save face weights to h5 file
                    h5_file = f'./test_feats/{model.mode}_{config.run_name}_{config.DATA.DATASET}_{total_step+1}.h5'
                    f = h5py.File(h5_file, 'w')
                    f['face_weights'] = test_res['face_weights']
                    f.close()
                    # save qe model
                    model.save_ckpt(f'./checkpoints/MEME/{config.log_prefix}-{config.run_name}-{config.DATA.DATASET}-{total_step+1}.pth')
                elif 'score' in model.mode:
                    test_res = test(model, None, None, None, config.DATA.DATASET)
                    log_dict['GR_top1'] = test_res['GR_top1']
                    log_dict['GR_mAP'] = test_res['GR_mAP']
                    log_dict['CC_top1'] = test_res['CC_top1']
                    log_dict['CC_mAP'] = test_res['CC_mAP']
                    log_dict[r'TAR@1.00%FAR'] = test_res[r'TAR@1.00%FAR']
                    log_dict[r'FNIR@1.00%FPIR'] = test_res[r'FNIR@1.00%FPIR']
                    log_dict['GR_overall'] = test_res['GR_top1'] + test_res['GR_mAP'] + test_res[r'TAR@1.00%FAR'] - test_res[r'FNIR@1.00%FPIR']
                    log_dict['CC_overall'] = test_res['CC_top1'] + test_res['CC_mAP']
                    gr_overall = log_dict['GR_overall'] * 100
                    cc_overall = log_dict['CC_overall'] * 100
                    overall = gr_overall
                    log_dict['overall'] = overall
                    print(f'total_step {total_step+1}, GR_overall: {gr_overall:.2f}, CC_overall: {cc_overall:.2f}')
                    if overall > best_overall:
                        best_overall = overall
                        best_top1 = test_res['GR_top1']
                        best_map = test_res['GR_mAP']
                        best_cctop1 = test_res['CC_top1']
                        best_ccmap = test_res['CC_mAP']
                        best_tar = test_res[r'TAR@1.00%FAR']
                        best_fnir = test_res[r'FNIR@1.00%FPIR']
                        best_fnir_std = test_res[r'FNIR@1.00%FPIR_std']
                        early_stop_counter = 0
                        best_model = copy.deepcopy(model)
                        best_step = total_step
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= config.early_stop_patience:
                            print('Early stopping triggered. Stop training.')
                            log_dict['best_overall'] = best_overall
                            logging.info(f'important params: train_batch: {config.DATA.TRAIN_BATCH}, num_sample: {config.DATA.SAMPLING_STEP}')
                            logging.info(f'best top1: {best_top1:.1%}, best mAP: {best_map:.1%}, best cc top1: {best_cctop1:.1%}, best cc mAP: {best_ccmap:.1%}, best TAR: {best_tar:.1%}, best FNIR: {best_fnir:.1%}±{best_fnir_std:.1%}, best overall: {best_overall:.2f}')
                            os.makedirs('./checkpoints/QME', exist_ok=True)
                            best_model.save_ckpt(f'./checkpoints/QME/{config.log_prefix}-{config.DATA.DATASET}-bs{config.DATA.TRAIN_BATCH}-seq{config.DATA.SAMPLING_STEP}-{best_overall:.2f}-{best_step+1}.pth')
                            wandb.log(log_dict)
                            return  

            wandb.log(log_dict)

            if config.max_step and total_step >= config.max_step:
                print(f'Reached max_step {config.max_step}. Stop training.')
                if 'score' in model.mode and 'best_model' in locals():
                    log_dict['best_overall'] = best_overall
                    logging.info(f'important params: train_batch: {config.DATA.TRAIN_BATCH}, num_sample: {config.DATA.SAMPLING_STEP}')
                    logging.info(f'best top1: {best_top1:.1%}, best mAP: {best_map:.1%}, best cc top1: {best_cctop1:.1%}, best cc mAP: {best_ccmap:.1%}, best TAR: {best_tar:.1%}, best FNIR: {best_fnir:.1%}±{best_fnir_std:.1%}, best overall: {best_overall:.2f}')
                    os.makedirs('./checkpoints/QME', exist_ok=True)
                    best_model.save_ckpt(f'./checkpoints/QME/{config.log_prefix}-{config.DATA.DATASET}-bs{config.DATA.TRAIN_BATCH}-seq{config.DATA.SAMPLING_STEP}-{best_overall:.2f}-{best_step+1}.pth')
                    wandb.log(log_dict)
                return  
 
def parse_option():
    parser = argparse.ArgumentParser(description='')
    # Training
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--max_step', type=int, default=1500)
    parser.add_argument('--test_per_step', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    # parser.add_argument('--train_batch', type=int, default=4)
    # parser.add_argument('--num_sample', type=int, default=3)
    # model settings
    parser.add_argument('--backbone_cfg', type=str, default='./configs/backbone_cfg-ltcc.yaml')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_cfg-ltcc.yaml')
    parser.add_argument('--mode', type=str, default='mod_qe', choices=['mod_qe', 'score_loss'], help='model version name. score_loss for training quality estimator, mod_qe: training QME with quality estimator')
    parser.add_argument('--use_qe', type=str2bool, default=True)
    parser.add_argument('--rank_threshold', type=int, default=3)
    # parser.add_argument('--resume_from_ckpt', type=str, default='')

    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory", default='/data')
    parser.add_argument('--dataset', type=str, default='ltcc', help="ccvid, mevid, ltcc", choices=['ccvid', 'mevid', 'ltcc'])
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    
    # log settings
    parser.add_argument('--log_prefix', type=str, default='lsf')
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--debug', type=str2bool, default=True)

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29600'
    logging.basicConfig(
        # filename='ccvid-cal_qe-1129.log',
        # filemode='a',
        format='%(asctime)s - %(levelname)s : %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    
    config = parse_option()
    train_cfg = yaml.safe_load(open(config.train_cfg, 'r'))
    backbone_cfg = yaml.safe_load(open(config.backbone_cfg, 'r'))
    # overwrite bs and num_sample
    config.defrost()
    if 'train_batch' in train_cfg.keys():
        config.DATA.TRAIN_BATCH = train_cfg['train_batch']
    if 'num_sample' in train_cfg.keys():
        config.DATA.SAMPLING_STEP = train_cfg['num_sample']
    config.freeze()
    

    train(config, train_cfg, backbone_cfg)

    # DEBUG code
    # init dataset
    # config = parse_option()
    # backbone_cfg = yaml.safe_load(open('./configs/backbone_cfg.yaml', 'r'))
    # trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(backbone_cfg['model_list'], config)
    
    # init model
    # model = QME(backbone_cfg)
    # model = model.cuda()
    
    # print('/---------------Training set--------------------/')
    # for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
    #     print(batch_idx, [imgs[model_name].shape for model_name in backbone_cfg['model_list']], pids)
    #     if batch_idx > 3:
    #         break

    
    # print('/---------------Query set--------------------/')
    # for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(queryloader):
    #     print(batch_idx, imgs[backbone_cfg['model_list'][0]].shape, pids)
    #     if batch_idx > 3:
    #         break
        
    # print('/---------------Gallery set--------------------/')
    # for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(galleryloader):
    #     print(batch_idx, imgs[backbone_cfg['model_list'][0]].shape, pids)
    #     if batch_idx > 3:
    #         break
    


