import os
import einops
import yaml
import numpy as np
from math import pi
from functools import partial
from timm.layers import trunc_normal_
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ParameterDict
from itertools import combinations

from WBModules import * 
from tools.eval_metrics import *


MODEL_MAPPING_DICT = {'biggait': 'body_data',
                      'cal': 'body_data',
                      'cal-ccvid': 'body_data',
                      'cal-mevid': 'body_data',
                      'cal-ltcc': 'body_data',
                      'agrl': 'body_data',
                      'aim': 'body_data',
                      'adaface': 'face_data'}

MODEL_DIM_KEYS = {'adaface': 512, 'kprpe': 512, 'arcface': 512, 'insightface': 512}


def hook_fn(module, input, output):
    if isinstance(output, list):
        for i, tensor in enumerate(output):
            module.register_buffer(f'hook_output_{i}', tensor)
    else:
        module.register_buffer('hook_output', output)

def normalize_embeddings(a, eps=1e-5):
    # a dim: (N, d)
    a_n = torch.clip(a.norm(dim=-1)[:, None], min=eps)
    a_norm = a / a_n
    return a_norm

def build_face_backbone(backbone_cfg, mode='kprpe'):
    """
    ckpt_idr must be absolute path
    register the hook_fn for your face model to get the intermediate features from backbone blocks
    We only implement the hook_fn for adaface and kprpe right now
    """
    if 'adaface' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_adaface_vit_base_webface4m", 
                                      save_path=backbone_cfg['adaface_cache_path'], 
                                      HF_TOKEN=backbone_cfg['HF_TOKEN'])
        # register hook for intermediate features
        first_hook_block = model.model.net.blocks[8] if hasattr(model.model, 'net') else model.model.model.net.blocks[8]
        second_hook_block = model.model.net.blocks[16] if hasattr(model.model, 'net') else model.model.model.net.blocks[16]
        first_hook_block.register_forward_hook(hook_fn)
        second_hook_block.register_forward_hook(hook_fn)
    elif 'kprpe' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_adaface_vit_base_kprpe_webface4m", 
                                        save_path=backbone_cfg['kprpe_cache_path'], 
                                        HF_TOKEN=backbone_cfg['HF_TOKEN'])
        first_hook_block = model.module.net.blocks[8] if hasattr(model.module, 'net') else model.module.model.net.blocks[8]
        second_hook_block = model.module.net.blocks[16] if hasattr(model.module, 'net') else model.module.model.net.blocks[16]
        first_hook_block.register_forward_hook(hook_fn)
        second_hook_block.register_forward_hook(hook_fn)
    elif 'arcface' in mode:
        model = load_model_by_repo_id(repo_id="minchul/cvlface_arcface_ir101_webface4m", 
                                        save_path=backbone_cfg['arcface_cache_path'], 
                                        HF_TOKEN=backbone_cfg['HF_TOKEN'])
        
    else:
        raise NotImplementedError('not supported face model mode!')
    model.eval()
    model.mode = mode
    print(f'face backbone model using {mode} model')
    # print(f'load checkpoint from {ckpt_path}')
    return model

def build_gait_backbone(backbone_cfg, mode='biggait'):
    
    if 'biggait' in mode:
        ckpt_path = backbone_cfg['biggait_backbone_path']
        model_cfg_path = backbone_cfg['biggait_cfg_path']
        model_cfg = yaml.safe_load(open(model_cfg_path, 'r'))['model_cfg']
        model = GaitModel(model_cfg=model_cfg)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.mode = mode
    else:
        raise NotImplementedError('not supported gait model mode!')
    print(f'gait backbone model using {mode} model')
    print(f'load checkpoint from {ckpt_path}')
    return model

def build_body_backbone(backbone_cfg, mode='cal'):
    if 'cal' in mode:
        from WBModules.CAL.configs.default_img import _C as C_img
        from WBModules.CAL.configs.default_vid import _C as C_vid

        from WBModules.CAL.models.img_resnet import ResNet50
        from WBModules.CAL.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50

        factory = {
        'resnet50': ResNet50,
        'c2dres50': C2DResNet50,
        'i3dres50': I3DResNet50,
        'ap3dres50': AP3DResNet50,
        'nlres50': NLResNet50,
        'ap3dnlres50': AP3DNLResNet50,
        }
        if 'ccvid' in mode:
            config = C_img.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-ccvid_config_path'])
            ckpt_path = backbone_cfg['cal-ccvid_backbone_path']       
        elif 'mevid' in mode:
            config = C_vid.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-mevid_config_path'])
            ckpt_path = backbone_cfg['cal-mevid_backbone_path']
        elif 'ltcc' in mode:
            config = C_img.clone()
            config.defrost()
            config.merge_from_file(backbone_cfg['cal-ltcc_config_path'])
            ckpt_path = backbone_cfg['cal-ltcc_backbone_path']           
        model = factory[config.MODEL.NAME](config)
        model_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_dict)
        if 'ccvid' in mode or 'ltcc' in mode:
            first_hook_block = model.base[5]
            second_hook_block = model.base[6]
        elif 'mevid' in mode:
            first_hook_block = model.layer2
            second_hook_block = model.layer3
        first_hook_block.register_forward_hook(hook_fn)
        second_hook_block.register_forward_hook(hook_fn) 
    elif 'agrl' in mode:
        from WBModules.AGRL.torchreid import models
        model = models.init_model(name='vmgn', num_classes=104, loss={'xent', 'htri'},
                            last_stride=1, num_parts=3, num_scale=1,
                            num_split=4, pyramid_part=True, num_gb=2,
                            use_pose=False, learn_graph=True, consistent_loss=True,
                            bnneck=False, save_dir='./')
        ckpt_path = backbone_cfg['agrl_backbone_path']
        model_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        model.load_state_dict(model_dict)
    elif 'aim' in mode:
        from WBModules.AIM_CCReID.configs.default_img import _C as C_img
        from WBModules.AIM_CCReID.models.img_resnet import ResNet50
        config = C_img.clone()
        config.defrost()
        config.merge_from_file(backbone_cfg['aim_config_path'])
        model = ResNet50(config)
        ckpt_path = backbone_cfg['aim_backbone_path']
        model_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        raise NotImplementedError('not supported body model mode!')
    model.mode = mode
    model.eval()
    print(f'body backbone model using {mode} model')
    print(f'load checkpoint from {ckpt_path}')
    return model


class QME(nn.Module):
    """
    Model Wrapper to handle different backbone models
    """
    def __init__(self, backbone_cfg, num_experts=2, mlp_ratio=3, out_dim=1, 
                 dropout_rate=.0, mode='original', train_label=None, cotrain_qe=False, use_qe=False, qe_mode='adaface'):
        """
        original: original backbone structure
        """
        super(QME, self).__init__()
        self.mode = mode
        self.use_qe = use_qe
        self.dropout_rate = dropout_rate
        self.comb_tracker = {}
        # Init backbone
        self.model_list = backbone_cfg['model_list']
        
        self.backbone_dict = nn.ModuleDict()
        self.face_model = nn.ModuleDict()
        for model_name in self.model_list:
            if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name:
                self.face_model[model_name] = build_face_backbone(backbone_cfg, mode=model_name)
            elif 'biggait' in model_name:
                self.backbone_dict[model_name] = build_gait_backbone(backbone_cfg, mode=model_name)
            elif 'cal' in model_name or 'agrl' in model_name or 'aim' in model_name:
                self.backbone_dict[model_name] = build_body_backbone(backbone_cfg, mode=model_name)
            else:
                raise NotImplementedError('not implemented backbone model')
        
        for backbone in self.backbone_dict.keys():
            for param in self.backbone_dict[backbone].parameters():
                param.requires_grad = False
        
        if 'score' in self.mode:
            print(f'QME model using {self.mode} model')
            # self.fgb = MoNormQE(model_list=self.model_list, num_experts=num_experts, mlp_ratio=mlp_ratio, out_dim=out_dim, qe_ckpt_path=backbone_cfg['qe_ckpt_path'],
                            #   f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024, dropout_rate=dropout_rate, cotrain_qe=cotrain_qe)
            self.fgb = MoNormQE_dev(model_list=self.model_list, num_experts=num_experts, mlp_ratio=mlp_ratio, out_dim=out_dim, qe_ckpt_path=backbone_cfg['qe_ckpt_path'],
                                    dropout_rate=dropout_rate, cotrain_qe=cotrain_qe, use_qe=use_qe, qe_mode=qe_mode)
        elif 'mod_qe' in self.mode:
            print(f'QME model using {self.mode} model')
            self.fgb = Face_Quality_Estimator(patch_dim=512, dropout_rate=dropout_rate)
            # self.fgb = Quality_Estimator(patch_dim=[512, 1024], dropout_rate=dropout_rate)
        else:
            print(f'QME model uses original model')
        
        print('backbone model list:', self.model_list)
        print(f'model config: num_experts:{num_experts}, mlp_ratio:{mlp_ratio}, out_dim:{out_dim}, dropout_rate:{dropout_rate}, use_qe:{use_qe}')

    def save_ckpt(self, ckpt_path, optimizer=None, scheduler=None):
        checkpoint = {
            'model_state_dict': self.saved_model().state_dict()
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, ckpt_path)

    def saved_model(self):
        # return trainable param of model without backbone
        return self.fgb
        
    def load_ckpt(self, ckpt_path, optimizer=None, scheduler=None):
        if self.mode == 'original':
            print(f'QME model uses original model, no need to load ckpt')
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.fgb.load_state_dict(checkpoint['model_state_dict'])
            print(f'QME model resumes from ckpt {ckpt_path}')
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def get_face_feat(self, face=None, mode='kprpe', aggregated=True):
        # input dim: (B, C, T, 112, 112)
        # output dim: face:(B, N, 512), intermediate_feats: (B, T, num_blocks=2, num_patches=196, c=512)
        if face is None or face.numel()==0:
            return None, None, None
        bs = face.shape[0]
        intermediate_feats = []
        template_score = None
        if 'kprpe' in mode or 'adaface' in mode:
            with torch.no_grad():
                if face is None:
                    return None, None
                bs, n = face.shape[0:2]
                face = einops.rearrange(face, 'b c t h w -> (b t) c h w')
                face = self.face_model[mode](face)
                face = einops.rearrange(face, '(b t) d  -> b t d', b=bs)  # (B, N, 512)
            for idx in [8, 16]:
                intermediate_feats.append(self.face_model[mode].model.net.blocks[idx].hook_output) # (B * T, num_patches, c)
            intermediate_feats = partial(nn.LayerNorm, eps=1e-6)(intermediate_feats[0].shape[-1],
                                                                            elementwise_affine=False)(
                                                                                torch.stack(intermediate_feats, dim=0))
            intermediate_feats = einops.rearrange(intermediate_feats, 'blk (bs t) p c -> bs t blk p c', bs=bs).contiguous()
        elif 'arcface' in mode:
            with torch.no_grad():
                if face is None:
                    return None, None
                bs, n = face.shape[0:2]
                face = einops.rearrange(face, 'b c t h w -> (b t) c h w')
                face = self.face_model[mode](face)
                face = einops.rearrange(face, '(b t) d  -> b t d', b=bs)  # (B, N, 512)
        
        if aggregated:
            face = face.mean(dim=1)
        torch.cuda.empty_cache()
        return face, intermediate_feats, template_score

    def get_gait_feat(self, gait=None, aggregated=True, mode='biggait'):
        # input dim:(B, C, T, 256, 128)
        # output dim: gait:(B, N, 4096), intermediate_feats: (B, N, num_blocks=4, num_patches=162, c=384)
        if gait is None or gait.numel()==0:
                return None, None
        gait = einops.rearrange(gait, 'b c t h w -> b t c h w')
        if 'biggait' in mode:
            with torch.no_grad():
                bs, n = gait.shape[0:2]
                intermediate_feats = \
                    self.backbone_dict[mode].backbone(einops.rearrange(gait, 'b n c h w -> (b n) c h w'), is_training=True)[
                        "x_norm_patchtokens_mid4"].contiguous()
                intermediate_feats = einops.rearrange(intermediate_feats, '(bs n) p (b c) -> bs n b p c', bs=bs,
                                                    b=4).contiguous()
                gait = self.backbone_dict[mode](([gait], [0], ['_'], ['_'], None))['inference_feat']['embeddings']  # [bs, c, n, p]

                if aggregated:
                    # finalize output
                    gait = einops.rearrange(gait, 'n c s p -> s (n c p)')
                    gait = torch.max(gait, dim=0)[0]  # [n, c, p]
                    gait = einops.rearrange(gait, '(n c p) -> n c p', n=bs, c=512, p=16)
                    gait = self.backbone_dict[mode].gait_net.FCs(gait)  # [n, c//2, p]
                    gait = gait.flatten(1)  # (B, 4096)
                else:
                    gait = einops.rearrange(gait, 'n c s p -> s (n c p)')
                    gait = torch.max(gait, dim=0)[0]  # [n, c, p]
                    gait = einops.rearrange(gait, '(n c p) -> n c p', n=bs, c=512, p=16)
                    gait = self.backbone_dict[mode].gait_net.FCs(gait)  # [n, c//2, p]
                    gait = gait.flatten(1).unsqueeze(1)  # (B, 1, 4096)
                    # previous frame-level gait feature
                    # gait = einops.rearrange(gait, 'bs c n p -> (bs n) c p')
                    # gait = self.backbone_dict[mode].gait_net.FCs(gait)  # [bs*n, c//2, p]
                    # gait = einops.rearrange(gait, '(bs n) c p -> bs n (c p)', bs=bs, n=n)  # (B, N, 4096)
            # take middle 2 blocks
            intermediate_feats = intermediate_feats[:, :, 1:3]
        torch.cuda.empty_cache()
        return gait, intermediate_feats

    def get_body_feat(self, body=None, aggregated=True, mode='vit'):
        # input dim: (B, C, T, 256, 128)
        # output dim: body:(B, T, 512), intermediate_feats: (B, T, num_blocks=4, num_patches=256, c=1024)
        # change block_idx from body_vit/model.py
        if body is None or body.numel()==0:
            return None, None
        bs, c, t, h, w = body.shape
        if 'cal' in mode:
            intermediate_feats = []
            with torch.no_grad():
                if 'ccvid' in mode or 'ltcc' in mode:
                    body = einops.rearrange(body, 'b c t h w -> (b t) c h w')
                    body = self.backbone_dict[mode](body)
                    body = einops.rearrange(body, '(b t) d -> b t d', b=bs)  # (B, N, 4096)
                    if aggregated:
                        body = body.mean(dim=1) # (B, 4096)
                    for idx in [5,6]:
                        intermediate_feats.append(einops.rearrange(self.backbone_dict[mode].base[idx].hook_output, '(bs t) c h w ->bs t c h w', bs=bs)) # list of (B * T, C=, H, W)
                else:
                    # video-based backbone (MEVID)
                    body = self.backbone_dict[mode](body) # (B, 2048)
                    if not aggregated:
                        body = body.unsqueeze(1)  # (B, 1, 2048)
                    for idx in [self.backbone_dict[mode].layer2, self.backbone_dict[mode].layer3]:
                        intermediate_feats.append(einops.rearrange(idx.hook_output, 'bs c t h w ->bs t c h w')) # list of (B * T, C=, H, W)                
                
        elif 'agrl' in mode:
            with torch.no_grad():
                # generate pose related graph for pretrained AGRL follow the paper MEVID
                adj_size = sum(calc_splits(4))
                adj_size = adj_size * t * 1
                adj = torch.ones((bs, adj_size, adj_size))
                # adj = adj.view(bs * t, adj.size(-1), adj.size(-1))
                body = einops.rearrange(body, 'b c t h w -> b t c h w').contiguous()
                body = self.backbone_dict[mode](body, adj)  # (b, 4096)
                intermediate_feats = []
                if not aggregated:
                    body = body.unsqueeze(1) # (b, 1, 4096)

        elif 'aim' in mode:
            with torch.no_grad():
                body = einops.rearrange(body, 'b c t h w -> (b t) c h w')
                _, body = self.backbone_dict[mode](body)
                body = einops.rearrange(body, '(b t) d -> b t d', b=bs)  # (B, N, 4096)
                if aggregated:
                    body = body.mean(dim=1) # (B, 4096)
                    
            intermediate_feats = []

        torch.cuda.empty_cache()
        return body, intermediate_feats
    
    def get_feats(self, inputs, aggregated=True):
        feats_list = {}
        interm_feats_list = {}
        # get feats & scores
        for model_name in self.model_list:
            if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name:
                face_feat, face_intermediate, template_score = self.get_face_feat(inputs[model_name], aggregated=aggregated, mode=model_name)
                feats_list[model_name] = face_feat
                interm_feats_list[model_name] = face_intermediate
            elif 'biggait' in model_name:
                gait_feat, gait_intermediate = self.get_gait_feat(inputs[model_name], aggregated=aggregated, mode=model_name)
                feats_list[model_name] = gait_feat
                interm_feats_list[model_name] = gait_intermediate
            elif 'cal' in model_name or 'agrl' in model_name or 'aim' in model_name:
                body_feat, body_intermediate = self.get_body_feat(inputs[model_name], aggregated=aggregated, mode=model_name)
                feats_list[model_name] = body_feat
                interm_feats_list[model_name] = body_intermediate
            else:
                raise ValueError(f'model_name {model_name} is not supported')
        return feats_list, interm_feats_list
        
    def get_scores(self, probe_feats_list, gallery_feats_list, norm_method='none'):
        """
        probe_feats_list: dict, from get_feats()
        gallery_feats_list: dict, input variable
        """
        scores_list = {}
        # get scores
        for model_name in self.model_list:
            if probe_feats_list[model_name] is not None:
                face_score = sim_fn(probe_feats_list[model_name], gallery_feats_list[model_name], model_name=model_name, norm_method=norm_method)
                scores_list[model_name] = face_score
            else:
                scores_list[model_name] = None
                
        return scores_list
    
    def forward(self, inputs, center_feats, labels=None, norm_method='none'):
        """
        :param face: (bs, n, 3, 112, 112)
        :param gait: (bs, n, 3, 256, 128)
        :param body: (bs, n, 3, 224, 224) (transformed)
        :return:
        """        
        # get feats & scores
        feats_list, interm_feats_list = self.get_feats(inputs)
        # interm_feats_list['adaface_norm'] = feats_list['adaface'].norm(dim=-1)
        scores_list = self.get_scores(feats_list, center_feats, norm_method=norm_method)
        for key in scores_list.keys():
            feats_list[f'{key}_scores'] = scores_list[key]
        feats_list['concat_scores'] = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1)
        feats_list['concat_scores'] = einops.rearrange(feats_list['concat_scores'], 'b n g -> b g n')
        if self.mode == 'original':
            return feats_list
        elif 'score' in self.mode:
            # dropout modality
            if self.training:
                for model_name in self.model_list:
                    if torch.randn(1).item() < self.dropout_rate:
                        scores_list[model_name] = torch.zeros_like(scores_list[model_name])
            res = self.fgb(scores_list, interm_feats_list, labels)
            feats_list.update(res)
        elif 'mod_qe' in self.mode:
            # res = self.fgb(face_interm=face_intermediate, 
            #                gait_interm=gait_intermediate,
            #                body_interm=body_intermediate)
            for model_name in self.model_list:
                if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name:
                    face_interm_feats = interm_feats_list[model_name]
                    res = self.fgb(face_interm=face_interm_feats)
                # elif 'cal' in model_name:
                #     res = self.fgb(interm_feat=interm_feats_list[model_name])
                #     res = {'body_weights': res, 'body_scores': scores_list[model_name]}
            feats_list.update(res)
        for model_name in self.model_list:
            if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name:
                feats_list['face_scores'] = feats_list[f'{model_name}_scores']
            # if 'biggait' in model_name or 'openset' in model_name:
            #     feats_list['gait_scores'] = feats_list[model_name]
            # if 'cal' in model_name or 'agrl' in model_name:
            #     feats_list['body_scores'] = feats_list[model_name]
        return feats_list
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def train(self, mode=True):
        super().train(mode)
        self.training = mode
        for model in self.backbone_dict.keys():
            self.backbone_dict[model].eval()

class MoNormQE(nn.Module):
    """ Learnable normalization with MoE"""

    def __init__(self, model_list=[], num_experts=2, mlp_ratio=1, out_dim=3, 
                 f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024, dropout_rate=.0, norm_method='bn', 
                 qe_ckpt_path=None, cotrain_qe=False, use_qe=True):
        super(MoNormQE, self).__init__()
        
        print(f'initializing MoNormQE model with {num_experts} experts')
        self.num_experts = num_experts
        self.model_list = model_list
        self.cotrain_qe = cotrain_qe

        self.experts = nn.ModuleList([LSN(model_list=model_list, mlp_ratio=mlp_ratio, out_dim=out_dim,
                                           f_patch_dim=f_patch_dim, g_patch_dim=g_patch_dim, b_patch_dim=b_patch_dim, dropout_rate=dropout_rate, norm_method=norm_method) for _ in range(num_experts)])
        
        # self.routers = nn.Sequential(nn.Linear(3, num_experts), nn.Softmax(dim=-1))
        self.apply(self._init_weights)
        # Modality-specific Quality Estimator (QE)
        # self.qe = Mod_QE(f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024)
        self.qe = Face_Quality_Estimator(patch_dim=512, dropout_rate=dropout_rate)

        if not cotrain_qe:
            self.qe.load_state_dict(torch.load(qe_ckpt_path)['model_state_dict'])
            print(f'load qe model from {qe_ckpt_path}')
            self.qe.eval()
            
            for param in self.qe.parameters():
                param.requires_grad = False  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_rankx(self, scores, labels):
            # scores: (B, N), labels: (B,)
            sorted_indices = scores.argsort(dim=1, descending=True)
            ranks = torch.zeros_like(labels)
            for i in range(labels.size(0)):
                ranks[i] = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
            return ranks + 1
            
    def forward(self, scores_list, interm_feats_list, labels=None):
        """
        :param scores_list: dict of score matrices with size (B, G) wo normalization
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :param gait_interm: (B, N, num_blocks=2, num_patches=1024, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        :return:
        """
        device = next(self.parameters()).device
        result = {}
        # get face quality weights
        with torch.no_grad():
            # weights = self.qe(face_interm, gait_interm, body_interm)
            weights = self.qe([interm_feats_list[k] for k in interm_feats_list.keys() if MODEL_MAPPING_DICT[k]=='face_data'][0])
        result.update(weights)
        face_weights = result['face_weights'] # (B, 1)        
        
        # Weighted sum method
        scores = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1) # (B, num_mod, G)
        fuse_scores = face_weights * self.experts[0](scores)['fuse_scores'] + \
                        (1 - face_weights) * self.experts[1](scores)['fuse_scores']
        
        result['fuse_scores'] = fuse_scores

        return result
    
    @torch.no_grad()
    def inference(self, scores_list, face_weights, gait_weights=None, body_weights=None):
        """
        :param scores: score matrices with size (B, num_mod, G) wo normalization
        :param face_weights: (B, 1)
        """
        device = next(self.parameters()).device
        result = {}
        scores = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1) # (B, num_mod, G)
        scores = scores.to(device)
        face_weights = face_weights.to(device)
        
        fuse_scores = face_weights * self.experts[0](scores)['fuse_scores'] + \
                        (1 - face_weights) * self.experts[1](scores)['fuse_scores']
        
        result['fuse_scores'] = fuse_scores
        return result
    
    def train(self, mode=True):
        super().train(mode)
        self.training = mode
        # freeze all backbone models
        if hasattr(self, 'qe'):
            if self.cotrain_qe:
                self.qe.train(mode)
            else:
                self.qe.eval()    


class MoNormQE_dev(nn.Module):
    """ Learnable normalization with MoE"""

    def __init__(self, model_list=[], num_experts=2, mlp_ratio=1, out_dim=3, 
                 f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024, dropout_rate=.0, norm_method='bn', 
                 qe_ckpt_path=None, cotrain_qe=False, use_qe=True, qe_mode='cal'):
        super(MoNormQE_dev, self).__init__()
        
        print(f'initializing MoNormQE DEV model with {num_experts} experts')
        self.num_experts = num_experts
        self.model_list = model_list
        self.cotrain_qe = cotrain_qe
        self.use_qe = use_qe
        if not self.use_qe:
            self.weights = 1 / num_experts
            print(f'not using QE weight, using fixed weight {self.weights}')

        # self.experts_indices = generate_combinations(len(model_list))
        # self.num_experts = len(self.experts_indices)
        self.experts = nn.ModuleList([LSN(model_list=model_list, mlp_ratio=mlp_ratio, out_dim=out_dim,
                                           f_patch_dim=f_patch_dim, g_patch_dim=g_patch_dim, b_patch_dim=b_patch_dim, dropout_rate=dropout_rate, norm_method=norm_method) for _ in range(self.num_experts)])
        # self.router = nn.Sequential(nn.Linear(len(model_list) * 4, self.num_experts), nn.Softmax(dim=-1))
        self.apply(self._init_weights)
        # Modality-specific Quality Estimator (QE)
        if MODEL_MAPPING_DICT[qe_mode] == 'face_data':
            self.qe = Face_Quality_Estimator(patch_dim=512, dropout_rate=dropout_rate)
        else:
            # concat channel for layer 2 and 3
            self.qe = Quality_Estimator(patch_dim=[512+1024], dropout_rate=dropout_rate)

        if not cotrain_qe:
            self.qe.load_state_dict(torch.load(qe_ckpt_path)['model_state_dict'])
            print(f'load qe model from {qe_ckpt_path}')
            self.qe.eval()
            
            for param in self.qe.parameters():
                param.requires_grad = False  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, scores_list, interm_feats_list, labels=None):
        """
        :param scores_list: dict of score matrices with size (B, G) wo normalization
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :param gait_interm: (B, N, num_blocks=2, num_patches=1024, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        :return:
        """
        device = next(self.parameters()).device
        result = {}
        # get face quality weights
        if self.use_qe:
            with torch.no_grad():                
                # (OPTIONAL) use adaface_norm as the quality weights
                # weights = interm_feats_list['adaface_norm'].reshape(-1, 1)
                # weights = {'face_weights': 1 - 1 / weights}
                
                # (OPTIONAL) use CAL QE
                # weights = self.qe([interm_feats_list[k] for k in interm_feats_list.keys() if 'cal' in k][0])
                # weights = {'face_weights': weights}
                
                # use face quality estimator 
                weights = self.qe([interm_feats_list[k] for k in interm_feats_list.keys() if MODEL_MAPPING_DICT[k]=='face_data'][0])
        else:
            weights = {'face_weights': torch.tensor(self.weights, device=device).reshape(1, 1)}
        result.update(weights)
        face_weights = result['face_weights'] # (B, 1)
        
        scores = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1) # (B, num_mod, G)
        # weighted-sum method
        # fuse_scores = torch.sum(torch.stack([face_weights * self.experts[i](scores)['fuse_scores'] for i in range(len(self.experts))]), dim=0)
        
        # QE-guided method
        fuse_scores = face_weights * self.experts[0](scores)['fuse_scores'] + \
                        (1 - face_weights) * self.experts[1](scores)['fuse_scores']
        result['fuse_scores'] = fuse_scores
        return result
    
    @torch.no_grad()
    def inference(self, scores_list, face_weights, gait_weights=None, body_weights=None):
        """
        :param scores: score matrices with size (B, num_mod, G) wo normalization
        :param face_weights: (B, 1)
        """
        device = next(self.parameters()).device
        result = {}
        scores = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1) # (B, num_mod, G)
        scores = scores.to(device)
        if self.use_qe:
            face_weights = face_weights.to(device)
        else:
            face_weights = torch.full((scores.size(0), 1), self.weights, device=device)
        # weighted-sum method
        # fuse_scores = torch.sum(torch.stack([face_weights * self.experts[i](scores)['fuse_scores'] for i in range(len(self.experts))]), dim=0)
        
        # QE-guided method
        fuse_scores = face_weights * self.experts[0](scores)['fuse_scores'] + \
                        (1 - face_weights) * self.experts[1](scores)['fuse_scores']
        
        result['fuse_scores'] = fuse_scores
        return result
    
    def train(self, mode=True):
        super().train(mode)
        self.training = mode
        # freeze all backbone models
        if hasattr(self, 'qe'):
            if self.cotrain_qe:
                self.qe.train(mode)
            else:
                self.qe.eval()    
 
class LSN(nn.Module):
    """ Learnable Score Normalization"""

    def __init__(self, model_list=[], mlp_ratio=1, out_dim=3, 
                 f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024, dropout_rate=.0,
                 norm_method='bn'):
        super(LSN, self).__init__()
        print(f'initializing LSN model')
        self.model_list = model_list
        num_mod = len(model_list)
        # Learnable Score Normalization (LSN)
        if norm_method == 'bn':
            print('using Batch Normalization')
            self.bn = nn.BatchNorm1d(num_mod)
        self.norm_method = norm_method
        # self.score_mlp = Mlp(num_mod, num_mod * mlp_ratio, out_dim, nn.Identity, drop=dropout_rate)
        self.score_mlp = Mlp(num_mod, num_mod * mlp_ratio, out_dim, nn.SELU, drop=dropout_rate)
        # self.score_mlp_restr = Mlp(num_mod, num_mod * mlp_ratio, out_dim, nn.SELU, drop=dropout_rate)
        # self.score_mlp = AttnBlock(dim=num_mod, num_heads=1, mlp_ratio=3. ,qkv_bias=True, qk_scale=None, attn_drop=0., drop=dropout_rate, act_layer=nn.SELU)
        self.apply(self._init_weights)

        # Modality-specific Quality Estimator (QE)
        # self.fqe = Face_Quality_Estimator(patch_dim=f_patch_dim, dropout_rate=dropout_rate)
        # self.fqe.load_state_dict(torch.load('models/3mod_splitcrossattn_modweight-idx8-16_frame_stylemlp_centerdist_hardminingv2_rank50_mseloss-0.9755-3.pth')['model_state_dict'])
        # print('load fqe model')
        # self.fqe.eval()
        # for param in self.fqe.parameters():
        #     param.requires_grad = False
        
        # self.f_style_mlp = Mlp(f_patch_dim * 4, f_patch_dim * 2, 512, nn.ReLU, drop=dropout_rate)
        # self.g_style_mlp = Mlp(g_patch_dim * 2, g_patch_dim * 1, 512, nn.ReLU, drop=dropout_rate)
        # self.b_style_mlp = Mlp(b_patch_dim * 2, b_patch_dim * 1, 512, nn.ReLU, drop=dropout_rate)
        # self.f_weight_assigner = nn.Sequential(nn.Linear(512, 1))
        # self.g_weight_assigner = nn.Sequential(nn.Linear(512, 1))
        # self.b_weight_assigner = nn.Sequential(nn.Linear(512, 1))
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def make_style(self, x):
        # blk, B, P, C = x.shape
        mean = x.mean(-2, keepdim=False)
        std = x.std(-2, keepdim=False)
        style = torch.stack([mean, std], dim=-2)
        return style
    
    def cal_rankx(self, scores, labels):
        # scores: (B, N), labels: (B,)
        sorted_indices = scores.argsort(dim=1, descending=True)
        ranks = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            ranks[i] = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        return ranks + 1
        
    def get_qe(self, face_interm=None, gait_interm=None, body_interm=None):
        """
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :param gait_interm: (B, N, num_blocks=2, num_patches=1024, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        """
        device = next(self.parameters()).device
        result = {}
        bs, num_frames = (face_interm.shape[0], face_interm.shape[1]) if face_interm is not None else \
            ((gait_interm.shape[0], gait_interm.shape[1]) if gait_interm is not None else (
                body_interm.shape[0], body_interm.shape[1]))
        # Face
        if face_interm is not None:
            result['face_weights'] = self.fqe(face_interm).mean(dim=1)
            # proj_face_feat = face_interm
            # proj_face_feat = einops.rearrange(proj_face_feat, 'bs n blk p c -> blk bs n p c').contiguous()
            # face_style = self.make_style(proj_face_feat)  # (blk, B, N, num_stat, 512)
            # face_style = einops.rearrange(face_style, 'blk bs n stat c -> bs n (blk stat c)').contiguous()  # (B, N, blk*num_stat*512)
            # face_style = self.f_style_mlp(face_style)  # (B, N, 512)
            # face_weight = self.f_weight_assigner(face_style)  # (B, N, 1)
            # result['face_weights'] = nn.Sigmoid()(face_weight).mean(dim=1)
        else:
            # zero features
            result['face_weights'] = torch.zeros(bs, 1, device=device)
        
        # Gait
        if gait_interm is not None:
            # style for body/gait doesn't work
            proj_gait_feat = gait_interm
            proj_gait_feat = einops.rearrange(proj_gait_feat, 'bs n blk p c -> bs n p (blk c)').contiguous()
            gait_style = self.g_style_mlp(proj_gait_feat) # (B, N, p, 512)
            gait_style = gait_style.mean(dim=-2) # (B, N, 512)
            gait_weight = self.g_weight_assigner(gait_style)
            result['gait_weights'] = nn.Sigmoid()(gait_weight).mean(dim=1)
        else:
            # zero features
            result['gait_weights'] = torch.zeros(bs, 1, device=device)
        
        # Body
        if body_interm is not None:
            proj_body_feat = body_interm
            proj_body_feat = einops.rearrange(proj_body_feat, 'bs n blk p c -> bs n p (blk c)').contiguous()
            body_style = self.b_style_mlp(proj_body_feat) # (B, N, p, 512)
            body_style = body_style.mean(dim=-2) # (B, N, 512)
            body_weight = self.b_weight_assigner(body_style)
            result['body_weights'] = nn.Sigmoid()(body_weight).mean(dim=1)
        else:
            # zero features
            result['body_weights'] = torch.zeros(bs, 1, device=device)
        return result
        
    def forward(self, scores, labels=None):
        """
        :param scores: score matrices with size (B, num_mod, G) wo normalization
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :param gait_interm: (B, N, num_blocks=2, num_patches=1024, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        :return:
        """
        # assert face_interm is not None or gait_interm is not None or body_interm is not None, 'at least one input is not None'
        device = next(self.parameters()).device
        result = {}
        # bs, num_frames = (face_interm.shape[0], face_interm.shape[1]) if face_interm is not None else \
        #     ((gait_interm.shape[0], gait_interm.shape[1]) if gait_interm is not None else (
        #         body_interm.shape[0], body_interm.shape[1]))
        
        # scores = torch.cat([face_score.unsqueeze(1), gait_score.unsqueeze(1), body_score.unsqueeze(1)], dim=1) # (B, 3, G)
        if self.norm_method == '2mean':
            norm_scores = self.bn(scores, labels)
        else:
            norm_scores = self.bn(scores)
        norm_scores = einops.rearrange(norm_scores, 'b mod g -> b g mod')
        
        fuse_scores = self.score_mlp(norm_scores).mean(dim=-1)
        result['fuse_scores'] = fuse_scores # (B, G)

        return result
    
    @torch.no_grad()
    def inference(self, scores_list):
        # for one sample
        result = {}
        device = next(self.parameters()).device
        scores = torch.cat([scores_list[model_name].unsqueeze(1) for model_name in self.model_list], dim=1) # (B, num_mod, G)
        scores = scores.to(device)
        norm_scores = self.bn(scores)
        norm_scores = einops.rearrange(norm_scores, 'b mod g -> b g mod')
        
        # high_quality_mask = face_weight.squeeze(-1) > 0.4

        # low_quality_mask = ~high_quality_mask
        
        # high_quality_indices = torch.nonzero(high_quality_mask).squeeze(1)
        # low_quality_indices = torch.nonzero(low_quality_mask).squeeze(1)

        # high_quality_scores = norm_scores[high_quality_mask]
        # low_quality_scores = norm_scores[low_quality_mask]

        # fuse_scores = torch.empty_like(norm_scores, device=device)
        # if high_quality_scores.size(0) > 0:
        #     high_quality_fuse_scores = self.score_mlp(high_quality_scores)
        #     fuse_scores[high_quality_indices] = high_quality_fuse_scores

        # if low_quality_scores.size(0) > 0:
        #     low_quality_fuse_scores = self.score_mlp_restr(low_quality_scores)
        #     fuse_scores[low_quality_indices] = low_quality_fuse_scores
        # fuse_scores = fuse_scores.mean(dim=-1)
        
        # fuse_scores, attn_map  = self.score_mlp(norm_scores, return_attn=True)
        # fuse_scores = fuse_scores.mean(dim=-1).cpu().numpy() # (B, G)
        # result['attn_map'] = attn_map.squeeze().cpu().numpy() # (G, G)
        
        # fuse_scores = self.score_mlp(norm_scores).mean(dim=-1).cpu().numpy() # (B, G)
        # norm_scores = torch.stack([norm_scores[:, :, 0] * face_weight,
        #                          norm_scores[:, :, 1] * 1, 
        #                          norm_scores[:, :, 2] * 1], dim=-1)
        fuse_scores = self.score_mlp(norm_scores).mean(dim=-1) # (B, G)
        result['fuse_scores'] = fuse_scores 
    
        return result
    
    def train(self, mode=True):
        super().train(mode)
        self.training = mode
        # freeze all backbone models
        if hasattr(self, 'fqe'):
            self.fqe.eval()

class Mod_QE(nn.Module):
    """ Learnable normalization and QE version"""

    def __init__(self, f_patch_dim=512, g_patch_dim=384, b_patch_dim=1024, dropout_rate=.0,
                 norm_method='bn'):
        super(Mod_QE, self).__init__()
        print(f'initializing Mod_QE model')
        self.face_qe = Quality_Estimator(patch_dim=f_patch_dim, num_blk=2, dropout_rate=dropout_rate)
        self.gait_qe = Quality_Estimator(patch_dim=g_patch_dim, num_blk=2, dropout_rate=dropout_rate)
        self.body_qe = Quality_Estimator(patch_dim=b_patch_dim, num_blk=2, dropout_rate=dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_rankx(self, scores, labels):
        # scores: (B, N), labels: (B,)
        sorted_indices = scores.argsort(dim=1, descending=True)
        ranks = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            ranks[i] = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        return ranks + 1
        
    def forward(self, face_interm=None, gait_interm=None, body_interm=None, labels=None):
        """
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :param gait_interm: (B, N, num_blocks=4, num_patches=162, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        :return:
        """
        assert face_interm is not None or gait_interm is not None or body_interm is not None, 'at least one input is not None'
        device = next(self.parameters()).device
        result = {}
        bs, num_frames = (face_interm.shape[0], face_interm.shape[1]) if face_interm is not None else \
            ((gait_interm.shape[0], gait_interm.shape[1]) if gait_interm is not None else (
                body_interm.shape[0], body_interm.shape[1]))
            
        # Face
        if face_interm is not None:
            result['face_weights'] = self.face_qe(face_interm)['face_weights']
        else:
            # zero features
            result['face_weights'] = torch.zeros(bs, 1, device=device)
        
        # Gait
        if gait_interm is not None:
            result['gait_weights'] = self.gait_qe(gait_interm)
        else:
            # zero features
            result['gait_weights'] = torch.zeros(bs, 1, device=device)
        
        # Body
        if body_interm is not None:
            result['body_weights'] = self.body_qe(body_interm)
        else:
            # zero features
            result['body_weights'] = torch.zeros(bs, 1, device=device)
        return result
          
class Face_Quality_Estimator(nn.Module):
    """ FQE version"""

    def __init__(self, patch_dim=512, dropout_rate=.0):
        super(Face_Quality_Estimator, self).__init__()
        print(f'initializing FQE model')
        self.f_style_mlp = Mlp(patch_dim * 4, patch_dim * 2, patch_dim, nn.ReLU, drop=dropout_rate)
        # self.f_style_mlp = Mlp(patch_dim, patch_dim // 2, patch_dim, nn.ReLU, drop=dropout_rate)
        self.weight_assigner = nn.Sequential(nn.Linear(patch_dim, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def make_style(self, x):
        blk, B, N, C = x.shape
        mean = x.mean(2, keepdim=False)
        std = x.std(2, keepdim=False)
        style = torch.stack([mean, std], dim=-2)
        return style

    def cal_rankx(self, scores, labels):
        # scores: (B, N), labels: (B,)
        sorted_indices = scores.argsort(dim=1, descending=True)
        ranks = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            ranks[i] = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        return ranks + 1

    def forward(self, face_interm=None):
        """
        :param face_interm: (B, N, num_blocks=2, num_patches=196, c=512)
        :return: quality weights for batch (B, 1)
        """
        device = next(self.parameters()).device
        result = {}
        if face_interm.ndim == 5:
            bs, num_frames = (face_interm.shape[0], face_interm.shape[1])
            reshape_face_interm = einops.rearrange(face_interm, 'bs n blk p c -> blk (bs n) p c').contiguous() # (blk, B*N, num_stat, 512)
        else:
            assert face_interm.ndim == 4, f'face_interm shape {face_interm.shape}'
            num_frames = face_interm.shape[0] # B=1
            reshape_face_interm = einops.rearrange(face_interm, 'n blk p c -> blk n p c').contiguous() # (blk, B*N, num_stat, 512)

        face_style = self.make_style(reshape_face_interm)  # (blk, B*N, num_stat, 512)
        face_style = einops.rearrange(face_style, 'blk b n c -> b (blk n c)').contiguous()  # (B*N, blk*num_stat*512)
        face_style = self.f_style_mlp(face_style)  # (B*N, 512)
        face_weight = self.weight_assigner(face_style)  # (B*N, 1)
        if face_interm.ndim == 5:
            face_weight = einops.rearrange(face_weight, '(b n) d -> b n d', b=bs).contiguous()  # (B, N, 1)

        # for input is style features (b, n , 64)
        # face_style = self.f_style_mlp(face_interm) # (B, N, 64)
        # face_weight = self.weight_assigner(face_style)  # (B, N, 1)
        
        face_weight = nn.Sigmoid()(face_weight).mean(dim=1)
        result['face_weights'] = face_weight
        return result

class Quality_Estimator(nn.Module):
    """General QE version"""
    def __init__(self, patch_dim=[512, 1024], dropout_rate=.0, arch='resnet50'):
        super(Quality_Estimator, self).__init__()
        print(f'initializing Quality Estimator model')
        mlp_out_dim = 512
        self.arch = arch
        patch_dim = np.array(patch_dim).sum()
        self.style_mlp = Mlp(patch_dim * 2, patch_dim, mlp_out_dim, nn.ReLU, drop=dropout_rate)
        self.weight_assigner = nn.Sequential(nn.Linear(mlp_out_dim, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def make_style(self, x):
        if self.arch == 'transformer':
            blk, B, N, C = x.shape
            mean = x.mean(2, keepdim=False)
            std = x.std(2, keepdim=False)
            style = torch.stack([mean, std], dim=-2)
        elif self.arch == 'resnet50':
            B, N, C, H, W = x.shape
            mean = x.view(B, N, C,-1).mean(-1, keepdim=False)
            std = x.view(B, N, C,-1).std(-1, keepdim=False)
            style = torch.stack([mean, std], dim=-1)
        return style

    def cal_rankx(self, scores, labels):
        # scores: (B, N), labels: (B,)
        sorted_indices = scores.argsort(dim=1, descending=True)
        ranks = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            ranks[i] = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        return ranks + 1
        
    def forward(self, interm_feat=None):
        """
        :param gait_interm: (B, N, num_blocks=2, num_patches=1024, c=384)
        :param body_interm: (B, N, num_blocks=2, num_patches=256, c=1024)
        :param body_interm: [(B, N, C=2, num_patches=256, c=1024)]

        :return: weights (N, 1)
        """
        bs, num_frames = interm_feat[0].shape[0], interm_feat[0].shape[1]
        if interm_feat is not None:
            interm_feat = torch.concat([self.make_style(f) for f in interm_feat], dim=2) # (B, N, channel=512+1024, num_stat=2)
            proj_feat = interm_feat
            proj_feat = einops.rearrange(proj_feat, 'bs n c stat -> bs n (c stat)').contiguous()
            style_feat = self.style_mlp(proj_feat) # (B, N, 512)
            weights = self.weight_assigner(style_feat)
            weights = nn.Sigmoid()(weights).mean(dim=1)
        else:
            # zero features
            device = next(self.parameters()).device
            weights = torch.zeros(bs, 1, device=device)
        
        return weights

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def sim_fn(probe_feats, gallery_feats, model_name, norm_method='none'):
        # simiarity function for different models (e.g., cos_sim/euc_dist)
        if 'kprpe' in model_name or 'adaface' in model_name or 'arcface' in model_name or 'cal' in model_name or 'agrl' in model_name or 'aim' in model_name:
            scores = F.cosine_similarity(probe_feats.unsqueeze(1), gallery_feats.unsqueeze(0), dim=2)
        elif 'biggait' in model_name:
            bs = probe_feats.shape[0]
            num_g = gallery_feats.shape[0]
            # euc distance
            gait_score = torch.norm(gallery_feats.reshape(num_g, -1, 16).unsqueeze(0) - probe_feats.reshape(bs, -1, 16).unsqueeze(1), p=2, dim=2) #(P, G, 16)
            gait_score = gait_score.mean(dim=-1) #(P, G)
            scores = 1 / (1 + gait_score)
        else:
            raise ValueError(f'unknown model name: {model_name}')
        if norm_method == 'none':
            scores = scores
        return scores

def compare_model_weights(model1, model2):
    # Get the state dictionaries of the two models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if the state dictionaries are of the same size
    if len(state_dict1) != len(state_dict2):
        print('The state dictionaries have different sizes')
        return False

    # Check if the keys of the state dictionaries match
    if state_dict1.keys() != state_dict2.keys():
        print('The state dictionaries do not have the same keys')
        return False

    # Check if the values (weights) of the state dictionaries match
    for key in state_dict1:
        flag = True
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f'key {key} is not equal')
            flag = False
        if flag:
            return True
        else:
            return False

def load_model(train_cfg=None):
    if train_cfg is None:
        train_cfg=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'configs/train_cfg.yaml'
        )
    train_cfg = yaml.safe_load(open(train_cfg, 'r'))
    model = MoNormQE(model_list=train_cfg['model_list'], mlp_ratio=train_cfg['mlp_ratio'], out_dim=train_cfg['out_dim'], qe_ckpt_path=None)
    
    checkpoint = torch.load(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
          train_cfg['ckpt_path']),
        map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"load model from {train_cfg['ckpt_path']}")
    model.eval()
    return model
