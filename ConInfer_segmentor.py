import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import *
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler

import os
import random
from PIL import Image
import numpy as np

from torchvision import transforms
from fast_pytorch_kmeans import KMeans
from gmm import *
import concurrent.futures

@MODELS.register_module()
class ConInferSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 gmm_temp=50.0,
                 attn_lambda=0.0,
                 dataset_type= None,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')
        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.device = device
        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize
        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        
        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)
        
        if dataset_type in ['UAVidDataset', 'VDDDataset']:
            # dinov3 dinov3_vith16plus model
            REPO_DIR = "/data/users/cwy/0_datasets/github_source_packge/dinov3-main"
            WEIGHT_DIR = "/data/users/cwy/0_datasets/model_weight/DINOV3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=WEIGHT_DIR)
        else:
            # # dinov3 sat493m model
            REPO_DIR = "/data/users/cwy/0_datasets/github_source_packge/dinov3-main"
            WEIGHT_DIR = "/data/users/cwy/0_datasets/model_weight/DINOV3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source="local", weights=WEIGHT_DIR)
        
        self.dinov3.eval().to(device).half()
        
        self.dino_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448), antialias=True),
            transforms.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            )
        ])
        
        num_workers = 32
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.gmm_temp = gmm_temp
        self.attn_lambda = attn_lambda
        self.dataset_type = dataset_type
        
    def forward_feature(self, img, return_feat=False):
        batchsize = img.shape[0]
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token)

        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            cls_logits = image_cls_token @ self.query_features.T

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(batchsize, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()
            image_features = image_features.view(batchsize, self.feat_dim, image_w * image_h).permute(0, 2, 1)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits.unsqueeze(1) * self.cls_token_lambda
        
        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]

        logits = logits.permute(0, 2, 1).reshape(batchsize, out_dim, w, h)
        feats = image_features.permute(0, 2, 1).reshape(batchsize, self.feat_dim, w, h)
        if return_feat:
            return logits, feats
        else:
            return logits

    # ConInfer Inference
    @torch.no_grad()
    def predict(self, inputs, data_samples):
        clip_input = inputs.to(self.device).half()
        dino_input = self.get_dino_inputs(data_samples)

        clip_logit, clip_feats = self.dense_forward(clip_input, is_clip=True)
        dino_feats = self.dense_forward(dino_input, is_clip=False)

        B, P, C = clip_logit.shape
        B, P, D1 = clip_feats.shape
        B, P, D2 = dino_feats.shape
        H_patch, W_patch = [x // 16 for x in clip_input.shape[2:]]
        clip_logit = clip_logit.view(B*P, C)
        clip_feats = clip_feats.view(B*P, D1)
        dino_feats = dino_feats.view(B*P, D2)

        del clip_feats, dino_input
        z = gmm_fitting(dino_features=dino_feats.float(), y_hat=clip_logit.float(), temp=self.gmm_temp)
        clip_seg_logits = z.view(B, P, C).permute(0, 2, 1).view(B, C, H_patch, W_patch)
        return self.postprocess_result(clip_seg_logits, data_samples)

    # Baseline Inference: maskclip，clearclip，SCLIP，GEM, qqkkvv
    @torch.no_grad()
    def predict1(self, inputs, data_samples):
        clip_input = inputs.to(self.device).half()
        if self.clip_type == 'BLIP':
            clip_input = F.interpolate(clip_input, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(clip_input, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
            image_features /= image_features.norm(dim=-1, keepdim=True)
            clip_logit = image_features @ self.query_features.T
        elif self.model_type == 'GEM':
            image_features = self.net.visual(clip_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            clip_logit = image_features @ self.query_features.T
        else:
            clip_logit, _ = self.dense_forward(clip_input, is_clip=True)
        
        B, P, C = clip_logit.shape
        H_patch = W_patch = int(P ** 0.5)
        clip_seg_logits = clip_logit.permute(0, 2, 1).view(B, C, H_patch, W_patch)
        return self.postprocess_result(clip_seg_logits, data_samples)
    
    # SegEarth-OV Inference
    @torch.no_grad()
    def predict2(self, inputs, data_samples):
        inputs = inputs.half()
        image_logits = self.forward_feature(inputs)
        return self.postprocess_result(image_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            single_patch_logits = seg_logits[i].unsqueeze(0)
            target_size = data_samples[i].metainfo['ori_shape']

            single_seg_logits = F.interpolate(
                single_patch_logits,
                size=list(target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  

            single_seg_logits = single_seg_logits * self.logit_scale
            single_seg_logits = single_seg_logits.softmax(0)

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                temp_logits = single_seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx).to(temp_logits.device)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                single_seg_logits = (temp_logits * cls_index).max(1)[0]

            seg_pred = single_seg_logits.argmax(0, keepdim=True)
            confidence_mask = single_seg_logits.max(0, keepdim=True)[0] < self.prob_thd
            seg_pred[confidence_mask] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(data=single_seg_logits),
                'pred_sem_seg': PixelData(data=seg_pred.squeeze(0))
            })
            
        return data_samples

    def _load_and_transform_single_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            dino_input = self.dino_transform(image)
            return dino_input
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    def get_dino_inputs(self, data_samples):
        img_paths = [ds.metainfo['img_path'] for ds in data_samples]
        results = self.thread_pool_executor.map(
            lambda path: self._load_and_transform_single_image(path),
            img_paths
        )
        dino_inputs_list = [res for res in results if res is not None]
        if not dino_inputs_list:
            raise RuntimeError("All images failed to load in get_dino_inputs.")
        dino_inputs_batch = torch.stack(dino_inputs_list).to(self.device).half()
        return dino_inputs_batch
        
    def dense_forward(self, inputs_batch, is_clip):
        chunk_size = 50 # Adjust according to your video memory size.
        chunk_feature = []
        chunk_resluts = []
        for i in range(0, inputs_batch.shape[0], chunk_size):
            chunk = inputs_batch[i:i+chunk_size]
            if is_clip:
                dense_feats = self.net.encode_image(chunk, self.model_type, ignore_residual=self.ignore_residual, output_cls_token=self.output_cls_token, attn_lambda=self.attn_lambda)
                dense_feats /= dense_feats.norm(dim=-1, keepdim=True)
                dense_logit = dense_feats @ self.query_features.T
                chunk_feature.append(dense_feats)
                chunk_resluts.append(dense_logit)
            else:
                feature_dict = self.dinov3.forward_features(chunk)
                dense_feats = feature_dict['x_norm_patchtokens']
                dense_feats /= dense_feats.norm(dim=-1, keepdim=True)
                chunk_resluts.append(dense_feats)
        chunk_resluts = torch.cat(chunk_resluts, dim=0)
        if is_clip:
            chunk_feature = torch.cat(chunk_feature, dim=0)
            return chunk_resluts, chunk_feature
        
        return chunk_resluts

    def compute_result(self, z, B, dino_masks_original):
        _, C = z.shape
        K = self.n_clusters
        z = z.view(B, K, C)
        clip_seg_logits = torch.einsum('bkhw,bkp->bhwp', dino_masks_original.half(), z.half())
        clip_seg_logits = clip_seg_logits.permute(0, 3, 1, 2)  # [B, P, h_img, w_img]
        return clip_seg_logits

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices