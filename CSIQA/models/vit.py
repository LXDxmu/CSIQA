# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from torchinfo import summary
from torchvision.transforms.functional import resize
import numpy as np
from numpy import *

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj1 = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

        # self.scale_proj = nn.Conv2d(
        #     in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        # )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # self.pooling = nn.AdaptiveAvgPool2d((14, 14))

    # def forward_multiscale(self, x):
    #     s = resize(x, (112, 112))
    #     s = self.scale_proj(s)
    #     return s

    def forward(self, x):
        # s = self.forward_multiscale(x)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # s = s.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,#384
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        juery=1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.juery=juery
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lambda_mgd =0.3

    def forward(self, x ):#mask_map:[B,196,384]
        # mask_map = X[1]
        # x=self.norm1(X[0])
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #q:[3,6,198,64]

        q = q * self.scale
        if self.training:
            # mat = torch.rand((B, self.num_heads, N, N))  # [12 ,6, 197, 197]
            mask = torch.ones((B, self.num_heads, N, N)).cuda()
            # mask_map = mask_map.view(B, 1, -1, C)
            # mask_map = mask_map.expand(B, self.num_heads, -1,-1)
            for i in range(1,self.juery+1):
                # mask[:, :, -1, :-1] = torch.randint(0, 2, (B, self.num_heads, N-1))
                # mask[:, :, -i, :-(i+2)] = torch.where(torch.rand(B, self.num_heads, N-1) > 0, torch.ones(1), torch.zeros(1))
                #m:
                mask[:, :, -i, :-1] = torch.where(torch.rand(B, self.num_heads, N-1) > 1-self.lambda_mgd, torch.zeros(1), torch.ones(1))
                mask[:, :, -i, 196] = 1
                #0Nx
                mask[:, :, :, -i] = 0
                #1
                mask[:, :,-i, -i] = 1
                # mask_visual=mask.detach().cpu().numpy()
                # mask_visual_map = mask_map.detach().cpu().numpy()

            log_mat = torch.log(mask).cuda()
            # log_mat_visual=log_mat.cpu().numpy()
            attn = log_mat + q @ k.transpose(-2, -1)
            # attn_visual = attn.cpu().numpy()
            attn = attn.softmax(dim=-1)
            # attn1_visual = attn.cpu().numpy()
            attn = self.attn_drop(attn)
            #attn:torch.Size([16, 6, 197, 197])
            #v:torch.Size([16, 6, 197, 64])
            #attn @ v:torch.Size([16, 6, 197, 64])
        else:
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x_visual = x.cpu().numpy()
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Experts_MOS(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        juery_nums=6,
    ):
        super().__init__()
        # self.juery = juery_nums
        bunch_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            dropout=0.0,
            nhead=6,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(embed_dim * 4),
            norm_first=True,
        )
        self.bunch_decoder = nn.TransformerDecoder(bunch_layer, num_layers=1)
        # self.bunch_embedding = nn.Parameter(torch.randn(1, juery_nums, embed_dim))
        self.heads1 = nn.Linear(embed_dim*2, 1, bias=False)
        # trunc_normal_(self.bunch_embedding, std=0.02)

    def forward(self, x, ref, local): #x:[32,196,384] ref:[32,384] #local:[32,3,384]
        B, L, D = x.shape
        _, juery_num, _ = local.shape
        # local=F.normalize(local)
        # bunch_embedding = self.bunch_embedding.expand(B, -1, -1) #32,3,384
        ref = ref.view(B, 1, -1) #[32,1,384]
        # ref = ref.expand(B, self.juery, -1) #[32,6,384]
        # local = local.expand(B, self.juery, -1)  # [32,6,384]
        # local_all = local + bunch_embedding # [32,6,384]
        # ref_all = ref + bunch_embedding
        # ref = ref + bunch_embedding
        # local = local + bunch_embedding
        x_ref = self.bunch_decoder(ref, x)
        x_local = self.bunch_decoder(local, x)
        x = torch.cat([x_ref, x_local], dim=2)
        # ref_all = torch.cat((local,ref), dim=1) #[32,3,384]
        # output_embedding =  ref + bunch_embedding
        # output_embedding = torch.cat((local,output_embedding), dim=1)
        # x = self.bunch_decoder(output_embedding, x)
        # x = torch.cat([x[:, 0], x[:, 1]], dim=1)
        x = self.heads1(x)
        x = x.view(B, -1).mean(dim=1)
        return x.view(B, 1)


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        )
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp1 = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = (
            x
            + self.drop_path(self.attn(self.norm1(x)))
            + self.drop_path(self.attn1(self.norm11(x)))
        )
        x = (
            x
            + self.drop_path(self.mlp(self.norm2(x)))
            + self.drop_path(self.mlp1(self.norm21(x)))
        )
        return x

class Loss_contrastive(torch.nn.Module):
    def __init__(self):
        super(Loss_contrastive,self).__init__()

    def forward(self,x,weight_pos ,weight_neg ,temperature):
        B, _,_ = x.shape #[B,0.1*B]
        image_features1 = x.view(B, -1)
        image_features1 = (image_features1 / image_features1.norm(dim=1, keepdim=True))  # (B,197,384)
        sim_matrix_positive = torch.exp(torch.mm(image_features1, image_features1.t().contiguous()) / temperature)  # (B,B)
        sim_matrix_positive1 = ((sim_matrix_positive * weight_pos).sum(1)) / ((weight_pos).sum(1))  # 按行就和
        sim_matrix_negative1 = (sim_matrix_positive * weight_neg).sum(1)
        Loss = (- torch.log(sim_matrix_positive1 / (sim_matrix_positive1 + sim_matrix_negative1))).mean()

        return Loss

class vit_models(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_scale=1e-4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = 196
        self.juery = 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.local_token = nn.Parameter(torch.randn(1, self.juery, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )
        width = 12
        scale = width ** -0.5
        self.norm = norm_layer(embed_dim)
        self.proj1 = nn.Parameter(scale * torch.randn(embed_dim, 32))
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        # self.head = (
        #     nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # )
        self.head = Experts_MOS(embed_dim=384)

        self.head_project = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 128)
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.local_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token" ,"local_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    # def forward_pos_scale(self):
    #     pos_embed = self.pos_embed.transpose(1, 2).view(1, -1, 14, 14)
    #     pos_embed = F.interpolate(pos_embed, (7, 7), mode="bilinear").flatten(2)
    #     return pos_embed.transpose(1, 2)

    def forward_features(self, x , score ,score_t):
        # self.training=True
        B = x.shape[0] #x:[B,3,224,224]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        local_tokens = self.local_token.expand(B, -1, -1)

        x = self.patch_embed(x) #x:[B,196,384]
        x = x + self.pos_embed
        x = torch.cat((local_tokens, cls_tokens, x), dim=1)

        if self.training:
            #找对比学习的图片
            # 1.GT的相似矩阵
            # image_score1 = score / score.norm(dim=0, keepdim=True).expand(B, B)  # (B,1)->(B,B)
            image_score1 = score.expand(B, B)  # (B,1)->(B,B)
            image_score2 = image_score1.t()  # (B,B)->(B,B)
            logits_per_score = abs(image_score1 - image_score2)  # (B,B)
            max_count = (torch.sum(logits_per_score == 0, dim=1)).max()
            # 2.ouput的相似矩阵
            # image_score_t = score_t / score_t.norm(dim=0, keepdim=True).expand(B, B)  # (B,1)->(B,B)
            image_score_t = score_t .expand(B, B)
            image_score_t1 = image_score_t.t()  # (1,B)->(B,B)
            logits_per_score_t = abs(image_score_t - image_score_t1)
            # print(logits_per_score)
            # print(logits_per_score_t)
            threshold = logits_per_score  # (B,B)
            # print(threshold)
            threshold_t = logits_per_score / (logits_per_score_t) #threshold_t 没有0只有nan
            #logits_per_score有0（正样本）就会变nan，变nan在下面的取负样本，即最大值时就会被取到，这就互相矛盾了，因此需要把这种情况剔除
            threshold_t = torch.where(torch.isnan(threshold_t), torch.zeros_like(threshold_t), threshold_t)
            # print(threshold_t)
            # print(threshold_t)
            # 找每一行最小值的索引

            # 首先定义0.2个batchsize是正类(GT)
            pos_define = torch.topk(threshold, k=round(0.2 * B)+1, dim=1, largest=False)[1]  # GT接近
            # print(torch.topk(threshold, k=round(0.2 * B)+1, dim=1, largest=False)[0])
            # print(pos_define)
            # print('---------------------')
            # 首先定义0.6个batchsize是负类(GT)
            neg_define = torch.topk(threshold, k=round(0.6 * B), dim=1)[1]  # GT较远
            # print(neg_define)
            # print('---------------------')
            # 从threshold中0.0到0.1里面取正例作为简单正样本(GT简单正样本)
            idx = torch.topk(threshold, k=round(0.1 * B)+1, dim=1, largest=False)[1]  # GT接近
            # print(idx)
            # print('---------------------')
            # 从threshold中0.0到0.2里面取负例作为简单负样本(GT简单负样本)
            idx_negative = torch.topk(threshold, k=round(0.4 * B), dim=1)[1]  # GT较远
            # print(idx_negative)
            # print('---------------------')
            # 从threshold_t中0.0到0.1里面取正例作为困难正样本(output，要与GT取交集)
            idx_t = torch.topk(threshold_t, k=round(0.1 * B), dim=1, largest=False)[1]  # # GT近 ,output远
            # print(idx_t)
            # print('---------------------')
            # 从threshold_t中0.0到0.1里面取负例作为困难负样本(output，要与GT取交集)
            idx_negative_t = torch.topk(threshold_t, k=round(0.4 * B), dim=1)[1]  # GT远,output较近
            # print(idx_negative_t)
            # print('---------------------')


            candidates_t = []  # output接近
            candidates_negative_t = []  # output较远
            for i in range(B):
                # (output，要与GT取交集)
                # 找到GT近output远的困难正样本索引并将简单正样本与困难正样本合并
                intersection1 = np.union1d(np.intersect1d(pos_define[i][1:].cpu().numpy(), idx_t[i].cpu().numpy()),idx[i][1:].cpu().numpy())
                intersection_pos = torch.tensor(list(intersection1))
                # 找到GT远output近的困难负样本索引并将简单负样本与困难负样本合并
                intersection2 = np.union1d(np.intersect1d(neg_define[i].cpu().numpy(), idx_negative_t[i].cpu().numpy()),idx_negative[i].cpu().numpy())
                intersection_neg = torch.tensor(list(intersection2))

                candidates_t.append(intersection_pos.unsqueeze(dim=0))  # GT近output远 candidates_t是列表
                candidates_negative_t.append(intersection_neg.unsqueeze(dim=0))  # GT远output近
            # print("GT close out far")
            # print(candidates_t)
            # print("GT far out close")
            # print(candidates_negative_t)

            weight_pos = torch.zeros_like(logits_per_score)
            weight_neg = torch.zeros_like(logits_per_score)
            for i, (cols1, cols2) in enumerate(zip(candidates_t, candidates_negative_t)):
                weight_pos[i, cols1.long()] = 1
                weight_neg[i, cols2.long()] = 1
            # print(N_pos)
            # N_neg = weight_neg.sum(1)
            # print(N_neg)
        Loss_sum=0

        mlp_inner_feature = []
        # criterion = torch.nn.L1Loss()

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.training:
                Loss_con = Loss_contrastive().cuda()
                Loss = Loss_con(x, weight_pos, weight_neg, 0.3)
                Loss_sum += 0.1 * Loss
            # mlp_inner_feature.append(x[:, self.juery+1:, :])
        x = self.norm(x)
        z = x

        return z[:, 0:self.juery], z[:, self.juery], z[:, self.juery+1:, :], Loss_sum
        # return x

    def forward(self, x , score=0 ,score_t=0): #pred_masks[B,224,224] #samples: [B,3,224,224]
        local, ref, x ,Loss= self.forward_features(x ,score, score_t)
        # normalized features
        x = self.head(x, ref ,local)
        # x = self.head(ref)
        return x,Loss


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
def build_vit(
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    block_layers=Layer_scale_init_Block,
    pretrained=False,
    pretrained_model_path="",
):
    model = vit_models(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        block_layers=block_layers,
    )
    if pretrained:
        assert pretrained_model_path != ""
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = checkpoint["model"]
        # del state_dict["head.weight"]
        # del state_dict["head.bias"]
        model.load_state_dict(state_dict, strict=False)
        # with torch.no_grad():
        #     model.patch_embed.scale_proj.weight.copy_(
        #         state_dict["patch_embed.proj.weight"]
        #     )
        #     model.patch_embed.scale_proj.bias.copy_(state_dict["patch_embed.proj.bias"])
        del checkpoint
        torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    vit = build_vit(
        pretrained=True,
        pretrained_model_path="D:/mac/Data-Eff-IQA/deit_3_small_224_1k.pth",
    )
    pre = PatchEmbed(embed_dim=384)
    summary(vit, ((16, 3, 224, 224),(16, 3, 224, 224),(16, 1),(16, 1)),device=torch.device("cpu"))
