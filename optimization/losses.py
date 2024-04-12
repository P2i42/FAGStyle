# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn
import optimization.shapeSpace as shapeSpace

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance

def d_geo_loss(x, y):

    distance = shapeSpace.getShapeSpaceDistance(x, y)

    return distance

def d_clip_dir_loss(x_embd,y_embd,prompt_x_embd,prompt_y_embd):
    d_img = x_embd - y_embd # (32, 512)
    d_txt = prompt_x_embd - prompt_y_embd

    d_img = F.normalize(d_img, dim=-1)
    d_txt = F.normalize(d_txt, dim=-1)
    
    distance = 1 - (d_img @ d_txt.t()).squeeze() # (32, )
    

    return distance


def d_clip_fags_dir_loss(x_embd, y_embd, prompt_x_embd, prompt_y_embd):
    # d_img = shapeSpace.getShapeSpaceDistance(x_embd, y_embd)   # (32, 512)
    # d_txt = shapeSpace.getShapeSpaceDistance(prompt_x_embd, prompt_y_embd)

    # d_img = F.normalize(d_img, dim=-1)
    # d_txt = F.normalize(d_txt, dim=-1)
    # d_img = (x_embd * y_embd).sum(dim=1) #(32, )
    # d_txt = shapeSpace.getShapeSpaceDistance(prompt_x_embd, prompt_y_embd)
    # distance = 1 - (d_img @ d_txt.t()).squeeze()
    #
    # return distance
    d_img = x_embd - y_embd  # (32, 512)
    d_txt = prompt_x_embd - prompt_y_embd

    d_img = F.normalize(d_img, dim=-1)
    d_txt = F.normalize(d_txt, dim=-1)

    distance = 1 - (d_img @ d_txt.t()).squeeze()  # (32, )

    return distance

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def mse_loss(x_in, y_in):
    mse = torch.nn.MSELoss()
    return mse(x_in,y_in)

def get_features(image, model, layers=None):
    
    if layers is None:
        layers = {'0': 'conv1_1', 
                  '2': 'conv1_2', 
                  '5': 'conv2_1',  
                  '7': 'conv2_2',
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def zecon_loss_direct(Unet, x_in, y_in,t):
    total_loss = 0
    nce_layers = [0,2,5,8,11]
    num_patches=256

    l2norm = Normalize(2)
    feat_q = Unet.forward_enc(x_in,t, nce_layers)
    #list 5 (1, 256, 256, 256); (1,256,128,128); (1,512,64,64); (1,512,32,32)
    feat_k = Unet.forward_enc(y_in,t, nce_layers)
    # list 5 (1, 256, 256, 256); (1,256,128,128); (1,512,64,64); (1,512,32,32)
    patch_ids = []
    feat_k_pool = []
    feat_q_pool = []
    
    for feat_id, feat in enumerate(feat_k):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = np.random.permutation(feat_reshape.shape[1])
        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        
        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        patch_ids.append(patch_id)
        x_sample = l2norm(x_sample)
        feat_k_pool.append(x_sample)
    
    for feat_id, feat in enumerate(feat_q):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = patch_ids[feat_id]

        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        x_sample = l2norm(x_sample)
        feat_q_pool.append(x_sample)

    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        loss = PatchNCELoss(f_q, f_k)
        total_loss += loss.mean()
    return total_loss.mean()


def scc_loss_direct(Unet, x_in, y_in, t):
    total_loss = 0
    nce_layers = [0, 2, 5, 8, 11]
    num_patches = 256
    Loss_fn_smth = nn.SmoothL1Loss().cuda()
    pool_dict = {}
    pool_dict[4] = nn.AdaptiveAvgPool2d((1, 1))
    pool_dict[8] = nn.AdaptiveAvgPool2d((2, 2))
    pool_dict[16] = nn.AdaptiveAvgPool2d((4, 4))
    pool_dict[32] = nn.AdaptiveAvgPool2d((16, 16))
    pool_dict[64] = nn.AdaptiveAvgPool2d((32, 32))
    pool_dict[128] = nn.AdaptiveAvgPool2d((64, 64))
    pool_dict[256] = nn.AdaptiveAvgPool2d((128, 128))

    l2norm = Normalize(2)
    feat_q = Unet.forward_enc(x_in, t, nce_layers)
    # list 5 (1, 256, 256, 256); (1,256,128,128); (1,512,64,64); (1,512,32,32)
    feat_k = Unet.forward_enc(y_in, t, nce_layers)
    # list 5 (1, 256, 256, 256); (1,256,128,128); (1,512,64,64); (1,512,32,32)


    for idxx, (feat_s, feat_t) in enumerate(zip(feat_q, feat_k)):
        feat_size = feat_s.size(2)
        feat_channel = feat_s.size(1)
        batch = feat_s.size(0)
        ShapeSpace_proj = True

        if ShapeSpace_proj:
            if feat_size >= 32 and feat_size <= 256:
                feat_s = feat_s.reshape(batch, -1)
                feat_t = feat_t.reshape(batch, -1)

                # feat_s = feat_s.reshape(args.batch, -1, 2)
                # feat_t = feat_t.reshape(args.batch, -1, 2)
                feat_s = shapeSpace.project(feat_s)
                feat_t = shapeSpace.project(feat_t)

                inner_embeddings = []
                # distr = torch.distributions.dirichlet.Dirichlet(torch.ones(batch) / 1.0, validate_args=None)
                # alpha = distr.sample((batch,)).cuda()
                # for alphai in alpha:
                #     shapeInterp = shapeSpace.ShapeInterpolator(feat_s, alphai)
                #     inner_pts = shapeInterp.generate()
                #     inner_pts = inner_pts.reshape(-1).unsqueeze(0)
                #     inner_pts = inner_pts.reshape(feat_channel, feat_size, feat_size)
                #     inner_embeddings.append(inner_pts)
                # feat_s = torch.stack(inner_embeddings)
                feat_s = feat_s.reshape(batch, feat_channel, feat_size, feat_size)
                feat_t = feat_t.reshape(batch, feat_channel, feat_size, feat_size)

        feat = torch.cat([feat_s, feat_t], dim=0)
        batchsize = feat.size(0)
        feat = F.normalize(feat, dim=1)
        vector_len = feat_s.size(1)
        if feat_size >= 32 and feat_size <= 256:
            feat = pool_dict[feat_size](feat)  # [batch * 2, 128, 128, 128]
            feat_size = int(feat_size / 2)
            window_size = int(feat_size / 2)
            strid = int(feat_size / 2)  # 64
            ud = int(window_size / 2)  # ud:32
            unfold_feat = F.unfold(feat, (window_size, window_size), stride=strid)
            patch_num = unfold_feat.size(-1)
            unfold_feat = unfold_feat.resize(batchsize, vector_len, window_size * window_size,
                                             patch_num)
            unfold_feat = unfold_feat.permute(0, 3, 2, 1).reshape(batchsize * patch_num,
                                                                  window_size * window_size, vector_len)
            self_sim = torch.matmul(unfold_feat, unfold_feat.transpose(1, 2)).reshape(batchsize,
                                                                                      patch_num,
                                                                                      window_size * window_size,
                                                                                      window_size * window_size)
            self_sim = torch.chunk(self_sim, 2, dim=0)

            total_loss += Loss_fn_smth(self_sim[0], self_sim[1])

            feat = feat[:, :, ud:feat_size - ud, ud:feat_size - ud]  # [batch * 2, 128, 64, 64]
            unfold_feat = F.unfold(feat, (window_size, window_size), stride=strid)
            patch_num = unfold_feat.size(-1)
            unfold_feat = unfold_feat.resize(batchsize, vector_len, window_size * window_size,
                                             patch_num)
            unfold_feat = unfold_feat.permute(0, 3, 2, 1).reshape(batchsize * patch_num,
                                                                  window_size * window_size, vector_len)
            self_sim = torch.matmul(unfold_feat, unfold_feat.transpose(1, 2)).reshape(batchsize,
                                                                                      patch_num,
                                                                                      window_size * window_size,
                                                                                      window_size * window_size)
            self_sim = torch.chunk(self_sim, 2, dim=0)

            total_loss += Loss_fn_smth(self_sim[0], self_sim[1])

        else:
            continue

    return total_loss

def PatchNCELoss(feat_q, feat_k, batch_size=1, nce_T = 0.07):
    # feat_q : n_patch x 512
    # feat_q : n_patch x 512
    batch_size = batch_size
    nce_T = nce_T
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask_dtype = torch.bool

    num_patches = feat_q.shape[0]
    dim = feat_q.shape[1]
    feat_k = feat_k.detach()
    
    # pos logit 
    l_pos = torch.bmm(
        feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
    l_pos = l_pos.view(num_patches, 1)

    # reshape features to batch size
    feat_q = feat_q.view(batch_size, -1, dim)
    feat_k = feat_k.view(batch_size, -1, dim)
    npatches = feat_q.size(1)
    l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

    # diagonal entries are similarity between same features, and hence meaningless.
    # just fill the diagonal with very small number, which is exp(-10) and almost zero
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
    l_neg_curbatch.masked_fill_(diagonal, -10.0)
    l_neg = l_neg_curbatch.view(-1, npatches)

    out = torch.cat((l_pos, l_neg), dim=1) / nce_T

    loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))

    return loss




