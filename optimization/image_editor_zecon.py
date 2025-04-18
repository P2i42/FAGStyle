import os
from pathlib import Path
from collections import defaultdict
from optimization.constants import ASSETS_DIR_NAME
from utils.metrics_accumulator import MetricsAccumulator
import optimization.shapeSpace as shapeSpace

from numpy import random
from optimization.augmentations import ImageAugmentations as ImageAugmentations
from optimization.augmentations_swin import SwinImageAugmentations
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss, d_clip_dir_loss, mse_loss, get_features, zecon_loss_direct, scc_loss_direct, d_geo_loss, d_clip_fags_dir_loss
import numpy as np

from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from torchvision import models
from utils.visualization import  show_edited_masked_image
import matplotlib.pyplot as plt

# import ipdb

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults(self.args)

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)
        if self.args.data == 'imagenet':
            self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
            self.model.load_state_dict(
            torch.load(
                "./ckpt/256x256_diffusion_uncond.pt",
                map_location="cpu",
            )
        )
        elif self.args.data == 'ffhq':
            self.model_config.update(
            {
                "num_channels": 128,
                "num_head_channels": 64,
                "num_res_blocks":1,
                "attention_resolutions": "16",
                "resblock_updown": True,
                "use_fp16": False,
            }
            )
            self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
            self.model.load_state_dict(
            torch.load(
                # "./ckpt/ffhq_10m.pt",
                "./ckpt/ffhq_baseline.pt",
                map_location="cpu",
            )
        )

        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        
        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
            
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        self.image_augmentations = ImageAugmentations(224, self.args.aug_prob, self.args.patch_min, self.args.patch_max, patch=False)
        self.patch_augmentations = ImageAugmentations(224, self.args.aug_prob, self.args.patch_min, self.args.patch_max, patch=True)
        self.swin_patch_augmentations = SwinImageAugmentations(224, self.args.aug_prob, patch=True)
        
        self.metrics_accumulator = MetricsAccumulator()
        self.losses_dict = defaultdict(lambda: [])
       
        if self.args.l_vgg > 0:
            self.vgg = models.vgg19(pretrained=True).features
            self.vgg.to(self.device)
            self.vgg.eval().requires_grad_(False)
        
        self.vgg_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep
    
    def clip_global_loss(self,x_in,text_embed):
        clip_loss = torch.tensor(0)
        augmented_input = self.image_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def clip_fags_loss(self, x_in, text_embed):
        clip_loss = torch.tensor(0)
        if not self.args.swin:
            augmented_input = self.patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(2)  # (32,3,224,224)
        else:
            augmented_input = self.swin_patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(
                2)  # (32,3,224,224)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()  # (32, 512)


        batch = image_embeds.size(0)

        # image_embeds = image_embeds.reshape(batch, -1)

        image_embeds = image_embeds.reshape(batch, -1, 2)
        image_embeds = shapeSpace.project(image_embeds)

        inner_embeddings = []
        distr = torch.distributions.dirichlet.Dirichlet(torch.ones(batch) / 1.0, validate_args=None)
        alpha = distr.sample((batch,)).to(self.device)
        for alphai in alpha:
           shapeInterp = shapeSpace.ShapeInterpolator(image_embeds, alphai)
           inner_pts = shapeInterp.generate()
           inner_pts = inner_pts.reshape(-1).unsqueeze(0)
           # inner_pts = inner_pts.reshape(feat_channel, feat_size, feat_size)
           inner_embeddings.append(inner_pts)
        image_embeds = torch.stack(inner_embeddings)

        text_embed_proj = text_embed
        text_embed_proj = shapeSpace.project(text_embed_proj.reshape(1, -1, 2))
        text_embed_proj = text_embed_proj.reshape(1, -1)

        dists = d_geo_loss(image_embeds, text_embed_proj)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i:: self.args.batch_size].mean()

        return clip_loss

    def clip_global_patch_loss(self, x_in, text_embed):
        # text_embed (1, 512)
        clip_loss = torch.tensor(0)
        if not self.args.swin:
            augmented_input = self.patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(
                2)  # (32,3,224,224)
        else:
            augmented_input = self.swin_patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(
                2)  # (32,3,224,224)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float() # (32, 512)
        dists = d_clip_loss(image_embeds, text_embed)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss
    
    def clip_dir_loss(self, x_in, y_in, text_embed, text_y_embed):
        clip_loss = torch.tensor(0)
        
        augmented_input_x = self.image_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
        augmented_input_y = self.image_augmentations(y_in,num_patch=self.args.n_patch).add(1).div(2)
        
        clip_in_x = self.clip_normalize(augmented_input_x)
        clip_in_y = self.clip_normalize(augmented_input_y)
        
        image_embeds_x = self.clip_model.encode_image(clip_in_x).float()
        image_embeds_y = self.clip_model.encode_image(clip_in_y).float()
        dists = d_clip_dir_loss(image_embeds_x, image_embeds_y, text_embed, text_y_embed)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss
        
    def clip_dir_patch_loss(self, x_in, y_in, text_embed, text_y_embed):
        clip_loss = torch.tensor(0)


        if not self.args.swin:
            augmented_input_x = self.patch_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
            augmented_input_y = self.patch_augmentations(y_in, num_patch=self.args.n_patch, is_global=True).add(1).div(
                2)
        else:
            augmented_input_x = self.swin_patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(2)
            augmented_input_y = self.swin_patch_augmentations(y_in, num_patch=self.args.n_patch, is_global=True).add(1).div(
                2)
        # augmented_input_y = self.patch_augmentations(y_in,num_patch=self.args.n_patch,is_global=True).add(1).div(2)

        clip_in_x = self.clip_normalize(augmented_input_x)
        clip_in_y = self.clip_normalize(augmented_input_y)
        image_embeds_x = self.clip_model.encode_image(clip_in_x).float()
        image_embeds_y = self.clip_model.encode_image(clip_in_y).float()
        dists = d_clip_dir_loss(image_embeds_x, image_embeds_y, text_embed, text_y_embed)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def clip_fags_dir_patch_loss(self, x_in, y_in, text_embed, text_y_embed):
        clip_loss = torch.tensor(0)
        if not self.args.swin:
            augmented_input_x = self.patch_augmentations(x_in,num_patch=self.args.n_patch).add(1).div(2)
            augmented_input_y = self.patch_augmentations(y_in, num_patch=self.args.n_patch, is_global=True).add(1).div(
                2)
        else:
            augmented_input_x = self.swin_patch_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(2)
            augmented_input_y = self.swin_patch_augmentations(y_in, num_patch=self.args.n_patch, is_global=True).add(1).div(
                2)

        clip_in_x = self.clip_normalize(augmented_input_x)
        clip_in_y = self.clip_normalize(augmented_input_y)
        image_embeds_x = self.clip_model.encode_image(clip_in_x).float()
        image_embeds_y = self.clip_model.encode_image(clip_in_y).float()

        batch = image_embeds_x.size(0)

        image_embeds_x = image_embeds_x.reshape(batch, -1, 2)
        image_embeds_x = shapeSpace.project(image_embeds_x)

        image_embeds_y = image_embeds_y.reshape(batch, -1, 2)
        image_embeds_y = shapeSpace.project(image_embeds_y)

        inner_embeddings = []
        inner_embeddings_y = []
        distr = torch.distributions.dirichlet.Dirichlet(torch.ones(batch) / 1.0, validate_args=None)
        alpha = distr.sample((batch,)).to(self.device)
        for alphai in alpha:
            shapeInterp = shapeSpace.ShapeInterpolator(image_embeds_x, alphai)
            inner_pts = shapeInterp.generate()
            inner_pts = inner_pts.reshape(-1).unsqueeze(0)
            # inner_pts = inner_pts.reshape(feat_channel, feat_size, feat_size)
            inner_embeddings.append(inner_pts)

            shapeInterp_y = shapeSpace.ShapeInterpolator(image_embeds_y, alphai)
            inner_pts_y = shapeInterp_y.generate()
            inner_pts_y = inner_pts_y.reshape(-1).unsqueeze(0)
            # inner_pts = inner_pts.reshape(feat_channel, feat_size, feat_size)
            inner_embeddings_y.append(inner_pts_y)
        image_embeds_x = torch.cat(inner_embeddings)
        image_embeds_y = torch.cat(inner_embeddings_y)

        text_embed_proj = text_embed
        text_embed_proj = shapeSpace.project(text_embed_proj.reshape(1, -1, 2))
        text_embed_proj = text_embed_proj.reshape(1, -1)

        text_embed_proj_y = text_y_embed
        text_embed_proj_y = shapeSpace.project(text_embed_proj_y.reshape(1, -1, 2))
        text_embed_proj_y = text_embed_proj_y.reshape(1, -1)

        dists = d_clip_fags_dir_loss(image_embeds_x, image_embeds_y, text_embed_proj, text_embed_proj_y)
        for i in range(self.args.batch_size):
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss
    
    def zecon_loss(self, x_in, y_in, t):
        loss = zecon_loss_direct(self.model, x_in, y_in, torch.zeros_like(t,device=self.device))
        return loss.mean()

    def scc_loss(self, x_in, y_in, t):
        loss = scc_loss_direct(self.model, x_in, y_in, torch.zeros_like(t,device=self.device))
        return loss.mean()
    
    def mse_loss(self,x_in, y_in):
        loss = mse_loss(x_in,y_in)
        return loss.mean()
    
    def vgg_loss(self,x_in, y_in):
        content_features = get_features(self.vgg_normalize(x_in), self.vgg)
        target_features = get_features(self.vgg_normalize(y_in), self.vgg)
        loss = 0

        loss += torch.mean((target_features['conv1_1'] - content_features['conv1_1']) ** 2)
        loss += torch.mean((target_features['conv2_1'] - content_features['conv2_1']) ** 2)
        # loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        # loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        return loss.mean()

    def plot_metrics_and_save(self, filename="metrics_plot.png"):
        # Check if there are any metrics to plot
        if not self.metrics_accumulator.accumulator:
            print("No metrics to plot.")
            return

        plt.figure(figsize=(10, 5))

        for metric_name, values in self.metrics_accumulator.accumulator.items():
            iterations = list(range(1, len(values) + 1))
            plt.plot(iterations, values,marker='o', label=metric_name)
            for i, value in enumerate(values):
                plt.text(iterations[i], values[i], f"{value:.2f}", fontsize=8)

        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Loss Metrics Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory

    def edit_image_by_prompt(self):
        
        text_embed = self.clip_model.encode_text(
            clip.tokenize(self.args.prompt_tgt).to(self.device)
        ).float()
        text_y_embed = self.clip_model.encode_text(
            clip.tokenize(self.args.prompt_src).to(self.device)
        ).float()

        # text_embed from prompt target
        # text_y_embed from source target
        # x is output image
        # y is source image

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        visualization_path = visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )

        def cond_fn(x, t, y=None):
            if self.args.prompt_tgt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                
                loss = torch.tensor(0)
                if self.args.l_clip_global != 0:
                    clip_loss = self.clip_global_loss(x_in, text_embed) * self.args.l_clip_global
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                    # self.losses_dict["clip_loss"].append(clip_loss.item())

                if self.args.l_clip_fags_patch != 0:
                    fags_loss = self.clip_fags_loss(x_in, text_embed) * self.args.l_clip_fags_patch
                    loss = loss + fags_loss
                    self.metrics_accumulator.update_metric("fags_loss", fags_loss.item())
                    # self.losses_dict["fags_loss"].append(fags_loss.item())

                if self.args.l_clip_global_patch != 0:
                    clip_patch_loss = self.clip_global_patch_loss(x_in, text_embed) * self.args.l_clip_global_patch
                    loss = loss + clip_patch_loss
                    self.metrics_accumulator.update_metric("clip_patch_loss", clip_patch_loss.item())
                    # self.losses_dict["clip_patch_loss"].append(clip_patch_loss.item())
                
                if self.args.l_clip_dir != 0:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    clip_dir_loss = self.clip_dir_loss(x_in, y_in, text_embed, text_y_embed) * self.args.l_clip_dir
                    loss = loss + clip_dir_loss
                    self.metrics_accumulator.update_metric("clip_dir_loss", clip_dir_loss.item())
                    # self.losses_dict["clip_dir_loss"].append(clip_dir_loss.item())

                if self.args.l_clip_dir_patch != 0:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    clip_dir_patch_loss = self.clip_dir_patch_loss(x_in, y_in, text_embed, text_y_embed) * self.args.l_clip_dir_patch
                    loss = loss + clip_dir_patch_loss
                    self.metrics_accumulator.update_metric("clip_dir_patch_loss", clip_dir_patch_loss.item())
                    # self.losses_dict["clip_dir_patch_loss"].append(clip_dir_patch_loss.item())

                if self.args.l_clip_fags_dir_patch != 0:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    clip_fags_dir_patch_loss = self.clip_fags_dir_patch_loss(x_in, y_in, text_embed, text_y_embed) * self.args.l_clip_fags_dir_patch
                    loss = loss + clip_fags_dir_patch_loss
                    self.metrics_accumulator.update_metric("clip_fags_dir_patch_loss", clip_fags_dir_patch_loss.item())
                    # self.losses_dict["clip_fags_dir_patch_loss"].append(clip_fags_dir_patch_loss.item())

                if self.args.l_zecon != 0:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    zecon_loss = self.zecon_loss(x_in, y_in,t) * self.args.l_zecon
                    loss = loss + zecon_loss
                    self.metrics_accumulator.update_metric("zecon_loss", zecon_loss.item())
                    # self.losses_dict["zecon_loss"].append(zecon_loss.item())

                if self.args.l_scc != 0:
                    y_t = self.diffusion.q_sample(self.init_image, t)
                    y_in = self.init_image * fac + y_t * (1 - fac)
                    scc_loss = self.scc_loss(x_in, y_in, t) * self.args.l_scc
                    self.metrics_accumulator.update_metric("scc_loss", scc_loss.item())
                    # self.losses_dict["scc_loss"].append(scc_loss.item())
                
                if self.args.l_mse != 0 and t.item() < 700:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    mse_loss = self.mse_loss(x_in, y_in) * self.args.l_mse
                    loss = loss + mse_loss
                    self.metrics_accumulator.update_metric("mse_loss", mse_loss.item())
                    # self.losses_dict["mse_loss"].append(mse_loss.item())
                
                if self.args.l_vgg != 0 and t.item() < 800:
                    y_t = self.diffusion.q_sample(self.init_image,t)
                    y_in = self.init_image * fac + y_t * (1 - fac)

                    vgg_loss = self.vgg_loss(x_in, y_in) * self.args.l_vgg
                    loss = loss + vgg_loss
                    self.metrics_accumulator.update_metric("vgg_loss", vgg_loss.item())
                    # self.losses_dict["vgg_loss"].append(vgg_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                    # self.losses_dict["range_loss"].append(r_loss.item())
                
                return -torch.autograd.grad(loss, x)[0]
            

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):

            self.losses_dict.clear()

            fw = self.args.diffusion_type.split('_')[0]
            bk = self.args.diffusion_type.split('_')[-1]
            
            # Forward DDIM
            if fw == 'ddim':
                print("Forward Process to noise")
                noise = self.diffusion.ddim_reverse_sample_loop(
                    self.model, 
                    self.init_image, 
                    clip_denoised=False,
                    skip_timesteps=self.args.skip_timesteps,
                    )
            
            # Forward DDPM
            elif fw == 'ddpm':
                init_image_batch = torch.tile(self.init_image, dims=(self.args.batch_size, 1, 1, 1))
                noise = self.diffusion.q_sample(
                        x_start=init_image_batch,
                        t=torch.tensor(self.diffusion.num_timesteps-int(self.args.skip_timesteps), dtype=torch.long, device=self.device),
                        noise=torch.randn((self.args.batch_size,3,self.model_config["image_size"],self.model_config["image_size"]), device=self.device),
                    )
            else:
                raise ValueError

            # Reverse DDPM
            if bk == 'ddpm':
                samples = self.diffusion.p_sample_loop_progressive(
                    self.model,
                    (
                        self.args.batch_size,
                        3,
                        self.model_config["image_size"],
                        self.model_config["image_size"],
                    ),
                    noise = noise if fw=='ddim' else None,
                    clip_denoised=False,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=self.args.skip_timesteps,
                    init_image=self.init_image,
                )
            
            # Reverse DDIM
            elif bk == 'ddim':
                samples = self.diffusion.ddim_sample_loop_progressive(
                    self.model,
                    (
                        self.args.batch_size,
                        3,
                        self.model_config["image_size"],
                        self.model_config["image_size"],
                    ),
                    noise = noise,
                    clip_denoised=False,
                    model_kwargs={},
                    cond_fn=cond_fn,  
                    progress=True,
                    skip_timesteps=self.args.skip_timesteps,
                    eta=self.args.eta,
                )
            
            else:
                raise ValueError

            
            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1


            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    filename = Path(self.args.init_image).stem
                    visualization_path = visualization_path.with_name(
                        f"{filename}_{self.args.prompt_tgt}_{iteration_number}{visualization_path.suffix}"
                    )

                    loss_path = str(visualization_path).replace('.png', '_losses.png')
                    self.plot_metrics_and_save(loss_path)
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)

                        # filename = Path(self.args.init_image).stem
                        # visualization_path = visualization_path.with_name(
                        #     f"{filename}_{self.args.prompt_tgt}_{iteration_number}{visualization_path.suffix}"
                        # )
                        #
                        # loss_path = str(visualization_path).replace('.png','_losses.png')
                        # self.plot_metrics_and_save(loss_path)

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            show_edited_masked_image(
                                title=self.args.prompt_tgt,
                                source_image=self.init_image_pil,
                                edited_image=pred_image_pil,
                                path=visualization_path,
                            )
                            
                            visualization_path2 = str(visualization_path).replace('.png','_output.png')
                            pred_image_arr = np.array(pred_image_pil)
                            plt.imsave(visualization_path2, pred_image_arr)

