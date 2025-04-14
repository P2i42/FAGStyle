#### Environment setting
**Python** 3.8.5 \
**Torch** 1.11.0 
```
$ conda env create -f environment.yml
$ conda activate fags
```
Our source code relies on [ZeCon](https://github.com/YSerin/ZeCon).

#### Pre-trained model
Download the model weights trained on [imagenet](https://github.com/openai/guided-diffusion) dataset.

Create a folder **'./ckpt/'** and then place the downloaded weights into the folder.

#### Image manipulation
In order to transfer style, run:
```
python main.py --output_path [output_dir_path] --init_image [source_image_path] --data 'imagenet' --prompt_tgt [target_style_prompt] --prompt_src [source_style_prompt] --skip_timesteps 25 --timestep_respacing 50 --diffusion_type 'ddim_ddpm' --l_clip_global 0 --l_clip_global_patch 20000 --l_clip_fags_patch 20000 --l_clip_dir 0 --l_clip_dir_patch 20000 --l_clip_fags_dir_patch 20000 --l_zecon 1000 --l_scc 1000 --l_mse 1000 --l_vgg 100  --swin  --gpu_id 0 --n_patch 49
```

For Example:
```
python main.py --output_path './result' --init_image './src_image/apple.jpg' --data 'imagenet' --prompt_tgt 'Cubism' --prompt_src 'Photo' --skip_timesteps 25 --timestep_respacing 50 --diffusion_type 'ddim_ddpm' --l_clip_global 0 --l_clip_global_patch 20000 --l_clip_fags_patch 20000 --l_clip_dir 0 --l_clip_dir_patch 20000 --l_clip_fags_dir_patch 20000 --l_zecon 1000 --l_scc 1000 --l_mse 1000 --l_vgg 100  --swin  --gpu_id 0 --n_patch 49
```
