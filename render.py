#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
    hist_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hist")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)

    height, width = views[0][0].shape[1:3]
    cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, (viewimg, view) in enumerate(tqdm(views, desc="Rendering progress")):
        view = view.cuda()

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        img = cv2.cvtColor((rendering.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cap.write(img)
        gt = viewimg[0:3, :, :]
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        import matplotlib.pyplot as plt
        data1 = render_pkg["neural_opacity"]
        data2 = render_pkg["marginal"]
        data3 = render_pkg["opacity"]

        # Convert tensors to numpy arrays for plotting
        data1_np = data1.cpu().numpy().flatten()
        data2_np = data2.cpu().numpy().flatten()
        data3_np = data3.cpu().numpy().flatten()

        # Plotting histograms side by side
        fig, axs = plt.subplots(1, 3, figsize=(30, 6), sharey=True)

        # Plotting the first histogram
        axs[0].hist(data1_np, bins=100, alpha=0.75, color='blue')
        axs[0].set_title('Histogram of Torch Tensor Data 1')
        axs[0].set_xlabel('Neural Opacity')
        axs[0].set_ylabel('Frequency')
        axs[0].set_ylim(0, 400000)  # Set y-axis range
        axs[0].grid(True)

        # Plotting the second histogram
        axs[1].hist(data2_np, bins=100, alpha=0.75, color='green')
        axs[1].set_title('Histogram of Torch Tensor Data 2')
        axs[1].set_xlabel('Marginal')
        axs[0].set_ylim(0, 400000)  # Set y-axis range
        axs[1].grid(True)

        # Plotting the third histogram
        axs[2].hist(data3_np, bins=100, alpha=0.75, color='red')
        axs[2].set_title('Histogram of Torch Tensor Data 3')
        axs[2].set_xlabel('Opacity')
        axs[0].set_ylim(0, 400000)  # Set y-axis range
        axs[2].grid(True)

        # Save the plot as an image file
        plt.savefig(os.path.join(hist_path, '{0:05d}'.format(idx) + ".png"))
        plt.close()

        # import pdb; pdb.set_trace()

    cap.release()

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.time_dim, dataset.time_embedding)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
