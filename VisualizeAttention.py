# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='NeededFiles/dino_deitsmall8_pretrain.pth', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher2", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="Testbilder/testbild.png", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(64, 64), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='AttentionMapVis', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()


    # build model
    model = vits.__dict__["vit_small"](
        patch_size=args.patch_size,
        num_classes=0
        # stochastic depth
    )

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    imagedir_path = "Testbilder"

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for image_path2 in os.listdir(imagedir_path):
        imgname = image_path2
        image_path = imagedir_path + "/" + image_path2

        # open image

        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f"Provided image path {image_path} is non valid.")
            sys.exit(1)

        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size  # amount of tokens width
        h_featmap = img.shape[-1] // args.patch_size  # amount of tokens height
        attentions2 = model.get_last_selfattention(img.to(device))
        #attentions2 = model.get_first_attention(img.to(device))

        nh = attentions2.shape[1]  # number of heads



        # we keep only the output patch attention

        attentions = attentions2[0, :, 0, 1:].reshape(nh, -1)
        # query pixel = cls token

        attentions = attentions.reshape(nh, w_featmap, h_featmap)  # headnum, tokens-x, tokens-y : pixel -> token
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
            0].cpu().numpy()

        #attentions[attentions < 0.01] = 0
        # headnum, w of img, h of img

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                     os.path.join(args.output_dir, imgname))
        for j in range(nh):
            fname = os.path.join(args.output_dir, imgname + "head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")