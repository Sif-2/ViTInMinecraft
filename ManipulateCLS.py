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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer_injection as vits

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='NeededFiles/dino_deitsmall8_pretrain.pth', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument('--layertoinject', default=12, type=int)
    parser.add_argument('--lettertoinject', default=0, type=int)
    parser.add_argument("-usebias", default=False, action="store_true")
    parser.add_argument("--image_path1", default="TestbilderInj/cat.png", type=str, help="Path of the image to load.")
    parser.add_argument("--image_path2", default="TestbilderInj/dog.jpg", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(128, 128), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='CLSInjectionTests', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    img_path1 = args.image_path1
    img_path2 = args.image_path2
    # build model
    args.usebias = True
    if(args.usebias):

        model = vits.__dict__["vit_small"](
            patch_size=args.patch_size,
            num_classes=0
            # stochastic depth
        )
    else:
        model = vits.__dict__["vit_tiny"](
            patch_size=4,
            num_classes=0,
            qkv_bias=False
            # stochastic depth
        )

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
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

    with open(img_path1, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    with open(img_path2, 'rb') as f2:
        img2 = Image.open(f2)
        img2 = img2.convert('RGB')
    img = transform(img).unsqueeze(dim = 0)
    img2 = transform(img2).unsqueeze(dim = 0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size
    if(args.lettertoinject == 0):

        attentions, cls = model.get_last_self_and_cls(img.to(device), args.layertoinject)
        attentions2, cls2 = model.get_last_self_and_cls(img2.to(device), args.layertoinject)

        attentionsInje = model.inject_cls(img.to(device), args.layertoinject, cls2)
        attentionsInje2 = model.inject_cls(img2.to(device), args.layertoinject, cls)
    else:
        attentions, cls = model.get_last_self_and_clsqkv(img.to(device), args.layertoinject, args.lettertoinject)
        attentions2, cls2 = model.get_last_self_and_clsqkv(img2.to(device), args.layertoinject, args.lettertoinject)

        attentionsInje = model.inject_clsqkv(img.to(device), args.layertoinject, cls2, args.lettertoinject)
        attentionsInje2 = model.inject_clsqkv(img2.to(device), args.layertoinject, cls, args.lettertoinject)
    nh = attentions.shape[1]
    for i in range(4):
        imgname = "cat"
        if (i == 1):
            attentions = attentions2
            imgname = "dog"
        if (i == 2):
            attentions = attentionsInje
            imgname = "catswap"
        if (i == 3):
            attentions = attentionsInje2
            imgname = "dogswap"
        # 6 heads
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
            0].cpu().numpy()
        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        for j in range(nh):
            fname = os.path.join(args.output_dir, imgname + "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")







