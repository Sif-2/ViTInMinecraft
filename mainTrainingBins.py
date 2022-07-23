import argparse
import glob
import math

import PIL.ImageFont
import torchvision
import matplotlib

import random
import statistics
import cv2

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import ctypes
from sklearn.cluster import KMeans

from natsort import natsorted
import collections
from nets import *
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn.functional as F
import pickle
import minerl
import gzip
from PIL import Image, ImageDraw, ImageFont
from operator import itemgetter

import vision_transformer as vits
from torchvision import transforms as pth_transforms

libgcc_s = ctypes.CDLL('libgcc_s.so.1')  # libgcc_s.so.1 error workaround

device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
#device = T.device("cpu")

FOURCC = {
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
}


def get_moving_avg(x, n=10):
    cumsum = np.cumsum(x)
    return (cumsum[n:] - cumsum[:-n]) / n


class Handler():
    def __init__(self, args: argparse.Namespace):
        self.I = None
        self.train_loader = None
        self.Y = None
        self.X = None
        self.critic = None
        self.args = args
        argdict = args.__dict__
        self.font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 10)
        print("device:", device)
        self.models = dict()
        self.criticname = "critic"

        self.reset_models()
        self.models[self.criticname] = self.critic
        self.critic_args = "-".join([f"{a}={argdict[a]}" for a in
                                     ["arch", "rewidx", "cepochs", "datamode", "datasize", "shift", "chfak",
                                      "dropout", "usebins"] if argdict[a]])

        # SETUP PATHS
        self.path = f"{args.name}/"
        self.train_path = self.path + "train/"
        self.result_path = self.path + "results/"
        self.save_path = self.path + "saves/"
        self.data_path = "runs/data/straight/"
        self.save_paths = {
            self.criticname: f"{self.save_path}critic-{self.critic_args}.pt",
        }

    def reset_models(self):
        args = self.args
        if(args.usebias):
            self.critic = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.dropout, num_classes=0)
        else:
            self.critic = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.dropout, num_classes=0, qkvbias=False)
        if os.path.isfile(args.pretrain):
            state_dict = T.load(args.pretrain, map_location="cpu")
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.critic.load_state_dict(state_dict, strict=False)
            print(
                'Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrain,
                                                                                msg))

    def load_data(self, batch_size=10):
        args = self.args
        X, Y, I = self.collect_data()

        self.X, self.Y, self.I = X, Y, I

        print("dataset shapes", X.shape, Y.shape, self.X.shape, self.Y.shape)
        if args.oneep:
            self.X2, self.Y2, self.I2 = self.collect_data(epis=True)
            self.ep_loader = T.utils.data.DataLoader(
                T.utils.data.TensorDataset(T.from_numpy(self.X2),
                                           T.from_numpy(self.Y2).t(),
                                           T.arange(self.X2.shape[0], dtype=T.int32)),
                batch_size=1000)
        self.train_loader = T.utils.data.DataLoader(
            T.utils.data.TensorDataset(T.from_numpy(self.X),
                                       T.from_numpy(self.Y).t(),
                                       T.arange(self.X.shape[0], dtype=T.int32)),
            batch_size=batch_size, shuffle=True)

    def load_models(self, modelnames=[]):
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            save_path = self.save_paths[model]
            if not os.path.exists(save_path):
                if not self.args.train:
                    print(f"{save_path} not found")
                return False
            print("loading:", save_path)
            self.models[model].load_state_dict(T.load(save_path, map_location=T.device(device)))
        return True

    def save_models(self, modelnames=[]):
        os.makedirs(self.save_path, exist_ok=True)
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            save_path = self.save_paths[model]
            print("saving:", save_path)
            T.save(self.models[model].state_dict(), save_path)

    def critic_pipe(self):
        args = self.args
        loader = self.train_loader
        bin_size = 1 / 192

        if args.cload and self.load_models([self.criticname]):
            print("loaded critic, no new training")
            return

        transform = pth_transforms.Compose(
            [

                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        # Setup save path and Logger
        result_path = self.path + "critic/"
        os.makedirs(result_path, exist_ok=True)

        llog = []
        llog2 = []

        critic = self.critic
        critic = critic.to(device)
        opti = T.optim.Adam(critic.parameters())
        # Epoch and Batch Loops
        for epoch in range(self.args.cepochs):
            for b_idx, (X, Y, I) in enumerate(loader):
                # SHIFT
                if args.shift:
                    X = self.shift_batch(X)

                rewardtensorlist = []
                # FORMATING
                XP = X.permute(0, 3, 1, 2).float().to(device) / 255.0
                XP = transform(XP)
                Y = Y[:, args.rewidx].float().to(device)

                if args.usebins:

                    for i in range(Y.shape[0]):
                        bin_number = int((Y[i].item()) / bin_size)
                        if bin_number == 192:
                            bin_number = 191
                        rewardtensor = T.zeros(192)
                        rewardtensor[bin_number] = 1
                        rewardtensorlist.append(rewardtensor)
                    rewardtensorlist = T.stack(rewardtensorlist).to(device)
                pred = critic(XP)
                # pred = pred.softmax(dim=-1)
                loss = 0

                if args.usebins:
                    loss = F.cross_entropy(pred, T.max(rewardtensorlist, 1)[1])
                    # loss = F.mse_loss(pred[T.max(rewardtensorlist, 1)[1]], reward)
                else:
                    loss = F.mse_loss(pred, Y)
                print(f"critic e{epoch + 1} b{b_idx}", loss.item(), end="\r")
                opti.zero_grad()
                loss.backward()
                opti.step()
                llog.append(loss.item())

                # VIZ -----------------------------------
                if not b_idx % 100:  # VISUALIZE
                    vizs = []

                    viz = X.cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(viz))
                    draw = ImageDraw.Draw(img)
                    if args.usebins:
                        for i, value in enumerate(T.max(rewardtensorlist, 1)[1].tolist()):
                            x, y = int(i * img.width / len(X)), 1
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    else:
                        for i, value in enumerate(Y.tolist()):
                            x, y = int(i * img.width / len(X)), 1
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)
                    for i, value in enumerate(pred.softmax(dim=-1).tolist()):
                        x, y = int(i * img.width / len(X)), int(1 + img.height / 2)
                        if args.usebins:
                            if not b_idx % 500 and args.visbar:
                                x2 = range(192)
                                y2 = value
                                plt.clf()
                                plt.bar(x2, y2)
                                plt.xlabel('bin')
                                plt.ylabel("Value")
                                plt.title('Softmax predicted value')
                                plt.savefig(result_path + f"e{epoch}_b{b_idx}_bar{i}.png")
                            predvalue = 0
                            for j, nr in enumerate(value):
                                predvalue += j * nr
                            draw.text((x, y), str(round(predvalue, 3)), fill=(255, 255, 255), font=self.font)
                            fakeloss = float(abs(predvalue - T.max(rewardtensorlist, 1)[1][i]))
                            llog2.append(fakeloss)
                        else:
                            draw.text((x, y), str(round(value, 3)), fill=(255, 255, 255), font=self.font)

                    # plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path + f"e{epoch}_b{b_idx}.png")

            if not (epoch + 1) % args.saveevery:
                self.save_models(modelnames=[self.criticname])

            plt.clf()
            plt.plot(get_moving_avg(llog, 30), label="Train Loss")
            plt.ylim(0, plt.ylim()[1])
            plt.legend()
            plt.savefig(result_path + "_loss.png")

            plt.clf()
            plt.plot(get_moving_avg(llog2, 30), label="Difference between values")
            plt.ylim(0, plt.ylim()[1])
            plt.legend()
            plt.savefig(result_path + "_fakeloss.png")

        print()

    def createVis(self, numimgs, layer):
        inp = "./OutputDir"
        os.makedirs("kmeanimgs", exist_ok=True)
        rltrenner = Image.open("NeededFiles/rltrenner.png")
        rltrenner = rltrenner.convert("RGB")

        yseperator = 20
        print("Creating images")
        for idx, numimg in enumerate(tqdm(range(numimgs))):
            imagesvert = []
            name = "img" + str(numimg)

            aimg = Image.open("OutputDir/" + name + ".png")
            aimg = aimg.convert("RGB")

            for depth in layer:
                imagesinline = [aimg]

                for letter in range(3):
                    templist = natsorted(glob.glob(os.path.join(inp, name + "qdepth" + str(depth) + "head" + "*.jpg")))
                    if letter == 1:
                        templist = natsorted(
                            glob.glob(os.path.join(inp, name + "kdepth" + str(depth) + "head" + "*.jpg")))
                        imagesinline.append(aimg)
                    if letter == 2:
                        templist = natsorted(
                            glob.glob(os.path.join(inp, name + "vdepth" + str(depth) + "head" + "*.jpg")))
                        imagesinline.append(aimg)
                    for timg in templist:
                        timg = Image.open(timg)
                        timg = timg.convert("RGB")
                        imagesinline.append(timg)
                    if letter is not 2:
                        imagesinline.append(rltrenner)

                widths, heights = zip(*(i.size for i in imagesinline))

                total_width = sum(widths)
                max_height = max(heights)

                new_im = Image.new('RGB', (total_width, max_height))

                x_offset = 0
                for im in imagesinline:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                imagesvert.append(new_im.convert('RGB'))

            widths, heights = zip(*(i.size for i in imagesvert))

            total_height = sum(heights) + (len(layer) * yseperator)
            max_width = max(widths)

            new_im2 = Image.new('RGB', (max_width, total_height))

            y_offset = 0
            for im in imagesvert:
                new_im2.paste(im, (0, y_offset))
                y_offset += im.size[1] + yseperator

            new_im2.save(os.path.join("kmeanimgs/", "img" + str(idx) + ".jpg"))

    def shift_batch(self, X):
        xshift = int(self.args.shift * T.rand(1))
        if T.rand(1) > 0.5:
            # X = T.cat((X[:, :, yshift:], X[:, :, :yshift]), dim=2)
            X = T.cat((X[:, :, xshift:], X[:, :, :xshift]), dim=2)
        else:
            X = T.cat((X[:, :, -xshift:], X[:, :, :-xshift]), dim=2)
        return X

    def collect_data(self, epis=False):
        args = self.args
        datadir = self.data_path
        envname = args.envname
        mode = args.datamode
        filepath = datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle"
        print("collecting dataset at", filepath)
        if (not epis):
            if os.path.exists(filepath):
                print("loading existing dataset...")
                with gzip.open(datadir + f"{envname}-{mode}-{args.datasize}-[{args.gammas}].pickle", 'rb') as fp:
                    X, Y, I = pickle.load(fp)
                print("finished loading exisiting dataset")
                return X, Y, I

        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRL{envname}VectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment=f'MineRL{envname}VectorObf-v0')
        data = minerl.data.make(f'MineRL{envname}VectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()

        size = args.datasize
        if (epis):
            size = 200
        # np.random.default_rng().shuffle(names)
        X = np.zeros((size, 64, 64, 3), dtype=np.uint8)
        Y = np.zeros((7, size), dtype=np.float)
        I = np.zeros(size, dtype=np.uint16)
        print("collecting straight data set with", args.datasize, "frames")

        if (args.oneep):
            pickedep = args.pickep
            names.remove(pickedep)
            if epis:
                names = [pickedep]

        # DEV
        full_ep_lens = 0

        runidx = 0
        for name_idx, name in enumerate(names):
            # print(name)
            print("percentage of episodes used so far:", round(name_idx / len(names) * 100),
                  "dataset size:", runidx,
                  "full ep lens:", full_ep_lens)
            # EXTRACT EPISODE
            state, action, reward, _, done = zip(*data.load_data(name))
            reward = np.array(reward)
            pov = np.stack([s['pov'] for s in state])

            # get full ep len of all
            full_ep_lens += len(pov)

            if mode == "begin":
                # Only take frames until first reward:
                add = np.argmax(reward > 0) + 1 if reward.any() else add
                if add > 1000:
                    continue
                # print("first reward frame idx", add)
                reward = reward[:add]
            elif mode == "trunk":
                mask = [True] + [np.sum(reward[max(0, i - 35):i]) == 0 for i in range(1, len(reward))]
                pov = pov[mask]
                reward = reward[mask]

            add = min(size - runidx, len(pov))
            reward = reward[:add]
            reward = (reward > 0).astype(np.float)
            X[runidx:runidx + add] = pov[:add]
            Y[0, runidx:runidx + add] = reward
            I[runidx:runidx + add] = range(len(pov))[:add]

            for rewidx, gamma in \
                    enumerate(args.gammas.split('-')):
                # FORMATING RAW REWARD
                gamma = float(gamma)
                local_reward = reward.copy()
                for i in range(2, add + 1):
                    last = gamma * local_reward[-i + 1]
                    current = local_reward[-i]
                    local_reward[-i] = min(current + last, 1)

                Y[rewidx + 1, runidx:runidx + add] = local_reward

            runidx += add
            if epis:
                X = X[:runidx]
                Y = Y[:, :runidx]
                I = I[:runidx]
            if runidx >= size:
                break

        # SAVE AS ZIPPED FILE
        if (not epis):
            with gzip.GzipFile(filepath, 'wb') as fp:
                pickle.dump((X[:runidx], Y[:, :runidx], I[:runidx]), fp)

        # DEV
        print("full ep length:", full_ep_lens, "beginning percentage", size / full_ep_lens)

        return X, Y, I

    def clean_data(self, vis=0):
        args = self.args
        visdir = f"train/data/straight/{args.datasize}-vis/"
        os.makedirs(visdir, exist_ok=True)
        datadir = f"train/data/straight/"
        os.makedirs(datadir, exist_ok=True)
        filepath = datadir + f"{args.datasize}-clean.pickle"
        chunklen = 100
        distance_between_consecutive_rewards = chunklen
        distance_to_reward = 80
        shift = 20

        if os.path.exists(filepath):
            print("loading clean dataset...")
            with gzip.open(datadir + f"{args.datasize}-clean.pickle", 'rb') as fp:
                X, Y, I = pickle.load(fp)
            print("finished loading clean dataset")
            return X, Y, I

        # VIZ CHOPS METHOD
        def save_frame(name, frame):
            path = f"{visdir}" + name + f".png"
            dirpath = os.path.sep.join(path.split(os.path.sep)[:-1])
            # print(path, dirpath)
            os.makedirs(dirpath, exist_ok=True)
            plt.imsave(path, frame)

        X, Y, I = self.collect_data()
        Y = Y[0]

        # EXTRACT CHOPS
        chops = np.nonzero(Y)[0]
        choptimes = I[chops]
        # print("raw chops:", chops)
        deltas = choptimes[1:] - choptimes[:-1]
        big_enough_deltas = deltas > distance_between_consecutive_rewards
        negative_deltas = deltas < 0
        accepted_chop_times = big_enough_deltas | negative_deltas
        clean_chops = np.concatenate((chops[None, 0], chops[1:][accepted_chop_times]))
        # print("clean chops:", clean_chops)

        # EXCTRACT FAR FROM CHOPS
        faridxs = [i for i in range(len(X)) if not
        ((Y[max(i - distance_to_reward, 0):i + distance_to_reward] &
          ((I[max(i - distance_to_reward, 0):i + distance_to_reward] - I[i]) > 0)).any() or Y[i])]
        # print(faridxs)
        # print("faridxs:", set(faridxs).intersection(set(chops)))

        shift_chops = clean_chops[I[clean_chops] >= shift] - shift
        chunk_chops = shift_chops[I[shift_chops] >= chunklen]
        clean_idxs = np.concatenate([1 + np.arange(i - 100, i) for i in chunk_chops])
        for i in range(5):
            Y[chunk_chops - i] = 1
        # print(clean_idxs)
        # print(Y[clean_idxs+20])

        print("ratio of raw chops to all frames:", len(chops) / len(X))
        print("ratio of cleaned chops to raw chops:", len(clean_chops) / len(chops))
        print("ratio of cleaned chops to all frames:", len(clean_chops) / len(X))
        print("final size of clean chunked dataset", len(clean_idxs), "out of", len(X))

        # SAVE CLEAN DATA
        X, Y, I = X[clean_idxs], Y[clean_idxs], I[clean_idxs]
        with gzip.open(filepath, 'wb') as fp:
            pickle.dump((X, Y, I), fp)

        if vis:
            n_samples = 1000

            # clean data chunks
            print(f"saving first {n_samples // chunklen} cleaned chunks")
            for fix, chix in enumerate(clean_idxs[:n_samples]):
                save_frame(f"chunks/{fix // chunklen}/{fix % chunklen}", X[chix])

            if vis > 1:
                # first chops
                print("saving first chops")
                for fix, chix in enumerate(clean_chops[:n_samples]):
                    save_frame(f"first/first-{fix}", X[chix])

                # consec chops
                print("saving consecutive chops")
                for fix, chix in enumerate(list(set(chops).difference(set(clean_chops)))[:n_samples]):
                    save_frame(f"consec/consec-{fix}", X[chix])

                print("saving shifted first chops")
                # shifted chops
                for shift in [5, 10, 15, 20]:
                    shift_chops = clean_chops[I[clean_chops] >= shift] - shift
                    for fix, chix in enumerate(shift_chops[:n_samples]):
                        save_frame(f"shift/shift-{shift}-{fix}", X[chix])

                print("saving far from reward chops")
                # far from chops
                for fix, chix in enumerate(faridxs[:n_samples]):
                    save_frame(f"far/far-{fix}", X[chix])

        return X, Y, I


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-kmean", action="store_true")
    parser.add_argument("-predonly", action="store_true")
    parser.add_argument("-oneep", action="store_true")

    parser.add_argument("-critic", type=bool, default=True)
    parser.add_argument("-cload", type=bool, default=True)
    parser.add_argument("-dinonorm", default=False, action="store_true",
                        help="resizes the dataset and applies dino "
                             "normalization")
    parser.add_argument("-visbar", default=False, action="store_true")
    parser.add_argument("-freeze", type=bool, default=False)

    parser.add_argument("-usebins", default=False, action="store_true")
    parser.add_argument("-usebias", default=True, action="store_true")
    parser.add_argument('--arch', default='vit_tiny', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=(64, 64), type=int, nargs="+", help="Resize image.")

    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--pickep', default='v3_absolute_grape_changeling-7_14600-16079', type=str)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--envname", type=str, default="Treechop")
    parser.add_argument("--visname", type=str, default="curves")
    parser.add_argument("--datamode", type=str, default="trunk")
    parser.add_argument("--purevis", type=str, default="")
    parser.add_argument("--sortidx", type=int, default=1)
    parser.add_argument("--chfak", type=int, default=1)
    parser.add_argument("--shift", type=int, default=12)
    parser.add_argument("--neck", type=int, default=32)
    parser.add_argument("--clossfak", type=int, default=5)
    parser.add_argument("--cepochs", type=int, default=15)
    parser.add_argument("--rewidx", type=int, default=1)
    parser.add_argument("--gammas", type=str, default="0.98-0.97-0.96-0.95")
    parser.add_argument("--datasize", type=int, default=5000)
    parser.add_argument("--name", type=str, default="default-model")
    parser.add_argument("--model", type=str, default="default-model")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--saveevery", type=int, default=5)

    args = parser.parse_args()
    args.workers = (5, 1, 1)
    args.name = args.model

    # print(args)
    H = Handler(args)
    if args.train or args.kmean:
        if (args.oneep):
            H.load_data(batch_size=1)
        else:
            H.load_data(batch_size=64)
    if args.cload:
        H.load_models(modelnames=[H.criticname])
    if args.train:
        if args.critic:
            H.critic_pipe()
            H.save_models(modelnames=[H.criticname])


if __name__ == "__main__":
    main()
