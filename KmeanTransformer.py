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

        # L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def reset_models(self):
        args = self.args

        self.critic = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.dropout, num_classes=0)
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
        # if (args.dinonorm):
        #     self.train_loader = T.utils.data.DataLoader(
        #         T.utils.data.TensorDataset(self.X,
        #                                    T.from_numpy(self.Y).t(),
        #                                    T.arange(self.X.shape[0], dtype=T.int32)),
        #         batch_size=batch_size, shuffle=True)
        # else:
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


    def save_models(self, modelnames=[]):
        os.makedirs(self.save_path, exist_ok=True)
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            save_path = self.save_paths[model]
            print("saving:", save_path)
            T.save(self.models[model].state_dict(), save_path)

    def kmean_video(self):
        OutputDir = "OutputDir/"
        os.makedirs("OutputDir", exist_ok=True)
        counter = 0
        imgsToPrint = 5
        clusternum = 5
        layerToUse = range(12)
        numKmeanimgs = 100
        args = self.args
        loader = self.train_loader
        imgsforkmean = []
        critic = self.critic
        critic = critic.to(device)
        for p in critic.parameters():
            p.requires_grad = False
        critic.eval()
        transform = pth_transforms.Compose(
            [

                pth_transforms.Resize(128),
                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        # kmean predictor
        qpredictorlist = []
        kpredictorlist = []
        vpredictorlist = []

        # counts appearance of pixel
        kmeancountq = []
        kmeancountk = []
        kmeancountv = []
        for b_idx, (X, Y, I) in enumerate(loader):
            Y = Y[:, args.rewidx].float().to(device)
            Y2 = (Y.cpu().numpy() > 0.7).astype(np.bool)
            XP = X.permute(0, 3, 1, 2).float().to(device) / 255.0
            XP = transform(XP)

            if (len(imgsforkmean)) < numKmeanimgs:
                for i in range(len(Y2)):
                    if (Y2[i] == True):
                        imgsforkmean.append(XP[i])
                if (len(imgsforkmean)) >= numKmeanimgs:
                    print("Getting kqv values: \n")
                    qvalues = []
                    kvalues = []
                    vvalues = []
                    for idx, z in enumerate(tqdm(imgsforkmean)):
                        kqvvaluelist = critic.init_kmean_predictor(z.to(device).unsqueeze(0))
                        for i in range(len(kqvvaluelist)):
                            if idx == 0:
                                qvalues.append(kqvvaluelist[i]["qvalue"])
                                kvalues.append(kqvvaluelist[i]["kvalue"])
                                vvalues.append(kqvvaluelist[i]["vvalue"])
                            else:
                                qvalues[i] = T.cat((qvalues[i], kqvvaluelist[i]["qvalue"]), 0)
                                kvalues[i] = T.cat((kvalues[i], kqvvaluelist[i]["kvalue"]), 0)
                                vvalues[i] = T.cat((vvalues[i], kqvvaluelist[i]["vvalue"]), 0)
                        del kqvvaluelist
                    print("\n Generate Kmean \n ")
                    for z in layerToUse:  # tqdm(range(len(qvalues))):
                        print("\n Layer " + str(z) + "\n")
                        qpredictor, kpredictor, vpredictor = critic.getKMeansPredictor(qvalues[z], kvalues[z],
                                                                                       vvalues[z], clusternum)
                        qpredictorlist.append(qpredictor)
                        kpredictorlist.append(kpredictor)
                        vpredictorlist.append(vpredictor)

            else:
                testimgs = []
                for l in range(XP.shape[0]):
                    testimgs.append(XP[l])

                for idx, img in enumerate(testimgs):
                    kqvvaluelist = critic.init_kmean_predictor(img.to(device).unsqueeze(0))
                    isHigh = "Low"
                    if Y2[idx]:
                        isHigh = "High"

                    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                                 os.path.join(OutputDir,
                                                              "img" + str(counter) + ".png"))  # + str(isHigh)

                    for posid, i in enumerate(layerToUse):  # range(len(kqvvaluelist)):
                        for f in range(3):
                            qvalue = kqvvaluelist[i]["qvalue"]
                            predictor = qpredictorlist
                            letter = "q"
                            if f == 1:
                                qvalue = kqvvaluelist[i]["kvalue"]
                                predictor = kpredictorlist
                                letter = "k"
                            if f == 2:
                                qvalue = kqvvaluelist[i]["vvalue"]
                                predictor = vpredictorlist
                                letter = "v"
                            reshapesize = int(math.sqrt(qvalue.size(2) - 1))
                            kronsize = args.patch_size
                            for j in range(6):
                                attention2 = qvalue[0, j, 1:, :]
                                kmeanspred = predictor[posid][j].predict(attention2.cpu().numpy())  # 0 -> i
                                test = collections.Counter(kmeanspred)
                                if (counter == 0):
                                    if f == 0:
                                        kmeancountq.append([[test, isHigh]])
                                    if f == 1:
                                        kmeancountk.append([[test, isHigh]])
                                    if f == 2:
                                        kmeancountv.append([[test, isHigh]])
                                else:
                                    if f == 0:
                                        kmeancountq[j].append([test, isHigh])
                                    if f == 1:
                                        kmeancountk[j].append([test, isHigh])
                                    if f == 2:
                                        kmeancountv[j].append([test, isHigh])
                                kmeanspred = kmeanspred.reshape(reshapesize, reshapesize)
                                kmeanspred = np.kron(kmeanspred, np.ones((kronsize, kronsize)))
                                # kmeanspredtensor = torch.tensor(kmeanspred)
                                # kmeanspred = nn.functional.interpolate(kmeanspredtensor, scale_factor=16, mode="nearest").cpu().numpy()
                                fname = os.path.join(OutputDir, "img" + str(counter) + letter + "depth" + str(i)
                                                     + "head" + str(j) + ".jpg")
                                plt.imsave(fname=fname, arr=kmeanspred, format='jpg')

                    counter += 1
                    if counter == imgsToPrint:
                        self.createVis(counter, layerToUse)
                        exit()

                print("hi")
    def kmean_pipe(self):
        OutputDir = "OutputDir/"
        os.makedirs("OutputDir", exist_ok=True)
        pickedImages = []

        thresholdkmeanlow = 0.20
        thresholdkmeanhigh = 0.80
        amountCluster = 5
        layerToUse = [9]
        numKmeanimgs = 100
        loadKmeanPredictor = True
        thresholdkmean = 0.8
        secondClusterNum = 5
        if (self.args.oneep):
            thresholdkmeanlow = 0
            thresholdkmeanhigh = 0
        valueCritic = NewCritic(bottleneck=32, chfak=1, dropout=0.3)
        valueCritic.load_state_dict(T.load("NeededFiles/valueCritic30.pt", map_location=device))
        valueCritic = valueCritic.to(device)
        for p in valueCritic.parameters():
            p.requires_grad = False
        valueCritic.eval()
        loader = self.train_loader
        loaderep = self.ep_loader
        imgsforkmean = []
        critic = self.critic
        critic = critic.to(device)

        # pickedvidletter = [1, 0, 0, 0, 1, 0, 1, 1, 0, 1]
        #
        # pickedvidhead = [5, 0, 3, 2, 3, 1, 1, 2, 5, 0]
        #
        # pickedvidcluster = [4, 1, 0, 1, 0, 3, 4, 4, 4, 3]


        pickedvidletter = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        pickedvidhead = [4, 5, 5, 3, 0, 2, 3, 1, 0, 2]
        pickedvidcluster = [3, 3, 3, 4, 4, 3, 0, 1, 2, 3]


        for p in critic.parameters():
            p.requires_grad = False
        critic.eval()
        transform = pth_transforms.Compose(
            [

                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        resizer480 = pth_transforms.Compose(
            [

                pth_transforms.Resize(512),
            ]
        )

        # kmean predictor
        qpredictorlist = []
        kpredictorlist = []
        vpredictorlist = []

        afterPredictor = []

        for b_idx, (X, Y, I) in enumerate(loader):
            # Y = Y[:, args.rewidx].float().to(device)

            XPnorm = X.permute(0, 3, 1, 2).float().to(device) / 255.0
            Y = valueCritic(XPnorm)
            Y2 = (Y.cpu().numpy() > thresholdkmean).astype(np.bool)
            # Y2img = (Y.cpu().numpy() > thresholdimg).astype(np.bool)
            XP = resizer480(XPnorm)
            XP = transform(XP)  # dino norm

            if (len(imgsforkmean)) < numKmeanimgs:
                for i in range(len(Y2)):
                    if (Y2[i] == True):
                        imgsforkmean.append(XP[i])
                    if (len(imgsforkmean)) >= numKmeanimgs:
                        break
                if (len(imgsforkmean)) >= numKmeanimgs:
                    if(loadKmeanPredictor):
                        print("LOADING KMEAN PREDICTOR")
                        with open("kmeanFirstIteration", "rb") as fp:
                            predictorlist = pickle.load(fp)
                            qpredictorlist = predictorlist[0]
                            kpredictorlist = predictorlist[1]
                            vpredictorlist = predictorlist[2]

                    print("Getting kqv values: \n")
                    qvalues = []
                    kvalues = []
                    vvalues = []
                    for idx, z in enumerate(tqdm(imgsforkmean)):
                        kqvvaluelist = critic.init_kmean_predictor(z.to(device).unsqueeze(0))
                        for i in range(len(kqvvaluelist)):
                            if idx == 0:
                                qvalues.append(kqvvaluelist[i]["qvalue"].cpu().numpy())
                                kvalues.append(kqvvaluelist[i]["kvalue"].cpu().numpy())
                                vvalues.append(kqvvaluelist[i]["vvalue"].cpu().numpy())
                            else:
                                qvalues[i] = np.concatenate((qvalues[i], kqvvaluelist[i]["qvalue"].cpu().numpy()), 0)
                                kvalues[i] = np.concatenate((kvalues[i], kqvvaluelist[i]["kvalue"].cpu().numpy()), 0)
                                vvalues[i] = np.concatenate((vvalues[i], kqvvaluelist[i]["vvalue"].cpu().numpy()), 0)
                                # qvalues[i] = T.cat((qvalues[i], kqvvaluelist[i]["qvalue"]), 0)
                                # kvalues[i] = T.cat((kvalues[i], kqvvaluelist[i]["kvalue"]), 0)
                                # vvalues[i] = T.cat((vvalues[i], kqvvaluelist[i]["vvalue"]), 0)
                        del kqvvaluelist
                        if not loadKmeanPredictor:
                            print("\n Generate Kmean \n ")
                            for z in layerToUse:  # tqdm(range(len(qvalues))):
                                print("\n Layer " + str(z) + "\n")
                                qpredictor, kpredictor, vpredictor = critic.getKMeansPredictor(qvalues[z], kvalues[z],
                                                                                               vvalues[z], amountCluster)
                                qpredictorlist.append(qpredictor)
                                kpredictorlist.append(kpredictor)
                                vpredictorlist.append(vpredictor)

                    for pred2 in range(10):
                        anLetter = pickedvidletter[pred2]
                        anHead = pickedvidhead[pred2]
                        anCluster = pickedvidcluster[pred2]
                        qkvvalues = None
                        predictor2 = None
                        if (anLetter == 0):
                            qkvvalues = qvalues[layerToUse[0]]
                            predictor2 = qpredictorlist[0][anHead]
                        if (anLetter == 1):
                            qkvvalues = kvalues[layerToUse[0]]
                            predictor2 = kpredictorlist[0][anHead]
                        if (anLetter == 2):
                            qkvvalues = vvalues[layerToUse[0]]
                            predictor2 = vpredictorlist[0][anHead]
                        qkvtemp = qkvvalues[:, :, 1:, :]  # remove cls
                        qkvtemp = qkvtemp[:, anHead, :, :]  # take head
                        a = qkvtemp.shape[0]
                        b = qkvtemp.shape[1]
                        c = qkvtemp.shape[2]
                        qkvtemp = qkvtemp.reshape(a * b, c)
                        predlabels = predictor2.predict(qkvtemp)
                        qkvtemp2 = None
                        secCluFirst = True
                        for counter3 in tqdm(range(len(qkvtemp))):
                            if (predlabels[counter3] == anCluster):
                                if (secCluFirst):
                                    qkvtemp2 = np.expand_dims(qkvtemp[counter3], axis=0)
                                    secCluFirst = False
                                else:
                                    qkvtemp2 = np.concatenate((qkvtemp2, np.expand_dims(qkvtemp[counter3], axis=0)), axis=0)
                        print("secondCluster" + str(pred2))
                        afterkqvpredictor = KMeans(n_clusters=secondClusterNum).fit(qkvtemp2)
                        afterPredictor.append(afterkqvpredictor)
                    break

        for b_idx, (X, Y, I) in enumerate(loaderep):
            XPnorm = X.permute(0, 3, 1, 2).float().to(device) / 255.0
            Y = valueCritic(XPnorm)
            Y2high = (Y.cpu().numpy() > thresholdkmeanhigh).astype(np.bool)
            Y2low = (Y.cpu().numpy() < thresholdkmeanlow).astype(np.bool)
            # Y2img = (Y.cpu().numpy() > thresholdimg).astype(np.bool)
            XP = resizer480(XPnorm)
            XP = transform(XP)  # dino norm
            testimgs = []

            for l in range(XP.shape[0]):
                testimgs.append(XP[l])

            for idx, img in enumerate(tqdm(testimgs)):
                kqvvaluelist = critic.init_kmean_predictor(img.to(device).unsqueeze(0))
                isHigh = "Med"
                if Y2high[idx]:
                    isHigh = "High"
                if Y2low[idx]:
                    isHigh = "Low"
                if isHigh == "High" or isHigh == "Low":
                    torchvision.utils.save_image(torchvision.utils.make_grid(resizer480(XPnorm[idx])),
                                                 os.path.join(OutputDir,
                                                              "img" + str(idx) + "val" + str(
                                                                  round(Y[idx].item(), 2)) + ".jpg"))  # + str(isHigh)
                    pickedImages.append(X[idx])
                    for pickedcell in range(10):
                        vidletter = [pickedvidletter[pickedcell]]
                        vidhead = [pickedvidhead[pickedcell]]
                        vidcluster = pickedvidcluster[pickedcell]
                        for posid, i in enumerate(layerToUse):  # range(len(kqvvaluelist)):

                            for f in vidletter:
                                qvalue = kqvvaluelist[i]["qvalue"]
                                predictor = qpredictorlist
                                letter = "q"
                                if f == 1:
                                    qvalue = kqvvaluelist[i]["kvalue"]
                                    predictor = kpredictorlist
                                    letter = "k"
                                if f == 2:
                                    qvalue = kqvvaluelist[i]["vvalue"]
                                    predictor = vpredictorlist
                                    letter = "v"
                                reshapesize = int(math.sqrt(qvalue.size(2) - 1))
                                kronsize = 16
                                if (reshapesize == 8):
                                    kronsize = 60
                                if (reshapesize == 15):
                                    kronsize = 32

                                for j in vidhead:
                                    attention2 = qvalue[0, j, 1:, :]
                                    kmeanspred = predictor[posid][j].predict(attention2.cpu().numpy())  # 0 -> i
                                    kmeanspredAfter = afterPredictor[pickedcell].predict(attention2.cpu().numpy())
                                    kmeanspred = kmeanspred.reshape(reshapesize, reshapesize)
                                    kmeanspredAfter = kmeanspredAfter.reshape(reshapesize, reshapesize)

                                    kmeanspred = np.kron(kmeanspred, np.ones((kronsize, kronsize)))
                                    kmeanspred[kmeanspred != vidcluster] = -1
                                    kmeanspred[kmeanspred == vidcluster] = 1
                                    kmeanspred[kmeanspred == -1] = 0

                                    beforeimg = XPnorm[idx]
                                    beforeimg2 = resizer480(beforeimg)
                                    secondClusterImageList = []
                                    for cln in range(secondClusterNum):
                                        tempimg = beforeimg2.clone().detach()
                                        secondClusterImageList.append(tempimg)
                                    for y in range(reshapesize):
                                        y2 = y * 16
                                        for x in range(reshapesize):
                                            x2 = x * 16
                                            if kmeanspred[x2, y2] != 1:
                                                beforeimg2[:, x2:(x2 + 16), y2:(y2 + 16)] = beforeimg2[:, x2:(x2 + 16),
                                                                                            y2:(y2 + 16)] * 0.1
                                                for cln3 in range(secondClusterNum):
                                                    secondClusterImageList[cln3][:, x2:(x2 + 16), y2:(y2 + 16)] = \
                                                        secondClusterImageList[cln3][:, x2:(x2 + 16),
                                                        y2:(y2 + 16)] * 0.1

                                            else:
                                                for cln2 in range(secondClusterNum):
                                                    if kmeanspredAfter[x, y] != cln2:
                                                        secondClusterImageList[cln2][:, x2:(x2 + 16), y2:(y2 + 16)] = \
                                                            secondClusterImageList[cln2][:, x2:(x2 + 16),
                                                            y2:(y2 + 16)] * 0.1

                                    torchvision.utils.save_image(torchvision.utils.make_grid(beforeimg2),
                                                                 os.path.join(OutputDir,
                                                                              "img" + str(idx) + "letter"
                                                                              + str(letter)
                                                                              + "head" + str(j) + "cluster" + str(
                                                                                  vidcluster) + ".jpg"))
                                    for secId, secimg in enumerate(secondClusterImageList):
                                        torchvision.utils.save_image(torchvision.utils.make_grid(secimg),
                                                                     os.path.join(OutputDir,
                                                                                  "img" + str(idx) + "letter"
                                                                                  + str(letter)
                                                                                  + "head" + str(j) + "cluster" + str(
                                                                                      vidcluster) +
                                                                                  "secondCluster" + str(secId)
                                                                                  + ".jpg"))

            print("batch loaded")
            finallist = []
            finallist.append(qpredictorlist)
            finallist.append(kpredictorlist)
            finallist.append(vpredictorlist)
            finallist.append(afterPredictor)
            with open("predictor", "wb") as fp:
                pickle.dump(finallist, fp)


            self.create_kmean_video(numSecCluster = secondClusterNum)


    def createPredictor(self):
        os.makedirs("Predictors", exist_ok=True)
        amountCluster = 30
        layerToUse = range(12)
        numKmeanimgs = 100  #saddasasd
        thresholdkmean = 0

        valueCritic = NewCritic(bottleneck=32, chfak=1, dropout=0.3)
        valueCritic.load_state_dict(T.load("NeededFiles/valueCritic30.pt", map_location=device))
        valueCritic = valueCritic.to(device)
        for p in valueCritic.parameters():
            p.requires_grad = False
        valueCritic.eval()
        loader = self.train_loader
        imgsforkmean = []
        critic = self.critic
        critic = critic.to(device)

        for p in critic.parameters():
            p.requires_grad = False
        critic.eval()
        transform = pth_transforms.Compose(
            [

                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        resizesize = 512
        resizer480 = pth_transforms.Compose(
            [

                pth_transforms.Resize(resizesize),
            ]
        )
        # kmean predictor


        for b_idx, (X, Y, I) in enumerate(loader):
            # Y = Y[:, args.rewidx].float().to(device)

            XPnorm = X.permute(0, 3, 1, 2).float().to(device) / 255.0
            Y = valueCritic(XPnorm)
            Y2 = (Y.cpu().numpy() >= thresholdkmean).astype(np.bool)
            # Y2img = (Y.cpu().numpy() > thresholdimg).astype(np.bool)
            XP = resizer480(XPnorm)
            XP = transform(XP)  # dino norm

            if (len(imgsforkmean)) < numKmeanimgs:
                for i in range(len(Y2)):
                    if (Y2[i] == True):
                        imgsforkmean.append(XP[i])
                if (len(imgsforkmean)) >= numKmeanimgs:
                    print("Getting kqv values: \n")
                    qvalues = []
                    kvalues = []
                    vvalues = []
                    for idx, z in enumerate(tqdm(imgsforkmean)):
                        kqvvaluelist = critic.init_kmean_predictor(z.to(device).unsqueeze(0))
                        for i in range(len(kqvvaluelist)):
                            if idx == 0:
                                qvalues.append(kqvvaluelist[i]["qvalue"].cpu().numpy())
                                kvalues.append(kqvvaluelist[i]["kvalue"].cpu().numpy())
                                vvalues.append(kqvvaluelist[i]["vvalue"].cpu().numpy())
                            else:
                                qvalues[i] = np.concatenate((qvalues[i], kqvvaluelist[i]["qvalue"].cpu().numpy()), 0)
                                kvalues[i] = np.concatenate((kvalues[i], kqvvaluelist[i]["kvalue"].cpu().numpy()), 0)
                                vvalues[i] = np.concatenate((vvalues[i], kqvvaluelist[i]["vvalue"].cpu().numpy()), 0)
                        del kqvvaluelist
                    print("\n Generate Kmean \n ")
                    for z in layerToUse:  # tqdm(range(len(qvalues))):
                        print("\n Layer " + str(z) + "\n")
                        qpredictor, kpredictor, vpredictor = critic.getKMeansPredictor(qvalues[z], kvalues[z],
                                                                                       vvalues[z], amountCluster)

                        qpredictorlist = []
                        kpredictorlist = []
                        vpredictorlist = []
                        qpredictorlist.append(qpredictor)
                        kpredictorlist.append(kpredictor)
                        vpredictorlist.append(vpredictor)
                        finallist = []
                        finallist.append(qpredictorlist)
                        finallist.append(kpredictorlist)
                        finallist.append(vpredictorlist)
                        with open("Predictors/predictor" + str(z), "wb") as fp:
                            pickle.dump(finallist, fp)
                    exit()


    def create_kmean_video(self, numSecCluster):
        numFirstCluster = 5
        lineLen = numSecCluster + 1
        allimages = natsorted(glob.glob(os.path.join("./OutputDir", "img" + "*.jpg")))
        numDivider = ( 2 * numFirstCluster * (lineLen) ) + 1
        amountImg = int(len(allimages) / numDivider)
        logger = open("logfororder.txt", "a+")
        isFirst = True
        imagesForVideo = []
        args = self.args
        logger.write("Used episode: " + str(args.pickep) + "\n")

        for imgind in tqdm(range(amountImg)):
            startind = numDivider * imgind
            endind = numDivider * (imgind + 1)
            imagesVert = []
            imagesLine = allimages[startind:endind]
            firstElem = imagesLine[-1]
            imagesLine = imagesLine[:-1]
            imagesLine.insert(0, firstElem)
            if isFirst:
                logger.write("Order \n")
                for x in imagesLine:
                    logger.write(str(x) + "\n")
                isFirst = False
            val = str(imagesLine[0][-8:-4])
            if val[0] == 'l':
                val = val[1:]

            origImg = Image.open(imagesLine[0])
            origImg = origImg.convert("RGB")
            drawimg = ImageDraw.Draw(origImg)
            font = ImageFont.truetype(r'NeededFiles/Ubuntu-M.ttf', 50)
            drawimg.text((15, 15), val, (237, 230, 211), font)

            for i in range(10):
                imagesHor = []
                imagesHor.append(origImg)
                startLine = 1 + (i * lineLen)
                endLine = 1 + ((i + 1) * lineLen)
                oneLine = imagesLine[startLine:endLine]
                for imagex in oneLine:
                    imgx = Image.open(imagex)
                    imgx = imgx.convert("RGB")
                    imagesHor.append(imgx)
                widths, heights = zip(*(i.size for i in imagesHor))

                total_width = sum(widths)
                max_height = max(heights)

                new_im = Image.new('RGB', (total_width, max_height))

                x_offset = 0
                for im in imagesHor:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                imagesVert.append(new_im)

            widths, heights = zip(*(i2.size for i2 in imagesVert))

            total_height = sum(heights)
            max_width = max(widths)

            new_im2 = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for im in imagesVert:
                new_im2.paste(im, (0, y_offset))
                y_offset += im.size[1]
            new_im2 = cv2.cvtColor(np.array(new_im2), cv2.COLOR_RGB2BGR)
            imagesForVideo.append(new_im2)

        img_array = imagesForVideo
        xsize = imagesForVideo[0].shape[0]
        ysize = imagesForVideo[0].shape[1]
        outpsize = (ysize, xsize)

        # Get size of the first image
        out = cv2.VideoWriter(
            os.path.join("./", "video." + "mp4"),
            FOURCC["mp4"],
            5,
            outpsize,
        )
        print("Creating Video")
        for i in tqdm(range(len(img_array))):
            out.write(img_array[i])
        out.release()
        print("Done")

    def kmean_pipe_old(self):
        OutputDir = "OutputDirOld/"
        os.makedirs("OutputDirOld", exist_ok=True)
        logger = open("log.txt", "a+")
        counter = 0
        pickedImages = []
        pickedImagesKMeanVis = [[], [], []]
        kmeanvisfirst = True  # fixed val

        thresholdkmeanlow = 0.20
        thresholdkmeanhigh = 0.80
        takeLowestClusterNumPix = True
        amountCluster = 5
        imgsToPrint = 200 #saddasasd
        highvalimgtouse = 50
        usehighimgonmly = False
        layerToUse = [9]
        logger.write("LayerUsed:" + str(layerToUse) + "\n")
        numKmeanimgs = 100  #saddasasd
        thresholdkmean = 0
        logger.write("Threshold for Kmeanimgs:" + str(thresholdkmean) + "\n")
        logger.write("highvalimgtouse:" + str(highvalimgtouse) + "\n")
        logger.write("amountCluster:" + str(amountCluster) + "\n")
        logger.write("takeLowestClusterNumPix:" + str(takeLowestClusterNumPix) + "\n")
        logger.write("thresholdkmeanlow:" + str(thresholdkmeanlow) + "\n")
        logger.write("thresholdkmeanhigh:" + str(thresholdkmeanhigh) + "\n")
        numlettertouse = 2
        valueCritic = NewCritic(bottleneck=32, chfak=1, dropout=0.3)
        valueCritic.load_state_dict(T.load("NeededFiles/valueCritic30.pt", map_location=device))
        valueCritic = valueCritic.to(device)
        for p in valueCritic.parameters():
            p.requires_grad = False
        valueCritic.eval()
        loader = self.train_loader
        imgsforkmean = []
        critic = self.critic
        critic = critic.to(device)

        controlImg = None
        for c in range(amountCluster):
            if c == 0:

                controlImg = np.ones((3, 10)) * c
            else:
                temparr = np.ones((3, 10)) * c
                controlImg = np.append(controlImg, temparr, axis=0)
        plt.imsave(fname="./controlimg.png", arr=controlImg, format='png', vmin=0, vmax=amountCluster, cmap='hsv')

        for p in critic.parameters():
            p.requires_grad = False
        critic.eval()
        transform = pth_transforms.Compose(
            [

                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        invTransform = pth_transforms.Compose(
            [pth_transforms.Normalize(mean=[0., 0., 0.],
                                      std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
             pth_transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                      std=[1., 1., 1.]),
             ]
        )
        resizesize = 512
        resizefactor = int(512 / 16)
        resizer480 = pth_transforms.Compose(
            [

                pth_transforms.Resize(resizesize),
            ]
        )
        resizer64 = pth_transforms.Compose(
            [

                pth_transforms.Resize(64),
            ]
        )

        # kmean predictor
        qpredictorlist = []
        kpredictorlist = []
        vpredictorlist = []

        # counts appearance of pixel
        kmeancountq = []
        kmeancountk = []
        kmeancountv = []
        for b_idx, (X, Y, I) in enumerate(loader):
            # Y = Y[:, args.rewidx].float().to(device)

            XPnorm = X.permute(0, 3, 1, 2).float().to(device) / 255.0
            Y = valueCritic(XPnorm)
            Y2 = (Y.cpu().numpy() > thresholdkmean).astype(np.bool)
            Y2high = (Y.cpu().numpy() > thresholdkmeanhigh).astype(np.bool)
            Y2low = (Y.cpu().numpy() < thresholdkmeanlow).astype(np.bool)
            # Y2img = (Y.cpu().numpy() > thresholdimg).astype(np.bool)
            XP = resizer480(XPnorm)
            XP = transform(XP)  # dino norm
            if (len(imgsforkmean)) < numKmeanimgs:
                for i in range(len(Y2)):
                    if (Y2[i] == True):
                        imgsforkmean.append(XP[i])
                if (len(imgsforkmean)) >= numKmeanimgs:
                    print("Getting kqv values: \n")
                    qvalues = []
                    kvalues = []
                    vvalues = []
                    for idx, z in enumerate(tqdm(imgsforkmean)):
                        kqvvaluelist = critic.init_kmean_predictor(z.to(device).unsqueeze(0))
                        for i in range(len(kqvvaluelist)):
                            if idx == 0:
                                qvalues.append(kqvvaluelist[i]["qvalue"].cpu().numpy())
                                kvalues.append(kqvvaluelist[i]["kvalue"].cpu().numpy())
                                vvalues.append(kqvvaluelist[i]["vvalue"].cpu().numpy())
                            else:
                                qvalues[i] = np.concatenate((qvalues[i], kqvvaluelist[i]["qvalue"].cpu().numpy()), 0)
                                kvalues[i] = np.concatenate((kvalues[i], kqvvaluelist[i]["kvalue"].cpu().numpy()), 0)
                                vvalues[i] = np.concatenate((vvalues[i], kqvvaluelist[i]["vvalue"].cpu().numpy()), 0)
                                # qvalues[i] = T.cat((qvalues[i], kqvvaluelist[i]["qvalue"]), 0)
                                # kvalues[i] = T.cat((kvalues[i], kqvvaluelist[i]["kvalue"]), 0)
                                # vvalues[i] = T.cat((vvalues[i], kqvvaluelist[i]["vvalue"]), 0)
                        del kqvvaluelist
                    print("\n Generate Kmean \n ")
                    for z in layerToUse:  # tqdm(range(len(qvalues))):
                        print("\n Layer " + str(z) + "\n")
                        qpredictor, kpredictor, vpredictor = critic.getKMeansPredictor(qvalues[z], kvalues[z],
                                                                                       vvalues[z], amountCluster)
                        qpredictorlist.append(qpredictor)
                        kpredictorlist.append(kpredictor)
                        vpredictorlist.append(vpredictor)


            else:
                finallist = []
                finallist.append(qpredictorlist)
                finallist.append(kpredictorlist)
                finallist.append(vpredictorlist)
                with open("kmeanFirstIteration", "wb") as fp:
                    pickle.dump(finallist, fp)
                testimgs = []

                for l in range(XP.shape[0]):
                    testimgs.append(XP[l])

                for idx, img in enumerate(testimgs):
                    kqvvaluelist = critic.init_kmean_predictor(img.to(device).unsqueeze(0))
                    isHigh = "Med"
                    if Y2high[idx]:
                        isHigh = "High"
                    if Y2low[idx]:
                        isHigh = "Low"
                    if isHigh == "High" or isHigh == "Low":
                        torchvision.utils.save_image(torchvision.utils.make_grid(resizer480(XPnorm[idx])),
                                                     os.path.join(OutputDir,
                                                                  "img" + str(counter) + ".jpg"))  # + str(isHigh)
                        pickedImages.append(X[idx])

                        for posid, i in enumerate(layerToUse):  # range(len(kqvvaluelist)):

                            for f in range(numlettertouse):
                                qvalue = kqvvaluelist[i]["qvalue"]
                                predictor = qpredictorlist
                                letter = "q"
                                if f == 1:
                                    qvalue = kqvvaluelist[i]["kvalue"]
                                    predictor = kpredictorlist
                                    letter = "k"
                                if f == 2:
                                    qvalue = kqvvaluelist[i]["vvalue"]
                                    predictor = vpredictorlist
                                    letter = "v"
                                reshapesize = int(math.sqrt(qvalue.size(2) - 1))
                                kronsize = 16
                                if (reshapesize == 8):
                                    kronsize = 60
                                if (reshapesize == 15):
                                    kronsize = 32

                                for j in range(6):
                                    attention2 = qvalue[0, j, 1:, :]
                                    kmeanspred = predictor[posid][j].predict(attention2.cpu().numpy())  # 0 -> i
                                    test = collections.Counter(kmeanspred)
                                    if (counter == 0):
                                        if f == 0:
                                            kmeancountq.append([[test, isHigh]])
                                        if f == 1:
                                            kmeancountk.append([[test, isHigh]])
                                        if f == 2:
                                            kmeancountv.append([[test, isHigh]])
                                    else:
                                        if f == 0:
                                            kmeancountq[j].append([test, isHigh])
                                        if f == 1:
                                            kmeancountk[j].append([test, isHigh])
                                        if f == 2:
                                            kmeancountv[j].append([test, isHigh])
                                    kmeanspred = kmeanspred.reshape(reshapesize, reshapesize)

                                    kmeanspred = np.kron(kmeanspred, np.ones((kronsize, kronsize)))
                                    if kmeanvisfirst:
                                        pickedImagesKMeanVis[f].append([kmeanspred])
                                    else:
                                        pickedImagesKMeanVis[f][j].append(kmeanspred)
                                    # kmeanspredtensor = torch.tensor(kmeanspred)
                                    # kmeanspred = nn.functional.interpolate(kmeanspredtensor, scale_factor=16, mode="nearest").cpu().numpy()
                                    fname = os.path.join(OutputDir, "img" + str(counter) + letter + "depth" + str(i)
                                                         + "head" + str(j) + ".jpg")
                                    plt.imsave(fname=fname, arr=kmeanspred, format='jpg', vmin=0, vmax=amountCluster,
                                               cmap='hsv')
                            kmeanvisfirst = False
                        counter += 1
                        if(counter % 20 == 0):
                            print("LOADED IMGS" + str(counter))

                    if (not self.args.oneep):
                        if counter == imgsToPrint:
                            finalresults = []

                            qkvhighvalues = []
                            qkvlowvalues = []
                            highvalueindex = []
                            lowvalueindex = []
                            for i in range(counter):
                                if (kmeancountq[0][i][1] == "High"):
                                    highvalueindex.append(i)
                                elif (kmeancountq[0][i][1] == "Low"):
                                    lowvalueindex.append(i)
                                else:
                                    print("Not expected String")
                            if (usehighimgonmly):
                                temp2 = highvalueindex
                                lowvalueindex = temp2[highvalimgtouse:]
                                highvalueindex = temp2[:highvalimgtouse]
                            else:

                                highvalueindex = highvalueindex[:highvalimgtouse]  ###############TEST
                            for ind in tqdm(highvalueindex):
                                pickedlowimgs = random.sample(lowvalueindex, 10)
                                for i in tqdm(range(numlettertouse)):
                                    qkvletter = "q"
                                    if (i == 1):
                                        qkvletter = "k"
                                    if (i == 2):
                                        qkvletter = "v"
                                    kmeancluster = pickedImagesKMeanVis[i]

                                    for headnum, kmeanhead in enumerate(kmeancluster):
                                        highvaluecluster = kmeanhead[ind]

                                        lowvalueclusterlist = []
                                        pickedArrList = []
                                        for lowind in pickedlowimgs:
                                            lowvalueclusterlist.append([kmeanhead[lowind], lowind])

                                        for cluster in range(amountCluster):
                                            pickedArr = None
                                            pickedArrCount = 1000000
                                            pickedArrNumPixel = -1
                                            newArrHighVal = highvaluecluster.copy()

                                            newArrHighVal[newArrHighVal != cluster] = -1
                                            newArrHighVal[newArrHighVal == cluster] = 1
                                            newArrHighVal[newArrHighVal == -1] = 0
                                            # for x in range(480):
                                            #     for y in range(480):
                                            #         if highvaluecluster[x, y] == cluster:
                                            #             newArrHighVal[x, y] = 1

                                            for lowcluster in lowvalueclusterlist:
                                                newArr = lowcluster[0].copy()
                                                newArrCount = 0
                                                count_lowcluster = 0
                                                for y in range(resizesize):
                                                    for x in range(resizesize):
                                                        if lowcluster[0][x, y] == cluster:
                                                            count_lowcluster += 1
                                                            if highvaluecluster[x, y] == cluster:
                                                                newArrCount += 1

                                                pickDeciderIfEqual = False
                                                if(newArrCount == pickedArrCount):
                                                    if(pickedArrNumPixel == -1):
                                                        pickDeciderIfEqual = True
                                                        print("irgendwas ist falsch")
                                                        exit()
                                                    else:
                                                        if(takeLowestClusterNumPix):
                                                            if(count_lowcluster < pickedArrNumPixel):
                                                                pickDeciderIfEqual = True
                                                        else:
                                                            if(count_lowcluster > pickedArrNumPixel):
                                                                pickDeciderIfEqual = True

                                                if (newArrCount < pickedArrCount) or pickDeciderIfEqual:
                                                    # for x in range(480):
                                                    #     for y in range(480):
                                                    #         if lowcluster[0][x, y] == cluster:
                                                    #             newArr[x, y] = 1

                                                    newArr[newArr != cluster] = -1
                                                    newArr[newArr == cluster] = 1
                                                    newArr[newArr == -1] = 0

                                                    pickedArr = [newArr, lowcluster[1], newArrHighVal]
                                                    pickedArrCount = newArrCount
                                                    pickedArrNumPixel = count_lowcluster
                                            pickedArrList.append(pickedArr)
                                            del pickedArr

                                        beforeimg = pickedImages[ind].unsqueeze(0)
                                        beforeimg2 = beforeimg.permute(0, 3, 1, 2).float().to(device) / 255.0
                                        beforeimgval = valueCritic(beforeimg2)

                                        beforeimg2 = resizer480(beforeimg2)
                                        imgtest = resizer64(beforeimg2)
                                        # testval = valueCritic(imgtest)
                                        for cluster, pickedArr in enumerate(pickedArrList):
                                            highimg = beforeimg2.clone().detach()
                                            beforelowimg = pickedImages[pickedArr[1]].unsqueeze(0)

                                            beforelowimg2 = beforelowimg.permute(0, 3, 1, 2).float().to(device) / 255.0
                                            # beforelowimgval = valueCritic(beforelowimg2)

                                            beforelowimg2 = resizer480(beforelowimg2)

                                            for y in range(resizefactor):
                                                y2 = y * 16
                                                for x in range(resizefactor):
                                                    x2 = x * 16
                                                    if pickedArr[2][x2, y2] == 1:
                                                        highimg[:, :, x2:(x2 + 16), y2:(y2 + 16)] \
                                                            = beforelowimg2[:, :, x2:(x2 + 16), y2:(y2 + 16)]
                                            # if (pickedArr[0].max() == 1):
                                            #     savedPixelLowValue = []
                                            #     for y in range(30):
                                            #         y2 = y * 16
                                            #         for x in range(30):
                                            #             x2 = x * 16
                                            #             if pickedArr[0][x2, y2] == 1:
                                            #                 savedPixelLowValue.append(
                                            #                     beforelowimg2[:, :, x2:(x2 + 16), y2:(y2 + 16)])
                                            #     numPixel = len(savedPixelLowValue)
                                            #     counterPixel = 0
                                            #     for y in range(30):
                                            #         y2 = y * 16
                                            #         for x in range(30):
                                            #             x2 = x * 16
                                            #             if pickedArr[2][x2, y2] == 1:
                                            #                 highimg[:, :, x2:(x2 + 16), y2:(y2 + 16)] = \
                                            #                     savedPixelLowValue[counterPixel]
                                            #                 counterPixel += 1
                                            #                 if (counterPixel == numPixel):
                                            #                     counterPixel = 0

                                            afterimg = resizer64(highimg)
                                            # torchvision.utils.save_image(torchvision.utils.make_grid(
                                            #     afterimg, ),
                                            #     os.path.join(OutputDir,
                                            #                  str(ind) + "cluster" +
                                            #                  str(cluster) + "img64" + ".png"))  # + str(isHigh)
                                            afterimgval = valueCritic(afterimg)
                                            deltaval = abs(beforeimgval - afterimgval)
                                            deltaval2 = round((beforeimgval - afterimgval).item(), 2)
                                            torchvision.utils.save_image(torchvision.utils.make_grid(
                                                highimg),
                                                os.path.join(OutputDir,
                                                             str(ind) + "cluster" +
                                                             str(cluster) + qkvletter + "head" +
                                                             str(headnum) + "lowind" + str(pickedArr[1]) + "delta: "
                                                             + str(deltaval2) + ".jpg"))  # + str(isHigh)

                                            # print("delta value:" + str(deltaval))
                                            savedcluster = {"delta": deltaval, "highvalimgind": ind,
                                                            "qkv": qkvletter, "head": headnum, "cluster": pickedArr[0],
                                                            "lowvalimgind": pickedArr[1], "clusternum": str(cluster)}
                                            # plt.imsave(fname="./test123.png", arr=pickedArr[0] ,format='jpg')
                                            finalresults.append(savedcluster)
                                            del highimg
                            sortedResults = sorted(finalresults, key=itemgetter("delta"), reverse=True)[0:10]

                            imglen = int(len(finalresults) / (30 * numlettertouse))
                            # mean calc
                            deltafinal = []
                            for z in range(30 * numlettertouse):
                                deltatemp = []
                                for imgn in range(imglen):
                                    a = (imgn * (30 * numlettertouse)) + z
                                    deltafi = finalresults[a].get("delta")
                                    deltatemp.append(deltafi.item())
                                deltamean = statistics.mean(deltatemp)
                                deltastd = statistics.pstdev(deltatemp)
                                deltafinal.append([deltamean, deltastd])
                            deltafinal2 = []
                            for id3, k in enumerate(deltafinal):
                                deltafinal2.append([id3, k[0], k[1]])
                            deltafinal2 = sorted(deltafinal2, key=lambda x: x[1], reverse=True)
                            calcletter = []
                            calchead = []
                            calccluster = []
                            for i in range(10):
                                stringBez, valuesBez = self.numToBez(deltafinal2[i][0], numlettertouse,
                                                                     amountCluster)
                                stringDelta = "DeltaMean: " + str(deltafinal2[i][1]) + " DeltaSTD: " + str(
                                    deltafinal2[i][2]) \
                                              + " Cell: " + stringBez
                                print(stringDelta)
                                calcletter.append(valuesBez[0])
                                calchead.append(valuesBez[1])
                                calccluster.append(valuesBez[2])
                                logger.write(stringDelta + "\n")
                            logger.write("pickedvidletter = " + str(calcletter) + "\n")
                            logger.write("pickedvidhead = " + str(calchead) + "\n")
                            logger.write("pickedvidcluster = " + str(calccluster) + "\n")

                            os.makedirs("FinalResults", exist_ok=True)
                            for indx, sortedResult in enumerate(sortedResults):
                                fname = os.path.join("FinalResults",
                                                     str(indx) + "Delta" + str(sortedResult.get("delta")) +
                                                     "highind" + str(sortedResult.get("highvalimgind")) +
                                                     "clusternum" + str(sortedResult.get("clusternum")) +
                                                     "letter" + str(sortedResult.get("qkv")) +
                                                     "head" + str(sortedResult.get("head")) +
                                                     "lowind" + str(sortedResult.get("lowvalimgind")) + ".jpg")

                                plt.imsave(fname=fname, arr=sortedResult.get("cluster"), format='jpg')
                            logger.close()
                            exit()

                            # for i in range(3):
                            #     kmeancount = kmeancountq
                            #     if i == 1:
                            #         kmeancount = kmeancountk
                            #     if i == 2:
                            #         kmeancount = kmeancountv
                            #     finalhighvalues = []
                            #     finallowvalues = []
                            #     for kmeanhead in kmeancount:
                            #         lowvalues = {"0" : 0, "1" : 0, "2" : 0, "3" : 0, "4" : 0, }
                            #         highvalues = {"0" : 0, "1" : 0, "2" : 0, "3" : 0, "4" : 0, }
                            #         lowcounter = 0
                            #         highcounter = 0
                            #         for headvalue in kmeanhead:
                            #             if(headvalue[1] == "Low"):
                            #                 kmkeys = headvalue[0].keys()
                            #                 for kmkey in kmkeys:
                            #                     val = headvalue[0].get(kmkey)
                            #                     lowvalues[str(kmkey)] += val
                            #                 lowcounter += 1
                            #             else:
                            #                 kmkeys = headvalue[0].keys()
                            #                 for kmkey in kmkeys:
                            #                     val = headvalue[0].get(kmkey)
                            #                     highvalues[str(kmkey)] += val
                            #                 highcounter += 1
                            #         for num in range(len(lowvalues)):
                            #             lowvalues[str(num)] /= lowcounter
                            #             highvalues[str(num)] /= highcounter
                            #         finallowvalues.append(lowvalues)
                            #         finalhighvalues.append(highvalues)
                            #     qkvhighvalues.append(finalhighvalues)
                            #     qkvlowvalues.append(finallowvalues)
                            #
                            # print("test")
                            #
                            #
                            #
                            #
                            #
                            # self.createVis(counter, layerToUse)
                            # exit()

                    print("batch loaded")

    def numToBez(self, beznum, numlettertouse2, numCluster):

        for x in (["q", "k", "v"][:numlettertouse2]):
            for y in range(6):
                for z in range(numCluster):
                    x2 = 0
                    if (x == "k"):
                        x2 = 1
                    if (x == "v"):
                        x2 = 2
                    tempvalue = z + (y * 5) + (x2 * 30)
                    if (tempvalue == beznum):
                        stringtoprint = x + "head" + str(y) + "cluster" + str(z)
                        bezvalues = [x2, y, z]

                        return stringtoprint, bezvalues

    def createVis(self, numimgs, layer):
        inp = "./OutputDir"
        os.makedirs("kmeanimgs", exist_ok=True)
        rltrenner = Image.open("NeededFiles/rltrenner.png")
        rltrenner = rltrenner.convert("RGB").crop((0, 0, 20, 128))

        yseperator = 2
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
    parser.add_argument("-kmeanvideo", action="store_true")

    parser.add_argument("-critic", type=bool, default=True)
    parser.add_argument("-cload", type=bool, default=True)
    parser.add_argument("-dinonorm", default=False, action="store_true",
                        help="resizes the dataset and applies dino "
                             "normalization")
    parser.add_argument("-visbar", default=False, action="store_true")
    parser.add_argument("-freeze", type=bool, default=False)

    parser.add_argument("--usebins", default=False, action="store_true")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=(128, 128), type=int, nargs="+", help="Resize image.")

    parser.add_argument('--pretrain', default='NeededFiles/dino_deitsmall8_pretrain.pth', type=str)
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
    if args.kmean:
        if (args.oneep or args.kmeanvideo):
            H.load_data(batch_size=1)
        else:
            H.load_data(batch_size=64)
    if args.kmean:
        if(args.predonly):
            H.createPredictor()
        elif (args.oneep):
            H.kmean_pipe()
        elif (args.kmeanvideo):
            H.kmean_video()
        else:
            H.kmean_pipe_old()


if __name__ == "__main__":
    main()

