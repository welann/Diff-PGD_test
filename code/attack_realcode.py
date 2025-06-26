from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL

# from advertorch.attacks import LinfPGDAttack
import matplotlib.pylab as plt
import time
import glob

from attack_tools import gen_pgd_confs


class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t

    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t

        x = x * 2 - 1

        t = torch.full((x.shape[0],), t).long().to(x.device)
        print("sdedit x_t shape", x.shape)
        print("t shape", t.shape)
        print("=============")
        # 把正常图像加噪声，得到x_t
        x_t = self.diffusion.q_sample(x, t)

        sample = x_t
        print("sdedit sample shape", sample.shape)
        print("=============")
        # print(x_t.min(), x_t.max())

        # si(x_t, 'vis/noised_x.png', to_01=True)

        indices = list(range(t + 1))[::-1]

        # visualize
        l_sample = []
        l_predxstart = []

        for i in indices:

            # out = self.diffusion.ddim_sample(self.model, sample, t)
            out = self.diffusion.ddim_sample(
                self.model, sample, torch.full((x.shape[0],), i).long().to(x.device)
            )

            sample = out["sample"]

            l_sample.append(out["sample"])
            l_predxstart.append(out["pred_xstart"])

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def forward(self, x):

        out = self.sdedit(x, self.t)  # [0, 1]
        print("out1 shape", out.shape)
        print("=============")
        out = self.classifier(out)
        print("out2 shape", out.shape)
        print("=============")
        return out


class Attack_Realcode(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t

    def attack_method_v1(
        self,
        classifier_model_list=[],
        mask_list=[],
        label_list=[],
        pgd_conf=[],
    ):
        pass

    def forward(self, x):
        x_t = self.diffusion.q_sample(x, self.t)
        x_t = self.model(x_t)
        x_t = self.classifier(x_t)
        return x_t


def Realattack(
    classifier,
    device,
    respace,
    t,
    eps=16,
    iter=10,
    name="attack_global",
    alpha=2,
    version="v1",
):
    model_path = "/kaggle/working/models/256x256_diffusion_uncond.pt"

    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))

    classifier = get_archs(classifier, "imagenet")
    classifier = classifier.to(device)
    classifier.eval()

    # model 用来预测噪声， diffusion用来生成去噪后的图片
    model, diffusion = get_imagenet_dm_conf(
        device=device, respace=respace, model_path=model_path
    )

    net = Denoised_Classifier(diffusion, model, classifier, t)
    # 生成一个三通道224大小的随机变量
    random_input = torch.rand((1, 3, 224, 224)).to(device)
    
    # 输入sdedit并打印形状
    net.sdedit(random_input, t)
    


if __name__ == "__main__":
    Realattack(
        classifier="resnet50",
        device=0,
        respace="ddim50",
        t=3,
    )
