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



if __name__ == "__main__":
    Realattack(
        classifier="resnet50",
        device=0,
        respace="ddim50",
        t=3,
    )
