from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
)

# from advertorch.attacks import LinfPGDAttack
import matplotlib.pylab as plt
import time
import glob

from attack_tools import gen_pgd_confs

from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms import InterpolationMode


def get_dataset_tinyimagenet(split="valid"):
    ds = load_dataset("zh-plus/tiny-imagenet")
    dataset = ds[split]
    jitter = Compose(
        [
            Resize(224),
            # CenterCrop(224),
            ToTensor(),
        ]
    )

    def transforms(examples):
        examples["modified_image"] = [
            jitter(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    dataset.set_transform(transforms)
    return dataset


def get_mask(image, classifiermodel, threshold=0.5):
    classifier = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier = classifier.to(image.device)
    input_tensor = normalize(
        resize(image, (224, 224)) / 255.0,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    cam_extractor = SmoothGradCAMpp(classifier)
    # Preprocess your data and feed it to the model
    out = classifier(input_tensor)
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    activation_map = activation_map[0]
    activation_map = torch.where(
        activation_map > threshold,
        activation_map,
        torch.tensor(0.0).to(activation_map.device),
    )  # 将小于阈值的权重设为 0

    mask = resize(activation_map, (224, 224), interpolation=InterpolationMode.BICUBIC)

    return mask


def get_classifier(classifier_name):
    if classifier_name == "resnet18":
        return torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif classifier_name == "resnet50":
        return torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif classifier_name == "resnet101":
        return torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Invalid classifier name: {classifier_name}")


class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t

    def attack_it(self, x, t, to_01=True):
        # 1. 读取mask
        # 2. 正常图像加噪声
        # 3. 进行去噪，并攻击
        # 4. 得到攻击后的图像
        # assume the input is 0-1
        # assume the input is 0-1
        mask = get_mask(x, self.classifier, 0.5)

        t_int = t
        x = x * 2 - 1
        t = torch.full((x.shape[0],), t).long().to(x.device)
        # 把正常图像加噪声，得到x_t
        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

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

        # visualize
        si(torch.cat(l_sample), "l_sample.png", to_01=True)
        si(torch.cat(l_predxstart), "l_pxstart.png", to_01=True)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t

        x = x * 2 - 1

        t = torch.full((x.shape[0],), t).long().to(x.device)
        # 把正常图像加噪声，得到x_t
        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

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

        # visualize
        si(torch.cat(l_sample), "l_sample.png", to_01=True)
        si(torch.cat(l_predxstart), "l_pxstart.png", to_01=True)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def forward(self, x):

        out = self.sdedit(x, self.t)  # [0, 1]
        out = self.classifier(out)
        return out


@torch.no_grad()
def generate_x_adv_denoised_v2(
    x, y, mask, diffusion, model, classifier, pgd_conf, device, t
):
    # x shape 224x224x3
    net = Denoised_Classifier(diffusion, model, classifier, t)

    delta = torch.zeros(x.shape).to(x.device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    attack_iter = pgd_conf["iter"]

    x_diff = net.sdedit(x + delta, t).detach()

    for pgd_iter_id in range(attack_iter):

        x_diff = net.sdedit(x + delta, t).detach()

        x_diff.requires_grad_()

        with torch.enable_grad():

            loss = loss_fn(classifier(x_diff), y)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)

    # delta = mask * delta
    x_adv = torch.clamp(x + delta, 0, 1)
    print("Done")
    return x_adv.detach()


def generate_x_adv_denoised_v3(
    x, y, mask, diffusion, model, classifier, pgd_conf, device, t
):
    """
    把x当作待优化的参数，使用优化器进行优化
    """
    # x shape 224x224x3
    net = Denoised_Classifier(diffusion, model, classifier, t)

        # 将 x 设置为可训练的参数
    x_adv = x.clone().detach().requires_grad_(True)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    attack_iter = pgd_conf["iter"]

    # 使用优化器来更新 x_adv
    optimizer = torch.optim.Adam([x_adv], lr=0.01)

    for pgd_iter_id in range(attack_iter):
        optimizer.zero_grad()

        x_diff = net.sdedit(x_adv, t)  # 不再 detach，因为 x_adv 需要梯度

        loss = loss_fn(classifier(x_diff), y)

        loss.backward()

        # 应用梯度
        optimizer.step()

        # 投影步骤：保持扰动在 eps 范围内，并限制像素值在 [0, 1] 范围内
        delta = x_adv - x
        delta = torch.clamp(delta, -eps, eps)
        x_adv.data = torch.clamp(x + delta, 0, 1)  # 使用 data 避免 autograd 记录

    print("Done")
    return x_adv.detach()

def Attack_Global(
    classifier,
    device,
    model_path,
    respace,
    t,
    eps=16,
    attack_iter=10,
    name="attack_global",
    alpha=2,
    version="v1",
    skip=100,
):

    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=attack_iter, input_range=(0, 1))

    save_path = (
        f"vis/{name}_{version}/{classifier}_eps{eps}_iter{attack_iter}_{respace}_t{t}/"
    )

    mp(save_path)

    # classifier = get_archs(classifier, "imagenet")
    classifier = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    classifier = classifier.to(device)
    classifier.eval()

    new_classifier=torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
    new_classifier=new_classifier.to(device)
    new_classifier.eval()
     
    # dataset = get_dataset("imagenet", split="test")
    dataset = get_dataset_tinyimagenet()

    model, diffusion = get_imagenet_dm_conf(
        device=device, respace=respace, model_path=model_path
    )

    c = 0
    total_image = 0
    success_image = 0
    for i in tqdm(range(dataset.__len__())):
        if i % skip != 0:
            continue
        total_image += 1
        print(f"success_image: {success_image/total_image}")

        print(f"{c}/{dataset.__len__()//skip}")

        x, y = dataset[i]["modified_image"], dataset[i]["label"]
        x = x[None,].to(device)
        y = torch.tensor(y)[None,].to(device)
        y_pred = classifier(x).argmax(1)  # original prediction

        x_cla_label=new_classifier(x).argmax(1)
        mask = get_mask(x, new_classifier, 0.5)

        time_st = time.time()

        if version == "v1":
            print("v1 not implemented")
        elif version == "v2":
            x_adv = generate_x_adv_denoised_v2(
                x, y_pred, mask, diffusion, model, classifier, pgd_conf, device, t
            )
        elif version == "v3":
            x_adv = generate_x_adv_denoised_v3(
                x, y_pred, mask, diffusion, model, classifier, pgd_conf, device, t
            )

        cprint("time: {:.3}".format(time.time() - time_st), "g")

        with torch.no_grad():

            net = Denoised_Classifier(diffusion, model, classifier, t)

            pred_x0 = net.sdedit(x_adv, t)

        pkg = {
            "x": x,
            "y": y,
            "x_adv": x_adv,
            "x_adv_diff": pred_x0,
        }

        # print("x_adv: ", x_adv.min(), x_adv.max(), (x - x_adv).abs().max())

        torch.save(pkg, save_path + f"{i}.bin")
        si(torch.cat([x, x_adv, pred_x0], -1), save_path + f"{i}.png")
        print(
            "y_pred, x_cla_label,x_adv, pred_x0: ",
            y_pred,
            x_cla_label,
            new_classifier(x_adv).argmax(1),
            new_classifier(pred_x0).argmax(1),
        )
        if new_classifier(x_adv).argmax(1) != x_cla_label:
            success_image += 1

        c += 1



# Attack_Global('resnet101', 1, 'ddim100', t=2, eps=16, iter=1)

# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=1)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=2)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=5)
# Attack_Global('resnet50', 3, 'ddim100', t=2, eps=16, iter=10)
# Attack_Global('resnet50', 3, 'ddim100', t=3, eps=16, iter=10)


# eps = 8 #####################
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=1, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=2, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=5, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=10, name='attack_global_new2')


# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=1, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=2, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim100', t=2, eps=8, iter=5, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim50', t=2, eps=8, iter=10, name='attack_global_new2')


# eps=32 ######################
Attack_Global(
    "resnet50",
    0,
    "/root/Diff-PGD_test/models/256x256_diffusion_uncond.pt",
    "ddim50",
    t=3,
    eps=16,
    attack_iter=20,
    name="attack_global_gradpass",
    alpha=2,
    version="v2",
)
# Attack_Global('resnet50', 0, 'ddim50', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=2)

# Attack_Global('resnet50', 0, 'ddim40', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=4)


# Attack_Global('resnet50', 5, 'ddim10', t=2, eps=32, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim40', t=2, eps=16, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new2')

# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new')
# Attack_Global('resnet50', 1, 'ddim20', t=2, eps=16, iter=10, name='attack_global_new')
