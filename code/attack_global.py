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

        # visualize
        si(torch.cat(l_sample), "l_sample.png", to_01=1)
        si(torch.cat(l_predxstart), "l_pxstart.png", to_01=1)

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


# def generate_x_adv_denoised(x, y, diffusion, model, classifier, pgd_conf, device, t):

#     net = Denoised_Classifier(diffusion, model, classifier, t)

#     adversary = LinfPGDAttack(
#         net,
#         loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
#         eps=pgd_conf["eps"],
#         nb_iter=pgd_conf["iter"],
#         eps_iter=pgd_conf["alpha"],
#         rand_init=True,
#         targeted=False,
#     )

#     x_adv = adversary.perturb(x, y)

#     return x_adv


@torch.no_grad()
def generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t):
    # x shape 224x224x3
    net = Denoised_Classifier(diffusion, model, classifier, t)

    delta = torch.zeros(x.shape).to(x.device)
    print("delta shape", delta.shape)
    print("=============")
    # delta.requires_grad_()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    iter = pgd_conf["iter"]

    for pgd_iter_id in range(iter):

        x_diff = net.sdedit(x + delta, t).detach()

        x_diff.requires_grad_()

        with torch.enable_grad():

            loss = loss_fn(classifier(x_diff), y)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)
    print("Done")

    x_adv = torch.clamp(x + delta, 0, 1)
    print("x_adv shape", x_adv.shape)
    print("=============")
    return x_adv.detach()


@torch.no_grad()
def generate_x_adv_denoised_v3_mirror_descent(
    x, y, diffusion, model, classifier, pgd_conf, device, t
):

    net = Denoised_Classifier(diffusion, model, classifier, t)

    delta = torch.zeros(x.shape).to(x.device)
    delta.requires_grad_()  # 确保delta可以计算梯度

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    iter = pgd_conf["iter"]

    for pgd_iter_id in range(iter):

        # 计算sdedit后的净化图像，并分离计算图以防止梯度流回扩散模型本身
        # 这一步与原Diff-PGD逻辑保持一致，因为它利用了扩散模型进行图像净化
        x_diff = net.sdedit(x + delta, t).detach()

        # 确保x_diff的梯度可以被计算，以便计算对抗性损失的梯度
        x_diff.requires_grad_()

        with torch.enable_grad():
            # 计算对抗性损失，目标是最大化分类器的损失
            loss = loss_fn(classifier(x_diff), y)

            # 反向传播计算损失关于净化图像x_diff的梯度
            loss.backward()

            # 获取梯度的符号（或直接梯度，取决于PGD变体）
            # 对于标准PGD通常使用梯度的符号方向
            grad_direction = x_diff.grad.data.sign()

        # 梯度更新步骤
        # PGD的更新是 x_t+1 = Project(x_t + alpha * sign(grad))
        # 在这里，delta是扰动，所以是 delta = Project(delta + alpha * sign(grad))
        delta.data += grad_direction * alpha

        # **修改为基于ℓ2范数的投影：**
        # PGD使用ℓ∞投影，而典型的镜像梯度下降（当Bregman散度为平方欧几里得距离时）
        # 则对应于ℓ2投影。
        # 这里对整个扰动张量delta应用ℓ2范数限制。
        # 如果delta的ℓ2范数超过eps，则将其缩放到eps
        delta_norm = torch.norm(delta.data, p=2)
        if delta_norm > eps:
            delta.data = delta.data * (eps / delta_norm)

        # 确保扰动后的图像像素值在有效范围内 [7]
        # 这一步在PGD和Mirror Descent中都是常见的，以保持图像的有效性
        x_adv_current = torch.clamp(x + delta, 0, 1)
        # 更新delta以确保其始终是x_adv_current - x
        # 这一步是关键，以确保delta始终代表从原始图像x到当前对抗样本x_adv_current的扰动
        # 且满足像素值范围[7]
        delta.data = x_adv_current - x

    print("Done")

    # 返回最终的对抗样本
    return x_adv_current.detach()


@torch.no_grad()
def generate_x_adv_denoised_v4(
    x,
    y,
    diffusion,
    model,
    classifier,
    pgd_conf,
    device,
    t,
    use_full_grad: bool = False,  # 新参数：是否使用完整梯度而非符号梯度
    momentum_beta: float = 0.0,  # 新参数：动量因子 (0.0 表示不使用动量)
    alpha_decay_factor: float = 1.0,
):  # 新参数：学习率衰减因子 (1.0 表示不衰减)

    net = Denoised_Classifier(diffusion, model, classifier, t)

    delta = torch.zeros(x.shape).to(x.device)
    delta.requires_grad_()  # 确保 delta 可以被更新并跟踪操作（尽管此处梯度不回传至此）

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf["eps"]
    alpha = pgd_conf["alpha"]
    iter = pgd_conf["iter"]

    # 初始化动量缓冲区
    momentum_buffer = torch.zeros_like(delta.data)

    for pgd_iter_id in range(iter):
        # 计算当前迭代的步长 (应用衰减)
        current_alpha = alpha * (alpha_decay_factor**pgd_iter_id)

        # 核心：执行 SDEdit。这里的 .detach() 是实现“加速版 Diff-PGD v2”的关键。
        # 它确保梯度只在 x_diff 及其下游（分类器）计算，而不回传通过 SDEdit 到 x + delta。
        x_diff = net.sdedit(x + delta, t).detach()
        x_diff.requires_grad_()  # 启用对 x_diff 的梯度计算

        with torch.enable_grad():
            loss = loss_fn(classifier(x_diff), y)
            loss.backward()

            # 获取损失对 x_diff 的梯度 (这对应于论文中的 ∇_x0_t loss)
            grad = x_diff.grad.data

            # 根据参数选择使用完整梯度或符号梯度
            if use_full_grad:
                update_direction = grad
            else:
                update_direction = grad.sign()

        # 应用动量（如果启用）
        if momentum_beta > 0:
            # 动量更新公式：v = beta * v + (1 - beta) * g
            momentum_buffer = (
                momentum_beta * momentum_buffer + (1 - momentum_beta) * update_direction
            )
            update_term = momentum_buffer
        else:
            update_term = update_direction

        # 更新扰动 delta
        # 使用 .data 进行就地更新，避免创建新的计算图节点
        delta.data += update_term * current_alpha

        # 将 delta 投影回 L_inf 范数球
        delta.data = torch.clamp(delta.data, -eps, eps)

    print("Done")

    # 生成最终的对抗样本，并裁剪到 [35] 范围
    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()  # 返回时也进行 detach，确保没有不必要的梯度信息


def Attack_Global(
    classifier,
    device,
    model_path,
    respace,
    t,
    eps=16,
    iter=10,
    name="attack_global",
    alpha=2,
    version="v1",
    skip=200,
):

    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))

    save_path = f"vis/{name}_{version}/{classifier}_eps{eps}_iter{iter}_{respace}_t{t}/"

    mp(save_path)

    classifier = get_archs(classifier, "imagenet")

    classifier = classifier.to(device)
    classifier.eval()

    dataset = get_dataset("imagenet", split="test")

    model, diffusion = get_imagenet_dm_conf(
        device=device, respace=respace, model_path=model_path
    )

    c = 0

    # for i in tqdm(range(dataset.__len__())):
    for i in range(1000):
        if i % skip != 0:
            continue
        time_st = time.time()
        print(f"{c}/{dataset.__len__()//skip}")

        x, y = dataset[i]
        x = x[None,].to(device)
        y = torch.tensor(y)[None,].to(device)
        print("first =============")
        print("x shape", x.shape)
        print("=============")
        y_pred = classifier(x).argmax(1)  # original prediction

        if version == "v1":
            print("v1 not implemented")
            # x_adv = generate_x_adv_denoised(
            #     x, y_pred, diffusion, model, classifier, pgd_conf, device, t
            # )
        elif version == "v2":
            x_adv = generate_x_adv_denoised_v2(
                x, y_pred, diffusion, model, classifier, pgd_conf, device, t
            )
        elif version == "v3":
            x_adv = generate_x_adv_denoised_v3_mirror_descent(
                x, y_pred, diffusion, model, classifier, pgd_conf, device, t
            )
        elif version == "v4":
            x_adv = generate_x_adv_denoised_v4(
                x,
                y_pred,
                diffusion,
                model,
                classifier,
                pgd_conf,
                device,
                t,
                True,
                0.9,
                0.9,
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

        print("x_adv: ", x_adv.min(), x_adv.max(), (x - x_adv).abs().max())

        torch.save(pkg, save_path + f"{i}.bin")
        si(torch.cat([x, x_adv, pred_x0], -1), save_path + f"{i}.png")
        print(
            "y_pred, x_adv, pred_x0: ",
            y_pred,
            classifier(x_adv).argmax(1),
            classifier(pred_x0).argmax(1),
        )

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
    "/kaggle/working/models/256x256_diffusion_uncond.pt",
    "ddim50",
    t=3,
    eps=16,
    iter=20,
    name="attack_global_gradpass",
    alpha=2,
    version="v4",
)
# Attack_Global('resnet50', 0, 'ddim50', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=2)

# Attack_Global('resnet50', 0, 'ddim40', t=3, eps=16, iter=10, name='attack_global_gradpass', alpha=4)


# Attack_Global('resnet50', 5, 'ddim10', t=2, eps=32, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim40', t=2, eps=16, iter=10, name='attack_global_new2')
# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new2')

# Attack_Global('resnet50', 1, 'ddim30', t=2, eps=16, iter=10, name='attack_global_new')
# Attack_Global('resnet50', 1, 'ddim20', t=2, eps=16, iter=10, name='attack_global_new')
