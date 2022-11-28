"""
    子集选取策略。
    我们复现了多种子集选取策略，如random、uncertainty、kcenter等
"""
import copy

import heapq
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import vgg, resnet
from torchvision.transforms import transforms

def select_sub(thief_img, target, Di_soft, k, select_type, init_seed_img,device):
    if select_type == 'random':
        return random(thief_img, k)
    if select_type == 'uncertainty':
        return uncertainty(Di_soft, k)
    if select_type == 'kcenter':
        return kcenter(thief_img, k, init_seed_img,device)
    if select_type == 'DFAL':
        return DFAL(thief_img, target, k,device)
    assert False, '子集选取策略错误:' + select_type

#random 子集选取策略
def random(thief_img,k):
    return np.random.choice(range(thief_img.shape[0]), k, replace=False)

#uncertainty 子集选取策略
def uncertainty(Di_label,k):
    t = -(torch.log(Di_label)*Di_label).sum(axis=1)
    t = t.cpu().numpy()
    t = t.flatten()
    t = list(t)
    idxs = heapq.nlargest(k, range(len(t)), t.__getitem__)
    return idxs

#kcenter 子集选取策略
def kcenter(thief_imgs, k, initial_seed_imgs,device):
    ans = []
    imgs_num = thief_imgs.shape[0]

    for i in range(k):
        min_dis = torch.full((imgs_num,1),torch.inf).to(device)
        thief_imgs = thief_imgs.to(device)
        initial_seed_imgs = initial_seed_imgs.to(device)
        for i in range(initial_seed_imgs.shape[0]):
            center_img = initial_seed_imgs[i:i+1].to(device)
            new_dis = torch.sum((thief_imgs - center_img) * (thief_imgs - center_img), dim=(1,2,3)).reshape((imgs_num,1))

            min_dis = torch.minimum(min_dis, new_dis)
        for center_idx in ans:
            center_img = thief_imgs[center_idx:center_idx+1]
            new_dis = torch.sum((thief_imgs - center_img) * (thief_imgs - center_img), dim=(1,2,3)).reshape((imgs_num,1))
            min_dis = torch.minimum(min_dis, new_dis)
        max_min_dis_idx = torch.argmax(min_dis)
        ans.append(max_min_dis_idx.item())
    return ans

#DeepFool系列 子集选取策略
def DFAL(thief_imgs,target,k,device):
    pert = []
    for i in range(thief_imgs.shape[0]):
        pert.append(DeepFool(thief_imgs[i:i+1,:],target,device))
    pert = np.array(pert)
    ans = pert.argsort()[0:k][::-1]
    ans.sort()
    return list(ans)

def DeepFool(thief_image, target, device, max_iter=50):
    target.requires_grad_()

    input_shape = thief_image.cpu().numpy().shape

    pert_image = copy.deepcopy(thief_image)
    pert_image = pert_image.to(device)  # ****
    pert_image = Variable(pert_image, requires_grad=True)


    out = target.forward(pert_image)
    num_class = out.shape[1]
    label = out.argmax(axis=1).reshape(-1, 1)

    r_total = torch.zeros(input_shape).to(device)
    loop_i = 0
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = torch.inf
        out[0, label].backward(retain_graph=True)
        grad_origin = pert_image.grad

        for k in range(num_class):
            if k==label:
                continue
            pert_image.grad.zero_()
            out[0, k].backward(retain_graph=True)

            cur_grad = pert_image.grad
            print(cur_grad.sum())
            # set new w_k and new f_k
            w_k = cur_grad - grad_origin
            f_k = (out[0, k] - out[0, label])

            pert_k = abs(f_k)/(torch.linalg.norm(w_k.flatten())+1e-5)# 怕分母太小，多加了一个1e-5

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k


        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / torch.linalg.norm(w)
        r_total = r_total + r_i
        pert_image = pert_image + r_i

        out = target.forward(pert_image)
        k_i = out.argmax(axis=1)
        loop_i += 1

    return pert.item()

def DeepFool_parallel(thief_images, target, device, max_iter=50):

    target.requires_grad_()

    input_shape = thief_images.cpu().numpy().shape

    pert_images = copy.deepcopy(thief_images)
    pert_images = pert_images.to(device)  # ****
    pert_images = Variable(pert_images, requires_grad=True)


    out = target.forward(pert_images)
    num_class = out.shape[1]
    labels = out.argmax(axis=1).reshape(-1, 1)

    r_total = torch.zeros(input_shape).to(device)
    loop_i = 0
    k_i = labels

    while k_i == labels and loop_i < max_iter:
        pert = torch.full(input.shape[0],torch.inf)
        out[0, labels].backward(retain_graph=True)
        grad_origin = pert_images.grad

        for k in range(num_class):
            pert_images.grad.zero_()
            out[:, k].backward(retain_graph=True)

            cur_grad = pert_images.grad
            print(cur_grad.sum())
            # set new w_k and new f_k
            w_k = cur_grad - grad_origin
            f_k = (out[:, k] - out[:, labels])

            pert_k = abs(f_k)/(torch.linalg.norm(w_k.flatten())+1e-5)# 怕分母太小，多加了一个1e-5

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k


        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / torch.linalg.norm(w)
        r_total = r_total + r_i
        pert_images = pert_images + r_i

        out = target.forward(pert_images)
        k_i = out.argmax(axis=1)
        loop_i += 1

    return pert.item()

if __name__ == "__main__":
    net = vgg.vgg19(num_classes=10)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224)
    ])
    my_data = torchvision.datasets.CIFAR100(root='CIFAR-100', train=True, download=True, transform=t)
    loader = DataLoader(dataset=my_data, batch_size=2, shuffle=True, num_workers=2, drop_last=False)
    net = net.to("cuda")
    for data in loader:
        image,_ = data
        image = image[0:1,:].to("cuda")
        print(image.shape)
        print(DeepFool(image,net))
        break

