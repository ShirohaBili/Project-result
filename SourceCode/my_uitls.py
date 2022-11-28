"""
    本代码实现了模型创建等基本操作
    我们在这里加入了很多模型的架构预设，方便选择，但并不是每一种都用上了
    同时，我们加入了Cifar-10等数据集作为前期的测试数据集，限于篇幅，这一部分并未在报告中提及。
"""

import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import copy
import torch
from torchvision.models import vgg, resnet


class Adaptive_Net(nn.Module):
    def __init__(self, model,class_num):
        super(Adaptive_Net, self).__init__()
        
        self.resnet_layer = copy.deepcopy(model)
        self.final_Linear_layer_1 = nn.Sequential(nn.ReLU(),nn.Linear(1000, class_num))
        

    def forward(self, x):
        # print(x.shape)
        x = self.resnet_layer(x)
        
        x = self.final_Linear_layer_1(x)
        # print(x.shape)
        return x

#选择模型的架构
def create_model(struct,class_num,pretrain,load_path,if_modified):
    # print('num_class:')
    # print(class_num)
    net = None
    class_num = int(class_num)
    if if_modified:
        if struct[0:3] == 'VGG':
            kind = struct[3:]
            if kind == '11':
                net = vgg.vgg11_bn(pretrained=False)
            if kind == '13':
                net = vgg.vgg13_bn(pretrained=False)
            if kind == '16':
                net = vgg.vgg16_bn(pretrained=False)
            if kind == '19':
                net = vgg.vgg19_bn(pretrained=False)
            # net = vgg.VGG(struct, class_num,sizes)
        if struct[0:6] == 'ResNet':
            kind = struct[6:]
            if kind == '18':
                net = resnet.resnet18(pretrained=False)
            if kind == '34':
                net = resnet.resnet34(pretrained=False)
            if kind == '50':
                net = resnet.resnet50(pretrained=False)
            if kind == '101':
                net = resnet.resnet101(pretrained=False)
            if kind == '152':
                net = resnet.resnet152(pretrained=False)
        net = Adaptive_Net(net,class_num)
        pass
    else:
        if struct[0:3] == 'VGG':
            kind = struct[3:]
            if kind == '11':
                net = vgg.vgg11_bn(pretrained=False,num_classes=class_num)
            if kind == '13':
                net = vgg.vgg13_bn(pretrained=False,num_classes=class_num)
            if kind == '16':
                net = vgg.vgg16_bn(pretrained=False,num_classes=class_num)
            if kind == '19':
                net = vgg.vgg19_bn(pretrained=False,num_classes=class_num)
            # net = vgg.VGG(struct, class_num,sizes)
        if struct[0:6] == 'ResNet':
            kind = struct[6:]
            if kind == '18':
                net = resnet.resnet18(pretrained=False,num_classes=class_num)
            if kind == '34':
                net = resnet.resnet34(pretrained=False,num_classes=class_num)
            if kind == '50':
                net = resnet.resnet50(pretrained=False,num_classes=class_num)
            if kind == '101':
                net = resnet.resnet101(pretrained=False,num_classes=class_num)
            if kind == '152':
                net = resnet.resnet152(pretrained=False,num_classes=class_num)
    if (pretrain and load_path != None):
        print(load_path)
        # print(net)
        net.load_state_dict(torch.load(load_path))
    
    return net


#导入数据集
def load_data(data_name,transform_train,transform_test):
    print(data_name)
    train_data, test_data, class_num = None, None, None
    if data_name=='CIFAR-100':
        train_data = torchvision.datasets.CIFAR100(root=data_name, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(root=data_name, train=False, download=True, transform=transform_test)
        class_num = 100
    if data_name=='CIFAR-10':
        train_data = torchvision.datasets.CIFAR10(root=data_name,train=True,download=True,transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(root=data_name, train=False, download=True, transform=transform_test)
        class_num = 10
    if data_name=='MNIST':
        train_data = torchvision.datasets.MNIST(root=data_name, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.MNIST(root=data_name, train=False, download=True, transform=transform_test)
        class_num = 10
    if data_name=='EMNIST':
        split = input("emnist usage:")
        train_data = torchvision.datasets.EMNIST(root=data_name, split=split, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.EMNIST(root=data_name, split=split, train=False, download=True, transform=transform_test)
        class_num = 10
    if data_name == 'KMNIST':
        train_data = torchvision.datasets.KMNIST(root=data_name, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.KMNIST(root=data_name, train=False, download=True, transform=transform_test)
        class_num = 10
    if data_name=='Tiny-ImageNet':
        pass
    if data_name=='places365':
        train_data = ImageFolder(root='../input/places365/train', transform=transform_train)
        test_data = ImageFolder(root='../input/places365/val', transform=transform_test)
        class_num = -1
    if data_name=='CUB-200':
        train_data = ImageFolder(root='../input/CUB_200_2011/images', transform=transform_train)
        test_data = None
        class_num = -1
    if data_name=='classified_thief':
        train_data = ImageFolder(root='../input/classified_thief', transform=transform_train)
        test_data = None
        class_num = 5
    if data_name=='classified_validate':
        train_data = None
        test_data = ImageFolder(root='../input/classified_validate', transform=transform_test)
        class_num = 5
    
    return train_data,test_data,class_num

#比较替代模型和目标模型的agreement
def compare(substitute, target, compare_data, device):
    if compare_data == None:
        return -1
    with torch.no_grad():
        substitute.eval()
        if target == None:
            loader = DataLoader(dataset=compare_data, batch_size=200, shuffle=False, drop_last=False, num_workers=10)
            total_batch = 0
            total_right = 0
            i = 0
            for data in loader:
                img, label = data
                img = img.to(device)
                label = label.to(device)
                total_batch += img.shape[0]
                sub_out = substitute.forward(img)
                # print(sub_out.shape)
                sub_labels = sub_out.argmax(dim=1)
                # print(label)
            
                i += 1
                total_right += (sub_labels == label).sum()
        else:
            loader = DataLoader(dataset=compare_data, batch_size=200, shuffle=False, drop_last=False, num_workers=10)
            target.eval()
            total_batch = 0
            total_right = 0
            for data in loader:
                img, _ = data
                img = img.to(device)
                total_batch += img.shape[0]
                sub_labels = substitute.forward(img).argmax(dim=1)
                tar_labels = target.forward(img).argmax(dim=1)
                total_right += (sub_labels == tar_labels).sum()
        substitute.train()
        return (total_right / total_batch).item()
    pass





