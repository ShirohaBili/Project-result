"""
本代码实现了模型窃取的过程
"""
import argparse
from random import randint

import numpy as np
import torch
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import my_uitls
import subset_select_strategy as sss

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('-cuda', default=False, action="store_true", help='enables cuda')
parser.add_argument('-epochs', type=int, default=5, help='training epochs')
parser.add_argument('-pretrain', default=False, action='store_true', help='use the pretrained model')
parser.add_argument('-load_path', type=str, default=None, help='the pretrained modle')
parser.add_argument('-begin_epoch', type=int, default=1, help='the begin epoch number')
parser.add_argument('-modified', default=False, action="store_true", help='use modified model')
parser.add_argument('struct', type=str, default="ResNet18",
                    help='model structure: VGG11/13/16/19 ResNet18/34/50/101/152')
parser.add_argument('thief_data', type=str, default="classified_thief", help='thief_data source folder')
parser.add_argument('test_data', type=str, default="classified_validate", help='test_data source folder')
parser.add_argument('select_type', type=str, help='the subset selct strategy')
parser.add_argument('save_path', type=str, default="ResNet18", help='the file to save the trained model')

opt = parser.parse_args()
cuda = opt.cuda
epochs = opt.epochs
pretrain = opt.pretrain
load_path = opt.load_path
begin_epoch = opt.begin_epoch
struct = opt.struct
thief_data = opt.thief_data
test_data = opt.test_data
select_type = opt.select_type
save_path = opt.save_path
modified = opt.modified

#利用窃取数据集训练
def train_with_classified_thief(epochs, begin_epoch, pretrain, load_path, save_path, struct, thief_data_name,
                                test_data_name, select_type,modified):
    class_num = -1
    train_data = None
    test_data = None
    transform_train = None
    transform_test = None
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224))
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224))]
    )

    train_data, _, class_num = my_uitls.load_data(thief_data_name, transform_train, transform_test)
    _, test_data, _ = my_uitls.load_data(test_data_name, transform_train, transform_test)
    substitute = my_uitls.create_model(struct, class_num, pretrain, load_path, modified)

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

    substitute = substitute.to(device)
    substitute.requires_grad_()

    train_batch = 256

    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=True, num_workers=5,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=train_batch, shuffle=False, num_workers=5,
                                 drop_last=False)

    record = open('record.txt','a')
    record.writelines(save_path +' '+ struct +' '+ select_type)
    record.writelines('\n')
    record.close()
    
    
    # iterative training
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        print('epoch:' + str(begin_epoch + epoch))

        # print('epoch begin')
        train_iter = 0

        substitute.train()
        times = 0
        total_loss = 0
        # print('begin load')
        for data in train_dataloader:
            thief_img, thief_label = data
            total_loss += choose_thief_img(thief_img, thief_label, class_num, substitute, 10, 5,
                                                 randint(1, 10000),
                                                 device, select_type)
            times += 1
            # print('begin load')
        train_acc = my_uitls.compare(substitute, None, train_data, device)
        test_acc = my_uitls.compare(substitute, None, test_data, device)
        info_str = 'train_loss:%.5f' % (total_loss/times) + '  train_acc:%.5f' % train_acc + '  test_acc:%.5f'%test_acc
        print(info_str)
        
        print('------')
        torch.cuda.empty_cache()
        
        if True or (begin_epoch + epoch) % 10 == 0 or (begin_epoch + epoch) % 50 == 1:
            # torch.save(substitute.cpu().state_dict(), save_path + '-epoch' + str(begin_epoch + epoch))
            # substitute = substitute.to(device)
            
            record = open('record.txt','a')
            record.writelines('epoch:%3d'%(begin_epoch + epoch) + '   ')
            record.writelines(info_str)
            record.writelines('\n')
            record.close()

    substitute = substitute.cpu()
    torch.save(substitute.state_dict(), save_path)
    
    record = open('record.txt','a')
    record.writelines('\n\n\n')
    record.close()

#随机选取图片进行偷取，返回偷取得到的loss值
def choose_thief_img(thief_img, thief_label, class_num, substitute, iter_num, k, seed, device,
                                         select_type):
    # print("test point 1")
    substitute.train()
    substitute.resnet_layer.requires_grad = False

    substitute.to(device)
    target = None
    thief_img = thief_img.to(device)

    data_num = thief_img.shape[0]
    np.random.seed(seed)
    all_idxs = range(data_num)
    idxs = list(np.random.choice(all_idxs, k, replace=False))
    init_seed_idxs = idxs.copy()
    Si_img = thief_img[idxs]

    Si_target_label = thief_label[idxs]
    remain_thief_img = np.delete(thief_img.cpu(), axis=0, obj=idxs)
    remain_thief_label = np.delete(thief_label.cpu(), axis=0, obj=idxs)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, substitute.parameters()), lr=1e-5, weight_decay=1e-2)

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    # print("test point 2")
    for i in range(iter_num):
        # print("test point 2-1")
        Si_img = Si_img.to(device)
        rows = np.arange(k).reshape(-1, 1)
        # with torch.no_grad():
        #     target_label = Si_target_label
        # Si_label_grid = np.zeros((k, class_num))
        # Si_label_grid[rows, target_label.cpu()] = 1
        # Si_label_grid = torch.tensor(Si_label_grid)
        # Si_label_grid = Si_label_grid.to(device)

        Si_target_label = Si_target_label.to(device)
        # training substitute
        optimizer.zero_grad()
        output = substitute.forward(Si_img)
        # soft = torch.softmax(output,axis=0)
        loss = loss_fn(output, Si_target_label)
        total_loss += loss
        loss.backward()
        optimizer.step()
        # print("test point 2-2")
        # querying substitute
        with torch.no_grad():
            Di_soft = torch.softmax(substitute.forward(remain_thief_img.to(device)), axis=1)
        idxs = sss.select_sub(remain_thief_img, substitute, Di_soft, k, select_type, thief_img[init_seed_idxs], device)

        # print("test point 2-3")
        Si_img = remain_thief_img[idxs]
        Si_target_label = remain_thief_label[idxs]

        remain_thief_img = np.delete(remain_thief_img, axis=0, obj=idxs)
        remain_thief_label = np.delete(remain_thief_label, axis=0, obj=idxs)
        torch.cuda.empty_cache()
    # print("test point 3")
    torch.cuda.empty_cache()
    return total_loss/(iter_num*k)


if __name__ == '__main__':
    train_with_classified_thief(epochs, begin_epoch, pretrain, load_path, save_path, struct, thief_data, test_data,
                                select_type,modified)
