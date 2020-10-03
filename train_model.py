import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import readMhd, readCsv
from lndbDataset import lcdDataset
from loss_func import BCEDiceLoss
from model import NestedUNet
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score


def count_params(model): #計算網路參數
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(10)

    reader = readCsv('./train_valid_CTs.csv')
    data_list = readCsv('./data_list_train_valid.csv')
    reader = reader[1:]
    data_list = np.array(data_list[1:])

    tr_data = list()
    tr_label = list()
    for number in range(len(reader)): #讀取檔案製作訓練資料集[病人,幾張,3,Y,X]
        [scan,spacing,origin,transfmat] = readMhd('traindata/LNDb-{:04}.mhd'.format(int(reader[number][0])))
        [mask,spacing,origin,transfmat] = readMhd('trainlabel_1c1t/LNDb-{:04}_w_Text_merged-mask.mhd'.format(int(reader[number][0])))
        for z in range(scan.shape[0]):
            tr_data.append(scan[z])
            tr_label.append(mask[z])
    print("loading finished.")

    model = NestedUNet(crop_size=64, have_nd=False).to(device)
    #pre_weight = ""
    #model.load_state_dict(torch.load(pre_weight, map_location=device))
    print("amount of parameters: {}".format(count_params(model)))

    # hyperparameters
    learning_rate = 3e-4
    batch_size = 32
    num_epoch = 1000

    # setting
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    #pre_optim = ""
    #optimizer = torch.load(pre_optim)
    kf = KFold(n_splits=4)
    print("setting done")    

    print("ready to train")
    sigmoid = nn.Sigmoid()
    for epoch in range(0, num_epoch, 4):#訓練開始
        idx_epoch = 0

        for train_idx, valid_idx in kf.split(tr_data):
            ds_train = lcdDataset([tr_data[i] for i in train_idx], [tr_label[i] for i in train_idx], data_list[train_idx], 64, True)
            ds_valid = lcdDataset([tr_data[i] for i in valid_idx], [tr_label[i] for i in valid_idx], data_list[valid_idx], 64, True)
            dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
            dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)

            model.train()
            begin_epoch = time.time() 
            sum_loss = 0
            for idx, (data, mask) in enumerate(dl_train):
                data = data.to(device)
                mask = mask.to(device)

                output = model(data)
                pred = 0
                for i in range(4):
                    pred += sigmoid(output[i])
                pred /= 4
                loss = criterion(pred.squeeze(), mask.squeeze(), 0.5, 1, 1e-5)
                sum_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end_step = time.time()
                print("\rEpoch [{}/{}] - {}/{} - {}s - Loss: {:.4f}".format(epoch+idx_epoch+1, num_epoch, idx+1, len(ds_train), end_step - begin_epoch, loss), end='')
            end_epoch = time.time()
            print("\rEpoch [{}/{}] - {}/{} - {}s/epoch - AverageLoss: {} - ".format(epoch+idx_epoch+1, num_epoch, len(ds_train), len(ds_train), end_epoch - begin_epoch, sum_loss/len(dl_train)), end='')

            model.eval()
            val_loss_list = np.array([0, 0, 0, 0], dtype=np.float32)
            for idx, (data, mask) in enumerate(dl_valid):
                with torch.no_grad():
                    data = data.to(device)
                    mask = mask.to(device)

                    output = model(data)
                    for i in range(4):
                        pred = sigmoid(output[i]).detach().to('cpu').numpy().squeeze()
                        pred = np.where(pred > 0.5, 1, 0)
                        mask_gt = np.where(mask.detach().to('cpu').numpy() > 0, 1, 0)
                        val_loss_list[i] += jaccard_score(mask_gt.flatten(), pred.flatten())
            end_epoch = time.time()
            print("ValidLoss: {} - {} s/epoch".format(val_loss_list/len(dl_valid), end_epoch - begin_epoch))

            torch.save(model.state_dict(), './ckp/model_parameters/unet_param_epoch_{}.pkl'.format(epoch+idx_epoch+1))
            torch.save(optimizer, './ckp/optimizer/optimizer_epoch_{}.pkl'.format(epoch+idx_epoch+1))

            idx_epoch += 1