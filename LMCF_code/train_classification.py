import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import torch
import joblib
import random
import json
import math
import sys
import argparse
import numpy as np
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from mae_model_classificaiton import fusion_model_mae_2
from util import Logger, get_patients_information1, get_all_ci, get_val_ci, adjust_learning_rate
from mae_utils import generate_mask
from bls import BLS_Genfeatures
import time
from test_classification import train_threshold,test_threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bls_train_classification(all_data,train_data,test_data,patients):
    x_cli = []
    x_cli1 = []
    for id in train_data:
        # print(len(all_data[id].x_cli.numpy()))
        x_cli.append(all_data[id].x_cli.cpu().numpy())

    print(len(x_cli))
    x_train_cli = np.stack(x_cli, axis=0)

    print(x_train_cli.shape)
    # print(x_train_cli[3])

    for id in test_data:
        x_cli1.append(all_data[id].x_cli.cpu().numpy())
    print(len(x_cli1))
    # print(x_cli_test[0])
    x_test_cli = np.stack(x_cli1, axis=0)
    print(x_test_cli.shape)

    x_rna = []
    x_rna1 = []
    for id in train_data:
        # print(len(all_data[id].x_cli.numpy()))
        x_rna.append(all_data[id].x_rna.cpu().numpy())

    print(len(x_rna))
    x_train_rna = np.stack(x_rna, axis=0)

    print(x_train_rna.shape)
    # print(x_train_rna[0])

    for id in test_data:
        x_rna1.append(all_data[id].x_rna.cpu().numpy())
    print(len(x_rna1))
    # print(x_rna_test[0])
    x_test_rna = np.stack(x_rna1, axis=0)
    print(x_test_rna.shape)

    x_cli_train ,x_cli_test = BLS_Genfeatures(x_train_cli,x_test_cli,10,10,50,0.8)
    x_rna_train ,x_rna_test = BLS_Genfeatures(x_train_rna,x_test_rna,10,10,50,0.8)
    # print('xxxxxxxxx',x_cli_train.shape)
    x_cli = np.vstack((x_cli_train,x_cli_test))
    x_rna = np.vstack((x_rna_train,x_rna_test))
    # if np.any(np.isnan(x_cli)):
    #     print("数组中包含NaN值")
    # else:
    #     print("数组中不包含NaN值")
    print(x_cli_train.shape)
    print(x_cli_test.shape)
    print(x_cli.shape)
    # print(x_cli)
    print(x_rna_train.shape)
    print(x_rna_test.shape)
    print(x_rna.shape)

    t_rna_feas = {}
    i=-1
    for id in train_data:
        i+=1
        t_rna_feas[id] = x_rna[i]

    for id in test_data:
        i+=1
        t_rna_feas[id] = x_rna[i]
    t_cli_feas = {}
    j=-1
    for id in train_data:
        j+=1
        t_cli_feas[id] = x_cli[j]

    for id in test_data:
        j+=1
        t_cli_feas[id] = x_cli[j]

    for i in range(150):
        zz = []
        for x in t_rna_feas:
            #         print(x,len(t_rna_feas[x]))
            zz.append(t_rna_feas[x][i])
        # print('zz',zz)
        zz = np.array(zz)
        maxx = np.max(zz)
        minn = np.min(zz)
        t_rna_fea = t_rna_feas
        for x in t_rna_feas:
            t_rna_fea[x][i] = (t_rna_feas[x][i] - (maxx + minn) / 2) / (maxx - minn) * 2
    onehot_rna = {}
    for x in patients:
        tmp = np.zeros((len(t_rna_fea[x]), 1024))
        k = 0
        for i, z in enumerate(t_rna_fea[x]):
            tmp[k][i] = t_rna_fea[x][i]
            k += 1
        onehot_rna[x] = tmp
        # if np.sum(np.isnan(tmp)) > 0:
        #     print("数组中包含NaN值")
        # else:
        #     print("数组中不包含NaN值")

    for i in range(150):
        zz = []
        for x in t_cli_feas:
            #         print(x,len(t_cli_feas[x]))
            # print(t_cli_feas[x])
            zz.append(t_cli_feas[x][i])
        # print('zz',zz)
        zz = np.array(zz)
        maxx = np.max(zz)
        minn = np.min(zz)
        t_cli_fea = t_cli_feas
        for x in t_cli_feas:
            t_cli_fea[x][i] = (t_cli_feas[x][i] - (maxx + minn) / 2) / (maxx - minn) * 2
    onehot_cli = {}
    for x in patients:
        tmp = np.zeros((len(t_cli_fea[x]), 1024))
        k = 0
        for i, z in enumerate(t_cli_fea[x]):
            tmp[k][i] = t_cli_fea[x][i]
            k += 1
        onehot_cli[x] = tmp
        # if np.sum(np.isnan(tmp)) > 0:
        #     print("数组中包含NaN值")
        # else:
        #     print("数组中不包含NaN值")

    # node_rna = torch.tensor(onehot_rna[id], dtype=torch.float)
    # node_cli = torch.tensor(onehot_cli[id], dtype=torch.float)
    return onehot_rna,onehot_cli


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def train_a_epoch(model, train_data, all_data, patient_sur_type, batch_size, optimizer, args):
    model.train()
    mse_loss_of_mae = 0.0
    mes_loss_of_mae = nn.MSELoss()  # 均方误差损失估计方法
    loss_function = nn.BCELoss(weight=None, reduction='mean')
    iter = 0
    all_loss = 0

    for i_batch, id in enumerate(train_data):
        # print(id)
        # raise 1
        iter+=1
        graph = all_data[id].to(device)
        use_type_eopch = args.train_use_type
        # 0和1组成的随机mask (1, 1, len(args.train_use_type))=(1, 1, 3)
        mask = generate_mask(num=len(args.train_use_type))
        fea_dict, (one_y, multi_y) = model(graph, use_type_eopch, use_type_eopch, mask, mix=args.mix)[-2:]
        # print(one_y)
        # raise 1
        if args.add_mse_loss_of_mae:
            mse_loss_of_mae = args.mse_loss_of_mae_factor * mes_loss_of_mae(input=fea_dict['mae_out'][mask[0]],
                                                                            target=fea_dict['mae_labels'][mask[0]])
        if iter == 1:
            loss_surv = 0
        output_tensor = multi_y
        # print(output_tensor.shape)
        # print(output_tensor[0].unsqueeze(0))

        loss1 = loss_function(output_tensor[0],patient_sur_type[id].to(device)).unsqueeze(0)
        # print(loss1)
        loss2 = loss_function(output_tensor[1],patient_sur_type[id].to(device)).unsqueeze(0)
        one_loss = torch.cat((loss1,loss2))
        loss3 = loss_function(output_tensor[2], patient_sur_type[id].to(device)).unsqueeze(0)
        one_loss = torch.cat((one_loss,loss3))#一个样本的损失
        # print(loss)
        loss_surv += torch.mean(one_loss)
        # print(loss_surv)



        # loss = loss_surv  # loss是过程量 loss = loss_surv + mse_loss_of_mae / iter
        if args.add_mse_loss_of_mae:  # <<<
            loss_surv += mse_loss_of_mae



        if iter % batch_size == 0 or i_batch == len(train_data) - 1:#一个batch
            optimizer.zero_grad()
            all_loss += loss_surv
            loss = loss_surv/args.batch_size
            # print('batch loss',loss)
            loss.backward()  # loss_surv + mse_loss_of_mae
            optimizer.step()
            loss_surv = 0  # 每个batch loss重置

    return all_loss/iter


def test(model, test_data, all_data, patient_sur_type, args):
    from sklearn.metrics import accuracy_score
    model.eval()
    prd_y, tar_y = [],[]
    for i_batch, id in enumerate(test_data):
        graph = all_data[id].to(device)
        one_y, _= model(graph,train_use_type=args.train_use_type,use_type=args.train_use_type,mix=args.mix)[-1]
        prd_y.append(one_y.cpu().detach().numpy())
        tar_y.append(patient_sur_type[id].numpy())
    #
    # thresholds = np.arange(0.1, 1.0, 0.1)
    # best_acc = 0
    # best_threshold = 0.5
    #
    # for thresh in thresholds:
    #     y_pred_label = (prd_y >= thresh).astype(int)
    #     acc = accuracy_score(tar_y, y_pred_label)
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_threshold = thresh
    # print(f"best_threshold:{best_threshold}")
    y_pred_label = [1 if prob >= 0.5 else 0 for prob in prd_y]
    accuracy = accuracy_score(tar_y, y_pred_label)
    # print(prd_y)
    print(f'test acc:{accuracy}\n')
    # print(len(prd_y[0]))
    # print(len(tar_y[0]))
    return accuracy

def main(args):
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    model_type = args.model_type
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr

    label = "{} {} lr_{} {}_coxloss".format(cancer_type, details, lr, format_of_coxloss)

    if args.add_mse_loss_of_mae:
        label = label + " {}*mae_loss".format(args.mse_loss_of_mae_factor)

    if args.img_cox_loss_factor != 1:
        label = label + " img_ft_{}".format(args.img_cox_loss_factor)
    if args.rna_cox_loss_factor != 1:
        label = label + " rna_ft_{}".format(args.rna_cox_loss_factor)
    if args.cli_cox_loss_factor != 1:
        label = label + " cli_ft_{}".format(args.cli_cox_loss_factor)
    if args.mix:
        label = label + " mix"
    if args.train_use_type != None:
        label = label + ' use_'
        for x in args.train_use_type:
            label = label + x

    print(label)

    if cancer_type == 'mydataset':
        patients = joblib.load('patients.pkl')
        sur_and_time = joblib.load('sur_and_time.pkl')
        if model_type == 'hgcn':
            all_data = joblib.load(r'E:\Project\HGCN-main\gendata\all_data_hgcn_mydataset.pkl')
        seed_fit_split = joblib.load('mydataset_split_classificaiton.pkl')
    if cancer_type == 'mydataset1':
        patients = joblib.load(r'E:\Project\HGCN-main\dataset1\patients.pkl')
        sur_and_time = joblib.load('E:\Project\HGCN-main\dataset1\sur_and_time.pkl')
        if model_type == 'hgcn':
            all_data = joblib.load(r'E:\Project\HGCN-main\dataset1\all_data_hgcn_mydataset1.pkl')
        seed_fit_split = joblib.load('E:\Project\HGCN-main\dataset1\mydataset1_split_classificaiton.pkl')

    patient_sur_type, patient_and_time, kf_label = get_patients_information1(patients, sur_and_time)

    repeat = -1
    for seed in range(start_seed, start_seed + repeat_num):
        repeat += 1
        setup_seed(0)

        model = fusion_model_mae_2(in_feats=1024,
                                    n_hidden=args.n_hidden,
                                    out_classes=args.out_classes,
                                    dropout=drop_out_ratio,
                                    train_type_num=len(args.train_use_type)
                                    ).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        if cancer_type == 'mydataset':
            train_data = seed_fit_split[0][0]
            test_data = seed_fit_split[0][1]
        elif cancer_type == 'mydataset1':
            train_data = seed_fit_split[0]
            test_data = seed_fit_split[1]

        def bls(all_data, train_data, test_data, patients):
            node_rna, node_cli = bls_train_classification(all_data, train_data, test_data, patients)
            for id in patients:
                all_data[id].x_rna = torch.tensor(node_rna[id], dtype=torch.float).to(device)
                all_data[id].x_cli = torch.tensor(node_cli[id], dtype=torch.float).to(device)
                all_data[id].to(device)
            return all_data

        if model_type == 'my':
            if cancer_type == 'mydataset':
                # all_data = joblib.load('all.pkl')
                all_data = joblib.load(r'E:\Project\HGCN-main\gendata\all_150.pkl')
                all_data = bls(all_data, train_data, test_data, patients)
            elif cancer_type == 'mydataset1':
                all_data = joblib.load(r'E:\Project\HGCN-main\dataset1\alldata.pkl')
                all_data = bls(all_data, train_data, test_data, patients)


        print(len(train_data), len(test_data))


        best_acc = 0
        train_start_time = time.time()
        for epoch in range(epochs):
            print(f'epoch:{epoch+1}')
            if if_adjust_lr:
                adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=args.adjust_lr_ratio)
            loss = train_a_epoch(model, train_data,all_data,patient_sur_type,batch_size,optimizer,args).item()
            print('loss:',loss)

            # _, best_shreshold = train_threshold(model, train_data, all_data, patient_sur_type, args)
            # accuracy = test_threshold(model, test_data, all_data, patient_sur_type, args,best_shreshold)
            accuracy = test(model, test_data, all_data, patient_sur_type, args)

            if accuracy > best_acc:
                best_acc = accuracy
                print(f'**best_acc:{best_acc}**')
                # torch.save(model.state_dict(), 'best_weights.pth')
        import datetime
        train_time = time.time() - train_start_time
        train_time = str(datetime.timedelta(seconds=int(train_time)))
        print(f'train time:{train_time}')

        print(f'\n**best_acc:{best_acc}**\n')
        # torch.save(model.state_dict(), 'weights.pth')
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="my", help="my/hgcn")
    parser.add_argument("--cancer_type", type=str, default="mydataset", help="Cancer type:mydataset/mydataset1/lihc/kirc/luad/ucec")
    parser.add_argument("--img_cox_loss_factor", type=float, default=5, help="img_cox_loss_factor")
    parser.add_argument("--rna_cox_loss_factor", type=float, default=1, help="rna_cox_loss_factor")
    parser.add_argument("--cli_cox_loss_factor", type=float, default=5, help="cli_cox_loss_factor")
    parser.add_argument("--train_use_type", type=list, default=['img', 'rna', 'cli'],
                        help='train_use_type,Please keep the relative order of img, rna, cli')
    parser.add_argument("--format_of_coxloss", type=str, default="multi", help="format_of_coxloss:multi,one")
    parser.add_argument("--add_mse_loss_of_mae", action='store_true', default=True, help="add_mse_loss_of_mae")###
    parser.add_argument("--mse_loss_of_mae_factor", type=float, default=5, help="mae_loss_factor")
    parser.add_argument("--start_seed", type=int, default=0, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="fusion_model_mae_2", help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.6, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of model training")  # 0.00003
    parser.add_argument("--epochs", type=int, default=30, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Data volume of model training once")  # 32
    parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")
    parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension")
    parser.add_argument("--mix", action='store_true', default=True, help="mix mae")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=True, help="fixed division/random division")
    parser.add_argument("--details", type=str, default='', help="Experimental details")

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise
