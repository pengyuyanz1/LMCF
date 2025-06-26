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

    x_cli_train ,x_cli_test = BLS_Genfeatures(x_train_cli,x_test_cli,10,10,100,0.8)
    x_rna_train ,x_rna_test = BLS_Genfeatures(x_train_rna,x_test_rna,10,10,100,0.8)
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

    for i in range(200):
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

    for i in range(200):
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



def train_threshold(model, train_data, all_data, patient_sur_type, args):
    from sklearn.metrics import accuracy_score
    model.eval()
    prd_y, tar_y = [], []
    for i_batch, id in enumerate(train_data):
        graph = all_data[id].to(device)
        one_y, _ = model(graph, train_use_type=args.train_use_type, use_type=args.train_use_type, mix=args.mix)[-1]
        prd_y.append(one_y.cpu().detach().numpy())
        tar_y.append(patient_sur_type[id].numpy())

    thresholds = np.arange(0.45, 0.55, 0.01)
    best_acc = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred_label = (prd_y >= thresh).astype(int)
        acc = accuracy_score(tar_y, y_pred_label)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    print(f"best_threshold:{best_threshold}")

    y_pred_label = [1 if prob >= best_threshold else 0 for prob in prd_y]
    accuracy = accuracy_score(tar_y, y_pred_label)
    # print(prd_y)
    print(f'train acc:{accuracy}\n')
    # print(len(prd_y[0]))
    # print(len(tar_y[0]))
    return accuracy,best_threshold


def test_threshold(model, test_data, all_data, patient_sur_type, args,best_threshold):
    from sklearn.metrics import accuracy_score
    model.eval()
    prd_y, tar_y = [],[]
    for i_batch, id in enumerate(test_data):
        graph = all_data[id].to(device)
        one_y, _= model(graph,train_use_type=args.train_use_type,use_type=args.train_use_type,mix=args.mix)[-1]
        prd_y.append(one_y.cpu().detach().numpy())
        tar_y.append(patient_sur_type[id].numpy())

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
    y_pred_label = [1 if prob >= best_threshold else 0 for prob in prd_y]
    accuracy = accuracy_score(tar_y, y_pred_label)
    # print(prd_y)
    print(f'test acc:{accuracy}\n')
    # print(len(prd_y[0]))
    # print(len(tar_y[0]))
    from sklearn.metrics import roc_auc_score

    # 假设 y_true 是真实标签列表（0或1），y_pred 是预测概率列表（正类的概率）
    auc = roc_auc_score(tar_y, prd_y)
    print("AUC值：", auc)
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
            all_data = joblib.load('all.pkl')
            all_data = bls(all_data, train_data, test_data, patients)
        elif cancer_type == 'mydataset1':
            all_data = joblib.load(r'E:\Project\HGCN-main\dataset1\alldata.pkl')
            all_data = bls(all_data, train_data, test_data, patients)

    model.load_state_dict(torch.load('weights.pth'))
    _,best_shreshold = train_threshold(model, train_data,all_data,patient_sur_type,args)
    acc=test_threshold(model, test_data, all_data, patient_sur_type, args,best_shreshold)

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hgcn", help="my/hgcn")
    parser.add_argument("--cancer_type", type=str, default="mydataset1", help="Cancer type:mydataset/mydataset1/lihc/kirc/luad/ucec")
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
    parser.add_argument("--drop_out_ratio", type=float, default=0.5, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate of model training")  # 0.00003
    parser.add_argument("--epochs", type=int, default=26, help="Cycle times of model training")
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
