import joblib
import numpy as np
from bls import BLS_Genfeatures
import torch





# train_data = seed_fit_split[0][0]
# x_cli=[]
# for id in train_data:
    # print(id)
    # print(all_data[id].x_cli.numpy().shape)
    # x_cli.append(all_data[id].x_cli.numpy().ravel())
# x_train = np.stack(x_cli, axis=0)
# print(x_train.shape)
# print(patient_sur_type)
# val_data = seed_fit_split[0][1]
# for id in val_data:
    # print(id)
    # print(all_data[id].x_cli.numpy().shape)
    # x_cli.append(all_data[id].x_cli.numpy().ravel())
# x_train = np.stack(x_cli, axis=0)
# print(x_train.shape)
# test_data = seed_fit_split[0][2]
# x_cli1=[]
# for id in test_data:
    # print(id)
    # print(all_data[id].x_cli.numpy().shape)
    # x_cli1.append(all_data[id].x_cli.numpy().ravel())
# x_test = np.stack(x_cli1, axis=0)
# print(x_test.shape)
# train ,test = BLS_Genfeatures(x_train,x_test,10,10,500,0.8)
# print(train.shape)
# print(test.shape)
# print(train)
# print(test)
def bls_train(all_data,train_data,val_data,test_data,patients):
    x_cli = []
    x_cli1 = []
    for id in train_data:
        # print(len(all_data[id].x_cli.numpy()))
        x_cli.append(all_data[id].x_cli.cpu().numpy())
    for id in val_data:
        x_cli.append((all_data[id].x_cli.cpu().numpy()))
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
    for id in val_data:
        x_rna.append((all_data[id].x_rna.cpu().numpy()))
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

    x_cli_train ,x_cli_test = BLS_Genfeatures(x_train_cli,x_test_cli,5,10,50,0.8)
    x_rna_train ,x_rna_test = BLS_Genfeatures(x_train_rna,x_test_rna,5,10,50,0.8)
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
    for id in val_data:
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
    for id in val_data:
        j+=1
        t_cli_feas[id] = x_cli[j]
    for id in test_data:
        j+=1
        t_cli_feas[id] = x_cli[j]

    for i in range(100):
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

    for i in range(100):
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


def bls_train_tcga(all_data,train_data,val_data,test_data,patients):
    x_cli = []
    x_cli1 = []
    for id in train_data:
        # print(len(all_data[id].x_cli.numpy()))
        x_cli.append(all_data[id].x_cli.cpu().numpy())
    for id in val_data:
        x_cli.append((all_data[id].x_cli.cpu().numpy()))
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


    x_cli_train ,x_cli_test = BLS_Genfeatures(x_train_cli,x_test_cli,10,10,100,0.8)
    # print('xxxxxxxxx',x_cli_train.shape)
    x_cli = np.vstack((x_cli_train,x_cli_test))
    # if np.any(np.isnan(x_cli)):
    #     print("数组中包含NaN值")
    # else:
    #     print("数组中不包含NaN值")
    print(x_cli_train.shape)
    print(x_cli_test.shape)
    print(x_cli.shape)
    # print(x_cli)



    t_cli_feas = {}
    j=-1
    for id in train_data:
        j+=1
        t_cli_feas[id] = x_cli[j]
    for id in val_data:
        j+=1
        t_cli_feas[id] = x_cli[j]
    for id in test_data:
        j+=1
        t_cli_feas[id] = x_cli[j]


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
    return onehot_cli


if __name__ == '__main__':
    patients = joblib.load('patients.pkl')
    sur_and_time = joblib.load('sur_and_time.pkl')
    all_data = joblib.load('all_data111.pkl')
    seed_fit_split = joblib.load('mydata_split.pkl')


    train_data = seed_fit_split[0][0]
    val_data = seed_fit_split[0][1]
    test_data = seed_fit_split[0][2]
    node_rna,node_cli=bls_train(all_data,train_data,val_data,test_data,patients)
    # for i in node_cli.keys():
    #     print(node_cli[i].shape)
    #     print(node_rna[i].shape)