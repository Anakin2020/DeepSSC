import os
import concurrent.futures
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from model import VAE_EAD
from utils import evaluate, extractEdgesFromMatrix
import threading
import argparse
Tensor = torch.cuda.FloatTensor
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--setting', type=str, default='default')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_file', type=str)
parser.add_argument('--data_stage1', type=str)
parser.add_argument('--data_stage2', type=str)
parser.add_argument('--data_stage3', type=str)
parser.add_argument('--data_stage4_Lym', type=str)
parser.add_argument('--data_stage4_GM', type=str)
parser.add_argument('--data_stage4_ME', type=str)
parser.add_argument('--net_file', type=str, default='',)
parser.add_argument('--alpha', type=float, default=100)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--l', type=float, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_step_size', type=int, default=0.99)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--K1', type=int, default=1)
parser.add_argument('--K2', type=int, default=2)
parser.add_argument('--save_name', type=str, default='/tmp')
opt = parser.parse_args()
class DeepSSC_model:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')
    def initalize_A(self, data):
        num_genes = data.shape[1]
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A
    def init_data(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data = sc.read(self.opt.data_file)
        gene_name = list(data.var_names)
        data_values = data.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))
        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name
    def init_data1(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data1 = sc.read(self.opt.data_stage1)
        gene_name1 = list(data1.var_names)
        data_values = data1.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data1 = pd.DataFrame(data_values, index=list(data1.obs_names), columns=gene_name1)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes1, num_nodes1 = data1.shape[1], data1.shape[0]
        Evaluate_Mask1 = np.zeros([num_genes1, num_genes1])
        TF_mask1 = np.zeros([num_genes1, num_genes1])
        for i, item in enumerate(data1.columns):
            for j, item2 in enumerate(data1.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask1[i, j] = 1
                if item2 in TF:
                    TF_mask1[i, j] = 1
        feat_train = torch.FloatTensor(data1.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader1 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        truth_df = pd.DataFrame(np.zeros([num_genes1, num_genes1]), index=data1.columns, columns=data1.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges1 = set(zip(idx_send, idx_rec))
        return dataloader1, Evaluate_Mask1, num_nodes1, num_genes1, data1, truth_edges1, TF_mask1, gene_name1

    def init_data2(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data2 = sc.read(self.opt.data_stage2)
        gene_name2 = list(data2.var_names)
        data_values = data2.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data2 = pd.DataFrame(data_values, index=list(data2.obs_names), columns=gene_name2)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes2, num_nodes2 = data2.shape[1], data2.shape[0]
        Evaluate_Mask2 = np.zeros([num_genes2, num_genes2])
        TF_mask2 = np.zeros([num_genes2, num_genes2])
        for i, item in enumerate(data2.columns):
            for j, item2 in enumerate(data2.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask2[i, j] = 1
                if item2 in TF:
                    TF_mask2[i, j] = 1
        feat_train = torch.FloatTensor(data2.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader2 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        truth_df = pd.DataFrame(np.zeros([num_genes2, num_genes2]), index=data2.columns, columns=data2.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges2 = set(zip(idx_send, idx_rec))
        return dataloader2, Evaluate_Mask2, num_nodes2, num_genes2, data2, truth_edges2, TF_mask2, gene_name2

    def init_data3(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data3 = sc.read(self.opt.data_stage3)
        gene_name3 = list(data3.var_names)
        data_values = data3.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data3 = pd.DataFrame(data_values, index=list(data3.obs_names), columns=gene_name3)##########
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes3, num_nodes3 = data3.shape[1], data3.shape[0]
        Evaluate_Mask3 = np.zeros([num_genes3, num_genes3])##########
        TF_mask3 = np.zeros([num_genes3, num_genes3])
        for i, item in enumerate(data3.columns):
            for j, item2 in enumerate(data3.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask3[i, j] = 1
                if item2 in TF:
                    TF_mask3[i, j] = 1
        feat_train = torch.FloatTensor(data3.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader3 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)##########
        truth_df = pd.DataFrame(np.zeros([num_genes3, num_genes3]), index=data3.columns, columns=data3.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges3 = set(zip(idx_send, idx_rec))
        return dataloader3, Evaluate_Mask3, num_nodes3, num_genes3, data3, truth_edges3, TF_mask3, gene_name3
    def init_data4(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data4 = sc.read(self.opt.data_stage4_Lym)
        gene_name4 = list(data4.var_names)
        data_values = data4.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data4 = pd.DataFrame(data_values, index=list(data4.obs_names), columns=gene_name4)##########
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes4, num_nodes4 = data4.shape[1], data4.shape[0]
        Evaluate_Mask4 = np.zeros([num_genes4, num_genes4])##########
        TF_mask4 = np.zeros([num_genes4, num_genes4])
        for i, item in enumerate(data4.columns):
            for j, item2 in enumerate(data4.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask4[i, j] = 1
                if item2 in TF:
                    TF_mask4[i, j] = 1
        feat_train = torch.FloatTensor(data4.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader4 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)##########
        truth_df = pd.DataFrame(np.zeros([num_genes4, num_genes4]), index=data4.columns, columns=data4.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges4 = set(zip(idx_send, idx_rec))
        return dataloader4, Evaluate_Mask4, num_nodes4, num_genes4, data4, truth_edges4, TF_mask4, gene_name4


    def init_data5(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data5 = sc.read(self.opt.data_stage4_GM)
        gene_name5 = list(data5.var_names)
        data_values = data5.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0)+ 1e-20)
        data5 = pd.DataFrame(data_values, index=list(data5.obs_names), columns=gene_name5)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes5, num_nodes5 = data5.shape[1], data5.shape[0]
        Evaluate_Mask5 = np.zeros([num_genes5, num_genes5])##########
        TF_mask5 = np.zeros([num_genes5, num_genes5])
        for i, item in enumerate(data5.columns):
            for j, item2 in enumerate(data5.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask5[i, j] = 1
                if item2 in TF:
                    TF_mask5[i, j] = 1
        feat_train = torch.FloatTensor(data5.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader5 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)##########
        truth_df = pd.DataFrame(np.zeros([num_genes5, num_genes5]), index=data5.columns, columns=data5.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges5 = set(zip(idx_send, idx_rec))
        return dataloader5, Evaluate_Mask5, num_nodes5, num_genes5, data5, truth_edges5, TF_mask5, gene_name5


    def init_data6(self):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data6 = sc.read(self.opt.data_stage4_ME)
        gene_name6 = list(data6.var_names)
        data_values = data6.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0) + 1e-20)
        data6 = pd.DataFrame(data_values, index=list(data6.obs_names), columns=gene_name6)  ##########
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes6, num_nodes6 = data6.shape[1], data6.shape[0]
        Evaluate_Mask6 = np.zeros([num_genes6, num_genes6])
        TF_mask6 = np.zeros([num_genes6, num_genes6])
        for i, item in enumerate(data6.columns):
            for j, item2 in enumerate(data6.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask6[i, j] = 1
                if item2 in TF:
                    TF_mask6[i, j] = 1
        feat_train = torch.FloatTensor(data6.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader6 = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)  ##########
        truth_df = pd.DataFrame(np.zeros([num_genes6, num_genes6]), index=data6.columns, columns=data6.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges6 = set(zip(idx_send, idx_rec))
        return dataloader6, Evaluate_Mask6, num_nodes6, num_genes6, data6, truth_edges6, TF_mask6, gene_name6
    def train_model(self):
        opt = self.opt
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name = self.init_data()
        adj_A_init = self.initalize_A(data)
        vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader1, Evaluate_Mask1, num_nodes1, num_genes1, data1, truth_edges1, TFmask1, gene_name1 = self.init_data1()
        adj_A_init1 = self.initalize_A(data1)
        vae1 = VAE_EAD(adj_A_init1, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_1 = optim.RMSprop(vae1.parameters(), lr=opt.lr)
        optimizer2_1 = optim.RMSprop([vae1.adj_A], lr=opt.lr * 0.2)
        w_stage1_2 = torch.load('data_mHSC/data/w/w_stage2.pt')
        w_stage1_2 = w_stage1_2.to(vae1.adj_A.device)
        l1_1 = vae1.adj_A - w_stage1_2
        l1_1 = l1_1.detach()
        optimizer_stage1_1 = optim.RMSprop([l1_1], lr=opt.lr * 0.01)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader2, Evaluate_Mask2, num_nodes2, num_genes2, data2, truth_edges2, TFmask2, gene_name2 = self.init_data2()
        adj_A_init2 = self.initalize_A(data2)
        vae2 = VAE_EAD(adj_A_init2, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_2 = optim.RMSprop(vae2.parameters(), lr=opt.lr)
        optimizer2_2 = optim.RMSprop([vae2.adj_A], lr=opt.lr * 0.2)
        w_stage2_1 = torch.load('data_mHSC/data/w/w_stage1.pt')
        w_stage2_1 = w_stage2_1.to(vae2.adj_A.device)
        l2_1 = vae2.adj_A - w_stage2_1
        l2_1 = l2_1.detach()
        optimizer_stage2_1 = optim.RMSprop([l2_1], lr=opt.lr * 0.01)
        w_stage2_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
        w_stage2_3 = w_stage2_3.to(vae2.adj_A.device)
        l2_2 = vae2.adj_A - w_stage2_3
        l2_2 = l2_2.detach()
        optimizer_stage2_2 = optim.RMSprop([l2_2], lr=opt.lr * 0.01)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader3, Evaluate_Mask3, num_nodes3, num_genes3, data3, truth_edges3, TFmask3, gene_name3 = self.init_data3()
        adj_A_init3 = self.initalize_A(data3)
        vae3 = VAE_EAD(adj_A_init3, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_3 = optim.RMSprop(vae3.parameters(), lr=opt.lr)
        optimizer2_3 = optim.RMSprop([vae3.adj_A], lr=opt.lr * 0.2)
        w_stage3_2 = torch.load('data_mHSC/data/w/w_stage2.pt')
        w_stage3_2 = w_stage3_2.to(vae3.adj_A.device)
        l3_1 = vae3.adj_A - w_stage3_2
        l3_1 = l3_1.detach()
        optimizer_stage3_1 = optim.RMSprop([l3_1], lr=opt.lr * 0.01)
        w_stage3_ME = torch.load('data_mHSC/data/w/w_stage4_ME.pt')
        w_stage3_ME = w_stage3_ME.to(vae3.adj_A.device)
        l3_2 = vae3.adj_A - w_stage3_ME
        l3_2 = l3_2.detach()
        optimizer_stage3_2 = optim.RMSprop([l3_2], lr=opt.lr * 0.01)
        w_stage3_GM = torch.load('data_mHSC/data/w/w_stage4_GM.pt')
        w_stage3_GM = w_stage3_GM.to(vae3.adj_A.device)
        l3_3 = vae3.adj_A - w_stage3_GM
        l3_3 = l3_3.detach()
        optimizer_stage3_3 = optim.RMSprop([l3_3], lr=opt.lr * 0.01)
        w_stage3_Lym = torch.load('data_mHSC/data/w/w_stage4_Lym.pt')
        w_stage3_Lym = w_stage3_Lym.to(vae3.adj_A.device)
        l3_4 = vae3.adj_A - w_stage3_Lym
        l3_4 = l3_4.detach()
        optimizer_stage3_4 = optim.RMSprop([l3_4], lr=opt.lr * 0.01)
        scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_3, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader4, Evaluate_Mask4, num_nodes4, num_genes4, data4, truth_edges4, TFmask4, gene_name4 = self.init_data4()
        adj_A_init4 = self.initalize_A(data4)
        vae4 = VAE_EAD(adj_A_init4, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_4 = optim.RMSprop(vae4.parameters(), lr=opt.lr)
        optimizer2_4 = optim.RMSprop([vae4.adj_A], lr=opt.lr * 0.2)
        w_stage4_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
        w_stage4_3 = w_stage4_3.to(vae4.adj_A.device)
        l4_1 = vae4.adj_A - w_stage4_3
        l4_1 = l4_1.detach()
        optimizer_stage4_1 = optim.RMSprop([l4_1], lr=opt.lr * 0.01)
        scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer_4, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader5, Evaluate_Mask5, num_nodes5, num_genes5, data5, truth_edges5, TFmask5, gene_name5 = self.init_data5()
        adj_A_init5 = self.initalize_A(data5)
        vae5 = VAE_EAD(adj_A_init5, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_5 = optim.RMSprop(vae5.parameters(), lr=opt.lr)
        optimizer2_5 = optim.RMSprop([vae5.adj_A], lr=opt.lr * 0.2)
        w_stage5_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
        w_stage5_3 = w_stage5_3.to(vae5.adj_A.device)
        l5_1 = vae5.adj_A - w_stage5_3
        l5_1 = l5_1.detach()
        optimizer_stage5_1 = optim.RMSprop([l5_1], lr=opt.lr * 0.01)
        scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer_5, step_size=opt.lr_step_size, gamma=opt.gamma)
        dataloader6, Evaluate_Mask6, num_nodes6, num_genes6, data6, truth_edges6, TFmask6, gene_name6 = self.init_data6()
        adj_A_init6 = self.initalize_A(data6)
        vae6 = VAE_EAD(adj_A_init6, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer_6 = optim.RMSprop(vae6.parameters(), lr=opt.lr)
        optimizer2_6 = optim.RMSprop([vae6.adj_A], lr=opt.lr * 0.2)
        w_stage6_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
        w_stage6_3 = w_stage6_3.to(vae6.adj_A.device)
        l6_1 = vae6.adj_A - w_stage6_3
        l6_1 = l6_1.detach()
        optimizer_stage6_1 = optim.RMSprop([l6_1], lr=opt.lr * 0.01)
        scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer_6, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        def train_vae(vae):
            vae.train()
        thread = threading.Thread(target=train_vae, args=(vae,))
        thread1 = threading.Thread(target=train_vae, args=(vae1,))
        thread2 = threading.Thread(target=train_vae, args=(vae2,))
        thread3 = threading.Thread(target=train_vae, args=(vae3,))
        thread4 = threading.Thread(target=train_vae, args=(vae4,))
        thread5 = threading.Thread(target=train_vae, args=(vae5,))
        thread6 = threading.Thread(target=train_vae, args=(vae6,))
        thread.start()
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread5.start()
        thread6.start()
        thread.join()
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
        thread5.join()
        thread6.join()
        for epoch in range(opt.n_epochs + 1):
            loss_w_stage= []
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            loss_all1, mse_rec1, loss_kl1, data_ids1, loss_tfs1, loss_sparse1 = [], [], [], [], [], []
            loss_all2, mse_rec2, loss_kl2, data_ids2, loss_tfs2, loss_sparse2 = [], [], [], [], [], []
            loss_all3, mse_rec3, loss_kl3, data_ids3, loss_tfs3, loss_sparse3 = [], [], [], [], [], []
            loss_all4, mse_rec4, loss_kl4, data_ids4, loss_tfs4, loss_sparse4 = [], [], [], [], [], []
            loss_all5, mse_rec5, loss_kl5, data_ids5, loss_tfs5, loss_sparse5 = [], [], [], [], [], []
            loss_all6, mse_rec6, loss_kl6, data_ids6, loss_tfs6, loss_sparse6 = [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                vae.adj_A.requires_grad = False
                vae1.adj_A.requires_grad = False
                vae2.adj_A.requires_grad = False
                vae3.adj_A.requires_grad = False
                vae4.adj_A.requires_grad = False
                vae5.adj_A.requires_grad = False
                vae6.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
                vae1.adj_A.requires_grad = True
                vae2.adj_A.requires_grad = True
                vae3.adj_A.requires_grad = True
                vae4.adj_A.requires_grad = True
                vae5.adj_A.requires_grad = True
                vae6.adj_A.requires_grad = True
            for i, (data_batch,data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, data_batch6) \
                    in enumerate(zip(dataloader,dataloader1, dataloader2,dataloader3,dataloader4,dataloader5,dataloader6), 0):
                optimizer.zero_grad()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                optimizer_3.zero_grad()
                optimizer_4.zero_grad()
                optimizer_5.zero_grad()
                optimizer_6.zero_grad()
                inputs, data_id, dropout_mask = data_batch
                inputs1, data_id1, dropout_mask1 = data_batch1
                inputs2, data_id2, dropout_mask2 = data_batch2
                inputs3, data_id3, dropout_mask3 = data_batch3
                inputs4, data_id4, dropout_mask4 = data_batch4
                inputs5, data_id5, dropout_mask5 = data_batch5
                inputs6, data_id6, dropout_mask6 = data_batch6
                inputs = Variable(inputs.type(Tensor))
                inputs1 = Variable(inputs1.type(Tensor))
                inputs2 = Variable(inputs2.type(Tensor))
                inputs3 = Variable(inputs3.type(Tensor))
                inputs4 = Variable(inputs4.type(Tensor))
                inputs5 = Variable(inputs5.type(Tensor))
                inputs6 = Variable(inputs6.type(Tensor))
                data_ids.append(data_id.cpu().detach().numpy())
                data_ids1.append(data_id1.cpu().detach().numpy())
                data_ids2.append(data_id2.cpu().detach().numpy())
                data_ids3.append(data_id3.cpu().detach().numpy())
                data_ids4.append(data_id4.cpu().detach().numpy())
                data_ids5.append(data_id5.cpu().detach().numpy())
                data_ids6.append(data_id6.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=None,temperature=temperature, opt=opt)
                loss1, loss_rec1, loss_gauss1, loss_cat1, dec1, y1, hidden1 = vae1(inputs1, dropout_mask=None,temperature=temperature, opt=opt)
                loss2, loss_rec2, loss_gauss2, loss_cat2, dec2, y2, hidden2 = vae2(inputs2, dropout_mask=None,temperature=temperature, opt=opt)
                loss3, loss_rec3, loss_gauss3, loss_cat3, dec3, y3, hidden3 = vae3(inputs3, dropout_mask=None,temperature=temperature, opt=opt)
                loss4, loss_rec4, loss_gauss4, loss_cat4, dec4, y4, hidden4 = vae4(inputs4, dropout_mask=None,temperature=temperature, opt=opt)
                loss5, loss_rec5, loss_gauss5, loss_cat5, dec5, y5, hidden5 = vae5(inputs5, dropout_mask=None, temperature=temperature, opt=opt)
                loss6, loss_rec6, loss_gauss6, loss_cat6, dec6, y6, hidden6 = vae6(inputs6, dropout_mask=None,temperature=temperature, opt=opt)
                sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
                sparse_loss1 = opt.alpha * torch.mean(torch.abs(vae1.adj_A))
                w1_2 = torch.load('data_mHSC/data/w/w_stage2.pt')
                w_stage1_2_loss = opt.l * torch.mean(torch.abs(vae1.adj_A - w1_2))
                sparse_loss2 = opt.alpha * torch.mean(torch.abs(vae2.adj_A))
                w2_1 = torch.load('data_mHSC/data/w/w_stage1.pt')
                w_stage2_1_loss = opt.l * torch.mean(torch.abs(vae2.adj_A - w2_1))
                w2_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
                w_stage2_3_loss = opt.l * torch.mean(torch.abs(vae2.adj_A - w2_3))
                sparse_loss3 = opt.alpha * torch.mean(torch.abs(vae3.adj_A))
                w3_2 = torch.load('data_mHSC/data/w/w_stage2.pt')
                w_stage3_2_loss = opt.l * torch.mean(torch.abs(vae3.adj_A - w3_2))
                w3_Lym = torch.load('data_mHSC/data/w/w_stage4_Lym.pt')
                w_stage3_Lym_loss = opt.l * torch.mean(torch.abs(vae3.adj_A - w3_Lym))
                w3_GM = torch.load('data_mHSC/data/w/w_stage4_GM.pt')
                w_stage3_GM_loss = opt.l * torch.mean(torch.abs(vae3.adj_A - w3_GM))
                w3_ME = torch.load('data_mHSC/data/w/w_stage4_ME.pt')
                w_stage3_ME_loss = opt.l * torch.mean(torch.abs(vae3.adj_A - w3_ME))
                sparse_loss4 = opt.alpha * torch.mean(torch.abs(vae4.adj_A))
                w4_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
                w_stage4_3_loss = opt.l * torch.mean(torch.abs(vae4.adj_A - w4_3))
                sparse_loss5 = opt.alpha * torch.mean(torch.abs(vae5.adj_A))
                w5_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
                w_stage5_3_loss = opt.l * torch.mean(torch.abs(vae5.adj_A - w5_3))
                sparse_loss6 = opt.alpha * torch.mean(torch.abs(vae6.adj_A))
                w6_3 = torch.load('data_mHSC/data/w/w_stage3.pt')
                w_stage6_3_loss = opt.l * torch.mean(torch.abs(vae6.adj_A - w6_3))
                loss = loss + sparse_loss
                loss1 = loss1 + sparse_loss1 + w_stage1_2_loss   ########
                loss2 = loss2 + sparse_loss2 + w_stage2_1_loss + w_stage2_3_loss
                loss3 = loss3 + sparse_loss3 + w_stage3_2_loss + w_stage3_Lym_loss + w_stage3_ME_loss + w_stage3_GM_loss
                loss4 = loss4 + sparse_loss4 + w_stage4_3_loss
                loss5 = loss5 + sparse_loss5 + w_stage5_3_loss
                loss6 = loss6 + sparse_loss6 + w_stage6_3_loss
                loss_sum = loss + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                loss_sum.backward()
                loss_rec_sum =loss_rec + loss_rec1 + loss_rec2 + loss_rec3 + loss_rec4 + loss_rec5 + loss_rec6
                mse_rec.append(loss_rec_sum.item())
                loss_all.append(loss_sum.item())
                loss_gauss_sum = loss_gauss + loss_gauss1+loss_gauss2+loss_gauss3+loss_gauss4+loss_gauss5+loss_gauss6
                loss_cat_sum = loss_cat + loss_cat1+loss_cat2+loss_cat3+loss_cat4+loss_cat5+loss_cat6
                loss_kl.append(loss_gauss_sum.item() + loss_cat_sum.item())
                sparse_loss_sum = sparse_loss + sparse_loss1 + sparse_loss2 + sparse_loss3 + sparse_loss4 + sparse_loss5 + sparse_loss6
                loss_sparse.append(sparse_loss_sum.item())
                loss_w_stage.append(w_stage1_2_loss.item() + w_stage2_1_loss.item() + w_stage2_3_loss.item() + w_stage3_2_loss.item() +
                                    w_stage3_Lym_loss.item() + w_stage3_ME_loss.item() + w_stage3_GM_loss.item() +
                                    w_stage4_3_loss.item() + w_stage5_3_loss.item() + w_stage6_3_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1:
                    optimizer.step()
                    optimizer_1.step()
                    optimizer_2.step()
                    optimizer_3.step()
                    optimizer_4.step()
                    optimizer_5.step()
                    optimizer_6.step()
                else:
                    optimizer2.step()
                    optimizer2_1.step()
                    optimizer2_2.step()
                    optimizer2_3.step()
                    optimizer2_4.step()
                    optimizer2_5.step()
                    optimizer2_6.step()
                    optimizer_stage1_1.step()
                    optimizer_stage2_1.step()
                    optimizer_stage2_2.step()
                    optimizer_stage3_1.step()
                    optimizer_stage3_2.step()
                    optimizer_stage3_3.step()
                    optimizer_stage3_4.step()
                    optimizer_stage4_1.step()
                    optimizer_stage5_1.step()
                    optimizer_stage6_1.step()
            scheduler.step()
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()
            scheduler5.step()
            scheduler6.step()
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                vae_adj_A = vae1.adj_A + vae2.adj_A + vae3.adj_A + vae4.adj_A + vae5.adj_A + vae6.adj_A)/6
                Ep, Epr = evaluate(vae_adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                best_Epr = max(Epr,best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse),"SSC_loss:",np.mean(loss_w_stage))
        extractEdgesFromMatrix(vae_adj_A.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
            opt.save_name + '/result.tsv', sep='\t', index=False)
model = CT_VaeSSC_model(opt)
model.train_model()
