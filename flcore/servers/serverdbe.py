# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientdbe import clientDBE
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import os
import torch
import copy
import torch.nn as nn
from flcore.trainmodel.models import *
import re
from utils.proto_rag import *
class FedDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # initialization period
        self.set_clients(clientDBE)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train() # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if self.save_best_global_model and test_acc > self.best_metrics['test_acc']:
            self.best_metrics['test_acc'] = test_acc
            best_acc_str = f"{test_acc:.4f}"
            for c in self.clients:
                c.save_best_model(best_acc_str=best_acc_str)

    def load_best_pesonlized_model(self):
        algo_path = os.path.join(self.args.best_model_dir, self.dataset, self.algorithm)  # TODO:删除time编号
        timestamp_folder_path  = get_timestamp_folder(algo_path)
        file_path = os.path.join(algo_path, timestamp_folder_path)
        assert os.path.exists(file_path), f"Model path {file_path} does not exist!"

        best_acc_per_client = {}

        # 解析所有文件名，找到每个客户端的最佳精度
        for filename in os.listdir(file_path):
            match = re.search(r'Client_(\d+)_.*_(\d+\.\d+).pt', filename)
            if match:
                client_id, acc = int(match.group(1)), float(match.group(2))
                if client_id not in best_acc_per_client or acc > best_acc_per_client[client_id]:
                    best_acc_per_client[client_id] = acc

        print("最佳精度记录:", best_acc_per_client)


        for c in self.clients:
            best_acc_str = f"{best_acc_per_client[c.id]:.4f}"

            model_file_path = os.path.join(file_path, f"Client_{c.id}_best_model_{best_acc_str}.pt")
            state_dict = torch.load(model_file_path)
            c.model.load_state_dict(state_dict)

            # 强制把model拆分为表征层和决策层
            if not isinstance(c.model,BaseHeadSplit):
                head = copy.deepcopy(c.model.fc)
                c.model.fc = nn.Identity()
                c.model = BaseHeadSplit(c.model, head)

            client_mean_path = os.path.join(file_path, f"Client_{c.id}_client_mean_{best_acc_str}.pt")
            client_mean = torch.load(client_mean_path)
            c.client_mean = client_mean

        # model_path = os.path.join("autodl", self.dataset)
        # model_path = os.path.join(model_path, self.algorithm)
        # assert (os.path.exists(model_path))
        # for c in self.clients:
        #     model_file_path = os.path.join(model_path, f"client_{c.id}_best_model_{best_acc_str}.pt")
        #     state_dict = torch.load(model_file_path)
        #     c.model.load_state_dict(state_dict)
        #
        #     client_mean_path = os.path.join(model_path, f"client_{c.id}_client_mean_{best_acc_str}.pt")
        #     client_mean = torch.load(client_mean_path)
        #     c.client_mean = client_mean
        #     # # 强制把model拆分为表征层和决策层
        #     # if not isinstance(c.model,BaseHeadSplit):
        #     #     head = copy.deepcopy(c.model.fc)
        #     #     c.model.fc = nn.Identity()
        #     #     c.model = BaseHeadSplit(c.model, head)
        print(f'模型读取完成,开始评估')
        self.evaluate()
        print('-'*50)
