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

import copy
import torch
import torch.nn as nn
import time
from flcore.clients.clientgpfl import *
from flcore.servers.serverbase import Server
from threading import Thread
import re
from utils.proto_rag import get_timestamp_folder


class GPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        args.GCE = GCE(in_features=self.feature_dim,
                       num_classes=args.num_classes,
                       dev=args.device).to(args.device)
        args.CoV = CoV(self.feature_dim).to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGPFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate performance")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            self.global_GCE()
            self.global_CoV()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # train_acc = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
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
        # print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if self.save_best_global_model and test_acc > self.best_metrics['test_acc']:
            self.best_metrics['test_acc'] = test_acc
            best_acc_str = f"{test_acc:.4f}"
            for c in self.clients:
                c.save_best_model(best_acc_str=best_acc_str)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)

    def global_GCE(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.GCE)

        self.GCE = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.GCE.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_GCE(w, client_model)

        for client in self.clients:
            client.set_GCE(self.GCE)

    def add_GCE(self, w, GCE):
        for server_param, client_param in zip(self.GCE.parameters(), GCE.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_CoV(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.CoV)

        self.CoV = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.CoV.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_CoV(w, client_model)

        for client in self.clients:
            client.set_CoV(self.CoV)

    def add_CoV(self, w, CoV):
        for server_param, client_param in zip(self.CoV.parameters(), CoV.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_best_model(self, best_acc_str):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)

        model_path = os.path.join(self.save_folder_name, f"{self.role}_best_model_{best_acc_str}.pt")
        client_mean_path = os.path.join(self.save_folder_name, f"{self.role}_client_mean_{best_acc_str}.pt")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.client_mean, client_mean_path)
        print(f"Save the FedDBE's best local model files at: {self.save_folder_name}")

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

        # 加载最佳精度对应的文件
        for c in self.clients:
            best_acc_str = f"{best_acc_per_client[c.id]:.4f}"

            model_file_path = os.path.join(file_path, f"Client_{c.id}_best_model_{best_acc_str}.pt")
            state_dict = torch.load(model_file_path)
            c.model.load_state_dict(state_dict)

            file_types = ["GCE", "GCE_frozen", "CoV", "generic_conditional_input",
                          "personalized_conditional_input"]
            for file_type in file_types:
                path = os.path.join(file_path, f"Client_{c.id}_{file_type}_{best_acc_str}.pt")
                if os.path.exists(path):
                    setattr(c, file_type, torch.load(path,weights_only=False))
                else:
                    print(f"Warning: {path} not found!")

        print(f'最佳权重文件读取完成, 开始评估')
        self.evaluate()
        print('-' * 50)


class GCE(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class CoV(nn.Module):
    def __init__(self, in_dim):
        super(CoV, self).__init__()

        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)

        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out
