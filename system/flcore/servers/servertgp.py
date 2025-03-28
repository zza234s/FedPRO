import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import numpy as np
from utils.proto_rag import *
import re
class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            TGP = Trainable_Global_Prototypes(
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
            save_item(TGP, self.role, 'TGP', self.save_folder_name)
            print(TGP)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

        self.global_prototype = None

    # def prototype_plug_train(self):
    #     s_t = time.time()
    #     self.selected_clients = self.select_clients()
    #     # self.send_models()
    #     super().receive_protos()
    #     self.build_prototype_db()
    #     self.send_prototype_db()

    #     for i in range(self.adaptation_rounds):
    #         print(f"\n-------------Plus Round number: {i}-------------")
    #         self.RAF_evaluate()
    #         self.receive_proto_feedback()
    #         self.aggregate_feedback()
    #         self.update_prototype_weights()
    #         # self.update_proto_db()
    #         self.send_prototype_db()

    #         if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
    #             break
    #     self.save_results()
    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.update_TGP()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

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

            if self.global_prototype is not None:
                save_path = os.path.join(self.save_folder_name, f"best_global_prototype_{best_acc_str}.pt")
                torch.save(self.global_prototype, save_path)
                print(f"Save the best global prototype at: {save_path}")
    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

    def update_TGP(self):
        TGP = load_item(self.role, 'TGP', self.save_folder_name)
        TGP_opt = torch.optim.SGD(TGP.parameters(), lr=self.server_learning_rate)
        TGP.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = TGP(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)

                TGP_opt.zero_grad()
                loss.backward()
                TGP_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        save_item(TGP, self.role, 'TGP', self.save_folder_name)

        TGP.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = TGP(torch.tensor(class_id, device=self.device)).detach()
        self.global_prototype = global_protos
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

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

            global_proto_path = os.path.join(file_path, f"best_global_prototype_{best_acc_str}.pt")
            c.global_protos = torch.load(global_proto_path,weights_only=False)
            c.glbal_proto_for_check = copy.deepcopy(c.global_protos)
            # file_types = ["GCE", "GCE_frozen", "CoV", "generic_conditional_input",
            #               "personalized_conditional_input"]
            # for file_type in file_types:
            #     path = os.path.join(file_path, f"Client_{c.id}_{file_type}_{best_acc_str}.pt")
            #     if os.path.exists(path):
            #         setattr(c, file_type, torch.load(path,weights_only=False))
            #     else:
            #         print(f"Warning: {path} not found!")

        print(f'最佳权重文件读取完成, 开始评估')
        self.evaluate()
        print('-' * 50)


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out

