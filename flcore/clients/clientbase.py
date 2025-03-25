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
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from collections import defaultdict
from flcore.trainmodel.models import BaseHeadSplit
from utils.finch import FINCH
import torch.nn.functional as F
import time
from utils.proto_rag import weighted_voting
from utils.vis_utils import *
from torch.optim import Adam
import numpy as np
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from utils.proto_con_loss import ProtoConloss
import time


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        if args.load_best_local_model:
            path = os.path.join(args.save_folder_without_time, args.save_model_time,
                                f"Client_{id}_best_model_{args.load_acc_str}.pt")
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)
            print(f"client_{self.id} load model from {path}")

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.inference_cost = []

        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.use_rag = args.use_RAG
        self.prototype_db = defaultdict(list)
        self.vis_prototype = args.vis_prototype
        self.test_cnt = 0  # TODO: 这里用来临时记录客户端测试次数，用于测试时的可视化
        self.topk = args.topk
        self.feedback_stats = defaultdict(lambda: {
            "total_counts": 0,
            "error_counts": 0,
            "correct_counts": 0
        })
        self.cluster_method = args.cluster_method
        self.cluster_k = args.cluster_k
        # if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
        #     model = BaseHeadSplit(args, self.id).to(self.device)
        #     save_item(model, self.role, 'model', self.save_folder_name)
        self.proto_op_mode = args.proto_op_mode
        self.proto_tempure = args.proto_tempure
        self.one_class_flag =False

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        start_time = time.time()
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        end_time = time.time()
        print(f"client#{self.id} test time cost: {end_time - start_time:.4f}s")
        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def save_best_model(self, best_acc_str):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        save_path = os.path.join(self.save_folder_name, f"{self.role}_best_model_{best_acc_str}.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f'Save the best local model at: {save_path}')

    @torch.no_grad()
    def local_cluster_for_proto_gen(self):
        trainloader = self.load_train_data()
        self.model.eval()

        reps_dict = defaultdict(list)
        agg_local_protos = defaultdict(list)

        all_reps = []
        all_labels = []
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                owned_classes = y.unique()
                all_reps.append(rep)
                all_labels.append(y)
                for cls in owned_classes:
                    filted_reps = rep[y == cls].detach().clone()
                    reps_dict[cls.item()].append(filted_reps)

        all_reps = torch.cat(all_reps, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        self.save_item(all_reps, f'all_train_reps')
        self.save_item(all_labels, f'all_train_labels')

        # 输出未见类信息
        unseen_classes = set(range(self.num_classes)) - set(reps_dict.keys())
        if unseen_classes:
            print(f"Class(es) {unseen_classes} not in client {self.id}'s local training dataset")

        self.train_reps_dict = reps_dict

        # 选择聚类方法
        for cls, protos in reps_dict.items():
            protos_np = torch.cat(protos).detach().cpu().numpy()

            agg_selected_proto = []

            if self.cluster_method == "finch":
                # 使用 FINCH 进行聚类
                c, num_clust, req_c = FINCH(protos_np, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False,
                                            verbose=False)  # ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
                cluster_labels = c[:, -1]  # 取最后一级聚类结果

            elif self.cluster_method == "kmeans":
                # K-Means 聚类
                k = self.cluster_k if hasattr(self, 'cluster_k') and self.cluster_k > 0 else min(20, protos_np.shape[0])
                k = 1 if k > len(protos) else k
                cluster_labels = KMeans(n_clusters=k, random_state=42).fit_predict(protos_np)

            elif self.cluster_method == "dbscan":
                # DBSCAN 聚类
                eps = getattr(self, 'dbscan_eps', 0.5)  # 默认 eps=0.5
                min_samples = getattr(self, 'dbscan_min_samples', 5)  # 默认 min_samples=5
                cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(protos_np)

            elif self.cluster_method == "gmm":
                # 高斯混合模型（GMM）聚类
                k = self.cluster_k if hasattr(self, 'cluster_k') and self.cluster_k > 0 else min(20, protos_np.shape[0])
                k = 1 if k > len(protos) else k
                cluster_labels = GaussianMixture(n_components=k, random_state=42).fit_predict(protos_np)

            else:
                raise ValueError(f"Unknown clustering method: {self.cluster_method}")

            unique_clusters = np.unique(cluster_labels)

            for cluster_index in unique_clusters:
                selected_array = np.where(cluster_labels == cluster_index)
                selected_proto_list = protos_np[selected_array]

                if len(selected_proto_list) == 0:
                    continue  # 避免计算空数据

                proto = np.mean(selected_proto_list, axis=0, keepdims=True)
                agg_selected_proto.append(torch.tensor(proto, device=self.device))

            agg_local_protos[cls] = agg_selected_proto

        # # 确保所有类别都存在默认值
        # if agg_local_protos:
        #     default_shape = next(iter(agg_local_protos.values()))[0].shape[0]
        #     for label in range(self.num_classes):
        #         if label not in agg_local_protos:
        #             agg_local_protos[label] = [torch.zeros(default_shape, device=self.device)]

        # 确保字典按 key 排序
        agg_local_protos = dict(sorted(agg_local_protos.items()))
        self.protos = agg_local_protos

    # @torch.no_grad()
    # def local_cluster_for_proto_gen(self):
    #     trainloader = self.load_train_data()
    #     self.model.eval()
    #
    #     reps_dict = defaultdict(list)
    #     agg_local_protos = defaultdict(list)
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(trainloader):
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #
    #             rep = self.model.base(x)
    #
    #             owned_classes = y.unique()
    #             for cls in owned_classes:
    #                 filted_reps = rep[y == cls].detach()
    #                 reps_dict[cls.item()].append(filted_reps.detach().clone())
    #
    #     # 输出未见类信息
    #     unseen_classes = set(range(self.num_classes)) - set(reps_dict.keys())
    #     if unseen_classes != set():
    #         print(f"class(es) {unseen_classes} not in the client{self.id}'s  local training dataset")
    #
    #     # 聚类生成原型
    #     for cls, protos in reps_dict.items():
    #         protos_np = torch.cat(protos).detach().cpu().numpy()
    #         c, num_clust, req_c = FINCH(protos_np, initial_rank=None, req_clust=None, distance='cosine',
    #                                     ensure_early_exit=False,
    #                                     verbose=False)  # ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    #         m, n = c.shape
    #         class_cluster_list = []
    #         for index in range(m):
    #             class_cluster_list.append(c[index, -1])
    #
    #         class_cluster_array = np.array(class_cluster_list)
    #         uniqure_cluster = np.unique(class_cluster_array).tolist()
    #         agg_selected_proto = []
    #
    #         for _, cluster_index in enumerate(uniqure_cluster):
    #             selected_array = np.where(class_cluster_array == cluster_index)
    #             selected_proto_list = protos_np[selected_array]
    #             proto = np.mean(selected_proto_list, axis=0, keepdims=True)
    #
    #             agg_selected_proto.append(torch.tensor(proto))
    #         agg_local_protos[cls] = agg_selected_proto
    #
    #         # mean_proto = torch.cat(protos).mean(dim=0)
    #         # agg_local_protos[cls] = mean_proto
    #
    #     # for label in range(self.num_classes):
    #     #     if label not in agg_local_protos:
    #     #         agg_local_protos[label] = torch.zeros(list(agg_local_protos.values())[0].shape[0],
    #     #                                               device=self.device)
    #
    #     agg_local_protos = dict(sorted(agg_local_protos.items()))  # 确保上传的dict按key升序排序
    #     self.protos = agg_local_protos

    # def refine_proto_db(self, epochs=50):
    #     local_proto_ids = [(idx, meta['class']) for idx, meta in self.proto_meta.items()
    #                        if meta['CLIENT_ID'] == self.id]
    #     local_proto_ids = torch.tensor(local_proto_ids)
    #
    #     self.protoConLoss = ProtoConloss(temperature=self.proto_tempure)
    #
    #     self.margin = 0.1
    #
    #     # 创建可优化的原型参数
    #     P = torch.nn.Parameter(
    #         torch.tensor(
    #             copy.deepcopy(self.proto_db[local_proto_ids[:, 0]]),
    #             device=self.device,
    #             dtype=torch.float32
    #         )
    #     )
    #     proto_label = torch.tensor(local_proto_ids[:, 1], device=self.device)
    #     unique_classes = torch.unique(proto_label)
    #     class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    #
    #     if len(unique_classes) == 1:
    #         print(f"client#{self.id} 只有{len(unique_classes)}个类，暂不支持更新prototype")
    #         agg_local_protos = defaultdict(list)
    #         for cls, protos in zip(proto_label, P.clone().detach()):
    #             agg_local_protos[cls.item()].append(protos)
    #
    #         self.protos = agg_local_protos
    #         return
    #
    #     optimizer = Adam([P], lr=0.01)
    #     self.model.eval()
    #     loader = self.load_train_data()
    #
    #     # 预计算所有训练样本的特征表示
    #     all_features, all_labels = [], []
    #     with torch.no_grad():
    #         for x, y in loader:
    #             x = x.to(self.device)
    #             rep = self.model.base(x)
    #             all_features.append(rep)
    #             all_labels.append(y)
    #
    #     all_features = torch.cat(all_features, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     feature_loader = DataLoader(TensorDataset(all_features, all_labels), batch_size=1024, shuffle=True)
    #
    #     best_accuracy, no_improve_count = 0, 0
    #     best_prototypes = P.clone().detach()
    #     patience = 5
    #
    #     for epoch in range(epochs):
    #         total_loss, correct, total = 0, 0, 0
    #
    #         for features, labels in feature_loader:
    #             features, labels = features.to(self.device), labels.to(self.device)
    #
    #             # 计算样本表征与原型之间的余弦相似度
    #             sim_matrix = F.normalize(features, p=2, dim=1) @ F.normalize(P, p=2, dim=1).T
    #             pos_indices = torch.tensor([class_to_index[l.item()] for l in labels], device=self.device)
    #
    #
    #             if self.proto_op_mode == "infonce":
    #                 losses =self.protoConLoss(features, labels,  P, proto_label, len(unique_classes))
    #             else:
    #
    #                 # 计算类别的最大相似度
    #                 class_max_sims = torch.full((features.size(0), len(unique_classes)), -float('inf'),
    #                                             device=self.device)
    #                 for idx, cls in enumerate(unique_classes):
    #                     cls_mask = (proto_label == cls)
    #                     if cls_mask.any():
    #                         class_max_sims[:, idx] = sim_matrix[:, cls_mask].max(dim=1).values
    #
    #                 pos_sims = class_max_sims[torch.arange(features.size(0)), pos_indices]
    #                 neg_sims = class_max_sims.clone()
    #                 neg_sims.masked_fill_(
    #                     torch.arange(features.size(0), device=self.device).unsqueeze(1) == pos_indices.unsqueeze(0),
    #                     -float('inf'))
    #                 # 计算最大负类相似度
    #                 neg_max = neg_sims.max(dim=1).values
    #
    #                 # 计算 Hinge Loss 变体
    #                 losses = F.relu(self.margin - (pos_sims - neg_max))
    #
    #             batch_loss = losses.mean()
    #             total_loss += batch_loss.item()
    #
    #             optimizer.zero_grad()
    #             batch_loss.backward()
    #             optimizer.step()
    #
    #             pred_label = proto_label[sim_matrix.argmax(dim=1)]
    #             correct += (pred_label == labels).sum().item()
    #             total += labels.size(0)
    #
    #         epoch_accuracy = correct / total
    #         print(f"client#{self.id}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Acc: {epoch_accuracy:.4f}")
    #
    #         if epoch_accuracy > best_accuracy:
    #             best_accuracy, best_prototypes, no_improve_count = epoch_accuracy, P.clone().detach(), 0
    #         else:
    #             no_improve_count += 1
    #
    #         if no_improve_count >= patience:
    #             print(f"连续{patience}轮无改善，提前停止")
    #             break
    #
    #     print(f"优化完成，最佳准确率: {best_accuracy:.4f}")
    #     agg_local_protos = defaultdict(list)
    #     for cls, protos in zip(proto_label, best_prototypes):
    #         agg_local_protos[cls.item()].append(protos)
    #     self.protos = agg_local_protos

    def refine_proto_db(self, epochs=50):
        local_proto_ids = [(idx, meta['class']) for idx, meta in self.proto_meta.items()
                           if meta['CLIENT_ID'] == self.id]
        local_proto_ids = np.array(local_proto_ids)
        self.margin = 0.1
        # 创建可优化的原型参数
        P = torch.nn.Parameter(
            torch.tensor(
                copy.deepcopy(self.proto_db[local_proto_ids[:, 0]]),
                device=self.device,
                dtype=torch.float32
            )
        )
        proto_label = torch.tensor(local_proto_ids[:, 1], device=self.device)
        unique_classes = torch.unique(proto_label).tolist()
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

        if len(unique_classes) == 1:
            print(f"client#{self.id} 只有{len(unique_classes)}个类，暂不支持更新prototype")
            agg_local_protos = defaultdict(list)
            self.one_class_flag =True
            for cls, protos in zip(proto_label, P.clone().detach()):
                agg_local_protos[cls.item()].append(protos)

            self.protos = agg_local_protos
            return
        # 设置优化器
        optimizer = Adam([P], lr=0.01)  # TODO: 优化器参数待调整
        self.model.eval()

        loader = self.load_train_data()

        # 预先计算所有训练样本的特征表示（避免重复计算）
        all_features = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                rep = self.model.base(x)
                all_features.append(rep.cpu())
                all_labels.append(y)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        feature_dataset = TensorDataset(all_features, all_labels)
        feature_loader = DataLoader(feature_dataset, batch_size=1024, shuffle=True)

        # visualize_with_dim_reduction(all_features, all_labels, 0, P, proto_label, mode='pca', title=f'client#{self.id}原型更新前')

        # 优化循环
        best_accuracy = 0
        best_prototypes = P.clone().detach()
        patience = 100  # 早停耐心值
        no_improve_count = 0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (features, labels) in enumerate(feature_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # 计算表征与原型之间的余弦相似度（批量计算）
                features_norm = F.normalize(features, p=2, dim=1)
                P_norm = F.normalize(P, p=2, dim=1)
                sim_matrix = torch.mm(features_norm, P_norm.t())

                # 向量化计算类别相似度
                batch_size = features.size(0)
                num_classes = len(unique_classes)
                class_max_sims = torch.full((batch_size, num_classes), -float('inf'), device=self.device)
                for idx, cls in enumerate(unique_classes):
                    cls_mask = (proto_label == cls)
                    if cls_mask.sum() > 0:
                        class_max_sims[:, idx], _ = torch.max(sim_matrix[:, cls_mask], dim=1)

                # 批量计算损失
                pos_indices = torch.tensor([class_to_index[int(l.item())] for l in labels], device=self.device)
                pos_sims = class_max_sims[torch.arange(batch_size), pos_indices]
                # 为每个样本屏蔽正类：将对应位置设置为 - inf，方便计算负类最大相似度
                neg_sims = class_max_sims.clone()
                neg_sims[torch.arange(batch_size), pos_indices] = -float('inf')
                neg_max, _ = torch.max(neg_sims, dim=1)

                losses = torch.clamp(self.margin - (pos_sims - neg_max), min=0)
                batch_loss = losses.mean()
                total_loss += batch_loss.item()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # 计算正确率（使用最相似原型的类别）
                pred_proto_idx = torch.argmax(sim_matrix, dim=1)
                pred_label = proto_label[pred_proto_idx]
                correct += (pred_label == labels).sum().item()
                total += len(labels)

            # 计算当前epoch的准确率
            epoch_accuracy = correct / total if total > 0 else 0
            print(
                f"client#{self.id},, Epoch {epoch + 1}/{epochs} 完成, Loss: {total_loss:.4f}, Acc: {epoch_accuracy:.4f}")

            # 使用训练准确率判断
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_prototypes = P.clone().detach()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience or epoch_accuracy==1.0:
                print(f"连续{patience}轮无改善，提前停止")
                break
        print(f"优化完成，最佳准确率: {best_accuracy:.4f}")

        # visualize_with_dim_reduction(all_features, all_labels, 0, P, proto_label, mode='pca', title=f'_client#{self.id}原型更新后')
        agg_local_protos = defaultdict(list)
        for cls, protos in zip(proto_label, best_prototypes):
            agg_local_protos[cls.item()].append(protos)

        self.protos = agg_local_protos

    @torch.no_grad()
    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        reps_dict = defaultdict(list)
        agg_local_protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                owned_classes = y.unique()

                for cls in owned_classes:
                    filted_reps = rep[y == cls].detach()
                    reps_dict[cls.item()].append(filted_reps.data)

            for cls, protos in reps_dict.items():
                mean_proto = torch.cat(protos).mean(dim=0)
                agg_local_protos[cls] = mean_proto

        # 对于未见类，生成全zero tensor上传server
        unseen_classes = set(range(self.num_classes)) - set(agg_local_protos.keys())
        if unseen_classes != set():
            print(f"class(es) {unseen_classes} not in the client{self.id}'s  local training dataset")

        agg_local_protos = dict(sorted(agg_local_protos.items()))  # 确保上传的dict按key升序排序
        self.local_mean_protos = agg_local_protos

    def RAF(self, mode='train'):
        ensemble_acc = 0
        sample_num = 0

        correct_prototype = 0
        correct_local = 0
        correct_rag = 0
        all_features = []
        # all_retrieved_prototypes = []
        all_retrieved_meta = []

        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []

        res_dict = dict()

        if mode == 'train':
            loader = self.load_train_data()
        else:
            loader = self.load_test_data()

        self.model.eval()

        self.proto_db = self.proto_db.to(self.device)

        for idx, meta in self.proto_meta.items():
            self.proto_meta[idx]['weight'] = 1.0

        start_time = time.time()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                # proto_sim = self.cal_proto_similarity(rep)
                # rag_output = self.weighted_voting_by_sim(self.proto_db, self.proto_meta, proto_sim, self.num_classes)

                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                    rep, topk=self.topk)  # (B, top_k, embed_dim)
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes,
                                             retrievaled_similarities)

                local_model_output = self.model.head(rep)
                # rag_output=torch.softmax(rag_output,dim=1)

                proto_conf, _ = torch.max(retrievaled_similarities, dim=1)
                # proto_conf = torch.max(rag_output, dim=1).values

                alpha = (1.0 - proto_conf).unsqueeze(1)
                ensemble_output = (1 - alpha) * rag_output + alpha * local_model_output
                # local_softmax = F.softmax(local_model_output, dim=1)
                # local_conf = torch.max(local_softmax, dim=1).values
                # alpha = (1.0 - local_conf).unsqueeze(1)

                # ensemble_output = (1 - alpha) * local_model_output + alpha * rag_output
                # ensemble_output = local_model_output + rag_output

                correct_local += (torch.sum(torch.argmax(local_model_output, dim=1) == y)).item()
                correct_rag += (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
                ensemble_acc += (torch.sum(torch.argmax(ensemble_output, dim=1) == y)).item()
                sample_num += y.shape[0]

                y_prob.append(ensemble_output.detach().cpu().numpy())
                nc = self.num_classes
                # if self.num_classes == 2:
                #     nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                # if self.num_classes == 2:
                #     lb = lb[:, :2]
                y_true.append(lb)

                y_pred.extend(np.argmax(ensemble_output.detach().cpu().numpy(), axis=1).tolist())
                y_true_list.extend(y.tolist())

                # if self.vis_prototype and retrieve_prototypes is not None:
                #     all_features.append(rep)
                #     # all_retrieved_prototypes.append(retrieve_prototypes)
                #     all_retrieved_meta.extend(retrieve_meta_info)

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"client:{self.id}, RAF time cost: {inference_time:.4f}s")
        self.inference_cost.append(inference_time)

        # if self.vis_prototype and retrieve_prototypes is not None and self.id == 0:
        #     all_features = torch.cat(all_features, dim=0)
        #     # all_retrieved_prototypes = torch.cat(all_retrieved_prototypes, dim=0).squeeze()
        #     for mode in ['pca', 'tsne']:
        #         visualize_with_dim_reduction(
        #             all_features,
        #             y_true_list,
        #             0,
        #             self.proto_db,
        #             self.proto_meta,
        #             all_retrieved_meta,
        #             mode=mode,
        #             title=f'{self.role}, query_idx: {0}, test_cnt: {self.test_cnt}'
        #         )
        # visualize_tsne_with_class(all_features,
        #                           y_true_list,
        #                           0,
        #                           all_protos_tensor,
        #                           all_protos_meta,
        #                           all_retrieved_meta,
        #                           title=f'{self.role}, query_idx: {0}, test_cnt: {self.test_cnt}')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        # 存储结果至字典
        res_dict['ensemble_acc'] = ensemble_acc
        res_dict['sample_num'] = sample_num
        res_dict['auc'] = auc
        res_dict['correct_local'] = correct_local
        res_dict['correct_prototype'] = correct_prototype
        res_dict['correct_rag'] = correct_rag
        return res_dict

    def cal_proto_similarity(self, query_features, temperature=1.0, distance='cosine'):
        query_features = F.normalize(query_features, p=2, dim=1)  # (B, D)
        all_protos_tensor = F.normalize(self.proto_db, p=2, dim=1)  # (N, D)
        if distance == 'cosine':
            sims = torch.mm(query_features, all_protos_tensor.t())
        elif distance == 'rbf':
            sigma = 1.0
            diff = query_features.unsqueeze(1) - all_protos_tensor.unsqueeze(0)  # (B, N, D)
            d2 = torch.sum(diff ** 2, dim=-1)  # (B, N)
            sims = torch.exp(-d2 / (2 * sigma ** 2))
        else:
            raise ValueError(f"Invalid distance metric: {distance}")

        return sims

    # def weighted_retrieve_prototypes(self, query_features, topk=5, temperature=1.0, distance='cosine'):
    #     all_protos_tensor = self.proto_db.to(self.device)  # 存放所有原型的张量
    #     all_meta = self.proto_meta  # 存放每个原型对应的类别信息
    #
    #     # weights = torch.tensor([meta['weight'] for idx, meta in all_meta.items()]).to(self.device)  # (N,)
    #
    #     query_features = F.normalize(query_features, p=2, dim=1)  # (B, D)
    #     all_protos_tensor = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
    #     sims = torch.mm(query_features, all_protos_tensor.t())
    #
    #
    #     # 获取topk结果
    #     retrievaled_similarities, indices = torch.topk(sims, k=topk, dim=1)
    #
    #     prototypes_retrieved = all_protos_tensor[indices]  # (B, top_k, embed_dim)
    #     # indices 是 tensor，因此转换为 numpy 数组以便索引列表 all_meta
    #     meta_retrieved = []
    #     indices_np = indices.cpu().numpy()  # ndarray: shape (B, topk)
    #     for idx_list in indices_np:
    #         meta_retrieved.append([all_meta[i] for i in idx_list])
    #
    #     return prototypes_retrieved, meta_retrieved, indices_np, retrievaled_similarities

    # 计算通讯开销前用
    def weighted_retrieve_prototypes(self, query_features, topk=5, temperature=1.0, distance='cosine'):
        all_protos_tensor = self.proto_db.to(self.device)  # 存放所有原型的张量0.
        all_meta = self.proto_meta  # 存放每个原型对应的类别信息

        weights = torch.tensor([meta['weight'] for idx, meta in all_meta.items()]).to(self.device)  # (N,)

        # query_features = F.normalize(query_features, p=2, dim=1)  # (B, D)
        # all_protos_tensor = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
        # sims = torch.mm(query_features, all_protos_tensor.t())
        if distance == 'cosine':
            # 对向量进行L2归一化后计算余弦相似度
            query_norm = F.normalize(query_features, p=2, dim=1)  # (B, D)
            proto_norm = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
            sims = torch.mm(query_norm, proto_norm.t())
        elif distance == 'euclidean':
            # 计算欧式距离，距离越小相似度越高，故取负数
            sims = - torch.cdist(query_features, all_protos_tensor, p=2)
        elif distance == 'manhattan':
            # 计算曼哈顿距离，取负数
            sims = - torch.cdist(query_features, all_protos_tensor, p=1)
        elif distance == 'chebyshev':
            # 计算切比雪夫距离（各维度差值的最大值），取负数
            sims = - torch.cdist(query_features, all_protos_tensor, p=float('inf'))
        elif distance == 'rbf':
            sigma = 1.0
            diff = query_features.unsqueeze(1) - all_protos_tensor.unsqueeze(0)  # (B, N, D)
            d2 = torch.sum(diff ** 2, dim=-1)  # (B, N)
            # 应用高斯核公式
            sims = torch.exp(-d2 / (2 * sigma ** 2))
        else:
            raise ValueError(
                "不支持的距离指标：{}，请使用 'cosine'、'euclidean'、'manhattan' 或 'chebyshev'".format(distance)
            )

        # if distance == 'cosine':
        #     sims = torch.mm(query_features, all_protos_tensor.t())
        # elif distance == 'rbf':
        #     sigma = 1.0
        #     diff = query_features.unsqueeze(1) - all_protos_tensor.unsqueeze(0)  # (B, N, D)
        #     d2 = torch.sum(diff ** 2, dim=-1)  # (B, N)
        #     # 应用高斯核公式
        #     sims = torch.exp(-d2 / (2 * sigma ** 2))
        # else:
        #     raise ValueError(f"Invalid distance metric: {distance}")
        zero_weight_mask = (weights == 0)

        # 将原始相似度按温度缩放
        sims_scaled = sims / temperature

        # 应用权重掩码，将权重为0的原型对应的相似度设为一个非常小的值
        # 使用torch.finfo(sims.dtype).min可能会导致数值问题，所以使用一个足够小的负数
        min_value = -1e9  # 一个足够小的负数，确保不会被选中
        sims_masked = torch.where(zero_weight_mask.unsqueeze(0),
                                  torch.tensor(min_value, device=self.device),
                                  sims_scaled)

        # 应用非零权重
        # non_zero_weights = torch.where(zero_weight_mask,
        #                                torch.tensor(1.0, device=self.device),
        #                                weights)
        # sims_weighted = non_zero_weights * sims_masked
        sims_weighted = sims_masked

        # 获取topk结果
        retrievaled_similarities, indices = torch.topk(sims_weighted, k=topk, dim=1)

        prototypes_retrieved = all_protos_tensor[indices]  # (B, top_k, embed_dim)
        # indices 是 tensor，因此转换为 numpy 数组以便索引列表 all_meta
        meta_retrieved = []
        indices_np = indices.cpu().numpy()  # ndarray: shape (B, topk)
        for idx_list in indices_np:
            meta_retrieved.append([all_meta[i] for i in idx_list])

        return prototypes_retrieved, meta_retrieved, indices_np, retrievaled_similarities

    def non_trainable_prototype_retrieval(self, query_features, topk=5, temperature=0.1):
        # 1. 获取原型库与对应元信息，并转移到设备上
        proto_features = self.proto_db.to(self.device)  # (N, D)
        all_meta = self.proto_meta  # 存放每个原型的元信息
        # 获取每个原型的先验权重
        weights = torch.tensor([meta['weight'] for meta in all_meta]).to(self.device)  # (N,)

        # 2. 对查询特征和原型进行 L2 归一化
        query_norm = F.normalize(query_features, p=2, dim=1)  # (B, D)
        proto_norm = F.normalize(proto_features, p=2, dim=1)  # (N, D)

        # 3. 计算余弦相似度
        sims = torch.mm(query_norm, proto_norm.t())  # (B, N)

        # 4. 融合先验权重（直接逐元素相乘）
        sims = sims * weights  # (B, N)

        # 5. 温度缩放
        sims_scaled = sims / temperature  # 调整相似度分布的平滑性

        # 6. 采用 softmax 得到注意力权重（软选择方案）
        attention_weights = F.softmax(sims_scaled, dim=1)  # (B, N)
        aggregated_proto = torch.matmul(attention_weights, proto_features)  # (B, D)

        # 7. 同时保留 top-k 的硬选择结果
        _, indices = torch.topk(sims, k=topk, dim=1)
        prototypes_retrieved = proto_features[indices]  # (B, topk, D)

        # 将对应的 meta 信息提取出来
        meta_retrieved = []
        indices_np = indices.cpu().numpy()  # ndarray: (B, topk)
        for idx_list in indices_np:
            meta_retrieved.append([all_meta[i] for i in idx_list])

        return prototypes_retrieved, meta_retrieved, indices_np, aggregated_proto

    def get_proto_feedback(self, mode='train', topk=None, mask_other_client_proto=False, visualize=False,
                           phase="before"):
        """
        计算每个类别和每个原型的统计信息，包括：
        1. 每个类别的总预测次数、正确/错误分类次数
        2. 每个类别选中的原型分布
        3. 每个原型的被选次数、正确/错误贡献

        参数：
            mode: 'train' 或 'test'，决定使用训练数据还是测试数据
            topk: 可选参数，指定要检索的 top-k 原型（默认为 self.topk）
            mask_other_client_proto: 是否遮蔽其他客户端的原型
            visualize: 是否可视化t-SNE结果

        返回：
            feedback_stats: 按类别统计的预测情况
            proto_stats: 按原型统计的使用情况
        """
        if mask_other_client_proto:
            for idx, meta in self.proto_meta.items():
                if meta['CLIENT_ID'] != self.id:
                    self.proto_meta[idx]['weight'] = 0
        if topk is None:
            topk = self.topk  # 使用默认的 top-k 值

        self.feedback_stats = defaultdict(lambda: {
            "total_counts": 0,  # 该类别的总预测次数
            "error_counts": 0,  # 该类别的错误预测次数
            "correct_counts": 0,  # 该类别的正确预测次数
            "accuracy": 0.0,  # 该类别的预测准确率
            "selected_by_class": defaultdict(int)  # 记录该类别选中的原型信息
        })

        self.proto_stats = defaultdict(lambda: {
            "total_counts": 0,  # 该原型被选中的总次数
            "error_counts": 0,  # 该原型导致错误分类的次数
            "correct_counts": 0,  # 该原型导致正确分类的次数
            "accuracy": 0.0  # 该原型的正确率
        })

        self.model.eval()

        local_correct = 0
        rag_correct = 0
        sample_num = 0
        not_load_rep = True
        all_rag_preds = []


        all_embeddings = self.load_item(f'all_{mode}_reps')
        all_labels = self.load_item(f'all_{mode}_labels')
        if all_embeddings is not None and all_labels is not None:
            loader = DataLoader(TensorDataset(all_embeddings, all_labels), batch_size=1024, shuffle=False)
            not_load_rep= False
        else:
            # 选择数据加载器
            loader = self.load_train_data() if mode == 'train' else self.load_test_data()
            all_embeddings= []
            all_labels = []

        # # 用于t-SNE可视化的数据收集
        # if visualize:
        #     all_embeddings = []
        #     all_labels = []
        #     all_rag_preds = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # 计算样本表征
                if not_load_rep:
                    rep = self.model.base(x)
                else:
                    rep = x



                # 检索最近的原型
                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                    rep, topk=1)  # (B, top_k, embed_dim)

                # 使用加权投票进行分类
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes,
                                             retrievaled_similarities)

                local_output = self.model.head(rep)

                local_correct += (torch.sum(torch.argmax(local_output, dim=1) == y)).item()
                rag_correct += (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
                sample_num += y.shape[0]

                rag_preds = rag_output.argmax(dim=1)

                all_rag_preds.append(rag_preds.cpu())

                if not_load_rep:
                    all_embeddings.append(rep.cpu())
                    all_labels.append(y.cpu())


                # # 收集可视化数据
                # if visualize:
                #     all_embeddings.append(rep.cpu())
                #     all_labels.append(y.cpu())
                #     all_rag_preds.append(rag_preds.cpu())

                for i in range(x.shape[0]):  # 遍历 batch 中的样本
                    true_label = int(y[i])  # 真实类别
                    pred_label = int(rag_preds[i])  # 预测类别
                    correct_prediction = (pred_label == true_label)

                    # 更新类别统计
                    self.feedback_stats[true_label]["total_counts"] += 1
                    if correct_prediction:
                        self.feedback_stats[true_label]["correct_counts"] += 1
                    else:
                        self.feedback_stats[true_label]["error_counts"] += 1

                    # 统计该类别选中的原型信息 & 统计原型的贡献情况
                    for j in range(topk):  # 遍历 top-k 选出的原型
                        proto_id = int(indices_np[i, j])  # 选中的原型 ID

                        # 记录该类别选择了哪些原型
                        self.feedback_stats[true_label]["selected_by_class"][proto_id] += 1

                        # 记录该原型的使用统计
                        self.proto_stats[proto_id]["total_counts"] += 1
                        if correct_prediction:
                            self.proto_stats[proto_id]["correct_counts"] += 1
                        else:
                            self.proto_stats[proto_id]["error_counts"] += 1

        # 计算每个类别和每个原型的准确率
        for class_id, stats in self.feedback_stats.items():
            if stats["total_counts"] > 0:
                stats["accuracy"] = stats["correct_counts"] / stats["total_counts"]
                stats['error_rate'] = stats['error_counts'] / stats['total_counts']

        if not_load_rep:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            self.save_item(f'all_{mode}_reps', all_embeddings)
            self.save_item(f'all_{mode}_labels', all_labels)

        all_rag_preds = torch.cat(all_rag_preds, dim=0)
        local_proto_ids = [(idx, meta['class']) for idx, meta in self.proto_meta.items()
                           if meta['CLIENT_ID'] == self.id]
        local_proto_ids = np.array(local_proto_ids)

        P = torch.tensor(
                    copy.deepcopy(self.proto_db[local_proto_ids[:, 0]]),
                    device=self.device,
                    dtype=torch.float32
                )

        proto_label = torch.tensor(local_proto_ids[:, 1], device=self.device)
        unique_classes = torch.unique(proto_label).tolist()

        if visualize:
            self.save_item( all_rag_preds,f'{mode}_rag_preds_{phase}')
            self.save_item( P,f'local_protos_{phase}')
            self.save_item(proto_label,f'local_protos_label_{phase}', )
        # 进行t-SNE可视化
        # if visualize and all_embeddings:
        #     all_embeddings = torch.cat(all_embeddings, dim=0)
        #     all_labels = torch.cat(all_labels, dim=0)
        #     all_rag_preds = torch.cat(all_rag_preds, dim=0)
        #
        #     local_proto_ids = [(idx, meta['class']) for idx, meta in self.proto_meta.items()
        #                        if meta['CLIENT_ID'] == self.id]
        #     local_proto_ids = np.array(local_proto_ids)
        #
        #
        #
        #     # 创建可优化的原型参数
        #     P = torch.tensor(
        #             copy.deepcopy(self.proto_db[local_proto_ids[:, 0]]),
        #             device=self.device,
        #             dtype=torch.float32
        #         )
        #
        #     proto_label = torch.tensor(local_proto_ids[:, 1], device=self.device)
        #     unique_classes = torch.unique(proto_label).tolist()


            # if len(unique_classes)!=1:
            #     visualize_tsne_results(all_embeddings, all_labels, all_rag_preds, P=P, proto_label=proto_label,
            #                            title=f"{self.id}_tsne_{mode}_{phase}", mode=mode)

        return local_correct, rag_correct, sample_num

    def vis_proto_effectiveness(self, mode='train'):
        from vis_exp.prototype_vis import visualize_comparison_tsne
        all_embeddings = self.load_item(f'all_{mode}_reps')
        all_labels = self.load_item(f'all_{mode}_labels')
        rag_preds_before_op = self.load_item(f'{mode}_rag_preds_before')
        rag_preds_after_op =  self.load_item(f'{mode}_rag_preds_after')
        visualize_comparison_tsne(
            all_embeddings,
            all_labels,
            rag_preds_before_op,
            rag_preds_after_op,
            title=f"{self.dataset}_{self.algorithm}_client_{self.id}_{mode}_comarision",
            P=None,
            proto_label=None
        )

    # def get_proto_feedback(self, mode='train', topk=None, mask_other_client_proto=False):
    #     """
    #     计算每个类别和每个原型的统计信息，包括：
    #     1. 每个类别的总预测次数、正确/错误分类次数
    #     2. 每个类别选中的原型分布
    #     3. 每个原型的被选次数、正确/错误贡献
    #
    #     参数：
    #     - mode: 'train' 或 'test'，决定使用训练数据还是测试数据
    #     - topk: 可选参数，指定要检索的 top-k 原型（默认为 self.topk）
    #
    #     返回：
    #     - feedback_stats: 按类别统计的预测情况
    #     - proto_stats: 按原型统计的使用情况
    #     """
    #     if mask_other_client_proto:
    #         for idx, meta in self.proto_meta.items():
    #             if meta['CLIENT_ID'] != self.id:
    #                 self.proto_meta[idx]['weight'] = 0
    #     if topk is None:
    #         topk = self.topk  # 使用默认的 top-k 值
    #
    #     self.feedback_stats = defaultdict(lambda: {
    #         "total_counts": 0,  # 该类别的总预测次数
    #         "error_counts": 0,  # 该类别的错误预测次数
    #         "correct_counts": 0,  # 该类别的正确预测次数
    #         "accuracy": 0.0,  # 该类别的预测准确率
    #         "selected_by_class": defaultdict(int)  # 记录该类别选中的原型信息
    #     })
    #
    #     self.proto_stats = defaultdict(lambda: {
    #         "total_counts": 0,  # 该原型被选中的总次数
    #         "error_counts": 0,  # 该原型导致错误分类的次数
    #         "correct_counts": 0,  # 该原型导致正确分类的次数
    #         "accuracy": 0.0  # 该原型的正确率
    #     })
    #
    #     self.model.eval()
    #
    #     # 选择数据加载器
    #     loader = self.load_train_data() if mode == 'train' else self.load_test_data()
    #     local_correct = 0
    #     rag_correct = 0
    #     sample_num = 0
    #     with torch.no_grad():
    #         for x, y in loader:
    #             # 处理不同数据格式
    #             if isinstance(x, list):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #
    #             # 计算样本表征
    #             rep = self.model.base(x)
    #
    #             # 检索最近的原型
    #             retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
    #                 rep, topk=1)  # (B, top_k, embed_dim)
    #
    #             # 使用加权投票进行分类
    #             rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes,
    #                                          retrievaled_similarities)
    #
    #             local_output = self.model.head(rep)
    #
    #             local_correct += (torch.sum(torch.argmax(local_output, dim=1) == y)).item()
    #             rag_correct += (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
    #             sample_num += y.shape[0]
    #
    #             rag_preds = rag_output.argmax(dim=1)
    #
    #             for i in range(x.shape[0]):  # 遍历 batch 中的样本
    #                 true_label = int(y[i])  # 真实类别
    #                 pred_label = int(rag_preds[i])  # 预测类别
    #                 correct_prediction = (pred_label == true_label)
    #
    #                 # 更新类别统计
    #                 self.feedback_stats[true_label]["total_counts"] += 1
    #                 if correct_prediction:
    #                     self.feedback_stats[true_label]["correct_counts"] += 1
    #                 else:
    #                     self.feedback_stats[true_label]["error_counts"] += 1
    #
    #                 # 统计该类别选中的原型信息 & 统计原型的贡献情况
    #                 for j in range(topk):  # 遍历 top-k 选出的原型
    #                     proto_id = int(indices_np[i, j])  # 选中的原型 ID
    #
    #                     # 记录该类别选择了哪些原型
    #                     self.feedback_stats[true_label]["selected_by_class"][proto_id] += 1
    #
    #                     # 记录该原型的使用统计
    #                     self.proto_stats[proto_id]["total_counts"] += 1
    #                     if correct_prediction:
    #                         self.proto_stats[proto_id]["correct_counts"] += 1
    #                     else:
    #                         self.proto_stats[proto_id]["error_counts"] += 1
    #
    #     # 计算每个类别和每个原型的准确率
    #     for class_id, stats in self.feedback_stats.items():
    #         if stats["total_counts"] > 0:
    #             stats["accuracy"] = stats["correct_counts"] / stats["total_counts"]
    #             stats['error_rate'] = stats['error_counts'] / stats['total_counts']
    #
    #     # self.all_protos_meta = all_protos_meta
    #     return local_correct, rag_correct, sample_num

    def weighted_voting_by_sim(self, all_proto, all_meta, proto_sim, num_classes):
        """
        使用原型的类别信息进行加权投票
        :param all_proto: (num_proto,embed_dim) 原型张量
        :param all_meta: (num_proto,) 原型的元信息列表
        :param proto_sim: (batch_size, num_proto) 当前mini_batch中样本表征和所有原型的相似度
        :param num_classes: 类别数量
        :return: (batch_size,) 经过加权投票的类别预测
        """
        num_protos = all_proto.shape[0]
        batch_size = proto_sim.shape[0]

        # 归一化权重（可用 softmax 或者 L1 归一化）
        weights = torch.softmax(proto_sim, dim=1)  # (batch_size, num_protos)
        # visualize_heatmap(proto_sim)
        # 提取类别信息 (batch_size, top_k)
        class_labels = torch.tensor([meta['class'] for idx, meta in all_meta.items()], device=proto_sim.device)
        expanded_class_labels = class_labels.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_protos)

        # 计算类别得分
        # num_classes = class_labels.max().item() + 1  # 假设类别是从 0 开始的
        class_scores = torch.zeros((batch_size, num_classes), device=proto_sim.device)

        class_scores.scatter_add_(1, expanded_class_labels, weights)
        # for i in range(num_protos):
        #     class_scores.scatter_add_(1, class_labels[:, i].unsqueeze(-1), weights[:, i].unsqueeze(-1))

        # 选取得分最高的类别
        # predicted_labels = class_scores.argmax(dim=1)

        return class_scores


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, f"Client{role}_{item_name}.pt"))


def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, f"Client{role}_{item_name}.pt"), weights_only=False)
    except FileNotFoundError:
        print(f"Not Found: Client{role}_{item_name}")
        return None
    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
