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

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable
import os
from collections import defaultdict
from utils.proto_rag import weighted_voting
from utils.vis_utils import *
import copy
class clientDBE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.klw = args.kl_weight
        self.momentum = args.momentum
        self.global_mean = None

        trainloader = self.load_train_data()        
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)


    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.reset_running_stats()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                    
                # ====== begin
                rep = self.model.base(x)
                running_mean = torch.mean(rep, dim=0)

                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)

                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * running_mean
                
                if self.global_mean is not None:
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean)**2)
                    output = self.model.head(rep + self.client_mean)
                    loss = self.loss(output, y)
                    loss = loss + reg_loss * self.klw
                else:
                    output = self.model.head(rep)
                    loss = self.loss(output, y)
                # ====== end

                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.opt_client_mean.step()
                self.detach_running()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def detach_running(self):
        self.running_mean.detach_()

    def train_metrics(self):
        trainloader = self.load_train_data()
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
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        reps = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)

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
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def RAF(self, mode='test'):
        from utils.proto_rag import weighted_voting
        ensemble_acc = 0
        sample_num = 0

        correct_prototype = 0
        correct_local = 0
        correct_rag = 0

        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []

        res_dict = dict()

        if mode == 'train':
            loader = self.load_train_data()
        else:
            loader = self.load_test_data()
        start_time = time.time()
        self.model.eval()

        self.proto_db = self.proto_db.to(self.device)

        for idx, meta in self.proto_meta.items():
            self.proto_meta[idx]['weight'] = 1.0

        with torch.no_grad():
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                ##############
                local_model_output = self.model.head(rep + self.client_mean)
                ##############
                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                     rep, topk=self.topk)  # (B, top_k, embed_dim)
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes, retrievaled_similarities)

                proto_conf,_ = torch.max(retrievaled_similarities, dim=1)
                # proto_conf = torch.max(rag_output, dim=1).values
                alpha = (1.0 - proto_conf).unsqueeze(1)
                ensemble_output = (1 - alpha) * rag_output + alpha * local_model_output

                correct_local += (torch.sum(torch.argmax(local_model_output, dim=1) == y)).item()
                correct_rag += (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
                ensemble_acc += (torch.sum(torch.argmax(ensemble_output, dim=1) == y)).item()
                sample_num += y.shape[0]

                y_prob.append(ensemble_output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

                y_pred.extend(np.argmax(ensemble_output.detach().cpu().numpy(), axis=1).tolist())
                y_true_list.extend(y.tolist())

            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_cost.append(inference_time)

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



    def get_proto_feedback(self, mode='train', topk=None, mask_other_client_proto=False, visualize=False, phase='before'):
        """
        计算每个类别和每个原型的统计信息，包括：
        1. 每个类别的总预测次数、正确/错误分类次数
        2. 每个类别选中的原型分布
        3. 每个原型的被选次数、正确/错误贡献

        参数：
        - mode: 'train' 或 'test'，决定使用训练数据还是测试数据
        - topk: 可选参数，指定要检索的 top-k 原型（默认为 self.topk）

        返回：
        - feedback_stats: 按类别统计的预测情况
        - proto_stats: 按原型统计的使用情况
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

        # 选择数据加载器
        loader = self.load_train_data() if mode == 'train' else self.load_test_data()
        local_correct = 0
        rag_correct =0
        sample_num = 0

        # 用于t-SNE可视化的数据收集
        if visualize:
            all_embeddings = []
            all_labels = []
            all_rag_preds = []


        with torch.no_grad():
            for x, y in loader:
                # 处理不同数据格式
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 计算样本表征
                rep = self.model.base(x)
                local_output = self.model.head(rep + self.client_mean)

                # 检索最近的原型
                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                    rep, topk=1)  # (B, top_k, embed_dim)

                # 使用加权投票进行分类
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes,
                                             retrievaled_similarities)


                local_correct += (torch.sum(torch.argmax(local_output, dim=1) == y)).item()
                rag_correct += (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
                sample_num += y.shape[0]

                rag_preds = rag_output.argmax(dim=1)

                # 收集可视化数据
                if visualize:
                    all_embeddings.append(rep.cpu())
                    all_labels.append(y.cpu())
                    all_rag_preds.append(rag_preds.cpu())


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

        if visualize and all_embeddings and not self.one_class_flag:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
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
            if len(unique_classes)!=1:
                visualize_tsne_results(all_embeddings, all_labels, all_rag_preds, P=P, proto_label=proto_label,
                                   title=f"{self.id}_tsne_{mode}_{phase}", mode=mode)

        # self.all_protos_meta = all_protos_meta
        return local_correct,rag_correct,sample_num


    def save_best_model(self,best_acc_str):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)

        model_path = os.path.join(self.save_folder_name, f"{self.role}_best_model_{best_acc_str}.pt")
        client_mean_path = os.path.join(self.save_folder_name, f"{self.role}_client_mean_{best_acc_str}.pt")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.client_mean, client_mean_path)
        print(f"Save the FedDBE's best local model files at: {self.save_folder_name}")