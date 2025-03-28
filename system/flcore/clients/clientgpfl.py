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
import time
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client
import os
from collections import defaultdict
from utils.proto_rag import weighted_voting
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
class clientGPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.feature_dim = list(self.model.head.parameters())[0].shape[1]

        self.lamda = args.lamda
        self.mu = args.mu

        self.GCE = copy.deepcopy(args.GCE)
        self.GCE_opt = torch.optim.Adam(self.GCE.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=self.mu)
        self.GCE_frozen = copy.deepcopy(self.GCE)

        self.CoV = copy.deepcopy(args.CoV)
        self.CoV_opt = torch.optim.Adam(self.CoV.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.mu)

        self.generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        self.personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        trainloader = self.load_train_data()
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / torch.sum(
            self.sample_per_class)
        

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                feat_G = self.CoV(feat, self.generic_conditional_input)
                softmax_loss = self.GCE(feat_G, y)

                loss = self.loss(output, y)
                # loss += softmax_loss

                emb = torch.zeros_like(feat)
                for i, yy in enumerate(y):
                    emb[i, :] = self.GCE_frozen.embedding(yy).detach().data
                loss += torch.norm(feat_G - emb, 2) * self.lamda

                self.optimizer.zero_grad()
                self.GCE_opt.zero_grad()
                self.CoV_opt.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.GCE_opt.step()
                self.CoV_opt.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, base):
        self.global_base = base
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def set_GCE(self, GCE):
        self.generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        self.personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        embeddings = self.GCE.embedding(torch.tensor(range(self.num_classes), device=self.device))
        for l, emb in enumerate(embeddings):
            self.generic_conditional_input.data += emb / self.num_classes
            self.personalized_conditional_input.data += emb * self.sample_per_class[l]

        for new_param, old_param in zip(GCE.parameters(), self.GCE.parameters()):
            old_param.data = new_param.data.clone()

        self.GCE_frozen = copy.deepcopy(self.GCE)

    def set_CoV(self, CoV):
        for new_param, old_param in zip(CoV.parameters(), self.CoV.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                test_acc += (torch.sum(
                    torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

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
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                feat_G = self.CoV(feat, self.generic_conditional_input)
                softmax_loss = self.GCE(feat_G, y)

                loss = self.loss(output, y)
                loss += softmax_loss

                emb = torch.zeros_like(feat)
                for i, yy in enumerate(y):
                    emb[i, :] = self.GCE_frozen.embedding(yy).detach().data
                loss += torch.norm(feat_G - emb, 2) * self.lamda

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                
        return losses, train_num

    def RAF(self, mode='test'):
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

                rep = self.CoV(rep, self.personalized_conditional_input)
                local_model_output = self.model.head(rep)

                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                     rep, topk=self.topk)  # (B, top_k, embed_dim)
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes, retrievaled_similarities)

                proto_conf = torch.max(rag_output, dim=1).values
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

    def get_proto_feedback(self, mode='train', topk=None, mask_other_client_proto=False):
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
                ####
                rep = self.CoV(rep, self.personalized_conditional_input)
                ####

                local_output = self.model.head(rep)

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

        # self.all_protos_meta = all_protos_meta
        return local_correct,rag_correct,sample_num

    def refine_proto_db(self,epochs=50):
        local_proto_ids = [(idx, meta['class']) for idx, meta in self.proto_meta.items()
                           if meta['CLIENT_ID'] == self.id]
        local_proto_ids = np.array(local_proto_ids)
        self.margin=0.1
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

        if len(unique_classes)==1:
            print(f"client#{self.id} 只有{len(unique_classes)}个类，暂不支持更新prototype")
            agg_local_protos = defaultdict(list)
            for cls, protos in zip(proto_label, P.clone().detach()):
                agg_local_protos[cls.item()].append(protos)

            self.protos = agg_local_protos
            return
        # 设置优化器
        optimizer = Adam([P], lr=0.01) #TODO: 优化器参数待调整
        self.model.eval()

        loader = self.load_train_data()

        # 预先计算所有训练样本的特征表示（避免重复计算）
        all_features = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                ##################
                rep = self.model.base(x)
                rep = self.CoV(rep, self.personalized_conditional_input)
                ##################

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
        patience = 5  # 早停耐心值
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
            print(f"client#{self.id},, Epoch {epoch + 1}/{epochs} 完成, Loss: {total_loss:.4f}, Acc: {epoch_accuracy:.4f}")

                # 使用训练准确率判断
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_prototypes = P.clone().detach()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"连续{patience}轮无改善，提前停止")
                break
        print(f"优化完成，最佳准确率: {best_accuracy:.4f}")

        # visualize_with_dim_reduction(all_features, all_labels, 0, P, proto_label, mode='pca', title=f'_client#{self.id}原型更新后')
        agg_local_protos = defaultdict(list)
        for cls, protos in zip(proto_label,best_prototypes):
            agg_local_protos[cls.item()].append(protos)

        self.protos= agg_local_protos



    def save_best_model(self,best_acc_str):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)


        model_path = os.path.join(self.save_folder_name, f"{self.role}_best_model_{best_acc_str}.pt")
        GCE_path = os.path.join(self.save_folder_name, f"{self.role}_GCE_{best_acc_str}.pt")
        GCE_frozen_path = os.path.join(self.save_folder_name, f"{self.role}_GCE_frozen_{best_acc_str}.pt")
        CoV_path = os.path.join(self.save_folder_name, f"{self.role}_CoV_{best_acc_str}.pt")
        generic_conditional_input_path = os.path.join(self.save_folder_name, f"{self.role}_generic_conditional_input_{best_acc_str}.pt")
        personalized_conditional_input_path = os.path.join(self.save_folder_name, f"{self.role}_personalized_conditional_input_{best_acc_str}.pt")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.GCE, GCE_path)
        torch.save(self.GCE_frozen, GCE_frozen_path)
        torch.save(self.CoV, CoV_path)
        torch.save(self.generic_conditional_input, generic_conditional_input_path)
        torch.save(self.personalized_conditional_input, personalized_conditional_input_path)

        print(f"Save the GPFL's best local model files at: {self.save_folder_name}")