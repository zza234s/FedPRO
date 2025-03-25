import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
import math
from sklearn.preprocessing import label_binarize
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
from pycm import *
import torch.nn.functional as F
from utils.vis_utils import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.finch import FINCH




class ParameterFreeAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        embed_dim: 特征维度，用于缩放因子
        """
        super(ParameterFreeAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(embed_dim)  # 缩放因子

    def forward(self, query, key, value):
        """
        query: (B, 1, embed_dim)
        key:   (B, top_k, embed_dim)
        value: (B, top_k, embed_dim)
        """
        # 计算点积注意力分数，结果 shape 为 (B, 1, top_k)
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = scores * self.scale  # 缩放分数
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, top_k)
        # 计算加权求和得到融合后的表示
        attn_output = torch.bmm(attn_weights, value)  # (B, 1, embed_dim)
        return attn_output, attn_weights


class clientIDS_V1(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda

        self.lam = torch.tensor([1.0], requires_grad=True, device=self.device)
        self.gamma = torch.tensor([1.0], requires_grad=True, device=self.device)
        self.optimizer_learned_weight_for_inference = torch.optim.Adam([self.lam, self.gamma], lr=args.ensemble_lr)
        # self.optimizer_learned_weight_for_inference = torch.optim.SGD([self.lam], lr=1e-2)

        self.use_ensemble = args.use_ensemble

        # RAG增强特征
        self.use_rag = args.use_RAG
        self.prototype_db = defaultdict(list)
        self.attention = ParameterFreeAttention(args.hidden_dim)  # todo:hidden dim 两个数据集并不同意，要注意设置
        self.vis_prototype = args.vis_prototype
        self.test_cnt =0 #TODO: 这里用来临时记录客户端测试次数，用于测试时的可视化
        self.topk=args.topk
        self.feedback_stats = defaultdict(lambda: {
            "total_counts": 0,
            "error_counts": 0,
            "correct_counts": 0
        })
    def retrieve_prototypes(self, query_features, topk=5):
        all_protos_tensor = self.proto_db.to(self.device)  # 存放所有原型的张量
        all_meta = self.proto_meta  # 存放每个原型对应的类别信息

        # # 遍历 prototype_db，将每个原型及其元信息存入列表
        # for cls, proto_entries in self.prototype_db.items():
        #     for entry in proto_entries:
        #         all_protos.append(entry['proto'].squeeze())
        #         meta_info = {
        #             'class': cls,
        #             'source': entry['source'],
        #             'CLIENT_ID': entry['client_id']
        #         }
        #         all_meta.append(meta_info)


        # all_protos_tensor = torch.stack(all_protos).to(self.device)  # (Number of all prototypes, D)

        query_norm = F.normalize(query_features, p=2, dim=1)  # (B, D)
        protos_norm = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
        sims = torch.mm(query_norm, protos_norm.t()) # (B, N)

        _, indices = torch.topk(sims, k=topk, dim=1)

        prototypes_retrieved = all_protos_tensor[indices]  # (B, top_k, embed_dim)
        # indices 是 tensor，因此转换为 numpy 数组以便索引列表 all_meta
        meta_retrieved = []
        indices_np = indices.cpu().numpy()  # ndarray: shape (B, topk)
        for idx_list in indices_np:
            meta_retrieved.append([all_meta[i] for i in idx_list])

        return prototypes_retrieved, meta_retrieved, indices_np, all_protos_tensor,all_meta

    def weighted_retrieve_prototypes(self, query_features, topk=5):
        all_protos_tensor = self.proto_db.to(self.device)  # 存放所有原型的张量
        all_meta = self.proto_meta  # 存放每个原型对应的类别信息
        all_weight = []
        for idx, meta in enumerate(all_meta):
            all_weight.append(meta['weight'])
        weight = torch.tensor(all_weight).to(self.device)

        query_norm = F.normalize(query_features, p=2, dim=1)  # (B, D)
        protos_norm = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
        sims = weight * torch.mm(query_norm, protos_norm.t()) # (B, N)

        _, indices = torch.topk(sims, k=topk, dim=1)

        prototypes_retrieved = all_protos_tensor[indices]  # (B, top_k, embed_dim)
        # indices 是 tensor，因此转换为 numpy 数组以便索引列表 all_meta
        meta_retrieved = []
        indices_np = indices.cpu().numpy()  # ndarray: shape (B, topk)
        for idx_list in indices_np:
            meta_retrieved.append([all_meta[i] for i in idx_list])

        return prototypes_retrieved, meta_retrieved, indices_np, all_protos_tensor,all_meta





    def get_proto_feedback(self):
        self.feedback_stats = defaultdict(lambda: {
            "total_counts": 0,
            "error_counts": 0,
            "correct_counts": 0
        })
        self.model.eval()
        trainloader = self.load_train_data()
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                retrieve_prototypes, retrieve_meta_info, indices_np, all_protos_tensor, all_protos_meta = self.weighted_retrieve_prototypes(rep,topk=self.topk)  # (B, top_k, embed_dim)
                rag_output =  weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes)
                rag_preds = rag_output.argmax(dim=1)

                for i in range(x.shape[0]):
                    true_label = y[i]
                    for j in range(self.topk):
                        proto_id = int(indices_np[i, j])
                        if rag_preds[i] != true_label:
                            self.feedback_stats[proto_id]["error_counts"] += 1 # 若预测错误则记录错误次数
                        else:
                            self.feedback_stats[proto_id]["correct_counts"] += 1
                        self.feedback_stats[proto_id]["total_counts"] += 1


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
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                # retrieve_prototypes, retrieve_meta_info, indices_np, all_protos_tensor, all_protos_meta = self.retrieve_prototypes(rep,topk=self.topk)  # (B, top_k, embed_dim)
                retrieve_prototypes, retrieve_meta_info, indices_np, all_protos_tensor, all_protos_meta = self.weighted_retrieve_prototypes(rep,topk=self.topk)  # (B, top_k, embed_dim)
                local_model_output = self.model.head(rep)
                rag_output =  weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes)
                ensemble_output = 0.8*local_model_output + 0.2*rag_output


                # rag_preds = rag_output.argmax(dim=1)
                # for i in range(x.shape[0]):
                #     sample_rep = rep[i]  # (embed_dim,)
                #     true_label = y[i]
                #     for j in range(self.topk):
                #         proto_id = int(indices_np[i, j])
                #         if rag_preds[i] != true_label:
                #             self.feedback_stats[proto_id]["error_counts"] += 1 # 若预测错误则记录错误次数
                #         else:
                #             self.feedback_stats[proto_id]["correct_counts"] += 1
                #
                #         self.feedback_stats[proto_id]["total_counts"] += 1

                #         # 计算特征差异并累加
                #         diff = sample_rep - retrieve_prototypes[i, j]
                #         self.feedback_stats[proto_id]["diff_sum"] += diff
                # # ------------------------------



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

                if self.vis_prototype and retrieve_prototypes is not None:
                    all_features.append(rep)
                    # all_retrieved_prototypes.append(retrieve_prototypes)
                    all_retrieved_meta.extend(retrieve_meta_info)


            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_cost.append(inference_time)

            if self.vis_prototype and retrieve_prototypes is not None:
                all_features = torch.cat(all_features, dim=0)
                # all_retrieved_prototypes = torch.cat(all_retrieved_prototypes, dim=0).squeeze()
                visualize_tsne_with_class(all_features,
                                          y_true_list,
                                          0,
                                          all_protos_tensor,
                                          all_protos_meta,
                                          all_retrieved_meta,
                                         title = f'{self.role}, query_idx: {0}, test_cnt: {self.test_cnt}')

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
    def set_proto_db(self,proto_db,proto_meta):
        self.proto_db = proto_db
        self.proto_meta = proto_meta


    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def receive_proto_of_other_clients(self, client_ids, client_protos):
        self.client_protos = client_protos
        self.client_protos_ids = client_ids

    def build_prototype_db(self):
        #TODO 弃用
        self.prototype_db = defaultdict(list)  # reset the prototype db

        if self.global_protos is not None:
            for cls, proto in self.global_protos.items():
                self.prototype_db[cls].append({
                    'proto': proto.detach(),  # 原型张量
                    'source': 'global',  # 来源标记为 global
                    'client_id': None  # global 原型没有 client id
                })

        for client_id, client_proto in zip(self.client_protos_ids, self.client_protos):
            for cls, proto in client_proto.items():
                if type(proto) == type([]):
                    for i in range(len(proto)):
                        self.prototype_db[cls].append(
                            {
                                'proto': proto[i].detach(),
                                'source': 'client',
                                'client_id': client_id
                            }
                        )
                else:
                    self.prototype_db[cls].append({
                        'proto': proto.detach(),
                        'source': 'client',
                        'client_id': client_id
                    })

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
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                owned_classes = y.unique()
                for cls in owned_classes:
                    filted_reps = rep[y == cls].detach()
                    reps_dict[cls.item()].append(filted_reps.data)

                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     protos[y_c].append(rep[i, :].detach().data)
            for cls, protos in reps_dict.items():
                mean_proto = torch.cat(protos).mean(dim=0)
                agg_local_protos[cls] = mean_proto

        # 对于未见类，生成全zero tensor上传server
        unseen_classes = set(range(self.num_classes)) - set(agg_local_protos.keys())
        if unseen_classes != set():
            print(f"class(es) {unseen_classes} not in the client{self.id}'s  local training dataset")

        # for label in range(self.num_classes):
        #     if label not in agg_local_protos:
        #         agg_local_protos[label] = torch.zeros(list(agg_local_protos.values())[0].shape[0],
        #                                               device=self.device)

        agg_local_protos = dict(sorted(agg_local_protos.items()))  # 确保上传的dict按key升序排序
        self.protos = agg_local_protos

    @torch.no_grad()
    def local_cluster_for_proto_gen(self):
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

        # 输出未见类信息
        unseen_classes = set(range(self.num_classes)) - set(reps_dict.keys())
        if unseen_classes != set():
            print(f"class(es) {unseen_classes} not in the client{self.id}'s  local training dataset")

        # 聚类生成原型
        for cls, protos in reps_dict.items():
            protos_np = torch.cat(protos).detach().cpu().numpy()
            c, num_clust, req_c = FINCH(protos_np, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=True)
            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])

            class_cluster_array = np.array(class_cluster_list)
            uniqure_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []

            for _, cluster_index in enumerate(uniqure_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = protos_np[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(torch.tensor(proto))
            agg_local_protos[cls] = agg_selected_proto



            # mean_proto = torch.cat(protos).mean(dim=0)
            # agg_local_protos[cls] = mean_proto

        # for label in range(self.num_classes):
        #     if label not in agg_local_protos:
        #         agg_local_protos[label] = torch.zeros(list(agg_local_protos.values())[0].shape[0],
        #                                               device=self.device)

        agg_local_protos = dict(sorted(agg_local_protos.items()))  # 确保上传的dict按key升序排序
        self.protos = agg_local_protos
    def test_metrics(self):
        self.test_cnt+=1
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0

        correct_prototype = 0
        correct_local = 0
        correct_rag = 0

        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []



        start_time = time.time()
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                local_model_output = self.model.head(rep)

                output = local_model_output

                correct_local += (torch.sum(torch.argmax(local_model_output, dim=1) == y)).item()
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

                y_pred.extend(np.argmax(output.detach().cpu().numpy(), axis=1).tolist())
                y_true_list.extend(y.tolist())


            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_cost.append(inference_time)


        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc, correct_local, correct_prototype, self.lam.detach().data, self.gamma.detach().data,correct_rag

def weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, num_classes):
    """
    使用原型的类别信息进行加权投票
    :param retrieve_prototypes: (batch_size, top_k, embed_dim) 原型张量
    :param retrieve_meta_info: (batch_size, top_k) 原型的元信息列表
    :param rep: (batch_size, embed_dim) 查询样本的嵌入
    :return: (batch_size,) 经过加权投票的类别预测
    """
    batch_size, top_k, embed_dim = retrieve_prototypes.shape

    # 计算原型与当前样本的相似度（欧式距离/余弦相似度）
    similarity = torch.nn.functional.cosine_similarity(retrieve_prototypes, rep.unsqueeze(1),
                                                       dim=-1)  # (batch_size, top_k)

    # 归一化权重（可用 softmax 或者 L1 归一化）
    weights = torch.softmax(similarity, dim=1)  # (batch_size, top_k)

    # 提取类别信息 (batch_size, top_k)
    class_labels = torch.tensor([[meta['class'] for meta in sample_meta] for sample_meta in retrieve_meta_info],
                                device=rep.device)

    # 计算类别得分
    # num_classes = class_labels.max().item() + 1  # 假设类别是从 0 开始的
    class_scores = torch.zeros((batch_size, num_classes), device=rep.device)

    for i in range(top_k):
        class_scores.scatter_add_(1, class_labels[:, i].unsqueeze(-1), weights[:, i].unsqueeze(-1))

    # 选取得分最高的类别
    # predicted_labels = class_scores.argmax(dim=1)

    return class_scores