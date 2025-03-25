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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


class clientIDS_PLUS(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda

        self.cuda_cka = CudaCKA(args.device)
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
    def retrieve_prototypes(self, query_features, topk=5):
        if topk is None:
            topk = len(self.protos)
        similarities = []

        all_protos = []  # 存放所有原型的张量
        all_meta = []  # 存放每个原型对应的类别信息
        # 遍历 prototype_db，将每个原型及其元信息存入列表
        for cls, proto_entries in self.prototype_db.items():
            for entry in proto_entries:
                all_protos.append(entry['proto'])
                meta_info = {
                    'class': cls,
                    'source': entry['source']
                }
                if entry['source'] == 'client':
                    meta_info['client_id'] = entry['client_id']
                else:
                    meta_info['client_id'] = None
                all_meta.append(meta_info)

        if len(all_protos) == 0:
            # 如果没有原型，则返回空结果
            return [[] for _ in range(query_features.size(0))]

        all_protos_tensor = torch.stack(all_protos)  # (Number of all prototypes, D)

        query_norm = F.normalize(query_features, p=2, dim=1)  # (B, D)
        protos_norm = F.normalize(all_protos_tensor, p=2, dim=1)  # (N, D)
        sims = torch.mm(query_norm, protos_norm.t()) # (B, N)

        _, indices = torch.topk(sims, k=topk, dim=1)

        prototypes_retrieved = all_protos_tensor[indices]  # (B, top_k, embed_dim)
        # indices 是 tensor，因此转换为 numpy 数组以便索引列表 all_meta
        meta_retrieved = []
        indices_np = indices.cpu().numpy()  # shape (B, topk)
        for idx_list in indices_np:
            meta_retrieved.append([all_meta[i] for i in idx_list])

        return prototypes_retrieved, meta_retrieved, indices_np

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

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
                output = self.model(x)

                # rep = self.model.base(x)
                # rag_output = None
                # if self.prototype_db:
                #     retrieve_prototypes = self.retrieve_prototypes(rep)  # (B, top_k, embed_dim)
                #
                #     # RAG特征增强
                #     query = rep.unsqueeze(1)  # (B, 1, embed_dim)
                #     attn_output, attn_weights = self.attention(query, retrieve_prototypes, retrieve_prototypes)
                #     attn_output = attn_output.squeeze(1)  # (B, embed_dim)
                #
                #     enhanced_rep = rep + attn_output  # (B, embed_dim)
                #
                #     rag_output = self.model.head(enhanced_rep)
                #
                # local_model_output = self.model.head(rep)
                #
                # if rag_output is not None:
                #     output = local_model_output + rag_output
                # else:
                #     output = local_model_output

                # if self.global_protos is not None and self.use_ensemble:
                #     with torch.no_grad():
                #         global_ptorotypes = torch.stack(list(self.global_protos.values()))  # dict to tensor
                #         # prototype_sim = self.cuda_cka.linear_CKA(rep.detach().data, global_ptorotypes) #todo 找到支持广播的Linea CKA
                #         sim = torch.matmul(rep.detach().data, global_ptorotypes.T)
                #         # Normalize the similarity matrix according to the L1 norm
                #         norms = sim.norm(p=1, dim=1, keepdim=True)
                #         prototype_based_output = sim / (norms + 1e-5)
                #
                #     output = local_model_output + prototype_based_output + rag_output
                #     # output = self.lam * local_model_output + self.gamma * prototype_based_output
                #     # output = self.lam * local_model_output + (1-self.lam) * prototype_based_output
                # else:
                #     output = local_model_output
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                # self.optimizer_learned_weight_for_inference.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.optimizer_learned_weight_for_inference.step()
        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        # self.collect_protos()
        # self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def receive_proto_of_other_clients(self, client_ids, client_protos):
        self.client_protos = client_protos
        self.client_protos_ids = client_ids

    def build_prototype_db(self):
        self.prototype_db = defaultdict(list)  # reset the prototype db

        for cls, proto in self.global_protos.items():
            self.prototype_db[cls].append({
                'proto': proto.detach(),  # 原型张量
                'source': 'global',  # 来源标记为 global
                'client_id': None  # global 原型没有 client id
            })

        for client_id, client_proto in zip(self.client_protos_ids, self.client_protos):
            for cls, proto in client_proto.items():
                self.prototype_db[cls].append({
                    'proto': proto.detach(),
                    'source': 'client',
                    'client_id': client_id
                })

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
            print(
                f'client{self.id}: class(es) {unseen_classes} not in the local training dataset, add zero tensors to local prototype')
        for label in range(self.num_classes):
            if label not in agg_local_protos:
                agg_local_protos[label] = torch.zeros(list(agg_local_protos.values())[0].shape[0],
                                                      device=self.device)

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
        retrieve_prototypes =None

        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []

        all_features = []
        all_retrieved_prototypes = []
        all_retrieved_meta = []



        start_time = time.time()
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                rag_output = None
                if self.use_rag and self.prototype_db:
                    retrieve_prototypes, retrieve_meta_info = self.retrieve_prototypes(rep)  # (B, top_k, embed_dim)

                    # RAG特征增强
                    query = rep.unsqueeze(1)  # (B, 1, embed_dim)
                    attn_output, attn_weights = self.attention(query, retrieve_prototypes, retrieve_prototypes)
                    attn_output = attn_output.squeeze(1)  # (B, embed_dim)

                    enhanced_rep = rep + attn_output  # (B, embed_dim)
                    rag_output = self.model.head(enhanced_rep)
                    correct_rag+= (torch.sum(torch.argmax(rag_output, dim=1) == y)).item()
                local_model_output = self.model.head(rep)

                if rag_output is not None:
                    output = local_model_output + rag_output
                else:
                    output = local_model_output

                # if self.global_protos is not None and self.use_ensemble:
                #     global_ptorotypes = torch.stack(list(self.global_protos.values()))  # dict to tensor
                #     # prototype_sim = self.cuda_cka.linear_CKA(rep.detach().data, global_ptorotypes) #todo 找到支持广播的Linea CKA
                #     sim = torch.matmul(rep.detach().data, global_ptorotypes.T)
                #     # Normalize the similarity matrix according to the L1 norm
                #     prototype_based_output = sim / (sim.norm(p=1, dim=1, keepdim=True) + 1e-5)
                #     correct_prototype += (torch.sum(torch.argmax(prototype_based_output, dim=1) == y)).item()
                #
                #     # output = self.lam * local_model_output + (1-self.lam)* prototype_based_output  # ensemble_out
                #     # output = self.lam * local_model_output + self.gamma * prototype_based_output #ensemble_out
                #     output = local_model_output + prototype_based_output + rag_output
                # else:
                #     output = local_model_output
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

                if self.vis_prototype and retrieve_prototypes is not None:
                    all_features.append(rep)
                    all_retrieved_prototypes.append(retrieve_prototypes)
                    all_retrieved_meta.append(retrieve_meta_info)


            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_cost.append(inference_time)

            if self.vis_prototype and retrieve_prototypes is not None:
                all_features = torch.cat(all_features, dim=0)
                all_retrieved_prototypes = torch.cat(all_retrieved_prototypes, dim=0)
                visualize_tsne_with_class(all_features, y_true_list, 0, all_retrieved_prototypes, all_retrieved_meta,
                                         title = f'{self.role}, query_idx: {0}, test_cnt: {self.test_cnt}')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc, correct_local, correct_prototype, self.lam.detach().data, self.gamma.detach().data,correct_rag

    def learning_for_inferencen(self):
        self.model.eval()
        self.optimizer_learned_weight_for_inference.zero_grad()
        trainloader = self.load_train_data()
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            rep = self.model.base(x)
            local_model_output = self.model.head(rep)
            if self.global_protos is not None and self.use_ensemble:
                global_ptorotypes = torch.stack(list(self.global_protos.values()))


def visualize_tsne_with_class(all_features, all_labels, query_idx,
                              query_retrieved_prototypes, query_retrieved_meta,
                              title='TSNE Visualization'):
    """
    使用 TSNE 降维并可视化所有样本特征（all_features）和 query 检索到的原型，
    并根据不同 class 用不同颜色显示，同时在原型点旁标注出原型编号、所属 class 及其来源信息。

    参数:
      all_features: numpy 数组或 PyTorch 张量, shape = (N, D)
                    所有样本的特征表示。
      all_labels: list 或 numpy 数组, 长度为 N, 每个元素对应 all_features 中样本的 class。
      query_idx: int, 指定 query 在 all_features 中的索引。
      query_retrieved_prototypes: numpy 数组或 PyTorch 张量, shape = (K, D)
                                  query 检索到的原型特征。
      query_retrieved_meta: list of dict, 长度为 K，每个字典至少包含键:
                              "class": 原型所属 class，
                              "source": 原型来源（"global" 或 "client"），
                              若 "source" 为 "client"，可包含 "client_id"。
      title: str, 图像标题 (默认 'TSNE Visualization')。
    """
    # 如果输入为 PyTorch 张量，则转换为 numpy 数组
    if isinstance(all_features, torch.Tensor):
        all_features = all_features.detach().cpu().numpy()
    if isinstance(query_retrieved_prototypes, torch.Tensor):
        query_retrieved_prototypes = query_retrieved_prototypes.detach().cpu().numpy()

    # 将所有特征与检索到的原型合并，用于 TSNE 降维
    combined_features = np.concatenate([all_features, query_retrieved_prototypes], axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_features)

    # 分离降维后所有特征和原型对应的二维点
    N = all_features.shape[0]
    all_features_2d = combined_2d[:N, :]
    prototypes_2d = combined_2d[N:, :]

    # 从 all_labels 和 query_retrieved_meta 中提取所有 class 信息
    all_labels = np.array(all_labels)
    # 对于原型，我们假设 meta 中包含 "class" 键
    proto_labels = np.array([meta.get('class', 'unknown') for meta in query_retrieved_meta])

    # 计算所有出现的 class
    unique_classes = np.unique(np.concatenate([all_labels, proto_labels]))

    # 为不同的 class 分配不同颜色，使用 matplotlib 的 tab10 调色板
    cmap = plt.get_cmap("tab10")
    class_to_color = {cls: cmap(i % 10) for i, cls in enumerate(unique_classes)}

    plt.figure(figsize=(10, 8))

    # 绘制所有样本特征，根据所属 class 采用不同颜色
    for cls in np.unique(all_labels):
        indices = np.where(all_labels == cls)[0]
        plt.scatter(all_features_2d[indices, 0],
                    all_features_2d[indices, 1],
                    color=class_to_color[cls],
                    alpha=0.6,
                    label=f'Class {cls}')

    # 高亮指定 query 点（从 all_features 中提取）
    query_point = all_features_2d[query_idx]
    plt.scatter(query_point[0], query_point[1],
                color='red', marker='*', s=250,
                edgecolors='black', linewidths=1.5,
                label='Query')

    # 绘制检索到的原型，颜色与其所属 class 一致，标记为菱形，并在旁边添加注释
    for i, proto in enumerate(prototypes_2d):
        proto_cls = proto_labels[i]
        plt.scatter(proto[0], proto[1],
                    color=class_to_color[proto_cls],
                    marker='D', s=150,
                    edgecolor='black', linewidth=1)
        meta = query_retrieved_meta[i]
        source = meta.get('source', 'unknown')
        # 如果来源为 client，则附带 client_id
        if source == 'client':
            client_id = meta.get('client_id', 'unknown')
            label_text = f"P{i} (client: {client_id})"
        else:
            label_text = f"P{i} ({source})"
        # 也可以在注释中加入所属的 class 信息
        label_text = f"{label_text}\nClass {proto_cls}"
        plt.text(proto[0], proto[1], label_text,
                 fontsize=10, color=class_to_color[proto_cls],
                 horizontalalignment='left', verticalalignment='bottom')

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
