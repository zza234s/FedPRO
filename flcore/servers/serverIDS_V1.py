import torch
import time
import math
import numpy as np
from flcore.clients.clientIDS_V1 import clientIDS_V1
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict
import time
import copy
import os
import h5py
import math
import torch.nn as nn
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

"""
RLPO版本已实现部分：
在任意联邦学习算法训练完成之后，聚合并生成全局原型，随后将全局原型以及每个客户都安的本地原型广播至至每个客户端
生成topk个基于原型的预测结果，投票产生最终的分类结果

V1版本拟实现思路
1. 上传聚类原型至服务器端 （√）
2. 尝试在服务器端直接基于客户都安的反馈生成更好的原型
3. 是否能引入额外的后训练步骤，增强集成/投票的精度
"""

"""
实验结果
25/02/18: 上传聚类原型，每个客户端仅用所有客户端的聚类原型投票推理，NSLKDD_GLOB_0.5设定上精度增加----后续尝试发现结果不稳定，可能受聚类算法影响
"""


class FedIDS_V1(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientIDS_V1)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        # self.global_protos = [None for _ in range(args.num_classes)]
        self.global_protos = None

        self.sd = {}

        self.rs_local_test_acc = []
        self.rs_prototype_test_acc = []
        self.rs_all_lam = []
        self.rs_all_gamma = []
        self.rs_rag_test_acc = []
        self.rs_en_test_acc =[]

        self.current_round = 0
        self.best_round = 0
        self.best_metrics = defaultdict(float)

        self.adaptation_rounds = args.adaptation_rounds
        self.alpha = 1.0
        self.PrototypeRanker = PrototypeRanker(input_dim=2)
    def prototype_plug_train(self):
        s_t = time.time()
        self.selected_clients = self.select_clients()
        self.send_models()
        self.receive_protos()
        self.build_prototype_db()
        self.send_prototype_db()

        for i in range(self.adaptation_rounds):
            print(f"\n-------------Plus Round number: {i}-------------")
            self.RAF_evaluate()
            self.receive_proto_feedback()
            self.aggregate_feedback()
            self.update_prototype_weights()
            # self.update_proto_db()
            self.send_prototype_db()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        self.save_results()

    def send_protos(self):
        # 发送全局原型以及每个客户端的本地原型至所有客户端
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)
            client.receive_proto_of_other_clients(self.uploaded_ids, self.uploaded_protos)
            client.build_prototype_db()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def send_prototype_db(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_proto_db(proto_db=self.proto_db, proto_meta=self.proto_meta)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_proto_feedback(self):
        assert (len(self.selected_clients) > 0)

        self.proto_feedback_ids = []
        self.proto_feedback_list = []
        for client in self.selected_clients:
            client.get_proto_feedback()
            self.proto_feedback_ids.append(client.id)
            self.proto_feedback_list.append(client.feedback_stats)
    def aggregate_feedback(self):
        """
        聚合来自多个客户端的反馈信息
        :param client_feedback_list: 客户端反馈信息列表，每个元素为客户端的 feedback_stats 字典
        :return: aggregated_feedback: 聚合后的反馈字典
        """
        aggregated_feedback = {}
        for feedback in self.proto_feedback_list:
            for proto_id, stats in feedback.items():
                if proto_id not in aggregated_feedback:
                    aggregated_feedback[proto_id] = {
                        "total_counts": 0,
                        "error_counts": 0,
                        "correct_counts": 0,

                    }
                aggregated_feedback[proto_id]["total_counts"] += stats["total_counts"]
                aggregated_feedback[proto_id]["error_counts"] += stats["error_counts"]
                aggregated_feedback[proto_id]["correct_counts"] += stats["correct_counts"]

        z = 1.96
        for proto_id, stats in aggregated_feedback.items():
            n = stats["total_counts"]
            if n > 0:
                # 正确的数量
                correct = n - stats["error_counts"]
                p_hat = correct / n
                denominator = 1 + (z ** 2) / n
                numerator = p_hat + (z ** 2) / (2 * n) - z * math.sqrt(
                    (p_hat * (1 - p_hat)) / n + (z ** 2) / (4 * n ** 2))
                stats["acc_gain"] = numerator / denominator
            else:
                stats["acc_gain"] = 0


            self.proto_meta[proto_id]['acc_gain'] = stats['acc_gain']
            self.proto_meta[proto_id]['error_counts'] = stats['error_counts']
            self.proto_meta[proto_id]['total_counts'] = stats['total_counts']
            self.proto_meta[proto_id]['correct_counts'] = stats['correct_counts']

        self.agg_proto_feedback = aggregated_feedback

    def update_prototype_weights(self, k=10, b=0.5, alpha=0.1):
        def sigmoid(x, k=10, b=0.5):
            """Sigmoid 函数控制权重变化"""
            return 1 / (1 + math.exp(-k * (x - b)))

        for proto_meta in self.proto_meta:
            p_i = proto_meta['acc_gain']  # 使用 acc_gain 作为调整依据

            # 计算 Sigmoid 控制的更新系数
            S_p = sigmoid(p_i, k=k, b=b)

            # 计算最终权重
            new_weight = proto_meta['weight'] * S_p #+ alpha * (1 - S_p)

            # 转换为 tensor，并限制范围
            new_weight = torch.tensor(new_weight, dtype=torch.float32).to(self.device)
            new_weight = torch.clamp(new_weight, min=0.01, max=1.0)

            # 更新原型权重
            proto_meta['weight'] = new_weight

        # return [proto['weight'] for proto in self.proto_meta]

    def update_proto_db(self):
        pass
        """
        根据聚合的反馈信息更新原型。
        更新公式： p_j = p_j + alpha * (平均偏差)
        """
        for proto_id, stats in self.agg_proto_feedback.items():
            self.proto_meta[proto_id]['acc_gain'] = stats['acc_gain']
            self.proto_meta[proto_id]['acc_gain'] = stats['error_counts']
            self.proto_meta[proto_id]['acc_gain'] = stats['total_counts']

        print(f'已更新原型db')

    def send_local_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.receive_proto_of_other_clients(self.uploaded_ids, self.uploaded_protos)
            client.build_prototype_db()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            client.local_cluster_for_proto_gen()
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def build_prototype_db(self):
        all_protos = []  # 存放所有原型的张量
        all_meta = []  # 存放每个原型对应的类别信息
        proto_idx = 0
        if self.global_protos is not None:
            for cls, proto in self.global_protos.items():
                meta_info = {
                    'idx': proto_idx,
                    'class': cls,
                    'source': 'global',
                    'CLIENT_ID': None,
                    'acc_gain': 1.0,
                    'weight':1.0,
                    'error_counts': 0,
                    'correct_counts':0,
                    'total_counts': 0
                }
                all_protos.append(proto)
                all_meta.append(meta_info)
                proto_idx += 1

        for client_id, client_proto in zip(self.uploaded_ids, self.uploaded_protos):
            for cls, proto in client_proto.items():
                if type(proto) == type([]):
                    for i in range(len(proto)):
                        meta_info = {
                            'idx': proto_idx,
                            'class': cls,
                            'source': 'client',
                            'CLIENT_ID': client_id,
                            'weight': 1.0,
                            'acc_gain': 1.0,
                            'error_counts': 0,
                            'correct_counts': 0,
                            'total_counts': 0
                        }
                        all_protos.append(proto[i].squeeze())
                        all_meta.append(meta_info)
                        proto_idx += 1
                else:
                    meta_info = {
                        'idx': proto_idx,
                        'class': cls,
                        'source': 'client',
                        'CLIENT_ID': client_id,
                        'weight': 1.0,
                        'acc_gain': 1.0,
                        'error_counts': 0,
                        'correct_counts': 0,
                        'total_counts': 0
                    }
                    all_protos.append(proto[i])
                    all_meta.append(meta_info)
                    proto_idx += 1

        self.proto_meta = all_meta
        self.proto_db = torch.stack(all_protos)


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            if f'personalized_{client.id}' in self.sd:
                model = self.sd[f'personalized_{client.id}']
            else:
                model = self.global_model
                print(f'send the global model to client:{client.id}')
            client.set_parameters(model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        # indices = np.mean(np.stack(stats[4]), axis=0)

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
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))

        ###########################
        all_lam = stats[6]
        all_gamma = stats[7]
        local_test_acc = sum(stats[4]) * 1.0 / sum(stats[1])
        protoype_ts_acc = sum(stats[5]) * 1.0 / sum(stats[1])
        rag_acc = sum(stats[8]) * 1.0 / sum(stats[1])

        print(f'Averaged Local Test Accurancy: {local_test_acc}')
        print(f'Averaged Prototype Test Accurancy: {protoype_ts_acc}')
        print(f'Averaged RAG Test Accurancy: {rag_acc}')
        # print(f'all_lambda: {all_lam}')
        # print(f'all_gamma: {all_gamma}')
        # self.res_IDS_indicators.append(indices)
        self.rs_local_test_acc.append(local_test_acc)
        self.rs_prototype_test_acc.append(protoype_ts_acc)
        self.rs_rag_test_acc.append(rag_acc)
        self.rs_all_lam.append(all_lam)
        self.rs_all_gamma.append(all_gamma)
        self.res_std.append(np.std(accs))


        if self.save_best_global_model and  test_acc > self.best_metrics['test_acc']:
            self.best_metrics['test_acc'] = test_acc
            best_acc_str = f"{test_acc:.4f}"
            if not os.path.exists(self.save_folder_name):
                os.makedirs(self.save_folder_name)
            save_path = os.path.join(self.save_folder_name, f"best_global_model_{best_acc_str}.pt")
            torch.save(self.global_model.state_dict(), save_path)
            print(f"Save the best global model at: {save_path}")

    def save_item(self, item, item_name):
        file_path = f"../results/{self.dataset}_{self.algorithm}_{self.goal}_{self.times}.h5"
        with h5py.File(file_path, 'a') as hf:
            hf.create_dataset(item_name, data=item)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_indicators = []
        total_correct_local = []
        total_correct_prototype = []
        total_correct_rag = []
        all_lam = []
        all_gamma = []
        for c in self.clients:
            ct, ns, auc, correct_local, correct_prototype, lam, gamma, correct_rag = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            all_lam.append(lam)
            all_gamma.append(gamma)
            total_correct_local.append(correct_local * 1.0)
            total_correct_prototype.append(correct_prototype * 1.0)
            total_correct_rag.append(correct_rag * 1.0)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, total_correct_local, total_correct_prototype, all_lam, all_gamma, total_correct_rag

    def RAF_evaluate(self, acc=None, loss=None):
        stats = self.RAF_metrics_test()

        ensemble_test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        local_test_acc = sum(stats[3]) * 1.0 / sum(stats[1])
        protoype_test_acc = sum(stats[4]) * 1.0 / sum(stats[1])
        rag_test_acc = sum(stats[5]) * 1.0 / sum(stats[1])

        print("Averaged Test Accurancy: {:.4f}".format(ensemble_test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print(f'Averaged Local Test Accurancy: {local_test_acc}')
        print(f'Averaged Prototype Test Accurancy: {protoype_test_acc}')
        print(f'Averaged RAG Test Accurancy: {rag_test_acc}')

        self.rs_en_test_acc.append(ensemble_test_acc)
        self.rs_local_test_acc.append(local_test_acc)
        self.rs_rag_test_acc.append(rag_test_acc)
        self.res_std.append(np.std(accs))

        if acc == None:
            self.rs_test_acc.append(ensemble_test_acc)


    def RAF_metrics_test(self):
        num_samples = []

        tot_auc_ensmble = []
        tot_correct_local = []
        tot_correct_prototype = []
        tot_correct_rag = []
        tot_correct_ensemble = []

        for c in self.clients:
            res = c.RAF(mode='test')

            tot_correct_ensemble.append(res['ensemble_acc'] * 1.0)
            tot_auc_ensmble.append(res['auc'] * res['sample_num'] )
            num_samples.append(res['sample_num'] )
            tot_correct_local.append(res['correct_local'] * 1.0)
            tot_correct_prototype.append(res['correct_prototype'] * 1.0)
            tot_correct_rag.append(res['correct_rag'] * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct_ensemble, tot_correct_local,tot_correct_prototype,tot_correct_rag, tot_auc_ensmble

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_local_test_acc', data=self.rs_local_test_acc)
                hf.create_dataset('rs_en_test_acc', data=self.rs_en_test_acc)
                hf.create_dataset('rs_rag_test_acc', data=self.rs_rag_test_acc)
                # hf.create_dataset('rs_prototype_test_acc', data=self.rs_prototype_test_acc)

        if len(self.res_std):
            folder = "../results/acc_and_std/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = folder + self.dataset + "_" + self.algorithm + "_" + self.goal + "_" + str(self.times)
            np.save(path + "std.npy", np.array(self.res_std))
            np.save(path + "acc.npy", np.array(self.rs_test_acc))

    def plot_acc_curve(self, interval=1):
        """
        绘制测试精度曲线图

        参数：
        - interval: int, 每隔多少个点取一次数据，默认为 1。
        """
        # 从类属性中获取三种测试精度数据
        final_test_acc = self.rs_test_acc
        local_test_acc = self.rs_local_test_acc
        prototype_test_acc = self.rs_prototype_test_acc

        # 根据 interval 对数据采样，并构造对应的 x 轴数据
        final_test_acc = final_test_acc[::interval]
        local_test_acc = local_test_acc[::interval]
        # prototype_test_acc = prototype_test_acc[::interval]
        rounds = np.arange(len(final_test_acc))

        plt.figure(figsize=(8, 6))

        # 绘制三条曲线
        plt.plot(rounds, final_test_acc, label='Final Test Accuracy',
                 color='blue', linestyle='-', marker='o', markersize=5)
        plt.plot(rounds, local_test_acc, label='Local Test Accuracy',
                 color='red', linestyle='-', marker='s', markersize=5)
        # plt.plot(rounds, prototype_test_acc, label='Prototype Test Accuracy',
        #          color='green', linestyle='-', marker='^', markersize=5)
        plt.plot(rounds, self.rs_rag_test_acc[::interval], color='green', label='RAG Test Accuracy')
        # 设置图形属性
        plt.xlabel('Number of FL Rounds', fontsize=16)
        plt.ylabel('Test Accuracy', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.tight_layout()  # 自动调整布局
        plt.show()
    def load_global_model(self,path):
        state_dict = torch.load(path)
        self.global_model.load_state_dict(state_dict)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


class PrototypeRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出排序得分
        )

    def forward(self, prototype, usage_count, accuracy_gain):
        combined = torch.cat([prototype, usage_count, accuracy_gain])
        return self.mlp(combined)