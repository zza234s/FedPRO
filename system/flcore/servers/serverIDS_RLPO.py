import torch
import time
import math
import numpy as np
from flcore.clients.clientIDS_RLPO import clientIDS_RLPO
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict
import time
import copy
import os
import h5py
import torch.nn as nn
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

"""
2025/02/13想到的
大致思路：
在任意联邦学习算法训练完成之后，聚合并生成全局原型，随后将全局原型下发至每个客户端
参考RAG的思路，每个客户端使用全局原型增强表征，实现推理。（或是用集成学习的方式实现推理）---最后采用投票方式
#下面的原型优化在这个版本重未实现
每个客户端上传基于原型推理结果的训练精度至服务器（作为更新全局原型的反馈信息）
服务器基于接收到的反馈信息，生成更优的原型。
"""

"""
实验记录：
简单的检索+特征增强不太行
检索+投票确实能提升训练好的FedAvg的效果

"""


class FedIDS_RLPO(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientIDS_RLPO)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]

        self.cuda_cka = CudaCKA(args.device)
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

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # self.receive_protos()
            # self.global_protos = proto_aggregation(self.uploaded_protos)
            # self.send_protos()

            if self.args.p_sim_cal:
                self.sim_matrix = self.cal_sim_martrix()
                self.receive_models_for_personalized()
                self.generate_personalized_models()
                self.aggregate_parameters()
            else:
                self.receive_models()
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters()

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
        # self.save_global_model()
        # self.plot_acc_curve()

        print('###########################################')
        total_inference_cost = 0
        total_train_cost = 0
        for c in self.selected_clients:
            total_inference_cost += sum(c.inference_cost) / len(c.inference_cost)
            total_train_cost += c.train_time_cost['total_cost'] / c.train_time_cost['num_rounds']
        print("\nAverage training time cost per client per round.")
        print(total_train_cost / len(self.selected_clients))
        print("\nAverage inference time per client per round:")
        print(total_inference_cost / len(self.selected_clients))

    def prototype_plug_train(self):
        for i in range(self.adaptation_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Plus Round number: {i}-------------")
            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            # for client in self.selected_clients:
            #     client.RAF()

            self.RAF_evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        self.save_results()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)
            client.receive_proto_of_other_clients(self.uploaded_ids, self.uploaded_protos)
            client.build_prototype_db()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            client.collect_protos()
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def cal_sim_martrix(self):
        start_time = time.perf_counter()

        num_client = len(self.selected_clients)
        sim_matrix = torch.zeros((num_client, num_client))
        for i in range(num_client):
            for j in range(num_client):
                X = torch.stack(tuple(self.uploaded_protos[i].values()))  # (7,80) tensor
                Y = torch.stack(tuple(self.uploaded_protos[j].values()))  # (7,80) tensor
                sim_matrix[i][j] = self.cuda_cka.linear_CKA(X, Y)

        # Normalize the similarity matrix according to the L1 norm
        norms = sim_matrix.norm(p=1, dim=1, keepdim=True)
        sim_matrix = sim_matrix / norms

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time for calculating similarity matrix: {elapsed_time:0.4f} seconds")
        return sim_matrix

    def receive_models_for_personalized(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_models = []
        self.uploaded_state_dict = []
        for client in self.selected_clients:
            self.uploaded_models.append(client.model)
            self.uploaded_state_dict.append(client.model.state_dict())

    def generate_personalized_models(self):
        assert (len(self.uploaded_models) > 0)
        start_time = time.time()
        for i, c_id in enumerate(self.uploaded_ids):
            aggr_local_model_weights = self.aggregate(self.uploaded_state_dict, self.sim_matrix[i, :])
            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = aggr_local_model_weights
        end_time = time.time()
        print(f"Execution time for generating personalized models: {end_time - start_time:0.4f} seconds")

    def aggregate(self, state_dicts, ratio=None):
        aggr_theta = copy.deepcopy(self.uploaded_models[0])
        for param in aggr_theta.parameters():
            param.data.zero_()

        if ratio is not None:
            for name, params in aggr_theta.named_parameters():
                for j, model_state_dict in enumerate(state_dicts):
                    params.data += model_state_dict[name] * ratio[j]  # TODO: 将state_dicit存储起来优化运行速度

        return aggr_theta

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

            # with h5py.File(file_path, 'w') as hf:
            #     hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            #     hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            #     hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            #     hf.create_dataset('rs_loacal_test_acc', data=self.rs_local_test_acc)
            #     hf.create_dataset('rs_prototype_test_acc', data=self.rs_prototype_test_acc)
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


class Trainable_prototypes(nn.Module):
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
