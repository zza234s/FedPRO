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
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from collections import defaultdict
from flcore.trainmodel.models import *
import math
import re
from utils.proto_rag import *
from utils.proto_rag import PFL_algo
from utils.vis_utils import *
from utils.finch import FINCH
import pandas as pd


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        ###########################
        self.res_IDS_indicators = []
        self.res_std = []
        self.rs_rag_test_acc = []
        self.rs_en_test_acc = []
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
            args.save_folder_without_time = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        elif args.save_folder_name == 'autodl':
            args.save_folder_name_full = f'../../autodl-tmp/{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
            args.save_folder_without_time = f'../../autodl-tmp/{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        elif args.save_folder_name == 'zhl' or 'GPFL' in args.save_folder_name:
            args.save_folder_name_full = f'./{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
            args.save_folder_without_time = f'./{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full
        self.best_metrics = defaultdict(float)
        self.save_best_global_model = args.save_best_global_model
        self.global_prototype = None
        self.global_protos = None
        self.adaptation_rounds = args.adaptation_rounds
        self.rs_local_test_acc = []

        self.plug_result = dict()
        self.plug_topk = args.topk

        # proto db
        self.proto_idx = 0
        self.all_proto = []
        self.proto_meta = dict()

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

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
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        if len(self.rs_test_acc):
            folder = "../results/acc_and_std/"
            path = folder + self.dataset + "_" + self.algorithm + "_" + self.goal + "_" + str(self.times)
            np.save(path + "acc.npy", np.array(self.rs_test_acc))

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
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
            if not os.path.exists(self.save_folder_name):
                os.makedirs(self.save_folder_name)
            save_path = os.path.join(self.save_folder_name, f"best_global_model_{best_acc_str}.pt")
            torch.save(self.global_model.state_dict(), save_path)
            print(f"Save the best global model at: {save_path}")

            if self.global_prototype is not None:
                save_path = os.path.join(self.save_folder_name, f"best_global_prototype_{best_acc_str}.pt")
                torch.save(self.global_prototype, save_path)
                print(f"Save the best global prototype at: {save_path}")

        self.plug_result['best_acc_inital'] = test_acc

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def load_best_pesonlized_model(self):
        algo_path = os.path.join(self.args.best_model_dir, self.dataset, self.algorithm)  # TODO:删除time编号
        timestamp_folder_path = get_timestamp_folder(algo_path)
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

            # 强制把model拆分为表征层和决策层
            if not isinstance(c.model, BaseHeadSplit):
                head = copy.deepcopy(c.model.fc)
                c.model.fc = nn.Identity()
                c.model = BaseHeadSplit(c.model, head)

        print(f'最佳权重文件读取完成, 开始评估')
        self.evaluate()
        print('-' * 50)

    ###########################
    def prototype_plug_train(self):
        use_cluster = True
        use_global_protos = False
        server_cluster = False

        self.selected_clients = self.select_clients()

        self.receive_cluster_protos()
        # self.receive_local_protos()
        self.build_prototype_db(use_cluster=use_cluster, use_global_protos=use_global_protos)
        # self.save_item(self.proto_db, 'proto_db_before_optimization')
        # self.save_item(self.proto_meta, 'proto_meta_before_optimization')


        # self.build_prototype_db(use_cluster=self.args.use_cluster_proto, use_global_protos=self.args.use_global_proto)
        self.send_prototype_db()
        # self.RAF_evaluate()
        # self.receive_proto_feedback(mode='train', mask_other_client_proto=True,phase="before",visualize=True) #todo：保存每个client的embedding

        self.refine_then_upload_protos_for_clients(proto_epoch=self.args.proto_refine_epoch)

        self.build_prototype_db(use_cluster=use_cluster, use_global_protos=use_global_protos,
                                server_cluster=server_cluster)
        self.send_prototype_db()
        # self.save_item(self.proto_db, 'proto_db_after_optimization')
        # self.save_item(self.proto_meta, 'proto_meta_after_optimization')


        # self.receive_proto_feedback(mode='train', mask_other_client_proto=True,phase="after",visualize=True)

        self.RAF_evaluate()
        # self.vis_prototye_effectiveness()

    def vis_prototye_effectiveness(self):
        for client in self.clients:
            client.vis_proto_effectiveness()



        # self.receive_proto_feedback(mode='test')
        # for i in range(self.adaptation_rounds):
        #     print(f"\n-------------Plus Round number: {i}-------------")
        #     self.RAF_evaluate()
        #     self.receive_proto_feedback()
        #     self.aggregate_feedback()
        #     self.update_prototype_weights()
        #     # self.update_proto_db()
        #     self.send_prototype_db()
        #
        #     if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
        #         break
        # self.save_plug_results()
        # self.save_results()

    def receive_local_protos(self):
        self.global_protos = []
        self.lcp_ids = []
        for client in self.clients:
            client.collect_protos()
            self.lcp_ids.append(client.id)
            self.global_protos.append(client.local_mean_protos)

    def receive_cluster_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            client.local_cluster_for_proto_gen()
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def add_proto_to_db(self, protos, cls, source, client_id=None, ):
        meta_info = {
            'idx': self.proto_idx,
            'class': cls,
            'source': source,
            'CLIENT_ID': client_id,
            'acc_gain': 1.0,
            'weight': 1.0,
            'error_counts': 0,
            'correct_counts': 0,
            'total_counts': 0
        }
        self.proto_meta[self.proto_idx] = meta_info
        self.all_proto.append(protos)
        self.proto_idx += 1

    def build_prototype_db(self, use_cluster=False, use_global_protos=False, server_cluster=False):
        reps_dict = defaultdict(list)

        self.proto_meta = dict()
        self.all_proto = []
        self.proto_idx = 0
        if use_global_protos:
            for client_id, client_proto in zip(self.lcp_ids, self.global_protos):
                for cls, proto in client_proto.items():
                    if not server_cluster:
                        self.add_proto_to_db(proto, cls, 'global_mean', client_id)
                    else:
                        reps_dict[cls].append(proto)

        if use_cluster:
            for client_id, client_proto in zip(self.uploaded_ids, self.uploaded_protos):
                for cls, proto in client_proto.items():
                    for i in range(len(proto)):
                        if not server_cluster:
                            self.add_proto_to_db(proto[i].squeeze(), cls, 'client', client_id)
                        else:
                            reps_dict[cls].append(proto[i].squeeze())

        if server_cluster:
            agg_cluster_protos = dict()
            for cls, protos in reps_dict.items():
                protos_np = torch.stack(protos).detach().cpu().numpy()
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

                    proto_tensor = torch.tensor(proto).squeeze()
                    agg_selected_proto.append(proto_tensor)
                    self.add_proto_to_db(proto_tensor, cls, 'server_cluster', None)

                agg_cluster_protos[cls] = agg_selected_proto

        self.proto_db = torch.stack(self.all_proto)

    def send_prototype_db(self):
        assert (len(self.clients) > 0)
        print(f"本次类原型数量{len(self.proto_db)}")
        for client in self.clients:
            start_time = time.time()
            client.proto_db = copy.deepcopy(self.proto_db)
            client.proto_meta = copy.deepcopy(self.proto_meta)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def RAF_evaluate(self, acc=None, loss=None):
        stats = self.RAF_metrics_test()

        ensemble_test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        local_accs = [a / n for a, n in zip(stats[3], stats[1])]
        en_accs = [a / n for a, n in zip(stats[2], stats[1])]

        local_test_acc = sum(stats[3]) * 1.0 / sum(stats[1])
        protoype_test_acc = sum(stats[4]) * 1.0 / sum(stats[1])
        rag_test_acc = sum(stats[5]) * 1.0 / sum(stats[1])
        print(f"local_test_acc:{local_accs}")
        print(f"rag_test_acc:{[a / n for a, n in zip(stats[5], stats[1])]}")
        print(f"ensemble_test_acc:{en_accs}")

        print("Averaged Test Accurancy: {:.4f}".format(ensemble_test_acc))
        # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print(f'Averaged Local Test Accurancy: {local_test_acc}')
        print(f'Averaged Prototype Test Accurancy: {protoype_test_acc}')
        print(f'Averaged RAG Test Accurancy: {rag_test_acc}')

        self.rs_en_test_acc.append(ensemble_test_acc)
        self.rs_local_test_acc.append(local_test_acc)
        self.rs_rag_test_acc.append(rag_test_acc)
        self.res_std.append(np.std(local_accs))

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
            tot_auc_ensmble.append(res['auc'] * res['sample_num'])
            num_samples.append(res['sample_num'])
            tot_correct_local.append(res['correct_local'] * 1.0)
            tot_correct_prototype.append(res['correct_prototype'] * 1.0)
            tot_correct_rag.append(res['correct_rag'] * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct_ensemble, tot_correct_local, tot_correct_prototype, tot_correct_rag, tot_auc_ensmble

    def receive_proto_feedback(self, mode='test', mask_other_client_proto=False, phase="before", visualize=False):
        assert (len(self.selected_clients) > 0)

        self.proto_feedback_ids = []
        self.proto_feedback_list = []
        self.proto_class_stats_list = []

        tot_correct_local = []
        tot_num = []
        tot_correct_rag = []

        for client in self.clients:
            l_correct, rag_correct, sample_num = client.get_proto_feedback(mode,
                                                                           mask_other_client_proto=mask_other_client_proto,
                                                                           visualize=visualize,
                                                                           phase=phase)
            self.proto_feedback_ids.append(client.id)
            self.proto_feedback_list.append(client.proto_stats)
            self.proto_class_stats_list.append(client.feedback_stats)
            # if client.id ==2:
            #     visualize_proto_feedback(client.feedback_stats,client.proto_stats,client.proto_meta,client.id)

            tot_correct_local.append(l_correct)
            tot_correct_rag.append(rag_correct)
            tot_num.append(sample_num)

        print('-' * 50)
        print(f'client：{self.proto_feedback_ids} 的训练精度信息：')
        local_accs = [a / n for a, n in zip(tot_correct_local, tot_num)]
        rag_accs = [a / n for a, n in zip(tot_correct_rag, tot_num)]
        print(f'local_accs:{local_accs}')
        print(f'rag_accs:{rag_accs}')
        print('-' * 50)

    def refine_then_upload_protos_for_clients(self, proto_epoch):
        self.uploaded_ids = []
        self.uploaded_protos = []
        for c in self.clients:
            c.refine_proto_db(proto_epoch)
            self.uploaded_ids.append(c.id)
            self.uploaded_protos.append(c.protos)

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
            new_weight = proto_meta['weight'] * S_p  # + alpha * (1 - S_p)

            # 转换为 tensor，并限制范围
            new_weight = torch.tensor(new_weight, dtype=torch.float32).to(self.device)
            new_weight = torch.clamp(new_weight, min=0.01, max=1.0)

            # 更新原型权重
            proto_meta['weight'] = new_weight

        # return [proto['weight'] for proto in self.proto_meta]

    def load_best_global_model(self):
        algo_path = os.path.join(self.args.best_model_dir, self.dataset, self.algorithm)  # TODO:删除time编号
        timestamp_folder_path = get_timestamp_folder(algo_path)
        file_path = os.path.join(algo_path, timestamp_folder_path)
        assert os.path.exists(file_path), f"Model path {file_path} does not exist!"

        best_acc = 0.0
        # 解析所有文件名，找到每个客户端的最佳精度
        for filename in os.listdir(file_path):
            match = re.search(r'best_global_model_(\d+\.\d+)\.pt', filename)
            if match:
                acc = float(match.group(1))  # 提取精度数值
                if acc > best_acc:
                    best_acc = acc
                    best_model = filename

        print("最佳精度记录:", best_acc)

        # 加载最佳精度对应的文件
        model_file_path = os.path.join(file_path, best_model)
        state_dict = torch.load(model_file_path)

        self.global_model.load_state_dict(state_dict)

        for c in self.clients:
            c.model.load_state_dict(state_dict)
            # 强制把model拆分为表征层和决策层
            if not isinstance(c.model, BaseHeadSplit):
                head = copy.deepcopy(c.model.fc)
                c.model.fc = nn.Identity()
                c.model = BaseHeadSplit(c.model, head)
        print(f'最佳权重文件读取完成, 开始评估')
        self.evaluate()
        print('-' * 50)

    import os
    import pandas as pd

    def save_plug_results(self, save_dir="Plugins_Results"):
        """
        保存测试结果到 CSV 文件，确保路径存在，并按列存储数据。

        参数：
        - save_dir: 结果保存的目录（默认 'Plugins_Results'）
        """
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 定义 CSV 文件路径
        csv_filename = os.path.join(save_dir, f"{self.algorithm}_results.csv")

        # 组织数据（现在数据集是列，指标是行）
        data = {
            "topk": self.args.topk,
            "dataset": self.dataset,
            "local_aacuracy": self.rs_local_test_acc[-1] if self.rs_local_test_acc else None,
            "en_accuracy": self.rs_en_test_acc[-1] if self.rs_en_test_acc else None,
        }

        # 新数据的 DataFrame
        new_df = pd.DataFrame([data])  # 这里要用列表来确保是 DataFrame 而非 Series

        if os.path.exists(csv_filename):
            # 读取已有的 CSV 文件
            existing_df = pd.read_csv(csv_filename)

            # 追加新数据（不覆盖）
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df  # 直接使用新数据

        # 保存回 CSV
        combined_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")