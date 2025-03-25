import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict

from sklearn import metrics
from utils.proto_rag import *
from sklearn.preprocessing import label_binarize

class clientTGP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            model = self.model
            save_item(model, self.role, 'model', self.save_folder_name)
        self.skip_fl = args.skip_FL_training
        self.glbal_proto_for_check =None
    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    # we use MSE here following FedProto's official implementation, where lamda is set to 10 by default.
                    # see https://github.com/yuetan031/FedProto/blob/main/lib/update.py#L171
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.collect_protos()
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

    def test_metrics(self):
        testloader = self.load_test_data()
        if not self.skip_fl:
            model = load_item(self.role, 'model', self.save_folder_name)
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
            self.model= model
            self.global_protos = global_protos
        else:
            model = self.model
            global_protos = self.global_protos
        model.eval()

        test_acc = 0
        test_num = 0

        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        if not self.skip_fl:
            model = load_item(self.role, 'model', self.save_folder_name)
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
            self.model= model
            self.global_protos = global_protos
        else:
            model = self.model
            global_protos = self.global_protos

        # model = load_item(self.role, 'model', self.save_folder_name)
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
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
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def RAF(self, mode='train'):

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

        self.model.eval()

        self.proto_db = self.proto_db.to(self.device)

        for idx, meta in self.proto_meta.items():
            self.proto_meta[idx]['weight'] = 1.0

        start_time = time.time()

        global_protos = self.global_protos #在读取训练好模型的时候也读取了保存的全局原型


        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                retrieve_prototypes, retrieve_meta_info, indices_np, retrievaled_similarities = self.weighted_retrieve_prototypes(
                    rep, topk=self.topk)  # (B, top_k, embed_dim)
                rag_output = weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, self.num_classes,
                                             retrievaled_similarities)
                proto_conf, _ = torch.max(retrievaled_similarities, dim=1)

                local_model_output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                for i, r in enumerate(rep):
                    for j, pro in global_protos.items():
                        if type(pro) != type([]):
                            local_model_output[i, j] = self.loss_mse(r, pro)


                alpha = (1.0 - proto_conf).unsqueeze(1)
                ensemble_output = (1 - alpha) * rag_output + alpha * local_model_output*(-1)


                correct_local += (torch.sum(torch.argmin(local_model_output, dim=1) == y)).item() #这里和一般做法不同，一般是取最大值。但是FedTGP源代码相似度计算用mse,越小越相似
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
# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos