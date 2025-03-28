import torch
import os

PFL_algo = ['Local', 'FedDBE', 'GPFL', 'FedTGP', "FedGH", "FedProto"]


def weighted_voting(retrieve_prototypes, retrieve_meta_info, rep, num_classes, retrievaled_similarities):
    """
    使用原型的类别信息进行加权投票
    :param retrieve_prototypes: (batch_size, top_k, embed_dim) 原型张量
    :param retrieve_meta_info: (batch_size, top_k) 原型的元信息列表
    :param rep: (batch_size, embed_dim) 查询样本的嵌入
    :param num_classes: 类别数量
    :param sims: (batch_size, top_k) rep和所有原型的相似度
    :return: (batch_size,) 经过加权投票的类别预测
    """
    batch_size, top_k, embed_dim = retrieve_prototypes.shape

    # 计算原型与当前样本的相似度（欧式距离/余弦相似度）
    similarity = retrievaled_similarities

    # 归一化权重（可用 softmax 或者 L1 归一化）
    # weights = torch.softmax(similarity, dim=1)  # (batch_size, top_k)

    # 提取类别信息 (batch_size, top_k)
    class_labels = torch.tensor([[meta['class'] for meta in sample_meta] for sample_meta in retrieve_meta_info],
                                device=rep.device)

    # 计算类别得分
    # num_classes = class_labels.max().item() + 1  # 假设类别是从 0 开始的
    class_scores = torch.zeros((batch_size, num_classes), device=rep.device)

    for i in range(top_k):
        class_scores.scatter_add_(1, class_labels[:, i].unsqueeze(-1), similarity[:, i].unsqueeze(-1))

    # 选取得分最高的类别
    # predicted_labels = class_scores.argmax(dim=1)

    return class_scores



def get_timestamp_folder(folder_path):
    """ 获取指定文件夹下的唯一时间戳文件夹名称 """
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)) and folder.replace('.', '').isdigit():
            return folder  # 直接返回找到的时间戳文件夹名
    return None  # 如果没有找到，返回 None
