import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

def visualize_with_dim_reduction(all_features, all_labels, query_idx,
                                 all_prototypes, all_protos_meta, all_retrieved_meta=None,
                                 mode='tsne', title='Feature Visualization'):
    """
    使用 PCA、t-SNE 或 UMAP 进行降维并可视化所有样本特征和 query 检索到的原型。

    参数:
      - all_features: numpy 数组或 PyTorch 张量, shape = (N, D)，所有样本的特征表示
      - all_labels: list 或 numpy 数组, 长度为 N, 每个元素对应 all_features 中样本的 class
      - query_idx: int, 指定 query 在 all_features 中的索引
      - all_prototypes: numpy 数组或 PyTorch 张量, shape = (number of prototype, D)，query 检索到的原型特征
      - all_protos_meta: list of dict，每个 dict 包含对应原型的元数据信息，如 "class"、"source"、"client_id"
      - all_retrieved_meta: list of list, 长度为 N，每个子 list 包含 top_k 个 dict
      - mode: str, 降维方法，'pca', 'tsne' 或 'umap' (默认 'tsne')
      - title: str, 图像标题

    """

    # 若输入为 PyTorch 张量，则转换为 numpy 数组
    if isinstance(all_features, torch.Tensor):
        all_features = all_features.detach().cpu().numpy()
    if isinstance(all_prototypes, torch.Tensor):
        all_prototypes = all_prototypes.detach().cpu().numpy()

    # 归一化数据
    combined_features = np.concatenate([all_features, all_prototypes], axis=0)
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # 选择降维方法
    if mode == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization' + title
    elif mode == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization' + title
    elif mode == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        title = 'UMAP Visualization' + title
    else:
        raise ValueError("Invalid mode. Choose from ['pca', 'tsne', 'umap']")

    # 进行降维
    combined_2d = reducer.fit_transform(combined_features)

    # 分离降维后的样本点和原型点
    N = all_features.shape[0]
    all_features_2d = combined_2d[:N, :]
    prototypes_2d = combined_2d[N:, :]

    # 获取类别标签
    all_labels = np.array(all_labels)

    if isinstance(all_protos_meta,torch.Tensor):
        proto_labels=np.array(all_protos_meta.cpu()) #假如输入的不是meta，而是原型标签
    else:
        proto_labels = np.array([meta.get('class', 'unknown') for _, meta in all_protos_meta.items()])

    # 分配颜色
    unique_classes = np.unique(np.concatenate([all_labels, proto_labels]))
    cmap = plt.get_cmap("tab10")
    class_to_color = {cls: cmap(i % 10) for i, cls in enumerate(unique_classes)}

    plt.figure(figsize=(10, 8))

    # 绘制样本点
    for cls in np.unique(all_labels):
        indices = np.where(all_labels == cls)[0]
        plt.scatter(all_features_2d[indices, 0],
                    all_features_2d[indices, 1],
                    color=class_to_color[cls], alpha=0.6, label=f'Class {cls}')

    # 高亮 query
    query_point = all_features_2d[query_idx]
    query_cls = all_labels[query_idx]
    plt.scatter(query_point[0], query_point[1],
                color=class_to_color[query_cls], marker='*', s=250,
                edgecolors='black', linewidths=1.5,
                label='Query')

    # 绘制原型点
    for i, proto in enumerate(prototypes_2d):
        proto_cls = proto_labels[i]
        # meta = all_protos_meta[i]
        # source = meta.get('source', 'unknown')
        # marker = 's' if source == 'client' else 'D'  # 客户端: 方形, 全局: 菱形
        marker ='s'
        plt.scatter(proto[0], proto[1],
                    color=class_to_color[proto_cls],
                    marker=marker, s=50,
                    edgecolor='black', linewidth=1)

    plt.title(title)
    plt.legend()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_proto_feedback(feedback_stats, proto_stats, all_protos_meta, client_id):
    """
    可视化联邦学习原型统计信息：
    1. 每个类别的预测情况（总数、错误、正确）
    2. 每个类别选中的原型分布（红色 = 类别匹配，蓝色 = 类别不匹配，深浅直接反映选中次数）
    3. 每个原型的使用情况（正确数、错误数、总选中次数）

    参数：
    - feedback_stats: 按类别统计的信息（total_counts, correct_counts, error_counts, selected_by_class）
    - proto_stats: 按原型统计的信息（total_counts, correct_counts, error_counts）
    - all_protos_meta: 包含原型信息的列表，每个元素是一个字典，包含 CLIENT_ID、class 等信息
    - client_id: 该客户端的 ID
    """
    ### 1. 可视化每个类别的预测情况（柱状图） ###
    class_ids = sorted(feedback_stats.keys())  # 获取所有类别
    total_counts = [feedback_stats[c]["total_counts"] for c in class_ids]
    correct_counts = [feedback_stats[c]["correct_counts"] for c in class_ids]
    error_counts = [feedback_stats[c]["error_counts"] for c in class_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.4  # 柱状宽度
    x = np.arange(len(class_ids))

    ax.bar(x - width / 2, correct_counts, width, label="Correct Predictions", color="blue", alpha=0.7)
    ax.bar(x + width / 2, error_counts, width, label="Error Predictions", color="red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(class_ids)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    ax.set_title(f"Client {client_id}: Prediction Statistics by Class")
    ax.legend()
    plt.show()

    ### 2. 获取 proto_id -> 类别 & 客户端映射 ###
    proto_class_map = {}  # {proto_id: 类别}
    proto_client_map = {}  # {proto_id: client_id}

    for idx, proto in all_protos_meta.items():
        proto_id = proto["idx"]  # 原型 ID
        proto_class_map[proto_id] = proto["class"]  # 原型类别
        proto_client_map[proto_id] = proto["CLIENT_ID"]  # 原型来源客户端 ID

    ### 3. 统计所有选中的原型 ID ###
    unique_prototypes = sorted(set(p for c in class_ids for p in feedback_stats[c]["selected_by_class"].keys()))

    # 构造原型 ID 轴标签
    proto_labels = [f"{p}" for p in unique_prototypes]

    ### 4. 构造热力图数据 ###
    heatmap_data = np.zeros((len(class_ids), len(unique_prototypes)))  # 选中次数
    for i, c in enumerate(class_ids):
        for j, p in enumerate(unique_prototypes):
            selected_count = feedback_stats[c]["selected_by_class"].get(p, 0)
            proto_class = proto_class_map.get(p, -1)  # 获取原型的类别
            if proto_class == c:
                heatmap_data[i, j] = selected_count  # 类别匹配（红色）
            else:
                heatmap_data[i, j] = -selected_count  # 类别不匹配（蓝色）

    ### 5. 自定义颜色映射 ###
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmax = np.max(np.abs(heatmap_data))  # 取最大绝对值
    vmin = -vmax  # 保证蓝色和红色对称

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap=cmap, center=0, vmin=vmin, vmax=vmax,
                xticklabels=proto_labels, yticklabels=class_ids)
    plt.xlabel("Prototype ID")
    plt.ylabel("Class ID")
    plt.title(f"Client {client_id}: Prototype Selection (Red = Class Match, Blue = Mismatch)")
    plt.xticks(rotation=90)
    plt.show()

    ### 6. 原型统计可视化（柱状图） ###
    proto_ids = sorted(proto_stats.keys())
    proto_total = [proto_stats[p]["total_counts"] for p in proto_ids]
    proto_correct = [proto_stats[p]["correct_counts"] for p in proto_ids]
    proto_error = [proto_stats[p]["error_counts"] for p in proto_ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4
    x = np.arange(len(proto_ids))

    ax.bar(x - width / 2, proto_correct, width, label="Correct Selections", color="blue", alpha=0.7)
    ax.bar(x + width / 2, proto_error, width, label="Error Selections", color="orange", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(proto_ids, rotation=90)
    ax.set_xlabel("Prototype ID")
    ax.set_ylabel("Count")
    ax.set_title(f"Client {client_id}: Prototype Selection Statistics (Correct vs Error)")
    ax.legend()
    plt.show()


def visualize_heatmap(tensor):
    tensor = tensor.cpu().numpy()  # 转换为 NumPy
    plt.figure(figsize=(10, 6))
    sns.heatmap(tensor, cmap="coolwarm", annot=False)
    plt.xlabel("D (Feature Dimension)")
    plt.ylabel("N (Samples)")
    plt.title("Tensor Heatmap Visualization")
    plt.show()


def visualize_tsne_results(rep, y, rag_preds, title, P=None, proto_label=None,mode='test'):
    """
    使用t-SNE可视化样本表征和原型，并根据分类结果标记

    参数：
        rep: 样本表征向量
        y: 真实标签
        rag_preds: RAG模型预测的标签
        title: 图表标题和保存的文件名
        P: 原型张量，维度为 (number_of_prototypes, d)
        proto_label: 原型对应的类别标签
        mode: 'train' 或 'test'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    # 将张量转换为NumPy数组
    rep_np = rep.cpu().numpy()
    y_np = y.cpu().numpy()
    rag_preds_np = rag_preds.cpu().numpy()

    # 如果提供了原型，将原型与样本表征合并进行t-SNE
    if P is not None and proto_label is not None:
        P_np = P.cpu().numpy()
        proto_label_np = proto_label.cpu().numpy() if isinstance(proto_label, torch.Tensor) else np.array(proto_label)

        # 合并样本表征和原型
        combined_features = np.vstack([rep_np, P_np])

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        combined_embeddings_2d = tsne.fit_transform(combined_features)

        # 分离样本和原型的降维结果
        embeddings_2d = combined_embeddings_2d[:len(rep_np)]
        proto_embeddings_2d = combined_embeddings_2d[len(rep_np):]
    else:
        # 只对样本表征进行t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(rep_np)

    # 获取不同的类别
    unique_classes = np.unique(np.concatenate([y_np, proto_label_np]) if P is not None else y_np)
    # 创建颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

    plt.figure(figsize=(10, 8))

    # 为每个类别绘制样本点
    for i, cls in enumerate(unique_classes):
        # 找出属于该类别的样本索引
        idx = y_np == cls

        # 绘制正确分类的样本（圆点）
        correct = (rag_preds_np == y_np) & idx
        if np.any(correct):
            plt.scatter(
                embeddings_2d[correct, 0],
                embeddings_2d[correct, 1],
                c=[colors[i]],
                marker='o',
                label=f'Class {cls}',
                alpha=0.7
            )

        # 绘制错误分类的样本（红色叉号）
        incorrect = (rag_preds_np != y_np) & idx
        if np.any(incorrect):
            plt.scatter(
                embeddings_2d[incorrect, 0],
                embeddings_2d[incorrect, 1],
                c=[colors[i]],
                marker='x',
                s=100,
                edgecolors='red',
                linewidth=2,
                alpha=0.7
            )

    # 如果提供了原型，使用三角形标记绘制原型
    if P is not None and proto_label is not None:
        for i, cls in enumerate(unique_classes):
            # 找出属于该类别的原型索引
            proto_idx = proto_label_np == cls
            if np.any(proto_idx):
                plt.scatter(
                    proto_embeddings_2d[proto_idx, 0],
                    proto_embeddings_2d[proto_idx, 1],
                    c=[colors[i]],
                    marker='^',  # 三角形标记
                    s=150,  # 更大的尺寸
                    edgecolors='black',
                    linewidth=1.5,
                    alpha=0.9,
                    label=f'Proto {cls}'
                )

    # 设置图表标题和标签
    plt.title(f"{title}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # 创建一个没有重复标签的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{title}.png")
    plt.show()
    plt.close()

    return embeddings_2d


# def visualize_tsne_with_class(all_features, all_labels, query_idx,
#                               all_prototypes,all_protos_meta, all_retrieved_meta,
#                               title='TSNE Visualization'):
#     """
#     使用 TSNE 降维并可视化所有样本特征（all_features）和 query 检索到的原型，
#     并根据不同 class 用不同颜色显示，同时在原型点旁标注出原型编号、所属 class 及其来源信息。
#
#     参数:
#       all_features: numpy 数组或 PyTorch 张量, shape = (N, D)
#                     所有样本的特征表示。
#       all_labels: list 或 numpy 数组, 长度为 N, 每个元素对应 all_features 中样本的 class。
#       query_idx: int, 指定 query 在 all_features 中的索引。
#       all_prototypes: numpy 数组或 PyTorch 张量, shape = (number of prototype, D)
#                                   query 检索到的原型特征。
#       all_retrieved_meta: list of list, 长度为 N，每个子list里包含topk个dict，每个dict包含三个key-value对:
#                               "class": 原型所属 class，
#                               "source": 原型来源（"global" 或 "client"），
#                               若 "source" 为 "client"，可包含 "client_id"，如果为"global",则对应为None
#       title: str, 图像标题 (默认 'TSNE Visualization')。
#     """
#     # 如果输入为 PyTorch 张量，则转换为 numpy 数组
#     if isinstance(all_features, torch.Tensor):
#         all_features = all_features.detach().cpu().numpy()
#     if isinstance(all_prototypes, torch.Tensor):
#         all_prototypes = all_prototypes.detach().cpu().numpy()
#
#     # 将所有特征与检索到的原型合并，用于 TSNE 降维
#     combined_features = np.concatenate([all_features, all_prototypes], axis=0)
#
#     tsne = TSNE(n_components=2, random_state=42)
#     combined_2d = tsne.fit_transform(combined_features)
#
#     # 分离降维后所有特征和原型对应的二维点
#     N = all_features.shape[0]
#     all_features_2d = combined_2d[:N, :]
#     prototypes_2d = combined_2d[N:, :]
#
#     # 从 all_labels 和 query_retrieved_meta 中提取所有 class 信息
#     all_labels = np.array(all_labels)
#     # 对于原型，我们假设 meta 中包含 "class" 键
#     proto_labels = np.array([meta.get('class', 'unknown') for meta in all_protos_meta])
#
#     # 计算所有出现的 class
#     unique_classes = np.unique(np.concatenate([all_labels, proto_labels]))
#
#     # 为不同的 class 分配不同颜色，使用 matplotlib 的 tab10 调色板
#     cmap = plt.get_cmap("tab10")
#     class_to_color = {cls: cmap(i % 10) for i, cls in enumerate(unique_classes)}
#
#     plt.figure(figsize=(10, 8))
#
#     # 绘制所有样本特征，根据所属 class 采用不同颜色
#     for cls in np.unique(all_labels):
#         indices = np.where(all_labels == cls)[0]
#         plt.scatter(all_features_2d[indices, 0],
#                     all_features_2d[indices, 1],
#                     color=class_to_color[cls],
#                     alpha=0.6,
#                     label=f'Class {cls}')
#
#     # 高亮指定 query 点（从 all_features 中提取）
#     query_point = all_features_2d[query_idx]
#     plt.scatter(query_point[0], query_point[1],
#                 color='red', marker='*', s=250,
#                 edgecolors='black', linewidths=1.5,
#                 label='Query')
#
#     # 绘制所有原型，颜色与其所属 class 一致，标记为菱形，并在旁边添加注释
#     for i, proto in enumerate(prototypes_2d):
#         proto_cls = proto_labels[i]
#         plt.scatter(proto[0], proto[1],
#                     color=class_to_color[proto_cls],
#                     marker='D', s=150,
#                     edgecolor='black', linewidth=1)
#
#     # 基于all_retrieved_meta, 高亮检索到的原型
#     retrieved_meta = all_retrieved_meta[query_idx]
#
#
#     # source = meta.get('source', 'unknown')
#     # # 如果来源为 client，则附带 client_id
#     # if source == 'client':
#     #     client_id = meta.get('client_id', 'unknown')
#     #     label_text = f"P{i} (client: {client_id})"
#     # else:
#     #     label_text = f"P{i} ({source})"
#     # # 也可以在注释中加入所属的 class 信息
#     # label_text = f"{label_text}\nClass {proto_cls}"
#     # plt.text(proto[0], proto[1], label_text,
#     #          fontsize=10, color=class_to_color[proto_cls],
#     #          horizontalalignment='left', verticalalignment='bottom')
#     #
#     # plt.title(title)
#     # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     # plt.tight_layout()
#     # plt.show()