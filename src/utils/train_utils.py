from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger(__name__)

plt.rcParams.update({
    "font.family": "serif",        # 使用衬线字体
    "font.serif": ["Times New Roman"], # 指定Times New Roman
    "font.weight": "bold"        # 全局文本加粗
})


# def SaveMetaphorTypeFigure(args, tsne_dict):
#     # 标签映射
#     label_mapping = {
#         0: "Verb",
#         1: "Attribute",
#         2: "Noun"
#     }
#     X = tsne_dict['metaphor_type_vecs'][-2000:]  # 保持原始数据截取方式
#     Y = tsne_dict['metaphor_type_labels'][-2000:]
#
#     # 数据预处理
#     X = np.array(X)
#     Y = np.array(Y)
#
#     # 优化TSNE参数（关键改进）
#     tsne = TSNE(
#         n_components=2,  # 明确使用2维可视化
#         perplexity=30,  # 增大困惑度（原2太小，适合5-50之间）
#         learning_rate=200,  # 明确学习率
#         n_iter=1000,  # 增加迭代次数
#         init="pca",
#         random_state=42  # 添加随机种子保证可复现性
#     )
#     X_embedded = tsne.fit_transform(X)
#
#     # 可视化参数配置
#     num_classes = len(set(Y))
#     # 可视化参数配置
#     plt.style.use('seaborn-white')
#     colors = ['#FF7D40', '#00C957', '#1E90FF']
#
#     # 创建画布（增大尺寸和分辨率）
#     plt.figure(figsize=(10, 8), dpi=150)
#     ax = plt.gca()
#
#     # 绘制每个类别的散点
#     # 绘制每个类别的散点
#     legend_handles = []
#     for i in range(num_classes):
#         indices = np.where(Y == i)[0]
#         scatter = ax.scatter(
#             X_embedded[indices, 0],
#             X_embedded[indices, 1],
#             color=colors[i % len(colors)],
#             alpha=0.95,
#             edgecolor='g',
#             linewidth=0.3,
#             label=label_mapping[i]
#         )
#         legend_handles.append(scatter)
#
#     # 添加可视化元素
#     plt.title(f'Metaphor Type Visualization\nDataset: {args.dataset}', fontsize=14, pad=20)
#     plt.xlabel('Metaphor Type Representations', fontsize=12, fontstyle='italic')
#
#     # 优化坐标轴
#     plt.xticks([])  # 移除刻度（常见于t-SNE可视化）
#     plt.yticks([])
#     # 去除图形边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     # 添加图例
#     plt.legend(
#         handles=legend_handles,
#         loc='upper right',
#         frameon=False,
#         markerscale=2
#     )
#
#     # 优化布局并保存
#     plt.tight_layout()
#     filename = f'MetaphorType_TSNE_EPOCH={args.num_train_epochs}_DATASET={args.dataset}.png'
#     plt.savefig(filename, bbox_inches='tight', dpi=300)  # 提高保存质量
#     plt.close()  # 防止内存泄漏
#
#     logger.info(f"MetaphorType visualization for [{args.dataset}] saved as {filename}!")

def SaveMetaphorTypeFigure(args, tsne_dict):
    X = tsne_dict['metaphor_type_vecs'][-2000:]  # tsne_dict中的向量数组
    Y = tsne_dict['metaphor_type_labels'][-2000:]  # tsne_dict中的类别标签
    X = np.array(X)
    Y = np.array(Y)
    X_embedded = TSNE(n_components=3, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF7D40', '#00C957', '#1E90FF']
    figure = plt.figure(figsize=(5, 5), dpi=150)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('MetaphorTypeEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))

    logger.info(f"MetaphorType in [{args.dataset}] Figure has saved!")
    plt.show()

def SaveSentimentFigure(args, tsne_dict):
    X = tsne_dict['sentiment_vecs'][-2000:]  # tsne_dict中的向量数组
    Y = tsne_dict['sentiment_labels'][-2000:]  # tsne_dict中的类别标签
    X = np.array(X)
    Y = np.array(Y)
    X_embedded = TSNE(n_components=3, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF0000', '#EEB422', '#836FFF']
    figure = plt.figure(figsize=(5, 5), dpi=150)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('SentimentEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))

    logger.info(f"Sentiment in [{args.dataset}] Figure has saved!")

    plt.show()

# def SaveSentimentFigure(args, tsne_dict):
#     # 标签映射
#     label_mapping = {
#         0: "Positive",
#         1: "Negative",
#         2: "Neutral"
#     }
#
#     X = tsne_dict['sentiment_vecs'][-2000:]  # 保持原始数据截取方式
#     Y = tsne_dict['sentiment_labels'][-2000:]
#
#     # 数据预处理
#     X = np.array(X)
#     Y = np.array(Y)
#
#     # 优化TSNE参数（关键改进）
#     tsne = TSNE(
#         n_components=2,  # 明确使用2维可视化
#         perplexity=30,  # 增大困惑度（原2太小，适合5-50之间）
#         learning_rate=200,  # 明确学习率
#         n_iter=1000,  # 增加迭代次数
#         init="pca",
#         random_state=42  # 添加随机种子保证可复现性
#     )
#     X_embedded = tsne.fit_transform(X)
#
#     # 可视化参数配置
#     num_classes = len(set(Y))
#     # 可视化参数配置
#     plt.style.use('seaborn-white')
#     colors = ['#FF0000', '#EEB422', '#836FFF']
#
#     # 创建画布（增大尺寸和分辨率）
#     plt.figure(figsize=(10, 8), dpi=150)
#     ax = plt.gca()
#
#     # 绘制每个类别的散点
#     # 绘制每个类别的散点
#     legend_handles = []
#     for i in range(num_classes):
#         indices = np.where(Y == i)[0]
#         scatter = ax.scatter(
#             X_embedded[indices, 0],
#             X_embedded[indices, 1],
#             color=colors[i % len(colors)],
#             alpha=0.95,
#             edgecolor='g',
#             linewidth=0.3,
#             label=label_mapping[i]
#         )
#         legend_handles.append(scatter)
#
#     # 添加可视化元素
#     plt.title(f'Sentiment Visualization\nDataset: {args.dataset}', fontsize=14, pad=20)
#     plt.xlabel('Sentiment Representations', fontsize=12, fontstyle='italic')
#
#     # 优化坐标轴
#     plt.xticks([])  # 移除刻度（常见于t-SNE可视化）
#     plt.yticks([])
#     # 去除图形边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     # 添加图例
#     plt.legend(
#         handles=legend_handles,
#         loc='upper right',
#         frameon=False,
#         markerscale=2
#     )
#
#     # 优化布局并保存
#     plt.tight_layout()
#     filename = f'Sentiment_TSNE_EPOCH={args.num_train_epochs}_DATASET={args.dataset}.png'
#     plt.savefig(filename, bbox_inches='tight', dpi=300)  # 提高保存质量
#     plt.close()  # 防止内存泄漏
#
#     logger.info(f"Sentiment visualization for [{args.dataset}] saved as {filename}!")


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))
                    writer.write("{} = {}".format(key, str(metrics[key])))


# class ManualModelCheckpoint(ModelCheckpoint):
#     def on_epoch_end(self, trainer, pl_module):
#         # 每个 epoch 结束时手动保存模型
#         epoch = trainer.current_epoch
#         filepath = f"{pl_module.hparams.temp_dir}/epoch-{epoch:02d}"
#         pl_module.model.save_pretrained(filepath)
#         pl_module.tokenizer.save_pretrained(filepath)
#         print(f"Epoch {epoch} model have saved to -> {filepath}")