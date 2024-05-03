def log_train():
    log_file_path = 'output4.29.log'
    # 输出的CSV文件路径
    output_csv_path = 'ml_metrics_f.csv'
    # {'precision': array([0.23296358]), 'recall': array([0.23359739]), 'ndcg': array([0.30861389]), 'hit_ratio': array([0.93791391]), 'auc': 0.0}
    # @10 {'precision': array([0.27135762]), 'recall': array([0.14476232]), 'ndcg': array([0.30601046]), 'hit_ratio': array([0.8705298]), 'auc': 0.0}
    # 用于匹配日志行的正则表达式
    pattern = re.compile(r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\[([\d., ]+)\]\s*\|\s*\[([\d.]+)\]\s*\|\s*\[([\d.]+)\]\s*\|\s*\[([\d.]+)\]\s*\|\s*\[([\d.]+)\]\s*\|')

    # 准备写入CSV的数据
    csv_data = [['Epoch', 'Training Time', 'Testing Time', "rec loss","mae loss","cl loss", 'Recall', 'NDCG', 'Precision', 'Hit Ratio']]

    # 读取日志文件
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch_data = match.groups()
                epoch = epoch_data[0]
                training_time = epoch_data[1]
                testing_time = epoch_data[2]
                loss_values = epoch_data[3].split(', ')
                recall = epoch_data[4]
                ndcg = epoch_data[5]
                precision = epoch_data[6]
                hit_ratio = epoch_data[7]
                # 将损失值分开以适应列格式
                while len(loss_values) < 3:  # 确保有足够的损失值填充
                    loss_values.append('')
                csv_data.append([epoch, training_time, testing_time] + loss_values + [recall, ndcg, precision, hit_ratio])

    # 写入CSV文件
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f'Data extracted and saved to {output_csv_path}')
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    # 读取CSV文件
    df = pd.read_csv(output_csv_path)

    # 选择要规范化的列
    columns_to_scale = ["rec loss","mae loss","cl loss", 'Recall', 'NDCG', 'Precision', 'Hit Ratio']

    # 使用MinMaxScaler进行规范化
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # 绘制损失随epoch的变化图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1行2列，当前是第1个图
    for loss_col in ["rec loss","mae loss","cl loss"]:
        plt.plot(df_scaled['Epoch'], df_scaled[loss_col], label=loss_col)
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.legend()

    # 绘制度量随epoch的变化图
    plt.subplot(1, 2, 2)  # 1行2列，当前是第2个图
    for metric_col in ['Recall', 'NDCG', 'Precision', 'Hit Ratio']:
        plt.plot(df_scaled['Epoch'], df_scaled[metric_col], label=metric_col)
    plt.title('Metric vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Metric')
    plt.legend()

    plt.tight_layout()
    plt.show()

import csv
import re
#
# # # 日志文件路径

# import pandas as pd
#
# # 假设csv_path是你的CSV文件路径
# csv_path = 'bfs_rating_respure_train.csv'
#
# # 读取CSV文件
# df = pd.read_csv(csv_path)
#
# # 排除含有'var'的列
# columns_to_exclude = [col for col in df.columns if 'var' in col]
# df_filtered = df.drop(columns=columns_to_exclude)
#
# # 计算每列的平均值，自动忽略缺失值
# column_averages = df_filtered.mean()
#
# print("平均值：")
# print(column_averages)
def draw_metrics():
    import numpy as np
    import matplotlib.pyplot as plt

    # 数据
    metrics = ["NDCG", "hit_ratio"]
    Lables = ["BFS", "ERROR","ALL"]
    BFS_values = [0.41, 0.97]
    ERROR_values = [0.30, 0.93]
    ALL_values = [0.43, 0.98]

    x = np.arange(len(metrics))
    width = 0.2  # 条形图的宽度

    # 绘图
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, BFS_values, width, label='BFS')
    rects2 = ax.bar(x, ERROR_values, width, label='ERROR')
    rects3 = ax.bar(x + width, ALL_values, width, label='ALL')

    # 添加文本，标题，标签
    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric and group')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # 在条形图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    # 显示图表
    plt.show()
draw_metrics()