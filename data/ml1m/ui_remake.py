import pandas as pd

filemap={"newtrain.csv":"train","newtest.csv":"test","newvalid.csv":"valid"}
def get_relation_by_id(file):
    df = pd.read_csv(file)
    grouped = df.groupby('user')['ordered_item'].apply(list).reset_index()
    def format_row(row):
        return f"{row['user']} " + ' '.join(map(str, row['ordered_item']))

    formatted_data = grouped.apply(format_row, axis=1)

    with open(f'mapped_data/{filemap[file]}.txt', 'w') as file:
        for line in formatted_data:
            file.write(line + '\n')

# files=["newtrain.csv","newtest.csv","newvalid.csv"]
#     for file in files:
#         get_relation_by_id(file)
def read_and_parse_file(file_path):
    """
    读取并解析文件，返回一个字典，键是用户ID，值是物品ID的列表。
    """
    user_items = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            user_id = parts[0]
            items = parts[1:]
            if user_id in user_items:
                # 确保物品ID唯一
                user_items[user_id].update(items)
            else:
                user_items[user_id] = set(items)
    return user_items

def merge_user_items(user_items1, user_items2):
    """
    合并两个用户物品字典。
    """
    # 合并第二个字典到第一个
    for user_id, items in user_items2.items():
        if user_id in user_items1:
            user_items1[user_id].update(items)
        else:
            user_items1[user_id] = items
    return user_items1

def save_merged_data(merged_data, output_file):
    """
    将合并后的数据保存到文件。
    """
    with open(output_file, 'w') as file:
        for user_id, items in sorted(merged_data.items(), key=lambda x: int(x[0])):
            line = f"{user_id} " + ' '.join(sorted(items, key=int))
            file.write(line + '\n')

# 读取并解析两个文件
user_items1 = read_and_parse_file(f'/home/h666/ai/kgcl/data/movie/test.txt')
user_items2 = read_and_parse_file(f'/home/h666/ai/kgcl/data/movie/valid.txt')

# 合并数据
merged_data = merge_user_items(user_items1, user_items2)

# 保存合并后的数据到新文件
save_merged_data(merged_data, 'mapped_data/merged_output.txt')

print("Files have been merged and saved as 'merged_output.txt'.")
