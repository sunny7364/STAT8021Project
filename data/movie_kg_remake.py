import pandas as pd


import os
# print(os.getcwd())
# 步骤1: 读取i2kg_map.tsv和e_map.csv文件

i2kg_map = pd.read_csv('./i2kg_map.tsv', sep='\t', header=None, names=['origin_id',"title", 'url'])
e_map = pd.read_csv('kg/e_map.txt', sep='\t',header=None, names=['entity_id', 'url'])
i_map = pd.read_csv('i_map.txt', sep='\t',header=None, names=['item_id', 'origin_id'])
origin_id_to_item_id = {row['origin_id']: row['item_id'] for idx, row in i_map.iterrows()}
# 步骤2: 建立URL到新ID的映射
# 首先，给所有item分配新的entity ID
# item_url_to_new_id = {row['url']: idx for idx, row in i2kg_map.iterrows()}
# item_id_to_new_id = {row['origin_id']: idx for idx, row in i2kg_map.iterrows()}
item_url_to_new_id={}
item_id_to_new_id={}# mapped item id to ordered id
item_over=0
for idx, row in i2kg_map.iterrows():
    if row['origin_id'] in origin_id_to_item_id: # item kg contained in item map
        item_url_to_new_id[row['url']]=int(item_over)
        item_id_to_new_id[origin_id_to_item_id[row['origin_id']]]=int(item_over)
        item_over+=1

# item_id_to_new_id={origin_id_to_item_id[row['item_id']]: idx for idx, row in i_map.iterrows()}
# 然后，给剩余的entity分配新的ID，跳过已经分配给item的ID数量
non_item_start_id = item_over
non_item_to_new_id={}
entity_over=0
for idx, row in e_map.iterrows():
    if row["url"] not in item_url_to_new_id:
        non_item_to_new_id[row["url"]]=int(entity_over+non_item_start_id)
        entity_over+=1
# non_item_to_new_id = {row['url']: idx + non_item_start_id for idx, row in e_map.iterrows() if row["url"] not in item_to_new_id}

url_to_new_id = {**item_url_to_new_id, **non_item_to_new_id}
i2kg_map["ordered_id"] = i2kg_map["url"].map(item_url_to_new_id)
e_map["ordered_id"] = e_map["url"].map(url_to_new_id)

# 假设知识图谱的三元组存储在kg_triplets.csv文件中
df_train=pd.read_csv('kg/train.csv', sep='\t',header=None, names=['head', 'tail', 'relation'])
df_test=pd.read_csv('kg/test.csv', sep='\t',header=None, names=['head', 'tail', 'relation'])
df_val=pd.read_csv('kg/valid.csv', sep='\t',header=None, names=['head', 'tail', 'relation'])
kg_triplets = pd.concat([df_train, df_test, df_val]).drop_duplicates(subset=['head', 'tail', 'relation'])


def is_continuous_and_unique(arr):
    # 转换为集合，以移除可能的重复元素
    arr=list(arr)
    arr_set = set(arr)
    # arr.sort()
    for i in range(len(arr) - 1):
        # 如果相邻两个元素的差值大于1，则记录这对元素
        if arr[i + 1] - arr[i] != 1 :
            print(arr[i], arr[i + 1])
    # 检查无重复
    if len(arr) != len(arr_set):
        return False

    # 检查连续性
    if max(arr) - min(arr) + 1 == len(arr):
        return True
    else:
        return False

# kg_triplets = pd.read_csv('kg_triplets.csv', header=None, names=['head', 'tail', 'relation'])
relations_to_delete = [5,16,13,12]
#各种场景中指向更多信息，但它并不强制性地定义了两个资源之间的具体关系类型，只是一种通用的“更多信息”链接 5
# type 声明一个资源（实体）属于一个特定的类（class） 引入丰富的语义信息，帮助理解资源之间的关系，以及资源本身的性质和类别 8
#subject 资源的主题内容，这可以是关键词、分类代码或者描述性短语
print("total kg",len(kg_triplets))
kg_triplets = kg_triplets[~kg_triplets['relation'].isin(relations_to_delete)]
print("after delete",len(kg_triplets))
# kg_triplets.to_csv('combined_filtered_kg_triplets.csv', index=False)
kg_triplets['ordered_head'] = kg_triplets['head'].map(e_map.set_index('entity_id')['url']).map(url_to_new_id)
kg_triplets['ordered_tail'] = kg_triplets['tail'].map(e_map.set_index('entity_id')['url']).map(url_to_new_id)
# kg_triplets = kg_triplets.dropna()
# kg_triplets['ordered_head'] = kg_triplets['ordered_head'].astype(int)
# kg_triplets['ordered_tail'] = kg_triplets['ordered_tail'].astype(int)
i2kg_map = i2kg_map.dropna()
i2kg_map["ordered_id"] = i2kg_map["ordered_id"].astype(int)
e_map = e_map.dropna()
e_map["ordered_id"] = e_map["ordered_id"].astype(int)
i2kg_map.to_csv('new_i2kg_map.tsv', sep='\t', index=False)
e_map.to_csv('new_e_map.csv', sep='\t', index=False)
# kg_triplets.to_csv('updated_kg_triplets.csv', index=False)

def map_ui_id(file_path,item_id_to_new_id):
    data=pd.read_csv(file_path,sep='\t',header=None, names=['user', 'item', 'rating'])
    data['ordered_item']=data['item'].map(item_id_to_new_id)
    data = data.dropna()
    data['ordered_item'] = data['ordered_item'].astype(int)
    data.to_csv("new_"+file_path,index=False)
map_ui_id('train.csv',item_id_to_new_id)
map_ui_id('test.csv',item_id_to_new_id)
map_ui_id('valid.csv',item_id_to_new_id)

def new_relation_map():
    r_map = pd.read_csv('kg/r_map.txt', sep='\t', header=None, names=['relation_id', 'url'])
    new_relations = [row["url"] for i, row in r_map.iterrows() if i not in relations_to_delete]
    url2id_r = {rel: i for i, rel in enumerate(new_relations)}
    old2new_r = {i: url2id_r[row["url"]] for i, row in r_map.iterrows() if row["url"] in url2id_r}
    r_map['ordered_r'] = r_map['url'].map(url2id_r)

    r_map=r_map.dropna()
    r_map.to_csv('new_r_map.txt', sep='\t', index=False)
    with open('mapped_data/relation_list.txt', 'w') as f:
        f.write(f"org_id remap_id\n")
        for new_id, relation in enumerate(new_relations):
            f.write(f"{relation} {new_id}\n")

    kg_triplets['relation'] = kg_triplets['relation'].map(old2new_r)
    kg_triplets.to_csv('mapped_data/kg_final.txt', header=False, index=False, sep=' ', columns=['ordered_head', 'relation', 'ordered_tail'])
new_relation_map()