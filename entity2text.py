import json
import os
import urllib.parse, urllib.request
from collections import deque,defaultdict
import random

from tqdm import tqdm
import networkx as nx
import pandas as pd
from utils.data_loader import  train_user_set, test_user_set,n_items
from utils.parser import parse_args_kgsr as parse_args
import numpy as np
import pickle
def convert_id_format(s):
    dot_index = s.find('.')
    return '/' + s[:dot_index] + '/' + s[dot_index+1:]

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes,can_relation_range, inv_relation_range

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        print("bi edge")
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        can_relation_range=(min(can_triplets_np[:, 1]), max(can_triplets_np[:, 1]))#(last fm 1-9)
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1#(last fm 10-18)
        inv_relation_range = (min(inv_triplets_np[:, 1]), max(inv_triplets_np[:, 1]))
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        # print(can_relation_range, inv_relation_range)
    else:
        print("uni edge")
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets

# def construct_id2text(df,ranges):
#     # for entity in id_list:
#     #
#     #     row_index = df[df['id'] == entity].index
#     #     if not row_index.empty:
#     #         # 检查'text'列是否为NaN
#     #         if pd.isna(df.loc[row_index, 'text']).any():
#     #             # 根据org_id处理并更新'text'列
#     #             df.loc[row_index, 'text'] = df.loc[row_index, 'org_id'].apply(process_org_id)
#     # df=df.iloc[ranges[0]:ranges[1]]
#     # for row in df.iterrows():
#     #     if pd.isna(row["name"]):
#     #         id=convert__id_format(row["org_id"])
#     #         entity_name,description=get_text_by_id(id)
#     #         row["name"]=entity_name
#     #         row["description"]=description
#     def update_row(row):
#     # for row in df.iterrows():
#         if len(row["name"])==0:
#             id=convert__id_format(row["org_id"])
#             entity_name,description=get_text_by_id(id)
#             row["name"]=entity_name
#             row["description"]=description
#         return row
#     df = df.apply(update_row, axis=1)
#     return df
        # row_index = df[df['id'] == entity].index
class Entity2TextGraph:
    def __init__(self,args):
        self.multi_edge = False
        self.empty_id_text = []
        self.empty_mark = " "
        self.absence_mark="!"
        self.entity_idsdf = self.read_entity_ids(args)
        self.entity_idsdf.set_index('remap_id', inplace=True)
        self.relation_idsdf = self.read_realtion_ids(args)


        # self.graph = nx.Graph()
        # self.id2text = {}
        # self.text2id = {}
        # self.id2type = {}
        # self.type2id = {}
        # self.id2neighbors

    def build_graph(self,train_data, triplets,directed=False):
        if directed:
            ckg_graph= nx.MultiDiGraph()
        else:
            ckg_graph = nx.MultiGraph()
        rd = defaultdict(list)

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            rd[0].append([u_id, i_id])

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            ckg_graph.add_edge(h_id, t_id, key=r_id)

        return ckg_graph, rd
    def get_text_by_id(self,id):
        # id="/m/0v1_6dj"
        # api_key = open('AIzaSyARiLn1qCDEa8Hy6bL9sMuCs1u2-mIFTmI').read()
        api_key = 'AIzaSyARiLn1qCDEa8Hy6bL9sMuCs1u2-mIFTmI'
        query = 'Taylor Swift'
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
        params = {
            'ids':id,
            'limit': 10,
            'indent': True,
            'key': api_key,
        }
        url = service_url + '?' + urllib.parse.urlencode(params)
        response = json.loads(urllib.request.urlopen(url).read())
        if len(response['itemListElement'])==0:
            return "",""
        res=response['itemListElement'][0]['result']
        name=""
        description=""
        if "description" in res:
            description=res['description']
        if "name" in res:
            name=res['name']

        if len(response['itemListElement'])>1:
            print("more returned names",id)
            # maxScore=-np.inf
            # for i,element in enumerate(response['itemListElement']):
            #     #select element with max score
            #     if element['result']['resultScore']>=maxScore:
            #         maxScore=element['result']['resultScore']
            #         name=element['result']['name']
            #         description=element['result']['description']
                # name.append(element['result']['name'])
                # description.append(element['result']['description'])
        return name,description
          # print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')

    def read_entity_ids(self,args):
        directory = args.data_path + args.dataset + '/'
        import os
        if not os.path.exists(directory + 'entity_list.csv'):
            entity_ids = pd.read_csv(directory + 'entity_list.txt', sep='\s+', header=0)
            entity_ids['name'] = self.empty_mark
            entity_ids['description'] = self.empty_mark
            entity_ids.to_csv(directory + 'entity_list.csv', index=False)
        else:
            entity_ids = pd.read_csv(directory + 'entity_list.csv', header=0)

        if not os.path.exists(directory + '/user_id_kg/'):
            os.makedirs(directory + '/user_id_kg/')
        return entity_ids
    def read_realtion_ids(self,args):
        directory = args.data_path + args.dataset + '/'
        import os
        relation_ids = None
        if args.dataset == "last-fm":
            if not os.path.exists(directory + 'relation_list.csv'):
                relation_ids = pd.read_csv(directory + 'relation_list.txt', sep='\s+', header=0)
                relation_ids['org_id'] = relation_ids["org_id"].str.split(".").str[-1]
                df_inv = relation_ids.copy()
                max_relation = relation_ids['remap_id'].max()
                df_inv['remap_id'] = df_inv['remap_id'] + max_relation + 2
                relation_ids['remap_id'] += 1
                df_inv['org_id'] = 'inv.' + df_inv['org_id'].astype(str)

                relation_ids = pd.concat([relation_ids, df_inv]).reset_index(drop=True)
                relation_ids.rename(columns={'org_id': 'relation'}, inplace=True)
                relation_ids.to_csv(directory + 'relation_list.csv', index=False)
            else:
                relation_ids = pd.read_csv(directory + 'relation_list.csv', header=0)
            relation_ids.set_index('remap_id', inplace=True)
        return relation_ids
    def id2text(self,node_id):
        row=self.entity_idsdf.loc[node_id]
        if row['name'] != self.empty_mark:
            return row['name']
        entity_name=self.empty_mark
        if row['name'] == self.empty_mark and row['description'] != "no-name":#如果不在现存表里（description也为空）
            id = convert_id_format(row['org_id'])
            entity_name, description = self.get_text_by_id(id)
            if entity_name == "":
                entity_name = row['org_id']
                print("no name found for ", entity_name)
                entity_name=self.empty_mark
                description="no-name"
                self.empty_id_text.append({"map_id":node_id,"org_id":row['org_id']})
                # pd.DataFrame(self.empty_id_text).to_csv(args.data_path + args.dataset + '/empty_id_text.csv',index=False)
            self.entity_idsdf.loc[node_id]['name'] = entity_name
            self.entity_idsdf.loc[node_id]['description'] = description
        return entity_name

    def relation_id2text(self,relation_id):
        row = self.relation_idsdf.loc[relation_id]
        return row['relation']
    def plot_graph(self,G,savedPtah=None):
        import matplotlib.pyplot as plt
        G = nx.DiGraph(G)
        pos = nx.spring_layout(G)
        # nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, edge_color='k', linewidths=1,
                font_size=15, arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
        plt.savefig(savedPtah)


    def build_user_kg(self,ckg_graph,max_depth=4):
        sub_kg = nx.MultiGraph()
        #https://networkx.org/documentation/stable/reference/classes/multidigraph.html
        # sample a user from train_user_set
        id_nums=101
        u_id=0
        while id_nums>100:
            u_id =random.choice(list(train_user_set.keys()))#u_id+1#
            id_nums=len(train_user_set[u_id])
        degrees = ckg_graph.degree()
        max_item_ids = n_items-1
        high_degree_nodes = [node for node, degree in degrees if degree > 2000]
        for node_to_remove in high_degree_nodes:
            ckg_graph.remove_node(node_to_remove)

        train_i_ids= [node for node in  train_user_set[u_id] if node not in high_degree_nodes]
        test_i_ids = [node for node in test_user_set[u_id] if node not in high_degree_nodes]
        user_items={"train":train_i_ids,"test":test_i_ids}

        id_kgs=[]
        user_covered_items = []
        for data_type,i_ids in user_items.items():
            id_kg, item_in, visited=self.build_subgraph_bfs(i_ids, ckg_graph, max_item_ids, max_depth=max_depth,max_neighbors= 1500)
            id_kgs.append(id_kg)
            user_covered_items.append(item_in)
            print("len of items",len(i_ids))
            print("len of nodes",len(id_kg.nodes()))
            print("len of edges",len(id_kg.edges()))
            print("len of items touched",len(item_in))
        self.entity_idsdf.to_csv(args.data_path + args.dataset + '/entity_list.csv')
        pd.DataFrame(self.empty_id_text).to_csv(args.data_path + args.dataset + '/empty_id_text.csv',index=False)
        return u_id,id_kgs,user_covered_items
        # for i_id in i_ids:
    def build_subgraph_shortest_path(self,entity_id,ckg_graph,max_item_id,max_depth=1,max_neighbors=1000):
        subG = nx.MultiGraph()
        for i in range(len(entity_id)):
            for j in range(i + 1, len(entity_id)):
                source = entity_id[i]
                target = entity_id[j]
                # 对于每一对节点，找到所有最短路径
                for path in nx.all_shortest_paths(ckg_graph, source=source, target=target):
                    # 将路径上的边添加到子图中
                    nx.add_path(subG, path)

    def networkx_generate(self, entity_id, ckg_graph, max_item_id, max_depth=1, max_neighbors=1000):
        id_kg = ckg_graph.subgraph(entity_id)
        nodes = list(id_kg.nodes())
        id_kg, item_in, visited = self.build_subgraph_bfs(nodes, ckg_graph, max_item_id, max_depth=max_depth,
                                                          max_neighbors=max_neighbors)
        id_kg=self.attach_text2graph(id_kg, max_item_id)
        # for u, v, k,v in id_kg.edges(data=True, keys=True):
        #     del id_kg[u][v][k]
        #     id_kg[u][v]['relation'] =  self.relation_id2text(k)
        #     print(id_kg[u][v]['relation'])
        #     print(id_kg[u][v][k])
        return id_kg,item_in
    def attach_text2graph(self,id_kg,max_item_id):
        node_type = {}
        node_name = {}
        for node in id_kg.nodes():
            if node <= max_item_id:
                node_type[node] = 1
            else:
                node_type[node] = 0
            if node not in node_name:
                node_text = self.id2text(node)
                node_name[node] = node_text
        nx.set_node_attributes(id_kg, node_type, 'type')
        nx.set_node_attributes(id_kg, node_name, 'name')
        return id_kg
    def bfs_nodes(self,entity_id, ckg_graph,max_depth=4,max_neighbors=1000):
        visited = set()
        queue = deque([(node, 0) for node in entity_id])
        id_kg=nx.Graph()
        while queue:
            current_node, current_depth = queue.popleft()
            if current_node in visited or current_depth > max_depth:
                continue
            visited.add(current_node)
            neighbors = ckg_graph[current_node].items()
            if len(neighbors) > max_neighbors:
                print(current_node, "has too many neighbors:", len(neighbors))
                continue
            for neighbor, attrs in neighbors:
                if not id_kg.has_edge(current_node, neighbor):
                    id_kg.add_edge(current_node, neighbor)
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
        return id_kg

    def build_subgraph_bfs(self,entity_id, ckg_graph,max_item_id,max_depth=4,max_neighbors=1000):
        visited = set()
        item_in = {}
        id_kg = nx.MultiGraph()
        queue = deque([(node, 0) for node in entity_id])
        while queue:
            current_node, current_depth = queue.popleft()
            if current_node in visited or current_depth > max_depth:
                continue
            visited.add(current_node)
            neighbors = ckg_graph[current_node].items()
            if len(neighbors) > max_neighbors and current_node not in entity_id:
                print(current_node, "has too many neighbors:", len(neighbors))
                continue
            neighbors = random.sample(neighbors, int(len(neighbors)/2))
            # current_node_text= self.id2text(current_node)
            # if current_node <= max_item_id:
            #     node_type[current_node]=1
            #     node_type[current_node]=0
            # if current_node not in node_name:
            #     node_name[current_node]=current_node_text
            # if current_node_text==self.empty_mark:
                # continue
            for neighbor, attrs in neighbors:
                relations = [k for k, _ in attrs.items()]
                is_item = False
                if neighbor <= max_item_id:  # neighbor是物品,那么记录由哪个实体直接得到
                    is_item = True
                    # print("got items",neighbor, current_node, relations)
                    if neighbor not in item_in:
                        item_in[neighbor] = [(current_node, relations)]
                    else:
                        item_in[neighbor].append((current_node, relations))
                if self.multi_edge:
                    for r in relations:
                        id_kg.add_edge(current_node, neighbor, relation=self.relation_id2text(r))
                        # sub_kg.add_edge(current_node_text, neighbor_text, relation=self.relation_id2text(r))
                else:
                    # sub_kg.add_edge(current_node_text, neighbor_text, relation=self.relation_id2text(relations[0]))
                    id_kg.add_edge(current_node, neighbor, relation=self.relation_id2text(relations[0]))
                if is_item == False and neighbor not in visited :
                    # neighbor_text = self.id2text(neighbor)
                    # if neighbor_text==self.empty_mark:
                    #     current_node_text=current_node
                    queue.append((neighbor, current_depth + 1))
            # print(len(sub_kg.nodes()))
            # print(len(sub_kg.edges()))
            # print(item_in)
        return id_kg, item_in, visited
    def contruct_data(self,model_args, depth):
        graph=self.init_data(model_args)
        # items = pd.read_csv(directory + 'item_list.txt', sep='\s+', header=0)
        # item_ids = items['remap_id'].values.tolist()
        u_id, id_kgs,item_in =self.build_user_kg(graph, depth)
        return u_id, id_kgs,item_in
    def init_data(self,model_args):
        global args
        args = model_args
        directory = args.data_path + args.dataset + '/'
        print('reading train and test user-item set ...')
        train_cf = read_cf(directory + 'train.txt')
        test_cf = read_cf(directory + 'test.txt')
        print('interaction count: train %d, test %d' % (train_cf.shape[0], test_cf.shape[0]))
        remap_item(train_cf, test_cf)
        print('combinating train_cf and kg data ...')
        triplets = read_triplets(directory + 'kg_final.txt')
        print('building the graph ...')
        graph, relation_dict = self.build_graph(train_cf, triplets,False)
        self.directory = directory
        return graph
    def chekc_connectity(self,model_args):
        self.Di=False
        graph=self.init_data(model_args)
        connected_components = list(nx.connected_components(graph))
        print('number of connected components:', len(connected_components))
        for i, component in enumerate(connected_components):
            print('component %d: %d nodes' % (i, len(component)))
        # for u,train_items in train_user_set.items():
        #     items= train_items+test_user_set[u]
        #     u_graph= graph.subgraph(items)
        #     print(len(train_items),len(test_user_set[u]),len(u_graph.edges()))
    def generate_dataset(self,model_args,max_depth):
        graph= self.init_data(model_args)
        degrees = graph.degree()
        max_item_ids = n_items - 1
        high_degree_nodes = [node for node, degree in degrees if degree > 2000]
        for node_to_remove in high_degree_nodes:
            graph.remove_node(node_to_remove)
        small_users=[key for key, value in train_user_set.items() if len(value) < 100]
        n=100
        if len(small_users) < n:
            sampled_users = small_users
        else:
            sampled_users = random.sample(small_users, n)
        user_covered_items = {}
        for u_id in sampled_users:
            train_i_ids = [node for node in train_user_set[u_id] if node not in high_degree_nodes]
            test_i_ids = [node for node in test_user_set[u_id] if node not in high_degree_nodes]
            # user_items = {"train": train_i_ids, "test": test_i_ids}
            user_items = {"all": train_i_ids+test_i_ids}
            for data_type, i_ids in user_items.items():
                print("len of items", len(i_ids))
                # id_kg, item_in, visited = self.build_subgraph_bfs(i_ids, graph, max_item_ids,
                #                                                           max_depth=max_depth, max_neighbors=1000)
                id_kg,item_in=self.networkx_generate(i_ids, graph, max_item_ids,
                                                                          max_depth=max_depth, max_neighbors=100)
                user_covered_items[u_id]=item_in
                print("len of nodes", len(id_kg.nodes()))
                print("len of edges", len(id_kg.edges()))
                print("len of items touched", len(item_in))
                with open(args.data_path + args.dataset + '/user_id_kg/' + str(u_id) + '_Multi_' + data_type + '.pkl', 'wb') as f:
                    pickle.dump(id_kg, f)
            self.entity_idsdf.to_csv(args.data_path + args.dataset + '/entity_list.csv')
            pd.DataFrame(self.empty_id_text).to_csv(args.data_path + args.dataset + '/empty_id_text.csv', index=False)
        with open(args.data_path + args.dataset + '/user_covered_items.pkl', 'wb') as f:
                pickle.dump(user_covered_items, f)

    def find_connected_nodes(self,train_items, G, path_len_num=3):
        subGnodes = set(train_items)
        train_items = list(train_items)
        for i, source in enumerate(train_items):
            for j, target in enumerate(train_items[i + 1:]):
                if nx.has_path(G, source=source, target=target):
                    paths = list(nx.all_shortest_paths(G, source=source, target=target))
                    if len(paths) > path_len_num:
                        paths=paths[:path_len_num]
                    for path in paths:
                        for node in path:
                            subGnodes.add(node)
        return list(subGnodes)
    def regenerate_text_dataset(self,model_args,max_depth):
        ckg_graph = self.init_data(model_args)
        data_path = self.directory+'/user_id_kg/'
        model_data=[]
        for file in tqdm(os.listdir(data_path)):
            #如果文件名不包含Di，说明是原始数据
            if 'Di' not in file:
                user_data={}
                with open(data_path+file, "rb") as f:
                    G_loaded = pickle.load(f)
                user_data['user_id'] = int(file.split('_')[0])
                train_ids=train_user_set[user_data['user_id']]
                test_ids=test_user_set[user_data['user_id']]
                edges_text_list,train_list,test_list,neg_list= self.graph2DiWords(G_loaded,ckg_graph,train_ids,test_ids)
                user_data['graph']=edges_text_list
                user_data['train']=train_list
                user_data['test']=test_list
                user_data['neg']=neg_list
                model_data.append(user_data)
        with open(self.directory+'/users_kg_text.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    def get_sampled_users(self,adding=False):

        data_path = "/home/h666/ai/kgcl/data/last-fm" + '/user_id_kg/'
        chosen_users = []
        for file in os.listdir(data_path):
            if 'Di' not in file:
                chosen_users.append(int(file.split('_')[0]))
        if adding==False:
            return chosen_users
        sampled_users = random.sample(train_user_set.keys(), 1000)
        test_users=[]
        for u_id in sampled_users:
            if u_id not in chosen_users:
                test_users.append(u_id)
        return test_users
    def generate_id_datsets(self,model_args,max_depth):
        graph= self.init_data(model_args)
        degrees = graph.degree()
        max_item_ids = n_items - 1
        high_degree_nodes = [node for node, degree in degrees if degree > 2000]
        for node_to_remove in high_degree_nodes:
            graph.remove_node(node_to_remove)
        sampled_users = self.get_sampled_users(adding=True)
        data_type="pure_train"
        user_data={}
        for u_id in tqdm(sampled_users):
            train_i_ids = [node for node in train_user_set[u_id] if node not in high_degree_nodes]
            # train_i_ids=self.find_connected_nodes(train_i_ids,graph)
            id_kg=self.bfs_nodes(train_i_ids, graph, max_depth=max_depth, max_neighbors=200)
            user_data[u_id]=id_kg
            print("len of nodes", len(id_kg.nodes()),"len of edges", len(id_kg.edges()))
        directory = args.data_path + args.dataset + '/'
        train_cf = read_cf(directory + 'train.txt')
        triplets = read_triplets(directory + 'kg_final.txt')
        print('building the directed graph ...')
        biG, _ = self.build_graph(train_cf, triplets, True)
        for u_id, id_kg in user_data.items():
            id_kg=self.undirected2directed(biG, id_kg)
            with open(args.data_path + args.dataset + '/user_id_kg/' + str(u_id) + '_Multi_' + data_type + '.pkl', 'wb') as f:
                    pickle.dump(id_kg, f)
    def undirected2directed(self,biG, subG):
        id_kg=nx.MultiGraph()
        for u, v in subG.edges():
            if biG.has_edge(u, v):
                edge = biG.get_edge_data(u, v)
                for k, rel in edge.items():
                    id_kg.add_edge(u, v, self.relation_id2text(k))
            elif biG.has_edge(v, u):
                edge = biG.get_edge_data(v, u)
                for k, rel in edge.items():
                    id_kg.add_edge(v, u, self.relation_id2text(k))
        return id_kg
    def graph2DiWords(self,graph,ckg_graph,train_ids,test_ids):
            edges_text_list=[]
            max_item_ids = n_items - 1
            train_list=set()
            test_list=set()
            neg_list=set()
            def node2name(node):
                name=graph.nodes[node]['name']
                if name==self.empty_mark:
                    return str(node)
                return name
            for node in graph.nodes():
                if node<=max_item_ids:
                    if node in train_ids:
                        train_list.add(node2name(node))
                    elif node in test_ids:
                        test_list.add(node2name(node))
                    else:
                        neg_list.add(node2name(node))
            for u, v in graph.edges():
                bi=0
                if ckg_graph.has_edge(u, v):
                    edge = ckg_graph.get_edge_data(u, v)
                    u_name = node2name(u)
                    v_name = node2name(v)
                    for k,_ in edge.items():
                        relation = self.relation_id2text(k)
                        edges_text_list.append([u_name, relation, v_name])
                elif ckg_graph.has_edge(v, u):  # 如果有向图中的边方向是反的
                    edge = ckg_graph.get_edge_data(v, u)
                    u_name = node2name(v)
                    v_name = node2name(u)
                    for k, _ in edge.items():
                        relation = self.relation_id2text(k)
                        edges_text_list.append([u_name, relation, v_name])

            print(edges_text_list)
            return edges_text_list,train_list,test_list,neg_list


if __name__ == '__main__':
    """read args"""
    global args, device
    args = parse_args()
    args.inverse_r=False
    """build dataset"""
    ukg_built=Entity2TextGraph(args)
    ukg_built.multi_edge=True

    # print(len(ukg_built.entity_idsdf.loc[1]["name"]))
    depth = 2
    ukg_built.generate_id_datsets(args,depth)
    # ukg_built.regenerate_text_dataset(args,depth)
    # ukg_built.chekc_connectity(args)
    # ukg_built.generate_dataset(args,depth)
    # ukg_built.get_text_by_id(convert_id_format("m.0dm9rk5"))
    # ukg_built.get_text_by_id("m.03_t3bb")
    # u_id,id_kgs, item_in =ukg_built.contruct_data(args,depth)
    # ukg_built.plot_graph(sub_kg[0],args.data_path + args.dataset + '/subgraph.png')
    # with open(str(u_id)+"_graph.pkl", "wb") as f:
    #     pickle.dump(sub_kg[0], f)
    # 从Pickle文件读取图
    # with open(str(u_id)+"_graph.pkl", "rb") as f:
    #     G_loaded = pickle.load(f)
    #
    # for node,att in G_loaded.nodes(data=True):
    #     print(node,att)

    # G = nx.MultiDiGraph()
    # print(sub_kg[0].edges(data=True)[0])
    # for node_id, neighbor_id, relation_ids in sub_kg[0].edges(data=True):
    #
    #     current_node_text = id2text(node_id,entity_idsdf)#node_id_to_text[current_node_id]
    #     neighbor_text = id2text(neighbor_id,entity_idsdf)# node_id_to_text[neighbor_id]
    #     for relation_id in relation_ids:
    #         relation_text = relation_id_to_text(relation_id,relation_idsdf)
    #         G.add_edge(current_node_text, neighbor_text, relation=relation_text)
    # df=construct_id2text(entity_idsdf,(0,10000))
    # df.to_csv(args.data_path + args.dataset + '/entity_list.csv', index=False)
    # get_text_by_id("/m/0v1_6dj")