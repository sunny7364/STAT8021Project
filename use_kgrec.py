import json
import pickle

import pandas as pd
import setproctitle
import random
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time

from prettytable import PrettyTable
import datetime

from utils.evaluate import get_performance
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data,load_ui_data,load_kg_map
from modules.KGRec import KGRec
from utils.evaluate_kgsr import test, get_user_emb_item, save_embed, select_success_user, \
    ranklist_by_sorted, ranklist_by_heapq
from utils.wikiAPI import get_relation_by_id
from utils.helper import early_stopping, init_logger
from logging import getLogger
from utils.sampler import UniformSampler
import heapq
from collections import defaultdict

seed = 2020
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "utils/ext/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(seed)
except:
    sampling = UniformSampler(seed)

setproctitle.setproctitle('EXP@KGRec')

def neg_sampling_cpp(train_cf_pairs, train_user_dict):
    time1 = time()
    train_cf_negs = sampling.sample_negative(train_cf_pairs[:, 0], n_items, train_user_dict, 1)
    train_cf_negs = np.asarray(train_cf_negs)
    train_cf_triples = np.concatenate([train_cf_pairs, train_cf_negs], axis=1)
    time2 = time()
    logger.info('neg_sampling_cpp time: %.2fs', time2 - time1)
    logger.info('train_cf_triples shape: {}'.format(train_cf_triples.shape))
    return train_cf_triples

def get_feed_dict(train_cf_with_neg, start, end):
    feed_dict = {}
    entity_pairs = torch.from_numpy(train_cf_with_neg[start:end]).to(device).long()
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = entity_pairs[:, 2]
    feed_dict['batch_start'] = start
    return feed_dict
def generate_jiayangwant():
    select_success_user()
def load_similar_user():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device

    args = parse_args_kgsr()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    logger = getLogger()

    logger.info('PID: %d', os.getpid())
    logger.info(f"DESC: {args.desc}\n")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    save_path = args.out_dir + args.log_fn + '.ckpt'
    kgrec_model = "/home/h666/ai/kgcl/weights/test_kgrec.ckpt"
    args.data_type = "pure_train"
    model_dict = {
        'KGSR': KGRec,
    }

    # select_success_user(user_dict,n_params)
    model = model_dict[args.model]
    model = model(n_params, args, graph, mean_mat_list[0]).to(device)
    model.print_shapes()
    model.load_state_dict(torch.load(kgrec_model))
    model.eval()


def item_count():
    directory = args.data_path + args.dataset + '/'
    file_name=directory + 'train.txt'
    lines = open(file_name, "r").readlines()
    item_co = {}
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            if i_id in item_co:
                item_co[i_id] += 1
            else:
                item_co[i_id] = 1
    return item_co
def similar_users(u_id,user_gcn_emb,top_k):
    from sklearn.metrics.pairwise import cosine_similarity
    target_user_vector = user_gcn_emb[u_id].reshape(1, -1)
    similarities = cosine_similarity(target_user_vector, user_gcn_emb)[0]
    top_k_indices = np.argsort(similarities)[-top_k - 1:-1][::-1]  # 排除自己，所以是-top_k-1
    return top_k_indices.tolist()
def extract_similar_users(u_num=10):
    train_user_set = user_dict['train_user_set']

    user_gcn_embs=user_gcn_emb.cpu().numpy()
    user_simi_user = []
    i_c = item_count()
    # randomly sample 1000 users
    users_id=random.sample(range(n_users),u_num)
    for u in tqdm(users_id):
        u_data = {"uid": u}
        u_data["history"] = [(i, i_c[i]) for i in train_user_set[u]]
        u_data["history"]=  sorted(u_data["history"], key=lambda x: x[1], reverse=True)
        u_data["similar_users"] = []
        a_user_gcn_emb = user_gcn_emb[u]
        rec_items, rec_scores = get_user_recommendation(u, a_user_gcn_emb)
        u_data["rec_items"]=[ [item_id,item_score] for item_id,item_score in zip(rec_items,rec_scores) ]
        similarUsers = similar_users(u, user_gcn_embs, 3)
        for s_u in similarUsers:
            s_u_item_fre = [(i, i_c[i]) for i in train_user_set[s_u]]
            s_u_item_fre = sorted(s_u_item_fre, key=lambda x: x[1], reverse=True)
            u_data["similar_users"].append({"uid": s_u, "history": s_u_item_fre})
        user_simi_user.append(u_data)
    with open(str(u_num)+args.dataset+"_user_simi_user.json", "w") as f:
        for line in user_simi_user:
            f.write(json.dumps(line)+"\n")
def get_user_recommendation(u_id,u_g_embedding):
    item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
    i_g_embddings = entity_gcn_emb[item_batch].to(device)
    rate_batch = torch.matmul( torch.squeeze(u_g_embedding.to(device)),torch.squeeze(i_g_embddings, dim=1).t()) #model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
    training_items = train_user_set[u_id]
    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))
    item_score = {}
    rate_batch=torch.squeeze(rate_batch).cpu()
    for i in test_items:
        item_score[i] = rate_batch[i]
    K_max = int(max(Ks)/2)
    rating_score = list(item_score.items())
    rating_score.sort(key=lambda x: x[1], reverse=True)
    rec_items,rec_scores = [],[]
    for key, value in rating_score[:K_max]:
        if key  in missing_entity:
            print("missing entity",key)
            continue
        rec_items.append(key)
        rec_scores.append(round(value.item(),2))
    return rec_items,rec_scores
def read_prompt_data(file_name):
    train_js=[]
    with open(data_path + file_name,encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            train_js.append(json_obj)
    return train_js
def movie_prompt():
    train_js=read_prompt_data(train_pro_f_name)
    rec_js=read_prompt_data(rec_pro_f_name)

    question="You are a film recommendation expert. Please first understand the knowledge triples as natural language and then recommend top 3 films to the user."
    prompt_q = "\n\nHere is extracted film knowledge based on the user's film watching history:"
    system_prompt = "\nHere are the candidate films and their knowledge that this user might be interested in:" \

    fianl_prompt="\n\nPlease recommend top 3 films to the user with reasons."
    prompts=[]
    for i in range(len(train_js)):
        prompt = {}
        train_js[i]["prompt"]=gene_train_prompt(train_js[i])
        rec_js[i]["prompt"]=gene_rec_prompt(rec_js[i])
        prompt["uid"]=train_js[i]["uid"]
        prompt["question"]=question+prompt_q+train_js[i]["prompt"]
        prompt["system_prompt"]=system_prompt+rec_js[i]["prompt"]+fianl_prompt
        prompts.append(prompt)
        # p=question+prompt_q+train_js[i]["prompt"]
        # p+=system_prompt+rec_js[i]["prompt"]+fianl_prompt
        # prompt[train_js[i]["uid"]]=p
    with open(data_path + "ml_prompt_d4.json", "w", encoding="utf-8") as f:
        for js in prompts:
            f.write(json.dumps(js, ensure_ascii=False) + "\n")
    # with open(data_path + "ml_prompt2.json", "w", encoding="utf-8") as f:
    #     for uid, js in prompt.items():
    #         f.write(json.dumps(prompt[uid], ensure_ascii=False) + "\n")

    return prompt
def gene_train_prompt(js):
    prompt=""
    prompt+="\nSome relations with importance score: "
    relations=[]
    for i in range(len(js["relations"])):
        rel,sc=js["relations"][i][0],js["relations"][i][1]
        relations.append(rel+"("+str(sc)+")")
    prompt+=",".join(relations)
    prompt+="\nKnowledge triples(subject, relation, object) with rationals:\n"
    triples=[]
    for i in range(len(js["triples"])):
        triple=js["triples"][i]
        sc=js['triples_score'][i]
        if triple[1]=="subject":
            triple = "(" + triple[0] + ", " + triple[2] + ")"
        else:
            triple="("+triple[0]+", "+triple[1]+", "+triple[2]+")"
        triples.append(triple+" with rational "+str(sc))
    prompt+=";;".join(triples)
    return prompt

def gene_rec_prompt(js):
    # prompt="some candidate films with knowledge: "
    relations=[]
    item_p=""
    rec_num=7
    for i in range(len(js["rec_items"])):
        if i>=rec_num:
            break
        id,sc,click=js["rec_items"][i][0],js["rec_items"][i][1],js["rec_items"][i][2]
        knowledge=js["items_kg"][i]
        name=knowledge["item"]
        item_p+="\nThe film {}, id {} with recommendation score {} and rating count {}, ".format(name,id,sc,click)
        item_p+="has important triples: "
        triples=[]
        for triple in knowledge["triples"]:
            if triple[1] == "subject":
                triples.append("(" + str(triple[0]) + ", " + triple[2] + ")")
            else:
                triples.append("(" + str(triple[0]) + ", " + triple[1] + ", " + str(triple[2]) + ")")
        item_p+=";;".join(triples)
    return item_p
def user_kg_profile(uid,user_gcn_emb,model):

    # uid_pt = torch.LongTensor(np.array([uid])).to(device)
    a_user_gcn_emb = user_gcn_emb[uid]

    rec_items, rec_scores = get_user_recommendation(uid, a_user_gcn_emb)
    edge_attn_score, item_attn_mean = model.pre_item_scores(entity_gcn_emb, a_user_gcn_emb)

    edge_index = model.edge_index.to("cpu")
    edge_type = model.edge_type.to("cpu")
    edge_attn_score = edge_attn_score.to("cpu")
    item_attn_mean = item_attn_mean.to("cpu")
    items = train_user_set[uid]
    items = [i for i in items if i not in missing_entity]
    train_pro = {"uid": uid}
    rec_pro = {"uid": uid}

    i_c = item_count()
    rec_pro["rec_items"] = [[item_id, item_score, i_c[item_id]] for item_id, item_score in zip(rec_items, rec_scores) ]
    rec_pro["test_items"] = test_user_set[uid]
    recs_kg_profile(edge_index, edge_type, edge_attn_score, rec_items, rec_pro, model)
    items_kg_profile(edge_index,edge_type,edge_attn_score, items, train_pro, model)
    return train_pro, rec_pro
def recs_kg_profile(edge_index,edge_type,edge_attn_score, items, user_data, model):
    item_good_edges = model.extract_item_edges_scores(edge_index, edge_type,
                                    edge_attn_score, items, 8)
    kg_data_items = []
    for i, (i_edges, i_ets, i_scores) in item_good_edges.items():
        text_triples = []
        if i in missing_entity:
            continue
        id_triples = []
        item_as_subject={}
        "for triples with item as subject"
        item_as_object={}
        for j, (e, et, s) in enumerate(zip(i_edges.t(), i_ets, i_scores)):
            edge = e.tolist()
            if edge[0] in missing_entity or edge[1] in missing_entity:
                continue
            head=id_dict[edge[0]]["name"]
            tail=id_dict[edge[1]]["name"]
            relation = relation_dict[et.item() - 1]["name"]
            score=round(s.item(), 2)
            if edge[0] == i:
                head=i
            #     if relation not in item_as_object:
            #         item_as_object[relation] = [tail, score]
            #     else:
            #         item_as_object[relation].append([tail, score])
            #     #item_as_object.append([relation_dict[relation]["name"], tail,score])
            elif edge[1] == i:
                tail=i
            #     if relation not in item_as_subject:
            #         item_as_subject[relation] = [head, score]
            #     else:
            #         item_as_subject[relation].append([head, score])
            #     #item_as_subject.append([head, relation_dict[edge_type - 1]["name"],score])
            # else:
            text_triples.append([head, relation, tail,score])
            # item_kg_data= {"id":i,"item": id_dict[i]["name"], "as_subject": item_as_subject, "as_object": item_as_object, "triples": text_triples}
        item_kg_data = {"id": i, "item": id_dict[i]["name"],  "triples": text_triples}
        kg_data_items.append(item_kg_data)
    user_data["items_kg"]= kg_data_items

def items_kg_profile(edge_index,edge_type,edge_attn_score, items, user_data, model):
    top_edges, top_edge_types, top_edge_attn_scores, top_relations, top_r_values = model.extract_items_edge_scores(edge_index,edge_type,
        edge_attn_score, items, 2*len(items)    )
    text_triples = []
    id_triples = []
    top_relations=top_relations.tolist()
    top_r_values=top_r_values.tolist()
    relations = [(relation_dict[r-1]["name"], round(v, 2)) for (r, v) in
                 zip(top_relations,top_r_values)]
    user_data["relations"] = relations
    scores=[]
    for i, (e, et, s) in enumerate(zip(top_edges.t(), top_edge_types, top_edge_attn_scores)):
        edge = e.tolist()
        edge_type = et.item()
        scores.append(round( s.item(), 2))
        text_triple = (id_dict[edge[0]]["name"], relation_dict[edge_type-1]["name"], id_dict[edge[1]]["name"])
        id_triple = (edge[0], edge_type-1, edge[1])
        text_triples.append(text_triple)
        id_triples.append(id_triple)
    user_data["triples"] = text_triples
    user_data["triples_id"] = id_triples
    user_data["triples_score"] = scores

def test_given_negatives(x, data_path):
        rating = x[0]
        u = x[1]
        try:
            training_items = train_user_set[u]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = test_user_set[u]
        all_items = set(range(0, n_items))

        test_items = list(all_items - set(training_items))
        test_type = "pure_train"
        bfs_pos_test = []
        neg_test = []
        user_items = list(set(training_items + user_pos_test))
        file = data_path + str(u) + '_Multi_' + test_type + '.pkl'
        if 'Di' not in file:
            with open(file, "rb") as f:
                graph = pickle.load(f)
            for node in graph.nodes():
                if node < n_items - 1:
                    if node in user_pos_test:
                        bfs_pos_test.append(node)
                    if node not in user_items:
                        neg_test.append(node)
        wide_neg_test = list(all_items - set(neg_test) - set(user_items))
        non_bfs_pos_test = list(set(user_pos_test) - set(bfs_pos_test))
        utest_res = {'uid': u, "test_num": len(user_pos_test)}
        save_test_result(utest_res, "wide_pos", non_bfs_pos_test, rating)
        save_test_result(utest_res, "bfs_pos", bfs_pos_test, rating)
        save_test_result(utest_res, "neg", neg_test, rating)
        save_test_result(utest_res, "wide_neg", wide_neg_test, rating)
        user_pos_test = bfs_pos_test
        test_items = list(set(neg_test + bfs_pos_test))
        if args.test_flag == 'part':
            r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        else:
            r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
        return get_performance(user_pos_test, r, auc, Ks), utest_res

def save_test_result(utest_res, name, test_ids, ratings):
    utest_res[name + '_num'] = len(test_ids)
    ratings = [ratings[id] for id in test_ids]
    scores = ratings
    mean_score = np.mean(np.array(scores))
    var_score = np.var(np.array(scores))
    utest_res[name + '_ratings_mean'] = mean_score
    utest_res[name + '_ratings_var'] = var_score
def test_bfs_items():
        result = {'precision': np.zeros(len(Ks)),
                  'recall': np.zeros(len(Ks)),
                  'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)),
                  'auc': 0.}
        source_path = args.data_path + args.dataset
        data_path = source_path+  '/user_id_kg/'
        # ndcg [0.02704716]), 'hit_ratio': array([0.142]
        test_users = []
        for file in os.listdir(data_path):
            if 'Di' not in file:
                test_users.append(int(file.split('_')[0]))
        u_batch_size = 32
        test_users = list(set(test_users))
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0
        utest_ress = []
        test_type = "pure_train"
        entity_gcn_emb, user_gcn_emb = model.generate()
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.squeeze(i_g_embddings, dim=1).t()).detach().cpu()
            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = []

            for ur in user_batch_rating_uid:
                res1, res2 = test_given_negatives(ur,data_path)
                batch_result.append(res1)
                utest_ress.append(res2)
            # batch_result,utest_ress = pool.map(test_given_negatives, user_batch_rating_uid)
            count += len(batch_result)
            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users
        assert count == n_test_users
        pd.DataFrame(utest_ress).set_index('uid').to_csv(source_path + '/bfs_rating_res' + test_type + '.csv')
        return result
def gene_kg_data4prompt(u_num=1000):

    train_pros, rec_pros=[],[]
    entity_gcn_emb, user_gcn_emb = model.generate()
    model.item_edge_count=0
    model.item_over = 0
    users_id = random.sample(range(n_users), u_num)
    for uid in tqdm(users_id):
        train_pro, rec_pro = user_kg_profile(uid, user_gcn_emb, model)
        train_pros.append(train_pro)
        rec_pros.append(rec_pro)
    with open(data_path+train_pro_f_name, "w", encoding="utf-8") as f:
        for i in range(u_num):
            f.write(json.dumps(train_pros[i], ensure_ascii=False) + "\n")
    with open(data_path+rec_pro_f_name, "w", encoding="utf-8") as f:
        for i in range(u_num):
            f.write(json.dumps(rec_pros[i], ensure_ascii=False) + "\n")
if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device

    args = parse_args_kgsr()
    id_dict, relation_dict=load_kg_map(args)
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    Ks = eval(args.Ks)
    logger = getLogger()

    logger.info('PID: %d', os.getpid())
    logger.info(f"DESC: {args.desc}\n")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    save_path = args.out_dir + args.log_fn + '.ckpt'
    kgrec_model = "/home/h666/ai/kgcl/weights/ml_kgrec.ckpt"
    # kgrec_model = "/home/h666/ai/kgcl/weights/test_kgrec.ckpt"
    args.data_type = "pure_train"
    model_dict = {
        'KGSR': KGRec,
    }

    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    # select_success_user(user_dict,n_params)
    model = model_dict[args.model]
    model = model(n_params, args, graph, mean_mat_list[0]).to(device)
    model.print_shapes()
    model.load_state_dict(torch.load(kgrec_model))
    model.eval()
    # uid = 0
    data_path = args.data_path + args.dataset + '/'
    missing_entity=[i for i in range(n_entities) if i not in id_dict]
    print(missing_entity)
    with  torch.no_grad():
        # ret = test(model, user_dict, n_params)
        # print(ret)
        # res = test_bfs_items()
        # print(res)
        entity_gcn_emb, user_gcn_emb = model.generate()
        # movie_prompt()
        # rec_items, rec_scores = get_user_recommendation(5, a_user_gcn_emb)

        # print(test_bfs_items())
        # extract_similar_users(1000)
        # rec_pro_f_name="ml_kg_rec_profile_d4.json"
        # train_pro_f_name="ml_kg_train_profile_d4.json"
        # gene_kg_data4prompt(1000)
        # movie_prompt()


    # get_user_emb_item(10,model, user_dict, n_params)
    # ret = test(model, user_dict, n_params)
    # save_embed(model,n_params)
    #
    # print(ret['recall'],ret['ndcg'], ret['precision'], ret['hit_ratio'])
    # with open("/home/h666/ai/kgcl/user_simi_user.json", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         json_obj = json.loads(line)
    #         print(json_obj)
   
