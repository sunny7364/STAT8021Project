# A function to match news_id with news_title

import pandas as pd
import numpy as np

class NewsIDTitleMatching:

    def __init__(self) -> None:
        
        newsinfo_trainingdataset = pd.read_csv('/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/news.tsv', sep = '\t', header = None, index_col = None, names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entites'])
        newsinfo_testingdataset = pd.read_csv('/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/news_test.tsv', sep = '\t', header = None, index_col = None, names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entites'])
        newsinfo_validdataset = pd.read_csv('/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/news_valid.tsv', sep = '\t', header = None, index_col = None, names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entites'])
        newsinfo = pd.concat([newsinfo_trainingdataset, newsinfo_testingdataset, newsinfo_validdataset])
        newsinfo.drop_duplicates()
        self.news_info = newsinfo

    def news_id_title_matching(self, news_id):
        news_id = str(news_id)
        if news_id[0] == 'N':
            try:
                self.news_title = str(self.news_info.loc[self.news_info['News ID'] == news_id, 'Title'].values[0])
                # print("self.news_title ", self.news_title)
            except:
                print("bug1")
                breakpoint()
        if news_id[0] != 'N':
            try:
                self.news_title = str(self.news_info.loc[self.news_info['News ID'] == 'N' + str(news_id), 'Title'].values[0])
                # print("self.news_title ", self.news_title)
            except:
                print("bug2")
                breakpoint()
        return self.news_title
    

