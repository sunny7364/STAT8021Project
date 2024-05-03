from news_id_title_matching import NewsIDTitleMatching
import json

match = NewsIDTitleMatching()

class Read_json: 
    def __init__(self, path='') -> None:
        
        self.prompts = {}
        # 打开包含JSON数据的文件
        with open('/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/10_users.json', 'r', encoding='utf-8') as file:
            # 遍历文件中的每一行
            for line in file:
                # 解析JSON数据
                data = json.loads(line)
                
                system_prompt = ""
                user_history = ""
                user_recommed = ""
                similar_user_history = ""

                # 提取并打印uid
                # print(f"uid: {data['uid']}")
                # self.prompts[data['uid']] = system_prompt
                # print("abc: ", len(data['history']))
                for idx, item in enumerate(data['history']):
                    #print(f" {item}")
                    #print(idx)
                    user_history += match.news_id_title_matching(item[0])
                    user_history += ". The click times of this news: "
                    user_history += str(item[1])
                    user_history += '\n'
                    # if idx == 9:
                    #     print("item: ", item)
                    if idx == 1:#取自己浏览历史里top2的新闻
                        break
                # print(user_history)
                # exit()
                
                # 提取并打印similar_users
                similar_user_news = dict()#使用dict来装是为了去重，相似用户的浏览记录会有重复的新闻
                #下面开始遍历similar user和其下的news为了后续排序做准备
                for user in data['similar_users']:                  
                    for item in user['history']:
                        title = match.news_id_title_matching(item[0])
                        # similar_user_news.update([title, item[1]])
                        similar_user_news[title] = item[1]
                # print('\n', 'dict = :', similar_user_news)
                similar_user_news_list = [[key, value] for key, value in similar_user_news.items()]
                similar_user_news_list.sort(key = lambda x: x[1], reverse = True)
                # print('\n', 'list = :', similar_user_news_list)
                # exit()

                
                for idx, item in enumerate(similar_user_news_list):
                    # print(idx)
                    similar_user_history += item[0]
                    similar_user_history += ". The click times of this news: "
                    similar_user_history += str(item[1])
                    similar_user_history += '\n'
                    if idx == 3:#取相似用户浏览历史里top3的新闻
                        break
                # print(similar_user_history)
                # exit()
                
                # 提取并打印rec_items
                # print("rec_items:")
                for item in data['rec_items']:
                    user_recommed += match.news_id_title_matching(item[0])
                    user_recommed += ". Recommendation Score: "
                    user_recommed += str(item[1])
                    user_recommed += '\n'
                
                # # 在每个JSON对象之后打印一个分隔符，以便更清晰地区分它们
                # print("\n" + "-"*50 + "\n")
                # system_prompt= "You are a news recommendation expert. Please recommend news to the user below and provide a brief explanation.Here is the browsing history of this user:\n" + user_history + "\nHere are the news articles and recommendation score that this user might be interested in:\n" + user_recommed + "\nHere are the news articles that users similar to this user have browsed:\n" + similar_user_history + "\nPlease avoid recommending news articles that the user has already viewed."


                question =  "You are a news recommendation expert. Please recommend top 3 news to the user. Here is the browsing history of this user:"+ user_history
                system_prompt= "\nHere are the news articles and reocmmendation score that this user might be interested in:\n" + user_recommed + "\nHere are the news articles that users similar to this user have browsed:\n" + similar_user_history + "\nPlease avoid recommending news articles that the user has already viewed."
                question_prompt_list = [question, system_prompt]
                self.prompts[data['uid']] = question_prompt_list


    def print_id(self, id):
        try:
            # print(self.prompts[id])
            return self.prompts[id]
        except:
            print(id)


from test_gpt import LLM

my_llm = LLM()
L = Read_json()

# while True:
#     L.print_id(int(input("input: ")))

resultlist = []
for i in range(20):
    data = L.prompts[i]
    print(data)
    question = data[0]
    system_prompt = data[1]
    response = my_llm.call_llm(system_prompt, question)
    # response = i
    result = {
        "Question": question,
        "System Prompt": system_prompt,
        "Answer": response
    }
    resultlist.append(result)

    # Structure the data to be saved in JSON format
    data_to_save = {
        "Qusetion": question,
        "system_prompt": system_prompt,
        "answer": response
    }
      
    # Save the data in a JSON file
    with open("/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/1000user_simi_user_with_newstitle.json", "w") as f:
        for line in resultlist:
            f.write(json.dumps(line)+"\n") 



