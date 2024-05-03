import json
from test_gpt import LLM

my_llm = LLM()
resultlist = []

class Movie_Response:
    def __init__(self) -> None:
        total = 0
        with open("/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/movie_result.json", "r") as file: 
            total = 0
            for line in file:
                total += 1
        print(total)

        with open('/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/ml_prompt_d4.json', 'r', encoding='utf-8') as file:
            # 遍历文件中的每一行
            for idx, line in enumerate(file):
                if idx < total:
                    continue
                # 解析JSON数据
                data = json.loads(line)
                uid = data['uid']
                question = data['question']
                prompt = data['system_prompt'] + "\nPlease avoid recommending movies that the user has already viewed"
                response = my_llm.call_llm(prompt, question)
                result = {
                    "Question": question,
                    "System Prompt": prompt,
                    "Answer": response
                }
                if idx == 0:
                    with open("/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/movie_result.json", "w") as f:
                        
                        f.write(json.dumps(result)+"\n")
                else:
                    with open("/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/movie_result.json", "a") as f:
                            f.write(json.dumps(result)+"\n")
            #     resultlist.append(result)
            # # Save the data in a JSON file
            #     print(resultlist)
        # with open("/Users/bytedance/Desktop/钟韵涵个人文件夹/8021_project/user_rec_matching_title/test_sample/movie_result.json", "w") as f:
        #     for line in resultlist:
        #         f.write(json.dumps(line)+"\n") 
    
if __name__ == '__main__':
    Movie_Response()