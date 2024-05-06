import requests

class LLM:
    def __init__(self) -> None:
        # self.url = "https://mtc2023australia.openai.azure.com/openai/deployments/gpt-35-turbo-1106/chat/completions?api-version=2023-07-01-preview"
        # self.key = '30e1842328f34dbf8f6e234527af5403'
        
        self.url = "https://api.132006.xyz/v1/chat/completions"

    def call_llm(self, context, question):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key}'
        }

        request_data = {
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content':  question}
            ],
            'stream': False,
            'model': 'gpt-3.5-turbo',
            'temperature': 0.5,
            'presence_penalty': 0,
            'frequency_penalty': 0,
            'top_p': 1
        }

        response = requests.post(self.url, headers=headers, json=request_data)
        data = response.json()

        # 处理响应数据
        # print(data['choices'][0]['message']['content']) 
        print(data) 
        
        return data['choices'][0]['message']['content']

