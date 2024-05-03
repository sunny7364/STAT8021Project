import requests

def get_relation_by_id(pid):
    url = 'https://www.wikidata.org/w/api.php'

    # 配置请求参数
    params = {
        'action': 'wbgetentities',  # 指定操作为获取实体
        'ids': pid,  # 设置要查询的属性ID
        'props': 'labels',  # 请求实体的标签（labels）信息
        'languages': 'en',  # 请求标签的语言（这里使用英文）
        'format': 'json'  # 指定响应格式为JSON
    }

    # 发送GET请求
    response = requests.get(url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析响应中的JSON数据
        data = response.json()
        # 提取并打印属性的英文描述
        label = data['entities']
        name = label[pid]['labels']['en']['value']
        print(pid, '的描述是:', name)
    else:
        print(pid,'请求失败', response.status_code)
        name="unknown"

    return name