import os

# 定义一个函数来读取.dat文件
def read_dat_file(file_path):
    try:
        # 使用'with'语句打开文件，这样可以确保文件在使用后会被正确关闭
        with open(file_path, 'r') as file:
            # 读取文件内容
            data = file.read()
            # 返回文件内容
            return data
    except FileNotFoundError:
        print(f"文件{file_path}未找到。")
        return None
    except Exception as e:
        print(f"读取文件时出现错误：{e}")
        return None

# 定义一个函数来将数据写入到.txt和.csv文件
def write_to_txt_and_csv(data, original_file_path):
    if data is not None:
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(original_file_path)[0]
        
        # 写入.txt文件
        txt_file_path = f"{base_name}.txt"
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(data)
        print(f"数据已写入到{txt_file_path}")
        
        # 写入.csv文件
        csv_file_path = f"{base_name}.csv"
        with open(csv_file_path, 'w') as csv_file:
            csv_file.write(data)
        print(f"数据已写入到{csv_file_path}")

# 使用函数读取.dat文件
file_path = '/home/SENSETIME/yujiayang/Desktop/8021/ml1m/kg/valid.dat'  # 替换为你的.dat文件路径
data = read_dat_file(file_path)

# 将读取的数据写入到.txt和.csv文件
write_to_txt_and_csv(data, file_path)
