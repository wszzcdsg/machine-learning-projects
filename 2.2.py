import os
import pandas as pd

os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,Pave,127500\n') #每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

#2.2.2处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]#iloc[]函数用于根据索引选择数据
inputs = inputs.fillna(inputs.mean())#fillna()函数用于填充缺失值
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)#get_dummies()函数用于将类别变量转换为哑变量
print(inputs)

#2.2.3转换为张量格式
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)#values属性返回DataFrame中的值
print(X, y)