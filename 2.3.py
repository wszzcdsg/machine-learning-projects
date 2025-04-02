import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x ** y)  # **运算符是求幂运算

x = torch.arange(4)
print(x)

print(x[3])
print(len(x))

#2.3.3矩阵

A = torch.arange(20).reshape(5, 4)
print(A)

print(A.T)#转置

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)

print(B == B.T)#对称矩阵

#2.3.4张量

x = torch.arange(24).reshape(2, 3, 4)
print(x)

#2.3.5张量算法的基本性质

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B)

print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

#2.3.6降维

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

print(A.shape)
print(A.sum())#默认对所有元素求和

A_sum_axis0 = A.sum(axis=0)#axis=0表示按列求和
print(A_sum_axis0, A_sum_axis0.shape)

print(A.sum(axis=1))
print(A.sum(axis=1).shape)

print(A.sum(axis=[0, 1]))  # 同时对两个轴求和
print(A.sum(axis=[0, 1]).shape)

print(A.mean(), A.sum() / A.numel())#numel()函数返回张量中的元素个数

#非降维求和
sum_A = A.sum(axis=1, keepdims=True)#keepdims=True表示保留原始张量的维度
print(sum_A)
print(sum_A.shape)


print(A / sum_A)#广播机制

print(A.cumsum(axis=0))#cumsum()函数返回张量中从0到当前位置的元素的和

#2.3.7点积

y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y), torch.sum(x * y))#torch.dot()函数返回两个向量的点积

#2.3.8矩阵-向量积

print(A.shape, x.shape, A, x, torch.mv(A, x))#torch.mv()函数返回矩阵A和向量x的矩阵-向量积

#2.3.9矩阵-矩阵乘法

B = torch.ones(4, 3)
print(A, B, torch.mm(A, B))#torch.mm()函数返回矩阵A和矩阵B的矩阵-矩阵乘法

#2.3.10范数

u = torch.tensor([3.0, -4.0])
print(torch.norm(u))#torch.norm()函数返回向量u的范数

print(torch.abs(u).sum())#torch.abs()函数返回向量u的绝对值

print(torch.norm(torch.ones((4, 9))))#torch.norm()函数返回矩阵的Frobenius范数