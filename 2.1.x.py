import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
print(x + y, x - y, x * y, x / y, x ** y)

print(torch.exp(x))

x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)#dim=0表示按行拼接，dim=1表示按列拼接
print(torch.cat((x, y), dim=0), torch.cat((x, y), dim=1))

print(x == y)

print(x.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)

print(a + b)

print(x)
print(x[-1])
print(x[1:3])

x[1, 2] = 9#将第2行第3列的元素赋值为9
print(x)

x[0:2, :] = 12#将第1行和第2行的所有元素赋值为12
print(x)
before = id(y)#id()函数返回对象的内存地址
y = y + x
print(id(y) == before)#id()函数返回对象的内存地址

z = torch.zeros_like(y)#zeros_like()函数返回一个与y形状相同的全0张量
print('id(z):', id(z))
z[:] = x + y#将x+y的结果赋值给,并且地址不变
print('id(z):', id(z))

before = id(x)
x += y#将x+y的结果赋值给x
print(id(x) == before)#id()函数返回对象的内存地址

A = x.numpy()#将张量转换为NumPy数组
B = torch.tensor(A)#将NumPy数组转换为张量
print(type(A), type(B))

a = torch.tensor([3.5])#将Python标量转换为张量
print(a, a.item(), float(a), int(a))#item()函数将张量转换为Python标量
