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
