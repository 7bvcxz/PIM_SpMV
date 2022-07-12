import torch
import torchvision.models as models
import numpy as np
import random

model = models.resnet50(pretrained=True)

a = model.fc.weight
a = a.abs()
a = a > 0.027
print(a)

"""
total = 0
for i in range(1000):
    for j in range(2048):
        if a[i][j] == True:
            total = total + 1

print(1000*2048/4)
print(total)

b = a.sum(axis=0)
print(b)

max_col = b.argmax()
print(max_col)

c = np.zeros(2048)
for i in range(1000):
    if a[i][max_col] == True:
        for j in range(2048):
            if a[i][j] == True:
                c[j] = c[j] + 1

print(c)
"""
d = a.sum(axis=1)
print(d)
print(d.size())

for i in range(8):
    for j in range(4):
        d0 = d[i*125+31*j:i*125+31*j+31].sum()
        print(d0)
    print("--")

