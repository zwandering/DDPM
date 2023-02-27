import matplotlib.pyplot as plt
import math
import torch
t=torch.tensor([1,2,3])
print(10**t)
def f(x):
    return 20**x
plt.figure()
x=[i/100 for i in range(100)]
y=[f(i/100)for i in range(100)]
plt.plot(x,y)
plt.show()