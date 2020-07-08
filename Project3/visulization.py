import pickle
from pylab import mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']

file=open('./project3/model/PointNetCls-200-0.0100-50_loss.pickle','rb')
acc=pickle.load(file)
file.close()
x = range(1, len(acc)+1)

plt.title(u'Loss of Pointnet model')
plt.xlabel('Iteration')
plt.ylabel('Loss')
# plt.scatter(x, y, s, c, marker)
# x: x轴坐标
# y：y轴坐标
# s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
# c: 点的颜色 
# marker: 标记的样式 默认是 'o'
plt.legend()

plt.plot(x, acc,'m.-.', linewidth = 2)
plt.show()