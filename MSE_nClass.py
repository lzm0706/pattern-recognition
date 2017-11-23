import numpy as np
#数据
w1 = np.array([(0.1,1.1),(6.8,7.1),(-3.5,-4.1),(2.0,2.7),(4.1,2.8),(3.1,5.0),(-0.8,-1.3),(0.9,1.2),(5.0,6.4),(3.9,4.0)])
w2 = np.array([(7.1,4.2),(-1.4,-4.3),(4.5,0.0),(6.3,1.6),(4.2,1.9),(1.4,-3.2),(2.4,-4.0),(2.5,-6.1),(8.4,3.7),(4.1,-2.2)])
w3 = np.array([(-3.0,-2.9),(0.5,8.7),(2.9,2.1),(-0.1,5.2),(-4.0,2.2),(-1.3,3.7),(-3.4,6.2),(-4.1,3.4),(-5.1,1.6),(1.9,5.1)])
w4 = np.array([(-2.0,8.4),(-8.9,0.2),(-4.2,-7.7),(-8.5,-3.2),(-6.7,-4.0),(-0.5,-9.2),(-5.3,-6.7),(-8.7,-6.4),(-7.1,-9.7),(-8.0,-6.3)])

#增广表示
rownum=np.shape(w1)[0]
o=np.ones(rownum)
w1_z=np.c_[o,w1]
w2_z=np.c_[o,w2]
w3_z=np.c_[o,w3]
w4_z=np.c_[o,w4]

# train data and test data, 每个类别前8个为训练样本，后2个为测试样本
train1 = w1_z[:8, :]
test1 = w1_z[8:, :]
train2 = w2_z[:8, :]
test2 = w2_z[8:, :]
train3 = w3_z[:8, :]
test3 = w3_z[8:, :]
train4 = w4_z[:8, :]
test4 = w4_z[8:, :]
#合并样本为一个矩阵
y = np.r_[train1, train2, train3, train4]#Y
test = np.r_[test1, test2, test3, test4]
#构造B
b=np.zeros((32,4))
b[:8,:][:,0]=1
b[8:16,:][:,1]=1
b[16:24,:][:,2]=1
b[24:32,:][:,3]=1

labelt=np.array([0,0,1,1,2,2,3,3])#测试数据标注

#print(labelt)
#print(y)
#print(test)
#print(b)

def MSE_nClass(y,b):
    yp=yp=np.linalg.pinv(y)#伪逆矩阵
    a=np.dot(yp,b)#A
    #print(a)
    return a

a=MSE_nClass(y,b)
c=0
for i in range(np.shape(test)[0]):#一次取一个测试样本
    r=np.dot(a.T,test[i,:])#求一个样本在四个决策函数下的值
    if labelt[i]==np.where(r==np.max(r))[0]:#测试样本标注与决策函数结果最大时对应的标注作比较
        c+=1#计数正确次数
c_r=c/np.shape(test)[0]#正确率
print('正确率:',c_r)