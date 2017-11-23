import numpy as np
#数据
w1 = np.array([(0.1,1.1),(6.8,7.1),(-3.5,-4.1),(2.0,2.7),(4.1,2.8),(3.1,5.0),(-0.8,-1.3),(0.9,1.2),(5.0,6.4),(3.9,4.0)])
w2 = np.array([(7.1,4.2),(-1.4,-4.3),(4.5,0.0),(6.3,1.6),(4.2,1.9),(1.4,-3.2),(2.4,-4.0),(2.5,-6.1),(8.4,3.7),(4.1,-2.2)])
w3 = np.array([(-3.0,-2.9),(0.5,8.7),(2.9,2.1),(-0.1,5.2),(-4.0,2.2),(-1.3,3.7),(-3.4,6.2),(-4.1,3.4),(-5.1,1.6),(1.9,5.1)])
w4 = np.array([(-2.0,8.4),(-8.9,0.2),(-4.2,-7.7),(-8.5,-3.2),(-6.7,-4.0),(-0.5,-9.2),(-5.3,-6.7),(-8.7,-6.4),(-7.1,-9.7),(-8.0,-6.3)])
'''
print(np.shape(w1))
print(np.shape(w2))
print(np.shape(w3))
print(np.shape(w4))
'''
#增广表示
rownum=np.shape(w1)[0]
o=np.ones(rownum)
w1_z=np.c_[w1,o]
w2_z=np.c_[w2,o]
w3_z=np.c_[w3,o]
w4_z=np.c_[w4,o]

def batch_perception(w1,w2,a,l_r,n):#l_r:学习率,n:准则值
    samples2=-w2  #规范化
    samples1=w1
    samples=np.r_[samples1,samples2]
    
    #找错分样本
    mask=np.dot(a,samples.T)<=0
    #print (mask)
    error_samples=samples[mask]
    #print(error_samples)
    g=error_samples.sum(axis=0)#梯度
    count=0
    while np.linalg.norm(l_r*g)>=n:#判断是否到达收敛要求
        
        a=a+l_r*g#权值更新
        
        #找错分样本
        mask=np.dot(a,samples.T)<=0
        error_samples=samples[mask]
        g=error_samples.sum(axis=0)#梯度
        count+=1
    print('收敛步数：',count)
    print('权值向量：',a)

#初始化
a=np.zeros(np.shape(w1_z)[1])
l_r=1
n=0.5
#迭代计算
batch_perception(w1_z,w2_z,a,l_r,n)
batch_perception(w3_z,w2_z,a,l_r,n)

