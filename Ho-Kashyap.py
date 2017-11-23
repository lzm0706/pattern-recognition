import numpy as np
#数据
w1 = np.array([(0.1,1.1),(6.8,7.1),(-3.5,-4.1),(2.0,2.7),(4.1,2.8),(3.1,5.0),(-0.8,-1.3),(0.9,1.2),(5.0,6.4),(3.9,4.0)])
w2 = np.array([(7.1,4.2),(-1.4,-4.3),(4.5,0.0),(6.3,1.6),(4.2,1.9),(1.4,-3.2),(2.4,-4.0),(2.5,-6.1),(8.4,3.7),(4.1,-2.2)])
w3 = np.array([(-3.0,-2.9),(0.5,8.7),(2.9,2.1),(-0.1,5.2),(-4.0,2.2),(-1.3,3.7),(-3.4,6.2),(-4.1,3.4),(-5.1,1.6),(1.9,5.1)])
w4 = np.array([(-2.0,8.4),(-8.9,0.2),(-4.2,-7.7),(-8.5,-3.2),(-6.7,-4.0),(-0.5,-9.2),(-5.3,-6.7),(-8.7,-6.4),(-7.1,-9.7),(-8.0,-6.3)])

#增广表示
rownum=np.shape(w1)[0]
o=np.ones(rownum)
w1_z=np.c_[w1,o]
w2_z=np.c_[w2,o]
w3_z=np.c_[w3,o]
w4_z=np.c_[w4,o]

def ho_kashyap(a,n,bmin,w1,w2):#n<1,更新因子（学习率）
    samples2=-w2  #规范化
    samples1=w1
    samples=np.r_[samples1,samples2]
    b=np.ones(np.shape(samples)[0]).T #初始化b
    kmax=1000 #初始化最大迭代次数
    #print(kmax)
    k=0
    while k<kmax:
        k=k+1
        e=np.dot(samples,a)-b
        ep=0.5*(e+np.absolute(e))
        b=b+2*n*ep
        yp=np.linalg.pinv(samples)#求伪逆矩阵
        a=np.dot(yp,b)
        J_s = np.sum((np.dot(samples, a) - b) ** 2)#以准则函数代表训练误差
        if e[np.absolute(e)>bmin].size==0:#收敛条件判别
            print('a=\n',a,'\nb=\n',b)
            print('训练误差：',J_s)
            return a,b
            break
    print('no solutions found\n')
    print('a=\n',a,'\nb=\n',b)
    print('训练误差：',J_s)
    
#初始化
a=np.zeros(np.shape(w1_z)[1]).T
n=0.5#n<1,更新因子（学习率）
bmin=0.5
ho_kashyap(a,n,bmin,w2_z,w4_z)