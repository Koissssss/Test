import numpy as np
import matplotlib.pyplot as plt

class SLRFrame:
    def __init__(self):
        #用于记录乘子的移动轨迹
        self.Xdata = [] #x坐标
        self.Ydata = [] #y坐标

        #用于更新步长c的参数
        self.K = 0 #迭代次数
        self.M = 100
        self.R = 0.95
        self.alpha = 0.9

        self.LProblem = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1]) #原问题
        self.Lambda1 = np.array([-1.0, 0.2, -1.0, 0.2, -1.0, 0.2]) #约束1
        self.Lambda2 = np.array([-5.0, 1.0, -5.0, 1.0, -5.0, 1.0]) #约束2
        
        #初始化乘数λ0
        self.multilpler = np.array([0.0, 0.0])

        #计算初始解x0
        #乘子为0的情况下，L(x,λ)最小值显然在满足x=0的情况下得到
        self.x_sol = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        #估计最优解q*为180，以此计算步长
        self.c = (180 - self.countFunc()) / pow(np.linalg.norm(self.countGx()), 2)
        self.preGx = self.countGx()

    #计算L(x, λ)
    def countFunc(self):
        sum = 0
        for i in range(6):
            sum += self.LProblem[i] * self.x_sol[i] * self.x_sol[i]
            sum += self.multilpler[0] * self.Lambda1[i] * self.x_sol[i]
            sum += self.multilpler[1] * self.Lambda2[i] * self.x_sol[i]
        #约束内常数
        sum += self.multilpler[0] * 48
        sum += self.multilpler[1] * 250
        return sum

    #计算G(x)
    def countGx(self):
        return np.array([48 + np.dot(self.Lambda1, self.x_sol.transpose()), 250 + np.dot(self.Lambda2, self.x_sol.transpose())])

    #迭代18*2=36次
    def process(self):
        while(self.K < 18):
            self.cycle()
        self.end()

    #单次循环，对子问题1，2，3迭代一次，对子问题4，5，6迭代一次
    def cycle(self):
        #循环次数增加
        self.K += 1
        #更新α
        self.alpha *= 1 - 1 / (self.M * pow(self.K, 1 - 1 / pow(self.K, self.R)))

        #子问题1，2，3
        #更新步长c
        self.c = self.alpha * self.c * np.linalg.norm(self.preGx) / np.linalg.norm(self.countGx())
        self.preGx = self.countGx()
        #更新乘子
        self.multilpler += self.c * self.countGx()
        #确保乘子不小于0
        if self.multilpler[0] < 0:
            self.multilpler[0] = 0
        if self.multilpler[1] < 0:
            self.multilpler[1] = 0
        #记录乘子移动轨迹
        self.Xdata.append(self.multilpler[0])
        self.Ydata.append(self.multilpler[1])
        
        #对每个子问题进行二次函数最小值点的求解
        self.x_sol[0] = self.multilpler[0] + 5 * self.multilpler[1]
        self.x_sol[1] = -self.multilpler[0] - 5 * self.multilpler[1]
        self.x_sol[2] = self.multilpler[0] + 5 * self.multilpler[1]
        
        #子问题4，5，6，同上
        self.c = self.alpha * self.c * np.linalg.norm(self.preGx) / np.linalg.norm(self.countGx())
        self.preGx = self.countGx()
        self.multilpler += self.c * self.countGx()
        
        if self.multilpler[0] < 0:
            self.multilpler[0] = 0
        if self.multilpler[1] < 0:
            self.multilpler[1] = 0

        self.Xdata.append(self.multilpler[0])
        self.Ydata.append(self.multilpler[1])
        
        
        self.x_sol[3] = -self.multilpler[0] - 5 * self.multilpler[1]
        self.x_sol[4] = self.multilpler[0] + 5 * self.multilpler[1]
        self.x_sol[5] = -self.multilpler[0] - 5 * self.multilpler[1]

    def end(self):
        print(self.x_sol)
        print(self.countFunc())
        plt.plot(self.Xdata, self.Ydata, marker='o')
        plt.title('Simple Line Plot')
        plt.xlabel('λ1')
        plt.ylabel('λ2')
        plt.show()

problem = SLRFrame()
problem.process()