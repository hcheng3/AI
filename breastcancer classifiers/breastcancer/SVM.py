

'''this is my svm'''

import os
import math
import sys
from operator import xor
import random
import pickle

class svm_model:
    def __init__(self,num,dlt):
        self.num = num
        self.a = []
        self.b =0.0
        self.samples = []
        self.y = []
        self.k = []
        self.ygx = []
        self.e = []
        self.init_svm(dlt)

    def init_svm(self,dlt):
        fp= open("breast-cancer-wisconsin.data.txt","r")
        for i,line in enumerate(fp):
            if i <= self.num:
                if "?" in line:
                    line = line.replace("?","1")
                    line = line.split(",")
                    self.samples.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9])])
                else:
                    line = line.split(",")
                    self.samples.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9])])
                if int(line[10]) == 2:
                    self.y.append(1)
                elif int(line[10]) == 4:
                    self.y.append(-1)
                self.a.append(0)
        self.init_k(dlt)
        self.init_ygx()
        self.init_e()

    # Gaussian kernel function
    def kernel(self,i,j,dlt):
        if self.samples[i] == self.samples[j]:
            return math.exp(0)
        dlt = dlt
        result = 0.0
        for idx in range(len(self.samples[i])):
            result += math.pow(self.samples[i][idx] - self.samples[j][idx],2)
        result = math.exp(-result/(2*dlt*dlt))
        return result

    def init_k(self,dlt):
        for i in range(len(self.samples)):
            self.k.append([])
            for j in range(len(self.samples)):
                self.k[i].append(self.kernel(i,j,dlt))

    def gx(self,i):
        predict = 0.0
        for j in range(len(self.samples)):
            predict += self.a[j]*self.y[j]*self.k[i][j]
        predict +=self.b
        return predict

    def init_ygx(self):
        for j in range(len(self.samples)):
            self.ygx.append(self.y[j]*self.gx(j))

    def init_e(self):
        for j in range(len(self.samples)):
            self.e.append(self.gx(j)-self.y[j])

    def update_e(self):
        for j in range(len(self.samples)):
            self.e[j] = (self.gx(j)-self.y[j])

    def update_ygx(self):
        for j in range(len(self.samples)):
            self.ygx[j] = (self.y[j]*self.gx(j))

def select_a2_(i,svm_model):
    if svm_model.e[i] > 0:
        a2_idx = svm_model.e.index(min(svm_model.e))
    if svm_model.e[i] < 0:
        a2_idx = svm_model.e.index(max(svm_model.e))
    return a2_idx

def update_a2(a1idx,svm_model,C,minumstep):
    for i in range(len(svm_model.samples)):

        a2idx = random.randint(0,len(svm_model.samples)-1)
        while a2idx == a1idx:
            a2idx = random.randint(0,len(svm_model.samples)-1)

        k11 = svm_model.k[a1idx][a1idx]
        k22 = svm_model.k[a2idx][a2idx]
        k12 = svm_model.k[a1idx][a2idx]
        eta = k11 + k22 - 2 * k12
        if eta <=0:
            continue
        a2_new = svm_model.a[a2idx] + svm_model.y[a2idx] * (svm_model.e[a1idx] - svm_model.e[a2idx]) / eta
        L = 0.0
        H = 0.0
        a1_old = svm_model.a[a1idx]
        a2_old = svm_model.a[a2idx]
        if svm_model.y[a1idx] == svm_model.y[a2idx]:
            L = max(0, a2_old + a1_old - C)
            H = min(C, a2_old + a1_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(C, C + a2_old - a1_old)
        if L == H:
            continue
        if a2_new > H:
            a2_new = H
        if a2_new < L:
            a2_new = L

        if abs(a2_old-a2_new) < minumstep:
            continue
        break
    return a2_new,a2idx

def model_train(svm_model,C,T,timelimit,minumstep):
    updated = True
    times = 0

    while updated and times < timelimit:
        updated = False
        times += 1

        min_ygx= 100
         # go through the support vector first
        for j in range(len(svm_model.samples)):
            if 0 < svm_model.a[j] < C:
                if (svm_model.e[j]*svm_model.y[j] < -T and svm_model.a[j] < C) or (svm_model.e[j]*svm_model.y[j]> T and svm_model.a[j] > 0):
                    if svm_model.ygx[j]< min_ygx:
                        min_ygx = svm_model.ygx[j]
                        a1idx=j
        if min_ygx == 100:
            j = 0
            # if you cant find any vector break the kkt condition in the support vectors, go through all the vectors
            for j in range(len(svm_model.samples)):
                if (svm_model.e[j]*svm_model.y[j] < -T and svm_model.a[j] < C) or (svm_model.e[j]*svm_model.y[j]> T and svm_model.a[j] > 0):
                    if svm_model.ygx[j]< min_ygx:
                        min_ygx = svm_model.ygx[j]
                        a1idx=j
        if min_ygx != 100:
            updated = True

        if updated == True:
            a1_old = svm_model.a[a1idx]

            a2idx = select_a2_(a1idx,svm_model)

            k11 = svm_model.k[a1idx][a1idx]
            k22 = svm_model.k[a2idx][a2idx]
            k12 = svm_model.k[a1idx][a2idx]
            eta = k11 + k22 - 2 * k12

            if eta >0:
                a2_new = svm_model.a[a2idx] + svm_model.y[a2idx] * (svm_model.e[a1idx] - svm_model.e[a2idx]) / eta
                L = 0.0
                H = 0.0

                a2_old = svm_model.a[a2idx]
                if svm_model.y[a1idx] == svm_model.y[a2idx]:
                    L = max(0, a2_old + a1_old - C)
                    H = min(C, a2_old + a1_old)
                else:
                    L = max(0, a2_old - a1_old)
                    H = min(C, C + a2_old - a1_old)


                if a2_new > H:
                    a2_new = H
                if a2_new < L:
                    a2_new = L

                if L == H:
                    a2_new_cob = update_a2(a1idx,svm_model,C,minumstep)
                    a2_new =a2_new_cob[0]
                    a2idx = a2_new_cob[1]
            else:
                a2_new_cob = update_a2(a1idx,svm_model,C,minumstep)
                a2_new =a2_new_cob[0]
                a2idx = a2_new_cob[1]

            if abs(a2_old-a2_new) < minumstep :
                a2_new_cob = update_a2(a1idx,svm_model,C,minumstep)
                a2_new =a2_new_cob[0]
                a2idx = a2_new_cob[1]

            a1_new = a1_old + svm_model.y[a2idx] *svm_model.y[a1idx] * (a2_old - a2_new)
            b1_new = svm_model.b - svm_model.e[a1idx] - svm_model.y[a1idx] * svm_model.k[a1idx][a1idx] * (a1_new - a1_old) - svm_model.y[a2idx] * svm_model.k[a2idx][a1idx] * (a2_new - a2_old)
            b2_new = svm_model.b - svm_model.e[a2idx] - svm_model.y[a1idx] * svm_model.k[a1idx][a2idx] * (a1_new - a1_old) - svm_model.y[a2idx] * svm_model.k[a2idx][a2idx] * (a2_new - a2_old)
            if a1_new > 0 and a1_new < C:
                b_new = b1_new
            elif a2_new > 0 and a2_new < C:
                b_new = b2_new
            else:
                b_new = (b1_new+b2_new) / 2.0

            svm_model.a[a1idx] = a1_new
            svm_model.a[a2idx] = a2_new
            svm_model.b = b_new

            svm_model.update_ygx()
            svm_model.update_e()
            # print "iteration: %d, a1: %d, a2: %d" %(times,a1idx,a2idx)
            print "iteration: %d, a1: %d, a2: %d" %(times,a1idx,a2idx)
            # print "old a1: %f" %(a1_old)
            # print "old a2: %f" %(a2_old)
            # print "new a1: %f" %(a1_new)
            # print "new a2: %f" %(a2_new)

class test_model:
    def __init__(self,num,svm_model,dlt):
        self.samples = []
        self.y = []
        self.k = []
        self.num = num
        self.init_test(svm_model,dlt)

    def init_test(self,svm_model,dlt):
        fp= open("breast-cancer-wisconsin.data.txt","r")
        for i,line in enumerate(fp):
            if i > self.num:
                if "?" in line:
                    line = line.replace("?","0")
                    line = line.split(",")
                    self.samples.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9])])
                else:
                    line = line.split(",")
                    self.samples.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9])])
                if int(line[10]) == 2:
                    self.y.append(1)
                elif int(line[10]) == 4:
                    self.y.append(-1)
        self.init_tk(svm_model,dlt)


    def kernel1(self,i,j,svm_model,dlt):
        if self.samples[i] == svm_model.samples[j]:
            return math.exp(0)
        dlt = dlt
        result = 0.0
        for idx in range(len(svm_model.samples[i])):
            result += math.pow(self.samples[i][idx] - svm_model.samples[j][idx],2)
        result = math.exp(-result/(2*dlt*dlt))
        return result

    def init_tk(self,svm_model,dlt):
        for i in range(len(self.samples)):
            self.k .append([])
            for j in range(len(svm_model.samples)):
                self.k[i].append(self.kernel1(i,j,svm_model,dlt))

    def gxt(self,i,svm_model):
        predict = 0.0
        for j in range(len(svm_model.samples)):
            predict += svm_model.a[j]*svm_model.y[j]*self.k[i][j]
        predict +=svm_model.b
        if predict >0: return 1
        else: return -1

    def test(self,svm_model):
        right =0
        wrong = 0
        for i in range(len(self.samples)):
            pred = self.gxt(i,svm_model)
            if pred == self.y[i]:
                right +=1
            else:
                wrong +=1
        print "right:", right
        print "wrong:", wrong
        print "right percent %d %%:" %(float(right)/(right+wrong)*100)

class test_svm():
    def __init__(self,num,dlt,C,T,timeslimits,minstep):
        self.svm = svm_model(num,dlt)
        model_train(self.svm,C,T,timeslimits,minstep)


        print self.svm.a
        print self.svm.b
        self.test1 = test_model(num,self.svm,dlt)
    def testing(self):
        self.test1.test(self.svm)

# parameters
number_train = 400
dlt_guassian =7.95
C = 5.0
tolerence =0.0001
looptimelimits=200
minstep =0.0001


# if you want to change the parameters and train for yourself, just get rid of the pickle one and uncomment
#  the testsvm init line. just be aware the change of the dlt_guassian could affect the result alot, so you need to
# optimize the dlt_guassian if you change the amount of the data for train.

# testsvm = test_svm(number_train,dlt_guassian,C,tolerence,looptimelimits,minstep)
# pickle.dump(testsvm,open( "classifier.txt", "wb" ))

testsvm = test_svm(number_train,dlt_guassian,C,tolerence,looptimelimits,minstep)

# testsvm = pickle.load( open( "classifier.txt", "rb" ) )
testsvm.testing()
