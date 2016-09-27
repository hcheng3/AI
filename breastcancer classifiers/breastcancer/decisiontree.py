'''this is my decision tree'''
import numpy as np
import math
import copy

class Node(object):
    "Generic tree node."
    def __init__(self,subset,decision= None,label='root',leaf ='notleaf'):
        self.label = label
        self.subset = subset
        self.children = []
        self.decision = decision
        self.leaf = leaf

    def add_child(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

class tree():
    def __init__(self,num,min_example):
        self.num = num
        self.min_example = min_example
        self.examples = self.init_examples()
        self.root = self.train_tree()

    def init_examples(self):
        a = []
        fp= open("breast-cancer-wisconsin.data.txt","r")
        for i,line in enumerate(fp):
            if i <= self.num:
                if "?" in line:
                    line = line.replace("?","1")
                    line = line.split(",")
                    a.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9]),int(line[10])])
                else:
                    line = line.split(",")
                    a.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9]),int(line[10])])
        return np.array(a)

    def train_tree(self):
        attributes = []
        num1 = 0
        num2 = self.examples.shape[0]
        attributes_init = np.array(range(0,self.examples.shape[1]-1))
        start_attribute = self.infoGain(num1,num2,attributes_init)
        attributes = attributes_init.tolist()
        attributes.remove(start_attribute)
        return self._train_tree("root",num1,num2,start_attribute,attributes)

    # num1,num2: the first and last line which will be create to dict in the examples
    # best_A: index of attribute which will be used as key
    def create_dict(self,num1,num2,best_A):
        b = {}
        for i in range(num1,num2):
            key = self.examples[i][best_A]
            b.setdefault(key,[])
            b[key].append(self.examples[i])
        return b

    def find_common(self,num1,num2):
        count4 =0
        count2 =0
        for i in range(num1,num2):

            if self.examples[i][9] ==4:
                count4 +=1
            if self.examples[i][9] ==2:
                count2 +=1

        if count4 >= count2:
            return "-"
        else:return "+"
    # attributes = attributes - target_attribute
    # attributes is an array
    def _train_tree(self,key,num1,num2,target_attribute ,attributes):
        # need change to sort by the num1 to num2 lines
        # print "num1,num2:",num1,num2
        media =self.examples[num1:num2]

        self.examples[num1:num2] = media[media[:,target_attribute].argsort()]

        # print "num1,num2:",num1,num2
        # print "sort based on target attribute(column,decision):",target_attribute
        # print"key which is label:",key
        # print "sorted:"
        # print self.examples

        # print  self.examples
        node = Node((num1,num2))
        node.label = key
        node.decision = target_attribute

        # check whether we got a pure set or not,if its a pure set directly return the node with its label 4 or 2
        count4 = 0
        count2 = 0
        for i in range(num1,num2):
            if self.examples[i][9] ==4:
                count4 +=1
            if self.examples[i][9] != 4:
                break
        # print "count4:" ,count4
        if count4 == num2-num1:
            node.label = key
            node.leaf="-"

            return node
        for j in range(num1,num2):
            if self.examples[j][9] ==2:
                count2 +=1
            if self.examples[j][9] != 2:
                    break


        if count2 == num2-num1:
            node.label = key
            node.leaf  = "+"

            return node
        #if its not a pure set, we use the target_attribute as decision attribute
        node.decision = target_attribute
        # sort on this attribute ,key represent the calss I got after sort
        dictionary = self.create_dict(num1,num2,target_attribute)

        # cause the existing of the info gain heuristic so when this happened, there isn't any other
        # attribute which could provide more information for classification, so directly set this
        # as a node and assign 4 or 2 to it

        for key in dictionary:
            if len(dictionary[key]) == num2-num1:
                # print "n1 n2:",num1,num2
                node.label = key
                node.leaf  = self.find_common(num1,num2)
                return node

        # if the attributes left is 0 which mean, there is no more attributes could be based on to classify
        # or this subset is too small less than 5, we use it as a class
        # directly give it a 4 or 2
        # print "attributes:", attributes
        if len(attributes) == 0 or (num2-num1)< self.min_example:
            print "pruning"
            node.label = key
            node.leaf  = self.find_common(num1,num2)
            return node

        for key in dictionary:
            if len(dictionary[key]) != 0:
                subsize = self.find_sub(num1,num2,key,target_attribute)
                target_attribute_new = self.infoGain(subsize[0],subsize[1],attributes)
                attributes1=copy.deepcopy(attributes)
                # something wrong with attributes
                attributes1.remove(target_attribute_new)
                node.add_child(self._train_tree(key,subsize[0],subsize[1],target_attribute_new, attributes1))
        return node

    def find_sub(self,num1,num2,key,target_attribut):
        count=0
        for i in range(num1,num2):
            if self.examples[i][target_attribut]==key:
                break
        for j in range(num1,num2):
            if self.examples[j][target_attribut]==key:
                count += 1

        return i ,i+count

    def infoGain(self,num1,num2,attributes):
        hs_array = []
        for i in attributes:
            hs =0
            a = self.create_dict(num1,num2,i)
            for key in a:
                hs +=self.entropy(a,key)*len(a[key])/(num2-num1)
            hs_array.append((hs,i))
        return  min(hs_array)[1]

    def entropy(self,dictionary,key):
        count2 = 0
        count4 = 0
        for i in range(len(dictionary[key])):
            if dictionary[key][i][9] ==4:
                count4 +=1
            elif dictionary[key][i][9] ==2:
                count2 +=1

        proportion_2 = float(count2)/(count2+count4)
        proportion_4 = float(count4)/(count2+count4)
        if proportion_4 == 0.0 or proportion_2 ==0.0:
            return 0
        hs = -(float(proportion_2)*math.log(proportion_2,2)+float(proportion_4)*math.log(proportion_4,2))
        return hs

class test():
    def __init__(self,num):
        self.num =num
        self.test_example =self.init_test()

    def init_test(self):
        test = []
        fp= open("breast-cancer-wisconsin.data.txt","r")
        for i,line in enumerate(fp):
            if i > self.num:
                if "?" in line:
                    line = line.replace("?","1")
                    line = line.split(",")
                    test.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9]),int(line[10])])
                else:
                    line = line.split(",")
                    test.append([int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6]),int(line[7]),int(line[8]),int(line[9]),int(line[10])])
        return np.array(test)

    def testing(self,tree):
        right = 0
        wrong = 0
        for i in range(len(self.test_example)):
            b = self.predict(tree.root,i)
            if b =="right":
                right +=1
            elif b=="wong"or"unknow":
                wrong +=1
        print "right:", right
        print "wrong:", wrong
        print "right percent %f %%:" %(float(right)/(right+wrong)*100)

    def predict(self,node,i):
        if node.leaf == "+" and self.test_example[i][9] == 2 or node.leaf == "-" and self.test_example[i][9] == 4:
            return "right"
        if node.leaf == "+" and self.test_example[i][9] == 4 or node.leaf == "-" and self.test_example[i][9] == 2:
            return "wrong"
        new_node= None

        for x in node.children:
            if x.label == self.test_example[i][node.decision]:
                new_node = x
                break
        if new_node == None:
            dif = []
            for x in node.children:
                diff = math.fabs(x.label - self.test_example[i][node.decision])
                dif.append((diff,x))
            new_node = min(dif)[1]

        return self.predict(new_node,i)


train_num=579
min_for_subset=2


tree1 = tree(train_num,min_for_subset)
test1 = test(train_num)
test1.testing(tree1)