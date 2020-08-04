# @Time    : 2020/8/4 15:43
# @Author  : Libuda
# @FileName: knn_demo.py
# @Software: PyCharm

#这里用一个花瓣萼片的实例
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
# f = open("iris.data.csv", 'wb')              #可以保存数据
# f.write(str(iris))
# f.close()
knn.fit(iris.data, iris.target)                 #用KNN的分类器进行建模，这里利用的默认的参数，大家可以自行查阅文档
print( iris.data)
print( iris.target)
predictedLabel = knn.predict( [[0.1 ,0.2,5,1.9]])

print ("predictedLabel is :{}".format(predictedLabel))

#通过结合KNN本身的分类算法以及对前k个距离加权，来达到分类的目的 wk-nnc算法是对经典knn算法的改进，
# 这种方法是对k个近邻的样本按照他们距离待分类样本的远近给一个权值w w(i) = (h(k) - h(i)) / (h(k) - h(1)) w(i)是第i个近邻的权值，其中1<i<k,h(i)是待测样本距离第i个近邻的距离