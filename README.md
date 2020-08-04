参照贪心学院 机器学习高阶课程大纲 进行学习
太穷啦~  只能按照大纲自己找资料学习~
## 一.机器学习基础与凸优化
#### 1.1 kNN和Weighted kNN
    参考资料：
              https://blog.csdn.net/qq_43416572/article/details/100582970
              https://www.cnblogs.com/bigmonkey/p/7387943.html
              https://www.cnblogs.com/jyroy/p/9427977.html
     学习要求：
                1.简述knn算法的流程
                    对于一个未知的数据，从已知数据集中求出每个数据与其的"距离",取其中最接近的k个,然后通过多数表决的形式，即统计k个数据中种类最多的一个类，
                    我们认为这个未知数据为该类别。
                2.训练集和测试集是什么？
                    训练集可以理解为已知的带有标签的数据，测试集理解为未知数据。
                3.k值大小有什么影响？什么值最合适？
                    k值太小不具备抗干扰性 可能最近的几个k值中有噪音
                    k值太大不具备代表性 相当于较大领域中训练 近似误差较大
                    一般的k取值不会超过20，上限是n的开方，理论上训练集越大，k值越大。
                4.有哪些常用的度量距离呢？
                    欧氏距离，余弦值，相关度，曼哈顿距离(城市街区距离)。
                5.加权knn? 加的什么权？
                    反比例函数加权，高斯加权等等。个人感觉不必了解过多。
                    假如k=3
                    三个类别分别为A、A、B
                    一般来讲，A有2个，B有1个，那么判别结果为A
                    加权情况下，三个邻近的权重分别为A（0.8），A（0.6），B（0.5）
                    相当于最后有0.8+0.6=1.4（个）A，0.5（个）B，所以最后选A。

                    #通过结合KNN本身的分类算法以及对前k个距离加权，来达到分类的目的 wk-nnc算法是对经典knn算法的改进，
                    # 这种方法是对k个近邻的样本按照他们距离待分类样本的远近给一个权值w w(i) = (h(k) - h(i)) / (h(k) - h(1))
                    w(i)是第i个近邻的权值，其中1<i<k,h(i)是待测样本距离第i个近邻的距离


####  1.2 Approximated KNN算法
      个人觉得，只做简单了解就好，不必深究。
      Ann, Approximate Nearest Neighbor的缩写，就是近似最近邻搜索。
      参考资料：
                https://blog.csdn.net/suibianshen2012/article/details/101517801
                https://zhuanlan.zhihu.com/p/37381294
                https://www.ryanligod.com/2018/11/27/2018-11-27%20HNSW%20%E4%BB%8B%E7%BB%8D/#more
      学习要求：
                1.近似最近邻算法出现的意义是什么？
                       对于传统的knn算法，我们需要根据输入值去训练集中求出k个最近邻，通过多数表决的方式决定其类别。
                       在机器学习领域，语义检索，图像识别，推荐系统等方向常涉及到的一个问题是：给定一个向量X=[x1,x2,x3...xn]，
                       需要从海量的向量库中找到最相似的前K个向量。通常这些向量的维度很高，对于在线服务，用传统的方法查找是非常耗时的，
                       容易使得时延上成为瓶颈，因此业界通用的方式就是将最相似的查找转换成Ann问题。
                2.怎么样衡量其好坏?
                       衡量Ann算法好坏的一个依据是召回率，也就是通过Ann算法返回的k个结果与通过暴力查找的k的结果进行比较，如果完全一致，
                       则说明Ann算法有效。因为它节省了搜索时间效果却依然有效。
                3.有哪些ann算法？
                        目前的Ann算法有基于图的，基于树的，基于哈希等。

####  1.3 KD树,近似KD树
       简单了解就好~
       KD树是一种二叉树数据结构，可以用来进行高效的KNN计算。
       参考资料：
                https://www.joinquant.com/view/community/detail/dd60bd4e89761b916fe36dc4d14bb272
                https://zhuanlan.zhihu.com/p/23966698

## 二.SVM与集成模型
## 三.无监督模型与序列模型
## 四.深度学习
## 五.推荐系统与在线学习
## 六.贝叶斯模型
## 七.增强学习与其他前沿主题