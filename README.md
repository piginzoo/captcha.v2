# 使用CNN识别验证码

## 使用方法
注：
>    训练数据存放在data/validate/<bank>
>    验证数据存放在data/validate/<bank>
>    训练和识别时，不用敲目录名，会根据bank自动识别

训练：
    py train.py --bank renfa|jianhang|nonghang 
    如需帮助请输入：py train.py -h

识别一张图片：
    py predict.py --bank renfa --image abcd.jpg

随机识别100张：
    py predict.py --bank renfa --test 100

生成一张图片：（到out目录，用于观察图像处理情况）
    py lib/image_process.py  --bank renfa --image abcd.jpg
    

## 文件说明

- /     根目录下是几个入口文件
- lib   相关的python文件
- model 存放着训练完的文件
- log   训练的用的日志文件
- web   Restful的JSon API Web服务
- docker docker镜像和容器以及部署相关的脚本
- data是训练和验证数据
    - data/validate/<BankId> 验证数据，各个子目录是每个bank的
    - data/train/<BankId>    训练数据，各个子目录是每个bank的
    > renfa 
    > jianhang
    > nonghang
- bin   常用的一些批处理  

## 配置文件config.json

为了方便配置，我们做了一个json的配置文件，方便加入新的bank
>'''
>    "name": "renfa",    哪个bank
>    "width":160,        图像宽度    
>    "height":70,        图像高度
>    "number":4,         识别码长度（几个字符）
>    "charset": "0aA",   对应的字符集（比如是0:0-9,a:a-z,A:A-Z)
>    "img_binary_threshold":240  二值化时候的阈值，大部分bank不需要定义，只对少数bank需要
>    "img_remove_area"   图像去燥的时候，要去掉的图像噪点的大小
>''' 
    
## 关于部署

在服务器上采用了docker方式部署，只要运行 docker/build_run.sh 就可以build出来一个image，并生成一个容器。
    目录约定：代码一定要部署在/home/captcha目录下，模型放置在/home/captcha/model内。
    容器中依赖外部两个内容：
    1、日志：本考虑用run -v方式mount一个日志目录、文件上去，但是过于麻烦，现在采用直接输出到标准控制台上，需要查看的时候，使用docker logs -f/--tail N方式查看
    2、模型文件：使用run -v方式挂架/home/captcha/model目录到容器内的/home/captcha/model目录，里面存放着生产用的模型文件

## 批次文件
    bin/image.sh    用于测试图片处理情况（如二值化，去噪）
    bin/predict.sh  用于预测单张图片或者预测批量图片
    bin/stress.sh   用于做压力测试
    bin/train.sh    用于做训练
    docker/build_run.sh     用于构建一个生产镜像(image)并创建运行态的容器(container)
    docker/deploy.sh        用于重新部署代码
    web/startup.sh  用于启动Gunicorn Web服务器

## 环境依赖
    opencv      3.4.0   https://opencv.org/
    tensorflow  1.5     https://tensorflow.org/
    skimage     0.13.1  http://scikit-image.org/
    keras       2.1.3   https://keras.io/
    flask       0.12.2  http://flask.pocoo.org/

## 设计思路

### 问题提出：

过去采用切割的方法，使用传统的图像识别算法进行验证码识别，效果不是特别好，这次采用深度学习的方法来进行识别，并没有分割，而是直接一体扔给CNN去识别。

### 设计思路：

本打算还是采用切片的方式，即先对图片进行分割，然后进行单独的识别，交流后，他们之前已经进行过大量的尝试，拆分图片效果不好，所以果断放弃，直接采用一体识别，即把5个字符一起识别的方式。另外，图片进行了预处理，灰度化->二值化->去噪点因为担心识别效果，还是去网上做了一番调研，发现了一些解决方案，都可以支持：

参考：

- 1. <http://blog.csdn.net/Gavin__Zhou/article/details/52071797>

这个貌似不错，有github代码，考了粘连：<https://github.com/iamGavinZhou/py-captcha-breaking> <zxinbloom@gmail.com>

不过，这个哥们后来又撸了一个固定长度的，<http://blog.csdn.net/gavin__zhou/article/details/69049663>
我都得看看。

- 2. <https://www.jianshu.com/p/86489f1afd36>
这个是端到端的做法，不切割了，

(1). 把OCR的问题当做一个多标签学习的问题。4个数字组成的验证码就相当于有4个标签的图片识别问题（这里的标签还是有序的），用CNN来解决。

(2). 把OCR的问题当做一个语音识别的问题，语音识别是把连续的音频转化为文本，验证码识别就是把连续的图片转化为文本，用    CNN+LSTM+CTC来解决。作者玩的是第一种方法，我们可以试试也。

- 3. <https://github.com/moxiegushi/zhihu>
     从知乎上爬取，测试的，有点意思，先记下

- 4. <http://www.bubuko.com/infodetail-1877150.html>
     中文的，留着看，

- 5.<http://blog.csdn.net/yinchuandong2/article/details/40340735>
    这篇虽然老点，但是有cfs的切割，图像切割的办法，可以参考

- 6. <http://www.doc88.com/p-9874925209090.html>
       这篇是一个传统方法的，但是用到很多方法，可以参考，但是仅能在线，无法下载pdf

- 7.<https://www.baitiwu.com/t/topic/16027>
        最后看的这篇，挺神奇，思路简单粗暴啊，直接暴力破解，狠喜欢
- 8. <https://www.cnblogs.com/SeekHit/p/7806953.html>
        受不了了，这篇更暴力，要跑死的节奏啊，总共 62^4 种类型，采用 4 个 one-hot 编码分别表示 4 个字符取值。噢，晕了，其实和上一篇是同样的思路

- 9.<http://blog.topspeedsnail.com/archives/10858>
        再来一篇，撸起来，这个很赞，我喜欢的思路，又是不分割的，哈哈
        这哥们的代码貌似靠谱，但是有人反馈resize后就有问题了。我试试。

- 10.<https://github.com/luyishisi/Anti-Anti-Spider/tree/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81/tensorflow_cnn>
        
其中，9、10的思路最接近，于是果断采用，开始编写代码。

### 详细设计

设计思路比较清晰，就是把整个5位的验证码图像，作为一个整体喂给CNN网络，整个是输入的x，而对应的y是一个36(小写字符+数字）x识别字符数（5）=180的一个One-hot向量，遇到缺少的字符，那36位就全部置零。
样子如下：

>[0,0,0….1,….,0,0, | 0,0,0.1,….,0,0,...|,0,0,0….1,…...,0, |,0,0,0….1,…..,0,0] 

灌入后，在CNN网络的最后一层，要使用sigmod函数，而没有使用softmax，而损失函数依然是交叉熵？为什么要这么一个设计呢？

>model.add(Dense(num_classes, activation='sigmoid'))

>model.compile( loss='binary_crossentropy',

因为softmax往往最后得到的是多分类中1/N（我们这里就是1/180）的结果，但是我们的这个实际上要得到的是5/N（我们这里是5/180）的分类问题，也就是多个分类都满足要求的结果。这种情况下，就必须要采用binary_crossentropy的损失函数。这个损失函数，需要我们在喂给这个损失函数之前的张量要做一次sigmod变换。

最后在训练的评价函数上，通过一个自定义的损失函数，来对比标签的5/180的结果，和我的概率分布的结果，我的概率分布也是180维度的向量，但是是每个维度都有值，我需要先36个分割一个，分割出5个36维度的向量，然后找出36维度里面最高的那个维度，置为1，其他35维度置为0，形成一个36维度的one-hot向量，然后拼接成180维度的5/180，和标签5/180做对比，一致就认为ok，不一致就认为不ok。所以，这个过程，需要自定义一个metrics函数。

### 遇到问题

- 首先是图像预处理，采用opencv，做二值化处理；然后使用skimage对图像做了噪点去除；但是，对干扰线实在是去不掉，和也进行过交流，因为jianhang的图片的干扰线和图像的颜色很相近，因此很难去掉，所以最后作罢
- 9、10中使用的是直接的tensorflow，我没有采用他们的代码，而是使用keras，一个是我想对熟悉一些，另外一个原因是keras的代码更简洁和容易维护 
- 目前偶尔遇到一些数据是4位的，标准是5位的，对于缺位的处理是在字符阶段用"_"代替，在 
- 这里我有一个非常困惑的地方：

因为我理解的一般在交叉熵的计算上，都是两个概率分布的计算，既然是概率分布，一定是归一化得，而我们的样本，以及我们的CNN计算结果，显然都不是归一化的。我们的样本是180维度上有5个1，我们的结果上更是夸张，结果sum到一起连5都不是，而是约等于7。
    
>[DEBUG] 模型预测出来的结果是：

```
array([[2.27676210e-05, 1.47138140e-03, 2.10353086e-04, 6.11234969e-03,
        2.60898820e-03, 8.84016603e-03, 5.93697906e-01, 2.26049320e-04,
        1.70289129e-01, 7.46237068e-03, 9.22823325e-03, 3.46448243e-04,
        6.10056275e-04, 1.16427680e-02, 5.61858434e-03, 1.01848654e-02,
        2.52747093e-04, 1.50831360e-02, 8.88945724e-05, 7.16042996e-04,
[DEBUG] 预测结果shape是：(1, 180)
[DEBUG] 预测结果Sum为:6.8608074
```

- 在自定义metrics函数的编制过程中，遇到一些问题，因为这个函数实在编译阶段要构件好，而在运行态才执行的，所以，不能直接使用诸如ndarray的一些函数来做one-hot转化，比如to_categorical函数，否则，在编译阶段就会报错。后来全部采用tf的函数，如tf.shape/tf.argmax/tf.reduce_mean，保证他只是操作这些张量变量，保证编译器通过。
- 另外metrics的调试极其费劲，无法看到结果，尝试了tf.Print来打印，但是没凑效，不知道什么原因，下个阶段还是要继续深入