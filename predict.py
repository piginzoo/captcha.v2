#-*- coding:utf-8 -*-  
#__author__ = 'piginzoo'
#__date__ = '2018/2/1'
from __future__ import print_function

import os,sys,io,cv2,h5py,codecs,os,random,argparse,numpy as np,logging as logger
from skimage import color,io

from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard ,  ModelCheckpoint
import tensorflow as tf

from lib import label,image_process,log,crash_debug
from lib.cnn import CNN
from lib.model import Model
from lib.config import Config
'''
    做预测用
'''

weight_decay = 0.001
num_model = None

#入参是一个批量目录，出参是批量的字符串
#会判断否正确，以及整体正确率
def evaluate(image_dir,conf,model):
    
    x,y = image_process.load_all_image_by_dir(image_dir)
    e = model.evaluate(x,y,batch_size=100,verbose=0)
    logger.info(model.metrics_names)
    logger.info("评估结果,正确率：%f，损失值：%f",e[1],e[0])    

#入参是一张图片的全路径，出参数是预测的字符串
def predict(image_data,file_name,conf,model,from_web=False,graph=None):
    from threading import Thread, current_thread
    logger.debug('子进程:%s,父进程:%s,线程:%r', os.getpid(), os.getppid(),current_thread())

    logger.debug("predict()的image_data.shape:%r",image_data.shape)
    # image_gray = color.rgb2gray(image_data)
    x = image_process.preprocess_image(image_data,file_name,conf,debug=True)

    _y = None
    if graph: 
        with graph.as_default():
            logger.debug("使用存在的全局graph:%r",graph)
            _y = model.predict(x,batch_size=1,verbose=0)
    else:
        logger.debug("使用默认的graph:%r",graph)
        _y = model.predict(x,batch_size=1,verbose=0)         

    #不知道预测出来的是什么，所以，我要先看看
    logger.debug("模型预测出来的结果是：%r",_y)
    logger.debug("预测结果shape是：%r",_y.shape)
    logger.debug("预测结果Sum为:%r",np.sum(_y))

    return label.vector2label(_y,conf)

#正确率太TMD高了，搞的我都毛了，我要自己挨个试试
#这个函数没啥用，就是自己肉眼测试，看结果用的，所谓眼见为实
def predict_by_number(_dir,pred_number,conf,model):
    
    if not os.path.exists(_dir):
            logger.error("验证用数据%s不存在",_dir)
            return None

    file_list = os.listdir(_dir)
    random.shuffle(file_list)

    ok,fail = 0,0

    file_counter=0
    for file in file_list:
        file_counter+=1
        if file_counter > pred_number: break #只测试1000张

        try:
            image_data = cv2.imread(_dir+file)
        except IOError as e:
            logger.error("无法加载文件：%s",_dir+file)
            continue
        
        result = predict(image_data,file,conf,model) #识别出来的
        label = file.split("_")[0]
        #这个是nonghang数据，要做一下特殊处理
        #zzwk_6ccb06b59a09aa982bfac14b657a5b8f.jpg
        #一般的图片都是zzwk.jpg这种形式
        if len(label)> conf.number:
            logger.debug("传入的标签[%s]长度为[%d]，固定长度[%d]",label,len(label),conf.number)
            label = label[0:conf.number]        

        if result != label:
            fail+=1
            logger.error("标签%s和预测结果%s不一致!!!!!!!",label,result)
        else:
            ok+=1
            logger.info("标签%s vs 预测结果%s",label,result)

    logger.info("预测对%d vs 预测错%d",ok,fail)

'''
    测试一张图片：
        py predict.py -bank renfa --image abcd.jpg

    随机测试1000张：
        py predict.py -bank renfa --test 1000
        //注：数据存放在data/validate
'''
if __name__ == '__main__':
    log.logger_init()
    #解决stdout显示中文的问题
    #python3.x代码
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument("--bank",help="--bank 名称(jianhang|renfa|nonghang)")
    parser.add_argument("--image",help="--image 图片名称(位于data/validate/<bank代号>/)")
    parser.add_argument("--test",type=int,help="--test 需要测试的图片数量")

    args = parser.parse_args()
    config = Config()

    if args.bank == None:
        parser.print_help()
        exit()

    conf = config.get(args.bank)

    _model = Model()
    model = _model.load(conf)

    if(model == None):
        log.error("模型文件[%s]不存在", _model.model_path())

    if args.test == None and args.image == None:
        parser.print_help()
        exit()

    if args.image != None:
        image_path = "data/validate/"+args.bank+"/"+args.image
        image_data = cv2.imread(image_path)
        print("预测图片为："+image_path)
        print("预测结果为："+predict(image_data,args.image,conf,model))
        exit()

    if args.test != None:
        predict_by_number("data/validate/"+args.bank+"/",
            args.test,conf,model)