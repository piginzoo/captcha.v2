# coding: utf-8
'''
#-----------------------------------------------------------------------------------------                
                训练神经网络
    author:
        piginzoo
    date:
        2018/3
#------------------------------------------------------------------------------------------
'''  
import io,sys,cv2,h5py,codecs,os,random,argparse,numpy as np,logging as logger
from skimage import color

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard  
from keras.callbacks import ModelCheckpoint

from lib import label,image_process,log,crash_debug
from lib.cnn import CNN
from lib.model import Model
from lib.config import Config

__model = Model()

# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
def train(data_dir,conf,epochs=2,batch_size=100):
    image_width = conf.width
    image_height = conf.height
    _letters = conf.charset#字符集合
    _num_symbol = conf.number #识别字符的个数

    model = __model.load4train(conf)
    model_path = __model.model_path(conf)

    # 训练集
    x_train,y_train = image_process.load_all_image_by_dir(data_dir,conf)

    #训练期间保存checkpoint，防止整个crash
    #checkpoint_path = "num-model-{epoch:02d}-{val_acc:.2f}.hdf5" #这个是对save_best_only=False的文件名    
    checkpoint_path = __model.checkpoint_path(conf)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    logger.info("开始训练：")
    logger.info("\t训练样本数：%d",len(x_train))
    logger.info("\t训练总次数：%d",epochs)
    logger.info("\t每批次数量：%d",batch_size)

    # 令人兴奋的训练过程
    history = model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        verbose=1, 
        callbacks=[TensorBoard(log_dir='./log'),checkpoint],
        validation_split=0.1)#拿出10%来不参与训练，而用做中途的验证

    logger.info("训练结束！%r",history)

    model.save(model_path)
    logger.info("保存训练后的模型到：%s",model_path)

if __name__ == '__main__':

    log.logger_init()
    #解决stdout显示中文的问题
    #python3.x代码
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", help="bank名称:如'jianhang' 'renfa'")
    parser.add_argument("--epochs",default=2,type=int,help="训练多少个轮回，默认2次")    
    parser.add_argument("--batch",default=100,type=int,help="1次训练多少张图片，默认100张")    
    args = parser.parse_args()

    bank = args.bank
    if bank==None:
        parser.print_help()
        exit()

    config = Config()
    conf = config.get(bank)

    train(
        "data/train/"+conf.name+"/",
        conf,
        args.epochs,
        args.batch)