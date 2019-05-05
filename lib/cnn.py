# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
                
                定义一个神经网络，标准的CNN

    主要做了以下操作：
		
	

    author:
        piginzoo
    date:
        2018/2
#------------------------------------------------------------------------------------------
'''  
from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import logging as logger
from keras import utils 
from keras.models import load_model


class CNN:

	_dropout_rate = 0.25
	_num_symbol = 0
	_charset_num = 0

	def __init__(self, conf):
		self._num_symbol = conf.number	
		self._charset_num = len(conf.charset)


	#此函数很重要。别看短。
	#他是用来告诉tensorflow，你预测出来的和我给的label是否一致，
	#tf预测出来的是一个杂乱无章的180维向量，而label是一个onehot的，
	#例如：预测是[0.02,0.031,..,0.35,..,0.12]，label是[0,0..,1,..,0]
	def custom_accuracy(self,y_true, y_pred):
	    predict = tf.reshape(y_pred, [-1, self._num_symbol, self._charset_num])
	    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
	    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, self._num_symbol, self._charset_num]), 2)	    
	    correct_pred = tf.equal(max_idx_p, max_idx_l)
	    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
	    return tf.reduce_mean(tf.cast(_result, tf.float32))

	#input_shape，主要是确认
	def create_model(self,input_shape,num_classes):
		# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
		model = Sequential()
		# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
		# 卷积核的窗口选用3*3像素窗口
		model.add(Conv2D(32,(3,3),activation='relu',strides=1,input_shape=input_shape))
		# 池化层是2*2像素的
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# 对于池化层的输出，采用_dropout_rate概率的Dropout
		model.add(Dropout(self._dropout_rate))

		# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
		# 卷积核的窗口选用3*3像素窗口
		model.add(Conv2D(32,(3,3),activation='relu',strides=1,input_shape=input_shape))
		# 池化层是2*2像素的
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# 对于池化层的输出，采用_dropout_rate概率的Dropout
		model.add(Dropout(self._dropout_rate))

		# 展平所有像素，比如[36*100] -> [3600]
		model.add(Flatten())
		# 对所有像素使用全连接层，输出为128，激活函数选用relu
		model.add(Dense(1024, activation='relu'))
		# 对输入采用0.5概率的Dropout
		model.add(Dropout(self._dropout_rate))


		# 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
		# 这个注释掉了，categorical_crossentropy适合softmax，多分类选1个，不适合我们的场景
		#model.compile(loss=keras.metrics.categorical_crossentropy,
		#              optimizer=keras.optimizers.Adadelta(),
		#              metrics=['accuracy'])
		#改成binary_crossentropy，用于多分类选多个的场景
		#但是之前要加上一个sigmod层，参见例子：https://keras.io/getting-started/sequential-model-guide/#training
		model.add(Dense(num_classes, activation='sigmoid'))

		model.compile( 
				optimizer=keras.optimizers.Adadelta(),
	            loss='binary_crossentropy',
	            metrics=['accuracy',self.custom_accuracy])

		return model 

	def load_model(self,model_path):
		model = keras.models.load_model(model_path,
			custom_objects={'custom_accuracy': self.custom_accuracy})
		logger.info("成功加载模型[%s]",model_path)
		
		return model

if __name__ == '__main__':
	global letters
	letters = "123"
	import numpy as np
	a_true = np.array([[0,1,0],[0,1,0]])
	b_true = np.array([[0.5,1.2,0.3],[0.51,1.21,0.13]])
	print (custom_accuracy(a_true,b_true))#should be true
	a_true = np.array([[0,1,0]])
	b_true = np.array([[1.5,0.2,0.3]])
	print (custom_accuracy(a_true,b_true))#should be false
	