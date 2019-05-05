# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
				
		解析标签数据

	主要做了以下操作：
		由于文件名就是标签，所以这个标签就是解析文件名
	author:
		piginzoo
	date:
		2018/2
#------------------------------------------------------------------------------------------
'''
import numpy as np
import logging as logger
from lib import log
from keras.utils import np_utils 

#入参是一个文件名的字符串，如 uv24c
#输出是一个62x5的一个5个one-hot向量组成的向量
def label2vector(file_name,conf):
	length = len(file_name)

	#这个是为了华星提供的农行数据，要做一下特殊处理
	#zzwk_6ccb06b59a09aa982bfac14b657a5b8f.jpg
	#一般的图片都是zzwk.jpg这种形式
	if length> conf.number:
		logger.debug("传入的标签[%s]长度为[%d]，固定长度[%d]",file_name,length,conf.number)
		file_name = file_name[0:conf.number]

	logger.debug("识别的文件对应标签为%s",file_name)

	char_index=[]
	for c in file_name:
		try:
			i = conf.charset.index(c)
			char_index.append(i)
		except ValueError as e:
			raise Exception(c+" is not in charset,Filename:"+file_name)
		
	one_hots = np_utils.to_categorical(char_index,len(conf.charset))

	if length< conf.number:
		for i in range(conf.number - length):
			one_hots = np.vstack(
				(one_hots,
				np.zeros(len(conf.charset))))

	# print( one_hots)
	result = np.hstack(one_hots)
	# print( result)
	logger.debug( "标签数据:%r",result.shape)
	return result


#入参是一个62x5的一个5个one-hot向量组成的向量,注意，是一维的
#输出参是一个文件名的字符串，如 uv24c
def vector2label(vector,conf):

	length = len(conf.charset)

	logger.debug("输入的hot向量是%r",vector)
	logger.debug("输入向量shape是%r",vector.shape)
	# assert vector.shape == (conf.number*length,)

	#把一维62x5=180的向量，reshape成一个二维的向量
	devided_vectors = np.reshape(vector,(-1,length))

	result = ""
	#Keras的to_categorical反向方法，就把1-hot变成一个数，就是字符串里的位置
	for one_hot in devided_vectors:
		logger.debug("当前的one-hot向量：%r",one_hot)
		#全0向量就忽略
		if np.count_nonzero(one_hot)==0:
			logger.debug("此向量为全0，解析为空！！！")
			continue
			
		index = np.argmax(one_hot)	
		letter = conf.charset[index]
		logger.debug("解析字符的序号：%d,解析的字符为：%s",index,letter)
		result+= letter

	logger.debug( "解析出来的结果为:%r",result)
	return result


if __name__ == '__main__':
	
	a = label2vector("trai5")		    
	b = label2vector("trai5555")
	c = label2vector("tra5")
	d = label2vector("tra")		    		    
	print( a)
	print( b)
	print( c)
	print( d)

	print( vector2label(a))
	print( vector2label(b))
	print( vector2label(c))
	print( vector2label(d))

	f = np.array([2.27676210e-05, 1.47138140e-03, 2.10353086e-04, 6.11234969e-03,
        2.60898820e-03, 8.84016603e-03, 5.93697906e-01, 2.26049320e-04,
        1.70289129e-01, 7.46237068e-03, 9.22823325e-03, 3.46448243e-04,
        6.10056275e-04, 1.16427680e-02, 5.61858434e-03, 1.01848654e-02,
        2.52747093e-04, 1.50831360e-02, 8.88945724e-05, 7.16042996e-04,
        1.09516352e-03, 5.41963323e-04, 1.46554587e-02, 1.07519729e-02,
        3.71959788e-04, 8.43562651e-03, 1.05409988e-03, 4.03449638e-04,
        1.25252089e-04, 8.69100913e-03, 4.00895663e-02, 1.87198515e-03,
        8.79501924e-03, 1.46250022e-04, 2.56469881e-04, 1.25844454e-04,
        4.48116713e-04, 2.33228020e-02, 2.81759392e-04, 5.95326675e-03,
        9.48002096e-03, 9.97657888e-03, 9.10525327e-04, 6.13110373e-04,
        1.04311411e-03, 1.98717811e-04, 9.03097689e-01, 1.70895262e-04,
        1.80205703e-03, 9.09789465e-04, 1.61770552e-01, 1.21310656e-03,
        5.97630918e-04, 4.25095262e-04, 1.81629584e-05, 3.66338383e-04,
        6.33585674e-04, 9.88863991e-04, 1.14032604e-01, 3.77260149e-02,
        1.52453285e-04, 4.97384579e-04, 1.12607400e-03, 1.73787586e-02,
        2.80013337e-04, 4.02341643e-03, 2.58398103e-03, 2.57012225e-03,
        7.97528587e-03, 3.32528874e-02, 2.19268096e-03, 5.31593442e-01,
        3.67596367e-04, 2.66813964e-04, 7.35380361e-03, 1.01331470e-03,
        3.95892896e-02, 1.29315793e-03, 6.56478712e-03, 1.51120737e-04,
        2.06782985e-02, 8.14618485e-04, 5.47068799e-03, 3.94553026e-05,
        5.50394226e-03, 9.96713996e-01, 1.34009104e-02, 8.88924624e-05,
        4.90950944e-04, 3.67864908e-04, 5.04667521e-04, 3.92648275e-04,
        6.69848581e-04, 1.78790092e-03, 8.46620928e-03, 6.10376708e-03,
        3.36909317e-03, 1.92471838e-03, 3.14407237e-02, 1.32358691e-05,
        1.02250793e-04, 1.27569856e-05, 7.56577700e-02, 9.90313129e-05,
        1.02341932e-03, 1.23595601e-04, 2.16793414e-05, 1.13001071e-04,
        3.67587811e-04, 4.33387235e-04, 7.17533112e-05, 3.58651479e-04,
        6.45336183e-03, 8.88123934e-04, 1.70656536e-02, 5.77677702e-06,
        1.52134849e-03, 1.59175333e-03, 1.01942876e-02, 4.92385996e-04,
        4.98368870e-04, 2.49108858e-03, 6.99159550e-03, 1.21457539e-04,
        2.17163091e-04, 2.40839273e-03, 1.14973605e-04, 5.34953608e-04,
        8.98304433e-02, 3.49930255e-03, 2.38535866e-01, 1.12532064e-01,
        5.20306465e-04, 9.38495807e-03, 5.26941791e-02, 3.09547968e-03,
        2.88778479e-04, 3.06648028e-04, 6.65633082e-02, 7.39186583e-03,
        8.51062059e-01, 2.16002017e-02, 1.81221461e-03, 1.62647327e-03,
        4.67675636e-05, 4.35031839e-02, 1.24836195e-04, 1.59627656e-04,
        5.96818235e-03, 2.22883057e-02, 4.71321307e-03, 2.96168327e-02,
        1.96822472e-02, 1.17240031e-03, 8.59424123e-04, 2.73402169e-04,
        9.45712731e-04, 1.26615632e-05, 1.36439758e-03, 7.50226855e-01,
        1.43466284e-04, 7.73136489e-05, 1.56502356e-03, 1.14007469e-03,
        4.02029417e-02, 8.66075454e-04, 5.82826743e-03, 3.13856639e-04,
        6.81801117e-04, 1.24372200e-05, 5.78947802e-05, 2.57605780e-03,
        2.51208003e-05, 2.68695742e-01, 4.39417374e-04, 2.46306658e-02,
        2.51952652e-03, 2.51299283e-03, 5.82860708e-02, 1.98105886e-03])	
	print( vector2label(f))