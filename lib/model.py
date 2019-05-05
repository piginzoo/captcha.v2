#-*- coding:utf-8 -*-  
#__author__ = 'piginzoo'
#__date__ = '2018/3/10'
from __future__ import print_function
import logging as logger,os
from lib import label,image_process,log,crash_debug
from lib.cnn import CNN
from lib.config import Config
from keras import backend as K
# import pdb 
class Model:

	__model_repository = {}


	def __init__(self,base_dir="model/"):
		self.base_dir = base_dir

	def model_path(self,conf):
		return self.base_dir+"captcha."+conf.name+".h5"

	def checkpoint_path(self,conf):
		return self.base_dir+"checkpoint."+conf.name+".hdf5"

	def load_all(self,confs):
		for c in confs:
			self.load(c)

	def load(self,conf):
		# pdb.set_trace()
		if self.__model_repository.get(conf.name,None): 
			logger.debug("模型 [%s] 已经缓存",conf.name)
			return self.__model_repository[conf.name]

		_model_path = self.model_path(conf)
		if not os.path.exists(_model_path):
			logger.error("模型文件%s不存在",_model_path)
			return None
		logger.debug("模型 [%s] 缓存不存在，需要加载",conf.name)	
		cnn = CNN(conf)
		model = cnn.load_model(_model_path)
		self.__model_repository[conf.name] = model
		return model

    #train的时候，需要特殊处理，直接从checkpoint中加载，不需要加载结果模型
    #原因是，有可能上次train的过程中crash掉，所以索性直接从checkpont里加载    
	def load4train(self,conf):
		cnn = CNN(conf)

		#如果checkpoint文件存在，就加载之
		_checkpoint_path = self.checkpoint_path(conf)
		if os.path.exists(_checkpoint_path):
			return cnn.load_model(_checkpoint_path)        	
		logger.info("模型checkpoint文件[%s]不存在" % _checkpoint_path)

		#如果checkpoint文件不存在，再尝试加载model文件
		_model_path = self.model_path(conf)
		if os.path.exists(_model_path):
			return cnn.load_model(_model_path)        	
		logger.info("模型文件[%s]不存在" % _model_path)


		input_shape = None 
		if K.image_data_format() == 'channels_first': 
			input_shape = (1,conf.height,conf.width)
		else:
			input_shape = (conf.height,conf.width,1)   #这个shape不包含第一个维度，也就是图片数量

			 
		char_dimention = conf.number*len(conf.charset)    
		model = cnn.create_model(input_shape,char_dimention)
		logger.info("找不到存在的模型或者checkpoint，创建了新模型")
		return model