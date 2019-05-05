#-*- coding:utf-8 -*- 
import json

# import pdb
class Config:

	conf = None


	def __init__(self,config_file='./config.json'):
		self.conf = json.load(open(config_file))

	def all(self):
		confs = []
		# pdb.set_trace()
		for k in self.conf:
			confs.append(self.ConfigEntry(self.conf[k]))
		return confs

	'''
		"name": "renfa", 	哪个bank
		"width":160,		图像宽度	
		"height":70,		图像高度
		"number":4,			识别码长度（几个字符）
		"charset": "0aA",	对应的字符集（比如是0:0-9,a:a-z,A:A-Z)
		"img_binary_threshold":240 	二值化时候的阈值，大部分bank不需要定义，只对少数bank需要
		"img_remove_area"	图像去燥的时候，要去掉的图像噪点的大小
	'''
	class ConfigEntry:
		width = 0
		height = 0
		charset = ""
		name = ""
		number = 0
		img_binary_threshold = -1
		img_remove_area = -1
		mask_pad = 0
		inverse_color = False

		# 验证码中的字符, 就不用汉字了
		_number = ['0','1','2','3','4','5','6','7','8','9']
		alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

		def charset(self,flag):
			if flag=='0a': return self._number + self.alphabet
			if flag=='0aA': return self._number + self.alphabet + self.ALPHABET
			if flag=='0A': return self._number + self.ALPHABET
			if flag=='aA': return self.alphabet + self.ALPHABET
			if flag=='a': return self.alphabet
			if flag=='0': return self._number
			return []

		def __init__(self,conf):
			self.name = conf['name']
			self.width = int(conf.get('width','0'))
			self.height = int(conf.get('height','0'))
			self.charset = self.charset(conf.get('charset',''))
			self.number = int(conf.get('number','0'))
			self.img_binary_threshold = int(conf.get('img_binary_threshold','-1')) #允许不配置
			self.img_remove_area = int(conf.get('img_remove_area','-1')) #允许不配置
			self.mask_pad = int(conf.get('mask_pad','0')) #允许不配置
			self.inverse_color = conf.get('inverse_color',False)#黑白两色需要反转
			

			if self.width == 0: raise ValueError("配置错误：宽度设置错误"+str(conf))
			if self.height == 0: raise ValueError("配置错误：高度设置错误"+str(conf))
			if len(self.charset) == 0: raise ValueError("配置错误：字符集设置错误"+str(conf))
			if self.number == 0: raise ValueError("配置错误：字符数量设置错误"+str(conf))

		def __str__(self):

			return 	"name:" + self.name + ",\
					width:"+ str(self.width) + ",\
	        		height:" + str(self.height) + ",\
	        		charset:"+ str(self.charset) + ",\
	        		number:" + str(self.number) + ",\
	        		img_binary_threshold:" + str(self.img_binary_threshold) + ",\
	        		mask_pad:" + str(self.mask_pad) + ",\
	        		img_remove_area:" + str(self.img_remove_area)+ ",\
	        		inverse_color:" + str(self.inverse_color)

	#获得某个bank的配置	
	def get(self,bank):
		bank_conf = self.conf.get(bank,None)
		if bank_conf == None: raise ValueError('Invalid Bank:'+bank)
		return self.ConfigEntry(bank_conf)

if __name__ == '__main__':
	config = Config()
	print (config.get('jianhang'))
	print (config.get('renfa'))
	print (config.get('bsb'))