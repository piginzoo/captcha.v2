# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
				
				此文件用来加载一张图片，并且进行预处理

	主要做了以下操作：
		1.通过阈值调整去掉左右留白
		2.通过阈值调整去掉干扰线
		3.把突变成黑底的，白字的，100x36的，干净输入
	操作:
		二值化，顶格，字符分割等
	依赖库及版本:
		openCV > 2.4.x, skimage >= 0.9.x
	author:
		piginzoo
	date:
		2018/3
#------------------------------------------------------------------------------------------
'''
import sys,os,io
#这个是为了在单独运行image_process.py的时候（因为他在lib里面），处理路径引用问题，把他所在的上层目录放到路径中
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 得到上上一层路径
sys.path.append(base_dir)

import math,cv2,h5py,codecs,os,random,argparse,numpy as np,logging as logger,time as t
from skimage.measure import regionprops
from skimage import morphology,color,data,filters,io
from skimage.morphology import label,disk
from lib import log,crash_debug,label as label_process
from lib.config import Config
from keras import backend as K

def dump_array_detail(arr):
	logger.debug("Dump the array:")
	for row in arr:
		logger.debug(row)

def output_img(name,img,debug=False):
	if debug == False: return

	file_name = os.path.basename(name)
	#调试用，不用打开了，否则，20000张图片，会撑爆硬盘的
	cv2.imwrite("out/"+file_name+'.jpg',img)

#这个函数专门用于特殊处理“邮储bank的”验证码，
def process_psbc_red(img,debug):
	#把RGB通道转成GBR通道，靠，skimage和opencv的读取的rgb通道是不一样的：
	#skimage：RGB，opencv是：BGR，所以要转一下
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	#利用inRange函数来保留红色，红色H值范围是0-8和160-180
	mask1 = cv2.inRange(hsv,(160,50,50),(180,255,255))
	mask2 = cv2.inRange(hsv,(0,50,50),(8,255,255))
	mask = cv2.bitwise_or(mask1, mask2)
	res = cv2.bitwise_and(img,img,mask=mask)
	res = res + 255#得到是黑背景，所以要装成白背景，其他图片处理都是白背景的，保持一致
	output_img("去红色",res,debug)
	return res


#加载一个图像，返回对应的张量[channel,row,conf.height,conf.width]
#入参是灰度图像，为什么是灰度
#出参是图像数据(ndarry的)，和图像的标签（文件名，这里就是'aaaaa'）
def preprocess_image(img_rgb_data,file_name,conf,debug=False):

	if img_rgb_data is None:
		logger.error("图像数据为空")
		return None

	output_img(file_name+"原图",img_rgb_data,debug)

	#实在代码是太ugly了，由于邮储太特殊，只认红色的，所以hardcode一下吧
	if conf.name=="psbc":
		logger.debug("特殊处理邮储bank图片，只识别红色字体")
		img_rgb_data = process_psbc_red(img_rgb_data,debug)

	#这个方法不靠谱
	#image_gray = color.rgb2gray(img)
	image_gray = cv2.cvtColor(img_rgb_data,cv2.COLOR_BGR2GRAY)
	logger.debug( "灰度图像的shape:%r",image_gray.shape)
	output_img(file_name+"灰度图",image_gray,debug)

	#1.根据阈值进行分割
	if conf.img_binary_threshold == -1:
		thresh = filters.threshold_otsu(image_gray) #自动探测一个阈值
	else:
		logger.debug("使用自定义的阈值来二值化：%d",conf.img_binary_threshold)
		thresh = conf.img_binary_threshold #从配置中读取阈值

	t =(image_gray <= thresh)

	#如果配置中需要黑白颠倒，就把颜色最后颠倒一下
	if conf.inverse_color: t = ~t 

	#2.删除掉小的区块，面积是minsize=25,25是个拍脑袋的经验值，可以在配置文件中修改
	output_img(file_name+"二值化",t*255,debug)#*255是为了变成白色的用于显示
	min_size = 25	    #默认要删除掉的噪点大小
	if conf.img_remove_area !=-1: min_size = conf.img_remove_area
	t = morphology.remove_small_objects(t,min_size=min_size,connectivity=1)
	output_img(file_name+"删除小块",t*255,debug)#*255是为了变成白色的用于显示

	t = t.astype(np.float32)

	#3.有些图像带黑框，用一个框的mask去掉
	pad = conf.mask_pad
	if pad != 0:
		logger.debug("去掉黑框，pad=%d",conf.mask_pad)
		mask = np.zeros(t.shape, dtype = "uint8")
		#画一个需要的地方是1，不需要的地方是0的矩形框，框的厚度是pad
		cv2.rectangle(mask,(pad,pad), (t.shape[1]-2*pad, t.shape[0]-2*pad), 1, -1)
		# output_img(file_name+"黑框mask",mask,debug)#*255是为了变成白色的用于显示
		t = cv2.bitwise_and(t,t,mask=mask)
		output_img(file_name+"去掉黑框",t*255,debug)#*255是为了变成白色的用于显示

	#防止有的图像不是规定的conf.widthxheight，有必要都规范化一下
	if t.shape[1] < conf.width:
		logger.debug("图像宽度%d不够，补齐到%d", t.shape[1] , conf.width)
		t = np.concatenate((t, np.zeros((t.shape[0], conf.width - t.shape[1]), dtype='uint8')), axis=1)
	if t.shape[0] < conf.height:
		logger.debug("图像高度%d不够，补齐到%d", t.shape[0] , conf.height)
		t = np.concatenate((t, np.zeros((conf.height - t.shape[0], t.shape[1]), dtype='uint8')), axis=0)
	output_img(file_name+"调整大小",t*255,debug)#*255是为了变成白色的用于显示

	#如果图像大了，要剪切成规定大小（经过上面的处理，宽和高至少都是比规定尺寸大了）
	if t.shape[1] > conf.width or t.shape[0] > conf.height:
		logger.debug("图像从%r Resize到 %r",t.shape,(conf.width,conf.height))
		t = cv2.resize(t,(conf.width,conf.height))
		output_img(file_name+"Resize",t*255,debug)#*255是为了变成白色的用于显示

	#同时把原来白色的地方变成1，黑的地方为0，这个是训练要求，有数字显示的地方是1，没有的是0
	t = t > 0


	#变成一个四维numpy.ndarray,为了和[图片个数？,image channel,conf.height,conf.width]
	#tensorflow和thenano的格式要求不一样，图像通道的位置反着，烦人，做一下处理
	if K.image_data_format() == 'channels_first':
		I = t.astype(np.float32).reshape((1, 1, conf.height, conf.width)) 
	else:
	 	I = t.astype(np.float32).reshape((1, conf.height, conf.width, 1)) 

	logger.debug( "图像数据:%r",I.shape)
	return I#返回一个图像的矩阵



#"data/"
#返回是个4维度的numpy.ndarray[20000,1,100,36]，
#20000是图片数
#1是图像通道，灰度的
#100x36图像
def load_all_image_by_dir(path,conf):

	start = t.time()
	data = []
	label = []
	file_list = os.listdir(path)
	debug_count = 0

	for file in file_list:
		# debug_count+=1
		# if debug_count>100: break #调试用，省的加载那么长时间图片

		if (file.find(".jpg")==-1 and 
			file.find(".jpeg")==-1 and
			file.find(".png")==-1 and 
			file.find(".bmp")==-1 and
			file.find(".gif")==-1): continue

		file_name = file.split(".")[0]
		img_data = cv2.imread(path+file)
		one_img = preprocess_image(img_data,file_name,conf)
		if (one_img is None): 
			logger.debug("处理图像失败%s，处理下一张",file)
			continue
		
		try:
			v = label_process.label2vector(file_name,conf)
			label.append(v)
		except Exception as e:
			import traceback
			traceback.print_exc()
			logger.error("忽略此文件%s,原因：%r",file_name,str(e))
			continue

		data.append(one_img)

	#把数组堆叠到一起形成一个[20000,100,36,1]的张量	
	# print(len(data))
	image_data = np.vstack(data)
	label_data = np.vstack(label)

	logger.info("images data loaded:%r",image_data.shape)
	end = t.time()
	logger.info("加载图像使用了%d秒...",(end-start))

	return image_data,label_data

if __name__ == '__main__':
	log.logger_init()
    #解决stdout显示中文的问题
    #python3.x代码
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

	#建行的验证码尺寸
	#测试整个data目录下的图片处理
	#data,label = load_all_image_by_dir("temp/",IMG_WIDTH,IMG_HEIGHT,letter,4)
	#logger.debug("加载了图像：%r,标签：%r",data.shape,label.shape)
	#测试单张图片，可以看out输出
	#preprocess_image("data/validate/renfa/5yhg.jpg",IMG_WIDTH,IMG_HEIGHT)  	
	parser = argparse.ArgumentParser()
	parser.add_argument("--bank",help="--bank bank名称(jianhang|renfa|nonghang)")
	parser.add_argument("--image",help="--image 图片名称(位于data/validate/<bank代号>/)")

	args = parser.parse_args()
	config = Config()

	if args.bank == None:
	    parser.print_help()
	    exit()
	if args.image == None:
	    parser.print_help()
	    exit()

	conf = config.get(args.bank)

	import sys
	sys.path.append("..")

	if args.image != None:
		image_path = "data/train/"+args.bank+"/"+args.image
		try:
			image_data = cv2.imread(image_path)
		except IOError as e:
			logger.error("无法加载文件：%s",image_path)
		preprocess_image(image_data,args.image,conf,debug=True)