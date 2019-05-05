#-*- coding:utf-8 -*- 
from flask import Flask,jsonify,request,abort,render_template

import base64,cv2,json,sys,numpy as np
sys.path.append("..")
from lib.config import Config
from lib.model import Model
from lib import log
import predict,os,io,sys
from skimage import color,io
import logging as logger
import tensorflow as tf

log.logger_init("../log.conf")
#解决stdout显示中文的问题
#python3.x代码
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


app = Flask(__name__, static_url_path='')

from threading import Thread, current_thread
logger.debug('子进程:%s,父进程:%s,线程:%r', os.getpid(), os.getppid(),current_thread())

config = Config('../config.json')

try:
    confs = config.all()
except Exception as e:
    logger.error("配置文件解析出错：%r",e)
    exit()


model = Model("../model/")

try:
    model.load_all(confs)
except Exception as e:
    logger.error("加载Model出错：%r",e)
    exit()

#注意！一定要放到加载了模型之后
graph = tf.get_default_graph()
logger.debug("全局获得Graph:%r",graph)

#读入的buffer是个纯byte数据
def process(buffer,bank):
    logger.debug("从web读取数据len:%r",len(buffer))

    if len(buffer)==0: return False,"Image is null"

    #先给他转成ndarray(numpy的)
    data_array = np.frombuffer(buffer,dtype=np.uint8)

    #从ndarray中读取图片，有raw数据变成一个图片rgb数据
    #出来的数据，其实就是有维度了，就是原图的尺寸，如160x70
    image_rgb_data = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
    if image_rgb_data is None:
        logger.error("图像解析失败")#有可能从字节数组解析成图片失败
        return None

    logger.debug("从字节数组变成图像的shape:%r",image_rgb_data.shape)


    conf = config.get(bank)

    #后面对图像进行了规整化处理，所以这里的校验就不需要了
    #参见 image_process.preprocess_image方法
    # target_shape = (conf.height,conf.width,3)
    # if(image_rgb_data.shape!=target_shape):
    #     return False,"We need"+str(target_shape)+" image,you provide"+str(image_rgb_data.shape)

    _model = model.load(conf)
    if(_model==None):
        return False,"Model not exist"

    try:
        logger.debug("传入预测的Graph:%r",graph)
        return True,predict.predict(image_rgb_data,
                "__",conf,_model,
                from_web=True,graph=graph)
    except Exception as e:
        logger.error("预测出现错误：%r",str(e))
        #虽然catch住了，还是要把stack打出，方便调试
        import traceback
        traceback.print_exc()
        return False,str(e)

@app.route("/")
def index():
    with open("../version") as f:
        version = f.read()

    return render_template('index.html',version=version)

#base64编码的图片识别
@app.route('/captcha',methods=['POST'])  
def captcha():  

    bank = request.form.get('bank','')

    if bank=="": 
        logger.error("无法从request中获取bank值")
        return jsonify({'success':'false','reason':'bank id not provide'})

    base64_data = request.form.get('image','')

    #去掉可能传过来的“data:image/jpeg;base64,”HTML tag头部信息
    index = base64_data.find(",")
    if index!=-1: base64_data = base64_data[index+1:]


    buffer = base64.b64decode(base64_data)
    
    try:
        success,result = process(buffer,bank)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r",e)
        return jsonify({'success':'false','reason':str(e)}) 
    
    if success: 
        if result is None:
            return jsonify({'success':'false','reason':'image resolve fail'}) 
        else:
            return jsonify({'success':'true','result':result})
    else:
        return jsonify({'success':'false','result':result}) 
    
#图片的识别
@app.route('/captcha_recognise',methods=['POST'])  
def captcha_recognise():  
    bank = request.form.get('bank','')

    if bank=="": 
        logger.error("无法从request中获取bank值")
        return jsonify({'success':'false','reason':'bank id not provide'})

    data = request.files['image']

    buffer = data.read()

    success,result = process(buffer,bank)

    if success: 
        if result is None:
            return jsonify({'success':'false','reason':'image resolve fail'}) 
        else:
            return jsonify({'success':'true','result':result})
    else:
        return jsonify({'success':'false','result':result}) 


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)