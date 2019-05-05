#-*- coding:utf-8 -*- 
# 设置默认的level为DEBUG
# 设置log的格式
import logging as logger
from logging import config
# logger.basicConfig(
#     level=logger.DEBUG, 
#     format="[%(levelname)s] %(message)s"
# )

def logger_init(conf_file="log.conf"):
    logger.config.fileConfig(conf_file)