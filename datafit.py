import numpy as np
import imageio
import os
from utils.utils import get_shape
SHAPE = (1526, 3331)

def raw2tif(file_name, save_path = 'data/tif/'):
    """
    转换类型 raw -> tif 格式

    :file_name str:  文件路径
    :file_name str:  字符串 相对或者绝对路径
    """
    raw_img_shape = None
    file_ini = file_name[:-3] + "ini"
    if os.path.exists(file_ini):
        raw_img_shape = get_shape(file_ini)

    if raw_img_shape is None:
        raw_img_shape = SHAPE
    raw = np.memmap( file_name, dtype = np.uint16, shape = raw_img_shape)  # 读取16位raw图片
    
    last_index = file_name.rfind('/')  # 找到倒数第一个逗号的索引
    substring = file_name[last_index:]  # 截取到倒数第一个/之前的部分
    file_name = substring[5:-4]
    file_name = file_name.replace('\\', '_')
    savename = save_path + file_name
    print(file_name)
    imageio.imsave(savename+'.tif', raw)  # 转换格式与保存,tif为RGB格式，如果normalize后的npy转成tif查看需要×256

def raw2np(file_name):
    """
    读取 raw -> numpy 格式

    :file_name str:  文件路径
    :file_name str:  字符串 相对或者绝对路径
    """
    raw_img_shape = None
    file_ini = file_name[:-3] + "ini"
    if os.path.exists(file_ini):
        raw_img_shape = get_shape(file_ini)

    if raw_img_shape is None:
        raw_img_shape = SHAPE
    raw = np.memmap( file_name, dtype = np.uint16, shape = raw_img_shape)  # 读取16位raw图片
    return raw

def normalized(data):
    data_normal = (data - data.min())/(data.max() - data.min())
    return data_normal

def raw2cv(file_path):
    data = raw2np(file_path)
    n_data = normalized(data)
    cv_data = np.array(n_data * 255, np.uint8)
    return cv_data