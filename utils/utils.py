import configparser
import chardet

CONFIG = configparser.ConfigParser()

def get_shape(file_path):
    """
    从ini文件获取shape,若无则返回None

    :file_path str:  文件路径
    :file_path str:  字符串 相对或者绝对路径
    """
    CONFIG.read( file_path, get_encoding(file_path))
    if CONFIG.has_option("width") and CONFIG.has_option("height"):
        return (int(CONFIG['height']), int(CONFIG['width']))
    else:
        return None
    

def get_encoding(file_path): 
    """
    二进制方式读取，获取字节数据，检测类型

    :file_path str:  文件路径
    :file_path str:  字符串 相对或者绝对路径
    """
    with open(file_path, 'rb') as f:
        data = f.read()
        return chardet.detect(data)['encoding']
    
def normalized(x):
    return (x - x.min())/(x.max()-x.min())