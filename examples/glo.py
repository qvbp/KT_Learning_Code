# # 使用一个内部的字典来存储全局变量
# _global_dict = {}
# def _init():
#     global _global_dict
#     _global_dict = {}

# def set_value(key, value):
#     _global_dict[key] = value

# def get_value(key, defValue=None):
#     return _global_dict[key]

# glo.py

# glo.py

_global_dict = None  # 使用一个模块级别变量来存储全局状态

def _init():
    """初始化全局变量字典"""
    global _global_dict
    if _global_dict is None:
        _global_dict = {}  # 初始化一次

def set_value(key, value):
    """设置全局变量"""
    if _global_dict is None:
        _init()  # 自动初始化
    _global_dict[key] = value

def get_value(key, defValue=None):
    """获取全局变量的值，不存在时返回默认值"""
    if _global_dict is None:
        _init()  # 自动初始化
    return _global_dict.get(key, defValue)

