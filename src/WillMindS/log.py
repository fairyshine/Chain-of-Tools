import logging 

def Log_Init(config):
    print('------------------------------------')
    print('--------------初始化日志--------------')
    # 实例化一个 Logger 
    logger = logging.getLogger("WillMindS") #logger = logging.getLogger() 

    # 设置日志输出等级 
    logger.setLevel(logging.DEBUG) 

    # 设置日志输出格式 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 实例化写入日志文件的Handler
    file_handler = logging.FileHandler('my_log/{} | {}.log'.format(config.model_name,config.start_time))
    file_handler.setFormatter(formatter) 

    # 实例化实时输出的Handler
    try:
        from rich.logging import RichHandler
        shell_handler = RichHandler(rich_tracebacks=True)
    except ImportError:
        shell_handler = logging.StreamHandler(stream=None)
        shell_handler.setFormatter(formatter) 

    # 添加到 logger 
    logger.addHandler(file_handler) 
    logger.addHandler(shell_handler)

    # 输出日志 
    # logger.debug('this is a debug message') 
    # logger.info('this is an info message') 
    # logger.warning('this is a warning message') 
    # logger.error('this is an error message') 
    # logger.critical('this is a critical message')

    logger.info('------------------------------------')
    logger.info('------------日志初始化完成------------')

    return logger