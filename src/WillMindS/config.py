#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import random
import datetime

from omegaconf import OmegaConf
import numpy as np
import torch

class Config(object):
    def __init__(self):
        # 初始化设置
        args = self.__get_config()

        #将设置逐条引入类的实例变量
        print('------------------------------------')
        print('-------------读取超参数中-------------')
        ''' argparse引入
        for key in args.__dict__:
            print('key: ',key)
            setattr(self, key, args.__dict__[key])
        '''
        # omegeconf引入
        for key in args:
            print('key: ',key)
            setattr(self, key, args.__getattr__(key))

        self.start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # OR  str(datetime.datetime.now())[:-7]

        #####检测可使用的运算设备#####
        # if self.device_wanted=='cuda' and torch.cuda.is_available():
        #     CHIP="cuda"   #Nvidia - Compute Unified Device Architecture
        # elif self.device_wanted=='mps' and torch.backends.mps.is_built():
        #     CHIP="mps"    #Apple Silicon - API Metal - Metal Performance Shaders
        # else:
        #     CHIP="cpu"
        # 选择运算设备
        # self.device = None
        # if self.device_id >= 0 and CHIP != "cpu":
        #     self.device = torch.device('{}:{}'.format(CHIP,self.device_id))
        # else:pip 
        #     self.device = torch.device(CHIP)

        # 辅助处理 model_name 和 model_dir
        if self.model_name is None:
            self.model_name = 'model'
        self.model_dir = os.path.join(self.output_dir, self.model_name+"_"+self.start_time+'/')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 备份设置
        self.__config_backup(args)

        # 设置各类随机数
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'Config-setting filedir to set. All settings are in it.'
        parser.add_argument("--config_file", type=str, default="config/basic.yaml",
                        help="the filedir of config")
        args = parser.parse_args()

        args_dict=dict()
        for key in args.__dict__:
            args_dict[key]=args.__dict__[key]
        cli_conf=OmegaConf.create(args_dict)
        file_conf=OmegaConf.load(args.config_file)
        return OmegaConf.merge(cli_conf,file_conf)

    def __set_seed(self, seed=1116):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu
        torch.backends.cudnn.deterministic = True

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            OmegaConf.save(config=args, f=fw)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])

    def log_print_config(self,logger):
        logger.info('------------------------------------')
        logger.info('Here\'s the config:')
        for key in self.__dict__:
            logger.info(str(key)+' = '+str(self.__dict__[key]))

if __name__ == '__main__':
    config = Config()
    config.print_config()
