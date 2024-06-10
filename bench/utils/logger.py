import logging
import os


class Logger():
    logger = None
    log_name = 'default'

    def get_logger(self):
        if Logger.logger is not None:
            return Logger.logger
        log_file = '{}.log'.format(Logger.log_name)
        logging.basicConfig(level=logging.INFO,
                            filename=log_file,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        Logger.logger = logging.getLogger('FL-type')
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        Logger.logger.addHandler(sh)
        Logger.logger.info('logger init finished ---- log file: {}'.format(log_file))
        return Logger.logger

    def set_log_name(self, name):
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        while name[-4:] == '.log':
            name = name[:-4]
        Logger.log_name = name
