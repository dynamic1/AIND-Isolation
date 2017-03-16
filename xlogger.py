import logging
import logging.config
import sys


class xlogger(object):
    def __init__(self, logger_name='xLogger'):
        logging.config.fileConfig('xlogging.conf')

        # create logger
        self.__logger__ = logging.getLogger(logger_name)
        self.__context_stack__ = []
        self.__context__ = ""
        pass


    def __update_context__(self):
        self.__context__ = "@".join(self.__context_stack__)

    def set_context(self, str_context):
        self.__context_stack__= [ str_context ]
        self.__update_context__()

    def push_context(self, str_context):
        self.__context_stack__.append(f"[ {str_context} ]")
        self.__update_context__()

    def pop_context(self):
        self.__context_stack__.pop()
        self.__update_context__()

    def debug(self, message, meta=True):
        level = "debug"
        """
        obj = sys._getframe(1).f_code
        for attr in dir(obj):
            print("obj.%s = %s\n" % (attr, getattr(obj, attr)))

        exit()
        """
        if meta:
            self.__logger__.debug(f"{level}: {self.__context__}: {message}")
        else:
            self.__logger__.debug(message)

    def warning(self, message):
        level = "warning"
        self.__logger__.warning(f"{level}: {self.__context__}: {message}")

    def error(self, message):
        level = "error"
        self.__logger__.error(f"{level}: {self.__context__}: {message}")

    def print(self, message):
        return
        print(message)

    def get_context(self):
        return self.__context__

