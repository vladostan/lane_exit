# -*- coding: utf-8 -*-

import os
import sys

class Logger(object):
    def __init__(self, dirname, filename):
        self.terminal = sys.stdout
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()