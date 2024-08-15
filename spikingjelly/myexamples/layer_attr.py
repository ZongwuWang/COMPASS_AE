from pyngleton import Singleton

class layer_attr(Singleton):
    def __init__(self):
        self.layer_attr_list = []
        self.header = -1
        self.full = False
    def incStop(self):
        self.full = True
    def incHeader(self):
        if not self.full:
            self.header += 1
        else:
            self.header = (self.header + 1) % len(self.layer_attr_list)