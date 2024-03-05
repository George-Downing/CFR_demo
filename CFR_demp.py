import numpy as np
import ctypes
import io

class Node:
    MEM = {}
    def __init__(self, parent_id=0):
        self.parent = parent_id
        self.child = []
        self.MEM[id(self)] = id(self)

    def __repr__(self) -> str:
        stream = io.StringIO()
        print(id(self), ": ",  "parent=", self.parent, ", child=", self.child, sep="", file=stream)
        stream_val = stream.getvalue()
        stream.close()
        return stream_val

root = Node()
Node(parent_id=id(root))

for ptr in Node.MEM:
    print(ptr)
    print(ctypes.cast(ptr, ctypes.py_object).value)

    