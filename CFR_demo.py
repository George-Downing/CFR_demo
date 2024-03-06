import numpy as np
import ctypes
import io


class Ptr(int):
    HEAP = {}

    def __init__(self, val):
        # warning: be ware of conflict like which happened in malloc (currently turned off)
        pass
    
    def get_obj(self):
        return self.HEAP[self]
    def set_obj(self, obj):
        self.HEAP[self] = obj
    obj = property(get_obj, set_obj)

    @classmethod
    def ref(cls, obj):
        return cls(id(obj))
    
    @classmethod
    def register_into_heap(cls, obj):
        cls.ref(obj).obj = obj
    
    def free(self):
        del self.HEAP[self]
    
    '''
    @classmethod
    def malloc(cls):
        # warning: currently, pointers generated from id won't check occupied value and may result in a conflict.
        # this is because get_ptr_from_id is too straightforward, mapping will be added later.
        while True:
            p = Ptr(1 + np.random.randint(0, 2**24-1)*(2**24+1) + np.random.randint(0, 2**24+1))  # (0, 2**48)
            if p not in cls.HEAP.keys():
                break
        return p
    '''

class Node:
    def __init__(self, parent=Ptr(0)) -> None:
        self.parent = parent
        self.child = []
        Ptr.register_into_heap(self)

        if self.parent != Ptr(0):
            self.parent.obj.child.append(Ptr.ref(self))

    def __repr__(self) -> str:
        stream = io.StringIO()
        print("parent=", self.parent, ", child=", self.child, sep="", file=stream)
        stream_val = stream.getvalue()
        stream.close()
        return stream_val


root = Node()
Node(parent=Ptr.ref(root))

for p in Ptr.HEAP.keys():
    print(p, ":", p.obj)
    # print(ctypes.cast(p, ctypes.py_object).value)
