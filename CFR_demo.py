import numpy as np
import ctypes
import io


class act_t(str):
    @staticmethod
    def test():
        a = act_t("fold")
        print('act_t("fold"):', a)


class Ptr(int):
    HEAP = {}

    def __init__(self, val):
        # warning: beware of conflict like which happened in malloc (currently turned off)
        pass
    
    def get_obj(self):
        return self.HEAP[self]

    def set_obj(self, obj):
        self.HEAP[self] = obj

    o = property(get_obj, set_obj)

    @classmethod
    def p(cls, obj):
        return cls(id(obj))
    
    @classmethod
    def record(cls, obj):
        cls.p(obj).o = obj
    
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


class Node(object):
    def __init__(self, parent=Ptr(0), branch=act_t("")) -> None:
        self.parent = parent
        self.branch = branch
        self.child: dict[act_t, Ptr] = {}
        Ptr.record(self)

        if self.parent != Ptr(0):
            self.parent.o.child[self.branch] = Ptr.p(self)

    def __repr__(self) -> str:
        buff = io.StringIO()
        print("parent:", self.parent, file=buff, end=", ")
        print("branch:", self.branch, file=buff, end=", ")
        print("child:", self.child, file=buff)
        return buff.getvalue()


if __name__ == "__main__":
    root = Node(Ptr(0), act_t(""))
    Node(Ptr.p(root), act_t("raise"))

    for p in Ptr.HEAP.keys():
        print(p, ": ", p.o, sep="")
        # print(ctypes.cast(p, ctypes.py_object).value)
