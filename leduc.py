import numpy as np
import ctypes
import io


class act_t(str):
    @staticmethod
    def test():
        a = act_t("fold")
        print('act_t("fold"):', a)


class Node(object):
    def __init__(self, parent, branch) -> None:
        self.parent: NodePtr = parent
        self.branch: act_t = branch
        self.child: dict[act_t, NodePtr] = {}

        NodePtr.HEAP[self.p] = self
        if self.parent != NodePtr(0):
            self.parent.o.child[self.branch] = self.p

    def h(self) -> list[act_t]:
        p = self.p
        y: list[act_t] = []
        while p.o.parent != NodePtr(0):
            y.append(p.o.branch)
            p = p.o.parent
        y.reverse()
        return y

    def __call__(self, acts) -> "Node":
        p = self.p
        for a in acts:
            if a in p.o.child.keys():
                p = p.o.child[a]
            else:
                p = self.p
                print("Warning: invalid action sequence, nothing will take effect.")
                break
        return p.o

    def __repr__(self) -> str:
        buff = io.StringIO()
        print("h:", self.h(), file=buff, end=", ")
        print("act:", *self.child.keys(), file=buff, end="")
        return buff.getvalue()

    def get_p(self) -> "NodePtr":
        return NodePtr.p(self)
    p = property(get_p)


class NodePtr(int):
    HEAP: dict["NodePtr", Node] = {}

    def __init__(self, val) -> None:
        # warning: beware of conflict like which happened in malloc (currently turned off)
        pass

    def get_o(self) -> Node:
        return self.HEAP[self]

    def set_o(self, o) -> None:
        self.HEAP[self] = o

    o = property(get_o, set_o)

    @classmethod
    def p(cls, o) -> "NodePtr":
        return cls(id(o))

    @classmethod
    def record(cls, o) -> None:
        cls.p(o).o = o

    def free(self) -> None:
        del self.HEAP[self]

    def __repr__(self) -> str:
        return str(self.o.h())


if __name__ == "__main__":
    R = [0, 0]
    root = Node(NodePtr(0), "")
    Node(root.p, "R")
    Node(root.p, "C")
    Node(root.p, "F")

    Node(root(["R"]).p, "R")
    for p in NodePtr.HEAP.keys():
        print(p.o)
        # print(ctypes.cast(p, ctypes.py_object).value)
