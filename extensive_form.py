import io
import numpy as np


class act_t(str):
    @staticmethod
    def test():
        _ = act_t("fold")
        print('act_t("fold"):', _)

    @classmethod
    def cast(cls, l: list[str]):
        return [act_t(i) for i in l]


class player_t(str):  # -1(stochastic), 0, 1, ...
    @staticmethod
    def test():
        _ = player_t("Abc")
        print('player_t("Abc"):', _)


class Node(object):
    HEAP: dict["NodePtr", "Node"] = {}

    def __init__(self, parent: "NodePtr", branch: act_t) -> None:
        self.HEAP[self.p] = self

        self.parent: NodePtr = parent
        self.branch: act_t = branch
        self.child: dict[act_t, NodePtr] = {}

        self.I: InfoSetPtr = InfoSetPtr(0)
        self.pi: float = 1.0
        self.cfv: np.ndarray = np.array([0.0, 0.0])

    @property
    def p(self) -> "NodePtr":
        return NodePtr(id(self))

    def h(self) -> list:
        p = self.p
        y: list[act_t] = []
        while p.o.parent != NodePtr(0):
            y.append(p.o.branch)
            p = p.o.parent
        y.reverse()
        return y

    def __call__(self, acts) -> "NodePtr":
        p = self.p
        for a in acts:
            p = p.o.child[a]
        return p

    def __repr__(self) -> str:
        buff = io.StringIO()
        print(self.h(), file=buff, end="")
        return buff.getvalue()


class NodePtr(int):
    @property
    def o(self) -> Node:
        return Node.HEAP[self]

    @o.setter
    def o(self, o):
        Node.HEAP[self] = o

    def __repr__(self) -> str:
        return super(self.__class__, self).__repr__() + "->" + self.o.__repr__()


class InfoSet(list[NodePtr]):
    HEAP: dict["InfoSetPtr", "InfoSet"] = {}
    OBS_PTR: dict[str, "InfoSetPtr"] = {}

    def __init__(self, P: player_t, observation: str = ""):
        super().__init__()
        self.HEAP[self.p] = self

        self.P: player_t = P
        self.observation: str = observation

    @classmethod
    def get_ptr(cls, item):
        pass

    @property
    def p(self):
        return InfoSetPtr(id(self))

    def __repr__(self) -> str:
        buff = io.StringIO()
        print("{obs=", self.observation, file=buff, end=", ", sep="")
        print("N=", len(self), file=buff, end="}", sep="")
        return buff.getvalue()

    def print_pro(self) -> str:
        buff = io.StringIO()
        print("{P:", self.P, file=buff, end=", ")
        print("obs=", self.observation, file=buff, end=", ", sep="")
        print("N=", len(self), file=buff, end=", ", sep="")
        print("Nodes:", super(InfoSet, self).__repr__(), file=buff, end="}")
        return buff.getvalue()


class InfoSetPtr(int):
    @property
    def o(self):
        return InfoSet.HEAP[self]

    @o.setter
    def o(self, o):
        InfoSet.HEAP[self] = o

    def __repr__(self) -> str:
        return super(self.__class__, self).__repr__() + "->" + self.o.__repr__()


def rand_sig_numpy_ndarray(N: int) -> np.ndarray:
    val = np.random.rand(N - 1)
    val.sort()
    p = np.concatenate([val, np.array([1.0])], axis=0)
    p[1:] -= p[:-1]
    return p


def rand_sig_dict(acts: list[act_t]) -> dict[act_t, float]:
    sig: dict[act_t, float] = {}
    val = np.random.rand(len(acts)-1)
    val.sort()
    p = np.concatenate([val, np.array([1.0])], axis=0)
    p[1:] -= p[:-1]
    for i in range(len(acts)):
        sig[acts[i]] = p[i]
    return sig
