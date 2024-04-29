import os.path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle
import arithmetics
from graph import act_t, Node, NodePtr

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)

# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = ["black", "red", "green", "blue", "cyan", "magenta", "yellow"]

class Goofspiel(object):
    def __init__(self, CARD_NUM: int, upcard: list[int], weight: dict[str, list[float]]):
        # non-iterables
        self.CARD_NUM: int = CARD_NUM
        self.upcard: list[int] = upcard
        self.weight: dict[str, list[float]] = weight
        self.node_layer: list[list[NodePtr]] = []  # nodes in breadth-first layout

        # states
        self.time: int = 0
        self.reg: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # reg[play_name][info][act]
        self.sig: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # sig[play_name][info][act]
        self.opt: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # opt[play_name][info][act]
        self.tab: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # tab[play_name][state][act, act]
        self.cfv: dict[NodePtr, np.ndarray] = {}  # cfv[node][play_id]
        self.exp: dict[NodePtr, np.ndarray] = {}  # exp[state][play_id]

        # records
        self.TIMEs: np.ndarray = np.empty([0], int)
        self.REGs: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # REGs[play][info][act, t]
        self.SIGs: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # SIGs[play][info][act, t]
        self.OPTs: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # OPTs[play][info][act, t]
        self.CFVs: dict[NodePtr, np.ndarray] = {}  # CFVs[state][play, t]
        self.EXPs: dict[NodePtr, np.ndarray] = {}  # EXPs[state][play, t]

    def new_game(self):
        self.node_init()
        self.state_init()
        with open("node.db", "wb") as db:
            pickle.dump(self.CARD_NUM, db)
            pickle.dump(self.upcard, db)
            pickle.dump(self.weight, db)
            pickle.dump(self.node_layer, db)
            pickle.dump(Node.MEM_N, db)
            pickle.dump(Node.MEM, db)
        with open("state.db", "wb") as db:
            pickle.dump(self.reg, db)
            pickle.dump(self.time, db)

        try:
            shutil.rmtree("logs")
        except:
            pass
        try:
            shutil.rmtree("curves")
        except:
            pass

    def load_game(self):
        with open("node.db", "rb") as db:
            self.CARD_NUM = pickle.load(db)
            self.upcard = pickle.load(db)
            self.weight = pickle.load(db)
            self.node_layer = pickle.load(db)
            Node.MEM_N = pickle.load(db)
            Node.MEM = pickle.load(db)
        with open("state.db", "rb") as db:
            self.reg = pickle.load(db)
            self.time = pickle.load(db)
        self.var_alloc()
        self.payoff_fill()

    def save_checkpoint(self):
        with open("state.db", "wb") as db:
            pickle.dump(self.reg, db)
            pickle.dump(self.time, db)

        if "logs" not in os.listdir():
            os.mkdir("logs")
        else:
            if "TIMEs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "TIMEs.db"), "rb") as db:
                    TIMEs = pickle.load(db)
            if "REGs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "REGs.db"), "rb") as db:
                    REGs = pickle.load(db)
            if "SIGs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "SIGs.db"), "rb") as db:
                    SIGs = pickle.load(db)
            if "OPTs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "OPTs.db"), "rb") as db:
                    OPTs = pickle.load(db)
            if "CFVs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "CFVs.db"), "rb") as db:
                    CFVs = pickle.load(db)
            if "EXPs.db" in os.listdir("logs"):
                with open(os.path.join("logs", "EXPs.db"), "rb") as db:
                    EXPs = pickle.load(db)

            self.TIMEs = np.concatenate([TIMEs, self.TIMEs], 0)
            for i in ["A", "B"]:
                for n in self.REGs[i].keys():
                    self.REGs[i][n] = np.concatenate([REGs[i][n], self.REGs[i][n]], 1)
                for n in self.SIGs[i].keys():
                    self.SIGs[i][n] = np.concatenate([SIGs[i][n], self.SIGs[i][n]], 1)
                for n in self.OPTs[i].keys():
                    self.OPTs[i][n] = np.concatenate([OPTs[i][n], self.OPTs[i][n]], 1)
            for n in self.CFVs.keys():
                self.CFVs[n] = np.concatenate([CFVs[n], self.CFVs[n]], 1)
            for n in self.EXPs.keys():
                self.EXPs[n] = np.concatenate([EXPs[n], self.EXPs[n]], 1)

        with open(os.path.join("logs", "TIMEs.db"), "wb") as db:
            pickle.dump(self.TIMEs, db)
        with open(os.path.join("logs", "REGs.db"), "wb") as db:
            pickle.dump(self.REGs, db)
        with open(os.path.join("logs", "SIGs.db"), "wb") as db:
            pickle.dump(self.SIGs, db)
        with open(os.path.join("logs", "OPTs.db"), "wb") as db:
            pickle.dump(self.OPTs, db)
        with open(os.path.join("logs", "CFVs.db"), "wb") as db:
            pickle.dump(self.CFVs, db)
        with open(os.path.join("logs", "EXPs.db"), "wb") as db:
            pickle.dump(self.EXPs, db)

    def node_init(self):
        root = Node.new()
        root.o.branch = np.array([0, 0])
        root.o.actA = np.arange(self.CARD_NUM) + 1
        root.o.actB = np.arange(self.CARD_NUM) + 1

        parent = [root]
        child = []
        self.node_layer = []
        while len(parent):
            self.node_layer.append(parent)
            for n in parent:
                n.o.child = np.resize(n.o.child, [len(n.o.actA), len(n.o.actB)])
                for i, a in enumerate(n.o.actA):
                    for j, b in enumerate(n.o.actB):
                        n.o.child[i, j] = Node.new()
                        ch = n.o.child[i, j]
                        ch.o.parent = n
                        ch.o.branch = np.array([a, b])
                        ch.o.actA = n.o.actA[n.o.actA != a]
                        ch.o.actB = n.o.actB[n.o.actB != b]
                        child.append(ch)
            parent = child
            child = []

    def state_init(self):
        for layer in self.node_layer[0:-2]:
            for n in layer:
                self.reg["A"][n] = arithmetics.rand_f(1, len(n.o.actA) - 1)[0] * 5
                self.reg["B"][n] = arithmetics.rand_f(1, len(n.o.actB) - 1)[0] * 5

    def var_alloc(self):
        for layer in self.node_layer[0:-2]:
            for n in layer:
                self.sig["A"][n] = np.empty([len(n.o.actA)], dtype=float)
                self.sig["B"][n] = np.empty([len(n.o.actB)], dtype=float)
                self.tab["A"][n] = np.empty([len(n.o.actA), len(n.o.actB)], dtype=float)
                self.tab["B"][n] = np.empty([len(n.o.actA), len(n.o.actB)], dtype=float)
                self.opt["A"][n] = np.empty([len(n.o.actA)], dtype=float)
                self.opt["B"][n] = np.empty([len(n.o.actB)], dtype=float)
                self.cfv[n] = np.empty([2], dtype=float)
                self.exp[n] = np.empty([2], dtype=float)
        for layer in self.node_layer[-2:]:
            for n in layer:
                self.cfv[n] = np.empty([2], dtype=float)

    def payoff_fill(self):
        for n in self.node_layer[-1]:
            h = n.o.h()
            u, v = 0, 0
            for k, acts in enumerate(h):
                a, b = acts
                if a > b:
                    u += self.upcard[k] * self.weight["A"][k]
                    v -= self.upcard[k] * self.weight["B"][k]
                elif a < b:
                    u -= self.upcard[k] * self.weight["A"][k]
                    v += self.upcard[k] * self.weight["B"][k]
            self.cfv[n][:] = np.array([u, v], dtype=float)
        for n in self.node_layer[-2]:
            self.cfv[n][:] = self.cfv[n.o.get_child(0, 0)].copy()

    def log_alloc(self, T=500):
        box, acc = 100, 1
        while game.time > box:
            box = box * 2 if str(box)[0] != "2" else box // 2 * 5
            acc = acc * 2 if str(acc)[0] != "2" else acc // 2 * 5
        log_from = int(np.ceil(game.time / acc)) * acc

        self.TIMEs = np.arange(log_from, min(box, T), acc)
        while T > box:
            big_box = box * 2 if str(box)[0] != "2" else box // 2 * 5
            big_acc = acc * 2 if str(acc)[0] != "2" else acc // 2 * 5
            self.TIMEs = np.concatenate([self.TIMEs, np.arange(box, min(big_box, T), big_acc)], 0)
            box, acc = big_box, big_acc

        L = len(self.TIMEs)
        for layer in self.node_layer[0:-2]:
            for n in layer:
                self.REGs["A"][n] = np.empty([len(n.o.actA), L], dtype=float)
                self.REGs["B"][n] = np.empty([len(n.o.actB), L], dtype=float)
                self.SIGs["A"][n] = np.empty([len(n.o.actA), L], dtype=float)
                self.SIGs["B"][n] = np.empty([len(n.o.actB), L], dtype=float)
                self.OPTs["A"][n] = np.empty([len(n.o.actA), L], dtype=float)
                self.OPTs["B"][n] = np.empty([len(n.o.actB), L], dtype=float)
                self.CFVs[n] = np.empty([2, L], dtype=float)
                self.EXPs[n] = np.empty([2, L], dtype=float)

    def reg_upd(self, num=2):
        for layer in self.node_layer[0:-num]:  # cheat 20230315
            for n in layer:
                x = self.sig["A"][n][:]
                y = self.sig["B"][n][:]
                u = self.opt["A"][n][:]
                v = self.opt["B"][n][:]
                r = {"A": u - u.T @ x, "B": v - v.T @ y}
                r["A"][r["A"] < 0] *= 0.9999
                r["B"][r["B"] < 0] *= 0.9999

                self.reg["A"][n][:] = self.reg["A"][n][:] + r["A"]
                self.reg["B"][n][:] = self.reg["B"][n][:] + r["B"]
                self.reg["A"][n][self.reg["A"][n][:] < 0] = 0
                self.reg["B"][n][self.reg["B"][n][:] < 0] = 0
        self.time += 1

    def sync(self, num=2):
        for layer in self.node_layer[-1-num::-1]:  # cheat 20230315
            for n in layer:
                self.sig["A"][n][:] = self.reg["A"][n][:] / self.reg["A"][n][:].sum()
                self.sig["B"][n][:] = self.reg["B"][n][:] / self.reg["B"][n][:].sum()
                for i, a in enumerate(n.o.actA):
                    for j, b in enumerate(n.o.actB):
                        self.tab["A"][n][i, j], self.tab["B"][n][i, j] = self.cfv[n.o.get_child(i, j)][:]
                self.opt["A"][n][:] = self.tab["A"][n] @ self.sig["B"][n]
                self.opt["B"][n][:] = self.tab["B"][n].T @ self.sig["A"][n]
                self.cfv[n][0] = self.opt["A"][n].T @ self.sig["A"][n]
                self.cfv[n][1] = self.opt["B"][n].T @ self.sig["B"][n]
                self.exp[n][0] = self.opt["A"][n][:].max() - self.cfv[n][0]
                self.exp[n][1] = self.opt["B"][n][:].max() - self.cfv[n][1]

    def log_rec(self, t: int, num=2):
        for layer in self.node_layer[-1-num::-1]:  # cheat 20230315
            for n in layer:
                self.REGs["A"][n][:, t] = self.reg["A"][n][:].copy()
                self.REGs["B"][n][:, t] = self.reg["B"][n][:].copy()
                self.SIGs["A"][n][:, t] = self.sig["A"][n][:].copy()
                self.SIGs["B"][n][:, t] = self.sig["B"][n][:].copy()
                self.OPTs["A"][n][:, t] = self.opt["A"][n][:].copy()
                self.OPTs["B"][n][:, t] = self.opt["B"][n][:].copy()
                self.CFVs[n][:, t] = self.cfv[n].copy()
                self.EXPs[n][:, t] = self.exp[n].copy()


def run(game: Goofspiel, T=500):
    game.load_game()
    game.sync()
    game.log_alloc(T)

    t_start = time.time()
    t_ = 0
    while game.time < T:
        game.sync(num=4)  # cheat 2023-0315 (pruning)
        try:
            if game.time == game.TIMEs[t_]:
                game.log_rec(t_, num=4)  # cheat 2023-0315 (pruning)
                if game.time % 100 == 0:
                    print("milestone", game.time, "saved")
                t_ += 1
        except:
            reason = "t_: [array_index_exceed] when (game.time) is between (game.TIMEs[-1]+1) and (T-1). "
            reason += "Try-except is a low-cost version of pre-checking, so everything goes fine."
            pass
        game.reg_upd(num=3)
    print("iter:", time.time() - t_start)


def split_to_stackers(A: np.ndarray):
    M, N = A.shape
    y = []
    for i in range(M):
        y.append(A[i, :])
    return y


def output(game: Goofspiel):
    try:
        shutil.rmtree("curves")
    except:
        reason = "There is no directory (curves) to [rmtree]"
        pass
    finally:
        os.mkdir("curves")
        for d in ["cfv_A", "cfv_B", "ep", "reg_A", "reg_B", "sig_A", "sig_B"]:
            os.mkdir(os.path.join("curves", d))

    with open(os.path.join("logs", "TIMEs.db"), "rb") as db:
        game.TIMEs = pickle.load(db)
    with open(os.path.join("logs", "REGs.db"), "rb") as db:
        game.REGs = pickle.load(db)
    with open(os.path.join("logs", "SIGs.db"), "rb") as db:
        game.SIGs = pickle.load(db)
    with open(os.path.join("logs", "OPTs.db"), "rb") as db:
        game.OPTs = pickle.load(db)
    with open(os.path.join("logs", "CFVs.db"), "rb") as db:
        game.CFVs = pickle.load(db)
    with open(os.path.join("logs", "EXPs.db"), "rb") as db:
        game.EXPs = pickle.load(db)

    # reg
    t_start = time.time()
    for l in game.node_layer[0:1]:
        for n in l:
            h = str(n.o.h())
            fig = plt.figure(h, [8, 5], 96)

            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            colors = [COLORS[a % 10] for a in n.o.actA]
            ax.stackplot(game.TIMEs, *split_to_stackers(game.REGs["A"][n]), labels=n.o.actA, colors=colors, alpha=0.5)
            ax.set_title(r'$R_1$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("Regret", fontsize=20)
            fig.savefig(os.path.join("curves", "reg_A", h + ".png"))

            ax.clear()
            colors = [COLORS[b % 10] for b in n.o.actB]
            ax.stackplot(game.TIMEs, *split_to_stackers(game.REGs["B"][n]), labels=n.o.actB, colors=colors, alpha=0.5)
            ax.set_title(r'$R_2$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("Regret", fontsize=20)
            fig.savefig(os.path.join("curves", "reg_B", h + ".png"))
            plt.close(fig)
    print("reg:", time.time() - t_start)

    # sig
    t_start = time.time()
    for l in game.node_layer[0:1]:
        for n in l:
            h = str(n.o.h())
            fig = plt.figure(h, [8, 5], 96)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            colors = [COLORS[a % 10] for a in n.o.actA]
            ax.stackplot(game.TIMEs, *split_to_stackers(game.SIGs["A"][n]), labels=n.o.actA, colors=colors, alpha=0.5)
            ax.set_title(r'$\sigma_1$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("probability", fontsize=20)
            fig.savefig(os.path.join("curves", "sig_A", h + ".png"))

            ax.clear()
            colors = [COLORS[b % 10] for b in n.o.actB]
            ax.stackplot(game.TIMEs, *split_to_stackers(game.SIGs["B"][n]), labels=n.o.actB, colors=colors, alpha=0.5)
            ax.set_title(r'$\sigma_2$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("probability", fontsize=20)
            fig.savefig(os.path.join("curves", "sig_B", h + ".png"))
            plt.close(fig)
    print("sig:", time.time() - t_start)

    # cfv
    t_start = time.time()
    for l in game.node_layer[0:1]:
        for n in l:
            h = str(n.o.h())
            fig = plt.figure(h, [8, 5], 96)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.plot(game.TIMEs, game.CFVs[n][0], label="mixed")
            for i, a in enumerate(n.o.actA):
                ax.plot(game.TIMEs, game.OPTs["A"][n][i, :], label=a, alpha=0.3, color="C" + str(a))
            ax.set_title(r'$u_1$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("payoff", fontsize=20)
            fig.savefig(os.path.join("curves", "cfv_A", h + ".png"))

            ax.clear()
            ax.plot(game.TIMEs, game.CFVs[n][1], label="mixed")
            for j, b in enumerate(n.o.actB):
                ax.plot(game.TIMEs, game.OPTs["B"][n][j, :], label=b, alpha=0.3, color="C" + str(b))
            ax.set_title(r'$u_2$', fontsize=20)
            ax.grid()
            ax.legend()
            ax.set_xlabel("iteration", fontsize=20)
            ax.set_ylabel("payoff", fontsize=20)
            fig.savefig(os.path.join("curves", "cfv_B", h + ".png"))
            plt.close(fig)
    print("cfv:", time.time() - t_start)

    # ep-value
    t_start = time.time()
    for l in game.node_layer[0:2]:
        for n in l:
            h = str(n.o.h())
            fig: plt.Figure = plt.figure(h, [8, 8], 96)
            fig.add_axes([0.1, 0.1, 0.8, 0.8])
            fig.axes[0].loglog(game.TIMEs, game.EXPs[n].sum(axis=0), label=r'$ep$')
            fig.axes[0].loglog(game.TIMEs, game.EXPs[n][0, :], label=r'$ep_1$', alpha=0.3)
            fig.axes[0].loglog(game.TIMEs, game.EXPs[n][1, :], label=r'$ep_2$', alpha=0.3)
            fig.axes[0].set_title(r'$ep$', fontsize=20)
            fig.axes[0].set_xlabel("iteration", fontsize=20)
            fig.axes[0].set_ylabel("exploitability", fontsize=20)

            ax: plt.Axes
            for ax in fig.axes:
                ax.set_ylim(0.01, 4)
                ax.set_xlim(50, 64000)
                ax.grid()
                ax.legend()
            fig.savefig(os.path.join("curves", "ep", h + ".png"))
            plt.close(fig)
    print("exp:", time.time() - t_start)


if __name__ == "__main__":
    np.random.seed(133484)
    CARD_NUM = 5
    upcard = [1, 1, 1, 1, 1]
    weight = {"A": [5.00, 1.33, 2.71, 4.80, 2.24], "B": [4.10, 6.28, 3.33, 1.92, 3.60]}
    game = Goofspiel(CARD_NUM=CARD_NUM, upcard=upcard, weight=weight)
    #game.new_game()
    game.load_game()
    run(game, 64000)
    game.save_checkpoint()

    game = Goofspiel(CARD_NUM=CARD_NUM, upcard=upcard, weight=weight)
    game.load_game()
    output(game)
