import os
import time
import pickle
# from typing import Dict, Union, Any
# from matplotlib import pyplot as plt
import numpy as np
from extensive_form import player_t, act_t, Node, NodePtr, InfoSet, InfoSetPtr, rand_sig

'''
ABSOLUTE-STABLE, candidates:
    absolute-stable/const invariant readonly
WEAK-STATIC, candidates:
    weak/volatile-static/const (4 combinations), non-iterating-objects, inert-parameters
ITERATING-VARIABLES, candidates:
    dynamic iterating-variables
'''


class Leduc(object):
    A, B, LUCK = player_t("A"), player_t("B"), player_t("LUCK")
    round_template: list[list[tuple]] = [
        [([], ["r", "c"])],
        [(["r"], ["f", "r", "c"]), (["c"], ["r", "c"])],
        [(["r", "r"], ["f", "c"]), (["c", "r"], ["f", "r", "c"])],
        [(["c", "r", "r"], ["f", "c"])]
    ]
    folds_template: list[list[str]] = [["r", "f"], ["r", "r", "f"], ["c", "r", "f"], ["c", "r", "r", "f"]]
    calls_template: list[list[str]] = [["r", "c"], ["c", "c"], ["r", "r", "c"], ["c", "r", "c"], ["c", "r", "r", "c"]]
    round1_info_heads: dict[player_t, list[str]] = {A: ["J??", "Q??", "K??"], B: ["?J?", "?Q?", "?K?"]}
    round2_info_heads: dict[player_t, list[str]] = {A: [], B: []}
    round2_info_titles = {
        A: ["J?J", "J?Q", "J?K", "Q?J", "Q?Q", "Q?K", "K?J", "K?Q", "K?K"],
        B: ["?JJ", "?JQ", "?JK", "?QJ", "?QQ", "?QK", "?KJ", "?KQ", "?KK"]}
    for _P_name in [A, B]:
        for _info_title in round2_info_titles[_P_name]:
            for _c in calls_template:
                round2_info_heads[_P_name].append(_info_title + "".join(_c))
    added_money = [
        (["r", "f"], np.array([1, 0], dtype=float)),
        (["r", "r", "f"], np.array([1, 2], dtype=float)),
        (["c", "r", "f"], np.array([0, 1], dtype=float)),
        (["c", "r", "r", "f"], np.array([2, 1], dtype=float)),
        (["r", "c"], np.array([1, 1], dtype=float)),
        (["c", "c"], np.array([0, 0], dtype=float)),
        (["r", "r", "c"], np.array([2, 2], dtype=float)),
        (["c", "r", "c"], np.array([1, 1], dtype=float)),
        (["c", "r", "r", "c"], np.array([2, 2], dtype=float))
    ]

    def __init__(self):
        """
        node_layers: [1, 3, 9, **24** (=27-3), 48, 120, 120, **48** --[-24+96]-> **120**, 240, 600, 600, **240**]

        round1_roots: 24, round2_roots: 120 = 24*5

        round1_leaves: 96 = 24*4, round2_leaves: 1080 = 24*5*(4+5)
        """

        self.PRINT = "stdout"
        self.DATA = "data"
        self.LOGS = "logs"
        self.PNGS = "pngs"
        print("Leduc():", os.listdir())

        LUCK, A, B = Leduc.LUCK, Leduc.A, Leduc.B
        self.true_root: NodePtr = Node(NodePtr(0), act_t("CCCCCCCC")).p
        Node.roots.append(self.true_root)
        self.true_root.o.I = InfoSet(LUCK, "").p
        self.true_root.o.I.o.append(self.true_root)

        self.node_layers: list[list[NodePtr]] = [[self.true_root]]
        self.info_collect: dict[player_t, dict[str, InfoSetPtr]] = {LUCK: {"": self.true_root.o.I}, A: {}, B: {}}
        self.act: dict[player_t, dict[InfoSetPtr, list[act_t]]] = {LUCK: {self.true_root.o.I: act_t.cast(["J", "Q", "K"])}, A: {}, B: {}}

        self.round1_roots: list[NodePtr] = []
        self.round2_roots: list[NodePtr] = []
        self.leaves: list[NodePtr] = []

        self._card_num: dict[InfoSetPtr, dict[act_t, int]] = {self.true_root.o.I: {act_t("J"): 2, act_t("Q"): 2, act_t("K"): 2}}
        self._ante: dict[NodePtr, np.ndarray] = {}

        self.t = 0
        self.cfrs: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        self.sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {LUCK: {}, A: {}, B: {}}
        self.pi: dict[NodePtr, np.ndarray] = {}
        self.cfv: dict[NodePtr, np.ndarray] = {}
        self.cfri: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        self.ep: dict[player_t, dict[InfoSetPtr, float]] = {A: {}, B: {}}

        self.T: np.ndarray = np.zeros([16])
        self.CFRS: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        self.SIGMA: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        self.PI: dict[NodePtr, np.ndarray] = {}
        self.CFV: dict[NodePtr, np.ndarray] = {}
        self.CFRI: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        self.EP: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}

        # 0->1->2->3: player="LUCK", (EQUIVALENT to a SINGLE composite step)
        for l in range(3):
            self.node_layers.append([])
            for pa in self.node_layers[-2]:
                for a in self.act[LUCK][pa.o.I]:
                    ch = Node(pa, a).p
                    ch.o.parent.o.child[ch.o.branch] = ch
                    self.node_layers[-1].append(ch)
            if l == 2:
                break
            for n in self.node_layers[-1]:
                I = InfoSet(LUCK, "".join(n.o.h())).p
                I.o.append(n)
                n.o.I = I
                self.info_collect[LUCK][I.o.observation] = I
                self._card_num[n.o.I] = {b: self._card_num[n.o.parent.o.I][b] for b in self._card_num[n.o.parent.o.I]}
                self._card_num[n.o.I][n.o.branch] = self._card_num[n.o.parent.o.I][n.o.branch] - 1
                self.act[LUCK][n.o.I] = [b for b in self._card_num[n.o.I] if self._card_num[n.o.I][b] != 0]
        for n in self.node_layers[-1]:
            self.round1_roots.append(n)

        # round 1: 3->4->5->6->7, player="A"/"B"
        for l in range(len(Leduc.round_template)):
            player_i = A if l % 2 == 0 else B
            for info_r in Leduc.round1_info_heads[player_i]:  # diff
                for pa, acts in Leduc.round_template[l]:
                    acts = act_t.cast(acts)
                    obs = info_r + "".join(pa)
                    I = InfoSet(player_i, obs).p
                    self.info_collect[player_i][I.o.observation] = I
                    self.act[player_i][I] = acts
            self.node_layers.append([])
            for r in self.round1_roots:  # diff
                for pa, acts in Leduc.round_template[l]:
                    n = r.o(act_t.cast(pa))
                    h = n.o.h()
                    obs = ((h[0] + "?" + "?") if (player_i == A) else ("?" + h[1] + "?")) + "".join(h[3:])  # diff
                    I = self.info_collect[player_i][obs]
                    I.o.append(n)
                    n.o.I = I
                    for a in acts:
                        ch = Node(n, a).p
                        ch.o.parent.o.child[ch.o.branch] = ch
                        self.node_layers[-1].append(ch)
        for r in self.round1_roots:  # diff
            for n in [r.o(f) for f in Leduc.folds_template]:
                self.leaves.append(n)
            for n in [r.o(c) for c in Leduc.calls_template]:
                self.round2_roots.append(n)  # diff

        # round 2: 7->8->9->10->11, player="A"/"B"
        for l in range(len(Leduc.round_template)):
            player_i = A if l % 2 == 0 else B
            for info_r in Leduc.round2_info_heads[player_i]:  # diff
                for pa, acts in Leduc.round_template[l]:
                    acts = act_t.cast(acts)
                    obs = info_r + "".join(pa)
                    I = InfoSet(player_i, obs).p
                    self.info_collect[player_i][I.o.observation] = I
                    self.act[player_i][I] = acts
            self.node_layers.append([])
            for r in self.round2_roots:  # diff
                for pa, acts in Leduc.round_template[l]:
                    n = r.o(act_t.cast(pa))
                    h = n.o.h()
                    obs = ((h[0] + "?" + h[2]) if (player_i == A) else ("?" + h[1] + h[2])) + "".join(h[3:])  # diff
                    I = self.info_collect[player_i][obs]
                    I.o.append(n)
                    n.o.I = I
                    for a in acts:
                        ch = Node(n, a).p
                        ch.o.parent.o.child[ch.o.branch] = ch
                        self.node_layers[-1].append(ch)
        for r in self.round2_roots:  # diff
            for n in [r.o(f) for f in Leduc.folds_template]:
                self.leaves.append(n)
            for n in [r.o(c) for c in Leduc.calls_template]:
                self.leaves.append(n)  # diff

        # ante for round1 leaves (calls included) and round2 leaves
        for r in self.round1_roots:
            self._ante[r] = np.array([1, 1], dtype=float)
        for r in self.round1_roots:
            for h, m in Leduc.added_money:
                self._ante[r.o(h)] = self._ante[r] + 2 * m
        for r in self.round2_roots:
            for h, m in Leduc.added_money:
                self._ante[r.o(h)] = self._ante[r] + 4 * m

    def print_structures(self):
        np.set_printoptions(precision=4, suppress=True)
        with open(os.path.join(self.PRINT, "game.node_layers.txt"), "w") as f:
            for l in self.node_layers:
                for n in l:
                    print(n, file=f)
        with open(os.path.join(self.PRINT, "game.round1_roots.txt"), "w") as f:
            for n in self.round1_roots:
                print(n, file=f)
        with open(os.path.join(self.PRINT, "game.round2_roots.txt"), "w") as f:
            for n in self.round2_roots:
                print(n, file=f)
        with open(os.path.join(self.PRINT, "game.leaves.txt"), "w") as f:
            for n in self.leaves:
                print(n, file=f)
        with open(os.path.join(self.PRINT, "game.info_collect.txt"), "w") as f:
            for player_i in self.info_collect:
                for I in self.info_collect[player_i]:
                    print(I, ": ", self.info_collect[player_i][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "game.act_map.txt"), "w") as f:
            for player_i in self.info_collect:
                for I in self.act[player_i]:
                    print(I, ": ", self.act[player_i][I], file=f, sep="")

    def iters_init(self):
        # sigma_init (SET luck + INT rand_sig)
        # pi_init (zero-Padding + SET luck)
        # cfv_init (zero-padding + SET leaf)
        # cfri_init (zero-padding)
        # cfrs_init (zero-padding)
        # ep_init (zero-padding)

        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        # t_ptr = [0]

        # cfrs: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.cfrs[P_name][I] = np.zeros([len(act)], dtype=float)

        # sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {LUCK: {}, A: {}, B: {}}
        for obs in self.info_collect[LUCK]:
            I = self.info_collect[LUCK][obs]
            act = self.act[LUCK][I]
            self.sigma[LUCK][I] = np.array([self._card_num[I][a] for a in act]) / sum(self._card_num[I].values())
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.sigma[P_name][I] = rand_sig(len(act))

        # pi: dict[NodePtr, np.ndarray] = {}
        for l in self.node_layers:
            for n in l:
                self.pi[n] = np.array(0.0, dtype=float)
        self.pi[self.true_root] = np.array(1.0)
        for l in self.node_layers[0:3]:
            for n in l:
                act = self.act[LUCK][n.o.I]
                for i, a in enumerate(act):
                    self.pi[n.o.child[a]] = self.pi[n] * self.sigma[LUCK][n.o.I][i]

        # cfv: dict[NodePtr, np.ndarray] = {}
        for l in self.node_layers:
            for n in l:
                self.cfv[n] = np.array([0.0, 0.0], dtype=float)

        # round-1, fold leaves
        for r in self.round1_roots:
            for seq in Leduc.folds_template:
                leaf = r.o(seq)
                if len(seq) % 2 == 1:
                    self.cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                else:
                    self.cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
        # round-2, fold leaves
        for r in self.round2_roots:
            for seq in Leduc.folds_template:
                leaf = r.o(seq)
                if len(seq) % 2 == 1:
                    self.cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                else:
                    self.cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
        # round-2, call leaves
        for r in self.round2_roots:
            h = r.o.h()
            winner = "/"
            if h[0] == h[1]:
                winner = "/"  # the only way to tie, P=20%
            elif h[0] == h[2]:
                winner = "A"  # A win by match, P=20%
            elif h[1] == h[2]:
                winner = "B"  # B win by match, P=20%
            else:  # mutually distinct
                if h[0] == "K":
                    winner = "A"  # A win, P=13.33%
                elif h[0] == "Q":
                    if h[1] == "J":
                        winner = "A"  # A win, P=6.67%
                    elif h[1] == "K":
                        winner = "B"  # B win, P=6.67%
                elif h[0] == "J":
                    winner = "B"  # B win, P=13.33%
            for h in Leduc.calls_template:
                leaf = r.o(h)
                if winner == "A":
                    self.cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
                elif winner == "B":
                    self.cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                elif winner == "/":
                    self.cfv[leaf] = -self._ante[leaf] * 0.05

        # cfri: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.cfri[P_name][I] = np.zeros([len(act)], dtype=float)

        # ep: dict[player_t, dict[InfoSetPtr, float]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.ep[P_name][I] = 0.0

        # return t_ptr, cfrs, sigma, pi, cfv, cfri, ep

    def print_iters(self):  # , t_ptr, cfrs, sigma, pi, cfv, cfri, ep):
        np.set_printoptions(precision=4, suppress=True)
        t = self.t
        with open(os.path.join(self.PRINT, "cfrs_" + str(t) + ".txt"), "w") as f:
            for P_name in self.cfrs:
                for I in self.cfrs[P_name]:
                    print(I, ": ", self.cfrs[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "sigma_" + str(t) + ".txt"), "w") as f:
            for P_name in self.sigma:
                for I in self.sigma[P_name]:
                    print(I, ": ", self.sigma[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "pi_" + str(t) + ".txt"), "w") as f:
            for l in self.node_layers:
                for n in l:
                    print(n, self.pi[n], file=f)
        with open(os.path.join(self.PRINT, "cfv_" + str(t) + ".txt"), "w") as f:
            for n in self.cfv:
                print(n, self.cfv[n], file=f)
        with open(os.path.join(self.PRINT, "cfri_" + str(t) + ".txt"), "w") as f:
            for P_name in self.cfri:
                for I in self.cfri[P_name]:
                    print(I, ": ", self.cfri[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "ep_" + str(t) + ".txt"), "w") as f:
            for P_name in self.ep:
                for I in self.ep[P_name]:
                    print(I, ": ", self.ep[P_name][I], file=f, sep="")

    def sync(self):  # , sigma, pi, cfv, cfri, ep):
        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        # pi_sync:
        for l in range(len(Leduc.round_template)):
            P_name = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                for r in self.round1_roots:
                    n = r.o(pa)
                    for i, a in enumerate(act):
                        self.pi[n.o.child[a]] = self.pi[n] * self.sigma[P_name][n.o.I][i]
        for l in range(len(Leduc.round_template)):
            P_name = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                for r in self.round2_roots:
                    n = r.o(pa)
                    for i, a in enumerate(act):
                        self.pi[n.o.child[a]] = self.pi[n] * self.sigma[P_name][n.o.I][i]

        # cfv_sync (non-leaf-nodes):
        # parent actively collect the cfv (not child submit onward)
        for r in self.round2_roots:  # parallelize THIS level
            for l in range(len(Leduc.round_template))[::-1]:
                P_name = A if l % 2 == 0 else B
                for pa, act in Leduc.round_template[l]:
                    n = r.o(pa)
                    self.cfv[n] *= 0.0
                    for i, a in enumerate(self.act[P_name][n.o.I]):
                        self.cfv[n] += self.cfv[n.o.child[a]] * self.sigma[P_name][n.o.I][i]
        for r in self.round1_roots:  # parallelize THIS level
            for l in range(len(Leduc.round_template))[::-1]:
                P_name = A if l % 2 == 0 else B
                for pa, act in Leduc.round_template[l]:
                    n = r.o(pa)
                    self.cfv[n] *= 0.0
                    for i, a in enumerate(self.act[P_name][n.o.I]):
                        self.cfv[n] += self.cfv[n.o.child[a]] * self.sigma[P_name][n.o.I][i]
        for l in self.node_layers[2::-1]:
            for n in l:
                self.cfv[n] *= 0.0
                for i, a in enumerate(self.act[LUCK][n.o.I]):
                    self.cfv[n] += self.cfv[n.o.child[a]] * self.sigma[LUCK][n.o.I][i]

        # cfr_sync
        for P_id, P_name in enumerate([A, B]):
            for I in self.cfri[P_name]:
                self.cfri[P_name][I] *= 0.0
                for n in I.o:
                    val_diff = np.array([self.cfv[n.o.child[a]][P_id] for a in self.act[P_name][I]]) - self.cfv[n][P_id]
                    val_diff[val_diff < 0] *= 0  # strict cfr
                    self.cfri[P_name][I] += self.pi[n] * val_diff

        # ep_sync
        for P_name in [A, B]:
            for I in self.ep[P_name]:
                self.ep[P_name][I] = np.max(self.cfri[P_name][I])

    def step(self):
        self.t += 1

        # cfR_update (accumulated version)
        for P_name in self.cfrs:
            for I in self.cfrs[P_name]:
                self.cfrs[P_name][I] += self.cfri[P_name][I]

        # sigma_sync
        for P_name in self.cfrs:
            for I in self.cfrs[P_name]:
                norm = self.cfrs[P_name][I].sum()
                if norm > 0:
                    self.sigma[P_name][I] = self.cfrs[P_name][I] / norm

    def save(self):
        with open(os.path.join(self.DATA, "game.true_root.pkl"), "wb") as f:
            pickle.dump(self.true_root, f)
        with open(os.path.join(self.DATA, "Node.HEAP.pkl"), "wb") as f:
            pickle.dump(Node.HEAP, f)
        with open(os.path.join(self.DATA, "game.info_collect.pkl"), "wb") as f:
            pickle.dump(self.info_collect, f)

        with open(os.path.join(self.DATA, "t.pkl"), "wb") as f:
            pickle.dump(self.t, f)
        with open(os.path.join(self.DATA, "cfrs.pkl"), "wb") as f:
            pickle.dump(self.cfrs, f)
        with open(os.path.join(self.DATA, "sigma.pkl"), "wb") as f:
            pickle.dump(self.sigma, f)
        with open(os.path.join(self.DATA, "pi.pkl"), "wb") as f:
            pickle.dump(self.pi, f)
        with open(os.path.join(self.DATA, "cfv.pkl"), "wb") as f:
            pickle.dump(self.cfv, f)
        with open(os.path.join(self.DATA, "cfri.pkl"), "wb") as f:
            pickle.dump(self.cfri, f)
        with open(os.path.join(self.DATA, "ep.pkl"), "wb") as f:
            pickle.dump(self.ep, f)

        with open(os.path.join(self.LOGS, "T.pkl"), "wb") as f:
            pickle.dump(self.T, f)
        with open(os.path.join(self.LOGS, "CFRS.pkl"), "wb") as f:
            pickle.dump(self.CFRS, f)
        with open(os.path.join(self.LOGS, "SIGMA.pkl"), "wb") as f:
            pickle.dump(self.SIGMA, f)
        with open(os.path.join(self.LOGS, "PI.pkl"), "wb") as f:
            pickle.dump(self.PI, f)
        with open(os.path.join(self.LOGS, "CFV.pkl"), "wb") as f:
            pickle.dump(self.CFV, f)
        with open(os.path.join(self.LOGS, "CFRI.pkl"), "wb") as f:
            pickle.dump(self.CFRI, f)
        with open(os.path.join(self.LOGS, "EP.pkl"), "wb") as f:
            pickle.dump(self.EP, f)

    def load(self):
        with open(os.path.join(self.DATA, "game.true_root.pkl"), "rb") as f:
            game_true_root_old: NodePtr = pickle.load(f)
        with open(os.path.join(self.DATA, "Node.HEAP.pkl"), "rb") as f:
            Node_HEAP_old: dict[NodePtr, Node] = pickle.load(f)
        with open(os.path.join(self.DATA, "game.info_collect.pkl"), "rb") as f:
            game_info_collect_old: dict[player_t, dict[str, InfoSetPtr]] = pickle.load(f)

        with open(os.path.join(self.DATA, "t.pkl"), "rb") as f:
            t_old: int = pickle.load(f)
        with open(os.path.join(self.DATA, "cfrs.pkl"), "rb") as f:
            cfrs_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.DATA, "sigma.pkl"), "rb") as f:
            sigma_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.DATA, "pi.pkl"), "rb") as f:
            pi_old: dict[NodePtr, np.ndarray] = pickle.load(f)
        with open(os.path.join(self.DATA, "cfv.pkl"), "rb") as f:
            cfv_old: dict[NodePtr, np.ndarray] = pickle.load(f)
        with open(os.path.join(self.DATA, "cfri.pkl"), "rb") as f:
            cfri_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.DATA, "ep.pkl"), "rb") as f:
            ep_old: dict[player_t, dict[InfoSetPtr, float]] = pickle.load(f)

        with open(os.path.join(self.LOGS, "T.pkl"), "rb") as f:
            T_old: np.ndarray = pickle.load(f)
        with open(os.path.join(self.LOGS, "CFRS.pkl"), "rb") as f:
            CFRS_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.LOGS, "SIGMA.pkl"), "rb") as f:
            SIGMA_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.LOGS, "PI.pkl"), "rb") as f:
            PI_old: dict[NodePtr, np.ndarray] = pickle.load(f)
        with open(os.path.join(self.LOGS, "CFV.pkl"), "rb") as f:
            CFV_old: dict[NodePtr, np.ndarray] = pickle.load(f)
        with open(os.path.join(self.LOGS, "CFRI.pkl"), "rb") as f:
            CFRI_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
        with open(os.path.join(self.LOGS, "EP.pkl"), "rb") as f:
            EP_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)

        NodePtr_reverse: dict[NodePtr, NodePtr] = {}
        InfoSet_reverse: dict[InfoSetPtr, InfoSetPtr] = {}

        NodePtr_reverse[self.true_root] = game_true_root_old
        for l in self.node_layers:
            for n in l:
                for a in n.o.child:
                    NodePtr_reverse[n.o.child[a]] = Node_HEAP_old[NodePtr_reverse[n]].child[a]
        for P_name in self.info_collect:
            for obs in self.info_collect[P_name]:
                InfoSet_reverse[self.info_collect[P_name][obs]] = game_info_collect_old[P_name][obs]

        self.t = t_old
        for P_name in self.cfrs:
            for I in self.cfrs[P_name]:
                self.cfrs[P_name][I] = cfrs_old[P_name][InfoSet_reverse[I]]
        for P_name in self.sigma:
            for I in self.sigma[P_name]:
                self.sigma[P_name][I] = sigma_old[P_name][InfoSet_reverse[I]]
        for n in self.pi:
            self.pi[n] = pi_old[NodePtr_reverse[n]]
        for n in self.cfv:
            self.cfv[n] = cfv_old[NodePtr_reverse[n]]
        for P_name in self.cfri:
            for I in self.cfri[P_name]:
                self.cfri[P_name][I] = cfri_old[P_name][InfoSet_reverse[I]]
        for P_name in self.ep:
            for I in self.cfri[P_name]:
                self.ep[P_name][I] = ep_old[P_name][InfoSet_reverse[I]]

        L = len(T_old)
        if L <= len(self.T):
            self.T[0:L] = T_old[0:L]
            for P_name in self.CFRS:
                for I in self.CFRS[P_name]:
                    self.CFRS[P_name][I][:, 0:L] = CFRS_old[P_name][InfoSet_reverse[I]][:, 0:L]
            for P_name in self.SIGMA:
                for I in self.SIGMA[P_name]:
                    self.SIGMA[P_name][I][:, 0:L] = SIGMA_old[P_name][InfoSet_reverse[I]][:, 0:L]
            for n in self.PI:
                self.PI[n][0:L] = PI_old[NodePtr_reverse[n]][0:L]
            for n in self.CFV:
                self.CFV[n][:, 0:L] = CFV_old[NodePtr_reverse[n]][:, 0:L]
            for P_name in self.CFRI:
                for I in self.CFRI[P_name]:
                    self.CFRI[P_name][I][:, 0:L] = CFRI_old[P_name][InfoSet_reverse[I]][:, 0:L]
            for P_name in self.EP:
                for I in self.cfri[P_name]:
                    self.EP[P_name][I][0:L] = EP_old[P_name][InfoSet_reverse[I]][0:L]
        else:
            # this usually occurs when after save then load, but t_MAX was prescribed even smaller than self.t
            # in this situation, t_MAX is treated as if it is exactly self.t, 0 iterations will occur
            self.T = T_old
            for P_name in self.CFRS:
                for I in self.CFRS[P_name]:
                    self.CFRS[P_name][I] = CFRS_old[P_name][InfoSet_reverse[I]]
            for P_name in self.SIGMA:
                for I in self.SIGMA[P_name]:
                    self.SIGMA[P_name][I] = SIGMA_old[P_name][InfoSet_reverse[I]]
            for n in self.PI:
                self.PI[n] = PI_old[NodePtr_reverse[n]]
            for n in self.CFV:
                self.CFV[n] = CFV_old[NodePtr_reverse[n]]
            for P_name in self.CFRI:
                for I in self.CFRI[P_name]:
                    self.CFRI[P_name][I] = CFRI_old[P_name][InfoSet_reverse[I]]
            for P_name in self.EP:
                for I in self.cfri[P_name]:
                    self.EP[P_name][I] = EP_old[P_name][InfoSet_reverse[I]]
        return L

    @staticmethod
    def box_dt(t):
        box, dt = 32, 1
        while box <= t:
            box, dt = box * 2, dt * 2
        return box, dt

    def rec_init(self, t, t_MAX):
        box, dt = self.box_dt(t)
        T = np.arange(int(np.ceil(t / dt)) * dt, min(box, t_MAX), dt)

        while box <= t_MAX:
            box_old = box
            box, dt = box * 2, dt * 2
            T = np.concatenate([T, np.arange(box_old, min(box, t_MAX), dt)], 0)

    def logs_init(self, t_MAX):
        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        self.T = np.arange(0, t_MAX, 5)

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.CFRS[P_name][I] = np.zeros([len(act), len(self.T)], dtype=float)

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.SIGMA[P_name][I] = np.zeros([len(act), len(self.T)])

        for l in self.node_layers:
            for n in l:
                self.PI[n] = np.zeros([len(self.T)], dtype=float)

        for l in self.node_layers:
            for n in l:
                self.CFV[n] = np.zeros([2, len(self.T)], dtype=float)

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                self.CFRI[P_name][I] = np.zeros([len(act), len(self.T)], dtype=float)

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.EP[P_name][I] = np.zeros([len(self.T)])

        return 0  # 0 is for new instance and no load, 0 is t_idx at here

    def logs_rec(self, t_idx):
        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.CFRS[P_name][I][:, t_idx] = self.cfrs[P_name][I]

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.SIGMA[P_name][I][:, t_idx] = self.sigma[P_name][I]

        for l in self.node_layers:
            for n in l:
                self.PI[n][t_idx] = self.pi[n]

        for l in self.node_layers:
            for n in l:
                self.CFV[n][:, t_idx] = self.cfv[n]

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.CFRI[P_name][I][:, t_idx] = self.cfri[P_name][I]

        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                self.EP[P_name][I][t_idx] = self.ep[P_name][I]


def main():
    game = Leduc()
    game.print_structures()

    game.iters_init()
    t_MAX = 600
    t_idx = game.logs_init(t_MAX)
    t_idx = game.load()

    time_start = time.time()
    while game.t < t_MAX:
        game.sync()
        if game.t % 5 == 0:
            game.logs_rec(t_idx)
            t_idx += 1
            if game.t % 20 == 0:
                print(game.t, time.time() - time_start)
        game.step()
    game.save()

    # plot the curves
    # plot CFRS, SIGMA, CFRI, EP, dtype: np.array[P_name][I], total=288
    # plot pi, cfv, dtype: np.array[node], np.array[node][P_idx], total=10237 if full plotted



if __name__ == "__main__":
    # FLAG_PRINT = True if t_ptr[0] % 1 == 0 else False
    # if FLAG_PRINT:
    main()

