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

    def strategy_init(self):
        # sigma_init (SET luck + INT rand_sig)
        # pi_init (zero-Padding + SET luck)
        # cfv_init (zero-padding + SET leaf)
        # cfr_init (zero-padding)
        # cfR_init (zero-padding)
        # exploitability (zero-padding)

        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        cfR: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                cfR[P_name][I] = np.zeros([len(act)], dtype=float)

        sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {LUCK: {}, A: {}, B: {}}
        for obs in self.info_collect[LUCK]:
            I = self.info_collect[LUCK][obs]
            act = self.act[LUCK][I]
            sigma[LUCK][I] = np.array([self._card_num[I][a] for a in act]) / sum(self._card_num[I].values())
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                sigma[P_name][I] = rand_sig(len(act))

        pi: dict[NodePtr, np.ndarray] = {}
        for l in self.node_layers:
            for n in l:
                pi[n] = np.array(0.0, dtype=float)
        pi[self.true_root] = np.array(1.0)
        for l in self.node_layers[0:3]:
            for n in l:
                act = self.act[LUCK][n.o.I]
                for i, a in enumerate(act):
                    pi[n.o.child[a]] = pi[n] * sigma[LUCK][n.o.I][i]

        cfv: dict[NodePtr, np.ndarray] = {}
        for l in self.node_layers:
            for n in l:
                cfv[n] = np.array([0.0, 0.0], dtype=float)

        # round-1, fold leaves
        for r in self.round1_roots:
            for seq in Leduc.folds_template:
                leaf = r.o(seq)
                if len(seq) % 2 == 1:
                    cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                else:
                    cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
        # round-2, fold leaves
        for r in self.round2_roots:
            for seq in Leduc.folds_template:
                leaf = r.o(seq)
                if len(seq) % 2 == 1:
                    cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                else:
                    cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
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
                    cfv[leaf] = np.array([1, -1]) * self._ante[leaf][1] - self._ante[leaf] * 0.05
                elif winner == "B":
                    cfv[leaf] = np.array([-1, 1]) * self._ante[leaf][0] - self._ante[leaf] * 0.05
                elif winner == "/":
                    cfv[leaf] = -self._ante[leaf] * 0.05

        cfr: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                act = self.act[P_name][I]
                cfr[P_name][I] = np.zeros([len(act)], dtype=float)

        ep: dict[player_t, dict[InfoSetPtr, float]] = {A: {}, B: {}}
        for P_name in [A, B]:
            for obs in self.info_collect[P_name]:
                I = self.info_collect[P_name][obs]
                ep[P_name][I] = 0.0

        return cfR, sigma, pi, cfv, cfr, ep

    def print_iters(self, t, cfR, sigma, pi, cfv, cfr, ep):
        np.set_printoptions(precision=4, suppress=True)
        with open(os.path.join(self.PRINT, "cfR_sum_" + str(t) + ".txt"), "w") as f:
            for P_name in cfR:
                for I in cfR[P_name]:
                    print(I, ": ", cfR[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "sigma_" + str(t) + ".txt"), "w") as f:
            for P_name in sigma:
                for I in sigma[P_name]:
                    print(I, ": ", sigma[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "pi_" + str(t) + ".txt"), "w") as f:
            for l in self.node_layers:
                for n in l:
                    print(n, pi[n], file=f)
        with open(os.path.join(self.PRINT, "cfv_" + str(t) + ".txt"), "w") as f:
            for n in cfv:
                print(n, cfv[n], file=f)
        with open(os.path.join(self.PRINT, "cfr_" + str(t) + ".txt"), "w") as f:
            for P_name in cfr:
                for I in cfr[P_name]:
                    print(I, ": ", cfr[P_name][I], file=f, sep="")
        with open(os.path.join(self.PRINT, "ep_" + str(t) + ".txt"), "w") as f:
            for P_name in ep:
                for I in ep[P_name]:
                    print(I, ": ", ep[P_name][I], file=f, sep="")

    def sync(self, sigma, pi, cfv, cfr, ep):
        A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK

        # pi_sync:
        for l in range(len(Leduc.round_template)):
            P_name = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                for r in self.round1_roots:
                    n = r.o(pa)
                    for i, a in enumerate(act):
                        pi[n.o.child[a]] = pi[n] * sigma[P_name][n.o.I][i]
        for l in range(len(Leduc.round_template)):
            P_name = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                for r in self.round2_roots:
                    n = r.o(pa)
                    for i, a in enumerate(act):
                        pi[n.o.child[a]] = pi[n] * sigma[P_name][n.o.I][i]

        # cfv_sync (non-leaf-nodes):
        # parent actively collect the cfv (not child submit onward)
        for r in self.round2_roots:  # parallelize THIS level
            for l in range(len(Leduc.round_template))[::-1]:
                P_name = A if l % 2 == 0 else B
                for pa, act in Leduc.round_template[l]:
                    n = r.o(pa)
                    cfv[n] *= 0.0
                    for i, a in enumerate(self.act[P_name][n.o.I]):
                        cfv[n] += cfv[n.o.child[a]] * sigma[P_name][n.o.I][i]
        for r in self.round1_roots:  # parallelize THIS level
            for l in range(len(Leduc.round_template))[::-1]:
                P_name = A if l % 2 == 0 else B
                for pa, act in Leduc.round_template[l]:
                    n = r.o(pa)
                    cfv[n] *= 0.0
                    for i, a in enumerate(self.act[P_name][n.o.I]):
                        cfv[n] += cfv[n.o.child[a]] * sigma[P_name][n.o.I][i]
        for l in self.node_layers[2::-1]:
            for n in l:
                cfv[n] *= 0.0
                for i, a in enumerate(self.act[LUCK][n.o.I]):
                    cfv[n] += cfv[n.o.child[a]] * sigma[LUCK][n.o.I][i]

        # cfr_sync
        for P_id, P_name in enumerate([A, B]):
            for I in cfr[P_name]:
                cfr[P_name][I] *= 0.0
                for n in I.o:
                    val_diff = np.array([cfv[n.o.child[a]][P_id] for a in self.act[P_name][I]]) - cfv[n][P_id]
                    val_diff[val_diff < 0] *= 0  # strict cfr
                    cfr[P_name][I] += pi[n] * val_diff

        # ep_sync
        for P_name in [A, B]:
            for I in ep[P_name]:
                ep[P_name][I] = np.max(cfr[P_name][I])

    @staticmethod
    def step(cfr, cfR, sigma):
        # cfR_update (accumulated version)
        for P_name in cfR:
            for I in cfR[P_name]:
                cfR[P_name][I] += cfr[P_name][I]

        # sigma_sync
        for P_name in cfR:
            for I in cfR[P_name]:
                norm = cfR[P_name][I].sum()
                if norm > 0:
                    sigma[P_name][I] = cfR[P_name][I] / norm


def main():
    TIME = {}
    time_start = time.time()
    TIME["timer_start"] = time.time() - time_start
    print("timer_start", TIME["timer_start"])

    game = Leduc()
    game.print_structures()
    TIME["build"] = time.time() - time_start
    print("build", TIME["build"])

    cfR, sigma, pi, cfv, cfr, ep = game.strategy_init()
    game.print_iters(0, cfR, sigma, pi, cfv, cfr, ep)
    TIME["val_init"] = time.time() - time_start
    print("val_init", TIME["val_init"])

    FLAG_LOAD = True
    FLAG_LOAD_SUCCESS = False
    if FLAG_LOAD:
        try:
            with open(os.path.join(game.DATA, "game.true_root.pkl"), "rb") as f:
                game_true_root_old: NodePtr = pickle.load(f)
            with open(os.path.join(game.DATA, "Node.HEAP.pkl"), "rb") as f:
                Node_HEAP_old: dict[NodePtr, Node] = pickle.load(f)
            with open(os.path.join(game.DATA, "game.info_collect.pkl"), "rb") as f:
                game_info_collect_old: dict[player_t, dict[str, InfoSetPtr]] = pickle.load(f)

            with open(os.path.join(game.DATA, "cfR_sum.pkl"), "rb") as f:
                cfR_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
            with open(os.path.join(game.DATA, "sigma.pkl"), "rb") as f:
                sigma_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
            with open(os.path.join(game.DATA, "pi.pkl"), "rb") as f:
                pi_old: dict[NodePtr, np.ndarray] = pickle.load(f)
            with open(os.path.join(game.DATA, "cfv.pkl"), "rb") as f:
                cfv_old: dict[NodePtr, np.ndarray] = pickle.load(f)
            with open(os.path.join(game.DATA, "cfr.pkl"), "rb") as f:
                cfr_old: dict[player_t, dict[InfoSetPtr, np.ndarray]] = pickle.load(f)
            with open(os.path.join(game.DATA, "ep.pkl"), "rb") as f:
                ep_old: dict[player_t, dict[InfoSetPtr, float]] = pickle.load(f)
            FLAG_LOAD_SUCCESS = True
        except Exception as e:
            print("Structure/iteration data corrupted, no overwrite happened yet.")
            FLAG_LOAD_SUCCESS = False

    FLAG_REVERSE_SUCCESS = False
    FLAG_TRANSCRIPT_SUCCESS = False
    if FLAG_LOAD_SUCCESS:
        NodePtr_reverse: dict[NodePtr, NodePtr] = {game.true_root: game_true_root_old}
        InfoSet_reverse: dict[InfoSetPtr, InfoSetPtr] = {}
        try:
            for l in game.node_layers:
                for n in l:
                    for a in n.o.child:
                        NodePtr_reverse[n.o.child[a]] = Node_HEAP_old[NodePtr_reverse[n]].child[a]
            for P_name in game.info_collect:
                for obs in game.info_collect[P_name]:
                    I = game.info_collect[P_name][obs]
                    I_old = game_info_collect_old[P_name][obs]
                    InfoSet_reverse[I] = I_old
            FLAG_REVERSE_SUCCESS = True
        except Exception as e:
            print("Exception during reverse, no overwrite happened yet.")
            FLAG_REVERSE_SUCCESS = False

        if FLAG_REVERSE_SUCCESS:
            try:
                for P_name in cfR:
                    for I in cfR[P_name]:
                        cfR[P_name][I] = cfR_old[P_name][InfoSet_reverse[I]]
                for P_name in sigma:
                    for I in sigma[P_name]:
                        sigma[P_name][I] = sigma_old[P_name][InfoSet_reverse[I]]
                for n in pi:
                    pi[n] = pi_old[NodePtr_reverse[n]]
                for n in cfv:
                    cfv[n] = cfv_old[NodePtr_reverse[n]]
                for P_name in cfr:
                    for I in cfr[P_name]:
                        cfr[P_name][I] = cfr_old[P_name][InfoSet_reverse[I]]
                for P_name in ep:
                    for I in cfr[P_name]:
                        ep[P_name][I] = ep_old[P_name][InfoSet_reverse[I]]
                FLAG_TRANSCRIPT_SUCCESS = True
            except Exception as e:
                print("Exception during transcription, try rerun load or init a new one.")
                FLAG_TRANSCRIPT_SUCCESS = False

    for t in range(11, 21):
        flag_print = True if t % 1 == 0 else False
        game.sync(sigma, pi, cfv, cfr, ep)
        if flag_print:
            game.print_iters(t, cfR, sigma, pi, cfv, cfr, ep)
            TIME[t] = time.time() - time_start
            print(t, TIME[t])
        game.step(cfr, cfR, sigma)

    with open(os.path.join(game.DATA, "game.true_root.pkl"), "wb") as f:
        pickle.dump(game.true_root, f)
    with open(os.path.join(game.DATA, "Node.HEAP.pkl"), "wb") as f:
        pickle.dump(Node.HEAP, f)
    with open(os.path.join(game.DATA, "game.info_collect.pkl"), "wb") as f:
        pickle.dump(game.info_collect, f)

    with open(os.path.join(game.DATA, "cfR_sum.pkl"), "wb") as f:
        pickle.dump(cfR, f)
    with open(os.path.join(game.DATA, "sigma.pkl"), "wb") as f:
        pickle.dump(sigma, f)
    with open(os.path.join(game.DATA, "pi.pkl"), "wb") as f:
        pickle.dump(pi, f)
    with open(os.path.join(game.DATA, "cfv.pkl"), "wb") as f:
        pickle.dump(cfv, f)
    with open(os.path.join(game.DATA, "cfr.pkl"), "wb") as f:
        pickle.dump(cfr, f)
    with open(os.path.join(game.DATA, "ep.pkl"), "wb") as f:
        pickle.dump(ep, f)


if __name__ == "__main__":
    main()
