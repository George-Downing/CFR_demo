import os
import time
# import pickle
# from typing import Dict, Union, Any
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import numpy as np
# from numpy import ndarray

from extensive_form import player_t, act_t, Node, NodePtr, InfoSet, InfoSetPtr, rand_sig

PRINT_OUT = "print_out"


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
    many_round1_info_heads: dict[player_t, list[str]] = {A: ["J??", "Q??", "K??"], B: ["?J?", "?Q?", "?K?"]}
    many_round2_info_heads: dict[player_t, list[str]] = {A: [], B: []}
    many_round2_info_titles = {
        A: ["J?J", "J?Q", "J?K", "Q?J", "Q?Q", "Q?K", "K?J", "K?Q", "K?K"],
        B: ["?JJ", "?JQ", "?JK", "?QJ", "?QQ", "?QK", "?KJ", "?KQ", "?KK"]}
    for _player_i in many_round2_info_titles:
        for _info_title in many_round2_info_titles[_player_i]:
            for _c in calls_template:
                many_round2_info_heads[_player_i].append(_info_title + "".join(_c))
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
    ante: dict[NodePtr, np.ndarray] = {}

    def __init__(self):
        pass


def create_tree():
    # higher level
    LUCK, A, B = Leduc.LUCK, Leduc.A, Leduc.B
    true_root: NodePtr = Node(NodePtr(0), act_t("N/A")).p
    Node.roots.append(true_root)
    node_layers: list[list[NodePtr]] = [[true_root]]
    head_I: InfoSetPtr = InfoSet(LUCK, "").p
    info_collect: dict[player_t, dict[str, InfoSetPtr]] = {LUCK: {"": head_I}, A: {}, B: {}}
    act_map: dict[player_t, dict[InfoSetPtr, list[act_t]]] = {LUCK: {head_I: act_t.cast(["J1", "J2", "Q1", "Q2", "K1", "K2"])}, A: {}, B: {}}

    head_I.o.append(true_root)
    true_root.o.I = head_I
    card_num: dict[InfoSetPtr, dict[str, int]] = {head_I: {"J": 2, "Q": 2, "K": 2}}
    # 0->1->2->3: player="LUCK", (EQUIVALENT to a SINGLE composite step)

    for l in range(3):
        node_layers.append([])
        for pa in node_layers[-2]:
            for a in act_map[LUCK][pa.o.I]:
                ch = Node(pa, a).p
                ch.o.parent.o.child[ch.o.branch] = ch
                node_layers[-1].append(ch)
        if l == 2:
            break
        for n in node_layers[-1]:
            I = InfoSet(LUCK, "-".join(n.o.h())).p
            I.o.append(n)
            n.o.I = I
            info_collect[LUCK][I.o.observation] = I
            act_map[LUCK][n.o.I] = [b for b in act_map[LUCK][n.o.parent.o.I] if b != n.o.branch]

    # 3->4->5->6->7, 7->8->9->10->11, player="A"/"B"
    round1_roots: list[NodePtr] = node_layers[-1]
    round2_roots: list[NodePtr] = []
    leaves: list[NodePtr] = []
    # round 1
    for l in range(len(Leduc.round_template)):
        player_i = A if l % 2 == 0 else B
        for info_r in Leduc.many_round1_info_heads[player_i]:  # diff
            for pa, acts in Leduc.round_template[l]:
                acts = act_t.cast(acts)
                obs = info_r + "".join(pa)
                I = InfoSet(player_i, obs).p
                info_collect[player_i][I.o.observation] = I
                act_map[player_i][I] = acts
        node_layers.append([])
        for r in round1_roots:  # diff
            for pa, acts in Leduc.round_template[l]:
                n = r.o(act_t.cast(pa))
                h = n.o.h()
                obs = ((h[0][0] + "?" + "?") if (player_i == A) else ("?" + h[1][0] + "?")) + "".join(h[3:])  # diff
                I = info_collect[player_i][obs]
                I.o.append(n)
                n.o.I = I
                for a in acts:
                    ch = Node(n, a).p
                    ch.o.parent.o.child[ch.o.branch] = ch
                    node_layers[-1].append(ch)
    for r in round1_roots:  # diff
        for n in [r.o(f) for f in Leduc.folds_template]:
            leaves.append(n)
        for n in [r.o(c) for c in Leduc.calls_template]:
            round2_roots.append(n)  # diff
    # round 2
    for l in range(len(Leduc.round_template)):
        player_i = A if l % 2 == 0 else B
        for info_r in Leduc.many_round2_info_heads[player_i]:  # diff
            for pa, acts in Leduc.round_template[l]:
                acts = act_t.cast(acts)
                obs = info_r + "".join(pa)
                I = InfoSet(player_i, obs).p
                info_collect[player_i][I.o.observation] = I
                act_map[player_i][I] = acts
        node_layers.append([])
        for r in round2_roots:  # diff
            for pa, acts in Leduc.round_template[l]:
                n = r.o(act_t.cast(pa))
                h = n.o.h()
                obs = ((h[0][0] + "?" + h[2][0]) if (player_i == A) else ("?" + h[1][0] + h[2][0])) + "".join(h[3:])  # diff
                I = info_collect[player_i][obs]
                I.o.append(n)
                n.o.I = I
                for a in acts:
                    ch = Node(n, a).p
                    ch.o.parent.o.child[ch.o.branch] = ch
                    node_layers[-1].append(ch)
    for r in round2_roots:  # diff
        for n in [r.o(f) for f in Leduc.folds_template]:
            leaves.append(n)
        for n in [r.o(c) for c in Leduc.calls_template]:
            leaves.append(n)  # diff

    ante = Leduc.ante
    for r in round1_roots:
        ante[r] = np.array([1, 1], dtype=float)
    for r in round1_roots:
        for h, m in Leduc.added_money:
            ante[r.o(h)] = ante[r] + 2 * m
    for r in round2_roots:
        for h, m in Leduc.added_money:
            ante[r.o(h)] = ante[r] + 4 * m

    np.set_printoptions(precision=4, suppress=True)
    with open(os.path.join(PRINT_OUT, "node_layers.txt"), "w") as f:
        for l in node_layers:
            for n in l:
                print(n, file=f)
    with open(os.path.join(PRINT_OUT, "round1_roots.txt"), "w") as f:
        for n in round1_roots:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "round2_roots.txt"), "w") as f:
        for n in round2_roots:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "leaves.txt"), "w") as f:
        for n in leaves:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "info_collect.txt"), "w") as f:
        for player_i in info_collect:
            for I in info_collect[player_i]:
                print(I, ": ", info_collect[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "act_map.txt"), "w") as f:
        for player_i in info_collect:
            for I in act_map[player_i]:
                print(I, ": ", act_map[player_i][I], file=f, sep="")

    return node_layers, true_root, round1_roots, round2_roots, leaves, info_collect, act_map


def calculate_square(xy):
    x, y = xy
    return x**2 + y**2


def main():
    x = [5 * i - 20 for i in range(100)]
    y = [30 - 4 * i for i in range(100)]
    inputs = list(zip(x, y))
    with ProcessPoolExecutor() as executor:
        z = list(executor.map(calculate_square, inputs))
    print(z)


if __name__ == "__main__":
    TIME = {}
    time_start = time.time()
    TIME["start"] = time.time() - time_start

    A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK
    node_layers, true_root, r1_roots, r2_roots, leaves, info_collect, act_map = create_tree()
    ante = Leduc.ante
    # node_layers: NEW: [1, 3,  9,  24,  48, 120, 120,  48 (then  24 quited, and  96 joined),  240,  600,  600,  240]
    # node_layers: OLD: [1, 6, 30, 120, 240, 600, 600, 240 (then 120 quited, and 480 joined), 1200, 3000, 3000, 1200]
    # many_round1_roots: NEW: 24; OLD: 120
    # many_round2_roots: NEW: 120 = 24*5; OLD: 600 = 120*5
    # round1_leaves: NEW: 96 = 24*4; OLD: 480 = 120*4
    # round2_leaves: NEW: 1080 = 24*5*(4+5); OLD: 5400 = 120*5*(4+5)
    TIME["leduc"] = time.time() - time_start


    # sigma_init (rand_sig):
    sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {LUCK: {}, A: {}, B: {}}
    for obs in info_collect[LUCK]:
        I = info_collect[LUCK][obs]
        act = act_map[LUCK][I]
        sigma[LUCK][I] = np.ones([len(act)])/len(act)
    for player_i in [A, B]:
        for obs in info_collect[player_i]:
            I = info_collect[player_i][obs]
            act = act_map[player_i][I]
            sigma[player_i][I] = rand_sig(len(act))
    TIME["sigma_init"] = time.time() - time_start

    # pi_init (placeholder + luck):
    pi: dict[NodePtr, np.ndarray] = {true_root: np.array(1.0)}
    # placeholder
    for l in node_layers:
        for n in l:
            pi[n] = np.array(1.0)
    # luck's fixed mechanism
    for l in node_layers[0:3]:
        for n in l:
            act = act_map[LUCK][n.o.I]
            for i, a in enumerate(act):
                pi[n.o.child[a]] = pi[n] * sigma[LUCK][n.o.I][i]
    TIME["pi_init"] = time.time() - time_start

    # cfv_init (placeholder + set leaf nodes):
    cfv: dict[NodePtr, np.ndarray] = {}
    # placeholder
    for l in node_layers:
        for n in l:
            cfv[n] = np.array([0.0, 0.0], dtype=float)
    # round-1, fold leaves
    for r in r1_roots:
        for seq in Leduc.folds_template:
            leaf = r.o(seq)
            if len(seq) % 2 == 1:
                cfv[leaf] = np.array([-1, 1]) * ante[leaf][0] - ante[leaf] * 0.05
            else:
                cfv[leaf] = np.array([1, -1]) * ante[leaf][1] - ante[leaf] * 0.05
    # round-2, fold leaves
    for r in r2_roots:
        for seq in Leduc.folds_template:
            leaf = r.o(seq)
            if len(seq) % 2 == 1:
                cfv[leaf] = np.array([-1, 1]) * ante[leaf][0] - ante[leaf] * 0.05
            else:
                cfv[leaf] = np.array([1, -1]) * ante[leaf][1] - ante[leaf] * 0.05
    # round-2, call leaves
    for r in r2_roots:
        h = r.o.h()
        winner = "/"
        if h[0][0] == h[1][0]:
            winner = "/"  # the only way to tie, P=20%
        elif h[0][0] == h[2][0]:
            winner = "A"  # A win by match, P=20%
        elif h[1][0] == h[2][0]:
            winner = "B"  # B win by match, P=20%
        else:  # mutually distinct
            if h[0][0] == "K":
                winner = "A"  # A win, P=13.33%
            elif h[0][0] == "Q":
                if h[1][0] == "J":
                    winner = "A"  # A win, P=6.67%
                elif h[1][0] == "K":
                    winner = "B"  # B win, P=6.67%
            elif h[0][0] == "J":
                winner = "B"  # B win, P=13.33%
        for h in Leduc.calls_template:
            leaf = r.o(h)
            if winner == "A":
                cfv[leaf] = np.array([1, -1]) * ante[leaf][1] - ante[leaf] * 0.05
            elif winner == "B":
                cfv[leaf] = np.array([-1, 1]) * ante[leaf][0] - ante[leaf] * 0.05
            elif winner == "/":
                cfv[leaf] = -ante[leaf] * 0.05
    TIME["cfv_init"] = time.time() - time_start

    # cfr_init (placeholder):
    cfr: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}}
    for player_i in cfr:
        for obs in info_collect[player_i]:
            I = info_collect[player_i][obs]
            act = act_map[player_i][I]
            cfr[player_i][I] = rand_sig(len(act)) * 0.0
    TIME["cfr_init"] = time.time() - time_start

    with open(os.path.join(PRINT_OUT, "sigma.txt"), "w") as f:
        for player_i in sigma:
            for I in sigma[player_i]:
                print(I, ": ", sigma[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "pi.txt"), "w") as f:
        for l in node_layers:
            for n in l:
                print(n, pi[n], file=f)
    with open(os.path.join(PRINT_OUT, "cfv.txt"), "w") as f:
        for n in cfv:
            print(n, cfv[n], file=f)
    with open(os.path.join(PRINT_OUT, "cfr.txt"), "w") as f:
        for player_i in cfr:
            for I in cfr[player_i]:
                print(I, ": ", cfr[player_i][I], file=f, sep="")
    TIME["print_init"] = time.time() - time_start

    # ITERATIONS:
    # pi_refresh:
    for l in range(len(Leduc.round_template)):
        player_i = A if l % 2 == 0 else B
        for pa, act in Leduc.round_template[l]:
            for r in r1_roots:
                n = r.o(pa)
                for i, a in enumerate(act):
                    pi[n.o.child[a]] = pi[n] * sigma[player_i][n.o.I][i]
    for l in range(len(Leduc.round_template)):
        player_i = A if l % 2 == 0 else B
        for pa, act in Leduc.round_template[l]:
            for r in r2_roots:
                n = r.o(pa)
                for i, a in enumerate(act):
                    pi[n.o.child[a]] = pi[n] * sigma[player_i][n.o.I][i]

    # cfv_refresh (non-leaf-nodes):
    # parent actively collect the cfv (not child submit onward)
    for r in r2_roots:  # parallelize THIS level
        for l in range(len(Leduc.round_template))[::-1]:
            player_i = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                n = r.o(pa)
                cfv[n] *= 0.0
                for i, a in enumerate(act_map[player_i][n.o.I]):
                    cfv[n] += cfv[n.o.child[a]] * sigma[player_i][n.o.I][i]
    for r in r1_roots:  # parallelize THIS level
        for l in range(len(Leduc.round_template))[::-1]:
            player_i = A if l % 2 == 0 else B
            for pa, act in Leduc.round_template[l]:
                n = r.o(pa)
                cfv[n] *= 0.0
                for i, a in enumerate(act_map[player_i][n.o.I]):
                    cfv[n] += cfv[n.o.child[a]] * sigma[player_i][n.o.I][i]
    for l in node_layers[2::-1]:
        for n in l:
            cfv[n] *= 0.0
            for i, a in enumerate(act_map[LUCK][n.o.I]):
                cfv[n] += cfv[n.o.child[a]] * sigma[LUCK][n.o.I][i]

    # cfr_algorithm
    i = 0
    for player_i in cfr:
        for I in cfr[player_i]:
            for n in I.o:
                for ch in n.o.child:
                    i += 1
    print(i)
    TIME["iter"] = time.time() - time_start

    # print
    with open(os.path.join(PRINT_OUT, "sigma.txt"), "w") as f:
        for player_i in sigma:
            for I in sigma[player_i]:
                print(I, ": ", sigma[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "pi.txt"), "w") as f:
        for l in node_layers:
            for n in l:
                print(n, pi[n], file=f)
    with open(os.path.join(PRINT_OUT, "cfv.txt"), "w") as f:
        for n in cfv:
            print(n, cfv[n], file=f)
    with open(os.path.join(PRINT_OUT, "cfr.txt"), "w") as f:
        for player_i in cfr:
            for I in cfr[player_i]:
                print(I, ": ", cfr[player_i][I], file=f, sep="")
    TIME["print"] = time.time() - time_start


    # time complexity and performance
    i = 0
    for key in TIME.keys():
        print(key, TIME[key])
        i += 1
