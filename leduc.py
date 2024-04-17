import os
# import io
# import numpy as np
# import ctypes  # print(ctypes.cast(p, ctypes.py_object).value)
# import abc
# from typing import Union, TypeVar
import extensive_form
from extensive_form import *

PRINT_OUT = "print_out"


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # higher level
    A, B, LUCK = player_t("A"), player_t("B"), player_t("LUCK")
    true_root: NodePtr = Node(NodePtr(0), act_t("N/A")).p
    layers: list[list[NodePtr]] = [[true_root]]
    info_collect: dict[player_t, dict[str, InfoSetPtr]] = {A: {}, B: {}, LUCK: {}}
    act_map: dict[player_t, dict[InfoSetPtr, list[act_t]]] = {A: {}, B: {}, LUCK: {}}
    sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}, LUCK: {}}

    # 0->1->2->3: player="LUCK", (EQUIVALENT to a SINGLE composite step)
    # InfoSets:
    n = true_root
    I = InfoSet(LUCK, "-".join(n.o.h())).p
    I.o.append(n)
    n.o.I = I
    info_collect[LUCK][I.o.observation] = I
    act_map[LUCK][I] = [act_t(c) for c in ["J1", "J2", "Q1", "Q2", "K1", "K2"]]
    sigma[LUCK][I] = np.ones([len(act_map[LUCK][I])], dtype=float) / len(act_map[LUCK][I])
    # LUCK's play
    for l in range(3):
        # Nodes:
        layers.append([])
        for pa in layers[-2]:
            for a in act_map[LUCK][pa.o.I]:
                ch = Node(pa, a).p
                ch.o.parent.o.child[ch.o.branch] = ch
                layers[-1].append(ch)
        if l == 2:
            break
        # InfoSets:
        for n in layers[-1]:
            I = InfoSet(LUCK, "-".join(n.o.h())).p
            I.o.append(n)  # for loop omitted because of singleton
            n.o.I = I  # for loop omitted because of singleton
            info_collect[LUCK][I.o.observation] = I
            act_map[LUCK][n.o.I] = [b for b in act_map[LUCK][n.o.parent.o.I] if b != n.o.branch]
            sigma[LUCK][I] = np.ones([len(act_map[LUCK][I])], dtype=float) / len(act_map[LUCK][I])

    betting_round_layout = [
        [([], ["r", "c"])],
        [(["r"], ["f", "r", "c"]), (["c"], ["r", "c"])],
        [(["r", "r"], ["f", "c"]), (["c", "r"], ["f", "r", "c"])],
        [(["c", "r", "r"], ["f", "c"])]
    ]
    many_roots = layers[-1]
    many_info_roots: dict[player_t, list[str]] = {A: ["J??", "Q??", "K??"], B: ["?J?", "?Q?", "?K?"]}
    round1_leaves = []
    round2_roots = []

    for l in range(len(betting_round_layout)):
        layers.append([])
        for pa, acts in betting_round_layout[l]:
            acts = act_t.cast(acts)
            player_i = A if l % 2 == 0 else B
            for info_r in many_info_roots[player_i]:
                obs = info_r + "".join(pa)
                I = InfoSet(player_i, obs).p
                info_collect[player_i][I.o.observation] = I
                act_map[player_i][I] = acts
                sigma[player_i][I] = extensive_form.rand_sig_numpy_ndarray(len(act_map[player_i][I]))
            for r in many_roots:
                n = r.o(act_t.cast(pa))
                h = n.o.h()
                if player_i == A:
                    obs = h[0][0] + "?" + "?" + "".join(h[3:])
                    I = info_collect[A][obs]
                else:
                    obs = "?" + h[1][0] + "?" + "".join(h[3:])
                    I = info_collect[B][obs]
                I.o.append(n)
                n.o.I = I
                for a in acts:
                    ch = Node(n, a).p
                    ch.o.parent.o.child[ch.o.branch] = ch
                    layers[-1].append(ch)

    for r in many_roots:
        for n in [r.o(["r", "f"]), r.o(["r", "r", "f"]), r.o(["c", "r", "f"]), r.o(["c", "r", "r", "f"])]:
            round1_leaves.append(n)
        for n in [r.o(["r", "c"]), r.o(["c", "c"]), r.o(["r", "r", "c"]), r.o(["c", "r", "c"]), r.o(["c", "r", "r", "c"])]:
            round2_roots.append(n)

    with open(os.path.join(PRINT_OUT, "nodes.txt"), "w") as f:
        for l in range(len(layers)):
            for n in layers[l]:
                print(n, file=f)
    with open(os.path.join(PRINT_OUT, "many_roots.txt"), "w") as f:
        for n in many_roots:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "info_collect.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in info_collect[player_i]:
                print(I, ": ", info_collect[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "act_map.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in act_map[player_i]:
                print(I, ": ", act_map[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "sigma.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in sigma[player_i]:
                print(I, ": ", sigma[player_i][I], file=f, sep="")
