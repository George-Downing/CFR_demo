import os
import time
import pickle
import numpy as np

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

    def __init__(self):
        pass


def create_tree():
    # higher level
    A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK
    true_root: NodePtr = Node(NodePtr(0), act_t("N/A")).p
    Node.roots.append(true_root)
    node_layers: list[list[NodePtr]] = [[true_root]]
    info_collect: dict[player_t, dict[str, InfoSetPtr]] = {A: {}, B: {}, LUCK: {}}
    act_map: dict[player_t, dict[InfoSetPtr, list[act_t]]] = {A: {}, B: {}, LUCK: {}}

    # 0->1->2->3: player="LUCK", (EQUIVALENT to a SINGLE composite step)
    for n in node_layers[-1]:
        I = InfoSet(LUCK, "-".join(n.o.h())).p
        I.o.append(n)
        n.o.I = I
        info_collect[LUCK][I.o.observation] = I
        act_map[LUCK][n.o.I] = [act_t(c) for c in ["J1", "J2", "Q1", "Q2", "K1", "K2"]]
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
    many_round1_roots: list[NodePtr] = node_layers[-1]
    many_round2_roots: list[NodePtr] = []
    leaves: list[NodePtr] = []
    # round 1
    for l in range(len(Leduc.round_template)):
        node_layers.append([])
        for pa, acts in Leduc.round_template[l]:
            acts = act_t.cast(acts)
            player_i = A if l % 2 == 0 else B
            for info_r in Leduc.many_round1_info_heads[player_i]:  # diff
                obs = info_r + "".join(pa)
                I = InfoSet(player_i, obs).p
                info_collect[player_i][I.o.observation] = I
                act_map[player_i][I] = acts
            for r in many_round1_roots:  # diff
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
    for r in many_round1_roots:  # diff
        for n in [r.o(f) for f in Leduc.folds_template]:
            leaves.append(n)
        for n in [r.o(c) for c in Leduc.calls_template]:
            many_round2_roots.append(n)  # diff
    # round 2
    for l in range(len(Leduc.round_template)):
        node_layers.append([])
        for pa, acts in Leduc.round_template[l]:
            acts = act_t.cast(acts)
            player_i = A if l % 2 == 0 else B
            for info_r in Leduc.many_round2_info_heads[player_i]:  # diff
                obs = info_r + "".join(pa)
                I = InfoSet(player_i, obs).p
                info_collect[player_i][I.o.observation] = I
                act_map[player_i][I] = acts
            for r in many_round2_roots:  # diff
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
    for r in many_round2_roots:  # diff
        for n in [r.o(f) for f in Leduc.folds_template]:
            leaves.append(n)
        for n in [r.o(c) for c in Leduc.calls_template]:
            leaves.append(n)  # diff

    return true_root, node_layers, many_round1_roots, many_round2_roots, leaves, info_collect, act_map


if __name__ == "__main__":
    A, B, LUCK = Leduc.A, Leduc.B, Leduc.LUCK
    true_root, node_layers, many_round1_roots, many_round2_roots, leaves, info_collect, act_map = create_tree()
    np.set_printoptions(precision=4, suppress=True)
    with open(os.path.join(PRINT_OUT, "node_layers.txt"), "w") as f:
        for l in node_layers:
            for n in l:
                print(n, file=f)
    with open(os.path.join(PRINT_OUT, "many_round1_roots.txt"), "w") as f:
        for n in many_round1_roots:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "many_round2_roots.txt"), "w") as f:
        for n in many_round2_roots:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "leaves.txt"), "w") as f:
        for n in leaves:
            print(n, file=f)
    with open(os.path.join(PRINT_OUT, "info_collect.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in info_collect[player_i]:
                print(I, ": ", info_collect[player_i][I], file=f, sep="")
    with open(os.path.join(PRINT_OUT, "act_map.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in act_map[player_i]:
                print(I, ": ", act_map[player_i][I], file=f, sep="")

    sigma: dict[player_t, dict[InfoSetPtr, np.ndarray]] = {A: {}, B: {}, LUCK: {}}
    for obs in info_collect[LUCK]:
        I = info_collect[LUCK][obs]
        act = act_map[LUCK][I]
        sigma[LUCK][I] = np.ones([len(act)])/len(act)
    for player_i in [A, B]:
        for obs in info_collect[player_i]:
            I = info_collect[player_i][obs]
            act = act_map[player_i][I]
            sigma[player_i][I] = rand_sig(len(act))
    with open(os.path.join(PRINT_OUT, "sigma.txt"), "w") as f:
        for player_i in [LUCK, A, B]:
            for I in sigma[player_i]:
                print(I, ": ", sigma[player_i][I], file=f, sep="")

    pi: dict[NodePtr, float] = {true_root: 1.0}
    for l in node_layers[0:3]:
        for n in l:
            act = act_map[LUCK][n.o.I]
            for i, a in enumerate(act):
                pi[n.o.child[a]] = pi[n] * sigma[LUCK][n.o.I][i]
    for l in range(len(Leduc.round_template)):
        for pa, act in Leduc.round_template[l]:
            player_i = A if l % 2 == 0 else B
            for r in many_round1_roots:
                n = r.o(pa)
                for i, a in enumerate(act):
                    pi[n.o.child[a]] = pi[n] * sigma[player_i][n.o.I][i]
    for l in range(len(Leduc.round_template)):
        for pa, act in Leduc.round_template[l]:
            player_i = A if l % 2 == 0 else B
            for r in many_round2_roots:
                n = r.o(pa)
                for i, a in enumerate(act):
                    pi[n.o.child[a]] = pi[n] * sigma[player_i][n.o.I][i]
    with open(os.path.join(PRINT_OUT, "pi.txt"), "w") as f:
        for l in node_layers:
            for n in l:
                print(n, pi[n], file=f)

    cfv: dict[NodePtr, np.ndarray] = {}
    # for leaves:

    # for other nodes:
