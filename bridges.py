import os
import re
import time
import itertools
import numpy as np
from gpt import Model
from collections import defaultdict

# model that generates and computes the logits for the forward
# prediction of the tokens
fw_model = Model(run_name="forward", batch_size=10)
# same as forwaerts, but for the backward prediction of the tokens
# (trained on a dataset where all chars have been reverted)
bw_model = Model(run_name="backward", batch_size=10)

prefix = "Aha ! À nous les ponts !"
prefix_end = len(prefix)
suffix = "Et après ces travaux ils virent que les ponts étaient bons."[::-1]
suffix_end = len(suffix)

fw_tokens, _, scores, _ = fw_model.run(prefix=prefix, length=500)
# the backwards strands are generated backwards
bw_strands_rev = bw_model.gen(prefix=suffix, length=500)

# cuts the strand so that we don't end it in the middle of a word
def cleanup_strand(strand):
    return re.sub(r"[\t\n\s]+", " ", strand[: strand.rfind(" ")])


# cut at the last space
fw_strands = [cleanup_strand(fw_strand) for fw_strand in fw_model.decode(fw_tokens)]
bw_strands = [cleanup_strand(strand_rev)[::-1] for strand_rev in bw_strands_rev]

# these are the locations where we may want to cut the strands for recombinations
pattern = r"\s"
all_fw_cut_indices = [
    [
        prefix_end + match.start()
        for match in re.finditer(pattern, fw_strand[prefix_end:])
    ]
    for fw_strand in fw_strands
]
all_bw_cut_indices = [
    [match.start() for match in re.finditer(pattern, bw_strand[:-suffix_end])]
    for bw_strand in bw_strands
]

# for indices,strand in zip(all_fw_cut_indices, fw_strands):
#     for i in indices:
#         print(i)
#         print(strand)
#         print(strand[:i], "|", strand[i:])
#         print()

# print("------")

# for indices,strand in zip(all_bw_cut_indices, bw_strands):
#     for i in indices:
#         print(i)
#         print(strand)
#         print(strand[:i], "|", strand[i:])
#         print()

# returns a (fairly long) list of possible bridges, that we will then evaluate
# through their forward likelihood
def generate_possible_bridges(fw_strands, bw_strands):
    print("inside the possibilities")
    count = 0
    possible_bridges = []
    for (fw_strand, fw_cut_indices) in zip(fw_strands, all_fw_cut_indices):
        for fw_index in fw_cut_indices:
            for (bw_strand, bw_cut_indices) in zip(bw_strands, all_bw_cut_indices):
                for bw_index in bw_cut_indices:
                    # riddance of space in fw_strand, kept in bw_strand
                    possible_bridge = fw_strand[:fw_index] + bw_strand[bw_index:]
                    possible_bridge_cut = (
                        fw_strand[:fw_index] + " | " + bw_strand[bw_index:]
                    )
                    possible_bridges.append((possible_bridge, possible_bridge_cut))
                    count += 1
    print("-" * 40)
    print(f"count: {count}")
    print()
    return np.array(list(set(possible_bridges)))


def generate_final_n_grams(fw_strand, fw_cut_indices, n):
    n_grams = []
    begin_ends = []

    fw_cut_indices = fw_cut_indices + [len(fw_strand)]  # adds the end of the fw_strand
    for i in range(len(fw_cut_indices) - n):
        n_gram_begin = fw_cut_indices[i] + 1  # (included in the n-gram)
        n_gram_end = fw_cut_indices[i + n]  # (not included in the n-gram)
        n_gram = fw_strand[n_gram_begin:n_gram_end]
        n_grams.append(n_gram)
        begin_ends.append((n_gram_begin, n_gram_end))
    return (n_grams, begin_ends)


def generate_initial_n_grams(bw_strand, bw_cut_indices, n):
    n_grams = []
    begin_ends = []
    # adds the beginning of the bw_strand (-1 gets added with the +1 in n_gram_begin for i=0
    bw_cut_indices = [-1] + bw_cut_indices
    for i in range(len(bw_cut_indices) - n):
        n_gram_begin = bw_cut_indices[i] + 1  # (included in the n-gram)
        n_gram_end = bw_cut_indices[i + n]  # (not included in the n-gram)
        n_gram = bw_strand[n_gram_begin:n_gram_end]
        n_grams.append(n_gram)
        begin_ends.append((n_gram_begin, n_gram_end))
    return (n_grams, begin_ends)


# print(generate_final_n_grams("bonjour monsieur le prince machiavelique"), [7, 16, 19, 26], 2))


def generate_overlap_bridges(
    fw_strands, fw_cut_indices_lists, bw_strands, bw_cut_indices_lists, n
):
    fw_set = set()
    bw_set = set()
    for (fw_strand, fw_cut_indices) in zip(fw_strands, fw_cut_indices_lists):
        fw_set.update(generate_final_n_grams(fw_strand, fw_cut_indices, n)[0])
    bw_n_grams_lists = []
    for (bw_strand, bw_cut_indices) in zip(bw_strands, bw_cut_indices_lists):
        bw_set.update(generate_initial_n_grams(bw_strand, bw_cut_indices, n)[0])

    fw_bw_intersection = fw_set.intersection(bw_set)
    print(fw_bw_intersection)

    useful_fw_substrands = defaultdict(list)
    useful_bw_substrands = defaultdict(list)

    for (fw_strand, fw_cut_indices) in zip(fw_strands, fw_cut_indices_lists):
        final_n_grams, begin_ends = generate_final_n_grams(fw_strand, fw_cut_indices, n)
        for (final_n_gram, begin_end) in zip(final_n_grams, begin_ends):
            if final_n_gram in fw_bw_intersection:
                begin, end = begin_end
                useful_fw_substrand = fw_strand[:begin]
                useful_fw_substrands[final_n_gram].append(useful_fw_substrand)

    print("useful fw")
    print(useful_fw_substrands)

    for (bw_strand, bw_cut_indices) in zip(bw_strands, bw_cut_indices_lists):
        initial_n_grams, begin_ends = generate_initial_n_grams(
            bw_strand, bw_cut_indices, n
        )
        for (initial_n_gram, begin_end) in zip(initial_n_grams, begin_ends):
            if initial_n_gram in fw_bw_intersection:
                begin, end = begin_end
                useful_bw_substrand = bw_strand[end:]
                useful_bw_substrands[initial_n_gram].append(useful_bw_substrand)

    print("useful bw")
    print(useful_bw_substrands)

    overlap_bridges = set()

    for n_gram in fw_bw_intersection:
        n_gram_useful_fw_substrands = useful_fw_substrands[n_gram]
        n_gram_useful_bw_substrands = useful_bw_substrands[n_gram]

        fw_bw_pairs = itertools.product(
            n_gram_useful_fw_substrands, n_gram_useful_bw_substrands
        )
        overlap_bridges.update([fw + "\033[0;31m" + n_gram + "\033[0m" + bw for (fw, bw) in fw_bw_pairs])

    return overlap_bridges


# fw_strands = ["a b c d e f g x", "x z f g k p q", "c a d f g h"]
# bw_strands = ['h i j f g x k c d l m n g d', 'h g g i a b j k l m o p', 'a a h i j s d m', 'a d d c d e f']

# print('les strands')
# print('fw:')
# print(fw_strands)
# print()
# print('bw')
# print(bw_strands)
# prefix_end = 3
# suffix_end = 7

# pattern = r"\s"

# all_fw_cut_indices = [
#     [prefix_end + match.start() for match in re.finditer(pattern, fw_strand[prefix_end:])]
#     for fw_strand in fw_strands
# ]
# all_bw_cut_indices = [
#     [match.start() for match in re.finditer(pattern, bw_strand[:-suffix_end])]
#     for bw_strand in bw_strands
# ]

n = 15
bridges = generate_overlap_bridges(
    fw_strands, all_fw_cut_indices, bw_strands, all_bw_cut_indices, n
)
while not bridges:
    print(f"found no bridge for n = {n}, retrying with {n-1}")
    n = n - 1
    bridges = generate_overlap_bridges(
        fw_strands, all_fw_cut_indices, bw_strands, all_bw_cut_indices, n
    )


if not os.path.isdir("overlaps"):
    os.mkdir("overlaps")
fname = os.path.join("overlaps", time.strftime(f"%Y-%m-%d-%H:%M:%S.txt"))
with open(fname, "w") as o:
    for bridge in bridges:
        print("----")
        print(bridge)
        o.write(bridge + "\n")
        print()

    # select the fw_strand parts that gave something to the intersection
    # select the bw_strand parts that gave something to the intersection

    # if fw_bw_intersection:
    #     for fw_bw_elem in list(fw_set_intersection):
    #         eligible_fw_substrands = []
    #         for fw_strand in fw_strands:
    #             fw_final_n_grams = set(generate_final_n_grams(fw_strand))

    #             pass
    #         eligible_bw_substrands = []
    #         for bw_strand in bw_strands:
    #             pass
    #     return fw_strand, bw_strand


# print("forward strands:")
# print(fw_strands)
# print("-" * 40)
# print("backward strands:")
# print(bw_strands)
# print("-" * 40)

# possible_bridges = generate_possible_bridges(fw_strands, bw_strands)
# mode = "meanmin"
# perps = fw_model.get_perplexity(possible_bridges[:, 0], verbose=True, mode=mode)
# print("-" * 40)
# print()

# sorted_indz = perps.argsort()
# sorted_perps = perps[sorted_indz]
# sorted_bridges = possible_bridges[sorted_indz, 1]

# if not os.path.isdir("results"):
#     os.mkdir("results")
# fname = os.path.join("results", time.strftime(f"%Y-%m-%d-%H:%M:%S-{mode}.txt"))
# with open(fname, "w") as o:
#     print("now the results sorted:")
#     for sentence, perp in zip(sorted_bridges, sorted_perps):
#         stat = f"perp: {perp:21.17f} | {sentence}"
#         print(stat)
#         o.write(stat + "\n")
# print("-" * 40)
# print(f"written results to {fname}")
