import gc
import os
import re
import time
import itertools
import numpy as np
from gpt import Model
import tensorflow as tf
from collections import defaultdict


# cuts the strand so that we don't end it in the middle of a word
def cleanup_strand(strand, trim="start"):
    """
    Remove anything after the last space (trim = "end"), or anything before the
    first space (trim = "start"), then apply a regex to replace tabs/newlines
    or multiple spaces with just one.
    """
    if trim == "start":
        return re.sub(r"[\t\n\s]+", " ", strand)[strand.find(" ") :]
    elif trim == "end":
        return re.sub(r"[\t\n\s]+", " ", strand)[: strand.rfind(" ")]


def generate_ngrams(strand, cut_indices, n, verbose=False):
    """
    Using the indices of the separator, collect all n-grams for the given
    strand.
    """
    ngrams = []
    begin_ends = []

    for i in range(len(cut_indices) - n):
        begin = cut_indices[i] + 1  # (start index after space)
        end = cut_indices[i + n]  # (end index at space)
        n_gram = strand[begin:end]
        ngrams.append(n_gram)
        begin_ends.append((begin, end))

    if verbose:
        print_sep()
        underprint("generating n grams")
        print(strand)
        for n_gram, indz in zip(ngrams, begin_ends):
            print(f" - {indz}: '{n_gram}'")

    return {"ngrams": ngrams, "begin_ends": begin_ends}


def generate(
    fw_model, bw_model, prefix, suffix, length=50, rev_mode="tokens", verbose=False
):

    prefix_end = len(prefix)
    suffix_end = len(suffix)
    fw_tokens, _, scores, _ = fw_model.run(prefix=prefix, length=length)

    # cut at the last space
    fw_strands = [
        cleanup_strand(fw_strand, trim="end")
        for fw_strand in fw_model.decode(fw_tokens)
    ]

    if rev_mode == "tokens":
        # generate the tokens, then reverse them before decoding & clean-up
        bw_tokens, _, _, _ = bw_model.run(prefix=suffix, length=length, reverse=True)
        bw_strands = [
            cleanup_strand(strand_rev, trim="start")
            for strand_rev in bw_model.decode(bw_tokens[:, ::-1])
        ]
    elif rev_mode == "letters":
        # generate straight in string format and reverse before cleaning
        bw_strands_rev = bw_model.gen(prefix=suffix, length=length)
        bw_strands = [
            cleanup_strand(strand_rev, trim="end")[::-1]
            for strand_rev in bw_strands_rev
        ]

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

    if verbose:
        print_sep()
        underprint("fw strands & cuts:")
        print_strands(fw_strands, all_fw_cut_indices)
        print_sep()
        underprint("bw strands & cuts:")
        print_strands(bw_strands, all_bw_cut_indices)
        exit()

    return fw_strands, all_fw_cut_indices, bw_strands, all_bw_cut_indices


# ----------------------------------------------------------------------------------------
# concat


def generate_concat_bridges(
    fw_model, bw_model, prefix, suffix, length=50, rev_mode="tokens", write=True,
):

    print_sep()
    underprint("generating with concatenation")

    fw_strands, all_fw_cut_indices, bw_strands, all_bw_cut_indices = generate(
        fw_model, bw_model, prefix, suffix, length=length, rev_mode=rev_mode
    )

    # returns a (fairly long) list of possible bridges, that we will then evaluate
    # through their forward likelihood
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
    possible_bridges = np.array(list(set(possible_bridges)))
    print(f"found: {count} possible bridges...")
    print()

    mode = "meanmin"
    perps = fw_model.get_perplexity(
        possible_bridges[:, 0], verbose=True, mode=mode, batched=True
    )

    sorted_indz = perps.argsort()
    sorted_perps = perps[sorted_indz]
    sorted_bridges = possible_bridges[sorted_indz, 1]

    print_sep()
    underprint("now the results sorted:")
    for sentence, perp in zip(sorted_bridges, sorted_perps):
        stat = f"perp: {perp:21.17f} | {sentence}"
        print(stat)

    if write:
        if not os.path.isdir("results"):
            os.mkdir("results")
        fname = os.path.join("results", time.strftime(f"%Y-%m-%d-%H:%M:%S-{mode}.txt"))
        with open(fname, "w") as o:
            o.write(stat + "\n")
        print_sep()
        print(f"written results to {fname}")


# ----------------------------------------------------------------------------------------
# overlap


def generate_overlap_bridges(
    fw_model,
    bw_model,
    prefix,
    suffix,
    length=50,
    ngrams=2,
    rev_mode="tokens",
    verbose=False,
    write=False,
):

    print_sep()
    underprint("generating with overlaps")

    fw_strands, all_fw_cut_indices, bw_strands, all_bw_cut_indices = generate(
        fw_model, bw_model, prefix, suffix, length=length, rev_mode=rev_mode
    )

    # underprint("strands:")
    # print(*fw_strands, sep="\n\n")
    # print_sep()
    # print(*bw_strands, sep="\n\n")

    fw_data = defaultdict(dict)
    bw_data = defaultdict(dict)

    # gather all ngrams in sets
    all_fw_ngrams, all_bw_ngrams = set(), set()

    for (fw_strand, fw_cut_indices) in zip(fw_strands, all_fw_cut_indices):
        fw_data[fw_strand] = generate_ngrams(fw_strand, fw_cut_indices, n=ngrams)
        all_fw_ngrams.update(fw_data[fw_strand]["ngrams"])

    for (bw_strand, bw_cut_indices) in zip(bw_strands, all_bw_cut_indices):
        bw_data[bw_strand] = generate_ngrams(bw_strand, bw_cut_indices, n=ngrams)
        all_bw_ngrams.update(bw_data[bw_strand]["ngrams"])

    # are there any matches?
    fw_bw_intersection = all_fw_ngrams.intersection(all_bw_ngrams)

    while not fw_bw_intersection:
        print(
            f"({len(all_fw_ngrams)} fw & {len(all_bw_ngrams)} bw ngrams, {len(fw_data.keys())} total strands, yet no {ngrams}-intersections: generating MOA.)",
            end="\r"
        )
        _fw_strands, _all_fw_cut_indices, _bw_strands, _all_bw_cut_indices = generate(
            fw_model, bw_model, prefix, suffix, length=length, rev_mode=rev_mode
        )
        for (fw_strand, fw_cut_indices) in zip(_fw_strands, _all_fw_cut_indices):
            fw_data[fw_strand] = generate_ngrams(fw_strand, fw_cut_indices, n=ngrams)
            all_fw_ngrams.update(fw_data[fw_strand]["ngrams"])
        for (bw_strand, bw_cut_indices) in zip(_bw_strands, _all_bw_cut_indices):
            bw_data[bw_strand] = generate_ngrams(bw_strand, bw_cut_indices, n=ngrams)
            all_bw_ngrams.update(bw_data[bw_strand]["ngrams"])
        fw_bw_intersection = all_fw_ngrams.intersection(all_bw_ngrams)

    print()
    print(f"Aha! found {len(fw_bw_intersection)} intersection...")
    print_sep()

    useful_fw_substrands = defaultdict(list)
    useful_bw_substrands = defaultdict(list)

    for strand, strand_data in fw_data.items():
        for str_n_gram, str_begin_end in zip(
            strand_data["ngrams"], strand_data["begin_ends"]
        ):
            if str_n_gram in fw_bw_intersection:
                begin, end = str_begin_end
                useful_fw_substrand = strand[:begin]  # up to the n_gram but omitting it
                useful_fw_substrands[str_n_gram].append(useful_fw_substrand)

    for strand, strand_data in bw_data.items():
        for str_n_gram, str_begin_end in zip(
            strand_data["ngrams"], strand_data["begin_ends"]
        ):
            if str_n_gram in fw_bw_intersection:
                begin, end = str_begin_end
                useful_bw_substrand = strand[end:]  # from n_gram onward but omitting it
                useful_bw_substrands[str_n_gram].append(useful_bw_substrand)

    if verbose:
        print()
        underprint("intersections:")
        for inters in fw_bw_intersection:
            print(f"  '{inters}':")
            print()
            underprint("useful fw:", offset="    ")
            for useful_fw in useful_fw_substrands[inters]:
                print(f"    - {useful_fw}{inters}")
            print()
            underprint("useful bw:", offset="    ")
            for useful_bw in useful_bw_substrands[inters]:
                print(f"    - {inters}{useful_bw}")
            print()

    bridges = set()

    for n_gram in fw_bw_intersection:
        n_gram_useful_fw_substrands = useful_fw_substrands[n_gram]
        n_gram_useful_bw_substrands = useful_bw_substrands[n_gram]

        # cartesian product: [1,2] with [a,b] -> [1,a],[1,b],[2,a],[2,b]
        fw_bw_pairs = itertools.product(
            n_gram_useful_fw_substrands, n_gram_useful_bw_substrands
        )
        bridges.update(
            [fw + "\033[0;31m" + n_gram + "\033[0m" + bw for (fw, bw) in fw_bw_pairs]
        )

    print()
    underprint("les bridges:")
    for bridge in bridges:
        print(bridge)
        print("----")
    print()

    if write:
        if not os.path.isdir("overlaps"):
            os.mkdir("overlaps")
        fname = os.path.join("overlaps", time.strftime(f"%Y-%m-%d-%H:%M:%S.txt"))
        print("printing results to {fname}")
        with open(fname, "w") as o:
            for bridge in bridges:
                o.write(bridge + "\n")


# ----------------------------------------------------------------------------------------
# print


def print_strands(all_strands, all_indices):
    for indices, strand in zip(all_indices, all_strands):
        for i in indices:
            print(f"index: {i}")
            print(strand[:i], "|", strand[i:])
            print()


def underprint(s, offset=None):
    und = "-" * len(s)
    if offset:
        s = offset + s
        und = offset + und
    print(s)
    print(und)


def print_sep():
    print("-" * 40)


if __name__ == "__main__":

    """
    two learning modes:
        - tokens: the model has learnt on reversed token sequences
        - letters: the model has learnt on reversed strings
    """

    rev_mode = "tokens"
    # rev_mode = "letters"

    if len(tf.config.experimental.list_physical_devices("GPU")) > 1:
        print_sep()
        print("More than one GPU, attempting dual use.")
        run_name_fw = "forward"
        run_name_bw = "backward"
        device_fw = "GPU:0"
        device_bw = "GPU:1"
    else:
        run_name_fw = "run1"
        run_name_bw = "117Mr"
        device_fw = "GPU:0"
        device_bw = "GPU:0"

    length_desired = 50
    ngrams_desired = 4
    batch_size = 3

    # model that generates and computes the logits for the forward
    # prediction of the tokens
    fw_model = Model(run_name=run_name_fw, batch_size=batch_size, device=device_fw)
    # same as forwaerts, but for the backward prediction of the tokens
    # (trained on a dataset where all chars have been reverted)
    bw_model = Model(run_name=run_name_bw, batch_size=batch_size, device=device_bw)

    prefix = "Il était une fois une princesse qui"

    if rev_mode == "tokens":
        suffix = "heureux et eurent beaucoup d'enfants."
    if rev_mode == "letters":
        suffix = "heureux et eurent beaucoup d'enfants."[::-1]

    generate_concat_bridges(
        fw_model,
        bw_model,
        prefix,
        suffix,
        length=length_desired,
        rev_mode=rev_mode,
        write=False,
    )

    # generate_overlap_bridges(
    #     fw_model,
    #     bw_model,
    #     prefix,
    #     suffix,
    #     length=length_desired,
    #     ngrams=ngrams_desired,
    #     rev_mode=rev_mode,
    #     write=False,
    # )
