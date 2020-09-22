from tensorflow.core.protobuf import rewriter_config_pb2
from collections import defaultdict
from operator import itemgetter
from print_utils import term
import tensorflow as tf
import numpy as np
import encoder_hug
import encoder
import random
import model
import regex
import json
import sys
import os
import gc

# PYTHONPATH=src python bridges.py

# disabling some warnings
os.environ["KMP_WARNINGS"] = "off"


class Model:
    def __init__(
        self,
        model_name="frfw117",
        run_name="run1",
        device="/GPU:0",
        batch_size=1,
        encoder_type="default",
        special_tokens=["<|s|>", "<|e|>", "<|endoftext|>"],
        reverse=False,
    ):
        tf.compat.v1.reset_default_graph()
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.rewrite_options.layout_optimizer = (
            rewriter_config_pb2.RewriterConfig.OFF
        )
        self.sess = tf.compat.v1.Session(config=self.config)
        if encoder_type == "hug":
            self.enc = encoder_hug.get_encoder(
                model_name, "models", special_tokens=special_tokens
            )
            self.encoder_type = "hug"
            self.encoder = self.enc.tok.get_vocab()
            self.decoder = {v: k for k, v in self.encoder}
        else:
            self.enc = encoder.get_encoder(
                model_name, "models", special_tokens=special_tokens
            )
            self.encoder_type = "default"
            self.encoder = self.enc.encoder
            self.decoder = self.enc.decoder
        self.special_tokens = set(special_tokens)
        self.hparams = model.default_hparams()
        with open(f"models/{model_name}/hparams.json") as f:
            self.hparams.override_from_dict(json.load(f))
        self.hparams.add_hparam("precision", tf.float32)
        self.batch_size = batch_size
        self.context = tf.compat.v1.placeholder(
            tf.int32, [self.batch_size, None], name="context"
        )
        self.length = tf.compat.v1.placeholder(tf.int32, (), name="length")
        self.temperature = tf.compat.v1.placeholder(tf.float32, (), name="temperature")
        self.top_k = tf.compat.v1.placeholder(tf.int32, (), name="top_k")
        self.top_p = tf.compat.v1.placeholder(tf.float32, (), name="top_p")
        self.lgt = tf.compat.v1.placeholder(
            tf.float32, [None, len(self.encoder)], name="lgt"
        )
        self.softmax = tf.nn.softmax(self.lgt, axis=-1, name="final_softmax")
        self.reverse = reverse
        self.model = model.model
        # required to load checkpoint
        self.model(hparams=self.hparams, X=self.context)
        self.load_checkpoint(f"checkpoint/{run_name}")
        # sampling the network, self.output will be used throughout
        with tf.device(device):
            self.output = self.sample(
                length=self.length,
                context=self.context,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        # preemptively spit out all these warnrings
        self.dummy_run()

    # --------------------------------------------------------------------------------
    # Main uses:
    # - generate text
    # - generate text & logits

    def print(
        self,
        prefix="\n",
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        print_tokens=False,
    ):

        """# {{{
        Highest level function, simply prints generated texts. Invokes self.gen().
        Parameters:
        -----------
        prefix: string or list of list/np.arrays of tokens. If a string is
            passed, it will be used as a prefix for all batch_size generated sequences.
            When passing a list of lists/np.arrays of tokens (encoded text),
            each generated sequence will have its own prefix, and the number of sequences
            generated (the batch size) will be adjusted to match the number of
            given parallel prefixes.
        length: number of tokens to be generated (not string letters). Default: 5.
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        batch_size: int. Batch size, number of sequences produced in
            parallel. Will be overridden by the number of given sequences if
            not passing a string as prefix.
        print_tokens:
            boolean. Print raw tokens instead of decoding them to strings.
        """# }}}

        for seq in self.gen(
            prefix=prefix,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=batch_size,
        )["sequences"]:
            print("-" * 40)
            print(seq)
        print("-" * 40)

    def gen(
        self, prefix="\n", past=None, length=5, temperature=1, top_k=0, top_p=0.0, batch_size=None,
    ):

        """# {{{
        Higher level generation: input a sentence, get an array with n batches
        of continuations.
        Parameters:
        -----------
        prefix: string or list of list/np.arrays of tokens. If a string is
            passed, it will be used as a prefix for all batch_size generated sequences.
            When passing a list of lists/np.arrays of tokens (encoded text),
            each generated sequence will have its own prefix, and the number of sequences
            generated (the batch size) will be adjusted to match the number of
            given parallel prefixes.
        length: number of tokens to be generated (not string letters). Default: 5.
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        batch_size: int. Batch size, number of sequences produced in
            parallel. Will be overridden by the number of given sequences if
            not passing a string as prefix.
        return_tokens:
            boolean. Return tokens instead of decoding them to strings.
        Returns:
        --------
        a dictionary containig:
            sequences: the decoded generated sequences.
            tokens: the generated tokens.
            logits: all the scores for all tokens at each step.
        """# }}}

        context_tkns = self._check_prefix(prefix, batch_size)["context_tkns"]
        if past is None:
            results = self.sess.run(
                self.output,
                feed_dict={
                    self.length: length,
                    self.context: context_tkns,
                    self.temperature: temperature,
                    self.top_k: top_k,
                    self.top_p: top_p,
                },
            )
        else:
            results = self.sess.run(
                self.output,
                feed_dict={
                    self.length: length,
                    self.context: context_tkns,
                    self.past: past,
                    self.temperature: temperature,
                    self.top_k: top_k,
                    self.top_p: top_p,
                },
            )
        tkns, logits, past = itemgetter("tokens", "logits", "past")(results)
        return {
            "sequences": self.decode(tkns)
            if not self.reverse
            else self.decode(tkns[:, ::-1]),
            "tokens": tkns if not self.reverse else tkns[:, ::-1],
            "logits": logits if not self.reverse else logits[:, ::-1],
            "past": past,
        }

    def gen_until(
        self,
        prefix="\n",
        until="<|e|>",
        exclude_until=True,
        chunk_length=5,
        sanity_limit=100,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
    ):

        """# {{{
        Generate sequences until a special token (e.g. "<|endoftext|>") or a
        regex is found. The resulting sequences will not be of the same length.
        Beware, when using the string version of until, not to
        forget to escape regex-like chars (for instance '.' means any
        character, and '\.' a literal dot). The module used is the `regex`
        module (not `re`), which is a superset of 're' allowing for Perl/utf-8
        syntax.
        Parameters:
        -----------
        prefix: string or list of list/np.arrays of tokens. If a string is
            passed, it will be used as a prefix for all batch_size generated sequences.
            When passing a list of lists/np.arrays of tokens (encoded text),
            each generated sequence will have its own prefix, and the number of sequences
            generated (the batch size) will be adjusted to match the number of
            given parallel prefixes.
        until: the special token or regex to find. Defaults to '<|e|>'.
        exclude_until:  boolean. Whether to return the sequences with or without the
            searched item. Default: True (do not include until).
        chunk_length: int. Number of tokens to be generated in the loop. Default: 5.
        sanity_limit: int. Guarantee that the generation loop is interrupted after this
            number of iterations. Default: 200.
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        batch_size: int. Batch size, number of sequences produced in
            parallel. Will be overridden by the number of given sequences if
            not passing a string as prefix.
        return_tokens:
            boolean. Return tokens instead of decoding them to strings.
        Returns:
        --------
        if the searched string is a special token:
            a dictionary containig:
                sequences: the decoded generated sequences. A list of strings
                    of various lengths.
                tokens: the generated tokens. A list of np.arrays of various
                    lengths.
                logits: all the scores for all tokens at each step. A list of
                    np.arrays of various lengths n_sequences * (len_seq, n_vocab).
                logprobs: the normalized logits (after softmax). A list of
                    np.arrays of various lengths n_sequences * (len_seq, n_vocab).
                perplexities: the perplexity for each sentence, np.array shape: (n_sequences, 1)
                scores: sequence of logits (scores, unnormalized) for each sequence
                        shape: (batch_size, n_tokens)
                scores_min:  the min of scores, shape: (batch_size, 1)
                scores_max: the max of scores, shape: (batch_size, 1)
                scores_range: the range of scores, shape: (batch_size, 1)
                scores_mean: the mean of scores, shape: (batch_size, 1)
                scores_std: the standard deviation of scores, shape: (batch_size, 1)
        if the searched string is a regex:
            a dictionary containing:
                sequences: the generated sequences found.
        NOTE: as the results differ in length, the return type will be pure
        Python lists, and not numpy arrays.
        """# }}}

        pref_data = self._check_prefix(prefix, batch_size)
        prefix, prefix_enc, context_tkns = itemgetter(
            "prefix", "prefix_enc", "context_tkns"
        )(pref_data)
        if until in self.special_tokens:
            until = self.encode(until)[0]
            use_regex = False
        else:
            rr = regex.compile(regex.escape(until))
            use_regex = True
        if not use_regex:
            length_prefix = len(context_tkns[0])
            batch_data = [
                {"previous_length": length_prefix, "index": None, "seq": p,}
                for p in context_tkns
            ]
            i = 0
            while i < sanity_limit and not all(
                s["index"] is not None for s in batch_data
            ):
                tkns, logits = self.sess.run(
                    self.output,
                    feed_dict={
                        self.length: chunk_length,
                        self.context: context_tkns,
                        self.temperature: temperature,
                        self.top_k: top_k,
                        self.top_p: top_p,
                    },
                )
                batch_data = self._find_token(
                    until,
                    tkns,
                    batch_data=batch_data,
                    chunk_length=chunk_length,
                    exclude_until=exclude_until,
                )
                msg = f"\t\t{tkns[:, -chunk_length:]}".replace("\n", "")[
                    : term.width - 16
                ]
                print(msg)
                context_tkns = tkns
                i += 1
            tkns = [t[: batch_data[i]["index"]] for i, t in enumerate(tkns)]
            tkns = tkns if not self.reverse else [t[::-1] for t in tkns]
            logits = [l[: batch_data[i]["index"]] for i, l in enumerate(logits)]
            logits = logits if not self.reverse else [l[::-1] for l in logits]
            logprobs = []
            perplexities = []
            scores = []
            stats = []
            for lgt, tkn in zip(logits, tkns):
                lgpr = self.sess.run(self.softmax, feed_dict={self.lgt: lgt})
                tkn = tkn[1:] if len(tkn) > 1 else tkn
                scrs = np.nan_to_num([(lgpr[i, t]) for i, t in enumerate(tkn)])
                scores.append(scrs)
                perps = self._perplexities(scrs)
                perplexities.append(perps["perplexities"])
                stats.append(
                    {k: v for k, v in perps.items() if k is not "perplexities"}
                )
                logprobs.append(lgpr)

            return {
                "sequences": self.decode(tkns),
                "tokens": tkns,
                "logits": logits,
                "logprobs": logprobs,
                "scores": scores,
                "perplexities": np.array(perplexities),
                **{k: np.stack([st[k] for st in stats]) for k in stats[0].keys()},
            }
        else:
            seqs = (
                self.decode(context_tkns)
                if not self.reverse
                else self.decode(context_tkns[:, ::-1])
            )
            batch_data = [
                {"previous_length": len(seq), "index": None, "seq": seq,}
                for seq in seqs
            ]
            i = 0
            while i < sanity_limit and not all(
                s["index"] is not None for s in batch_data
            ):
                tkns, _ = self.sess.run(
                    self.output,
                    feed_dict={
                        self.length: chunk_length,
                        self.context: context_tkns,
                        self.temperature: temperature,
                        self.top_k: top_k,
                        self.top_p: top_p,
                    },
                )
                batch_data = self._find_regex(
                    until, tkns, batch_data, exclude_until=exclude_until
                )
                context_tkns = tkns
                i += 1
            return {
                "sequences": [s["seq"] for s in batch_data],
            }

    def gen_avoiding(
        self,
        prefix="\n",
        avoiding="<|e|>",
        length=5,
        sanity_limit=200,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
    ):

        """# {{{
        Generate beginnings of sequences avoiding a certain token. Useful when
        wanting continuations despite the fact that a likely outcome from the
        network's perspective is an end token. This function allows to generate
        beginnings that are guaranteed not to contain this particular special
        token or regex, and this batch can then be fed to other functions as
        prefixes.  Beware, when using the string version of
        avoiding, not to forget to escape regex-like chars (for
        instance '.' means any character, and '\.' a literal dot). The module
        used is the `regex` module (not `re`), which is a superset of 're'
        allowing for Perl/utf-8 syntax.
        Parameters:
        -----------
        prefix: string or list of list/np.arrays of tokens. If a string is
            passed, it will be used as a prefix for all batch_size generated sequences.
            When passing a list of lists/np.arrays of tokens (encoded text),
            each generated sequence will have its own prefix, and the number of sequences
            generated (the batch size) will be adjusted to match the number of
            given parallel prefixes.
        avoiding: the special token or regex to avoid.
        length: int. Number of tokens to be generated (not string letters). Default: 5.
        sanity_limit: int. Guarantee that the generation loop is interrupted after this
            number of iterations. Default: 200.
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        batch_size: int. Batch size, number of sequences produced in
            parallel. Will be overridden by the number of given sequences if
            not passing a string as prefix.
        return_tokens:
            boolean. Return tokens instead of decoding them to strings.
        Returns:
        --------
        a dictionary containig:
            sequences: the decoded generated sequences.
            tokens: the generated tokens.
            logits: all the scores for all tokens at each step.
            logprobs: the normalized logits (after softmax).
            perplexities: the perplexity for each sentence, shape: (n_sequences, 1)
            scores: sequence of logits (scores, unnormalized) for each sequence
                    shape: (batch_size, n_tokens)
            scores_min:  the min of scores, shape: (batch_size, 1)
            scores_max: the max of scores, shape: (batch_size, 1)
            scores_range: the range of scores, shape: (batch_size, 1)
            scores_mean: the mean of scores, shape: (batch_size, 1)
            scores_std: the standard deviation of scores, shape: (batch_size, 1)
        """# }}}

        context_tkns = self._check_prefix(prefix, batch_size)["context_tkns"]
        gen_tkns = []
        gen_logits = []
        cond = None
        i = 0
        while not cond:
            tkns, logits = self.sess.run(
                self.output,
                feed_dict={
                    self.length: length,
                    self.context: context_tkns,
                    self.temperature: temperature,
                    self.top_k: top_k,
                    self.top_p: top_p,
                },
            )
            i += 1
            if i > sanity_limit:
                break
            if isinstance(avoiding, str) and avoiding not in self.special_tokens:
                # print("searching for regex")
                generated = (
                    [self.decode(t[-length:]) for t in tkns]
                    if not self.reverse
                    else [self.decode(t[-length:][::-1]) for t in tkns]
                )
                for i, seq in enumerate(generated):
                    if not regex.search(avoiding, seq):
                        gen_tkns.append(tkns[i])
                        gen_logits.append(logits[i])
                        if len(gen_tkns) == self.batch_size:
                            cond = True
                            break
            else:
                # print("searching for token")
                if isinstance(avoiding, str):
                    avoiding = self.encode(avoiding)[0]
                for i, t in enumerate(tkns):
                    if avoiding not in t[-length:]:
                        gen_tkns.append(t)
                        gen_logits.append(logits[i])
                        if len(gen_tkns) == self.batch_size:
                            cond = True
                            break
            temperature += 0.1
        gen_tkns = gen_tkns if not self.reverse else [g[::-1] for g in gen_tkns]
        gen_logits = gen_logits if not self.reverse else [g[::-1] for g in gen_logits]
        logprobs = []
        perplexities = []
        scores = []
        stats = []
        for lgt, tkn in zip(gen_logits, gen_tkns):
            tkn = tkn[1:] if len(tkn) > 1 else tkn
            lgpr = self.sess.run(self.softmax, feed_dict={self.lgt: lgt})
            scrs = np.nan_to_num([(lgpr[i, t]) for i, t in enumerate(tkn)])
            scores.append(scrs)
            perps = self._perplexities(scrs)
            perplexities.append(perps["perplexities"])
            stats.append({k: v for k, v in perps.items() if k is not "perplexities"})
            logprobs.append(lgpr)
        return {
            "sequences": self.decode(gen_tkns)
            if not self.reverse
            else self.decode(gen_tkns[:, ::-1]),
            "tokens": np.array(gen_tkns),
            "logits": np.array(gen_logits),
            "logprobs": np.array(logprobs),
            "scores": np.array(scores),
            "perplexities": np.array(perplexities),
            **{k: np.stack([st[k] for st in stats]) for k in stats[0].keys()},
        }

    def run(
        self,
        prefix="\n",
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        return_perplexities=True,
        return_ranks=True,
    ):

        """# {{{
        Lower level generation: input a sentence, get n batches of generated
        tokens as well as the logits associated with each step.
        Parameters:
        -----------
        prefix: string or list of list/np.arrays of tokens. If a string is
            passed, it will be used as a prefix for all batch_size generated sequences.
            When passing a list of lists/np.arrays of tokens (encoded text),
            each generated sequence will have its own prefix, and the number of sequences
            generated (the batch size) will be adjusted to match the number of
            given parallel prefixes.
        length: number of tokens to be generated (not string letters). Default: 5.
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        batch_size: int. Batch size, number of sequences produced in
            parallel. Will be overridden by the number of given sequences if
            not passing a string as prefix.
        return_perplexities: boolean. Calculate the scores (logits assigned to
            tokens at each steps, that can be normalised into probabilities),
            use them to calculate the perplexities of the produced sequences,
            and return those as well as a dict with various statistics (min,
            max, range, mean, std) see self._stats().
        return_ranks: boolean. Calculate the ranks (ranks of the logits
            assigned to tokens at each steps: 0 for the most probable, then 1,
            all the way until n_vocab.), and return those as well as a dict
            with various statistics (min, max, range, mean, std) see
            self._stats().
        Returns:
        --------
        a dictionary containg:
            tokens: the produced batch of machine-readable subwords
                    shape: (batch_size, length)
            logits: the scores for the next token at each step
                    shape: (batch_size, n_tokens - 1, n_vocab)
            if return_perplexities, the dict will also include:
                scores: sequence of logits (scores, unnormalized) for each sequence
                        shape: (batch_size, n_tokens)
                probs: the normalized logits (softmaxed into probabilities)
                        shape: (batch_size, n_tokens - 1, n_vocab)
                perplexities: the perplexity for each sentence, shape: (batch_size, 1)
                scores_min:  the min of scores, shape: (batch_size, 1)
                scores_max: the max of scores, shape: (batch_size, 1)
                scores_range: the range of scores, shape: (batch_size, 1)
                scores_mean: the mean of scores, shape: (batch_size, 1)
                scores_std: the standard deviation of scores, shape: (batch_size, 1)
            if return_ranks, the dict will include the above and the following:
                ranks: sequence of ranks for each sequence, shape: (batch_size, n_tokens)
                ranks_min: the min of ranks, shape: (batch_size, 1)
                ranks_max: the max of ranks, shape: (batch_size, 1)
                ranks_range: the range of ranks, shape: (batch_size, 1)
                ranks_mean: the mean of ranks, shape: (batch_size, 1)
                ranks_std: the standard deviation of ranks, shape: (batch_size, 1)
        """# }}}

        context_tkns = self._check_prefix(prefix, batch_size)["context_tkns"]
        tkns, logits = self.sess.run(
            self.output,
            feed_dict={
                self.length: length,
                self.context: context_tkns,
                self.temperature: temperature,
                self.top_k: top_k,
                self.top_p: top_p,
            },
        )
        data = {
            "sequences": self.decode(tkns)
            if not self.reverse
            else self.decode(tkns[:, ::-1]),
            "tokens": tkns if not self.reverse else tkns[:, ::-1],
            "logits": logits if not self.reverse else logits[:, ::-1],
        }
        if return_perplexities or return_ranks:
            data.update(
                **self._perps_n_ranks(
                    data,
                    return_perplexities=return_perplexities,
                    return_ranks=return_ranks,
                )
            )
        return data

    def print_data(self, data, sort_by="perplexities"):
        assert sort_by in data, f"! Data keys:\n{data.keys()}"
        le_sorting_el = data[sort_by]
        sorted_seqs = data["sequences"][np.argsort(le_sorting_el.flatten())]
        sorted_stats = np.sort(le_sorting_el.flatten())
        for i, seq in enumerate(sorted_seqs):
            print(seq)
            print()
            msg = "stats:"
            sub = "-" * len(msg)
            print(f"\t\t{msg}")
            print(f"\t\t{sub}")
            print(f"\t\t{sort_by}: {sorted_stats[i]}")
            print()
            print("-" * 40)

    # --------------------------------------------------------------------------------
    # encoder/decoder utils

    def encode(self, s):
        """
        Encode a string, or an array of strings, into an array (or a batch) of
        machine-readable subwords.
        """
        if isinstance(s, str):
            return self.enc.encode(s)
        elif isinstance(s, (list, tuple, np.ndarray)):
            np.array([np.array(self.enc.encode(ss)) for ss in s], dtype=np.int32)

    def decode(self, s):
        """
        Decode an array (or a batch) of machine-readable subwords into an array
        of string(s).
        """
        if isinstance(s[0], (int, np.integer)):
            return self.enc.decode(s)
        elif isinstance(s, (list, tuple, np.ndarray)):
            return np.array([self.enc.decode(ss) for ss in s])

    def pad_sequences(
        self,
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",  # both pre
        truncating="pre",  # or post
        value=None,
    ):
        """
        Given an array of strings or tokens (variable lengths), will return a
        square matrix of padded tokens (equal lengths). Options include:
        padding 'pre' or 'post', truncating, according to maxlen, also
        'pre' or 'post', and value, that which one pads with (if None, the
        current encoder's token for space will be used).
        """
        # The pad value, space, often 220 in gpt-2
        if value is None:
            value = self.enc.encode(" ")[0]
        return tf.keras.preprocessing.sequence.pad_sequences(
            self.encode(sequences),
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
            value=value,
        )

    # --------------------------------------------------------------------------------
    # Plumbing:
    # ---------
    # - reset: change batch_size and other options
    #          (batch_size is reset automatically in most functions)
    # - load checkpoint
    # - change hparams
    # - check & reset graph with new batch size
    # - dummy run to clear out messages
    # - clear line print helper

    def reset(
        self, hparams_file=None, device="/GPU:0", batch_size=1, top_k=0.0, top_p=0.0
    ):
        """
        Resetting the network (params, batch_size, device).
        """
        self._check_hparams(hparams_file)
        self.batch_size = batch_size
        self.length = tf.compat.v1.placeholder(tf.int32, (), name="length")
        self.context = tf.compat.v1.placeholder(
            tf.int32, [self.batch_size, None], name="context"
        )
        self.model = model.model(hparams=self.hparams, X=self.context)
        with tf.device(device):
            self.output = self.sample(
                length=self.length,
                context=self.context,
                temperature=self.temperature,
                top_k=top_k,
                top_p=top_p,
            )

    def load_checkpoint(self, path="checkpoint/run1"):
        """
        Load a checkpoint manually.
        """
        self.ckpt = tf.train.latest_checkpoint(path)
        self.saver = tf.compat.v1.train.Saver(allow_empty=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print("-" * 40)
        print(f"loading checkpoint {self.ckpt}")
        self.saver.restore(self.sess, self.ckpt)


    def _check_hparams(self, hparams_file):
        """
        Loading params from file if it exists.
        Parameters:
        -----------
            - hparams_file: path to hparams json file.
        """
        if hparams_file is not None:
            print(f"reloading hparams from file {hparams_file}")
            with open(hparams_file) as f:
                self.hparams.override_from_dict(json.load(f))

    def _check_batch_size(self, batch_size, verbose=True):
        """
        Returns self.batch_size if batch_size is None.
        Else runs reset() to redraw the graph with a new batch_size.
        """
        if batch_size is None:
            batch_size = self.batch_size
        else:
            if batch_size != self.batch_size:
                if verbose:
                    print(
                        f"(Batch size changed from {self.batch_size} to {batch_size}, resetting graph.)"
                    )
                self.reset(batch_size=batch_size)

    def _check_prefix(self, prefix, batch_size):

        """# {{{
        Check whether prefix is a string or a list of tokens (for parallel
        generation). If passing tokens as prefix all sequences must have the
        same length. This will set the batch size according to either:
            the parameters given to the function calling this one
            (gen(), gen_until(), etc.), or
            the number of prefix token sequences given.
        Parameters:
        -----------
        prefix: string or list of lists/np arrays of tokens
        batch_size: int, setting the batch size for the case where the
            prefix is a string
        Returns:
        --------
        a dictionary containing:
            prefix: the prefix (or batch thereof, if different) as a
                list of strings
            prefix_enc: the encoded prefix (or batch thereof, if
                different)
            context_tkns: the context tokens, a batch of shape (batch_size, n_tokens),
                ready to be fed to the network
        """# }}}

        if isinstance(prefix, np.ndarray):
            if isinstance(prefix[0], np.integer):
                self._check_batch_size(batch_size)
                prefix_enc = prefix if not self.reverse else prefix_enc[:, ::-1]
                prefix = self.decode(prefix)
                context_tkns = self.batch_size * [prefix_enc]
            elif isinstance(prefix[0], np.ndarray):
                self._check_batch_size(prefix.shape[0])
                prefix_enc = prefix if not self.reverse else prefix_enc[:, ::-1]
                prefix = self.decode(prefix)
                context_tkns = prefix_enc
            else:
                raise TypeError(
                    """
                    When passing a nd.array as prefix, make sure it contains
                    either tokens (ints) or sequences of tokens (nd.arrays) of
                    ints.
                    """
                )
        elif isinstance(prefix, list):
            if isinstance(prefix[0], int):
                self._check_batch_size(batch_size)
                prefix_enc = prefix if not self.reverse else prefix[::-1]
                prefix = self.decode(prefix)
                context_tkns = self.batch_size * [prefix_enc]
            elif isinstance(prefix[0], list):
                it = iter(prefix)  # https://stackoverflow.com/a/35791116
                l = len(next(it))
                assert isinstance(prefix[0][0], (int, np.integer)) and all(
                    len(p) == l for p in it
                ), """
                When passing a list as prefix, the following two conditions must be
                met:
                    - the list contains tokenized sequences, not strings;
                    - all tokenized sequences must have the same length.
                In order to pass several prefixes to the generator,
                pre-tokenize them with encode()."""
                self._check_batch_size(len(prefix))
                prefix_enc = prefix if not self.reverse else [p[::-1] for p in prefix]
                prefix = self.decode(prefix)
                context_tkns = prefix_enc
            else:
                raise TypeError(
                    """
                    The prefix is must be a list of ints, of lists (equal
                    lengths), or of np.arrays.
                    """
                )
        else:
            self._check_batch_size(batch_size)
            prefix_enc = (
                self.encode(prefix) if not self.reverse else self.encode(prefix)[::-1]
            )
            prefix = [prefix]
            context_tkns = self.batch_size * [prefix_enc]

        data = {
            "prefix": prefix,
            "prefix_enc": prefix_enc,
            "context_tkns": context_tkns,
        }
        return data

    def dummy_run(self):
        """
        A dummy runs forces some libraries to open at the onset of the program,
        clearing out messages and warnings.
        """
        self.run("A", length=1, batch_size=self.batch_size)

    def clear_line(self):
        # https://stackoverflow.com/a/943921
        _, columns = os.popen("stty size", "r").read().split()
        print(" " * int(columns), end="\r")

    # --------------------------------------------------------------------------------
    # Bowels
    # sampling utils
    # - normalisation/softmax (not used in the end)
    # - tok_k, top_p
    # - generation step
    # - sampling loop

    def _find_token(
        self, token, tkns, batch_data, chunk_length, exclude_until=True,
    ):
        """
        Helper function for gen_until(): finds whether the special_token is
        present in the produced sequences, and returns a dictionary containing
        the current sequences, the index where the token has been found
        (including it or not depending on exclude_until, and the
        previous_length (to search only in the produced bits). This function
        works on the tokens only.
        """
        for i, seq in enumerate(tkns):
            if batch_data[i]["index"] is None:
                index = token in seq[batch_data[i]["previous_length"] :]
                if index:
                    ind = np.where(seq[batch_data[i]["previous_length"] :] == token)[0][
                        0
                    ]
                    batch_data[i]["index"] = batch_data[i]["previous_length"] + ind
                    if not exclude_until:
                        batch_data[i]["index"] += 1
                else:
                    batch_data[i]["previous_length"] += chunk_length
                    batch_data[i]["seq"] = seq
        # print(batch_data)
        return batch_data

    def _find_regex(
        self, rr, tkns, batch_data, exclude_until=True,
    ):
        """
        Helper function for gen_until(): finds whether the regex is
        present in the produced sequences, and returns a dictionary containing
        the current sequences, the index where the token has been found
        (including it or not depending on exclude_until, and the
        previous_length (to search only in the produced bits). This function
        converts the tokens into strings and then performs regex search.
        """
        seqs = self.decode(tkns) if not self.reverse else self.decode(tkns[:, ::-1])
        for i, seq in enumerate(seqs):
            if batch_data[i]["index"] is None:
                if not self.reverse:
                    ind = regex.search(
                        rr, batch_data[i]["seq"][batch_data[i]["previous_length"] :]
                    )
                else:
                    ind = regex.search(
                        rr, batch_data[i]["seq"][: -batch_data[i]["previous_length"]]
                    )
                if ind:
                    if not self.reverse:
                        batch_data[i]["index"] = (
                            ind.span()[0] if exclude_until else ind.span()[1]
                        )
                        ind = batch_data[i]["previous_length"] + batch_data[i]["index"]
                        batch_data[i]["seq"] = batch_data[i]["seq"][:ind]
                    else:
                        batch_data[i]["index"] = (
                            ind.span()[1] if exclude_until else ind.span()[0]
                        )
                        batch_data[i]["seq"] = batch_data[i]["seq"][
                            batch_data[i]["index"] :
                        ]
                else:
                    # batch_data[i]["length"] = len(seq) # used for debugging
                    batch_data[i]["previous_length"] = len(batch_data[i]["seq"])
                    batch_data[i]["seq"] = seq
        # print(batch_data)
        return batch_data

    def top_k_logits(self, logits, k):
        """
        Top K sampling. Selects the k tokens with the max probability for the
        next step, and reduces the rest to near zero.
        """
        if k == 0:
            # no truncation
            return logits

        def _top_k():
            values, _ = tf.nn.top_k(logits, k=k)
            min_values = values[:, -1, tf.newaxis]
            return tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )

        return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k(),)

    def top_p_logits(self, logits, p):
        """
        Nucleus sampling. Selects the logits the combined probability of which
        is smaller than p (a number between 0 & 1), and reduces the rest to
        near zero.
        Difference with top K: if p is .9, depending on the case, the number of
        possible tokens that have a combined probability of 90% can be small
        (e.g. only 5, the next step is highly predictable, <-> 5 tokens are
        highly likely, the rest not at all), whereas at other times there might
        be almost the entire vocabulary allowed (almost no preference, high
        randomness).
        """
        with tf.compat.v1.variable_scope("top_p_logits"):
            logits_sort = tf.sort(logits, direction="DESCENDING")
            probs_sort = tf.nn.softmax(logits_sort)
            probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
            logits_masked = tf.where(
                probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000
            )  # [batchsize, vocab]
            min_logits = tf.reduce_min(
                logits_masked, axis=1, keepdims=True
            )  # [batchsize, 1]
            return tf.where(
                logits < min_logits,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )

    def step(self, tokens, past=None):

        """# {{{
        One step of Transformation.
        Inputs:
        -------
        tokens: machine-readable subwords
        past: None at first, will contain the matrices of attention for
            each step (remembered in the loop of self.sample, so that they
            don't need to be recomputed each time).
        Returns:
        --------
        a dictionary containing:
            logits: probabilities for the next tokens at each step,
                shape: (batch_size, n_tokens - 1, n_vocab)
            presents: the attention matrices, shape:
                [batch_size, n_layer, 2, n_head, sequence, n_embd // n_head]
                (see model.past_shape and default_hparams())
        """# }}}

        lm_output = self.model(
            hparams=self.hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE
        )
        logits = lm_output["logits"][:, :, : self.hparams.n_vocab]
        presents = lm_output["present"]
        presents.set_shape(
            model.past_shape(hparams=self.hparams, batch_size=self.batch_size)
        )
        return {
            "logits": logits,
            "presents": presents,
        }

    def sample(
        self, length=5, context=None, temperature=1, top_k=0, top_p=0.0,
    ):

        """# {{{
        The Sample loop, using tf.while(), see:
        https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/while_loop
        The while loop must be constructed functionally, hence the
        complication.
        Because of the constraints of graph building, all shapes have to be
        known in advance and must be the same for the two main components, cond
        and body. They are passed through loop_vars.
        It is composed of:
            - a condition: (not used, always true)
            - a body: the core computation, which extracts the logits for the
              next step and samples a next token (applying temperature,
              optionally with top_k/top_p).
            - maximum_iteration: how the loop actually ends
            - shape_invariants: specifying in advance what the shapes of body's
              arguments, loop_vars, will be.
            - loop_vars: the variables passed to the loop (body/cond).
            - back_prop: Nein!
        Parameters:
        -----------
        length: number of tokens to be generated (not string letters). Default: 5.
        context: machine readable tokens fed into the network as prefix
            shape: (batch_size, n_tokens)
        temperature: float. Used when sampling. A higher temperature flattens the
            probability curve for the next tokens (things are more random, an unlikely
            choice has more chances to occur). A lower one means the reverse, the most
            likely events are even more likely to occur. With a low temperature, the
            network is more stable (but can end up just repeating itself or being flat);
            with a high temperature, the network is more 'creative', which can lead to
            unstable/chaotic outputs.
        top_k: int. The network samples only from the top_k likeliest tokens
            at each step. Default: 0 (deactivated).
        top_p: float, ]0,1]. Nucleus sampling. At each step, the network will sample
            from the most probable tokens the combined probabilities of which
            is at most top_p. Default: 0.0 (deactivated).
        Returns:
        --------
        tokens: machine-readable subwords
        all_logits: probabilities for the next token at each step
                    shape: (batch_size, n_tokens, n_vocab)
        all_scores: probabilities at each step for the sampled tokens
        """# }}}

        with tf.name_scope("sample_sequence"):
            # Don't feed the last context token -- leave that to the loop below
            # TODO: Would be slightly faster if we called step on the entire context,
            # rather than leaving the last token transformer calculation to the while loop.

            context_output = self.step(context[:, :-1])

            def body(all_logits, past, prev, output):
                next_outputs = self.step(prev[:, None], past=past)
                logits = next_outputs["logits"][:, -1, :] / tf.cast(
                    temperature, tf.float32
                )
                # going ever more flowy with a tf.cond, allowing for dynamic
                # resetting of top_p & top_k during generation
                def p():
                    return self.top_p_logits(logits, p=top_p)

                def k():
                    return self.top_k_logits(logits, k=top_k)

                logits = tf.cond(tf.greater(top_p, 0.0), p, k)
                # use the logits to sample an index from them (equivalent to a token)
                # sample shape: (batch_size, 1)
                samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)

                return [
                    # all logits [batch_size, seq_len, n_vocab]
                    tf.concat([all_logits, next_outputs["logits"]], axis=-2),
                    # all pasts/presents (attention matrices)
                    # [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]
                    tf.concat([past, next_outputs["presents"]], axis=-2),
                    # previous samples (last token)
                    tf.squeeze(samples, axis=[1]),
                    # sequences (all tokens)
                    tf.concat([output, samples], axis=1),
                ]

            def cond(*args):
                return True

            all_logits, _, _, tokens = tf.while_loop(
                cond=cond,
                body=body,
                maximum_iterations=length,
                loop_vars=[
                    context_output["logits"],   # all logits
                    context_output["presents"], # pasts/presents
                    context[:, -1],             # prev
                    context,                    # output
                ],
                shape_invariants=[
                    # all logits
                    tf.TensorShape([self.batch_size, None, self.hparams.n_vocab]),
                    # all pasts/presents (attention matrices)
                    # [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]
                    tf.TensorShape(
                        model.past_shape(
                            hparams=self.hparams, batch_size=self.batch_size
                        )
                    ),
                    # previous samples (last token)
                    tf.TensorShape([self.batch_size]),
                    # sequences (all tokens)
                    tf.TensorShape([self.batch_size, None]),
                ],
                back_prop=False,
            )

            return tokens, all_logits

    # --------------------------------------------------------------------------------
    # Perplexity works:
    # -----------------
    # - _stats: produce stats for a batch of arrays (min/max/range/mean/std)
    # - _perplexities: to be used in combination with the output of run (which
    #                  returns scores
    # - _ranks: returns the ranks of batched sentences, output by run
    # - get_logits: run existing tokens thru the network, returns logits
    # - get_rank: gets rank for one or more existing sentences
    # - get_perplexity: gets perplexity for one or more existing sentences

    def _stats(self, arr, name=None):

        """# {{{
        Parameters:
        -----------
        arr: an array/tuple/np.array of batch_size arrays/tuples/np.arrays.
        Returns:
        --------
        a dictionary of statistics:
            min:  the min, shape: (batch_size, 1)
            max: the max, shape: (batch_size, 1)
            range: the range, shape: (batch_size, 1)
            mean: the mean, shape: (batch_size, 1)
            std: the standard deviation, shape: (batch_size, 1)
        """# }}}

        if name and name[-1] != "_":
            name = f"{name}_"
        else:
            name = ""
        return {
            f"{name}min": np.min(arr, axis=-1, keepdims=True),
            f"{name}max": np.max(arr, axis=-1, keepdims=True),
            f"{name}range": np.ptp(arr, axis=-1, keepdims=True),
            f"{name}mean": np.mean(arr, axis=-1, keepdims=True),
            f"{name}std": np.std(arr, axis=-1, keepdims=True),
        }

    def _perps_n_ranks(self, data, return_perplexities=True, return_ranks=True):
        logprobs = self.sess.run(tf.nn.softmax(data["logits"], axis=-1))
        # extract scores & calculate perplexities
        seq_len = len(data["tokens"][0]) - 1
        indz = (
            tuple(((i,) * seq_len for i in range(self.batch_size))),
            self.batch_size * (tuple(range(seq_len)),),
            tuple(tuple(tkn) for tkn in data["tokens"][:, 1:]),
        )
        scores = logprobs[indz]
        if return_perplexities:
            perp_data = self._perplexities(scores)
            data.update(
                {
                    "logprobs": logprobs if not self.reverse else logprobs[:, ::-1],
                    "scores": scores if not self.reverse else scores[:, ::-1],
                    **perp_data,
                }
            )
        if return_ranks:
            ranks_data = self._ranks(logprobs, scores)
            data.update(**ranks_data)
        return data

    def _perplexities(self, scores):

        """# {{{
        Compute the perplexity given a batch of scores (computed by
        self.run()).
        Parameters:
        -----------
        scores: the softmaxed logits.
            shape: (batch_size, seq_len - 1)
        Returns
        -------
        a dictionary containing:
            perplexities: 2 ** mean(log2(scores)), shape: (batch_size, 1)
            scores_min: the min of scores, shape: (batch_size, 1)
            scores_max: the max of scores, shape: (batch_size, 1)
            scores_range: the range of scores, shape: (batch_size, 1)
            scores_mean: the mean of scores, shape: (batch_size, 1)
            scores_std: the standard deviation of scores, shape: (batch_size, 1)
        """# }}}

        return {
            "perplexities": 2 ** -np.mean(np.log2(scores), axis=-1, keepdims=True),
            **self._stats(scores, name="scores"),
        }

    def _ranks(self, probs, scores):

        """# {{{
        Compute the rank of tokens for a batch of (freshly generated) input seqences(s).
        Note: this assumes equal lengths for sequences.
        Parameters
        ----------
        probs: all probs of the seq batch. (produced by .run())
        scores: scores for the seq batch. (produced by .run())
        Returns:
        --------
        a dictionary containing:
            ranks: ranks of each token for the seq batch, shape: (batch_size, n_tokens)
            ranks_min:  the min of ranks, shape: (batch_size, 1)
            ranks_max: the max of ranks, shape: (batch_size, 1)
            ranks_range: the range of ranks, shape: (batch_size, 1)
            ranks_mean: the mean of ranks, shape: (batch_size, 1)
            ranks_std: the standard deviation of ranks, shape: (batch_size, 1)
        """# }}}

        logits_sorted = np.sort(probs)[..., ::-1]  # descending order
        ranks = np.where(logits_sorted == scores[..., None])[-1]
        # np.where flattens the results -> reshape to (batch_size, seq_len)
        ranks = ranks.reshape(probs.shape[0], -1)
        return {
            "ranks": ranks,
            **self._stats(ranks, name="ranks"),
        }

    # Two following functions adapted from @gpt2ent:
    # https://github.com/gpt2ent/gpt-2-simple/blob/652fdab80131ce83f8f1b6fd00f597dd48ae2e36/gpt_2_simple/gpt_2.py#L504

    def get_logits(self, context_tokens, last_only=False, verbose=True):

        """# {{{
        Generate the logits (probabilities of each token) at each step for a
        given one or more sequences of tokens. If computing logits for a batch
        of tokens, said batch must have been padded beforehand (all token
        sequences must have the same length).
        Returns:
        --------
        logits: array of shape: (batch_size, n_tokens, n_vocab)
                or, if last_only is True: (batch_size, 1, n_vocab)
        """# }}}

        if not isinstance(context_tokens[0], (list, tuple, np.ndarray)):
            self._check_batch_size(1, verbose=verbose)
            context = self.batch_size * [context_tokens]
        else:
            self._check_batch_size(len(context_tokens), verbose=verbose)
            context = context_tokens
        logits = self.sess.run(self.model, feed_dict={self.context: context})["logits"]
        # all logits starting from the second token, n logits for n tokens
        # shape (batch_size, n_tokens, vocab_size)
        if not last_only:
            return logits
        # logits for next token, None to keep dims intact
        return logits[:, None, -1, :]

    def group_seqs_by_len(self, tokens):
        def custom_dd():
            return {"n": 0, "seqs": []}

        grouped_tkns = defaultdict(custom_dd)
        for seq in tokens:
            grouped_tkns[len(seq)]["n"] += 1
            grouped_tkns[len(seq)]["seqs"].append(seq)
        return grouped_tkns

    def seqs_to_tkns(self, sequences):
        if isinstance(sequences, str):
            sequences = [sequences]
            tkns = self.encode(sequences)
        elif isinstance(sequences, (list, np.ndarray)):
            if isinstance(sequences[0], str):
                tkns = self.encode(sequences)
            elif isinstance(sequences[0], (int, np.integer)):
                tkns = [sequences]
            elif isinstance(sequences[0], (list, np.ndarray)):
                tkns = sequences
        return tkns

    def get_rank(self, sequences=["\n"], verbose=False):

        """# {{{
        Compute the rank of tokens for input sentence(s). Note: this assumes
        unequal lengths for sentences. For freshly neuroned batches of equal
        lengths, use the logits returned by self.run() and pass them to
        _ranks().
        Inputs
        ------
        sequences: str, list/np.array of strings, of tokens, of lists/np.arrays
            of tokens.
        verbose: print the progress made. Defaults to false.
        Returns
        -------
        a dictionary containing:
            ranks: list of ranks at each step for each sentence.
                shape: (n_sequences, seq_len)
                ! seq_len is the number of tokens after encoding
            ranks_min:  the min of ranks, shape: (n_sequences, 1)
            ranks_max: the max of ranks, shape: (n_sequences, 1)
            ranks_range: the range of ranks, shape: (n_sequences, 1)
            ranks_mean: the mean of ranks, shape: (n_sequences, 1)
            ranks_std: the standard deviation of ranks, shape: (n_sequences, 1)
        """# }}}

        if verbose:
            msg = "calculating ranks of existing sentences:"
            print(msg)
            print("-" * len(msg))
        tkns = self.seqs_to_tkns(sequences)
        tot = len(tkns)
        count_len = len(str(tot))  # just for formatting purposes
        # assuming varying sequence lengths, just use a plain loop
        # and run each of them through the network
        ranks_stats = []
        ranks = []
        for i, seq in enumerate(tkns):
            seq_len = len(seq)
            logits = self.get_logits(context_tokens=seq)
            # don't take the last one (predicting the token after our sentence)
            if seq_len > 1:
                seq_len = seq_len - 1
                trunc = seq[1:]
                logits = logits[:, :-1, :]
            scores = np.nan_to_num(
                [(logits[0, i, token]) for i, token in enumerate(trunc)]
            )
            logits_sorted = np.sort(logits)[..., ::-1]  # descending order
            r = np.where(logits_sorted == scores[..., None])[-1]
            ranks.append(r)
            stats = self._stats(r, name="ranks")
            if verbose:
                print("-" * 40)
                print(f"{i+1:{count_len}}/{tot} |")
                print()
                print("sequence:")
                print(sequences[i])
                print()
                for k, v in stats.items():
                    print(f"{k}: {v[0]}")
                print()
                print("ranks:")
                print(ranks[i])
                print()
            ranks_stats.append(stats)
        print()
        return {
            "ranks": ranks,
            **{
                k: np.stack([st[k] for st in ranks_stats])
                for k in ranks_stats[0].keys()
            },
        }

    def get_perplexity(
        self, sequences=["\n"], verbose=False,
    ):

        """# {{{
        Compute perplexity score(s) for input sentence(s). Note: this assumes
        unequal lengths for sentences. For freshly neuroned batches of equal
        lengths, use the scores returned by self.run() and pass them to
        _perplexities.
        Parameters:
        -----------
        sequences: str, list/np.array of strings, of tokens, of lists/np.arrays
            of tokens.
        verbose: print the progress made. Defaults to false.
        Returns:
        --------
        a dictionary containing:
            scores: list of scores at each step for each sentence.
                shape: (n_sequences, seq_len)
                ! seq_len is the number of tokens after encoding
            perplexities: array of perplexity score(s).
                shape: (n_sequences, 1)
            scores_min:  the min of scores, shape: (n_sequences, 1)
            scores_max: the max of scores, shape: (n_sequences, 1)
            scores_range: the range of scores, shape: (n_sequences, 1)
            scores_mean: the mean of scores, shape: (n_sequences, 1)
            scores_std: the standard deviation of scores, shape: (n_sequences, 1)
        """# }}}

        if verbose:
            msg = "calculating perplexity of existing sentences:"
            print(msg)
            print("-" * len(msg))
        tkns = self.seqs_to_tkns(sequences)
        tot = len(tkns)
        count_len = len(str(tot))  # just for formatting purposes
        # assuming varying sequence lengths, just use a plain loop
        # and run each of them through the network
        scores = []
        scores_stats = []
        perplexities = []
        for i, seq in enumerate(tkns):
            logits = self.get_logits(context_tokens=seq)
            # don't take the last one (predicting the token after our sentence)
            if len(seq) > 1:
                trunc = seq[1:]
                logits = logits[:, :-1, :]
            logprobs = self.sess.run(tf.nn.softmax(logits, axis=-1))
            s = np.nan_to_num(
                [(logprobs[0, i, token]) for i, token in enumerate(trunc)]
            )
            scores.append(s)
            stats = self._stats(s, name="scores")
            scores_stats.append(stats)
            perp = 2 ** -np.mean(np.log2(s), axis=-1, keepdims=True)
            perplexities.append(perp)
            if verbose:
                print("-" * 40)
                print(f"{i+1:{count_len}}/{tot} |")
                print()
                print("sequence:")
                print(sequences[i])
                print()
                print(f"perplexity: {perp[0]}")
                for k, v in stats.items():
                    print(f"{k}: {v[0]}")
                print()
                print("scores:")
                print(scores[i])
                print()
        return {
            "scores": scores,
            **{
                k: np.stack([st[k] for st in scores_stats])
                for k in scores_stats[0].keys()
            },
            "perplexities": np.array(perplexities),
        }


if __name__ == "__main__":
    m = Model(batch_size=20)
    data = m.run("LE COMTE.", length=200)
    indz = data["perplexities"][:, 0].argsort()[
        ::-1
    ]  # https://stackoverflow.com/a/2828121
    sorted_perplexities = data["perplexities"][indz]
    sorted_seqs = data["tokens"][indz]

    print("--------------------------------")
    for perp, sentence in zip(sorted_perplexities, m.decode(sorted_seqs)):
        print()
        print(sentence)
        print(f"\t\t\t\t---> perplexity: {perp[0]:.16f}")
        print("--------------------------------")
