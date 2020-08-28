from tensorflow.core.protobuf import rewriter_config_pb2
from collections import defaultdict
from operator import itemgetter
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
        self.batch_size = batch_size
        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        self.length = tf.compat.v1.placeholder(tf.int32, ())
        self.temperature = tf.compat.v1.placeholder(tf.float32, ())
        self.top_k = tf.compat.v1.placeholder(tf.int32, ())
        self.top_p = tf.compat.v1.placeholder(tf.float32, ())
        self.reverse = reverse
        # required to load checkpoint
        self.model = model.model(hparams=self.hparams, X=self.context)
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
        """
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

        """
        for seq in self.gen(
            prefix=prefix,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=batch_size,
            return_tokens=print_tokens,
        ):
            print("-" * 40)
            print(seq)
        print("-" * 40)

    def gen(
        self,
        prefix="\n",
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        return_tokens=False,
        skip_encoding=False,
    ):
        """
        Higher level generation: input a sentence, get an array with n batches
        of continuations.
        """
        pref_data = self._check_prefix(prefix, batch_size)
        prefix, pref, context_tkns = itemgetter("prefix", "pref", "context_tkns")(
            pref_data
        )
        tkns, _ = self.sess.run(
            self.output,
            feed_dict={
                self.length: length,
                self.context: context_tkns,
                self.temperature: temperature,
                self.top_k: top_k,
                self.top_p: top_p,
            },
        )
        if self.reverse:
            return self.decode(tkns[:, ::-1]) if not return_tokens else tkns[:, ::-1]
        else:
            return self.decode(tkns) if not return_tokens else tkns

    def gen_until(
        self,
        prefix="\n",
        until="<|e|>",
        exclude_until=True,
        limit=200,
        chunk_length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        return_tokens=False,
    ):
        pref_data = self._check_prefix(prefix, batch_size)
        prefix, pref, context_tkns = itemgetter("prefix", "pref", "context_tkns")(
            pref_data
        )
        if until in self.special_tokens:
            until = self.encode(until)[0]
            use_regex = False
        else:
            rr = regex.compile(regex.escape(until))
            use_regex = True
        if not use_regex:
            batch_data = [
                {"previous_length": len(pref), "index": None, "seq": pref,}
                for _ in range(self.batch_size)
            ]
            i = 0
            while i < limit and not all(s["index"] is not None for s in batch_data):
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
                batch_data = self._find_token(
                    until,
                    tkns,
                    batch_data=batch_data,
                    chunk_length=chunk_length,
                    exclude_until=exclude_until,
                )
                context_tkns = tkns
                i += 1
            tkns = [t[: batch_data[i]["index"]] for i, t in enumerate(tkns)]
            if self.reverse:
                tkns = [t[::-1] for t in tkns]
            return self.decode(tkns) if not return_tokens else tkns
        else:
            batch_data = [
                {"previous_length": len(prefix), "index": None, "seq": prefix,}
                for _ in range(self.batch_size)
            ]
            i = 0
            while i < limit and not all(s["index"] is not None for s in batch_data):
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
            return [s["seq"] for s in batch_data]

    def gen_avoiding(
        self,
        prefix,
        avoided_tkn_or_regex,
        chunk_length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        limit=100,
        return_tokens=False,
    ):
        pref_data = self._check_prefix(prefix, batch_size)
        prefix, pref, context_tkns = itemgetter("prefix", "pref", "context_tkns")(
            pref_data
        )
        cond = None
        i = 0
        while not cond:
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
            temperature += 0.1
            i += 1
            if i > limit:
                break
            print(tkns)
            if isinstance(avoided_tkn_or_regex, str):
                generated = (
                    [self.decode(t[-chunk_length:]) for t in tkns]
                    if not self.reverse
                    else [self.decode(t[-chunk_length:][::-1]) for t in tkns]
                )
                if not self.reverse:
                    cond = all(
                        not regex.search(avoided_tkn_or_regex, seq) for seq in generated
                    )
                else:
                    cond = all(
                        not regex.search(avoided_tkn_or_regex, seq) for seq in generated
                    )
            else:
                cond = all(avoided_tkn_or_regex not in t[-chunk_length:] for t in tkns)
        if self.reverse:
            return self.decode(tkns[:, ::-1]) if not return_tokens else tkns[:, ::-1]
        else:
            return self.decode(tkns) if not return_tokens else tkns

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
        """
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
            probs: the normalized logits (softmaxed into probabilities)
                    shape: (batch_size, n_tokens - 1, n_vocab)

            if return_perplexities:
            -----------------------
            scores: sequence of logits (scores, unnormalized) for each sequence
                    shape: (batch_size, n_tokens)
            perplexities: the perplexity for each sentence
                    shape: (batch_size, 1)
            scores_stats: a dict containing:
                min:  the min, shape: (batch_size, 1)
                max: the max, shape: (batch_size, 1)
                range: the range, shape: (batch_size, 1)
                mean: the mean, shape: (batch_size, 1)
                std: the standard deviation, shape: (batch_size, 1)

            if return_ranks:
            ----------------
            ranks: sequence of ranks for each sequence
                    shape: (batch_size, n_tokens)
            ranks_stats: a dict containing:
                min:  the min, shape: (batch_size, 1)
                max: the max, shape: (batch_size, 1)
                range: the range, shape: (batch_size, 1)
                mean: the mean, shape: (batch_size, 1)
                std: the standard deviation, shape: (batch_size, 1)
        """
        pref_data = self._check_prefix(prefix, batch_size)
        prefix, pref, context_tkns = itemgetter("prefix", "pref", "context_tkns")(
            pref_data
        )
        tokens, logits = self.sess.run(
            self.output,
            feed_dict={
                self.length: length,
                self.context: context_tokens,
                self.temperature: temperature,
                self.top_k: top_k,
                self.top_p: top_p,
            },
        )
        probs = self.sess.run(tf.nn.softmax(logits, axis=-1))
        data = {
            "tokens": tokens,
            "logits": logits,
            "probs": probs,
        }
        if return_perplexities or return_ranks:
            # extract scores & calculate perplexities
            seq_len = len(tokens[0]) - 1
            indz = (
                tuple(((i,) * seq_len for i in range(self.batch_size))),
                self.batch_size * (tuple(range(seq_len)),),
                tuple(tuple(tkn) for tkn in tokens[:, 1:]),
            )
            scores = probs[indz]
            perp_data = self._perplexities(scores)
            data.update(
                {"scores": scores, **perp_data,}
            )
            if return_ranks:
                ranks_data = self._ranks(probs, scores)
                data.update(**ranks_data)
        return data

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
            return [self.enc.encode(ss) for ss in s]

    def decode(self, s):
        """
        Decode an array (or a batch) of machine-readable subwords into an array
        of string(s).
        """
        if isinstance(s[0], (int, np.integer)):
            return self.enc.decode(s)
        elif isinstance(s, (list, tuple, np.ndarray)):
            return [self.enc.decode(ss) for ss in s]

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
        self._check_hparams(hparams_file)
        self.batch_size = batch_size
        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
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
        self.ckpt = tf.train.latest_checkpoint(path)
        self.saver = tf.compat.v1.train.Saver(allow_empty=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print("-" * 40)
        print(f"Loading checkpoint {self.ckpt}")
        self.saver.restore(self.sess, self.ckpt)

    def _check_hparams(self, hparams_file):
        if hparams_file is not None:
            print(f"Reloading hparams from file {hparams_file}")
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
        """
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
            - a dictionary containing:
                - "prefix": what was passed in, wrapped with an additional
                batch dimension if need be
                - "pref": the encoded prefix, equal or not to "prefix"
                - "context_tkns": the context tokens, a batch of shape (batch_size, n_tokens),
            ready to be fed to the network
        """
        if isinstance(prefix, list):
            if isinstance(prefix[0], (int, np.integer)):
                prefix = [prefix]
            else:
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
            if self.reverse:
                context_tkns = prefix[:, ::-1]
                pref = context_tkns
            else:
                context_tkns = prefix
                pref = prefix
        else:
            self._check_batch_size(batch_size)
            pref = (
                self.encode(prefix) if not self.reverse else self.encode(prefix)[::-1]
            )
            context_tkns = self.batch_size * [pref]
        return {
            "prefix": prefix,
            "pref": pref,
            "context_tkns": context_tkns,
        }

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
        """
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
                - logits: probabilities for the next tokens at each step,
                          shape: (batch_size, n_tokens - 1, n_vocab)
                - presents: the attention matrices, shape:
                            [batch_size, n_layer, 2, n_head, sequence, n_embd // n_head]
                            (see model.past_shape and default_hparams())
        """
        lm_output = model.model(
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
        """
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
        Inputs
        ------
            - context: machine readable tokens, shape: (batch_size, n_tokens)
        Returns
        -------
            - tokens: machine-readable subwords
            - all_logits: probabilities for the next token at each step
                          shape: (batch_size, n_tokens, n_vocab)
            - all_scores: probabilities at each step for the sampled tokens
        """

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
                    # all logits
                    tf.concat([all_logits, next_outputs["logits"]], axis=-2),
                    # all pasts/presents (attention matrices)
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

    def _stats(self, arr):
        """
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
        """
        lemin = np.min(arr, axis=-1, keepdims=True)
        lemax = np.max(arr, axis=-1, keepdims=True)
        return {
            "min": lemin,
            "max": lemax,
            "range": lemax - lemin,
            "mean": np.mean(arr, axis=-1, keepdims=True),
            "std": np.std(arr, axis=-1, keepdims=True),
        }


    def _perplexities(self, probs):
        """
        Compute the perplexity given a batch of probs (computed by
        self.run()).

        Parameters:
        -----------
        probs: the softmaxed logits.
            shape: (batch_size, seq_len - 1)

        Returns
        -------
        a dictionary containing:
            scores_stats: a dictionary of statistics:
                min:  the min, shape: (batch_size, 1)
                max: the max, shape: (batch_size, 1)
                range: the range, shape: (batch_size, 1)
                mean: the mean, shape: (batch_size, 1)
                std: the standard deviation, shape: (batch_size, 1)
            perplexities: 2 ** mean(log2(probs))
                shape: (batch_size, 1)
        """

        return {
            "scores_stats": self._stats(probs),
            "perplexities": 2 ** -np.mean(np.log2(probs), axis=-1, keepdims=True),
        }

    def _ranks(self, probs, scores):
        """
        Compute the rank of tokens for a batch of (freshly generated) input seqences(s).
        Note: this assumes equal lengths for sequences.

        Parameters
        ----------
        probs: all probs of the seq batch. (produced by .run())
        scores: scores for the seq batch. (produced by .run())

        Returns:
        --------
        a dictionary containing:
            ranks: ranks of each token for the seq batch.
                shape: (batch_size, n_tokens)
            ranks_stats: a dictionary of statistics:
                min:  the min, shape: (batch_size, 1)
                max: the max, shape: (batch_size, 1)
                range: the range, shape: (batch_size, 1)
                mean: the mean, shape: (batch_size, 1)
                std: the standard deviation, shape: (batch_size, 1)
        """
        logits_sorted = np.sort(probs)[..., ::-1]  # descending order
        ranks = np.where(logits_sorted == scores[..., None])[-1]
        # np.where flattens the results -> reshape to (batch_size, seq_len)
        ranks = ranks.reshape(probs.shape[0], -1)
        return {
            "ranks": ranks,
            "ranks_stats": self._stats(ranks),
        }

    # Two following functions adapted from @gpt2ent:
    # https://github.com/gpt2ent/gpt-2-simple/blob/652fdab80131ce83f8f1b6fd00f597dd48ae2e36/gpt_2_simple/gpt_2.py#L504

    def get_logits(self, context_tokens, last_only=False, verbose=True):
        """
        Generate the logits (probabilities of each token) at each step for a
        given one or more sequences of tokens. If computing logits for a batch
        of tokens, said batch must have been padded beforehand (all token
        sequences must have the same length).
        Returns:
        --------
            logits: array of shape: (batch_size, n_tokens, n_vocab)
                    or, if last_only is True: (batch_size, 1, n_vocab)
        """
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

    def get_rank(
        self, sentences=["\n"], return_all_ranks=False, verbose=False, batched=False,
    ):
        """
        Compute the rank of tokens for input sentence(s). Note: this assumes
        unequal lengths for sentences. For freshly neuroned batches of equal
        lengths, use the logits returned by self.run() and pass them to
        _ranks().
        Inputs
        ------
            - sentences: str or array
            - return_all_ranks: if True returns the arrays of ranks for all
                                steps. Defaults to False.
            - verbose: print the progress made. Defaults to false.
        Returns
        -------
            - seq_ranks: array of rank score(s), one per sentence, according to
                        'mode'. shape: (n_sentences,)
            - all_tkns_ranks: array of ranks of tokens at each step for each sentence.
                                shape: (n_sentences, seq_len)
                                ! seq_len is the number of tokens after encoding
        """
        if verbose:
            msg = "calculating ranks of existing sentences:"
            print(msg)
            print("-" * len(msg))
        if isinstance(sentences, str):
            sentences = [sentences]
        tkns = self.encode(sentences)
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
            data = self._stats(r)
            if verbose:
                print(f"{i+1:{count_len}}/{tot} | {sentences[i]}")
            ranks_stats.append(data)
        print()
        return {
            "ranks": ranks,
            "ranks_stats": ranks_stats,
        }

    def get_perplexity(
        self,
        sentences=["\n"],
        mode=None,
        return_scores=False,
        verbose=False,
        batched=False,
    ):
        """
        Compute perplexity score(s) for input sentence(s). Note: this assumes
        unequal lengths for sentences. For freshly neuroned batches of equal
        lengths, use the scores returned by self.run() and pass them to
        _perplexities.

        Parameters:
        -----------
        sentences: str or array of strings.
        return_scores: if True returns the arrays of scores. Defaults to False.
        verbose: print the progress made. Defaults to false.
        batched: boolean. If False, plain loop, constant batch size of 1,
           else, collecting seqences by tokenized length, processing
           them batch by batch. (*This is actually slower than the
           loop!*)

        Returns:
        --------
        perplexities: array of perplexity score(s).
            shape: (n_sentences,)
        scores (if verbose): array of scores at each step for each sentence.
            shape: (n_sentences, seq_len)
                   ! seq_len is the number of tokens after encoding
    """
        if verbose:
            msg = "calculating perplexity of existing sentences:"
            print(msg)
            print("-" * len(msg))
        if isinstance(sentences, str):
            sentences = [sentences]
        tkns = self.encode(sentences)
        if batched:
            perplexities = []
            all_scores = []
            grouped_tkns = self.group_seqs_by_len(tkns)
            print("sequences grouped as follows:")
            n_groups = 0
            for seq_len, data in sorted(grouped_tkns.items(), key=lambda d: d[1]["n"]):
                n_groups += 1
                n_seqs = data["n"]
                print(f" - {n_seqs} sequence(s) of length: {seq_len}.")
            print()

            print("processing batches:")
            c = 0
            count = 0
            n_gr_str = len(str(n_groups))
            for seq_len, data in sorted(
                grouped_tkns.items(), key=lambda d: d[1]["n"], reverse=True
            ):
                c += 1
                seqs = np.array(data["seqs"])
                n_seqs = data["n"]
                seq_pr = len(str(seq_len))  # formatting

                # if the seq is longer than one, don't take the last one
                # (predicting the token after our sentence)
                if seq_len > 1:
                    seq_len = seq_len - 1

                # TODO: - better division into chunks
                #       - use e.g. h5py to store data to disk as it's the RAM not GPU that is the bottleneck
                all_scores = []
                div = int(np.ceil(n_seqs / 10))
                while True:
                    seqs_chunks = np.array_split(seqs, div, axis=0)
                    try:
                        for i, s_chunk in enumerate(seqs_chunks):

                            tmp_n_seqs = len(s_chunk)
                            logits = self.get_logits(
                                context_tokens=s_chunk, verbose=False
                            )
                            count += tmp_n_seqs
                            self.clear_line()
                            if verbose:
                                print(
                                    f"{c:>{n_gr_str}}/{n_groups} | "
                                    + f"processing a batch of {tmp_n_seqs:3} "
                                    + f"sequence(s) of length: {seq_len:{seq_pr}}. "
                                    + f"(batch {i+1:2} of {div:2} done, "
                                    + f"{tmp_n_seqs:2} seq(s)) | total seqs "
                                    + f"so far: {count}",
                                    end="\r",
                                )

                            # shortening as above for seq_len: remove 1st tkn & last logit
                            if seq_len > 1:
                                s_chunk = s_chunk[:, 1:]
                                logits = logits[:, :-1, :]

                            indz = (
                                tuple(((i,) * seq_len for i in range(tmp_n_seqs))),
                                tmp_n_seqs * (tuple(range(seq_len)),),
                                tuple(tuple(tkn) for tkn in s_chunk),
                            )

                            scores = logits[indz]  # .copy()
                            all_scores.extend(s for s in scores)
                            perplexities.extend(
                                p for p in self._perplexities(scores).flatten()
                            )

                            gc.collect()

                        break
                    except KeyboardInterrupt:
                        print()
                        print("user aborted, bye.")
                        exit()
                    except Exception as e:
                        print(
                            f"oopsie, that batch ({n_seqs}/{div} ~= {n_seqs//div} seqs) was too big, dividing it into {div * 2} chunks of {n_seqs//(div * 2)} seqs..."
                        )
                        print(e)
                        div = div * 2
                print()

            return {
                "perplexities": perplexities,
                "scores": all_scores,
            }

        else:
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
                probs = self.sess.run(tf.nn.softmax(logits, axis=-1))
                s = np.nan_to_num(
                    [(probs[0, i, token]) for i, token in enumerate(trunc)]
                )
                scores.append(s)
                scores_stats.append(self._stats(s))
                perplexities.append(2 ** -np.mean(np.log2(s), axis=-1, keepdims=True))
                if verbose:
                    print(f"{i+1:{count_len}}/{tot} | {sentences[i]}")
                else:
                    print(f"({i+1:{count_len}}/{tot})", end="\r")
            print()
            return {
                "scores": scores,
                "scores_stats": scores_stats,
                "perplexities": perplexities,
            }


if __name__ == "__main__":
    m = Model(batch_size=20)
    tokens, logits, scores, perplexities = m.run("LE COMTE.", length=200)
    indz = perplexities[:, 0].argsort()[::-1]  # https://stackoverflow.com/a/2828121
    sorted_perplexities = perplexities[indz]
    sorted_seqs = tokens[indz]

    print("--------------------------------")
    for perp, sentence in zip(sorted_perplexities, m.decode(sorted_seqs)):
        print()
        print(sentence)
        print(f"\t\t\t\t---> perplexity: {perp[0]:.16f}")
        print("--------------------------------")
