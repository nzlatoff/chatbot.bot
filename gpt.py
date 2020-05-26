from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
import numpy as np
import encoder
import random
import model
import regex
import json
import sys
import os

# PYTHONPATH=src python bridges.py

# disabling some warnings
os.environ["KMP_WARNINGS"] = "off"


class Model:
    def __init__(
        self, model_name="117M", run_name="run1", device="/GPU:0", batch_size=1
    ):
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.rewrite_options.layout_optimizer = (
            rewriter_config_pb2.RewriterConfig.OFF
        )
        self.sess = tf.compat.v1.Session(config=self.config)
        self.enc = encoder.get_encoder(model_name, "models")
        self.hparams = model.default_hparams()
        with open(f"models/{model_name}/hparams.json") as f:
            self.hparams.override_from_dict(json.load(f))
        self.batch_size = batch_size
        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        self.length = tf.compat.v1.placeholder(tf.int32, ())
        self.temperature = tf.compat.v1.placeholder(tf.float32, ())
        self.top_k = tf.compat.v1.placeholder(tf.int32, ())
        self.top_p = tf.compat.v1.placeholder(tf.float32, ())
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

    def gen(
        self,
        prefix="\n",
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        reverse=False,
    ):
        """
        Higher level generation: input a sentence, get an array with n batches
        of continuations.
        """
        self.check_batch_size(batch_size)
        context_tokens = self.batch_size * [self.encode(prefix)]
        tkns, logits = self.sess.run(
            self.output,
            feed_dict={
                self.length: length,
                self.context: context_tokens,
                self.temperature: temperature,
                self.top_k: top_k,
                self.top_p: top_p,
            },
        )
        if reverse:
            return self.decode(tkns[:, ::-1])
        else:
            return self.decode(tkns)

    def run(
        self,
        prefix="\n",
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.0,
        batch_size=None,
        reverse=False,
    ):
        """
        Lower level generation: input a sentence, get n batches of generated
        tokens as well as the logits associated with each step.
        Returns:
        --------
            tokens: machine-readable subwords (from 1 to n_vocab)
            logits: probabilities for the next token at each step
                    shape: (batch_size, n_tokens - 1, n_vocab)
            scores: sequence of logits (probs) for each sequence
                    shape: (batch_size, n_tokens)
            perplexities: the perplexity for each sentence
                    shape: (batch_size, 1)
        """
        if reverse:
            pref = self.encode(prefix)[::-1]
        else:
            pref = self.encode(prefix)
        self.check_batch_size(batch_size)
        context_tokens = self.batch_size * [pref]
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

        # extract scores & calculate perplexities
        seq_len = len(tokens[0]) - 1
        indz = (
            tuple(((i,) * seq_len for i in range(self.batch_size))),
            self.batch_size * (tuple(range(seq_len)),),
            tuple(tuple(tkn) for tkn in tokens[:, 1:]),
        )
        scores = logits[indz]
        perplexities = self._perplexities(scores)

        return tokens, logits, scores, perplexities

    # --------------------------------------------------------------------------------
    # encoder/decoder utils

    def encode(self, s):
        """
        Encode a string, or an array of strings, into an array (or a batch) of
        machine-readable subwords.
        """
        if isinstance(s, str):
            return np.array(self.enc.encode(s))
        elif isinstance(s, (list, tuple, np.ndarray)):
            return [self.enc.encode(ss) for ss in s]

    def decode(self, s):
        """
        Decode an array (or a batch) of machine-readable subwords into an array
        of string(s).
        """
        if isinstance(s[0], int):
            return np.array(self.enc.decode(s))
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

    def reset(
        self, hparams_file=None, device="/GPU:0", batch_size=1, top_k=0.0, top_p=0.0
    ):
        self.check_hparams(hparams_file)
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

    def check_hparams(self, hparams_file):
        if hparams_file is not None:
            print(f"Reloading hparams from file {hparams_file}")
            with open(hparams_file) as f:
                self.hparams.override_from_dict(json.load(f))

    def check_batch_size(self, batch_size, verbose=True):
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

    def dummy_run(self):
        """
        A dummy runs forces some libraries to open at the onset of the program,
        clearing out messages and warnings.
        """
        self.run("A", length=1, batch_size=self.batch_size)

    # --------------------------------------------------------------------------------
    # Bowels
    # sampling utils
    # - normalisation/softmax (not used in the end)
    # - tok_k, top_p
    # - generation step
    # - sampling loop

    def normalize(self, logits, verbose=False):
        """
        Normalize a tensor of logits + softmaxing it.
        Inputs
        ------
            - logits shape: (batch_size, seq_len, n_vocab)
        Returns
        -------
            - logprobs: shape: (batch_size, seq_len, n_vocab)
        """
        mu = np.mean(logits, axis=-1, keepdims=True)
        lm = logits - mu
        le = np.exp(lm)
        logprobs = le / np.sum(le, axis=-1, keepdims=True)
        if verbose:
            return logprobs, mu, lm, le
        return logprobs

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
    # - _perplexities: to be used in combination with the output of run (which
    #                  returns scores
    # - get_logits: run existing tokens thru the network, returns logits
    # - get_perplexity: gets perplexity for one or more existing sentences

    def _perplexities(self, scores):
        """
        Compute the perplexity given a batch of scores (computed by
        self.run()).
        Inputs
        ------
            - scores: shape: (batch_size, seq_len - 1)
        Returns
        -------
            - perplexities: shape: (batch_size, 1)
        """
        return 2 ** (
            -np.mean(np.log2(np.exp(np.nan_to_num(scores))), axis=-1, keepdims=True)
        )

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
            self.check_batch_size(1, verbose=verbose)
            context = self.batch_size * [context_tokens]
        else:
            self.check_batch_size(len(context_tokens), verbose=verbose)
            context = context_tokens
        logits = self.sess.run(self.model, feed_dict={self.context: context})["logits"]
        # all logits starting from the second token, n logits for n tokens
        # shape (batch_size, n_tokens, vocab_size)
        if not last_only:
            return logits
        # logits for next token, None to keep dims intact
        return logits[:, None, -1, :]

    def get_perplexity(
        self, sentences=["\n"], mode="max", return_scores=False, verbose=False
    ):
        """
        Compute perplexity score(s) for input sentence(s). Note: this assumes
        unequal lengths for sentences. For freshly neuroned batches of equal
        lengths, use the scores returned by self.run() and pass them to
        _perplexities.
        Inputs
        ------
            - sentences: str or array
            - mode: 'max': take the maximum score for each sequence.
                    'mean': takes the mean + 2**log2(exp(scores)).
                    Defaults to 'max'.
            - return_scores: if True returns the arrays of scores. Defaults to False.
            - verbose: print the progress made. Defaults to false.
        Returns
        -------
            - perplexities: array of perplexity score(s).
                            shape: (n_sentences,)
            - scores (if verbose): array of scores at each step for each sentence.
                            shape: (n_sentences, seq_len)
                                   ! seq_len is the number of tokens after encoding
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        tkns = self.encode(sentences)
        if verbose: count_len = len(str(len(tkns))) # just for formatting purposes
        # assuming varying sequence lengths, just use a plain loop
        # and run each of them through the network
        perplexities = []
        all_scores = []
        if verbose:
            msg = "calculating perplexity of existing sentences:"
            print(msg)
            print("-"*len(msg))
        for i, seq in enumerate(tkns):
            shorten = True if len(seq) > 1 else False
            logits = self.get_logits(context_tokens=seq)
            # don't take the last one (predicting the token after our sentence)
            if shorten:
                logits = logits[:, :-1, :]
            trunc = seq[1:] if shorten else seq
            scores = np.nan_to_num(
                [(logits[0, i, token]) for i, token in enumerate(trunc)]
            )
            all_scores.append(scores)
            # exponentiate only the numbers after selection
            if mode == "min":
                perplexity = min(scores)
            if mode == "mean":
                perplexity = -np.mean(scores)
            if mode == "meanmin":
                perplexity = min(scores) - np.mean(scores)
            if verbose:
                print(f"{i+1:{count_len}} | {perplexity:20.17f} | {sentences[i]}")
            perplexities.append(perplexity)
        print()
        if return_scores:
            return np.array(perplexities), np.array(all_scores)
        return np.array(perplexities)


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
