from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
import numpy as np
import encoder
import sample
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
        self, run_name="run1", device="/GPU:0", batch_size=1, top_k=0.0, top_p=0.0
    ):
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.rewrite_options.layout_optimizer = (
            rewriter_config_pb2.RewriterConfig.OFF
        )
        self.sess = tf.compat.v1.Session(config=self.config)
        self.enc = encoder.get_encoder(f"{run_name}")
        self.hparams = model.default_hparams()
        with open(f"checkpoint/{run_name}/hparams.json") as f:
            self.hparams.override_from_dict(json.load(f))
        self.batch_size = batch_size
        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        self.length = tf.compat.v1.placeholder(tf.int32, ())
        self.temperature = tf.compat.v1.placeholder(tf.float32, ())
        self.model = model.model(hparams=self.hparams, X=self.context)
        self.load_checkpoint(f"checkpoint/{run_name}")
        with tf.device(device):
            self.output = self.sample(
                length=self.length,
                context=self.context,
                temperature=self.temperature,
                top_k=top_k,
                top_p=top_p,
            )
        # spit out all these warnrings
        self.dummy_run()

    # --------------------------------------------------------------------------------
    # Main uses:
    # - generate text
    # - generate text & logits

    def gen(self, prefix="\n", length=5, temperature=1, batch_size=None):
        """
        Higher level generation: input a sentence, get an array with n batches
        of continuations.
        """
        self.check_batch_size(batch_size)
        context_tokens = self.batch_size * [self.encode(prefix)]
        tkns, logitst = self.sess.run(
                self.output,
                feed_dict={
                    self.length: length,
                    self.context: context_tokens,
                    self.temperature: temperature,
                },
            )
        return self.decode(tkns)

    def run(self, prefix="\n", length=5, temperature=1, batch_size=None):
        """
        Lower level generation: input a sentence, get n batches of generated
        tokens as well as the logits associated with each step.
        Returns:
        --------
            tokens: machine-readable subwords (from 1 to n_vocab)
            logits: probabilities for the next token at each step
                    shape: (batch_size, n_tokens - 1, n_vocab)
        """
        self.check_batch_size(batch_size)
        context_tokens = self.batch_size * [self.encode(prefix)]
        return self.sess.run(
                self.output,
                feed_dict={
                    self.length: length,
                    self.context: context_tokens,
                    self.temperature: temperature,
                },
            )

    # --------------------------------------------------------------------------------
    # encoder/decoder utils

    def encode(self, s):
        if isinstance(s, str):
            return np.array(self.enc.encode(s))
        elif isinstance(s, (list, tuple, np.ndarray)):
            return np.array([self.enc.encode(ss) for ss in s])

    def decode(self, s):
        if isinstance(s[0], int):
            return np.array(self.enc.decode(s))
        elif isinstance(s, (list, tuple, np.ndarray)):
            return np.array([self.enc.decode(ss) for ss in s])

    # --------------------------------------------------------------------------------
    # sampling utils

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
        return tf.cond(
           tf.equal(k, 0),
           lambda: logits,
           lambda: _top_k(),
        )


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
        with tf.compat.v1.variable_scope('top_p_logits'):
            logits_sort = tf.sort(logits, direction='DESCENDING')
            probs_sort = tf.nn.softmax(logits_sort)
            probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
            logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
            min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
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
        Returns
        -------
            a dictionary containing:
                - logits: probabilities for the next tokens at each step,
                          shape: (batch_size, n_tokens - 1, n_vocab)
                - presents: the attention matrices, shape:
                            [batch_size, n_layer, 2, n_head, sequence, n_embd // n_head]
                            (see model.past_shape and default_hparams())
        """
        lm_output = model.model(
            hparams=self.hparams,
            X=tokens,
            past=past,
            reuse=tf.compat.v1.AUTO_REUSE
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
        self,
        length=5,
        context=None,
        temperature=1,
        top_k=0,
        top_p=0.0,
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
        Returns
        -------
            - tokens: machine-readable subwords
            - all_logits: probabilities for the next token at each step
                          shape: (batch_size, n_tokens, n_vocab)
        """

        with tf.name_scope("sample_sequence"):
            # Don't feed the last context token -- leave that to the loop below
            # TODO: Would be slightly faster if we called step on the entire context,
            # rather than leaving the last token transformer calculation to the while loop.
            context_output = self.step(context[:, :-1])

            def body(all_logits, past, prev, output):
                next_outputs = self.step(prev[:, tf.newaxis], past=past)
                logits = next_outputs["logits"][:, -1, :] / tf.cast(
                    temperature, tf.float32
                )
                if top_p > 0.0:
                    logits = self.top_p_logits(logits, p=top_p)
                else:
                    logits = self.top_k_logits(logits, k=top_k)
                # use the logits to sample an index from them (equivalent to a token)
                samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
                return [
                    tf.concat([all_logits, next_outputs["logits"]], axis=-2),
                    tf.concat([past, next_outputs["presents"]], axis=-2),
                    tf.squeeze(samples, axis=[1]),
                    tf.concat([output, samples], axis=1),
                ]

            def cond(*args):
                return True

            all_logits, _, _, tokens = tf.while_loop(
                cond=cond,
                body=body,
                maximum_iterations=length,
                loop_vars=[context_output["logits"], context_output["presents"], context[:, -1], context,],
                shape_invariants=[
                    tf.TensorShape([self.batch_size, None, self.hparams.n_vocab]),
                    tf.TensorShape(
                        model.past_shape(hparams=self.hparams, batch_size=self.batch_size)
                    ),
                    tf.TensorShape([self.batch_size]),
                    tf.TensorShape([self.batch_size, None]),
                ],
                back_prop=False,
            )

            return tokens, all_logits

    # --------------------------------------------------------------------------------
    # plumbing:
    # ---------
    # - load checkpoint
    # - change hparams
    # - check & reset graph with new batch size
    # - dummy run to clear out messages

    def load_checkpoint(self, path="checkpoint/run1"):
        self.ckpt = tf.train.latest_checkpoint(path)
        self.saver = tf.compat.v1.train.Saver(allow_empty=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print(f"Loading checkpoint {self.ckpt}")
        self.saver.restore(self.sess, self.ckpt)

    def check_hparams(self, hparams_file):
        if hparams_file is not None:
            print(f"Reloading hparams from file {hparams_file}")
            with open(hparams_file) as f:
                self.hparams.override_from_dict(json.load(f))

    def check_batch_size(self, batch_size):
        """
        Returns self.batch_size if batch_size is None.
        Else runs reset() to redraw the graph with a new batch_size.
        """
        if batch_size is None:
            batch_size = self.batch_size
        else:
            if batch_size != self.batch_size:
                print("(batch size changed, resetting graph)")
                self.reset(batch_size=batch_size)

    def reset(self, hparams_file=None, device="/GPU:0", batch_size=1, top_k=0.0, top_p=0.0):
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

    def dummy_run(self):
        """
        A dummy runs forces some libraries to open at the onset of the program,
        clearing out messages and warnings.
        """
        self.run("A", length=1, batch_size=self.batch_size)

    # --------------------------------------------------------------------------------
    # Perplexity works:
    # -----------------
    # Two following functions adapted from @gpt2ent:
    # https://github.com/gpt2ent/gpt-2-simple/blob/652fdab80131ce83f8f1b6fd00f597dd48ae2e36/gpt_2_simple/gpt_2.py#L504

    def get_logits(self, context_tokens, batch_size=None, last_only=False):
        """
        Generate the logits (probabilities of each token) at each step for a
        given sequence of tokens.
        Returns:
        --------
            logits: array of shape: (batch_size, n_tokens, n_vocab)
                    or, if last_only is True: (batch_size, 1, n_vocab)
        """
        self.check_batch_size(batch_size)
        # self.model, defined at initialization, returns logits & attention matrix
        out = self.sess.run(
            self.model, feed_dict={self.context: self.batch_size * [context_tokens]}
        )
        # all logits starting from the second token, n logits for n tokens
        # shape (batch_size, n_tokens, vocab_size)
        if not last_only:
            return out["logits"]
        # logits for next token, None to keep dims intact
        return out["logits"][None, -1, :]

    def get_perplexity(self, sentence=["\n"], batch_size=None, verbose=False):
        self.check_batch_size(batch_size)
        tkns = self.encode(sentence)
        len_tkns = len(tkns)
        logits = self.get_logits(
            context_tokens=tkns, batch_size=self.batch_size
        )[None, :-1, :] # None to keep dims intact
        # don't take the last one (predicting the token after our sentence)
        # normalizing logits for numerical stability (does not affect the result)
        mu = np.mean(logits, axis=-1, keepdims=True)
        lm = logits - mu
        le = np.exp(lm)
        logprobs = le / np.sum(le, axis=1, keepdims=True)
        return logprobs, tkns
        if verbose:
            print(f"sentence: (len: {len_tkns:2}): {tkns}")
            print(f"logit shape:    {len(logits.shape)}")
            print(f"logprobs shape: {logprobs.shape}")
            print()
            for i, tkn in enumerate(tkns[:-1]):
                print(f"{i:3}: token: {tkn:5} | probability: {logprobs[i, tkn]}")
            print()
        # scores = np.nan_to_num(
        #     [logprobs[:, i, token] for i, token in enumerate(tkns[:-1])]
        # )
        scores = np.nan_to_num(
            [
                [logprobs[b, i, token] for i, token in enumerate(tkns[:-1])]
                for b in range(self.batch_size)
            ]
        )
        perplexities = 2 ** (-np.mean(np.log2(scores), axis=-1))
        return perplexities

    def perp_test(self, sentence, verbose=False):
        if verbose:
            print(f"sentence:")
            print(sentence)
        print("-" * 10)
        print(f"perplexity: {self.get_perplexity(sentence=sentence, verbose=verbose)}")
        print()


if __name__ == "__main__":
    le_model_fw = Model(run_name="run1")

    # ----------------------------------------
    # try several runs, order by perplexity
    results = []
    for _ in range(10):
        sc = le_model_fw.run(" ", length=50)
        results.append({"cont": sc, "perp": le_model_fw.get_perplexity(sc)})

    results = sorted(results, key=lambda x: x["perp"])
    for i, r in enumerate(results):
        print()
        print(f"{i}:")
        s = r["cont"]
        print(s)
        print(r["perp"])
        print("-" * 20)

    # # ----------------------------------------
    # # try random continuations, order them by perplexity
    # le_model_bw = Model(run_name='run1')

    # sf = le_model_fw.run(' ', length=50)

    # results = []
    # for _ in range(10):
    #     sc = le_model_bw.run(' ', length=50)
    #     results.append({
    #         "cont": sc,
    #         "perp": le_model_fw.get_perplexity(sf+sc)
    #     })

    # results = sorted(results, key=lambda x: x["perp"], reverse=True)
    # for r in results:
    #     print()
    #     s = r["cont"]
    #     print(f"{sf} /// {s}")
    #     print(r["perp"])
    #     print("-"*20)

    # ----------------------------------------
    # hello from the two models

    # print('Le Model Forward:')
    # start_fw = 'Bonjour, '
    # tkns_fw = le_model_fw.encode(start_fw)
    # out = le_model_fw.run(1 * [tkns_fw], length=50)
    # print(le_model_fw.decode(out[0]))

    # print('-'*40)

    # print('Le Model Backward:')
    # start_bw = ' ,ruojnoB'
    # tkns_bw = le_model_bw.encode(start_bw)
    # out = le_model_bw.run(1 * [tkns_bw], length=50)
    # print(le_model_bw.decode(out[0]))

    # ----------------------------------------
    # # perplexity tests on sentences
    # print('-'*40)
    # print("Perplexity tests:")

    # le_model_fw.perp_test(sentence="Hello, my name is Mickey Mouse !")
    # le_model_fw.perp_test(sentence="Hello world!")
    # le_model_fw.perp_test(sentence="Hello, i23h5 wadhrwe 5h")
    # le_model_fw.perp_test(sentence="Hello, i23h5 tomorrow 5h")
    # le_model_fw.perp_test(sentence="Bonjour, i23h5 demain 5h")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Macron.")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne.")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Mon père est Poseidon.")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Je ne sais pas pourquoi je dis ce que je dis, mais je le dis.")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Mon nom est Licorne. Mon nom est Licorne.  Mon nom est Licorne. Mon nom est Licorne. Mon nom est Licorne.")
    # le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Je ne sais pas pourquoi je dis ce que je dis, mais je le dis. Dans tous les cas, il me semble important de continuer à parler autant que faire se peut, même si parfois le sens se perd dans les tourbillons.")
    # le_model_fw.perp_test(sentence="Bonjour, mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon. mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon. mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon.")
    # le_model_fw.perp_test(sentence="A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. ")
    # le_model_fw.perp_test(sentence="""L'homme qui, sur le trottoir, attendait l'omnibus Batignolles-Clichy-Odéon en même temps que moi, certainement je le connaissais, mais où l'avais-je vu, et comment s'appelait-il ? Cruelle énigme ! Sans être un jeune homme, c'était un homme jeune encore.  Ses traits, ses façons, toute son allure indiquaient un personnage inquiet, susceptible et ronchonneur.  Enfin l'omnibus arriva.  """)
