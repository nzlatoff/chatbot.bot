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
    def __init__(self, run_name="run1", device="/GPU:0", batch_size=1):
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
            self.output = sample.sample_sequence(
                hparams=self.hparams,
                length=self.length,
                start_token=None,
                context=self.context,
                batch_size=None,
                temperature=self.temperature,
                top_k=0.0,
                top_p=0.0,
            )

        # spit out all these warnrings
        self.dummy_run()

    def load_checkpoint(self, path="checkpoint/run1"):
        self.ckpt = tf.train.latest_checkpoint(path)
        self.saver = tf.compat.v1.train.Saver(allow_empty=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print(f"Loading checkpoint {self.ckpt}")
        self.saver.restore(self.sess, self.ckpt)

    def run(
        self, prefix="<|endoftext|>", length=5, temperature=1,
    ):
        context_tokens = 1 * [self.enc.encode(prefix)]
        return self.enc.decode(self.sess.run(
            self.output,
            feed_dict={
                self.length: length,
                self.context: context_tokens,
                self.temperature: temperature,
            },
        )[0])

    def dummy_run(self):
        self.run("A", length=1)

    # the two following functions adapted from here:
    # https://github.com/gpt2ent/gpt-2-simple/blob/652fdab80131ce83f8f1b6fd00f597dd48ae2e36/gpt_2_simple/gpt_2.py#L504

    def get_logits(self, context_tokens, batch_size=1, last_only=True):

        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])

        def step(hparams, tokens, past=None):
            lm_output = model.model(
                hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE
            )

            logits = lm_output["logits"][:, :, : hparams.n_vocab]
            presents = lm_output["present"]
            presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
            return {
                "logits": logits,
                "presents": presents,
            }

        output = step(self.hparams, context)

        out = self.sess.run(output, feed_dict={context: batch_size * [context_tokens]})

        # all logits starting from the second token, n logits for n tokens
        if not last_only:
            return out["logits"][0, :, :]
        return out["logits"][0, -1, :]  # logits for next token

    def get_perplexity(
        self, batch_size=1, sentence="<|endoftext|>", verbose=False
    ):

        tkns = self.enc.encode(sentence)
        len_tkns = len(tkns)
        logits = self.get_logits(
            context_tokens=tkns, batch_size=batch_size, last_only=False
        )

        # don't take the last one (predicting the token after our sentence)
        # print(f"logits before shortening: {logits.shape}")
        logits = logits[:-1, :]
        # normalizing logits for numerical stability
        # (does not affect the result)
        mu = np.mean(logits, axis=1)
        lm = logits - mu[:, None]
        le = np.exp(lm)
        logprobs = le / np.sum(le, axis=1)[:, None]

        if verbose:
            print(f"sentence: (len: {len_tkns:2}): {tkns}")
            print(f"logit shape:    {len(logits.shape)}")
            print(f"logprobs shape: {logprobs.shape}")
            print()
            for i, tkn in enumerate(tkns[1:]):
                print(f"{i:3}: token: {tkn:5} | probability: {logprobs[i, tkn]}")
            print()

        scores = np.nan_to_num(
            [logprobs[i, index] for i, index in enumerate(tkns[1:])]
        )
        perplexity = 2 ** (-np.mean(np.log2(scores)))
        return perplexity

    def perp_test(self, sentence, verbose=False):
        print(f"sentence:   {sentence}")
        print(f"perplexity: {self.get_perplexity(sentence=sentence, verbose=verbose)}")
        print()


if __name__ == '__main__':
    le_model_fw = Model(run_name='run1')
    # le_model_bw = Model(run_name='run1')

    # print('Le Model Forward:')
    # start_fw = 'Bonjour, '
    # tkns_fw = le_model_fw.enc.encode(start_fw)
    # out = le_model_fw.run(1 * [tkns_fw], length=50)
    # print(le_model_fw.enc.decode(out[0]))

    # print('-'*40)

    # print('Le Model Backward:')
    # start_bw = ' ,ruojnoB'
    # tkns_bw = le_model_bw.enc.encode(start_bw)
    # out = le_model_bw.run(1 * [tkns_bw], length=50)
    # print(le_model_bw.enc.decode(out[0]))

    print('-'*40)
    print("Perplexity tests:")

    le_model_fw.perp_test(sentence="Hello, my name is Mickey Mouse !")
    le_model_fw.perp_test(sentence="Hello world!")
    le_model_fw.perp_test(sentence="Hello, i23h5 wadhrwe 5h")
    le_model_fw.perp_test(sentence="Hello, i23h5 tomorrow 5h")
    le_model_fw.perp_test(sentence="Bonjour, i23h5 demain 5h")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Macron.")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne.")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Mon père est Poseidon.")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Je ne sais pas pourquoi je dis ce que je dis, mais je le dis.")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Mon nom est Licorne. Mon nom est Licorne.  Mon nom est Licorne. Mon nom est Licorne. Mon nom est Licorne.")
    le_model_fw.perp_test(sentence="Bonjour, mon nom est Licorne. Je ne sais pas pourquoi je dis ce que je dis, mais je le dis. Dans tous les cas, il me semble important de continuer à parler autant que faire se peut, même si parfois le sens se perd dans les tourbillons.")
    le_model_fw.perp_test(sentence="Bonjour, mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon. mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon. mon. mon. mon. mon.  mon. mon. mon. mon. mon. mon.")
    le_model_fw.perp_test(sentence="A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. ")
    le_model_fw.perp_test(sentence="""L'homme qui, sur le trottoir, attendait l'omnibus Batignolles-Clichy-Odéon en même temps que moi, certainement je le connaissais, mais où l'avais-je vu, et comment s'appelait-il ? Cruelle énigme ! Sans être un jeune homme, c'était un homme jeune encore.  Ses traits, ses façons, toute son allure indiquaient un personnage inquiet, susceptible et ronchonneur.  Enfin l'omnibus arriva.  """)
