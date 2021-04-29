# CHATBOT: GPT-2 Wrapper & Web Client

## Installation

This repository works in close partnership with a [fork of GPT-2](https://github.com/jchwenger/gpt-2/tree/chatbot). Pull this repo somewhere on your machine, for instance in the same directory as this one. The two repositories work hand in hand, here is the recommended folder structure:

```bash
chatbot/
|- gpt-2/
|- bot/
```

The main bulk of the installation consists in getting Tensorflow 1.4 to work. Please follow the installation procedure on [fork of GPT-2](https://github.com/jchwenger/gpt-2/tree/chatbot/README.md#Installation) that was used for the project. Once the installation is done, you can activate the conda environment like so:

```bash
conda activate chatbot
```

First, install the gpt-2 folder and the models (see the repo for instructions). Going back to `bot/`, add the following symlinks:

```bash
ln -s ../gpt-2/models
ln -s ../gpt-2/checkpoint
ln -s ../gpt-2/src
```

## Use

**Important note: don't forget to add the path to the /src folder using `PYTHONPATH=src`**

```
$ PYTHONPATH=src python client.py --help
usage: client.py [-h] [--model MODEL] [--run_name RUN_NAME]
                 [--server_name SERVER_NAME]
                 [--mode {legacy,reactive,autonomous,optimizer}]
                 [--device DEVICE] [--batch_size BATCH_SIZE]
                 [--temperature TEMPERATURE] [--top_p TOP_P] [--top_k TOP_K]
                 [--tempo TEMPO] [--local] [--heroku]
                 [--length_desired LENGTH_DESIRED] [--silence SILENCE]
                 [--pause PAUSE] [--rank_threshold RANK_THRESHOLD]
                 [--character CHARACTER] [--subtext SUBTEXT]
                 [--first_words FIRST_WORDS] [--wait WAIT]
                 [--bot_choice {sampling,min,max}] [--patience PATIENCE]
                 [--limit_prefix LIMIT_PREFIX]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model for forward model. Defaults to '117M'. (default:
                        117M)
  --run_name RUN_NAME   Run name for forward model. Defaults to 'run1'.
                        (default: run1)
  --server_name SERVER_NAME
                        Server name used in message. (default: Le Serveur)
  --mode {legacy,reactive,autonomous,optimizer}
  --device DEVICE       The GPU on which the net will be run. Default: /GPU:0.
                        (default: /GPU:0)
  --batch_size BATCH_SIZE
                        Number of sentences generated in parallel. Defaults to
                        1. (default: 1)
  --temperature TEMPERATURE
                        Temperature when sampling. (default: 0.9)
  --top_p TOP_P         Nucleus sampling when sampling: limit sampling to the
                        most likely tokens the combined probability of which
                        is at most p (sometimes the combined probability of a
                        few tokens reaches p (only a few likely choices),
                        sometimes many thousands are needed to reach the same
                        p (high uncertainty / many possible choices). Defaults
                        to 0.998 (1 to neutralise). (default: 0.998)
  --top_k TOP_K         Limit sampled tokens to the k most likely ones.
                        Defaults to 0 (deactivated). (default: 0)
  --tempo TEMPO         Length of pause for each step of interactive print
                        loop, in ms. (default: 0.1)
  --local               Run with local server, port 5100. (default: False)
  --heroku              Run with heroku. Default:
                        https://spark.theatrophone.fr/. (default: False)
  --length_desired LENGTH_DESIRED
                        Rank under which sentences are allowed to be sent.
                        Defaults to 25. (default: 500)
  --silence SILENCE     A random number between 0 and 1 is generated each time
                        the network receives a new message. If the number is
                        above the silence, the network answers. Must lie
                        withinin [0:1]. Defaults to 0 (the network answering
                        mechanism is fired every time). (default: 0.0)
  --pause PAUSE         The most time the bot sleeps between each new attempt
                        to produce text (for autonomous & optimizer modes. A
                        random number is generated, between 1 and pause,
                        before each call of the generate function. (default:
                        10)
  --rank_threshold RANK_THRESHOLD
                        Rank under which sentences are allowed to be sent.
                        Defaults to 25. (default: 25)
  --character CHARACTER
                        Character used by the network when answering. Defaults
                        to none. (default: )
  --subtext SUBTEXT     Additional text inserted at the end of each received
                        message, before the network produces the next
                        character & answer (can influence the overall theme
                        and stabilise the network). Defaults to nothing.
                        (default: )
  --first_words FIRST_WORDS
                        Additional text inserted at the start of each produced
                        message, after the character (influences the current
                        answer). If the character is not artificially set as
                        well, this start of line becomes the same as the
                        subtext: the text is added at the end of the received
                        messages, and the network is free to produce any
                        answer (coloured, however, by the context). Defaults
                        to nothing. (default: )
  --wait WAIT           Waiting time before activating the default message
                        choice. When generating with more than one message per
                        batch, the /master can choose the message, in the time
                        frame here defined, after which the standard choice
                        process kicks in. Defaults to 0 seconds (no wait, the
                        master sets it from /master). (default: 0)
  --bot_choice {sampling,min,max}
                        The bot's method for choosing messages. Available
                        options: sampling (random choice weighted by the
                        perplexities of the messages), min (perplexity), max
                        (perplexity) (default: sampling)
  --patience PATIENCE   Number of times the bot tolerates not to beat its own
                        record (in optimizer mode). It keeps generating
                        batches of sentences, keeping only the n best ones
                        (lowest perplexity). At first the bot is able to
                        produce sentences with a better perplexity rather
                        easily, but as the best ones are being saved each
                        round, it becomes ever more rare that it is even able
                        to produce a new sentence that makes it into the n
                        best ones. Patience is the number of times the bot is
                        allowed *not* to produce any new sentence making it
                        into the n best ones: once this happens, the batch is
                        submitted (either to be directly posted, or evaluated
                        by the master. Default: 3. (default: 3)
  --limit_prefix LIMIT_PREFIX
                        Preemptively limit the length of the prefix, to avoid
                        OOM issues. Defaults to 200. (default: 200)

```

## GPT-2 Wrapper

## Web Client
