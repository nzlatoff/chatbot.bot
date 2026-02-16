# CHATBOT: BOT

The Llama-cpp-python wrapper class & web client for loading a model and generate from it when connected to a [web server](https://github.com/nzlatoff/chatbot.interface). This work is a continuation from the project [CHATBOT](http://www.manufacture.ch/fr/4467/CHATBOT-jouer-et-dialoguer-avec-un-agent-conversationnel-acteur), led by [Nicolas Zlatoff](http://www.manufacture.ch/en/1695/Nicolas-Zlatoff) at the [Manufacture University of Performing Arts](http://www.manufacture.ch/en/) in Lausanne, Switzerland.

## Installation

Clone this repo somewhere on your machine:

```bash
git clone chatbot.bot
```

Then:

```bash
poetry install
```

This will also install llama.cpp as llama-cpp-python relies on it. Make sure to compile llama-cpp with the options according to your setup.

## Use

Without the [web server](https://github.com/nzlatoff/chatbot.interface) the client won't work. You will need to set host of the web server in an environment variable named `CHATBOST_HOST` and you will find an example in  `.env.template` ([here](https://github.com/nzlatoff/chatbot.bot/blob/master/.env.template)). It is also possible to run it locally, in one terminal window, and to pass `--local` to the client. The server we used required an authorization with username and password, located in the sampe place, which should be changed or removed.

```
usage: client_llama_cpp.py [-h] [--model MODEL] [--run_name RUN_NAME] [--server_name SERVER_NAME] [--mode {legacy,reactive,autonomous,optimizer}] [--base] [--tts]
                           [--batch_size BATCH_SIZE] [--n_gpu_layers N_GPU_LAYERS] [--temperature TEMPERATURE] [--top_p TOP_P] [--top_k TOP_K] [--tempo TEMPO] [--local]
                           [--length_desired LENGTH_DESIRED] [--silence SILENCE] [--pause PAUSE] [--rank_threshold RANK_THRESHOLD] [--character CHARACTER] [--subtext SUBTEXT]
                           [--first_words FIRST_WORDS] [--wait WAIT] [--bot_choice {sampling,min,max}] [--patience PATIENCE] [--limit_prefix LIMIT_PREFIX]
                           [--chunk_length CHUNK_LENGTH]

options:
  -h, --help            show this help message and exit
  --model MODEL         Model path (params). (default: 117M)
  --run_name RUN_NAME   Run name path (weights). (default: /home/spark/dev/models/mtext-050625_mistral-7B-v0.3_merged.Q8_0.gguf)
  --server_name SERVER_NAME
                        Server name used in message. (default: lActrice)
  --mode {legacy,reactive,autonomous,optimizer}
  --base                Run with base model (full completion without <|s|> and <|e|> separators) (default: False)
  --tts                 send text to tts if specified (default: False)
  --batch_size BATCH_SIZE
                        Number of sentences generated in parallel. (default: 1)
  --n_gpu_layers N_GPU_LAYERS
                        Number of GPU layers used when loading model (default: -1)
  --temperature TEMPERATURE
                        Temperature when sampling. (default: 0.75)
  --top_p TOP_P         Nucleus sampling when sampling: limit sampling to the most likely tokens the combined probability of which is at most p (sometimes the combined probability of
                        a few tokens reaches p (only a few likely choices), sometimes many thousands are needed to reach the same p (high uncertainty / many possible choices). (1 to
                        neutralise). (default: 0.95)
  --top_k TOP_K         Limit sampled tokens to the k most likely ones. (0 to neutralise). (default: 50)
  --tempo TEMPO         Length of pause for each step of interactive print loop, in ms. (default: 0.25)
  --local               Run with local server, port 5100. (default: False)
  --length_desired LENGTH_DESIRED
                        LEGACY ONLY (before end tokens were introduced). Length of text before the bot stops. (default: 500)
  --silence SILENCE     A random number between 0 and 1 is generated each time the network receives a new message. If the number is above the silence, the network answers. Must lie
                        withinin [0:1]. (when set to 0 the network answering mechanism is fired every time). (default: 0.0)
  --pause PAUSE         The most time the bot sleeps between each new attempt to produce text (for autonomous & optimizer modes. A random number is generated, between 1 and pause,
                        before each call of the generate function. (default: 10)
  --rank_threshold RANK_THRESHOLD
                        Rank under which sentences are allowed to be sent. (default: 25)
  --character CHARACTER
                        Character used by the network when answering. (default: )
  --subtext SUBTEXT     Additional text inserted at the end of each received message, before the network produces the next character & answer (can influence the overall theme and
                        stabilise the network). (default: )
  --first_words FIRST_WORDS
                        Additional text inserted at the start of each produced message, after the character (influences the current answer). If the character is not artificially set
                        as well, this start of line becomes the same as the subtext: the text is added at the end of the received messages, and the network is free to produce any
                        answer (coloured, however, by the context).. (default: )
  --wait WAIT           Waiting time before activating the default message choice. When generating with more than one message per batch, the /master can choose the message, in the
                        time frame here defined, after which the standard choice process kicks in. (0 means no wait, the master sets it from /master). (default: 0)
  --bot_choice {sampling,min,max}
                        The bot's method for choosing messages. Available options: sampling (random choice weighted by the perplexities of the messages), min (perplexity), max
                        (perplexity) (default: sampling)
  --patience PATIENCE   Number of times the bot tolerates not to beat its own record (in optimizer mode). It keeps generating batches of sentences, keeping only the n best ones
                        (lowest perplexity). At first the bot is able to produce sentences with a better perplexity rather easily, but as the best ones are being saved each round, it
                        becomes ever more rare that it is even able to produce a new sentence that makes it into the n best ones. Patience is the number of times the bot is allowed
                        *not* to produce any new sentence making it into the n best ones: once this happens, the batch is submitted (either to be directly posted, or evaluated by the
                        master. (default: 3)
  --limit_prefix LIMIT_PREFIX
                        Preemptively limit the length of the prefix, to avoid OOM issues. (default: 3900)
  --chunk_length CHUNK_LENGTH
                        Number of tokens requested during the gradual generation loop. (default: 5)
```

Note: the `--mode legacy` is intended to work with the [legacy web interface](https://github.com/jchwenger/chatbot.legacy) (but on a web server requiring a GPU, or only locally, and without special tokens, hence the new version). This was left aside as a work in progress, and is not guaranteed to work.

## Web Client

The Web Client has been conceived in partnership with the [server repo](https://github.com/nzlatoff/chatbot.interface). See this repo for how the below functionalities are integrated.


