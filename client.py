from base64 import b64encode
from threading import Lock
from gpt import Model
import numpy as np
import socketio
import argparse
import textwrap
import random
import regex
import time

# for random_threshold arg below
# https://stackoverflow.com/a/12117065
def float_range(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x


parser = argparse.ArgumentParser(
    description="""
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model",
    type=str,
    default="117M",
    help="Model for forward model. Defaults to '117M'.",
)

parser.add_argument(
    "--run_name",
    type=str,
    default="run1",
    help="Run name for forward model. Defaults to 'run1'.",
)

parser.add_argument(
    "--server_name",
    type=str,
    default="Le Serveur",
    help="Server name used in message.",
)

parser.add_argument("--new", action="store_true", help="Use the new generate function.")

parser.add_argument(
    "--agent", action="store_true", help="""Make the bot generate text autonomously""",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="""Number of sentences generated in parallel.  Defaults to 1.""",
)

parser.add_argument(
    "--temperature", type=float, default=0.9, help="Temperature when sampling.",
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.998,
    help="""Nucleus sampling when sampling: limit sampling to the most likely
    tokens the combined probability of which is at most p (sometimes the
    combined probability of a few tokens reaches p (only a few likely choices),
    sometimes many thousands are needed to reach the same p (high uncertainty /
    many possible choices). Defaults to 0.998 (1 to neutralise).""",
)

parser.add_argument(
    "--top_k",
    type=int,
    default=0,
    help="""Limit sampled tokens to the k most
    likely ones. Defaults to 0 (deactivated).""",
)

parser.add_argument(
    "--print_speed",
    type=float,
    default=0.1,
    help="Length of pause for each step of interactive print loop, in ms.",
)

parser.add_argument(
    "--local", action="store_true", help="Run with local server, port 5100.",
)

parser.add_argument(
    "--heroku",
    action="store_true",
    help="Run with heroku. Default: https://spark.theatrophone.fr/.",
)

parser.add_argument(
    "--length_desired",
    type=int,
    default=500,
    help="Rank under which sentences are allowed to be sent. Defaults to 25.",
)

parser.add_argument(
    "--random_threshold",
    type=float_range,
    default=0.0,
    help="""A random number between 0 and 1 is generated each time the network
    receives a new message. If the number is above the threshold, the network
    answers. Must lie withinin [0:1]. Defaults to 0 (the network answering
    mechanism is fired every time).""",
)

parser.add_argument(
    "--rank_threshold",
    type=int,
    default=25,
    help="Rank under which sentences are allowed to be sent. Defaults to 25.",
)

parser.add_argument(
    "--character",
    type=str,
    default="",
    help="Character used by the network when answering. Defaults to none.",
)

parser.add_argument(
    "--hidden_before_char",
    type=str,
    default="",
    help="""Additional text inserted at the end of each received message, before
    the network produces the next character & answer (can
    influence the overall theme and stabilise the network). Defaults to
    nothing.""",
)

parser.add_argument(
    "--hidden_after_char",
    type=str,
    default="",
    help="""Additional text inserted at the start of each produced message,
    after the character (influences the current answer). If the character is
    not artificially set as well, this start of line becomes the same as the
    hidden_before_char: the text is added at the end of the received messages,
    and the network is free to produce any answer (coloured, however, by the
    context). Defaults to nothing.""",
)

args = parser.parse_args()

sio = socketio.Client(logger=False, reconnection_delay_max=50)


def print_underlined(msg):
    print(msg)
    print("-" * len(msg))


def print_config():
    print("-" * 40)
    print("(settings:)")
    print()
    for k, v in vars(args).items():
        print(f"- {k}: {v}")
    print()


def pprint(
    msg, width=40, off="", sep="", sep_aft="", sp_bf=False, sp_aft=False, und=False
):
    if sp_bf:
        print()
    if sep:
        print(off + sep * width)
    if und:
        print(f"{off}{msg}")
        print(off + "-" * len(msg))
    else:
        print(
            "\n".join(
                textwrap.wrap(
                    msg, width=width, initial_indent=off, subsequent_indent=off
                )
            )
        )
    if sep_aft:
        print(off + sep_aft * width)
    if sp_aft:
        print()


LeLocle = Lock()

le_model = Model(
    model_name=args.model,
    run_name=args.run_name,
    batch_size=args.batch_size,
    special_tokens=["<|endoftext|>"]
    if not (args.new or args.agent)
    else ["<|s|>", "<|e|>", "<|endoftext|>"],
)

RESETTING_SESSION = False
IS_GENERATING = False

REPLIQUE_RE = regex.compile("<\|s\|>\n(.*?)\n+<\|e\|>", regex.DOTALL)
SEPARATORS = "\n<|e|>\n<|s|>\n"
END = "\n<|e|>\n"
START = "<|s|>\n"

MESSAGES = []
PREFIX = ""

TKNS = np.array([], dtype=np.int32)

print_config()


def fancy_typing(char, message):
    if args.print_speed > 0:
        for i in range(len(message) + 1):
            if should_sess_be_reset():
                return
            # print({ "id": sio.sid, "character": char, "message": # message[:i], "user": args.server_name})
            send_typing(
                {
                    "id": sio.sid,
                    "character": char,
                    "message": message[:i],
                    "user": args.server_name,
                }
            )
            time.sleep(args.print_speed)


def generate_new():

    global IS_GENERATING
    global TKNS

    # pprint("(tkns)", sep="-", sp_bf=True, sp_aft=True)
    # print(TKNS)

    pprint("(prefix)", sp_bf=True, off="\t\t\t", sp_aft=True)
    pprint(le_model.decode(TKNS), off="\t\t\t", sp_aft=True)

    IS_GENERATING = True

    if should_sess_be_reset():
        return

    # character & other injections
    if args.hidden_before_char:
        args.hidden_before_ch = args.hidden_before_char.strip()
        with LeLocle:
            TKNS = np.concatenate(
                (TKNS, le_model.encode(f"{args.hidden_before_char}\n"))
            )

    if not args.character and args.hidden_after_char:
        args.hidden_after_char = args.hidden_after_char.strip()
        with LeLocle:
            TKNS = np.concatenate(
                (TKNS, le_model.encode(f"{args.hidden_after_char}\n"))
            )

    end_pref = len(TKNS)

    if args.character:
        args.character = args.character.strip()
        char_encoded = le_model.encode(f"{args.character}\n")
        with LeLocle:
            TKNS = np.concatenate((TKNS, char_encoded))
        end_pref_after_injections = end_pref + len(char_encoded)
        if args.hidden_after_char:
            args.hidden_after_char = args.hidden_after_char.strip()
            after_char_encoded = le_model.encode(f"{args.hidden_after_char}")
            with LeLocle:
                TKNS = np.concatenate(
                    (TKNS, after_char_encoded)
                )
            end_pref_after_injections += len(after_char_encoded)

    data = le_model.gen_avoiding(
        TKNS,
        avoiding=le_model.encode("<|e|>"),
        length=10,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    tkns_batch = data["tokens"]

    pprint("(gen avoiding)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
    for i, seq in enumerate(data["sequences"]):
        pprint(seq, off="\t\t")
        pprint(f"| {data['perplexities'][i].item()} (perp)", off="\t\t", sep_aft="*")

    with LeLocle:
        TKNS = tkns_batch

    data = le_model.gen_until(
        prefix=TKNS,
        until="<|s|>",
        exclude_until=False,
        sanity_limit=300,
        chunk_length=5,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    tkns_batch = data["tokens"]

    # pprint("(generated: with hidden)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
    # for i, seq in enumerate(data["sequences"]):
    #     pprint(seq, off="\t\t")
    #     pprint(f"| {data['perplexities'][i].item()} (perp)", off="\t\t", sep_aft="*")

    if args.character:
        generated = []
        for tkns in tkns_batch:
            tmp_seq = le_model.decode(
                np.concatenate((char_encoded, tkns[end_pref_after_injections:]))
            )
            tmp_seq = tmp_seq.strip()
            generated.append(tmp_seq)
    else:
        generated = [
            seq.strip()
            for seq in le_model.decode([tkns[end_pref:] for tkns in tkns_batch])
        ]

    pprint("(generated)", off="\t", sep="-", sp_bf=True, sp_aft=True)

    chars = []
    messages = []
    for i, g in enumerate(generated):
        if g.find("\n") == -1:
            char = ""
            message = g
        else:
            char = g[: g.find("\n")]
            message = g[g.find("\n") + 1 :].strip()

        pprint(char, off="\t")
        pprint(message, off="\t")
        pprint(f"{data['perplexities'][i].item()} (perp)", off="\t", sep_aft="*")

        chars.append(char)
        messages.append(message)

    send_batch({
        "id": sio.sid,
        "chars": chars,
        "messages": messages,
        "perplexities": data["perplexities"].tolist(),
    })

    min_ind = np.argmin(data["perplexities"])
    char = chars[min_ind]
    message = messages[min_ind]
    pprint("(sent)", sep="-", sp_bf=True)
    pprint(char)
    pprint(message)
    pprint(f"| {np.min(data['perplexities'])} (perp)")

    fancy_typing(char, message)

    if should_sess_be_reset():
        return

    send_message({"character": char, "message": message, "user": args.server_name})

    with LeLocle:
        TKNS = tkns_batch[0]

    # else:
    #     pprint("(RANK INSUFFICIENT: NOT ANSWERING)", off="\t", sp_bf=True, sp_aft=True)

    IS_GENERATING = False
    if should_sess_be_reset():
        return


def generate():

    global IS_GENERATING
    global PREFIX
    global START

    IS_GENERATING = True

    if should_sess_be_reset():
        return

    # print("-"*40)
    # print("(prefix:)")
    # print(PREFIX)
    # print("-"*40)

    prefix_enc = le_model.encode(PREFIX)
    max_len = 1024 - args.length_desired
    if len(prefix_enc) > max_len:
        prefix_enc = prefix_enc[-max_len:]
        PREFIX = le_model.decode(prefix_enc)

    # add end of answer, store length of prefix
    end_pref = len(PREFIX)

    l = le_model.gen(
        prefix=PREFIX,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        length=args.length_desired,
    )["sequences"][0]
    generated = l[end_pref:]

    pprint("(raw)", off="\t\t\t", sep="-", sp_bf=True, sp_aft=True)
    pprint(generated, off="\t\t\t")

    r = regex.search(REPLIQUE_RE, generated)

    if r:
        repl = r.group(1)
        if repl.find("\n") == -1:
            char = ""
            message = regex.sub("<\|[es]\|>", "", repl)
        else:
            char = regex.sub("<\|[es]\|>", "", repl[: repl.find("\n")])
            message = regex.sub("<\|[es]\|>", "", repl[repl.find("\n") + 1 :])
        if args.character:
            char = args.character

        pprint("(generated)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
        pprint(repl, off="\t\t")

        pprint("(char)", off="\t", sep="-", sp_bf=True, sp_aft=True)
        pprint(char, off="\t", sp_aft=True)
        pprint("(message)", off="\t", sp_aft=True)
        pprint(message, off="\t")

        prefix_repl = f"{PREFIX} \u001b[31m|\u001b[0m {r.group(0)}"
        pprint(prefix_repl, off="\t")

        prefix_rank = le_model.get_rank(PREFIX)["ranks_mean"][0]
        prefix_repl_rank = le_model.get_rank(prefix_repl)["ranks_mean"][0]
        le_rank = le_model.get_rank(repl)["ranks_mean"][0]

        pprint(f"(prefix rank: {prefix_rank})", off="\t")
        pprint(f"(prefix & repl rank: {prefix_repl_rank})", off="\t")
        pprint(f"(rank: {le_rank})", off="\t")

        if le_rank < args.rank_threshold:
            pprint("(sent)", sep="-", sp_bf=True, sp_aft=True)
            pprint(char)
            pprint(message)

            fancy_typing(char, message)

            if should_sess_be_reset():
                return

            send_message(
                {"character": char, "message": message, "user": args.server_name}
            )
            PREFIX = f"{PREFIX}{START}{char}\n{message}"
        else:
            pprint(
                "(RANK INSUFFICIENT: NOT ANSWERING)", off="\t", sp_bf=True, sp_aft=True
            )
    else:
        pprint("(picked:)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
        pprint(l, off="\t\t", sp_aft=True)
        pprint("(MARKERS NOT FOUND: NOT ANSWERING)", off="\t\t")

    IS_GENERATING = False
    if should_sess_be_reset():
        return


def should_sess_be_reset():
    global RESETTING_SESSION
    global IS_GENERATING
    if RESETTING_SESSION:
        print(f"generation interrupted")
        print()
        print("=" * 40)
        IS_GENERATING = False
        RESETTING_SESSION = False
        return True


@sio.event
def connect():
    global TKNS
    print("connection established")
    print("-" * 40)
    sio.emit("new bot", args.server_name)
    if args.agent:
        with LeLocle:
            TKNS = le_model.encode("\n<|e|>\n<|s|>\n")
        while True:
            generate_new()
            time.sleep(10)


@sio.event
def connect_error(e):
    print("connection failed")
    print(e)


@sio.event
def disconnect():
    print("connection lost")
    print("-" * 40)


@sio.on("erase messages")
def reset_session():

    global RESETTING_SESSION
    global MESSAGES
    global PREFIX
    global TKNS

    print()
    print("=" * 40)
    print()
    print("resetting session")

    MESSAGES = []
    PREFIX = ""
    TKNS = np.array([], dtype=np.int32)
    if IS_GENERATING:
        RESETTING_SESSION = True
    else:
        print()
        print("=" * 40)


@sio.on("received")
def on_chat_message(data):

    global IS_GENERATING
    global MESSAGES
    global PREFIX
    global TKNS

    char = data["character"]
    msg = data["message"]

    pprint("(received)", off="\t", sep="-", sp_bf=True, sp_aft=True)
    if data["character"]:
        pprint(f"{char}", off="\t")
    if data["message"]:
        pprint(f"{msg}", off="\t", sp_aft=True)

    MESSAGES.append(data)
    character = data["character"]
    message = data["message"]
    if character:
        PREFIX = f"{PREFIX}{SEPARATORS}{character}\n{message}{END}"
        with LeLocle:
            TKNS = np.concatenate((TKNS, le_model.encode(f"{character}\n{message}")))
    else:
        PREFIX = f"{PREFIX}{SEPARATORS}{message}{END}"
        with LeLocle:
            TKNS = np.concatenate((TKNS, le_model.encode(f"{message}")))

    with LeLocle:
        TKNS = np.concatenate((TKNS, le_model.encode(f"{SEPARATORS}")))

    # pprint("(after reception, TKNS are now:)", sep="-", sp_bf=True)
    # print(TKNS[0], type(TKNS[0]))
    # pprint(le_model.decode(TKNS)[0], sp_aft=True)

    rand = random.random()
    pprint(f"(random has spoken: {rand})", off="\t", sp_bf=True)
    if not IS_GENERATING:
        if rand > args.random_threshold:
            pprint("(random is bountiful, let's generate)", off="\t", sp_aft=True)
            if args.new:
                generate_new()
            elif args.agent:
                pass
            else:
                generate()
    else:
        pprint("(is generating, not answering...)", off="\t", sp_aft=True)


@sio.on("get bot config")
def send_config():
    sio.emit(
        "config from bot",
        {
            "id": sio.sid,
            "user": args.server_name,
            "model": args.model,
            "run": args.run_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "print_speed": args.print_speed,
            "length_desired": args.length_desired,
            "random_threshold": args.random_threshold,
            "rank_threshold": args.rank_threshold,
            "character": args.character,
            "hidden_before_char": args.hidden_before_char,
            "hidden_after_char": args.hidden_after_char,
        },
    )


@sio.on("server sets bot config")
def set_config(data):
    if data["id"] == sio.sid:
        pprint("received config:", sep="-", sp_bf=True, und=True)
        for k, v in data.items():
            if k in {"user", "id", "run", "model"}:
                continue
            try:
                v = type(args.__getattribute__(k))(v)
                print(f"{k}: {v}")
                args.__setattr__(k, v)
            except:
                print(
                    f"\033[31m!! {k}: cannot cast {v} to {type(args.__getattribute__(k))}, ignoring...\033[0m"
                )
                continue
        print()


def send_typing(data):
    sio.emit("typing", data)

def send_batch(data):
    sio.emit("chat batch", data)

def send_message(data):
    # print("-"*40)
    # print("sending message:")
    # print(data)
    # print("-"*40)
    sio.emit("chat message", data)


user_pass = b64encode(b"username:password").decode("ascii")
if args.local:
    url = "http://localhost:5100"
    print("-" * 40)
    print(f"connecting to: {url}")
    sio.connect(url)
else:
    if args.heroku:
        url = "***HEROKU WEB ADDRESS***"
        print("-" * 40)
        print(f"connecting to: {url}")
        sio.connect(url)
    else:
        user_pass = b64encode(b"username:password").decode("ascii")
        url = "https://spark.theatrophone.fr"
        print("-" * 40)
        print(f"connecting to: {url}")
        sio.connect(url, {"Authorization": "Basic %s" % user_pass})
sio.wait()
