from base64 import b64encode
from gpt import Model
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
    many possible choices). Defaults to 0.998 (1 to neutralise).""",)

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


def pprint(msg, width=40, off="", sep="", sp_bf=False, sp_aft=False, und=False):
    if sp_bf:
        print()
    if sep:
        print(off + sep * width)
    if und:
        print(f"{off}{msg}")
        print(off + "-" * len(msg))
    else:
        print(
            "\n".join(textwrap.wrap(msg, width=width, initial_indent=off, subsequent_indent=off))
        )
    if sp_aft:
        print()


le_model = Model(model_name=args.model, run_name=args.run_name)

RESETTING_SESSION = False
IS_GENERATING = False

REPLIQUE_RE = regex.compile("<\|s\|>\n(.*?)\n+<\|e\|>", regex.DOTALL)
SEPARATORS = "\n<|e|>\n<|s|>\n"
END = "\n<|e|>\n"
START = "<|s|>\n"

MESSAGES = []
PREFIX = ""

print_config()


def generate(rank_threshold=25):

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

    # add END of answer, store length of PREFIX
    end_pref = len(PREFIX)

    l = le_model.gen(
        prefix=PREFIX,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        length=args.length_desired,
    )[0]
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

        prefix_rank = le_model.get_rank(PREFIX)[0]
        prefix_repl_rank = le_model.get_rank(prefix_repl)[0]
        le_rank = le_model.get_rank(repl)[0]

        pprint(f"(prefix rank: {prefix_rank})", off="\t")
        pprint(f"(prefix & repl rank: {prefix_repl_rank})", off="\t")
        pprint(f"(rank: {le_rank})", off="\t")

        if le_rank < rank_threshold:
            pprint("(sent)", sep="-", sp_bf=True, sp_aft=True)
            pprint(char)
            pprint(message)

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
    print("connection established")
    print("-" * 40)
    sio.emit("new bot", args.server_name)


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

    print()
    print("=" * 40)
    print()
    print("resetting session")

    MESSAGES = []
    PREFIX = ""
    if IS_GENERATING:
        RESETTING_SESSION = True
    else:
        print()
        print("=" * 40)


@sio.on("received")
def on_chat_message(data):

    global IS_GENERATING
    global PREFIX

    char = data["character"].replace("\n", "\t\n")
    msg = data["message"].replace("\n", "\t\n")

    pprint("(received)", off="\t", sep="-", sp_bf=True, sp_aft=True)
    if data["character"]:
        pprint(f"{char}", off="\t")
    if data["message"]:
        pprint(f"{msg}", off="\t")

    MESSAGES.append(data)
    character = data["character"]
    message = data["message"]
    if character:
        PREFIX = f"{PREFIX}{SEPARATORS}{character}\n{message}{END}"
    else:
        PREFIX = f"{PREFIX}{SEPARATORS}{message}{END}"

    rand = random.random()
    pprint(f"(random has spoken: {rand})", off="\t", sp_bf=True)
    if not IS_GENERATING:
        if rand > args.random_threshold:
            pprint("(random is bountiful, let's generate)", off="\t", sp_aft=True)
            generate(rank_threshold=args.rank_threshold)
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
