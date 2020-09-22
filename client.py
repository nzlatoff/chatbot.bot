from print_utils import print_underlined
from print_utils import print_config
from print_utils import pprint
from base64 import b64encode
from threading import Lock
from gpt import Model
import numpy as np
import socketio
import argparse
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


def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
    if x <= 0:
        raise argparse.ArgumentTypeError(f"{x} must be greater than 0.")
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
    "--mode",
    type=str,
    choices=["legacy", "reactive", "autonomous", "optimizer"],
    default="autonomous",
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
    "--sleepy_time",
    type=positive_float,
    default=10,
    help="""The most time the bot sleeps between each new attempt to produce
    text (for autonomous & optimizer modes. A random number is generated,
    between 1 and sleepy_time, before each call of the generate function.""",
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

parser.add_argument(
    "--wait_for_master",
    type=int,
    default=0,
    help="""Waiting time before activating the default message choice. When
    generating with more than one message per batch, the /master can choose the
    message, in the time frame here defined, after which the standard choice
    process kicks in. Defaults to 0 seconds (no wait, the master sets it from
    /master).""",
)

parser.add_argument(
    "--bot_choice",
    choices=("sampling", "min", "max"),
    type=str,
    default="sampling",
    help="""The bot's method for choosing messages. Available options: sampling
    (random choice weighted by the perplexities of the messages), min
    (perplexity), max (perplexity)""",
)

parser.add_argument(
    "--patience",
    type=int,
    default=3,
    help="""Number of times the bot tolerates not to beat its own record (in
    optimizer mode). It keeps generating batches of sentences, keeping only the
    n best ones (lowest perplexity). At first the bot is able to produce
    sentences with a better perplexity rather easily, but as the best ones
    are being saved each round, it becomes ever more rare that it is even able
    to produce a new sentence that makes it into the n best ones. Patience is
    the number of times the bot is allowed *not* to produce any new sentence
    making it into the n best ones: once this happens, the batch is submitted
    (either to be directly posted, or evaluated by the master. Default: 3.""",
)

parser.add_argument(
    "--limit_prefix",
    type=int,
    default=200,
    help="""Preemptively limit the length of the prefix, to avoid OOM issues.
    Defaults to 200.""",
)

args = parser.parse_args()


sio = socketio.Client(logger=False, reconnection_delay_max=50)

LeLocle = Lock()

le_model = Model(
    model_name=args.model,
    run_name=args.run_name,
    batch_size=args.batch_size,
    special_tokens=["<|endoftext|>"]
    if (args.mode == "legacy")
    else ["<|s|>", "<|e|>", "<|endoftext|>"],
)

# ----------------------------------------
# all the lovely globals

RESETTING = False
IS_GENERATING = False

REPLIQUE_RE = regex.compile("<\|s\|>\n(.*?)\n+<\|e\|>", regex.DOTALL)
SEPARATORS = "\n<|e|>\n<|s|>\n"
END = "\n<|e|>\n"
START = "<|s|>\n"

MESSAGES = []
PREFIX = ""

TKNS = np.array([], dtype=np.int32)
SEP_TKNS = np.array(le_model.encode(SEPARATORS))
SEP_TKNS_LEN = SEP_TKNS.size
RECEIVED_MSGS = np.array([], dtype=np.int32)
BATCH_MSG_IND = None

TKNS_LEN_THRESHOLD = args.limit_prefix

HAS_STARTED = False # for autonomous modes

# ----------------------------------------
# for printing see print_utils.py

print_config(args)

# ----------------------------------------
# utils


def reset_gen():

    global RECEIVED_MSGS
    global BATCH_MSG_IND
    global IS_GENERATING
    global HAS_STARTED
    global RESETTING
    global MESSAGES
    global PREFIX
    global TKNS

    send_three_dots()
    MESSAGES = []
    with LeLocle:
        RECEIVED_MSGS = np.array([], dtype=np.int32)
        IS_GENERATING = False
        BATCH_MSG_IND = None
        HAS_STARTED = False
        RESETTING = False
        TKNS = SEP_TKNS
        PREFIX = ""
    print(f"generation interrupted")
    print()
    print("=" * 40)
    print()
    return False

def index_from_master():

    global BATCH_MSG_IND

    i = 0
    print()
    while BATCH_MSG_IND == None:
        if RESETTING:
            return False
        print(
            f"\t(waiting for batch choice ({args.wait_for_master - i}))", end="     \r"
        )
        time.sleep(1)
        i += 1
        if i > args.wait_for_master + 2:
            print("\twaited enough, bot taking back control")
            with LeLocle:
                BATCH_MSG_IND = -1
    return True


def trim_tok(tkns):
    riddance = {
        le_model.encode(i)[0]
        for i in {" ", "\n", " ", "<|s|>", "<|e|>", "<|endoftext|>"}
    }
    # left trimming
    while tkns[0] in riddance:
        tkns = tkns[1:]
    # right trimming
    while tkns[-1] in riddance:
        tkns = tkns[:-1]
    return tkns


def fancy_tok_typing(tkns):
    pprint(
        f"(Alright, {args.server_name} sending les tokens to humans...)",
        sep="-",
        sp_bf=True,
        sp_aft=True,
    )
    tkns = trim_tok(tkns)
    total = len(tkns) + 1
    nl_ind = np.where(tkns == 201)[0]
    if nl_ind.size == 0:
        nl_ind = 1
    else:
        nl_ind = nl_ind[0]
    prev = ""
    for i in range(nl_ind, total):
        if RESETTING:
            return False
        if nl_ind == 1:
            char = ""
            message = le_model.decode(tkns[:i])
        else:
            char = le_model.decode(tkns[:nl_ind])
            message = le_model.decode(tkns[nl_ind + 1 : i + 2])
        msg = f"{tkns[:i+2]}".split("\n")
        current = msg[-1] + "\r"
        if len(current) < len(prev):
            print()
        print(msg[-1] + "\r", end="")
        prev = current
        send_typing(
            {
                "id": sio.sid,
                "character": char,
                "message": message,
                "user": args.server_name,
            }
        )
        time.sleep(args.print_speed)
    print()
    print()
    return True


def fancy_typing(char, message):
    pprint(
        f"(Awright, {args.server_name} sending le message to humans...)",
        sep="-",
        sp_bf=True,
        sp_aft=True,
    )
    total = len(message) + 1
    for i in range(total):
        if RESETTING:
            return False
        send_typing(
            {
                "id": sio.sid,
                "character": char,
                "message": message[:i],
                "user": args.server_name,
            }
        )
        time.sleep(args.print_speed)
    print()
    print()
    return True


def preprocess_prefix():

    global SEP_TKNS_LEN
    global SEP_TKNS
    global TKNS

    end_pref_orig = len(TKNS)
    end_pref = end_pref_orig
    end_pref_after_injections = end_pref_orig
    len_injections = 0

    pprint("(tkns)", sep="-", off="\t\t\t", sp_bf=True)
    pprint(str(TKNS), off="\t\t\t", sp_aft=True)
    pprint(f"(length: {end_pref_orig})", off="\t\t\t", sp_aft=True)

    pprint("(prefix)", sp_bf=True, off="\t\t\t", sp_aft=True)
    pprint(le_model.decode(TKNS), off="\t\t\t", sp_aft=True)

    if not args.character:

        # if no char injected, inject before markers:
        # - add hidden_after_char
        # - add hidden_after_char
        # - markers

        if args.hidden_after_char:
            args.hidden_after_char = args.hidden_after_char.strip()
            hidden_after_encoded = le_model.encode(f"\n{args.hidden_after_char}")
            len_injections += len(hidden_after_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, hidden_after_encoded))

        if args.hidden_before_char:
            args.hidden_before_ch = args.hidden_before_char.strip()
            hidden_before_encoded = le_model.encode(f"\n{args.hidden_before_char}")
            len_injections += len(hidden_before_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, hidden_before_encoded))

        # markers
        len_injections += SEP_TKNS_LEN
        with LeLocle:
            TKNS = np.concatenate((TKNS, SEP_TKNS))

        end_pref = end_pref_orig + len_injections

    else:

        # if char, add markers, then the rest

        len_injections += SEP_TKNS_LEN
        with LeLocle:
            TKNS = np.concatenate((TKNS, SEP_TKNS))

        if args.hidden_before_char:
            args.hidden_before_ch = args.hidden_before_char.strip()
            hidden_before_encoded = le_model.encode(f"{args.hidden_before_char}\n")
            len_injections += len(hidden_before_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, hidden_before_encoded))

        end_pref = end_pref_orig + len_injections

        args.character = args.character.strip()
        char_encoded = le_model.encode(f"{args.character}\n")
        with LeLocle:
            TKNS = np.concatenate((TKNS, char_encoded))
        end_pref_after_injections = end_pref + len(char_encoded)
        if args.hidden_after_char:
            args.hidden_after_char = args.hidden_after_char.strip()
            after_char_encoded = le_model.encode(f"{args.hidden_after_char} ")
            with LeLocle:
                TKNS = np.concatenate((TKNS, after_char_encoded))
            end_pref_after_injections += len(after_char_encoded)

    return end_pref_orig, end_pref, end_pref_after_injections


def select_in_batch(data, chars, messages):

    global BATCH_MSG_IND

    if BATCH_MSG_IND == -1:
        if args.bot_choice == "sampling":
            # smallest perps given most weight
            invert_perps = 1 - data["perplexities"]
            s = invert_perps.sum()
            normed = np.nan_to_num(invert_perps / s)
            with LeLocle:
                BATCH_MSG_IND = np.random.choice(
                    invert_perps.shape[0], 1, p=normed.flatten()
                ).item()
        elif args.bot_choice == "min":
            with LeLocle:
                BATCH_MSG_IND = np.argmin(data["perplexities"])
        elif args.bot_choice == "max":
            with LeLocle:
                BATCH_MSG_IND = np.argmax(data["perplexities"])
        char = chars[BATCH_MSG_IND]
        message = messages[BATCH_MSG_IND]
        pprint(
            f"({args.server_name} sending its own choice, message: {BATCH_MSG_IND+1})",
            sep="-",
            sp_bf=True,
            sp_aft=True,
        )
        pprint(char)
        pprint(message)
        pprint(f"(perp: {data['perplexities'][BATCH_MSG_IND].item()})")
    else:
        char = chars[BATCH_MSG_IND]
        message = messages[BATCH_MSG_IND]
        pprint(f"({args.server_name} sending le master's choice)", sep="-", sp_bf=True)
        pprint(char)
        pprint(message)
        pprint(f"(perp: {data['perplexities'][BATCH_MSG_IND].item()})")
    return char, message


def handle_error(fn_name, end_pref_orig, e, trimming_factor=5 / 6, sleep_for=5):

    global TKNS_LEN_THRESHOLD
    global TKNS

    send_three_dots()

    pprint(
        f"O.O.O.P.S. A problem ocurred during {fn_name}: {repr(e)}",
        sep="=",
        sp_bf=True,
    )
    pprint(
        f"! PERHAPS DANGEROUS LENGTH REACHED ? Trimming by {trimming_factor}, in case.",
    )

    two_thirds = int(end_pref_orig * trimming_factor)

    old_len = len(TKNS)
    next_len = old_len - 50
    with LeLocle:
        TKNS_LEN_THRESHOLD = next_len if next_len > 10 else old_len

    with LeLocle:
        TKNS = TKNS[two_thirds:]
    pprint(
        f"(Length was {old_len}, is now: {old_len - two_thirds}, capped to {TKNS_LEN_THRESHOLD} from now on, will also sleep for a bit while I'm at it...)",
        sp_aft=True,
        sep_aft="=",
    )
    time.sleep(sleep_for)


def trim_tokens(tkns, end_pref, end_pref_after_injections):
    if args.character:
        char_encoded = le_model.encode(f"{args.character}\n")
        generated = []
        trimmed = []
        for tkn in tkns:
            tt = tkn[end_pref_after_injections:]
            msg = np.concatenate((char_encoded, tt))
            trimmed.append(msg)
            tmp_seq = le_model.decode(msg)
            tmp_seq = tmp_seq.strip()
            generated.append(tmp_seq)
    else:
        trimmed = [tkns[end_pref:] for tkns in tkns]
        generated = [seq.strip() for seq in le_model.decode(trimmed)]
    return generated, trimmed


def extract_chars_msgs(generated, data):
    chars = []
    messages = []
    for i, g in enumerate(generated):
        if g.find("\n") == -1:
            char = ""
            message = g
        else:
            char = g[: g.find("\n")].strip()
            message = g[g.find("\n") + 1 :].strip()

        pprint(char, off="\t")
        pprint(message, off="\t")
        pprint(f"(perp: {data['perplexities'][i].item()})", off="\t")
        pprint("*", off="\t")

        chars.append(char)
        messages.append(message)
    return chars, messages


def le_random_wall(fn):
    global IS_GENERATING
    rand = random.random()
    pprint(f"(random has spoken: {rand})", off="\t", sp_bf=True)
    if rand > args.random_threshold:
        pprint(
            "(le grreat rrrandom is bountiful, let's generate)", off="\t", sp_aft=True
        )
        if not fn():
            return False
    else:
        pprint("(nope, the wall of random could not be passed)", off="\t", sp_aft=True)
        with LeLocle:
            IS_GENERATING = False
    return True


def le_warning(has_warned):
    if not has_warned:
        pprint(
            "(is generating, not answering...)\r", sep="-", sp_bf=True, sp_aft=True,
        )
    return True


def sleepy_times():
    r = np.random.uniform(1, 1 + args.sleepy_time)
    pprint(
        f"(sleepy timezz for {args.server_name}: rrandom gave me {r} second(s) to sleep.)",
        sep="-",
        sp_bf=True,
    )
    pprint(f"(The max could have been:  {1 + args.sleepy_time})", sp_aft=True)
    time.sleep(r)


def auto_loop(fn):
    global IS_GENERATING
    global SEP_TKNS
    global TKNS
    with LeLocle:
        TKNS = SEP_TKNS
    has_warned = False
    while True:
        if not IS_GENERATING:
            with LeLocle:
                IS_GENERATING = True
            print()
            has_warned = False
            if not le_random_wall(fn):
                break
            sleepy_times()
            if not HAS_STARTED:
                break
        else:
            has_warned = le_warning(has_warned)


# ----------------------------------------
# generation: mass production of sentences, selection of best ones


def generate_mass():

    global IS_GENERATING
    global RECEIVED_MSGS
    global BATCH_MSG_IND
    global RESETTING
    global TKNS

    send_typing(
        {"id": sio.sid, "character": "", "message": "", "user": args.server_name,}
    )

    if RESETTING:
        return reset_gen()

    if RECEIVED_MSGS.size > 0:
        pprint("(appending received messages)", sp_bf=True, off="\t\t\t", sp_aft=True)
        pprint(
            le_model.decode(RECEIVED_MSGS[:-SEP_TKNS_LEN]), off="\t\t\t", sp_aft=True
        )
        with LeLocle:
            RECEIVED_MSGS = RECEIVED_MSGS[:-SEP_TKNS_LEN]  # removing last separators
            TKNS = np.concatenate((TKNS, RECEIVED_MSGS))
            RECEIVED_MSGS = np.array([], np.int32)

    if RESETTING:
        return reset_gen()

    end_pref_orig, end_pref, end_pref_after_injections = preprocess_prefix()

    if RESETTING:
        return reset_gen()

    suitors = {
        "tokens": [],
        "perplexities": np.array([], dtype=np.int32),
        "chars": [],
        "messages": [],
    }

    patience = 0
    while patience < args.patience:

        if RESETTING:
            return reset_gen()

        # first produce a small bit avoiding the end token
        try:
            data = le_model.gen_avoiding(
                prefix=TKNS,
                avoiding=le_model.encode("<|e|>"),
                length=10,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=args.batch_size,
            )

        except Exception as e:
            handle_error("gen_avoiding", end_pref_orig, e)
            return reset_gen()

        pprint("(gen avoiding)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
        for i, tkn in enumerate(data["tokens"]):
            pprint(le_model.decode(tkn[end_pref_orig:]).strip(), off="\t\t")
            pprint(f"(perp: {data['perplexities'][i].item()})", off="\t\t")
            pprint("*", off="\t\t")

        if RESETTING:
            return reset_gen()

        # then produce the rest, until the end token
        try:
            pprint(
                "(generation proper, until reaching the end of each answer)",
                off="\t\t",
                sep="-",
                sp_bf=True,
                sp_aft=True,
            )
            data = le_model.gen_until(
                prefix=data["tokens"],
                until="<|s|>",
                exclude_until=False,
                sanity_limit=300,
                chunk_length=5,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )

        except Exception as e:
            handle_error("gen_until", end_pref_orig, e)
            return reset_gen()

        pprint("(done!)", off="\t\t", sp_bf=True, sp_aft=True)

        if suitors["perplexities"].size == 0:

            suitors["perplexities"] = data["perplexities"]
            suitors["tokens"] = data["tokens"]

            generated, data["trimmed"] = trim_tokens(
                data["tokens"], end_pref, end_pref_after_injections
            )

            if RESETTING:
                return reset_gen()

            pprint("(generated)", off="\t", sep="-", sp_bf=True, sp_aft=True)

            if RESETTING:
                return reset_gen()

            chars, messages = extract_chars_msgs(generated, data)

            suitors["chars"] = chars
            suitors["messages"] = messages

            if RESETTING:
                return reset_gen()

        else:

            concat_perps = np.concatenate(
                (suitors["perplexities"].flatten(), data["perplexities"].flatten())
            )
            n = args.batch_size  # get the batch_size smallest perps of 'em all
            n_best_indz = np.argpartition(concat_perps, n)
            n_best_perps = concat_perps[n_best_indz][:n]
            sorted_indz = np.argsort(n_best_perps)
            suitors["perplexities"] = n_best_perps[sorted_indz][:, None]

            former_suitors = set([t.tostring() for t in suitors["tokens"]])

            # use same partition to extract the sequences
            concat_seqs = np.array(suitors["tokens"] + data["tokens"])
            suitors["tokens"] = list(concat_seqs[n_best_indz][:n][sorted_indz])

            generated, data["trimmed"] = trim_tokens(
                data["tokens"], end_pref, end_pref_after_injections
            )

            if RESETTING:
                return reset_gen()

            pprint("(generated)", off="\t", sep="-", sp_bf=True, sp_aft=True)

            if RESETTING:
                return reset_gen()

            chars, messages = extract_chars_msgs(generated, data)

            if former_suitors == set([t.tostring() for t in suitors["tokens"]]):
                pprint(
                    "(NO UPDATE. Current production does not beat past record.)",
                    off="\t",
                    sep="-",
                    sp_bf=True,
                )
                patience += 1

            concat_chars = np.array(suitors["chars"] + chars)
            suitors["chars"] = list(concat_chars[n_best_indz][:n][sorted_indz])
            concat_messages = np.array(suitors["messages"] + messages)
            suitors["messages"] = list(concat_messages[n_best_indz][:n][sorted_indz])

            pprint("(current selection)", off="\t", sep="-", sp_bf=True, sp_aft=True)
            for i in range(n):
                pprint(suitors["chars"][i], off="\t")
                pprint(suitors["messages"][i], off="\t")
                pprint(f"(perp: {suitors['perplexities'][i].item()})", off="\t")
                pprint("*", off="\t")

            if RESETTING:
                return reset_gen()

            if patience < args.patience:
                send_batch(
                    {
                        "id": sio.sid,
                        "chars": suitors["chars"],
                        "messages": suitors["messages"],
                        "perplexities": suitors["perplexities"].tolist(),
                        "countdown": False,
                    }
                )
            # time.sleep(3)

    send_batch(
        {
            "id": sio.sid,
            "chars": suitors["chars"],
            "messages": suitors["messages"],
            "perplexities": suitors["perplexities"].tolist(),
            "seconds": args.wait_for_master,
            "countdown": True,
        }
    )

    if not index_from_master():
        return reset_gen()

    if RESETTING:
        return reset_gen()

    char, message = select_in_batch(suitors, suitors["chars"], suitors["messages"])

    if RESETTING:
        return reset_gen()

    send_ind()

    if RESETTING:
        return reset_gen()

    if not fancy_tok_typing(data["trimmed"][BATCH_MSG_IND]):
        return reset_gen()

    if RESETTING:
        return reset_gen()

    send_message({"character": char, "message": message, "user": args.server_name})

    with LeLocle:
        if not RESETTING:
            TKNS = suitors["tokens"][BATCH_MSG_IND]
        BATCH_MSG_IND = None
        IS_GENERATING = False

    return True

# ----------------------------------------
# generation: based on tokens


def generate_new():

    global TKNS_LEN_THRESHOLD
    global IS_GENERATING
    global RECEIVED_MSGS
    global BATCH_MSG_IND
    global RESETTING
    global TKNS

    send_typing(
        {"id": sio.sid, "character": "", "message": "", "user": args.server_name,}
    )

    # send_entrails(
    #     {
    #         "id": sio.sid,
    #         "entrails": f"Tokens: {str(TKNS)}" if TKNS.size > 0 else "",
    #         "user": args.server_name,
    #     }
    # )

    if RESETTING:
        return reset_gen()

    if RECEIVED_MSGS.size > 0:
        with LeLocle:
            RECEIVED_MSGS = RECEIVED_MSGS[:-SEP_TKNS_LEN]  # removing last separators
            pprint(
                "(appending received messages)", sp_bf=True, off="\t\t\t", sp_aft=True
            )
            pprint(le_model.decode(RECEIVED_MSGS), off="\t\t\t", sp_aft=True)
            TKNS = np.concatenate((TKNS, RECEIVED_MSGS))
            RECEIVED_MSGS = np.array([], np.int32)

    if TKNS_LEN_THRESHOLD and TKNS.size >= TKNS_LEN_THRESHOLD:
        pprint(
            "(REACHED THRESHOLD LENGTH, TRIMMING)",
            sp_bf=True,
            off="\t\t\t",
            sp_aft=True,
        )
        with LeLocle:
            TKNS = TKNS[-TKNS_LEN_THRESHOLD:]

    if RESETTING:
        return reset_gen()

    end_pref_orig, end_pref, end_pref_after_injections = preprocess_prefix()

    if RESETTING:
        return reset_gen()

    # first produce a small bit avoiding the end token
    try:
        pprint(
            "(starting generation, making sure we don't finish early)",
            off="\t\t",
            sep="-",
            sp_bf=True,
        )
        data = le_model.gen_avoiding(
            prefix=TKNS,
            avoiding=le_model.encode("<|e|>"),
            length=10,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )

    except Exception as e:
        handle_error("gen_avoiding", end_pref_orig, e)
        return reset_gen()

    # send_entrails(
    #     {
    #         "id": sio.sid,
    #         "entrails": f"Tokens:\n{str(data['tokens'])}\nLogprobs:\n{str(data['logprobs'])}\nPerplexities:\n{str(data['perplexities'])}",
    #         "user": args.server_name,
    #     }
    # )

    pprint("(done!)", off="\t\t", sp_aft=True)
    for i, tkn in enumerate(data["tokens"]):
        pprint(le_model.decode(tkn[end_pref_orig:]).strip(), off="\t\t")
        pprint(f"(perp: {data['perplexities'][i].item()})", off="\t\t")
        pprint("*", off="\t\t")

    if RESETTING:
        return reset_gen()

    # then produce the rest, until the end token
    try:
        pprint(
            "(generation proper, until reaching the end of each answer)",
            off="\t\t",
            sep="-",
            sp_bf=True,
            sp_aft=True,
        )
        data = le_model.gen_until(
            prefix=data["tokens"],
            until="<|s|>",
            exclude_until=False,
            sanity_limit=300,
            chunk_length=5,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )

    except Exception as e:
        handle_error("gen_until", end_pref_orig, e)
        return reset_gen()

    pprint("(done!)", off="\t\t", sp_bf=True, sp_aft=True)

    # sort the sequences
    sorted_indz = np.argsort(data["perplexities"], axis=0).flatten()
    data["perplexities"] = data["perplexities"][sorted_indz]
    data["tokens"] = np.asarray(data["tokens"])[sorted_indz]

    generated, data["trimmed"] = trim_tokens(
        data["tokens"], end_pref, end_pref_after_injections
    )

    if RESETTING:
        return reset_gen()

    pprint("(generated)", off="\t", sep="-", sp_bf=True, sp_aft=True)

    if RESETTING:
        return reset_gen()

    chars, messages = extract_chars_msgs(generated, data)

    if RESETTING:
        return reset_gen()

    send_batch(
        {
            "id": sio.sid,
            "chars": chars,
            "messages": messages,
            "perplexities": data["perplexities"].tolist(),
            "seconds": args.wait_for_master,
            "countdown": True,
        }
    )

    if not index_from_master():
        return reset_gen()

    if RESETTING:
        return reset_gen()

    char, message = select_in_batch(data, chars, messages)

    if RESETTING:
        return reset_gen()

    send_ind()

    send_direct_message(
        {"character": char, "message": message, "user": args.server_name, "id": sio.sid}
    )

    if not fancy_tok_typing(data["trimmed"][BATCH_MSG_IND]):
        return reset_gen()

    send_message({"character": char, "message": message, "user": args.server_name})

    if RESETTING:
        return reset_gen()

    with LeLocle:
        if not RESETTING:
            TKNS = data["tokens"][BATCH_MSG_IND]
        BATCH_MSG_IND = None
        IS_GENERATING = False

    return True


# ----------------------------------------
# legacy generation (based on strings)


def generate():

    global IS_GENERATING
    global PREFIX
    global START

    if RESETTING:
        return reset_gen()

    # print("-"*40)
    # print("(prefix:)")
    # print(PREFIX)
    # print("-"*40)

    prefix_enc = le_model.encode(PREFIX)
    max_len = 1024 - args.length_desired
    if len(prefix_enc) > max_len:
        prefix_enc = prefix_enc[-max_len:]
        with LeLocle:
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

            if RESETTING:
                return reset_gen()

            send_message(
                {"character": char, "message": message, "user": args.server_name}
            )
            with LeLocle:
                PREFIX = f"{PREFIX}{START}{char}\n{message}"
        else:
            pprint(
                "(RANK INSUFFICIENT: NOT ANSWERING)", off="\t", sp_bf=True, sp_aft=True
            )
    else:
        pprint("(picked:)", off="\t\t", sep="-", sp_bf=True, sp_aft=True)
        pprint(l, off="\t\t", sp_aft=True)
        pprint("(MARKERS NOT FOUND: NOT ANSWERING)", off="\t\t")

    with LeLocle:
        IS_GENERATING = False


@sio.event
def connect():
    print(f"{args.server_name} established connection")
    print("-" * 40)
    sio.emit("new bot", args.server_name)
    # if args.mode == "autonomous":
    #     auto_loop(generate_new)
    # elif args.mode == "optimizer":
    #     auto_loop(generate_mass)


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

    global BATCH_MSG_IND
    global RECEIVED_MSGS
    global HAS_STARTED
    global RESETTING
    global MESSAGES
    global PREFIX
    global TKNS

    print()
    print("=" * 40)
    print()
    print("resetting session")

    if IS_GENERATING:
        with LeLocle:
            RESETTING = True
    else:
        MESSAGES = []
        send_three_dots()
        with LeLocle:
            RECEIVED_MSGS = np.array([], dtype=np.int32)
            BATCH_MSG_IND = None
            HAS_STARTED = False
            TKNS = SEP_TKNS
            PREFIX = ""
        print("not generating, resetting variables")
        print()
        print("=" * 40)
        print()


@sio.on("received")
def on_chat_message(data):

    global IS_GENERATING
    global RECEIVED_MSGS
    global HAS_STARTED
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

    added_seps = False
    if character:
        with LeLocle:
            PREFIX = f"{PREFIX}{SEPARATORS}{character}\n{message}{END}"
            RECEIVED_MSGS = np.concatenate(
                (RECEIVED_MSGS, le_model.encode(f"{character}\n{message}"), SEP_TKNS)
            )
    else:
        with LeLocle:
            PREFIX = f"{PREFIX}{SEPARATORS}{message}{END}"
            RECEIVED_MSGS = np.concatenate(
                (RECEIVED_MSGS, le_model.encode(f"{message}"), SEP_TKNS)
            )

    # pprint("(after reception, TKNS are now:)", sep="-", sp_bf=True)
    # print(TKNS[0], type(TKNS[0]))
    # pprint(le_model.decode(TKNS)[0], sp_aft=True)

    # reactive mode, legacy or current
    if args.mode in {"legacy", "reactive"}:
        if not IS_GENERATING:
            with LeLocle:
                IS_GENERATING = True
            if args.mode == "legacy":
                le_random_wall(generate)
                sleepy_times()
            if args.mode == "reactive":
                le_random_wall(generate_new)
                sleepy_times()
        else:
            pprint("(is generating, not answering...)", off="\t", sp_aft=True)
    else:
        if not HAS_STARTED:
            with LeLocle:
                HAS_STARTED = True
            if args.mode == "autonomous":
                auto_loop(generate_new)
            elif args.mode == "optimizer":
                auto_loop(generate_mass)


@sio.on("get bot config")
def send_config():
    config = {
        "id": sio.sid,
        "user": args.server_name,
        "model": args.model,
        "run": args.run_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "print_speed": args.print_speed,
        "random_threshold": args.random_threshold,
    }
    if args.mode == "legacy":
        config.update(
            {
                "length_desired": args.length_desired,
                "rank_threshold": args.rank_threshold,
            }
        )
    else:
        config.update(
            {
                "character": args.character,
                "hidden_before_char": args.hidden_before_char,
                "hidden_after_char": args.hidden_after_char,
                "wait_for_master": args.wait_for_master,
                "sleepy_time": args.sleepy_time,
            }
        )
        if args.mode == "optimizer":
            config.update(
                {"patience": args.patience,}
            )
    sio.emit("config from bot", config)


@sio.on("server sets bot config")
def set_config(data):
    if data["id"] == sio.sid:
        pprint("received config:", sep="-", sp_bf=True, und=True)
        for k, v in data.items():
            if k in {"user", "id", "run", "model"}:
                continue
            try:
                if k in {"sleepy_time", "print_speed"}:
                    v = type(args.__getattribute__(k))(v)
                    v = v if v > 0 else 0.0001
                else:
                    v = type(args.__getattribute__(k))(v)
                print(f"{k}: {v}")
                args.__setattr__(k, v)
            except:
                print(
                    f"\033[31m!! {k}: cannot cast {v} to {type(args.__getattribute__(k))}, ignoring...\033[0m"
                )
                continue
        print()


@sio.on("server sends choice")
def set_message_choice(data):

    global BATCH_MSG_IND

    if data["id"] == sio.sid:
        with LeLocle:
            BATCH_MSG_IND = data["choice"]
        if BATCH_MSG_IND == -1:
            msg = f"(received batch choice: '-1' received, the bot will choose)"
        else:
            msg = f"(received batch choice: message chosen: {BATCH_MSG_IND})"
        pprint(msg, sep="-", off="\t", sp_aft=True)


def send_typing(data):
    sio.emit("typing", data)


def send_entrails(data):
    sio.emit("entrails", data)


def send_three_dots():
    send_typing(
        {"id": sio.sid, "character": "", "message": "(...)", "user": args.server_name,}
    )


def send_ind():
    sio.emit("bot confirms choice", {"id": sio.sid, "choice": BATCH_MSG_IND,})


def send_batch(data):
    sio.emit("chat batch", data)


def send_message(data):
    sio.emit("chat message", data)


def send_direct_message(data):
    sio.emit("direct chat message", data)


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
