from print_utils import print_underlined
from print_utils import print_config
from print_utils import pprint
from base64 import b64encode
from functools import partial
from print_utils import term
from threading import Lock
from gpt import Model
import numpy as np
import traceback
import textwrap
import argparse
import socketio
import blessed
import random
import string
import regex
import time
import sys

# numpy cosmetics
np.set_printoptions(formatter={"all": lambda x: f"{str(x):>{5}}"})

# for silence arg below
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
    help="Model path (params).",
)

parser.add_argument(
    "--run_name",
    type=str,
    default="run1",
    help="Run name path (weights).",
)

parser.add_argument(
    "--server_name",
    type=str,
    default="La Manufactrice",
    help="Server name used in message.",
)

parser.add_argument(
    "--mode",
    type=str,
    choices=["legacy", "reactive", "autonomous", "optimizer"],
    default="autonomous",
)

parser.add_argument(
    "--device",
    type=str,
    default="/GPU:0",
    help="The GPU on which the net will be run.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="""Number of sentences generated in parallel.""",
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
    many possible choices). (1 to neutralise).""",
)

parser.add_argument(
    "--top_k",
    type=int,
    default=0,
    help="""Limit sampled tokens to the k most
    likely ones. (0 to neutralise).""",
)

parser.add_argument(
    "--tempo",
    type=float,
    default=0.1,
    help="Length of pause for each step of interactive print loop, in ms.",
)

parser.add_argument(
    "--local", action="store_true", help="Run with local server, port 5100.",
)

parser.add_argument(
    "--length_desired",
    type=int,
    default=500,
    help="""LEGACY ONLY (before end tokens were introduced). Length of text
    before the bot stops.""",
)

parser.add_argument(
    "--silence",
    type=float_range,
    default=0.0,
    help="""A random number between 0 and 1 is generated each time the network
    receives a new message. If the number is above the silence, the network
    answers. Must lie withinin [0:1]. (when set to 0 the network answering
    mechanism is fired every time).""",
)

parser.add_argument(
    "--pause",
    type=positive_float,
    default=10,
    help="""The most time the bot sleeps between each new attempt to produce
    text (for autonomous & optimizer modes. A random number is generated,
    between 1 and pause, before each call of the generate function.""",
)

parser.add_argument(
    "--rank_threshold",
    type=int,
    default=25,
    help="Rank under which sentences are allowed to be sent.",
)

parser.add_argument(
    "--character",
    type=str,
    default="",
    help="Character used by the network when answering.",
)

parser.add_argument(
    "--subtext",
    type=str,
    default="",
    help="""Additional text inserted at the end of each received message, before
    the network produces the next character & answer (can
    influence the overall theme and stabilise the network).""",
)

parser.add_argument(
    "--first_words",
    type=str,
    default="",
    help="""Additional text inserted at the start of each produced message,
    after the character (influences the current answer). If the character is
    not artificially set as well, this start of line becomes the same as the
    subtext: the text is added at the end of the received messages,
    and the network is free to produce any answer (coloured, however, by the
    context)..""",
)

parser.add_argument(
    "--wait",
    type=int,
    default=0,
    help="""Waiting time before activating the default message choice. When
    generating with more than one message per batch, the /master can choose the
    message, in the time frame here defined, after which the standard choice
    process kicks in. (0 means no wait, the master sets it from /master).""",
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
    (either to be directly posted, or evaluated by the master.""",
)

parser.add_argument(
    "--limit_prefix",
    type=int,
    default=200,
    help="""Preemptively limit the length of the prefix, to avoid OOM issues.""",
)

args = parser.parse_args()

# ----------------------------------------
# socket & model init

sio = socketio.Client(logger=False, reconnection_delay_max=50)

LeLocle = Lock()

le_model = Model(
    model_name=args.model,
    run_name=args.run_name,
    device=args.device,
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

HAS_STARTED = False  # for autonomous modes

BOT_ID = f"bot-{''.join([random.choice(string.ascii_letters + string.digits) for _ in range(23)])}"

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
    pprint(
        f"thinking was stopped, wiping everything",
        sp_bf=True,
        sep="=",
        sp_aft=True,
        sep_aft="=",
    )
    return False


def index_from_master():

    global BATCH_MSG_IND

    i = 0
    pprint("")
    while BATCH_MSG_IND == None:
        if RESETTING:
            return False
        pprint(
            f"(waited for choice for {i + 1} seconds)     ", cr=True,
        )
        time.sleep(1)
        i += 1
        if i > args.wait + 10:
            pprint(f"waited enough, {args.server_name} taking back control!")
            with LeLocle:
                BATCH_MSG_IND = -1
    return True


def trim_tok(tkns):
    # print("TRIMMING TOKS:\n", tkns)
    # print()
    riddance = {
        le_model.encode(i)[0]
        for i in {" ", "\n", " ", "<|s|>", "<|e|>", "<|endoftext|>"}
    }
    # left trimming
    while tkns.size >= 1 and tkns[0] in riddance:
        tkns = tkns[1:]
    # right trimming
    while tkns.size >= 1 and tkns[-1] in riddance:
        tkns = tkns[:-1]
    return tkns


def fancy_tok_typing(tkns):
    """
    Somewhat nasty piece of work, playing with the Python print function
    (without carriage return, overwriting the current line, as well as sending
    things to the web at the same time).
    """
    pprint(
        f"(alright, {args.server_name} sending les tokens to humans...)",
        sep="-",
        sp_aft=True,
    )
    tkns = trim_tok(tkns)
    total = len(tkns)
    nl_ind = np.where(tkns == 201)[0]
    if nl_ind.size == 0:  # no newline separating char from msg
        nl_ind = 1
        send_entrails("[ ", pre=True)
    else:
        nl_ind = nl_ind[0]
        s = f"{tkns[:nl_ind + 1]}"
        if "\n" in s:  # if char on more than one line
            s = s.split("\n")  # split and send all
            [print(f"{ss}") for ss in s[:-1]]
            [send_entrails(f"{ss}", pre=True) for ss in s[:-1]]  # but the last one
            s = s[-1]
        print(f"{s[:s.rfind(']')]}\r", end="")
        send_entrails(f"{s[:s.rfind(']')]} ", pre=True)
        # breakpoint()
    prev = ""
    # i starts from after the char onward
    for i in range(nl_ind, total):
        # breakpoint()
        if RESETTING:
            return False
        if nl_ind == 1:  # if not, no char
            char = ""
            message = le_model.decode(tkns[:i])
        else:  # else, we split at that newline
            char = le_model.decode(tkns[:nl_ind])
            message = le_model.decode(tkns[nl_ind + 1 : i + 2])
        # np arrays are formatted, split at the new line, and only print the
        # last line (overwriting at each step unless we have reached a new line
        msg = f"{tkns[:i+2]}".split("\n")
        current = msg[-1][: msg[-1].rfind("]")] + "\r"
        if len(current) < len(prev):
            print()
            send_entrails(" ", pre=True)
        print(current, end="")
        t = tkns[i + 1 : i + 2]
        if t.size > 0:
            send_entrails(f"{t.item():>5} ", no_cr=True)
        prev = current
        send_typing(
            {"character": char, "message": message,}
        )
        time.sleep(args.tempo)
    print(msg[-1])
    send_entrails("]", no_cr=True)
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
            {"character": char, "message": message[:i],}
        )
        time.sleep(args.tempo)
    print()
    print()
    return True


def preprocess_prefix():

    global TKNS_LEN_THRESHOLD
    global SEP_TKNS_LEN
    global SEP_TKNS
    global TKNS

    # end_pref_orig: before all injections
    # if not char:
    #   end_pref: len after all injections
    #   end_pref_after_injections: unused (same as end_pref)
    # else:
    #   end_pref: len after subtext
    #   end_pref_after_injections: len after char

    end_pref_orig = len(TKNS)
    end_pref = end_pref_orig
    end_pref_after_injections = end_pref_orig
    len_injections = 0

    # pprint("(tkns)", sep="-", sp_bf=True)
    # pprint(str(TKNS), sp_aft=True)

    if not args.character:

        # if no char injected, inject before markers:
        # - add subtext
        # - add first_words
        # - markers

        if args.subtext:
            args.hidden_before_ch = args.subtext.strip()
            hidden_before_encoded = le_model.encode(f"\n{args.subtext}")
            len_injections += len(hidden_before_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, hidden_before_encoded))
            args.subtext = ""

        if args.first_words:
            args.first_words = args.first_words.strip()
            inject_after_encoded = le_model.encode(f"\n{args.first_words}")
            len_injections += len(inject_after_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, inject_after_encoded))
            args.first_words = ""

        # markers
        len_injections += SEP_TKNS_LEN
        with LeLocle:
            TKNS = np.concatenate((TKNS, SEP_TKNS))

        end_pref = end_pref_orig + len_injections
        end_pref_after_injections = end_pref

    else:

        # if char, add markers, then the rest

        len_injections += SEP_TKNS_LEN
        with LeLocle:
            TKNS = np.concatenate((TKNS, SEP_TKNS))

        if args.subtext:
            args.hidden_before_ch = args.subtext.strip()
            hidden_before_encoded = le_model.encode(f"{args.subtext}\n")
            len_injections += len(hidden_before_encoded)
            with LeLocle:
                TKNS = np.concatenate((TKNS, hidden_before_encoded))
            args.subtext = ""

        end_pref = end_pref_orig + len_injections

        args.character = args.character.strip()
        char_encoded = le_model.encode(f"{args.character}\n")
        with LeLocle:
            TKNS = np.concatenate((TKNS, char_encoded))
        end_pref_after_injections = end_pref + len(char_encoded)
        if args.first_words:
            args.first_words = args.first_words.strip()
            after_char_encoded = le_model.encode(f"{args.first_words}")
            with LeLocle:
                TKNS = np.concatenate((TKNS, after_char_encoded))
            args.first_words = ""
            # end_pref_after_injections += len(after_char_encoded)

    if TKNS.size >= TKNS_LEN_THRESHOLD:
        pprint(
            "(REACHED THRESHOLD LENGTH, TRIMMING)",
            sep="=",
            sp_bf=True,
            sep_aft="=",
            sp_aft=True,
        )
        with LeLocle:

            # readjust prefix params

            # print()
            # print('before')
            # print(f"end_pref: {end_pref}")
            # print(f"end_pref_orig: {end_pref_orig}")
            # print(f"end_pref_after_injections: {end_pref_after_injections}")
            # print(f"len_injections: {len_injections}")

            excess = TKNS.size - TKNS_LEN_THRESHOLD
            end_pref = end_pref - excess if excess < end_pref else 0
            end_pref_orig = end_pref_orig - excess if excess < end_pref_orig else 0
            end_pref_after_injections = end_pref_after_injections - excess if excess < end_pref_after_injections else 0
            if len_injections > TKNS_LEN_THRESHOLD: len_injections = TKNS_LEN_THRESHOLD

            # print()
            # print('after')
            # print(f"excess: {excess}")
            # print(f"end_pref: {end_pref}")
            # print(f"end_pref_orig: {end_pref_orig}")
            # print(f"end_pref_after_injections: {end_pref_after_injections}")
            # print(f"len_injections: {len_injections}")

            TKNS = TKNS[-TKNS_LEN_THRESHOLD:]

    if TKNS.size > SEP_TKNS_LEN:
        pprint("(currently in memory:)", sp_bf=True, sp_aft=True)
        pprint(le_model.decode(TKNS).strip(), sp_aft=True)
        pprint(f"(length: {end_pref})")
    else:
        pprint("(nothing in memory!)", sp_bf=True)

    return end_pref_orig, end_pref, end_pref_after_injections


# https://stackoverflow.com/a/50425683
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def select_in_batch(data, chars, messages):

    global BATCH_MSG_IND

    if BATCH_MSG_IND == -1:
        if args.bot_choice == "sampling":
            # smallest perps given most weight
            flipped = softmax(-data["perplexities"])
            with LeLocle:
                BATCH_MSG_IND = np.random.choice(
                    data["perplexities"].shape[0], 1, p=flipped.flatten()
                ).item()
        elif args.bot_choice == "min":
            with LeLocle:
                BATCH_MSG_IND = int(np.argmin(data["perplexities"]))
        elif args.bot_choice == "max":
            with LeLocle:
                BATCH_MSG_IND = int(np.argmax(data["perplexities"]))
        char = chars[BATCH_MSG_IND]
        message = messages[BATCH_MSG_IND]
        pprint(
            f"(ok, sending, message: {BATCH_MSG_IND+1})", sp_aft=True,
        )
        pprint(char.strip())
        pprint(message)
        pprint(
            f"(perplexity: {data['perplexities'][BATCH_MSG_IND].item()})",
        )
    else:
        char = chars[BATCH_MSG_IND]
        message = messages[BATCH_MSG_IND]
        pprint(f"(yawn! sending le master's choice)", sep="-", sp_bf=True)
        pprint(char)
        pprint(message)
        pprint(
            f"(perplexity: {data['perplexities'][BATCH_MSG_IND].item()})",
        )
    return char, message


def handle_error(fn_name, end_pref_orig, e, trimming_factor=1 / 6, sleep_for=5):

    global TKNS_LEN_THRESHOLD
    global TKNS

    send_three_dots()

    pprint(f"O.O.O.P.S. What ocurred during {fn_name}?", sep="=", sp_bf=True)
    # exc_type, exc_value, exc_traceback = sys.exc_info()
    # traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    # https://docs.python.org/3/library/traceback.html#traceback.format_exception
    [pprint(p, pre=True) for p in traceback.format_exc().split("\n") if p]
    if type(e).__name__ == "ResourceExhaustedError":
        pprint(
            f"DANGEROUS LENGTH REACHED ! Trimming by a factor of {trimming_factor} (approximately).",
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
    else:
        pprint("", sp_aft=True, sep_aft="=")


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


# https://stackoverflow.com/a/61421479
def unequal_lists_of_lists_to_np(a, b):
    if isinstance(a, list):
        l_a = len(a)
    else:
        a.shape[0]
        a = list(a)
    if isinstance(b, list):
        l_b = len(b)
    else:
        b.shape[0]
        b = list(b)
    container = np.empty(l_a + l_b, dtype=object)
    container[:l_a] = a
    container[l_a:] = b
    return container


def extract_chars_msgs(generated, data):

    # pprint("(generated)", sep="-", sp_bf=True, sp_aft=True)

    chars = []
    messages = []
    pprint("", sep="-")
    for i, g in enumerate(generated):
        if g.find("\n") == -1:
            char = ""
            message = g
        else:
            char = g[: g.find("\n")].strip()
            message = g[g.find("\n") + 1 :].strip()

        pprint(char)
        pprint(message)
        pprint(f"(perplexity: {data['perplexities'][i].item()})")
        if args.batch_size > 1 and i < args.batch_size - 1:
            pprint("*")

        chars.append(char)
        messages.append(message)
    return chars, messages


def process_received_messages():

    global RECEIVED_MSGS
    global TKNS

    if RECEIVED_MSGS.size > SEP_TKNS_LEN:
        # pprint("(appending received messages)", sp_bf=True, sp_aft=True)
        # pprint(le_model.decode(RECEIVED_MSGS), sp_aft=True)
        with LeLocle:  # removing last separators
            RECEIVED_MSGS = RECEIVED_MSGS[:-SEP_TKNS_LEN]
            TKNS = np.concatenate((TKNS, RECEIVED_MSGS))
            RECEIVED_MSGS = np.array([], np.int32)


def try_catch_wrapper(fn):

    global BATCH_MSG_IND
    global IS_GENERATING

    try:
        if not fn():
            return False
    except Exception as e:
        pprint(
            f"0.0.0.P.S. in function {fn.__name__}:", sp_bf=True, sep="=", sp_aft=True,
        )
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        print(exc_type, exc_type == "ValueError")
        pprint("(thinking is being stopped)", sp_bf=True, sp_aft=True, sep_aft="=")
        with LeLocle:
            BATCH_MSG_IND = None
            IS_GENERATING = False
        return False
    return True


def le_random_wall(fn):
    global IS_GENERATING
    rand = random.random()
    pprint(f"(random has spoken: {rand})", sp_bf=True)
    if rand > args.silence:
        pprint("(le grreat rrrandom is bountiful, let's think)",)
        if not try_catch_wrapper(fn):
            return False
    else:
        pprint("(nope, the wall of random could not be passed)", sp_aft=True)
        with LeLocle:
            IS_GENERATING = False
    return True


def le_warning(has_warned):
    if not has_warned:
        pprint(
            "(thinking already, not answering that...)\r",
            sep="-",
            sp_bf=True,
            sep_aft="-",
            sp_aft=True,
        )
    return True


def sleepy_times():
    r = np.random.uniform(1, 1 + args.pause)
    pprint(
        f"(sleepy timezz for {args.server_name}: {r:.1f} second(s) (max:  {1 + args.pause}).)",
        sep="-",
        sp_bf=True,
    )
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

    global TKNS_LEN_THRESHOLD
    global IS_GENERATING
    global RECEIVED_MSGS
    global BATCH_MSG_IND
    global RESETTING
    global TKNS

    send_typing(
        {"character": "", "message": "",}
    )

    if RESETTING:
        return reset_gen()

    process_received_messages()

    if RESETTING:
        return reset_gen()

    tkns_bckp = TKNS  # bckp in case batch skipped
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
                prefix=[TKNS] * args.batch_size,
                avoiding=le_model.encode("<|e|>")[0],
                length=10,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=args.batch_size,
            )

        except Exception as e:
            handle_error("gen_avoiding", end_pref, e)
            return reset_gen()

        # pprint("(gen avoiding)", sep="-", sp_bf=True, sp_aft=True)
        # for i, tkn in enumerate(data["tokens"]):
        #     pprint(le_model.decode(tkn[end_pref:]).strip())
        #     pprint(f"(perplexity: {data['perplexities'][i].item()})")
        #     if args.batch_size > 1:
        #         pprint("*")

        if RESETTING:
            return reset_gen()

        # then produce the rest, until the end token
        try:
            pprint(
                f"(Danger! {args.server_name} is about to think!)",
                sep="-",
                sp_bf=True,
                sp_aft=True,
            )
            data_until = le_model.gen_until(
                prefix=data["tokens"],
                until="<|s|>",
                exclude_until=False,
                sanity_limit=100,
                chunk_length=5,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                batch_size=args.batch_size,
                pprint=pprint,
            )
            # generator logic to extract intermediate results
            while True:
                result = next(data_gen)
                if isinstance(result, str):
                    pprint(result, term_trim=term.width, pre=True)
                else:
                    data = result
                    break

        except Exception as e:
            handle_error("gen_until", end_pref, e)
            return reset_gen()

        if suitors["perplexities"].size == 0:

            suitors["perplexities"] = data["perplexities"]
            suitors["tokens"] = data["tokens"]

            generated, data["trimmed"] = trim_tokens(
                data["tokens"], end_pref, end_pref_after_injections
            )

            if RESETTING:
                return reset_gen()

            pprint("(generated)", sep="-", sp_bf=True, sp_aft=True)

            if RESETTING:
                return reset_gen()

            chars, messages = extract_chars_msgs(generated, data)

            suitors["chars"] = chars
            suitors["messages"] = messages
            suitors["trimmed"] = data["trimmed"]

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
            concat_seqs = unequal_lists_of_lists_to_np(
                suitors["tokens"], data["tokens"]
            )
            suitors["tokens"] = list(concat_seqs[n_best_indz][:n][sorted_indz])

            generated, data["trimmed"] = trim_tokens(
                data["tokens"], end_pref, end_pref_after_injections
            )

            if RESETTING:
                return reset_gen()

            pprint("(generated)", sep="-", sp_bf=True, sp_aft=True)

            if RESETTING:
                return reset_gen()

            chars, messages = extract_chars_msgs(generated, data)

            if former_suitors == set([t.tostring() for t in suitors["tokens"]]):
                pprint(
                    f"(NO UPDATE. could not do better than past sentences.  {args.patience - patience - 1} more tries.)",
                    sp_bf=True,
                )
                patience += 1
            else:
                patience = 0

            concat_chars = unequal_lists_of_lists_to_np(suitors["chars"], chars)
            suitors["chars"] = list(concat_chars[n_best_indz][:n][sorted_indz])
            concat_messages = unequal_lists_of_lists_to_np(
                suitors["messages"], messages
            )
            suitors["messages"] = list(concat_messages[n_best_indz][:n][sorted_indz])
            concat_trimmed = unequal_lists_of_lists_to_np(
                suitors["trimmed"], data["trimmed"]
            )
            suitors["trimmed"] = list(concat_trimmed[n_best_indz][:n][sorted_indz])

            pprint("(sentences in my sack:)", sep="-", sp_bf=True, sp_aft=True)
            for i in range(n):
                pprint(suitors["chars"][i])
                pprint(suitors["messages"][i])
                pprint(f"(perplexity: {suitors['perplexities'][i].item()})")
                if args.batch_size > 1 and i != n - 1:
                    pprint("*")

            if RESETTING:
                return reset_gen()

            if patience < args.patience:
                send_batch(
                    {
                        "id": BOT_ID,
                        "chars": suitors["chars"],
                        "messages": suitors["messages"],
                        "perplexities": suitors["perplexities"].tolist(),
                        "countdown": False,
                    }
                )
            # time.sleep(3)

    send_batch(
        {
            "id": BOT_ID,
            "chars": suitors["chars"],
            "messages": suitors["messages"],
            "perplexities": suitors["perplexities"].tolist(),
            "seconds": args.wait,
            "countdown": True,
        }
    )

    if not index_from_master():
        return reset_gen()

    # batch skipped by master
    if BATCH_MSG_IND == -2:
        with LeLocle:
            TKNS = tkns_bckp
            BATCH_MSG_IND = None
            IS_GENERATING = False
        send_three_dots()
        return True

    if RESETTING:
        return reset_gen()

    char, message = select_in_batch(suitors, suitors["chars"], suitors["messages"])

    if RESETTING:
        return reset_gen()

    send_ind()

    if RESETTING:
        return reset_gen()

    if not fancy_tok_typing(suitors["trimmed"][BATCH_MSG_IND]):
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
        {"character": "", "message": "",}
    )

    if RESETTING:
        return reset_gen()

    process_received_messages()

    if RESETTING:
        return reset_gen()

    tkns_bckp = TKNS  # bckp in case batch skipped
    end_pref_orig, end_pref, end_pref_after_injections = preprocess_prefix()

    if RESETTING:
        return reset_gen()

    # first produce a small bit avoiding the end token
    try:
        # pprint(
        #     "(about to think, warming up, making sure it doesn't finish early)",
        #     sep="-",
        #     sp_bf=True,
        #     sp_aft=True,
        # )
        data = le_model.gen_avoiding(
            prefix=[TKNS] * args.batch_size,
            avoiding=le_model.encode("<|e|>")[0],
            length=10,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )

    except Exception as e:
        handle_error("gen_avoiding", end_pref, e)
        return reset_gen()

    # for i in range(args.batch_size):
    #     pprint(
    #         f"{le_model.decode(data['tokens'][i][end_pref:]).strip()}"
    #     )
    #     pprint(
    #         f"(perplexity: {data['perplexities'][i].item()})"
    #     )
    #     if args.batch_size > 1 and i != args.batch_size - 1:
    #         pprint("*")

    if RESETTING:
        return reset_gen()

    # then produce the rest, until the end token
    try:
        pprint(
            f"(Danger! {args.server_name} is about to think!)",
            sep="-",
            sp_bf=True,
            sp_aft=True,
        )
        data_gen = le_model.gen_until(
            prefix=data["tokens"],
            until="<|s|>",
            exclude_until=False,
            sanity_limit=100,
            chunk_length=5,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            batch_size=args.batch_size,
            pprint=True,
        )
        # generator logic to extract intermediate results
        while True:
            result = next(data_gen)
            if isinstance(result, str):
                pprint(result, term_trim=term.width, pre=True)
            else:
                data = result
                break

    except Exception as e:
        handle_error("gen_until", end_pref, e)
        return reset_gen()

    # sort the sequences, if the batch_size is greater than 1
    if args.batch_size > 1:
        sorted_indz = np.argsort(data["perplexities"], axis=0).flatten()
        data["perplexities"] = data["perplexities"][sorted_indz]

    generated, data["trimmed"] = trim_tokens(
        data["tokens"], end_pref, end_pref_after_injections
    )

    if RESETTING:
        return reset_gen()

    chars, messages = extract_chars_msgs(generated, data)

    if RESETTING:
        return reset_gen()

    send_batch(
        {
            "chars": chars,
            "messages": messages,
            "perplexities": data["perplexities"].tolist(),
            "seconds": args.wait,
            "countdown": True,
        }
    )

    if not index_from_master():
        return reset_gen()

    # batch skipped by master
    if BATCH_MSG_IND == -2:
        with LeLocle:
            TKNS = tkns_bckp
            BATCH_MSG_IND = None
            IS_GENERATING = False
        send_three_dots()
        return True

    if RESETTING:
        return reset_gen()

    char, message = select_in_batch(data, chars, messages)

    if RESETTING:
        return reset_gen()

    send_ind()

    send_direct_message(
        {"character": char, "message": message, "user": args.server_name, "id": BOT_ID}
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

    pprint("(raw)", sep="-", sp_bf=True, sp_aft=True)
    pprint(generated)

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

        pprint("(generated)", sep="-", sp_bf=True, sp_aft=True)
        pprint(repl)

        pprint("(char)", sep="-", sp_bf=True, sp_aft=True)
        pprint(char, sp_aft=True)
        pprint("(message)", sp_aft=True)
        pprint(message)

        prefix_repl = f"{PREFIX} \u001b[31m|\u001b[0m {r.group(0)}"
        pprint(prefix_repl)

        prefix_rank = le_model.get_rank(PREFIX)["ranks_mean"][0]
        prefix_repl_rank = le_model.get_rank(prefix_repl)["ranks_mean"][0]
        le_rank = le_model.get_rank(repl)["ranks_mean"][0]

        pprint(f"(prefix rank: {prefix_rank})")
        pprint(f"(prefix & repl rank: {prefix_repl_rank})")
        pprint(f"(rank: {le_rank})")

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
                "(RANK INSUFFICIENT: NOT ANSWERING)", sp_bf=True, sp_aft=True,
            )
    else:
        pprint("(picked:)", sep="-", sp_bf=True, sp_aft=True)
        pprint(l, sp_aft=True)
        pprint("(MARKERS NOT FOUND: NOT ANSWERING)")

    with LeLocle:
        IS_GENERATING = False


@sio.event
def connect():
    sio.emit("new bot", {"user": args.server_name, "id": BOT_ID})
    # pprint(f"connecting to: {sio.connection_url}", sep="=")
    pprint(f"{args.server_name} established connection", sep="=", sep_aft="=")
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

    if IS_GENERATING:
        with LeLocle:
            RESETTING = True
    else:
        MESSAGES = []
        send_three_dots()
        args.character = ""
        args.first_words = ""
        args.subtext = ""
        with LeLocle:
            RECEIVED_MSGS = np.array([], dtype=np.int32)
            BATCH_MSG_IND = None
            HAS_STARTED = False
            TKNS = SEP_TKNS
            PREFIX = ""
        pprint(
            f"{args.server_name} not thinking right now, wiping everything",
            sp_bf=True,
            sep="=",
            sp_aft=True,
            sep_aft="=",
        )


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

    pprint("(received)", sep="-", sp_bf=True, sp_aft=True)
    if data["character"]:
        pprint(f"{char}")
    if data["message"]:
        pprint(f"{msg}", sp_aft=True)

    MESSAGES.append(data)
    character = data["character"]
    message = data["message"]

    added_seps = False
    if character:
        with LeLocle:
            if args.mode == "legacy":
                PREFIX = f"{PREFIX}{SEPARATORS}{character}\n{message}{END}"
                if message:
                    PREFIX = f"{PREFIX}\n{message}{END}"
                else:
                    PREFIX = f"{PREFIX}{END}"
            else:
                RECEIVED_MSGS = np.concatenate(
                    (RECEIVED_MSGS, le_model.encode(f"{character}"))
                )
                if message:
                    RECEIVED_MSGS = np.concatenate(
                        (RECEIVED_MSGS, le_model.encode(f"\n{message}"), SEP_TKNS)
                    )
                else:
                    RECEIVED_MSGS = np.concatenate((RECEIVED_MSGS, SEP_TKNS))
    elif message:
        with LeLocle:
            if args.mode == "legacy":
                PREFIX = f"{PREFIX}{SEPARATORS}{message}{END}"
            else:
                RECEIVED_MSGS = np.concatenate(
                    (RECEIVED_MSGS, le_model.encode(f"{message}"), SEP_TKNS)
                )
    else:
        if args.mode == "legacy":
            if len(PREFIX) == 0:
                with LeLocle:
                    PREFIX = f"{SEPARATORS}"
        else:
            if TKNS.size == 0:
                pprint("(RECEIVED NOTHING, KICKSTARTING)")
                with LeLocle:
                    TKNS = SEP_TKNS

    # pprint("(after reception, TKNS are now:)", sep="-", sp_bf=True)
    # print(TKNS[0], type(TKNS[0]))
    # pprint(le_model.decode(TKNS)[0], sp_aft=True)

    # reactive mode, legacy or current
    if args.mode in {"legacy", "reactive"}:
        if not IS_GENERATING:
            with LeLocle:
                IS_GENERATING = True
            if args.mode == "legacy":
                if not le_random_wall(generate):
                    return
                sleepy_times()
            if args.mode == "reactive":
                if not le_random_wall(generate_new):
                    return
                sleepy_times()
        else:
            pprint("(is generating, not answering...)", sep_aft="-", sp_aft=True)
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
        "id": BOT_ID,
        "user": args.server_name,
        "model": args.model,
        "run": args.run_name,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "silence": args.silence,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "tempo": args.tempo,
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
                "first_words": args.first_words,
                "subtext": args.subtext,
                "wait": args.wait,
                "pause": args.pause,
            }
        )
        if args.mode == "optimizer":
            config.update(
                {"patience": args.patience,}
            )
    sio.emit("config from bot", config)


@sio.on("server sets bot config")
def set_config(data):
    global HAS_STARTED
    if data["id"] == BOT_ID:
        pprint("received config:", sep="-", sp_bf=True, und=True)
        longest = len(max(list(data.keys()), key=lambda x: len(x)))
        prev_batch = args.batch_size
        prev_mode = args.mode
        for k, v in data.items():
            if k in {"user", "id", "run", "model"}:
                continue
            try:
                if k in {"pause", "tempo"}:
                    v = type(args.__getattribute__(k))(v)
                    v = v if v > 0 else 0.0001
                else:
                    v = type(args.__getattribute__(k))(v)
                pprint(f"{k.replace('_', ' '):>{longest}}: {v}", pre=True)
                args.__setattr__(k, v)
            except:
                pprint(
                    f"{k.replace('_', ' '):>{longest}}: !! cannot cast '{v}' to {type(args.__getattribute__(k)).__name__}, ignoring...",
                    pre=True,
                )
                continue
        pprint("", sep="-")
        if prev_mode != args.mode:
            pprint(
                f"{args.server_name} is now {args.mode} (switched from {prev_mode}).",
                sp_aft=True,
                sep_aft="-",
                pre=True,
            )
            if args.mode == "reactive":
                with LeLocle:
                    HAS_STARTED = False
        if prev_batch != args.batch_size:
            global le_model
            pprint(
                f"WAIT! {args.server_name}'s batch size change in process...",
                sp_bf=True,
                sep="=",
            )
            le_model = Model(
                model_name=args.model,
                run_name=args.run_name,
                device=args.device,
                batch_size=args.batch_size,
                special_tokens=["<|endoftext|>"]
                if (args.mode == "legacy")
                else ["<|s|>", "<|e|>", "<|endoftext|>"],
            )
            # reset_session()
            pprint(
                f"Ya! Batch size changed to {args.batch_size}.",
                sp_aft=True,
                sep_aft="=",
            )


@sio.on("server sends choice")
def set_message_choice(data):

    global BATCH_MSG_IND

    if data["id"] == BOT_ID:
        with LeLocle:
            BATCH_MSG_IND = data["choice"]
        if BATCH_MSG_IND == -2:
            msg = f"(received choice: '-2', not sending)"
        if BATCH_MSG_IND == -1:
            msg = f"(received choice: '-1', {args.server_name} chooses)"
        else:
            msg = f"(received choice: {BATCH_MSG_IND})"
        pprint(msg, sep="-")


@sio.on("server requests new batch")
def gen_request(data):

    global IS_GENERATING
    global HAS_STARTED
    global TKNS

    if data["id"] == BOT_ID:
        if TKNS.size == 0:
            with LeLocle:
                TKNS = SEP_TKNS
        # reactive mode, legacy or current
        if args.mode in {"legacy", "reactive"}:
            if not IS_GENERATING:
                with LeLocle:
                    IS_GENERATING = True
                if args.mode == "legacy":
                    try_catch_wrapper(generate)
                    sleepy_times()
                if args.mode == "reactive":
                    try_catch_wrapper(generate_new)
                    sleepy_times()
            else:
                pprint("(is generating, not answering...)", sep_aft="-", sp_aft=True)
        else:
            if not HAS_STARTED:
                with LeLocle:
                    HAS_STARTED = True
                if args.mode == "autonomous":
                    auto_loop(generate_new)
                elif args.mode == "optimizer":
                    auto_loop(generate_mass)


def send_typing(data):
    sio.emit(
        "typing", {"id": BOT_ID, "user": args.server_name, "scroll": True, **data,}
    )


def send_entrails(data, **kwargs):
    sio.emit(
        "entrails",
        {"id": BOT_ID, "user": args.server_name, "entrails": data, **kwargs,},
    )


def send_three_dots():
    send_typing(
        {"character": "", "message": "(...)",}
    )


def send_ind():
    sio.emit("bot confirms choice", {"id": BOT_ID, "choice": BATCH_MSG_IND,})


def send_batch(data):
    sio.emit("chat batch", {"id": BOT_ID, **data,})


def send_message(data):
    sio.emit("chat message", data)


def send_direct_message(data):
    sio.emit("direct chat message", data)


# ----------------------------------------
# specifying print after send_entrails

pprint = partial(pprint, fn=send_entrails)

if args.local:
    url = "http://localhost:5100"
    sio.connect(url)
else:
    user_pass = b64encode(b"guest:vuVpm77e").decode("ascii")
    url = "https://chatbot.manufacture-recherche.ch/ "
    sio.connect(url, {"Authorization": "Basic %s" % user_pass})
sio.wait()
