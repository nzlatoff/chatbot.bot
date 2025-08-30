import argparse
import os
import random
import string
import time
from base64 import b64encode
from functools import partial
from dotenv import load_dotenv

import socketio

from print_utils import pprint
from print_utils import print_config

parser = argparse.ArgumentParser(
    description="""
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--server_name",
    type=str,
    default="lActrice",
    help="Server name used in message.",
)

parser.add_argument(
    "--local", action="store_true", help="Run with local server, port 5100.",
)

parser.add_argument(
    "--character",
    type=str,
    default="",
    help="Character used by the network when answering.",
)

args = parser.parse_args()

# ----------------------------------------
# socket & model init

sio = socketio.Client(logger=False, reconnection_delay_max=50)
BOT_ID = f"bot-{''.join([random.choice(string.ascii_letters + string.digits) for _ in range(23)])}"
load_dotenv()
BOT_TOKEN=os.getenv("BOT_TOKEN")
print_config(args)

def fancy_typing(char, message):
    pprint(
        f"(Awright, {args.server_name} sending le message to humans...)",
        sep="-",
        sp_bf=True,
        sp_aft=True,
    )
    total = len(message) + 1
    for i in range(total):
        send_typing(
            {"character": char, "message": message[:i],}
        )
        time.sleep(args.tempo)
    print()
    print()
    return True


@sio.event
def connect():
    sio.emit("new bot", {"user": args.server_name, "id": BOT_ID, "token": BOT_TOKEN})
    pprint(f"connecting to: {sio.connection_url}", sep="=")
    pprint(f"{args.server_name} established connection", sep="=", sep_aft="=")


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
    print("reset session")


@sio.on("received")
def on_chat_message(data):
    char = data["character"]
    msg = data["message"]

    print(f"(received) {char}: {msg}")
    send_message({"character": char, "message": "Hello :)", "user": args.server_name})

@sio.on("get bot config")
def send_config():
    config = {
        "id": BOT_ID,
        "user": args.server_name,
    }
    sio.emit("config from bot", config)


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
    user_pass = b64encode(b"guest:F89r$Q!Xw&HX").decode("ascii")
    url = "https://chatbot.manufacture-recherche.ch"
    sio.connect(url, {"Authorization": "Basic %s" % user_pass})
sio.wait()
