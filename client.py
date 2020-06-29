from base64 import b64encode
from gpt import Model
import socketio
import argparse
import random
import regex
import time

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
    "--local",
    action="store_true",
    help="Run with local server, port 5000.",
)

parser.add_argument(
    "--rank_threshold",
    type=int,
    default=25,
    help="Rank under which sentences are allowed to be sent. Defaults to 25.",
)

args = parser.parse_args()

sio = socketio.Client(logger=False, reconnection_delay_max=50)

le_model = Model(model_name=args.model, run_name=args.run_name)
print("-"*40)
print("server name:", args.server_name)
print("run name:", args.run_name)
print("rank rank_threshold", args.rank_threshold)
print("-"*40)

length_desired = 250

replique_re = regex.compile("<\|s\|>\n(.*?)\n<\|e\|>\n", regex.DOTALL)
separators = "\n<|e|>\n<|s|>\n"
end = "\n<|e|>\n"
start = "<|s|>\n"

messages = []
prefix = ""

def generate(rank_threshold=25):

    global marker_re
    global pref_re
    global prefix
    global start

    # print("-"*40)
    # print("prefix:")
    # print(prefix)
    # print("-"*40)

    prefix_enc = le_model.encode(prefix)
    max_len = 1024 - length_desired
    if len(prefix_enc) > max_len:
        prefix_enc = prefix_enc[-max_len:]
        prefix = le_model.decode(prefix_enc)

    # add end of answer, store length of prefix
    end_pref = len(prefix)

    l = le_model.gen(prefix=prefix, length=length_desired)[0]
    generated = l[end_pref:]

    # print(l[:end_pref])
    # print("-"*40)
    # print("generated:")
    # print(generated)
    # print("-"*40)

    r = regex.search(replique_re, generated)

    if r:
        repl = r.group(1)
        char = repl[:repl.find("\n")]
        message = repl[repl.find("\n") + 1:]
        print("\t(generated)")
        print("\t(char:)")
        print(f"\t{char}")
        print("\t(message:)")
        msg = message.replace("\n", "\n\t")
        print(f"\t{msg}")
        le_rank = le_model.get_rank(repl)[0]
        print(f"\t(rank: {le_rank})")
        print("\t" + "-"*40)
        if le_rank < rank_threshold:
            print()
            print(char)
            print(message)
            print()
            for i in range(len(message)):
                # print({ "id": sio.sid, "character": char, "message": # message[:i], "user": args.server_name})
                send_typing({ "id": sio.sid, "character": char, "message": message[:i], "user": args.server_name})
                time.sleep(.1)
            # send_typing({ "id": sio.sid, "character": "", "message": "", # "user": args.server_name})
            send_message({ "character": char, "message": message, "user": args.server_name})
            prefix = f"{prefix}{start}{char}\n{message}"
        else:
            print("\tnot answering")
            print()
    else:
        print("\tnot answering")
        print()

@sio.event
def connect():
    print("\tconnection established")
    print("\t" + "-"*40)
    sio.emit("new user", args.server_name)

@sio.event
def disconnect():
    print("\tconnection lost")

@sio.on("received")
def on_chat_message(data):

    global prefix

    char = data["character"].replace("\n", "\t\n")
    msg = data["message"].replace("\n", "\t\n")
    if data["character"]: print(f"\t{char}")
    if data["message"]: print(f"\t{msg}")

    messages.append(data)
    character = data["character"]
    message = data["message"]
    if character:
        prefix = f"{prefix}{separators}{character}\n{message}{end}"
    else:
        prefix = f"{prefix}{separators}{message}{end}"
    # print("prefix now:")

    # print(prefix)
    # print("-"*40)

    rand = random.random()
    print("\t" + "-"*40)
    print("\trandom has spoken:", rand)
    print("\t" + "-"*40)
    if rand > 0:
        # print("random has been bountiful, let's generate")
        generate(rank_threshold=args.rank_threshold)


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
    url = "http://localhost:5000"
    print(f"connecting to: {url}")
    print("-"*40)
    sio.connect(url)
else:
    url = "***HEROKU WEB ADDRESS***"
    print(f"connecting to: {url}")
    print("-"*40)
    sio.connect(url)
    # user_pass = b64encode(b"username:password").decode("ascii")
    # sio.connect("https://spark.theatrophone.fr",  { "Authorization" : "Basic %s" % user_pass})
sio.wait()
