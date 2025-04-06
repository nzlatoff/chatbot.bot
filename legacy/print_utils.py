import textwrap
import blessed

term = blessed.Terminal()


def print_underlined(msg):
    print(msg)
    print("-" * len(msg))


def print_config(args):
    print("-" * 40)
    print("(settings:)")
    print()
    for k, v in vars(args).items():
        print(f"- {k}: {v}")
    print()


def pprint(
    msg,
    width=term.width,
    off="",
    sep="",
    sep_aft="",
    sp_bf=False,
    sp_aft=False,
    und=False,
    term_trim=None,
    cr=False,
    fn=None,
    **kwargs,
):
    if fn is not None:
        if sp_bf:
            fn("", **kwargs)
        if sep:
            fn(sep, sep=True, **kwargs)
        if und:
            fn(f"{off}{msg}\n", und=True, pre=True,  **kwargs)
        elif cr:
            fn(f"{msg}", wipe=True, **kwargs)
        elif msg:
            fn(f"{msg}", **kwargs)
        if sep_aft:
            fn(sep_aft, sep=True, **kwargs)
        if sp_aft:
            fn("", **kwargs)

    if term_trim is not None:
        # wizardry: first char: ¬ to recognise it on the client side
        # no need on the terminal
        msg = msg[1: term_trim]

    if sp_bf:
        print()
    if sep:
        print(off + sep * width)
    if und:
        print(f"{off}{msg}")
        print(off + "-" * len(msg))
    elif cr:
        print(f"{off}{msg}", end="\r")
    elif msg:
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
