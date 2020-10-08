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
    msg, width=term.width, off="", sep="", sep_aft="", sp_bf=False, sp_aft=False, und=False
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


