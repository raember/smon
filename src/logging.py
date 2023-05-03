RED = 31
GREEN = 32
YELLOW = 33
BLUE = 34
MAGENTA = 35
CYAN = 36

FMT_INFO1 = f'\033[{BLUE}m'
FMT_INFO2 = f'\033[{BLUE};1m'
FMT_GOOD1 = f'\033[{GREEN}m'
FMT_GOOD2 = f'\033[{GREEN};1m'
FMT_WARN1 = f'\033[{YELLOW}m'
FMT_WARN2 = f'\033[{YELLOW};1m'
FMT_BAD1 = f'\033[{RED}m'
FMT_BAD2 = f'\033[{RED};1m'
FMT_RST = '\033[m'


def msg(s: str, level: int, color: int):
    if level == 0:
        print(f"\033[1;{color}m=>\033[m {s}\033[m")
    else:
        print(f"  \033[1;{color}m{'-' * level}>\033[m {s}\033[m")


def msg1(s: str):
    msg(s, 0, GREEN)


def msg2(s: str):
    msg(s, 1, BLUE)


def msg3(s: str):
    msg(s, 2, BLUE)


def msg4(s: str):
    msg(s, 3, BLUE)


def msg5(s: str):
    msg(s, 4, BLUE)


WARN = f"\033[1;{YELLOW}m"


def warn(s: str, level: int):
    msg(f"{WARN}{s}\033[m", level, YELLOW)


def warn1(s: str):
    warn(s, 0)


def warn2(s: str):
    warn(s, 1)


def warn3(s: str):
    warn(s, 2)


def warn4(s: str):
    warn(s, 3)


def warn5(s: str):
    warn(s, 4)


ERR = f"\033[1;{RED}m"


def err(s: str, level: int):
    msg(f"{ERR}{s}\033[m", level, RED)


def err1(s: str):
    err(s, 0)


def err2(s: str):
    err(s, 1)


def err3(s: str):
    err(s, 2)


def err4(s: str):
    err(s, 3)


def err5(s: str):
    err(s, 4)