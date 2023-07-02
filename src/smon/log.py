# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger
from contextlib import contextmanager
from typing import List, ContextManager

RED = 31
GREEN = 32
YELLOW = 33
BLUE = 34
MAGENTA = 35
CYAN = 36
LIGHT_GRAY = 37
DARK_GRAY = 90
LIGHT_RED = 91
LIGHT_GREEN = 92
LIGHT_YELLOW = 93
LIGHT_BLUE = 94
LIGHT_MAGENTA = 95
LIGHT_CYAN = 96
WHITE = 97

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


def msg6(s: str):
    msg(s, 5, BLUE)


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


def warn6(s: str):
    warn(s, 5)


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


def err6(s: str):
    err(s, 5)


def bar(s: str, level: int, color: int):
    print(f" {' ' * level}\033[1;{color}m|\033[m {s}\033[m")


def blue_bar(s: str, level: int):
    bar(s, level, BLUE)


# ├─
# │
# └─
#   ├─

class TreeLogger:
    def __init__(self, indent: int = 0, parent: 'TreeLogger' = None, fmt: List[int] = None,
                 include_root_hook: bool = True):
        self.indent = indent
        self.parent = parent
        if fmt is None:
            fmt = [BLUE]
        self.fmt = fmt
        self.nodes = []
        self.strings = []
        self.is_last = True
        self.include_root_hook = include_root_hook

    @contextmanager
    def add_node(self, fmt: List[int] = None) -> ContextManager['TreeLogger']:
        node = TreeLogger(parent=self, fmt=self.fmt if fmt is None else fmt)
        try:
            yield node
        finally:
            if len(self.nodes) > 0:
                self.nodes[-1].is_last = False
            self.nodes.append(node)

    def log(self, s: str = None):
        self.strings.append(s)

    def log_leaf(self, s: str):
        with self.add_node() as node:
            node.log(s)

    @property
    def path(self) -> List['TreeLogger']:
        tree = self
        path = []
        while tree is not None:
            path.append(tree)
            tree = tree.parent
        return path[::-1]

    def print_head(self, is_visible: bool, is_leaf: bool):
        if self.include_root_hook:
            if is_leaf:
                head = '└─ ' if self.is_last else '├─ '
            else:
                head = '   ' if self.is_last and not is_visible else '│  '
        else:
            head = ''
        fmt = self.fmt if self.parent is None else self.parent.fmt
        print(f"{' ' * self.indent}\033[{';'.join(map(str, fmt))}m{head}{FMT_RST}", end='')

    def print(self):
        paths = self.path
        # Print all lines of node
        is_first = True
        for s_line in self.strings:
            for parent in paths[:-1]:
                parent.print_head(is_visible=not parent.is_last, is_leaf=False)
            self.print_head(is_visible=not self.is_last, is_leaf=is_first)
            if s_line is not None:
                print(s_line)
            else:
                print()
            is_first = False

        # Print children
        for child in self.nodes:
            child.print()

    def __repr__(self):
        return '\n'.join(self.strings)


@contextmanager
def log_tree(indent: int, include_root_hook: bool = True) -> ContextManager[TreeLogger]:
    tree_log = TreeLogger(indent, include_root_hook=include_root_hook)
    try:
        yield tree_log
    finally:
        tree_log.print()
