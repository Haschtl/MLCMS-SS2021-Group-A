from matplotlib import pyplot as plt
from task2_subtask1 import subtask1
from task2_subtask2 import subtask2
from task2_subtask3 import subtask3


def console(_globals=None, _locals=None):
    """
    Opens interactive console with current execution state.
    Call it with: `console()`
    """
    if not _globals:
        _globals = globals()
    if not _locals:
        _locals = locals()
    import code
    import readline
    import rlcompleter
    context = _globals.copy()
    context.update(_locals)
    readline.set_completer(rlcompleter.Completer(context).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(context)
    shell.interact()


if __name__ == "__main__":
    plt.ion()
    if input("Execute subtask 1? [y/N]: ").lower() == "y":
        subtask1()
    if input("Execute subtask 2? [y/N]: ").lower() == "y":
        subtask2()
    if input("Execute subtask 3? [y/N]: ").lower() == "y":
        subtask3()
    plt.ioff()
    plt.show()
