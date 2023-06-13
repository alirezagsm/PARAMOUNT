from contextlib import contextmanager
import time

if __name__ == "__main__":
    raise Exception("!!!You can't run this file directly!!!")


class utils:
    @contextmanager
    def waiting():
        """appends waiting text at the end on stdout"""
        import sys

        old_stdout = sys.stdout
        print("task started")
        from shutil import get_terminal_size

        terminalsize = get_terminal_size().columns

        class CustomPrint:
            def __init__(self, stdout, terminalsize):
                self.stdout = stdout
                self.terminalsize = terminalsize

            def write(self, text):
                if len(text.rstrip()):
                    message = "waiting... "
                    if text[0] == "|":
                        self.stdout.write(f"\r{text}  {message}")
                    else:
                        self.stdout.write(f"\r{' '*terminalsize}\r{text}\n{message}")
                        with open("Glog.txt", "a", encoding="utf-8") as f:
                            f.write(text + "\n")

        sys.stdout = CustomPrint(sys.__stdout__, terminalsize=terminalsize)

        try:
            yield
        finally:
            sys.stdout = old_stdout
            print(f"\r{' '*terminalsize}\rtask completed")

    @staticmethod
    def progressbar(start, end, index, mod=1):
        """prints a progress bar works well with waiting"""
        realstart = (start // mod) * mod
        realend = (end // mod) * mod if end % mod != 0 else (end // mod - 1) * mod
        percent = float(index - realstart) / float(realend - realstart)

        from shutil import get_terminal_size

        stdsize = int(get_terminal_size().columns / 3)
        barLength = stdsize
        block = int(round(barLength * percent))
        pbar_char = "▆"
        empty_char = " "
        pbar = f"|{pbar_char * block + empty_char * (barLength - block)}| {round(percent * 100, 0):.0f}%"
        print(pbar)

    @staticmethod
    def saveit(obj: object, name: str) -> None:
        import pickle
        from pathlib import Path

        utils.ensure_dir(Path(name).parent)
        with open(name, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def loadit(name: str) -> object:
        import pickle

        with open(name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def ensure_dir(path):
        import os

        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

    @contextmanager
    def timeit(description="task duration") -> None:
        start = time.time()
        yield
        ellapsed_time = time.time() - start
        from datetime import timedelta

        print(f"{description}: {str(timedelta(seconds = ellapsed_time))}")
