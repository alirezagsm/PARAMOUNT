from contextlib import contextmanager
import time

if __name__ == "__main__":
    raise Exception("Please don't run this file directly")


class utils:
    """
    Utility functions for PARAMOUNT.

    This module provides various utility functions for tasks such as
    progress tracking, file saving/loading, directory management,
    DataFrame validation, and time measurement.
    """

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
        """saves an object to a file using pickle"""
        import pickle
        from pathlib import Path

        utils.ensure_dir(Path(name).parent)
        with open(name, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def loadit(name: str) -> object:
        """loads an object from a file using pickle"""
        import pickle

        with open(name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def ensure_dir(path):
        """ensures that the directory exists"""
        import os

        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

    @staticmethod
    def check_dd(path="", df=None) -> None:
        """checks if the DataFrame is clean"""
        import dask.dataframe as dd
        from pathlib import Path

        if df is None:
            df = dd.read_parquet(path)
        df_compute = df.compute()

        if df_compute.isna().sum().sum() != 0:
            raise ValueError("DataFrame contains NaN values")
        if df_compute.isnull().sum().sum() != 0:
            raise ValueError("DataFrame contains Null values")

        if isinstance(path, Path):
            df_name = path.name
        else:
            df_name = path.split("/")[-1]
        print(f"DataFrame {df_name} is clean")

    @contextmanager
    def timeit(description="task duration"):
        """measures the duration of a task"""
        start = time.time()
        yield
        ellapsed_time = time.time() - start
        from datetime import timedelta
        from pathlib import Path

        print(f"{description}: {str(timedelta(seconds = ellapsed_time))}")

    @staticmethod
    def save_to_parquet(
        df,
        path=".data",
        nparts=20,
        columns=None,
        column_dtype=None,
        imaginary_part=True,
    ):
        """saves a DataFrame to a parquet file"""
        import dask.dataframe as dd
        import dask.array as da
        import pandas as pd

        if isinstance(df, da.Array):
            df = dd.from_dask_array(df)

        if isinstance(df, pd.DataFrame):
            df = dd.from_pandas(df, npartitions=nparts)

        if columns is not None:
            df.columns = columns

        if column_dtype:
            df.columns = df.columns.astype(column_dtype)

        data_type = df.dtypes.iloc[0]
        if data_type == float:
            dd.to_parquet(
                df,
                f"{path}",
                compression="snappy",
                write_metadata_file=True,
            )
        elif data_type == complex:
            meta = {col: "float64" for col in df.columns}
            df_real = df.map(lambda x: x.real, meta=meta)
            dd.to_parquet(
                df_real,
                f"{path}",
                compression="snappy",
                write_metadata_file=True,
            )
            if imaginary_part:
                df_imag = df.map(lambda x: x.imag, meta=meta)
                dd.to_parquet(
                    df_imag,
                    f"{path}_imag",
                    compression="snappy",
                    write_metadata_file=True,
                )
        else:
            raise ValueError("Unsupported data type.")

    @staticmethod
    def load_from_parquet(path=".data", imaginary_part=True) -> object:
        """loads an object from a file using pickle"""
        import dask.dataframe as dd
        from pathlib import Path

        path_base = Path(path).parent
        filename = Path(path).name
        if imaginary_part and (path_base / f"{filename}_imag").exists():
            df_real = dd.read_parquet(f"{path}")
            df_imag = dd.read_parquet(f"{path}_imag")
            return df_real + 1j * df_imag
        else:
            return dd.read_parquet(f"{path}")
