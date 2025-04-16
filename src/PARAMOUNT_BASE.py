import os
import re
import psutil
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from pathlib import Path
import src.utils as utils


class Base:
    """
    Base class for handling data processing and visualization tasks in PARAMOUNT.

    This class provides methods for creating Dask clusters, reading and processing CSV files,
    converting data to Parquet format, performing signal analysis, and generating visualizations.
    It also includes utilities for setting analysis parameters and managing data transformations.

    Attributes:
        cluster (LocalCluster): The Dask cluster instance.
        client (Client): The Dask client instance.
    """

    def __init__(self, show_dashboard) -> None:
        self.cluster, self.client = self.create_cluster(show_dashboard)

    def create_cluster(self, show_dashboard):
        """
        create_cluster create a dask cluster. modify LocalCluster for parallel runs on
        remote clusters. webbrowser.open() shows the dask operation in browser.

        Returns:
            tuple: cluster, client
        """

        total_memory = psutil.virtual_memory().total
        core_count = psutil.cpu_count(logical=True)

        n_workers = max(6, core_count // 3)  # Number of workers
        threads_per_worker = 2  # Number of threads per worker
        memory_limit = f"{total_memory / 2**30}GB"  # Memory per worker = total_memory (over allocated)

        cluster = LocalCluster(
            dashboard_address="localhost:8000",
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            silence_logs=logging.WARN,
        )
        client = Client(cluster)

        print(client.cluster)
        print(client.dashboard_link)
        if show_dashboard:
            import webbrowser

            webbrowser.open(cluster.dashboard_link, new=2)

        return cluster, client

    @staticmethod
    def read_csv_columns(
        path_csv=Path.cwd(), skiprows=0, delimiter=",", boolPrint=True
    ):
        """
        read_csv_columns get columns of a csv file and print the headers


        Args:
            path_csv (str, optional): path to folder containing csv files. Defaults to Path.cwd().
            skiprows (int, optional): rows to skip in each csv file. Defaults to 0.
            delimiter (str, optional): delimiter used in csv files. Defaults to ",".
            boolPrint (bool, optional): whether to print our the columns. Defaults to True.

        Returns:
            list: the headers found in the first csv file
        """
        """

        Returns:
            list: headers in csv file
        """
        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))
        df = pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)
        print(df.head())
        headers = df.columns.tolist()
        if boolPrint:
            print(pd.DataFrame({"Available headers": headers}))
        return headers

    @staticmethod
    def check_csv(
        path_csv=Path.cwd(),
        delimiter=",",
        skiprows=0,
    ):
        """
        check_csv compares the number of rows and columns in multiple CSV files
        within a specified directory to ensure consistency.

        Args:
            path_csv (Path, optional): Path to the folder containing CSV files. Defaults to Path.cwd().
            delimiter (str, optional): Delimiter used in the CSV files. Defaults to ",".
            skiprows (int, optional): Number of rows to skip at the start of each CSV file. Defaults to 0.
        """
        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))
        result0 = pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)
        for path_csv in tqdm(pathlist[1:], "Comparing csv line numbers"):
            result = pd.read_csv(path_csv, sep=delimiter, skiprows=skiprows)
            if result.shape != result0.shape:
                print(" Mismatch found")
                print(f"{path_csv.name} {result.shape}")

    @staticmethod
    def get_folderlist(
        path=".data", boolPrint=True, ignoreStack=False, onlyStack=False
    ):
        """
        get_folderlist retrieves a list of folders in a given path.

        Args:
            path (str, optional): Path to look for folders. Defaults to ".data".
            boolPrint (bool, optional): Whether to print the list of folders. Defaults to True.
            ignoreStack (bool, optional): Whether to ignore folders starting with "stack". Defaults to False.
            onlyStack (bool, optional): Whether to include only folders starting with "stack". Defaults to False.

        Returns:
            list: List of available folders in the specified path.
        """
        files = Path.cwd() / path
        if not files.is_dir():
            os.makedirs(files)

        folderlist = [f for f in files.iterdir() if f.is_dir()]
        folderlist = [folder.parts[-1] for folder in folderlist]
        if ignoreStack:
            folderlist = [
                folder for folder in folderlist if folder.split("_")[0] != "stack"
            ]
        if onlyStack:
            if ignoreStack:
                print("WARNING: ignoreStack flag is ignored")
            folderlist = [
                folder for folder in folderlist if folder.split("_")[0] == "stack"
            ]
        if boolPrint:
            print(pd.DataFrame({"Available folders": folderlist}))
        return folderlist

    def csv_to_parquet(
        self,
        variables,
        coordinates="2D",
        path_csv=Path.cwd(),
        path_parquet=".data",
        i_start=0,
        i_end=None,
        delimiter=",",
        skiprows=0,
    ):
        """
        csv_to_parquet reads all CSV files in the specified path and saves the desired variables in Parquet format.
        Spatial values (x, y, z) are extracted from the first CSV file.

        Args:
            variables (list): List of variables to extract and save.
            coordinates (str): Cartesian coordinates to store, e.g., '2D' for 'xy' and '3D' for 'xyz'.
            path_csv (Path, optional): Path to the folder containing CSV files. Defaults to Path.cwd().
            path_parquet (str, optional): Path to save the Parquet database. Defaults to ".data".
            i_start (int, optional): Index of the first CSV file to process. Defaults to 0.
            i_end (int, optional): Index of the last CSV file to process. Defaults to None, which means all files.
            delimiter (str, optional): Delimiter used in the CSV files. Defaults to ",".
            skiprows (int, optional): Number of rows to skip in each CSV file. Defaults to 0.

        Raises:
            ValueError: If existing folders are found and the user chooses not to overwrite them.
        """

        variables = variables if type(variables) is list else [variables]

        self.make_dim(coordinates)

        utils.ensure_dir(path_parquet)

        v_ = variables.copy()
        v_ = self.overwrite_guard(path_parquet, v_)
        variables = v_

        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))
        if i_end is not None:
            if len(pathlist) < i_end:
                print("i_end is longer than pathlist")
                i_end = len(pathlist)
            pathlist = pathlist[i_start:i_end]
        else:
            pathlist = pathlist[i_start:]

        df = pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)

        try:
            for item in df.columns:
                if re.match(r"\s*" + "x", item, re.IGNORECASE):
                    variables.remove(item)
                    if "x" in self.dim:
                        utils.saveit(df[item], f"{path_parquet}/x.pkl")
                if re.match(r"\s*" + "y", item, re.IGNORECASE):
                    variables.remove(item)
                    if "y" in self.dim:
                        utils.saveit(df[item], f"{path_parquet}/y.pkl")
                if re.match(r"\s*" + "z", item, re.IGNORECASE):
                    variables.remove(item)
                    if "z" in self.dim:
                        utils.saveit(df[item], f"{path_parquet}/z.pkl")
        except Exception as e:
            print(f"warning: you did not specify all coordinate variables: {e}")

        results = [[] for x in range(len(variables))]

        for path_csv in tqdm(pathlist, "analyzing csv files"):
            for i, var in enumerate(variables):
                result = dd.read_csv(path_csv, sep=delimiter, skiprows=skiprows)
                results[i].append(result[var])

        for i, var in enumerate(tqdm(variables, "writing parquet database")):
            df = dd.concat(results[i], axis=1, ignore_unknown_divisions=True)
            df.columns = [path.stem for path in pathlist]
            dd.to_parquet(
                df.repartition(partition_size="150MB", force=True),
                f"{path_parquet}/{var.strip()}",
                compression="snappy",
                write_metadata_file=True,
            )

    @staticmethod
    def extract_csv_sequential(
        path_csv=Path.cwd(),
        path_save=".data",
        delimiter=",",
        x0=0,
        y0=0,
        tol=1e-3,
        skiprows=0,
        booldisplay=False,
    ):
        """
        extract_csv_sequential extracts data from a specific point in a sequential manner.

        Args:
            path_csv (Path, optional): Path to the folder containing CSV files. Defaults to Path.cwd().
            path_save (str, optional): Path to save the extracted results. Defaults to ".data".
            delimiter (str, optional): Delimiter used in the CSV files. Defaults to ",".
            x0 (float, optional): Target x-coordinate for point extraction. Defaults to 0.
            y0 (float, optional): Target y-coordinate for point extraction. Defaults to 0.
            tol (float, optional): Tolerance for point detection. Defaults to 1e-3.
            skiprows (int, optional): Number of rows to skip in each CSV file. Defaults to 0.
            booldisplay (bool, optional): Whether to display a scatter plot of the points. Defaults to False.
        """
        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))
        df = pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        tol = 1e-3

        for i in range(len(x)):
            if abs(x[i] - x0) < tol and abs(y[i] - y0) < tol:
                index = i
                break

        if booldisplay:
            import matplotlib.pyplot as plt
            from IPython.display import display

            fig, ax = plt.subplots()
            fig.set_size_inches(6, 15)
            ax.scatter(x, y, color="k")
            ax.scatter(x[index], y[index], color="r")
            display(fig)

        variables = df.columns[3:]

        results = []
        for path_csv in tqdm(pathlist, "analyzing csv files"):
            result = pd.read_csv(path_csv, sep=delimiter, skiprows=skiprows)
            results.append(result.iloc[index, :])
        results = pd.concat(results, axis=1)
        results.columns = [path.stem for path in pathlist]
        results = results.T
        print(results)
        utils.ensure_dir(path_save)
        for var in variables:
            utils.saveit(results[var], f"{path_save}/{var}")

    @staticmethod
    def read_csv_sequential(
        path_csv=Path.cwd(),
        path_save=".data",
        delimiter=",",
        skiprows=0,
    ):
        """
        read_csv_sequential reads values from a CSV database that may contain
        unclean data, such as units (e.g., "3.123 [m/s]") in each row.

        Args:
            path_csv (Path, optional): Path to the folder containing the CSV files. Defaults to Path.cwd().
            path_save (str, optional): Path to save the results. Defaults to ".data".
            delimiter (str, optional): Delimiter used in the CSV files. Defaults to ",".
            skiprows (int, optional): Number of rows to skip in each CSV file. Defaults to 0.
        """
        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))

        df = pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)
        df.applymap(lambda x: float(re.sub(r"\[(.*?)\]", "", x)))
        print(df.head())
        headers = df.columns.tolist()
        print(pd.DataFrame({"Available headers": headers}))
        variables = headers

        results = [[] for x in range(len(variables))]
        for path_csv in tqdm(pathlist, "analyzing csv files"):
            for i, var in enumerate(variables):
                result = pd.read_csv(path_csv, sep=delimiter, skiprows=skiprows)
                result = result.applymap(lambda x: float(re.sub(r"\[(.*?)\]", "", x)))
                results[i].append(result[var][0])

        for i, var in enumerate(variables):
            utils.ensure_dir(path_save)
            print(f"saving {var} {results[i][:10]}")
            utils.saveit(results[i], f"{path_save}/{var.strip()}")

    @staticmethod
    def fft_signal(signal, dt, path_save=Path.cwd(), fmax=3000):
        """
        fft_signal computes and plots the Fast Fourier Transform (FFT) of a given signal.

        Args:
            signal (Series): The input signal data.
            dt (float): The time interval between consecutive samples in the signal.
            path_save (Path, optional): The directory where the FFT plot will be saved. Defaults to the current working directory.
            fmax (int): The maximum frequency to display in the FFT plot.
        """
        import numpy as np
        from scipy.fft import fft, fftfreq
        import matplotlib.pyplot as plt

        plt.rc("font", family="Times New Roman")
        plt.rc("font", size=14)

        N = len(signal)
        T = dt

        yf = fft(signal)
        xf = fftfreq(N, T)[: N // 2]

        yff = 2.0 / N * np.abs(yf[1 : N // 2])
        xff = xf[1 : N // 2]

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 4)
        fig.patch.set_facecolor("w")
        ax.plot(xff, yff, "k", linewidth=0.75)
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, max(yff) * 1.05)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("FFT magnitude")

        ax.set_axisbelow(True)
        ax.grid(alpha=0.5, which="both")

        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(0.5)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(f"{path_save}/fft" + ".png", dpi=300, bbox_inches="tight")
        plt.close("all")
        plt.show()
        return xff, yff

    @staticmethod
    def check_parquet(path_parquet=".data"):
        """
        check_parquet reads and prints the head of all Parquet files in the specified path.

        Args:
            path_parquet (str, optional): Path to the folder containing Parquet files. Defaults to ".data".
        """
        pathlist = (Path.cwd() / path_parquet).glob("*/*.parquet")
        for path in tqdm(pathlist, "reading parquet databases"):
            print(path)
            df = dd.read_parquet(path, engine="pyarrow")
            print(df.head())
            print(len(df))

    @staticmethod
    def correlate(v1, v2):
        """
        correlate calculates the Pearson correlation coefficient and the maximum correlation
        found for a time lag between two signals.

        Args:
            v1 (array-like): First variable's time series.
            v2 (array-like): Second variable's time series.

        Returns:
            tuple: A tuple containing:
            - Pearson correlation coefficient (float).
            - Maximum correlation coefficient with time lag adjustment (float).
        """
        from scipy import signal

        corr = (
            signal.correlate(v1 - np.mean(v1), v2 - np.mean(v2), mode="full")
            / len(v1)
            / np.std(v1)
            / np.std(v2)
        )
        corr_smooth = (
            pd.DataFrame(corr)
            .rolling(window=max(int(len(v1) * 0.01), 5), center=True, closed="both")
            .mean()
        )
        lagindex = corr_smooth.iloc[:, 0].argmax()

        corrcoef = corr[int(len(corr) / 2)]
        corrcoef_adj = corr[lagindex]
        return corrcoef, corrcoef_adj

    def correlation_signals(
        self,
        path_signals=".signals",
        path_signals2=".signals2",
        path_viz=".viz",
    ):
        """
        correlation_signals plot correlation heatmap between signals.
        The signals are assumed to be in pickled format in two folders

        Args:
            path_signals (str, optional): folder for first set of signals. Defaults to ".signals".
            path_signals2 (str, optional): folder for second set of signals. Defaults to ".usv2".
            path_viz (str, optional): path to store plots in. Defaults to ".viz".
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        utils.ensure_dir(path_viz)

        pathlist = Path(path_signals).resolve().glob("*")
        pathlist2 = Path(path_signals2).resolve().glob("*")

        signals = []
        for path in pathlist:
            signals.append(pd.Series(utils.loadit(path), name=path.name))
        signaldf = pd.concat(signals, axis=1)

        signals = []
        for path in pathlist2:
            signals.append(pd.Series(utils.loadit(path), name=path.name))
        signaldf2 = pd.concat(signals, axis=1)

        dfcorr_adjusted = pd.DataFrame(
            np.nan, columns=range(signaldf.shape[1]), index=range(signaldf2.shape[1])
        )
        dfcorr = pd.DataFrame(
            np.nan, columns=range(signaldf.shape[1]), index=range(signaldf2.shape[1])
        )

        for i in tqdm(range(signaldf.shape[1]), "computing correlations"):
            for j in range(signaldf2.shape[1]):
                v1 = signaldf.iloc[:, i]
                v2 = signaldf2.iloc[:, j]
                corrcoef, corrcoef_adj = self.correlate(v1, v2)
                dfcorr.iat[i, j] = corrcoef
                dfcorr_adjusted.iat[i, j] = corrcoef_adj

        for name, df in zip(["corr", "corr_adjusted"], [dfcorr, dfcorr_adjusted]):
            fig, ax = plt.subplots()
            fig.set_size_inches(
                self.width * signaldf.shape[1] / signaldf2.shape[1], self.width
            )
            fig.patch.set_facecolor("w")

            ax1 = ax.imshow(df.abs(), interpolation="none", aspect="equal")
            ax1.set_clim(0, 1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right",
                size=self.width * 5 / 100,
                pad=self.width * 2 / 100,
            )
            cbar = fig.colorbar(ax1, cax=cax)
            cbar.ax.set_xlabel("Correlation Coefficient")

            for i in range(1, 1 + signaldf.shape[1]):
                ax.axhline(i - 0.5, color="w")
            for i in range(1, 1 + signaldf2.shape[1]):
                ax.axvline(i - 0.5, color="w")

            sigloc = np.arange(
                0,
                signaldf.shape[1],
                step=1,
            )
            sigloc2 = np.arange(
                0,
                signaldf2.shape[1],
                step=1,
            )
            ax.xaxis.set_ticks(
                sigloc,
                signaldf.columns,
                rotation=90,
                ha="center",
            )
            ax.yaxis.set_ticks(
                sigloc2,
                signaldf2.columns,
            )

            ax.xaxis.remove_overlapping_locs = True
            ax.yaxis.remove_overlapping_locs = False

            plt.savefig(
                f"{path_viz}/Sigs_{name}" + ".png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close("all")

    def make_dim(self, coordinates):
        """
        make_dim sets the analysis dimensions based on the provided coordinates.

        Args:
            coordinates (str): "2D" or "3D" to specify the dimensionality of the analysis.
        """
        coordinates = coordinates.lower().strip()
        if coordinates == "2d":
            self.dim = "xy"
        elif coordinates == "3d":
            self.dim = "xyz"
        else:
            raise ("please specify 2D or 3D as coordinates")

    def make_bounds(self, xyz):
        """
        make_bounds defines the analysis bounds and spatial resolution.

        Args:
            xyz (list): A list containing x, y, and optionally z coordinates of points.

        Returns:
            list: A list of bounds and spatial resolution. For 2D, it returns
              [xmin, xmax, ymin, ymax, res]. For 3D, it returns
              [xmin, xmax, ymin, ymax, zmin, zmax, res].
        """
        if self.dim == "xy":
            x = xyz[0]
            y = xyz[1]
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            lmax = max(xmax, ymax)
            lmin = max(xmin, ymin)
            res = (lmax - lmin) / 1000
            return [xmin, xmax, ymin, ymax, res]
        if self.dim == "xyz":
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            zmin = min(z)
            zmax = max(z)
            lmax = max(xmax, ymax, zmax)
            lmin = max(xmin, ymin, zmin)
            res = (lmax - lmin) / 75
            return [xmin, xmax, ymin, ymax, zmin, zmax, res]

    def set_time(self, dt, t0=0):
        """
        Sets the timestep and initial time for data acquisition.

        Args:
            dt (float): The time interval between consecutive data points.
            t0 (float): The starting time of the data acquisition.
        """
        try:
            self.data_skip
        except:
            self.data_skip = None

        if self.data_skip is not None:
            self.dt = dt * self.data_skip
        else:
            self.dt = dt
        self.t0 = t0

    def data_decimate(self, df, X1=False, X2=False):
        """
        data_decimate preprocesses a DataFrame by applying various transformations such as cutting off columns,
        skipping columns, and slicing columns based on certain conditions.

        Args:
            df (DataFrame): The DataFrame to be decimated.
            X1 (bool, optional): If True, excludes the last column. Defaults to False.
            X2 (bool, optional): If True, excludes the first column. Defaults to False.

        Returns:
            DataFrame: The transformed DataFrame after applying the specified operations.
        """
        try:
            self.data_cutoff
        except:
            self.data_cutoff = None

        try:
            self.data_cutin
        except:
            self.data_cutin = None

        try:
            self.data_skip
        except:
            self.data_skip = None

        if self.data_cutoff is not None:
            cutoff = int(df.shape[1] * self.data_cutoff / 100)
            try:
                df = df[:, :cutoff]
            except:
                df = df.iloc[:, :cutoff]

        if self.data_cutin is not None:
            cutin = int(df.shape[1] * self.data_cutin / 100)
            try:
                df = df[:, cutin:]
            except:
                df = df.iloc[:, cutin:]

        if self.data_skip is not None:
            df = df[:, :: self.data_skip]

        if X1:
            try:
                df = df[:, :-1]
            except:
                df = df.iloc[:, :-1]

        if X2:
            try:
                df = df[:, 1:]
            except:
                df = df.iloc[:, 1:]

        return df

    def set_data_cutin(self, data_cutin):
        """
        set_data_cutin sets the percentage of data to be included starting from the beginning of the dataset.

        Args:
            data_cutin (float): The percentage of data to include, specified as a value between 0 and 100.
        """
        self.data_cutin = data_cutin

    def set_data_cutoff(self, data_cutoff):
        """
        set_data_cutoff sets the percentage of data to be retained in the dataset.

        Args:
            data_cutoff (float): The percentage of data to retain, specified as a value between 0 and 100.
        """
        self.data_cutoff = data_cutoff

    def set_data_skip(self, data_skip):
        """
        set_data_skip sets the number of snapshots to be skipped between successive snapshots in the dataset.

        Args:
            data_skip (int): The number of snapshots to skip.
        """
        self.data_skip = data_skip

    def set_viz_params(
        self,
        dpi=300,
        linewidth=1.5,
        color="k",
        cmap="seismic",
        ax_width=0.5,
        font="Times New Roman",
        fontsize=14,
        height=4,
        width=5,
        contour_levels=20,
    ):
        """
        set_viz_params set visualization parameters

        Args:
            dpi (int, optional): dpi to save figures. Defaults to 300.
            linewidth (float, optional): line width. Defaults to 1.5.
            color (str, optional): line color. Defaults to 'k'.
            cmap (str, optional): color map to use. Defaults to 'seismic'.
            ax_width (float, optional): linewidth for axes of plots. Defaults to 0.5.
            font (str, optional): font family used in plots. Defaults to "Times New Roman".
            fontsize (int, optional): font size used in plots. Defaults to 14.
            height (int, optional): plot height in inches. Defaults to 4.
            width (int, optional): plot width in inches. Defaults to 5.
        """
        self.dpi = dpi
        self.linewidth = linewidth
        self.color = color
        self.cmap = cmap
        self.ax_width = ax_width
        self.font = font
        self.fontsize = fontsize
        self.width = width
        self.height = height
        self.contour_levels = contour_levels

    def dist_map(self, x, y, bounds):
        """
        dist_map generates a k-d tree distance map for all xy coordinates.
        This is used to mask visualization results for regions where no data exists.

        Args:
            x (list): List of x-coordinate values.
            y (list): List of y-coordinate values.
            bounds (list): Domain bounds for visualization in the format
            [xmin, xmax, ymin, ymax, resolution].

        Returns:
            ndarray: k-d tree distance map for all xy coordinate pairs.
        """
        from scipy.spatial import KDTree

        xx, yy = self.make_meshgrid(bounds)

        tree = KDTree(np.c_[x, y])
        dist, _ = tree.query(np.c_[xx.ravel(), yy.ravel()], k=1)
        dist = dist.reshape(xx.shape)
        return dist

    def make_meshgrid(self, bounds):
        """
        make_meshgrid generates a meshgrid for the domain described by its boundary and resolution.

        Args:
            bounds (list): Domain bounds for visualization. For 2D, it should be
                   [xmin, xmax, ymin, ymax, resolution]. For 3D, it should be
                   [xmin, xmax, ymin, ymax, zmin, zmax, resolution].

        Returns:
            list: A list containing numpy arrays representing the meshgrid.
              For 2D, it returns [xx, yy]. For 3D, it returns [xx, yy, zz].
        """
        if self.dim == "xy":
            xmin, xmax, ymin, ymax, res = bounds
            mgrid = np.meshgrid(
                np.arange(
                    xmin,
                    xmax + res,
                    res,
                ),
                np.arange(
                    ymin,
                    ymax + res,
                    res,
                ),
            )
            return mgrid

        if self.dim == "xyz":
            xmin, xmax, ymin, ymax, zmin, zmax, res = bounds
            mgrid = np.meshgrid(
                np.arange(
                    xmin,
                    xmax + res,
                    res,
                ),
                np.arange(
                    ymin,
                    ymax + res,
                    res,
                ),
                np.arange(
                    zmin,
                    zmax + res,
                    res,
                ),
            )
            return mgrid
