from dask.distributed import Client, LocalCluster
from dask import visualize
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
import re
from src.utils import utils
import dask.array as da
import webbrowser
from tqdm import tqdm
import shutil
import logging

# suppress dask user warnings
logging.getLogger("distributed.utils_perf").setLevel(logging.CRITICAL)
logging.getLogger("distributed.diskutils").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker_memory").setLevel(logging.CRITICAL)

# help pd output fit in console
pd.set_option("display.max_colwidth", 20)


class POD:
    """

    handle loading and processing csv datasets for POD analysis
    this class can read csv files in bulk and save to fast parquet format
    SVD analysis result are stored in parquet format for fast processing
    pd.Series are serialized and stored in pickle format
    Refer to svd_example.py, csv_example.py and viz_example.py for examples on how to utilize UTPOD

    Available Functions
    ----------
    create_cluster
    ead_csv_coordinates
    remove_coordinate_variables
    read_csv_columns
    get_folderlist
    csv_to_parquet
    check_parquet
    svd_save_usv
    set_time
    svd_viz

    """

    def __init__(self) -> None:

        self.cluster, self.client = self.create_cluster()

    def create_cluster(self):
        """
        create_cluster create a dask cluster. modify LocalCluster for parallel runs on
        remote clusters. webbrowser.open() shows the dask operation in browser.

        Returns:
            tuple: cluster, client
        """
        cluster = LocalCluster(dashboard_address="localhost:8000")
        client = Client(cluster)

        print(client.cluster)
        print(client.dashboard_link)
        webbrowser.open(cluster.dashboard_link, new=2)

        return cluster, client

    def read_csv_coordinates(self, df):
        """
        read_csv_coordinates read x y z cooridinates from dataframe

        Args:
            df (dataframe): dataframe containing cooridinates

        Returns:
            tuple: x, y, z series
        """
        strlist = ["x", "y", "z"]
        for item in df.columns:  # get node XYZ
            if re.match(r"\s*" + strlist[0], item, re.IGNORECASE):
                x = df[item]
                continue
            if re.match(r"\s*" + strlist[1], item, re.IGNORECASE):
                y = df[item]
                continue
            if re.match(r"\s*" + strlist[2], item, re.IGNORECASE):
                z = df[item]
                continue
        return x, y, z

    def remove_coordinate_variables(self, variables):
        """
        remove_coordinate_variables remove coordinate entries from variables list

        Returns:
            list: cleaned list of variables
        """
        strlist = ["x", "y", "z"]
        for item in variables[:]:  # get node XYZ
            if re.match(r"\s*" + strlist[0], item, re.IGNORECASE):
                variables.remove(item)
                continue
            if re.match(r"\s*" + strlist[1], item, re.IGNORECASE):
                variables.remove(item)
                continue
            if re.match(r"\s*" + strlist[2], item, re.IGNORECASE):
                variables.remove(item)
                continue
        return variables

    @staticmethod
    def read_csv_columns(
        path_csv=Path.cwd(), skiprows=0, delimiter=",", boolPrint=True
    ):
        """
        read_csv_columns get columns of a csv file and print the headers

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
    def get_folderlist(path=".data", boolPrint=True):
        """
        get_folderlist get list of folders in a given path

        Args:
            path (str, optional): path to look for folders. Defaults to ".data".
            boolPrint (bool, optional): whether to print out the folders. Defaults to True.

        Returns:
            list: list of available folders
        """
        files = Path.cwd() / path
        folderlist = [f for f in files.iterdir() if f.is_dir()]
        folderlist = [folder.parts[-1] for folder in folderlist]
        if boolPrint:
            print(pd.DataFrame({"Available folders": folderlist}))
        return folderlist

    def csv_to_parquet(
        self,
        variables,
        path_csv=Path.cwd(),
        path_parquet=".data",
        i_start=0,
        i_end=None,
        delimiter=",",
        skiprows=0,
    ):
        """
        read all csv files in path and save desired variables in parquet format

        Raises:
            ValueError: checking for existing folders and warn user about unwanted overwrites
        """
        variables = variables if type(variables) is list else [variables]
        for var in variables:
            if var.strip() in str(list((Path.cwd() / path_parquet).glob("*"))):
                while (
                    input(f"{var.strip()} folder already exists! Continue? [y/n]")
                    != "y"
                ):
                    self.client.shutdown()
                    raise ValueError("please correct your variable list")

        pathlist = sorted(Path(path_csv).resolve().glob("*.csv"))
        if i_end is not None:
            if len(pathlist) < i_end:
                print("i_end is longer than pathlist")
                i_end = len(pathlist)
            pathlist = pathlist[i_start:i_end]
        else:
            pathlist = pathlist[i_start:]

        x, y, z = self.read_csv_coordinates(
            pd.read_csv(pathlist[0], sep=delimiter, skiprows=skiprows)
        )
        utils.saveit(x, f"{path_parquet}/x.pkl")
        utils.saveit(y, f"{path_parquet}/y.pkl")
        variables = self.remove_coordinate_variables(variables)

        for var in tqdm(variables, "Writing parquet database"):
            results = []
            for path_csv in pathlist:
                result = dd.read_csv(path_csv, sep=delimiter, skiprows=skiprows)
                results.append(result[var])
            df = dd.concat(results, axis=1, ignore_unknown_divisions=True)
            df.columns = [path.stem for path in pathlist]
            dd.to_parquet(
                df.repartition(partition_size="150MB", force=True),
                f"{path_parquet}/{var.strip()}",
                compression="snappy",
                write_metadata_file=True,
            )

    @staticmethod
    def check_parquet(path_parquet=".data"):
        """
        check_parquet read and print head of all parquet files in path

        Args:
            path_parquet (str, optional): path to look for parquet files. Defaults to ".data".
        """
        pathlist = (Path.cwd() / path_parquet).glob("*/*.parquet")
        for path in tqdm(pathlist, "reading parquet databases"):
            print(path)
            df = dd.read_parquet(path, engine="pyarrow")
            print(df.head())
            print(len(df))

    def svd_save_usv(
        self,
        variables,
        path_parquet=".data",
        path_results_pod=".usv",
    ):
        """
        svd_save_usv compute distributed Singular Value Decomposition and store results in parquet format

        Args:
            variables (list or str): list of variables to consider
            path_parquet (str, optional): path to parquet datasets. Defaults to ".data".
            path_results_pod (str, optional): path to store SVD results in. Defaults to ".usv".

        Raises:
            ValueError: checking for existing folders and warn user about unwanted overwrites
        """
        variables = variables if type(variables) is list else [variables]
        for var in variables:
            if var in str(list((Path.cwd() / path_results_pod).glob("*"))):
                while (
                    input(f"{var.strip()} folder already exists! Continue? [y/n]")
                    != "y"
                ):
                    self.client.shutdown()
                    raise ValueError("please correct your variable list")
        shutil.copy(Path.cwd() / path_parquet / "x.pkl", path_results_pod)
        shutil.copy(Path.cwd() / path_parquet / "y.pkl", path_results_pod)
        for var in tqdm(variables, "Calculating POD modes"):
            path = Path.cwd() / path_parquet / f"{var}"
            df = dd.read_parquet(path, engine="pyarrow")
            u, s, v = da.linalg.svd(df.values)

            for name, item in zip(["u", "v"], [u, v]):
                result = dd.from_array(item)
                result.columns = result.columns.astype(str)
                dd.to_parquet(
                    result,
                    f"{path_results_pod}/{var}/{name}",
                    compression="snappy",
                    write_metadata_file=True,
                )
            utils.saveit(s.compute(), f"{path_results_pod}/{var}/s.pkl")

    def svd_viz(
        self,
        variables,
        modelist,
        dt,
        bounds,
        path_results_pod=".usv",
        t0=0,
        path_viz=".viz",
    ):
        """
        svd_viz distributed Singular Value Decomposition visualization

        Args:
            variables (list or str): list of variables to consider
            modelist (list): modes to consider
            dt (float): timestep of data acquisition
            bounds (list): domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]
            path_results_pod (str, optional): path to read SVD results from. Defaults to ".usv".
            t0 (int, optional): initial time to use in visualization. Defaults to 0.
        """
        variables = variables if type(variables) is list else [variables]
        modelist = modelist if type(modelist) is list else list(modelist)

        path_x = Path.cwd() / path_results_pod / "x.pkl"
        path_y = Path.cwd() / path_results_pod / "y.pkl"
        x = utils.loadit(path_x)
        y = utils.loadit(path_y)
        for var in tqdm(variables, "Visualizing POD modes"):

            path_u = Path.cwd() / path_results_pod / f"{var}" / "u"
            path_v = Path.cwd() / path_results_pod / f"{var}" / "v"
            path_s = Path.cwd() / path_results_pod / f"{var}" / "s.pkl"

            u = dd.read_parquet(path_u, engine="pyarrow")
            v = dd.read_parquet(path_v, engine="pyarrow")
            s = utils.loadit(path_s)

            self.uv_viz(x, y, u, v, f"{path_viz}/{var}", modelist, bounds)
            self.s_viz(s, f"{path_viz}/{var}", modelist)

    def set_time(self, dt, t0):
        """
        set_time set timestep and initial time of data acquisiton

        Args:
            dt (float): timestep
            t0 (float): initial time
        """
        self.dt = dt
        self.t0 = t0

    def uv_viz(self, x, y, u, v, path_viz, modelist, bounds):
        """
        uv_viz visualize u and v matrix of SVD result

        Args:
            x (series): x coordinate
            y (series): y coordinate
            u (dataframe): u matrix from SVD analysis
            v (dataframe): v matrix from SVD analysis
            path_viz (str): path to save results
            modelist (list): list of modes to visualize
            bounds (list): domain bounds for visualization [xmin,xmax,ymin,ymax,resolution]
        """
        from scipy.interpolate import griddata
        from scipy.signal import find_peaks
        import numpy as np

        import matplotlib.pyplot as plt

        plt.rc("font", family="Arial")
        plt.rc("font", size=9)

        # define 2D bounds and resolution

        xmin, xmax, ymin, ymax, res = bounds

        xx, yy = np.meshgrid(
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

        for mode in modelist:
            uu = u.iloc[:, mode].compute()
            zz = griddata((x, y), uu, (xx, yy), method="linear")

            fig, ax = plt.subplots(1)
            fig.set_size_inches(3, 4)
            fig.patch.set_facecolor("w")
            fig.dpi = 300
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"u {mode}")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect(1)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.5)

            ax.contourf(xx, yy, zz, 20, cmap="viridis")
            fig.tight_layout()
            plt.savefig(f"{path_viz}/u{mode}" + ".png")

            vv = v.compute().iloc[mode, :]
            tt = np.arange(self.t0, vv.shape[0] * self.dt, self.dt)

            fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(4, 6)
            fig.patch.set_facecolor("w")
            fig.dpi = 150
            ax[0].set_xlabel("Time [s]")
            ax[0].set_ylabel("Coefficient")
            ax[0].set_title(f"v {mode}")
            ax[0].grid(alpha=0.5)
            ax[0].plot(tt, vv, "k")

            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Power Spectral Density [db/Hz]")
            ax[1].grid(alpha=0.5)
            import matplotlib.mlab as mlab

            Pxx, freqs = mlab.psd(
                vv, Fs=1 / self.dt, window=mlab.window_hanning, detrend="linear"
            )
            dbPxx = 10 * np.log10(Pxx)
            peaks, _ = find_peaks(dbPxx)
            ax[1].plot(freqs, dbPxx, "k")
            npeaks = 2
            for n in range(0, npeaks):
                ax[1].scatter(
                    freqs[peaks[n]],
                    dbPxx[peaks[n]],
                    s=80,
                    facecolors="none",
                    edgecolors="r",
                )
                ax[1].annotate(
                    f"{freqs[peaks[n]]:0.0f}",
                    xy=(freqs[peaks[n]] * 1.01, dbPxx[peaks[n]] * 1.01),
                )
            fig.tight_layout()
            plt.show()
            plt.savefig(f"{path_viz}/v{mode}" + ".png")

    def s_viz(self, s, path_viz, modelist=20):
        """
        s_viz visualize s diagonal matrix of SVD result

        Args:
            s (series): eigenvalues of SVD analysis
            path_viz (str): path to save results
            modelist (int, optional): Defaults to 20.
        """
        import matplotlib.pyplot as plt

        plt.rc("font", family="Arial")
        plt.rc("font", size=9)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(4, 3)
        fig.patch.set_facecolor("w")
        fig.dpi = 300
        ax.set_xlabel("Mode")
        ax.set_ylabel("Energy")
        ax.set_yscale("log")
        ax.set_title(f"s")
        ax.set_aspect(1)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5)

        ax.plot(s[:modelist])
        fig.tight_layout()
        plt.show()
        plt.savefig(f"{path_viz}/s" + ".png")
