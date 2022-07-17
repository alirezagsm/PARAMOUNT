import os
import shutil
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import re
from src.utils import utils
from tqdm import tqdm

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
    read_csv_coordinates
    remove_coordinate_variables
    read_csv_columns
    get_folderlist
    csv_to_parquet
    check_parquet
    svd_save_usv
    set_time
    svd_viz

    """

    def __init__(self, show_dashboard=False) -> None:
        """
        initializes a PARAMOUNT POD class
        Args:
            show_dashboard (bool, optional): Whether to open dask dashboard in browser. Defaults to False.
        """

        self.cluster, self.client = self.create_cluster(show_dashboard)
        self.set_viz_params()

    def create_cluster(self, show_dashboard):
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
        if show_dashboard:
            import webbrowser

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

        for var in tqdm(variables, "writing parquet database"):
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
        for var in tqdm(variables, "calculating POD modes"):
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
        bounds,
        path_results_pod=".usv",
        path_viz=".viz",
        freq_max=3000,
        dist_map=False,
    ):
        """
        svd_viz distributed visualization of Singular Value Decomposition results

        Args:
            variables (list or str): list of variables to consider
            modelist (list): modes to consider
            dt (float): timestep of data acquisition
            bounds (list): domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]
            path_results_pod (str, optional): path to read SVD results from. Defaults to ".usv".
            path_viz (str, optional): path to store plots in. Defaults to ".viz".
            freq_max (float, optional): maximum frequency of interest for PSD plots. Defaults to 3kHz.
            dist_map (bool): whether to compute a mask for mode results. Defaults to False.
        """
        variables = variables if type(variables) is list else [variables]
        modelist = modelist if type(modelist) is list else list(modelist)

        path_x = Path.cwd() / path_results_pod / "x.pkl"
        path_y = Path.cwd() / path_results_pod / "y.pkl"
        x = utils.loadit(path_x)
        y = utils.loadit(path_y)

        if dist_map:
            dist = self.dist_map(x, y, bounds)
        else:
            dist = None

        if not os.path.exists(f"{path_viz}"):
            os.makedirs(f"{path_viz}")

        for var in tqdm(variables, "analyzing variables"):

            path_u = Path.cwd() / path_results_pod / f"{var}" / "u"
            path_v = Path.cwd() / path_results_pod / f"{var}" / "v"
            path_s = Path.cwd() / path_results_pod / f"{var}" / "s.pkl"

            u = dd.read_parquet(path_u, engine="pyarrow")
            v = dd.read_parquet(path_v, engine="pyarrow")
            s = utils.loadit(path_s)

            if not os.path.exists(f"{path_viz}/{var}"):
                os.makedirs(f"{path_viz}/{var}")

            self.uv_viz(
                x, y, u, v, f"{path_viz}/{var}", modelist, bounds, freq_max, dist
            )
            self.s_viz(s, f"{path_viz}/{var}")

    def s_viz_combined(
        self,
        variables,
        maxmode=100,
        path_results_pod=".usv",
        path_viz=".viz",
    ):
        """
        s_viz_combined visualization of all s energy values in one plot

        Args:
            variables (list or str): list of variables to consider
            maxmode (int): final mode to consider. Defaults to 100
            path_results_pod (str, optional): path to read SVD results from. Defaults to ".usv".
            path_viz (str, optional): path to store plots in. Defaults to ".viz".
        """

        variables = variables if type(variables) is list else [variables]

        if not os.path.exists(f"{path_viz}"):
            os.makedirs(f"{path_viz}")

        s_combined = pd.DataFrame(columns=variables)
        for var in variables:

            path_s = Path.cwd() / path_results_pod / f"{var}" / "s.pkl"
            s = utils.loadit(path_s)

            if not os.path.exists(f"{path_viz}/"):
                os.makedirs(f"{path_viz}/")
            mode_energy = [x**2 for x in s]
            mode_energy = mode_energy / sum(mode_energy) * 100
            cumsum = np.cumsum(mode_energy)
            s_combined[var] = cumsum[:maxmode]

        self.s_viz_combined_plot(s_combined, f"{path_viz}")

    def set_time(self, dt, t0=0):
        """
        set_time set timestep and initial time of data acquisiton

        Args:
            dt (float): timestep
            t0 (float): initial time
        """
        self.dt = dt
        self.t0 = t0

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
        width=6,
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
            width (int, optional): plot width in inches. Defaults to 6.
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

    def dist_map(self, x, y, bounds):
        """
        dist_map generate a kd-tree distance map for all xy coordinates. sed to mask the visualization results for which no data exists

        Args:
            x (list): list of x coordination values
            y (list): list of y coordination values
            bounds (list): domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]


        Returns:
            list: k-d tree distance map for all xy coordination pairs.
        """
        from scipy.spatial import KDTree

        xx, yy = self.make_meshgrid(bounds)

        tree = KDTree(np.c_[x, y])
        dist, _ = tree.query(np.c_[xx.ravel(), yy.ravel()], k=1)
        dist = dist.reshape(xx.shape)
        return dist

    def make_meshgrid(self, bounds):
        """
        make_meshgrid generates a meshgrid for the domain described by its boundary and meshgrid resolution

        Args:
            bounds (list): domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]


        Returns:
            ndarray: numpy meshgrid
        """
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

    def uv_viz(self, x, y, u, v, path_viz, modelist, bounds, freq_max, dist):
        """
        uv_viz visualize u and v matrix of SVD result. used in svd.viz.

        Args:
            x (series): x coordinate
            y (series): y coordinate
            u (dataframe): u matrix from SVD analysis
            v (dataframe): v matrix from SVD analysis
            path_viz (str): path to save results
            modelist (list): list of modes to visualize
            bounds (list): domain bounds for visualization [xmin,xmax,ymin,ymax,resolution]
            freq_max (int): maximum frequency to consider is PSD graphs
            dist (array): kd-tree distance map for the domain
        """
        from scipy.interpolate import griddata
        from scipy.signal import find_peaks

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

        plt.rc("font", family="Times New Roman")
        plt.rc("font", size=14)

        # define 2D bounds and resolution
        xmin, xmax, ymin, ymax, res = bounds
        xx, yy = self.make_meshgrid(bounds)

        for mode in tqdm(modelist, "creating POD plots", leave=False):
            uu = u.iloc[:, mode].compute()
            zz = griddata(
                (x, y),
                uu,
                (xx, yy),
                method="linear",
                fill_value=min(abs(uu)),
            )
            if dist is not None:
                # adjust this threshold according to your mesh size
                # this will mask out the parts of visualization for
                # which the distance between points exceeds a certain value
                zz[dist >= res * 13] = np.nan

            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect(1)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.5)
            zz[np.isnan(zz)] = np.min(abs(zz))

            ax.contourf(xx, yy, zz, 50, cmap=self.cmap, antialiased=True, extend="both")
            fig.tight_layout()
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            plt.savefig(
                f"{path_viz}/u{mode}" + ".png", dpi=self.dpi, bbox_inches="tight"
            )

            vv = v.compute().iloc[mode, :]
            tt = np.arange(self.t0, vv.shape[0] * self.dt, self.dt)

            fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(self.width, self.height * 2)
            fig.patch.set_facecolor("w")
            ax[0].set_xlabel("Time [s]")
            ax[0].set_ylabel("Coefficient")
            ax[0].grid(alpha=0.5)
            ax[0].set_xlim(tt[0], tt[-1])
            ax[0].plot(tt, vv, self.color, linewidth=self.linewidth)

            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Power Spectral Density [db/Hz]")
            ax[1].grid(alpha=0.5)
            ax[1].set_xlim(0, freq_max)

            Pxx, freqs = mlab.psd(
                vv, Fs=1 / self.dt, window=mlab.window_hanning, detrend="linear"
            )
            freqs = freqs[np.where(freqs < freq_max)]
            Pxx = Pxx[: len(freqs)]
            dbPxx = 10 * np.log10(Pxx)
            peaks, _ = find_peaks(dbPxx, prominence=3)
            ax[1].plot(freqs, dbPxx, self.color, linewidth=self.linewidth)
            npeaks = 3
            for n in range(0, min(npeaks, len(peaks))):
                ax[1].scatter(
                    freqs[peaks[n]],
                    dbPxx[peaks[n]],
                    s=80,
                    facecolors="none",
                    edgecolors="grey",
                )
                ax[1].annotate(
                    f"{freqs[peaks[n]]:0.0f}",
                    xy=(freqs[peaks[n]] + freq_max / 25, dbPxx[peaks[n]] * 0.99),
                )
            fig.tight_layout()
            for i in range(2):
                for axis in ["bottom", "left"]:
                    ax[i].spines[axis].set_linewidth(self.ax_width)
                for axis in ["top", "right"]:
                    ax[i].spines[axis].set_linewidth(0)
            plt.savefig(
                f"{path_viz}/v{mode}" + ".png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close('all')

    def s_viz(self, s, path_viz, maxmode=100):
        """
        s_viz visualize s diagonal matrix of SVD result

        Args:
            s (series): eigenvalues of SVD analysis
            path_viz (str): path to save results
            modelist (int, optional): Defaults to 20.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Cumulative share of mode energy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5, which="both")

        mode_energy = [x**2 for x in s]
        mode_energy = mode_energy / sum(mode_energy) * 100
        cumsum = np.cumsum(mode_energy)
        s = cumsum[:maxmode]
        ax.set_ylim(s[0] - 10, 100)
        ax.set_xlim(0, maxmode)
        ax.plot(s, self.color, linewidth=self.linewidth)
        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(f"{path_viz}/s" + ".png", dpi=self.dpi, bbox_inches="tight")
        plt.close('all')

    def s_viz_combined_plot(self, s, path_viz):
        """
        s_viz_combined visualize combined s plot

        Args:
            s (pd.Dataframe): eigenvalues of SVD analysis as columns of a dataframe
            path_viz (str): path to save results
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        plt.rc("font", family="Times New Roman")
        plt.rc("font", size=14)

        clrs_list = ["k", "b", "g", "r"]
        styl_list = ["-", "--", "-.", ":"]

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Cumulative share of mode energy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5, which="both")

        ax.set_ylim(min(s.min(axis=1)) - 10, 100)
        ax.set_xlim(0, s.shape[0])
        for i, var in enumerate(s.columns):
            label = re.sub(r"\[(.*?)\]", "", var)
            label = re.sub(r"\.", " ", label)
            clrr = clrs_list[i // 4]
            styl = styl_list[i % 4]
            ax.plot(s[var], linewidth=self.linewidth, label=label, color=clrr, ls=styl)

        ax.legend()
        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(
            f"{path_viz}/s_combined" + ".png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close('all')
