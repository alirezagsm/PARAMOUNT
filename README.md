<h1 align="left">
    <div>
    PARAMOUNT: Parallel Modal Analysis of Large Datasets
</h1>

<p align="left">
PARAMOUNT is a Python package developed at the University of Twente for performing Proper Orthogonal Decomposition (POD) and Dynamic Mode Decomposition (DMD) on large numerical and experimental datasets. Leverage the power of parallel processing to analyze massive amounts of data efficiently. A brief video introduction into the theory and methodology is presented [here](https://youtu.be/uz0q_TKrC84).
</p>

# Features

- **Distributed Processing:** Leverages multi-core architectures for significant speedups in data handling.
- **Proper Orthogonal Decomposition (POD):**

  - Perform distributed computation of POD to extract dominant spatial patterns.
  - Example Application: PARAMOUNT has been used to analyze large datasets from numerical simulations of a swirl burner. For details, see:

  > Alireza Ghasemi, et al. "Combustion Dynamics Analysis of a Pressurized Airblast Swirl Burner Using Proper Orthogonal Decomposition." *International Journal of Spray and Combustion Dynamics*, 2023. [DOI:10.1177/17568277231207252](http://journals.sagepub.com/doi/10.1177/17568277231207252)
  >
- **Dynamic Mode Decomposition (DMD):**

  - Compute DMD modes, eigenvalues, and generate future state predictions.
  - Includes support for multi-resolution DMD (MRDMD) to analyze data across various temporal scales.
- **Visualization:** Easily visualize POD/DMD modes and coefficients.

# Using PARAMOUNT

1. **Installation:** Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation:**
   - Data processing functionality of PARAMOUNT is particularly suited for CFD data analysis but can be adapted for other data formats.
   - Specify the variables of interest and convert the data into Parquet datasets for optimized performance and storage.
   - Refer to `csv_example` for a practical guide on using this feature.
3. **Analysis (POD/DMD/MRDMD):**
   - For POD: Utilize the `POD` class to compute the Singular Value Decomposition (SVD) from the prepared Parquet datasets. Results (U, S, V) will be stored. See `svd_example` for detailed usage.
   - For DMD: Use the `DMD` class, which builds upon the `POD` infrastructure, to perform Dynamic Mode Decomposition, compute eigenvalues, and analyze system dynamics. Refer to `dmd_example` for implementation details.
   - For Multi-Resolution DMD (MRDMD): Leverage the `MRDMD` functionality to analyze data across multiple DMD scales. See `mrdmd_example` for a step-by-step guide.
4. **Visualization:**
   - PARAMOUNT provides several visualization tools based on Matplotlib. Refer to the example files for guidance on how to use them.
   - 3D data can be interactively visualized with Plotly. See `viz_example` for instructions on how to utilize this feature.

**Proposed Project Folder Structure:**`</br>`
This is a sample project structure for using the PARAMOUNT library to perform POD and DMD analysis.

```
Project
├── myproject.py
├── .data
│   ├── variable_1
│   │   └── .parquet
│   ├── variable_2
│   │   └── .parquet
│   ├── x.pkl
│   └── y.pkl
├── .usv
│   ├── variable_1
│   │   ├── s.pkl
│   │   ├── u/.parquet
│   │   └── v/.parquet
│   ├── variable_2
│   │   ├── s.pkl
│   │   ├── u/.parquet
│   │   └── v/.parquet
│   ├── x.pkl
│   └── y.pkl
├── .dmd
│   ├── variable_1
│   │   ├── Atilde.pkl
│   │   ├── b.pkl
│   │   ├── lambda.pkl
│   │   ├── modes_imag/.parquet
│   │   ├── modes_real/.parquet
│   │   └── prediction/.parquet
│   ├── variable_2
│   │   ├── Atilde.pkl
│   │   ├── b.pkl
│   │   ├── lambda.pkl
│   │   ├── modes_imag/.parquet
│   │   ├── modes_real/.parquet
│   │   └── prediction/.parquet
├── .mrdmd
│   ├── variable_1/levels
│   │   ├── level_0/.parquet
│   │   ├── level_1/.parquet
│   │   └── ...
│   └── variable_2/levels
│       ├── level_0/.parquet
│       ├── level_1/.parquet
│       └── ...
├── .viz
│   ├── variable_1/results.png
│   └── variable_2/results.png
└── src
   ├── PARAMOUNT_BASE.py
   ├── PARAMOUNT_POD.py
   ├── PARAMOUNT_DMD.py
   └── utils.py
```

# Author and Acknowledgements

This package is developed by [Alireza Ghasemi](https://www.linkedin.com/in/alirezaaghasemi/) at University of Twente under the [MAGISTER](https://www.magister-itn.eu/) project. This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No. 766264.
