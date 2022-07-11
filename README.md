# PARAMOUNT: parallel modal analysis of large datasets

PARAMOUNT is a python package developed at University of Twente to perform modal analysis of large numerical and experimental datasets. Brief video introduction into the theory and methodology is presented [here](https://youtu.be/uz0q_TKrC84).

# Features

- Distributed processing of data on local machines or clusters using Dask Distributed
- Reading CSV files in glob format from specified folders
- Extracting relevant columns from CSV files and writing Parquet database for each specified variable
- Distributed computation of Proper Orthogonal Decomposition (POD)
- Writing U, S and V matrices into Parquet database for further analysis
- Visualizing POD modes and coefficients using pyplot

# Using PARAMOUNT

Make sure to install the dependencies by running `pip install -r requirements.txt`

Refer to csv_example to see how to use PARAMOUNT to read CSV files, write the variables of interest into Parquet datasets and inspect the final datasets.

Refer to svd_example to see how to read Parquet datasets, compute the Singular Value Decomposition, and store the results in Parquet format.

To visualize the results you can simply read the U, S and V parquet files and your plotting tool of choice. Examples are provided in viz_example.

Proposed project folder structure:
```
Project
├── myproject.py
├── .data
│   ├── variable 1
│   │   └── .parquet
│   ├── variable 2
│   │   └── .parquet
│   ├── x.pkl
│   └── y.pkl
├── .usv
│   ├── variable 1
│   │   ├── s.pkl
│   │   ├── u
│   │   │   └── .parquet
│   │   └── v
│   │       └── .parquet
│   ├── variable 2
│   │   ├── s.pkl
│   │   ├── u
│   │   │   └── .parquet
│   │   └── v
│   │       └── .parquet
│   ├── x.pkl
│   └── y.pkl
├── .viz
│   ├── variable 1
│   │   └── results.png   
│   └── variable 2
│       └── results.png         
└── src
    ├── PARAMOUNT.py
    └── utils.py
```


# Author and Acknowledgements

This package is developed by [Alireza Ghasemi](alireza.ghasemi@utwente.nl) at University of Twente under the [MAGISTER](https://www.magister-itn.eu/) project. This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No. 766264.