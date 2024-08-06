# Simultaneous whole-liver water T1 and T2 mapping

Example source code to perform motion-resolved reconstruction, dual-echo water-fat separation and dictionary matching based on the publication:

*Jonathan Stelter, Kilian Weiss, Lisa Steinhelfer, Veronika Spieker, Elizabeth Huaroc Moquillaza, Weitong Zhang, Marcus R. Makowski, Julia A. Schnabel, Bernhard Kainz, Rickmer F. Braren and Dimitrios C. Karampinos; Simultaneous whole‐liver water T1 and T2 mapping with isotropic resolution during free‐breathing, NMR in Biomedicine, DOI: 10.1002/nbm.5216, https://doi.org/10.1002/nbm.5216*

## 🚀 Setup

### Requirements

- Julia 1.9 (system-wide installation is recommened)
- Anaconda/mamba environment with Python 3.10

### Installing

1. **Create a new mamba environment:**

    ```shell
    mamba env create --name t1t2mapping --file environment_nocuda.yml
    mamba activate t1t2mapping
    which python
    ```

2. **Open Julia and instantiate/precompile new Julia environment:**

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ENV["PYTHON"] = "/path/to/envs/t1t2mapping/bin/python"
    Pkg.build("PyCall")
    ```

3. **Run processing script directly from the shell:**

    Replace `n_threads` with the number of threads you wish to use.

    ```shell
    julia -t n_threads -i scripts/processing_demo.jl
    ```

## Data
Raw data for the T1 mapping phantom and the dictionary are stored at the [Google Drive folder](https://drive.google.com/drive/folders/1LsoFVZ_pk_-EnxV4lYPALHji9xkFb5e4?usp=sharing).

## Authors and acknowledgment
* Jonathan Stelter - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)

**Dual-echo water-fat separation: https://github.com/BMRRgroup/2echo-WaterFat-hmrGC**

**Single-voxel spectroscopy processing: https://github.com/BMRRgroup/alfonso**

## License
This project is licensed as given in the LICENSE file. However, used submodules / projects may be licensed differently. Please see the respective licenses.