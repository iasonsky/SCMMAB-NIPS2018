# R Scripts for PAG to ADMG Conversion

This directory contains the R scripts that implement the PAG to ADMG conversion algorithms from the AAAI17 paper.

## Files

### `full_admg_learning.R`
Main R script containing all the functions for:
- PAG generation and manipulation
- ADMG enumeration from PAGs
- DAG to PAG conversion with latent variables
- Edge possibility evaluation (intelligent and brute force methods)
- ADMG validation and conversion utilities

**Key Functions:**
- `make_pag_from_amat()` - Create PAG from adjacency matrix
- `pag2admg()` - Convert PAG to equivalent ADMGs (intelligent method)
- `admg2dag2pag()` - Convert ADMG to DAG with latent nodes, then to PAG
- `isADMG()` - Validate if matrix represents a valid ADMG
- `eval_possibilities_ggm()` - Evaluate edge possibilities for GGM format

### `AAAI17_exp_script.R`
Demonstration script that reproduces the AAAI17 experiments:
- Creates PAG1 and PAG2 from the AAAI17 paper
- Converts each PAG to its set of equivalent ADMGs
- Demonstrates the complete workflow

## Dependencies

**Required R packages:**
```r
install.packages(c("pcalg", "ggm"))
```

## Usage

### In R:
```r
source("full_admg_learning.R")
source("AAAI17_exp_script.R")
```

### In Python (via rpy2):
```python
from pathlib import Path
import rpy2.robjects as ro

# Load the R scripts
script_dir = Path(__file__).parent / "r_scripts"
ro.r(f'source("{script_dir}/full_admg_learning.R")')
```

## Integration with Python

These R scripts are used by the Python wrapper in `npsem/aaai17_admg_wrapper.py` to provide:
- PAG to ADMG conversion functionality
- Mathematical verification of equivalence
- Visualization and analysis capabilities

The Python wrapper handles the R integration seamlessly, allowing users to work primarily in Python while leveraging the robust R implementation for the core algorithms.
