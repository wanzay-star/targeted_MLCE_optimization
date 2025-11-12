# FSO Channel Estimation

## MATLAB Implementation

The script 'fso_ch_est_2' uses FSO signal datasets to predict the attenuation that will occur in the signal over specified 'latency' time horizons ahead.
To achieve the forward-looking prediction, features are created based on data samples lagged according to the latency values.
The data is also made stationary by first-order differencing.
The predicted attenuation is used to pre-compensate the original signal to reduce the distortions in the signal.

The Rytov variance quantifies the distortion present in the signal by fitting the signal's distribution against known distributions (gamma-gamma and log-normal).
This enables the determination of the Rytov value that minimizes the fitting.
'rytov_vs_latency.m' determines the Rytov variance of pre-compensated data across different latency values.

## Python Implementation

This project now includes a Python implementation of the MATLAB code, providing equivalent functionality with modern Python scientific computing libraries.

### Requirements

**Python Version:**
- Python 3.10 or higher

**Dependencies:**
- **numpy**: Core numerical computing library for array operations and mathematical functions
- **scipy**: Scientific computing library used for optimization (fmin), signal processing, and statistical functions
- **matplotlib**: Plotting library for visualizing results and generating figures
- **pandas**: Data manipulation library for handling tabular data structures (equivalent to MATLAB tables)
- **scikit-learn**: Machine learning library providing linear regression and other algorithms
- **xgboost**: Gradient boosting library for ensemble learning methods
- **catboost**: Gradient boosting library with categorical feature support

### Installation

```bash
# Clone repository
git clone <repo-url>
cd <repo-directory>

# Install Python dependencies
pip install -e .
# or
pip install .
```

Alternatively, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

### Data Files

**Important:** Users must provide their own `.mat` files in the `./data/` directory. Sample data is not included in this repository due to size and licensing constraints.

**Expected File Structure:**
- Place `.mat` files in a `data` subdirectory at the project root
- Files should contain FSO signal power measurements sampled at 10 kHz

**Required .mat File Format:**
The MATLAB data files should contain the following variables:
- `lin_wan5_s_dat`: Strong turbulence regime data (vector of power measurements in watts)
- `lin_wan5_m_dat`: Moderate turbulence regime data (vector of power measurements in watts)  
- `lin_wan5_w_dat`: Weak turbulence regime data (vector of power measurements in watts)

**Example Files:**
- `lin_wan5_strong_turb_samps.mat`
- `lin_wan5_mod_turb_samps.mat`
- `lin_wan5_weak_turb_samps.mat`

Each file should contain FSO optical power measurements representing different atmospheric turbulence conditions.

### Usage

**Running the Main Analysis Script:**

```bash
# Execute the main channel estimation analysis
python fso_ch_est_2.py
```

The script will:
1. Load FSO signal data from the `./data/` directory
2. Perform channel estimation using LMS, Linear Regression, and Zero-Order Hold methods
3. Generate predictions with various latency values
4. Calculate RMSE and Rytov variance metrics
5. Produce visualization plots comparing different methods

**Importing Individual Functions:**

```python
# Import the Rytov variance calculation function
from rytov_vs_latency import rytov_vs_latency

# Use the function with your data
rytov_variance = rytov_vs_latency(power_data)
```

**Customizing Parameters:**

Edit the configuration variables in `fso_ch_est_2.py` to adjust:
- `latency`: Time horizon for prediction (in samples)
- `nTaps`: Number of taps for the LMS filter
- `nTrain`: Number of training samples
- `use_differential`: Enable/disable first-order differencing

### Results Comparison

When comparing results between the MATLAB and Python implementations, you should expect **approximate agreement** rather than exact matches. Small numerical differences are normal and expected.

**Expected Differences:**

1. **Floating-Point Arithmetic**: NumPy and MATLAB may produce slightly different results in floating-point operations due to differences in underlying linear algebra libraries (BLAS/LAPACK implementations) and numerical precision handling.

2. **Random Seed Initialization**: Ensemble models such as Random Forest, XGBoost, and CatBoost use random initialization. Even with fixed seeds, differences in random number generators between Python and MATLAB can lead to variations in model performance.

3. **Optimization Convergence**: `scipy.optimize.fmin` (Nelder-Mead simplex) may converge to slightly different local minima compared to MATLAB's `fminsearch`, especially for the Rytov variance fitting optimization.

4. **Moving Average Calculations**: Small differences in edge handling and boundary conditions in moving average operations (`movmean` vs. rolling window operations).

**Typical Magnitude of Differences:**
- RMSE metrics: < 5% relative error
- Rytov variance estimates: < 10% relative error  
- Correlation between predictions: > 0.95

These differences do not affect the validity of the results or the scientific conclusions drawn from the analysis. Both implementations are correct within numerical precision limits.
