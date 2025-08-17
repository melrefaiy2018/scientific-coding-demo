import glob
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import os


class SpectrumLoader:
    """Load experimental and simulated spectra with sorting and window normalization."""
    def __init__(self, wavelength_range=(650, 760)):
        self.wavelength_range = wavelength_range

    def load_exp_spectrum(self, npy_path):
        """
        Load the experimental spectrum from a NumPy array file, sort by wavelength,
        and normalize intensities by the peak within the analysis window.

        Parameters
        ----------
        npy_path : str
            Path to the experimental spectrum `.npy` file with shape (N, 2):
            column 0 = wavelength (nm), column 1 = intensity (a.u.).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with wavelengths (nm) and normalized intensities
            (unit peak within `self.wavelength_range`).
        """
        exp_wl_data_raw = np.load(npy_path)
        indices = np.argsort(exp_wl_data_raw[:, 0])
        exp_wl_data = exp_wl_data_raw[indices]
        mask = (exp_wl_data[:, 0] >= self.wavelength_range[0]) & (exp_wl_data[:, 0] <= self.wavelength_range[1])
        peak = np.max(exp_wl_data[mask, 1]) if np.any(mask) else np.max(exp_wl_data[:, 1])
        if peak != 0:
            exp_wl_data[:, 1] = exp_wl_data[:, 1] / peak
        return exp_wl_data

    def load_sim_pair(self, time_file, wl_file):
        """
        Load a pair of simulated datasets (time-resolved and wavelength-resolved),
        each from `.npy` files, and return sorted arrays.

        Parameters
        ----------
        time_file : str
            Path to the time-resolved simulation data `.npy` file (M, 2):
            column 0 = time (ns), column 1 = population/intensity (a.u.).
        wl_file : str
            Path to the wavelength-resolved simulation data `.npy` file (K, 2):
            column 0 = wavelength (nm), column 1 = intensity (a.u.).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (t_sim, y_sim, wl_sim, I_sim) where each array is sorted ascending
            by its x-axis (time or wavelength).
        """
        t_data = np.load(time_file)
        wl_data = np.load(wl_file)
        t_sorted = t_data[np.argsort(t_data[:, 0])]
        wl_sorted = wl_data[np.argsort(wl_data[:, 0])]
        return t_sorted[:, 0], t_sorted[:, 1], wl_sorted[:, 0], wl_sorted[:, 1]


class Normalizer:
    """Apply the exact normalization rules used in the current code."""
    def __init__(self, wavelength_range=(650, 760)):
        self.wavelength_range = wavelength_range

    def time_trace(self, y):
        """
        Normalize a time-resolved trace by its own maximum value.

        Parameters
        ----------
        y : np.ndarray
            Raw time-domain signal.

        Returns
        -------
        np.ndarray
            Normalized signal `y / max(y)` (returns `y` unchanged if max is 0).
        """
        m = np.max(y)
        return y / m if m != 0 else y

    def spectrum(self, wl, inten):
        """
        Normalize a spectrum by the peak intensity within the analysis window.

        Parameters
        ----------
        wl : np.ndarray
            Wavelengths (nm).
        inten : np.ndarray
            Raw intensities (a.u.).

        Returns
        -------
        np.ndarray
            Normalized intensities with unit peak in `self.wavelength_range`.
        """
        mask = (wl >= self.wavelength_range[0]) & (wl <= self.wavelength_range[1])
        peak = np.max(inten[mask]) if np.any(mask) else np.max(inten)
        peak = peak if peak != 0 else 1.0
        return inten / peak


class Metrics:
    """Compute the four error terms exactly as in the current script."""
    def __init__(self, exp_time_params, exp_wl_data, wavelength_range=(650, 760), roi=(710, 740)):
        self.exp_time_params = exp_time_params
        self.exp_wl_data = exp_wl_data
        self.wavelength_range = wavelength_range
        self.roi = roi

    @staticmethod
    def mse(y_true, y_pred):
        """
        Compute mean squared error between two arrays of equal length.

        Parameters
        ----------
        y_true : np.ndarray
            Reference values.
        y_pred : np.ndarray
            Predicted/compared values.

        Returns
        -------
        float
            Mean of squared residuals.
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def bi_exp_plot(t, a1_pct, tau1, a2_pct, tau2, tau_avg=None):
        """
        Bi-exponential decay evaluated for plotting using amplitude percentages.

        Parameters
        ----------
        t : np.ndarray
            Time (ns).
        a1_pct, a2_pct : float
            Amplitudes in percent (0–100).
        tau1, tau2 : float
            Decay lifetimes (ns).
        tau_avg : float, optional
            Unused here (present for interface consistency).

        Returns
        -------
        np.ndarray
            Decay values on `t`.
        """
        a1 = a1_pct / 100.0
        a2 = a2_pct / 100.0
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    @staticmethod
    def bi_exp_fit(t, a1_frac, tau1, a2_frac, tau2):
        """
        Bi-exponential decay used for fitting with fractional amplitudes.

        Parameters
        ----------
        t : np.ndarray
            Time (ns).
        a1_frac, a2_frac : float
            Amplitudes as fractions (0–1).
        tau1, tau2 : float
            Decay lifetimes (ns).

        Returns
        -------
        np.ndarray
            Decay values on `t`.
        """
        return a1_frac * np.exp(-t / tau1) + a2_frac * np.exp(-t / tau2)

    def time_mse(self, t_sim, pop_norm):
        """
        Compute MSE of the normalized simulated decay vs. the experimental
        bi-exponential reference on the simulation time grid.

        Parameters
        ----------
        t_sim : np.ndarray
            Simulation time points (ns).
        pop_norm : np.ndarray
            Normalized simulated decay.

        Returns
        -------
        float
            Mean squared error in the time domain.
        """
        exp_decay = self.bi_exp_plot(t_sim, **self.exp_time_params)
        return self.mse(exp_decay, pop_norm)

    def wl_mse(self, wl_sim, int_norm):
        """
        Compute MSE of the normalized simulated spectrum vs. the experimental
        spectrum, restricted to the analysis window (e.g., 650–760 nm).

        Parameters
        ----------
        wl_sim : np.ndarray
            Simulation wavelengths (nm).
        int_norm : np.ndarray
            Normalized simulated intensities.

        Returns
        -------
        float
            Mean squared error in the wavelength domain within the window.
        """
        exp_interp = np.interp(wl_sim, self.exp_wl_data[:, 0], self.exp_wl_data[:, 1])
        mask = (wl_sim >= self.wavelength_range[0]) & (wl_sim <= self.wavelength_range[1])
        return self.mse(exp_interp[mask], int_norm[mask])

    def roi_area_error(self, wl_sim, int_norm):
        """
        Compute the relative area error over the red-shoulder ROI (710–740 nm)
        using trapezoidal integration on normalized spectra.

        Parameters
        ----------
        wl_sim : np.ndarray
            Simulation wavelengths (nm).
        int_norm : np.ndarray
            Normalized simulated intensities.

        Returns
        -------
        float
            Relative area mismatch |A_sim - A_exp| / A_exp.
        """
        mask_sim = (wl_sim >= self.roi[0]) & (wl_sim <= self.roi[1])
        exp_wl = self.exp_wl_data[:, 0]
        exp_int = self.exp_wl_data[:, 1]
        mask_exp = (exp_wl >= self.roi[0]) & (exp_wl <= self.roi[1])
        sim_auc = np.trapz(int_norm[mask_sim], wl_sim[mask_sim])
        exp_auc = np.trapz(exp_int[mask_exp], exp_wl[mask_exp])
        return abs(sim_auc - exp_auc) / exp_auc if exp_auc != 0 else np.inf

    def tau_avg_error(self, t_sim, pop_norm):
        """
        Compute the relative error in average fluorescence lifetime (τ_avg).
        τ_avg is obtained by fitting a bi-exponential to the normalized
        simulated decay and comparing to the experimental τ_avg.

        Parameters
        ----------
        t_sim : np.ndarray
            Simulation time points (ns).
        pop_norm : np.ndarray
            Normalized simulated decay.

        Returns
        -------
        float
            Relative error in τ_avg (returns inf if fit fails).
        """
        try:
            popt, _ = curve_fit(
                self.bi_exp_fit,
                t_sim, pop_norm,
                p0=[0.5, 1.0, 0.5, 5.0],
                bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]),
                maxfev=5000
            )
            a1_fit, tau1_fit, a2_fit, tau2_fit = popt
            tau_avg_sim = (a1_fit * tau1_fit + a2_fit * tau2_fit) / (a1_fit + a2_fit)
            tau_avg_exp = self.exp_time_params['tau_avg']
            return abs(tau_avg_sim - tau_avg_exp) / tau_avg_exp
        except Exception:
            return np.inf


class Scorer:
    """Normalize-and-cap, then compute final score with geometric mean and hard penalty."""
    def __init__(self, caps=None, weights=None):
        self.caps = caps or {'time': 0.05, 'wl': 0.05, 'area': 0.20, 'tau': 0.10}
        self.weights = weights or {'time': 0.25, 'wl': 0.25, 'area': 0.25, 'tau': 0.25}

    def cap_errors(self, errs):
        """
        Normalize raw errors by their caps and clip to [0, 1].

        Parameters
        ----------
        errs : dict
            Raw errors with keys 'time', 'wl', 'area', 'tau'.

        Returns
        -------
        dict
            Normalized and capped errors with the same keys.
        """
        E_time = min(1.0, errs['time'] / self.caps['time'])
        E_wl   = min(1.0, errs['wl']   / self.caps['wl'])
        E_area = min(1.0, errs['area'] / self.caps['area'])
        E_tau  = min(1.0, errs['tau']  / self.caps['tau'])
        return {'time': E_time, 'wl': E_wl, 'area': E_area, 'tau': E_tau}

    def score(self, E):
        """
        Compute the final combined score from normalized errors.

        The score is defined as
            S = 0.5 * (w·E) + 0.5 * sqrt(E_time * E_wl),
        with a hard ×10 penalty if E_time > 0.9 or E_wl > 0.9.

        Parameters
        ----------
        E : dict
            Normalized errors with keys 'time', 'wl', 'area', 'tau'.

        Returns
        -------
        float
            Final combined score (lower is better).
        """
        weighted_avg = (self.weights['time'] * E['time'] +
                        self.weights['wl']   * E['wl'] +
                        self.weights['area'] * E['area'] +
                        self.weights['tau']  * E['tau'])
        balance = np.sqrt(E['time'] * E['wl'])
        S = 0.5 * weighted_avg + 0.5 * balance
        if E['time'] > 0.9 or E['wl'] > 0.9:
            S *= 10.0
        return S


class SimulationAnalyzer:
    """
    A class for analyzing spectral simulation results and comparing them to experimental data.
    """
    def __init__(self, base_dir, exp_wavelength_file, exp_time_params, wavelength_range=(650, 760), N_best=10):
        """
        Initialize the SimulationAnalyzer with the necessary parameters.
        (API unchanged; internally we now delegate to helper classes.)
        """
        self.base_dir = base_dir
        self.exp_wavelength_file = exp_wavelength_file
        self.exp_time_params = exp_time_params
        self.wavelength_range = wavelength_range
        self.N_best = N_best

        # Helper objects reflecting the current code behavior
        self.loader = SpectrumLoader(wavelength_range=self.wavelength_range)
        self.normalizer = Normalizer(wavelength_range=self.wavelength_range)
        self.exp_wl_data = None  # set in load_experimental_data()
        self.metrics = None      # set after exp spectrum is loaded
        self.scorer = Scorer(caps={'time':0.05,'wl':0.05,'area':0.20,'tau':0.10},
                             weights={'time':0.25,'wl':0.25,'area':0.25,'tau':0.25})

        # Results
        self.all_errors_sorted = None
        self.best_model = None

        # Output dirs
        self.output_dir = 'outputs'
        self.plots_dir = 'plots/best_models'
        self.npy_data_dir = 'npy_data'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.npy_data_dir, exist_ok=True)

        # Matplotlib config
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Set up matplotlib publication-quality settings."""
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8,
            'axes.linewidth': 1.2,
            'lines.linewidth': 1.5,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': False
        })
    
    def bi_exponential_decay_plot_func(self, t, a1_pct, tau1, a2_pct, tau2, tau_avg=None):
        """Bi-exponential decay function for plotting."""
        a1 = a1_pct / 100.0
        a2 = a2_pct / 100.0
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
    
    def bi_exponential_decay_fit_func(self, t, a1_frac, tau1, a2_frac, tau2):
        """Bi-exponential decay function for fitting."""
        return a1_frac * np.exp(-t / tau1) + a2_frac * np.exp(-t / tau2)
    
    def extract_params_from_name(self, run_name):
        """
        Extract parameters from the run name.
        
        Examples:
        Run_FL16_ISC5_coupling_100_CTenergy_14450 would extract:
        FL:16, ISC:5, coupling:100, CTenergy:14450
        """
        params = {}
        
        # Handle the FL and ISC parameters (uppercase letters followed by numbers)
        fl_isc_matches = re.findall(r'([A-Z]{2,})(\d+\.?\d*)', run_name)
        for key, value in fl_isc_matches:
            params[key] = float(value) if '.' in value else int(value)
        
        # Handle parameters with underscore format (like coupling_100)
        underscore_params = re.findall(r'([a-zA-Z]+)_(\d+\.?\d*)', run_name)
        for key, value in underscore_params:
            if key.lower() not in ['run']:  # Skip 'Run' prefix
                params[key] = float(value) if '.' in value else int(value)
        
        return params
    
    def extract_ratio_from_filename(self, filename):
        """Extract ratio information from a filename."""
        basename = os.path.basename(filename)
        ratio_match = re.search(r'(time|wavelength)_resolved_(.*?)_start_time', basename)
        return ratio_match.group(2) if ratio_match else "unknown"
    
    def compute_mse(self, y_true, y_pred):
        """Compute Mean Squared Error between two arrays."""
        return np.mean((y_true - y_pred) ** 2)
    
    def get_region_data(self, wl, inten, region=(710, 740)):
        """Extract data for a specific wavelength region."""
        mask = (wl >= region[0]) & (wl <= region[1])
        return wl[mask], inten[mask]
    
    def calculate_auc(self, wl, inten):
        """Calculate Area Under Curve using trapezoidal rule."""
        return np.trapz(inten, wl)
    
    def area_error(self, sim_auc, exp_auc):
        """Calculate relative error in area."""
        return abs(sim_auc - exp_auc) / exp_auc
    
    def load_experimental_data(self):
        """Load and process experimental wavelength data using SpectrumLoader."""
        try:
            self.exp_wl_data = self.loader.load_exp_spectrum(self.exp_wavelength_file)
            # Initialize metrics now that exp spectrum is available
            self.metrics = Metrics(self.exp_time_params, self.exp_wl_data, wavelength_range=self.wavelength_range, roi=(710,740))
            print("Successfully loaded experimental wavelength data.")
            return True
        except Exception as e:
            print(f"Error loading experimental wavelength file: {e}")
            return False
    
    def _save_npy_data(self, run_name, t_data, wl_data, ratio_str):
        """
        Save the numpy arrays for time-resolved and wavelength-resolved data.
        Now saves both simulation and experimental data for complete plotting.
        
        Parameters:
        -----------
        run_name : str
            Name of the simulation run
        t_data : numpy.ndarray
            Time-resolved data array
        wl_data : numpy.ndarray
            Wavelength-resolved data array
        ratio_str : str
            Ratio string extracted from filename
        """
        # Create subdirectory for this run
        run_subdir = os.path.join(self.npy_data_dir, run_name)
        os.makedirs(run_subdir, exist_ok=True)
        
        # Process time data
        t_sorted = t_data[np.argsort(t_data[:, 0])]
        t_sim = t_sorted[:, 0]
        pop_sim = self.normalizer.time_trace(t_sorted[:, 1])

        # Generate experimental time data for comparison
        exp_t = np.linspace(0, np.max(t_sim), len(t_sim))
        exp_pop = self.bi_exponential_decay_plot_func(exp_t, **self.exp_time_params)

        # Create combined time data array: [time, simulation, experimental]
        time_combined = np.column_stack([t_sim, pop_sim, np.interp(t_sim, exp_t, exp_pop)])

        # Process wavelength data
        wl_sorted = wl_data[np.argsort(wl_data[:, 0])]
        wl_sim = wl_sorted[:, 0]
        int_sim = self.normalizer.spectrum(wl_sim, wl_sorted[:, 1])
        # Generate experimental wavelength data for comparison
        exp_wl_interp = np.interp(wl_sim, self.exp_wl_data[:, 0], self.exp_wl_data[:, 1])
        # Create combined wavelength data array: [wavelength, simulation, experimental]
        wavelength_combined = np.column_stack([wl_sim, int_sim, exp_wl_interp])
        
        # Generate filenames
        time_filename = f"time_resolved_{ratio_str}.npy"
        wavelength_filename = f"wavelength_resolved_{ratio_str}.npy"
        
        # Full paths
        time_path = os.path.join(run_subdir, time_filename)
        wavelength_path = os.path.join(run_subdir, wavelength_filename)
        
        # Save the combined arrays
        np.save(time_path, time_combined)
        np.save(wavelength_path, wavelength_combined)
        
        # Create a readme file explaining the data format
        readme_path = os.path.join(run_subdir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write("Data Format Information\n")
            f.write("=" * 50 + "\n\n")
            f.write("Time-resolved data format (3 columns):\n")
            f.write("  Column 0: Time (ns)\n")
            f.write("  Column 1: Simulation (normalized)\n")
            f.write("  Column 2: Experimental (normalized)\n\n")
            f.write("Wavelength-resolved data format (3 columns):\n")
            f.write("  Column 0: Wavelength (nm)\n")
            f.write("  Column 1: Simulation (normalized)\n")
            f.write("  Column 2: Experimental (normalized)\n\n")
            f.write(f"Ratio: {ratio_str}\n")
            f.write(f"Run: {run_name}\n\n")
            f.write("Experimental Parameters Used:\n")
            for key, value in self.exp_time_params.items():
                f.write(f"  {key}: {value}\n")
        
        return time_path, wavelength_path
    
    def run_analysis(self):
        """Run the complete analysis workflow."""
        # Load experimental data
        if not self.load_experimental_data():
            print("Cannot proceed with analysis.")
            return False
        
        # Clean up existing plots
        self._cleanup_existing_plots()
        
        # Analyze all simulations
        print("\nAnalyzing simulation results...")
        if not self.analyze_simulations():
            return False
        
        # Display best model details
        print("\nShowing detailed information for the best model:")
        self.display_best_model()
        
        # Export data for the best model
        self.export_best_model_data()
        
        # Plot all top models
        print("\nGenerating plots for top models...")
        self.plot_best_models()
        
        print(f"\nAnalysis complete!")
        print(f"Results summary saved to '{self.output_dir}/best_model_errors.csv'")
        print(f"Best model detailed plot saved to '{self.output_dir}/best_model_detailed.png'")
        print(f"Plots for top {self.N_best} models saved to '{self.plots_dir}/' directory")
        print(f"Numpy data files saved to '{self.npy_data_dir}/' directory")
        return True
    
    def _cleanup_existing_plots(self):
        """Remove existing plot files to ensure clean output."""
        print("\nCleaning up existing plots...")
        plot_files = glob.glob(f"{self.plots_dir}/combined_plot_*.png")
        for file in plot_files:
            try:
                os.remove(file)
                print(f"Removed existing file: {file}")
            except Exception as e:
                print(f"Could not remove file {file}: {e}")
    
    def calculate_combined_score(self, time_mse, wavelength_mse, region_auc_err, tau_avg_err):
        """
        Calculate a combined score that prioritizes models performing well on both time 
        and wavelength dimensions simultaneously.
        
        Uses a geometric mean approach with penalties for poor performance in any metric.
        """
        # Normalize errors to similar scales
        max_acceptable_time_mse = 0.05
        max_acceptable_wl_mse = 0.05
        max_acceptable_auc_err = 0.2
        max_acceptable_tau_err = 0.1
        
        # Calculate normalized errors (capped at 1.0)
        norm_time_err = min(1.0, time_mse / max_acceptable_time_mse)
        norm_wl_err = min(1.0, wavelength_mse / max_acceptable_wl_mse)
        norm_auc_err = min(1.0, region_auc_err / max_acceptable_auc_err)
        norm_tau_err = min(1.0, tau_avg_err / max_acceptable_tau_err)
        
        # Weighted average of metrics
        w_time, w_wl, w_auc, w_tau = 0.25, 0.25, 0.25, 0.25
        weighted_avg = (w_time * norm_time_err + 
                        w_wl * norm_wl_err + 
                        w_auc * norm_auc_err + 
                        w_tau * norm_tau_err)
        
        # Calculate the geometric mean of time and wavelength errors
        # This heavily penalizes models that perform poorly in either dimension
        spectral_time_balance = np.sqrt(norm_time_err * norm_wl_err)
        
        # Final score combines weighted average with balance factor
        # Models must perform well on both curves to get a good score
        final_score = 0.5 * weighted_avg + 0.5 * spectral_time_balance
        
        # Add a massive penalty if either dimension is extremely poor
        if norm_time_err > 0.9 or norm_wl_err > 0.9:
            final_score *= 10.0
        
        return final_score

    def analyze_simulations(self):
        """
        Analyze all simulation results and identify the best models.
        Improved to ensure models simultaneously match both time and wavelength data.
        """
        run_dirs = glob.glob(os.path.join(self.base_dir, 'Run_*'))
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        
        if not run_dirs:
            print(f"No Run directories found in {self.base_dir}")
            return False
        
        all_errors = []
        
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            data_subdir = os.path.join(run_dir, 'Analysis_Data', 'Data')
            
            if not os.path.exists(data_subdir):
                print(f"Warning: Data directory not found for {run_name}, skipping...")
                continue
            
            time_files = sorted(glob.glob(os.path.join(data_subdir, 'time_resolved_*.npy')))
            wavelength_files = sorted(glob.glob(os.path.join(data_subdir, 'wavelength_resolved_*.npy')))
            
            if not time_files or not wavelength_files:
                print(f"Warning: No data files found for {run_name}, skipping...")
                continue
            
            best_score = np.inf
            best_pair = None
            best_data = {
                'time_mse': np.inf,
                'wavelength_mse': np.inf,
                'region_auc_err': np.inf,
                'tau_avg_err': np.inf
            }
            best_arrays = None  # Store the best data arrays
            
            # Now we'll examine every possible time-wavelength file pair
            for t_file in time_files:
                t_ratio = self.extract_ratio_from_filename(t_file)

                # Find matching wavelength file with same ratio
                matching_w_files = [w for w in wavelength_files if
                                self.extract_ratio_from_filename(w) == t_ratio]

                if not matching_w_files:
                    print(f"Warning: No matching wavelength file for {t_file}, skipping...")
                    continue

                for w_file in matching_w_files:
                    try:
                        # Load, sort, and normalize
                        t_sim, y_sim_raw, wl_sim, I_sim_raw = self.loader.load_sim_pair(t_file, w_file)
                        pop_sim = self.normalizer.time_trace(y_sim_raw)
                        int_sim = self.normalizer.spectrum(wl_sim, I_sim_raw)

                        # Errors
                        time_mse = self.metrics.time_mse(t_sim, pop_sim)
                        wavelength_mse = self.metrics.wl_mse(wl_sim, int_sim)
                        region_auc_err = self.metrics.roi_area_error(wl_sim, int_sim)
                        tau_avg_err = self.metrics.tau_avg_error(t_sim, pop_sim)

                        # Score
                        norm_errs = self.scorer.cap_errors({'time': time_mse, 'wl': wavelength_mse, 'area': region_auc_err, 'tau': tau_avg_err})
                        balanced_score = self.scorer.score(norm_errs)

                        # Store detailed error components for debugging
                        print(f"Model: {run_name}, Ratio: {t_ratio}")
                        print(f"  Time MSE: {time_mse:.6f}, WL MSE: {wavelength_mse:.6f}")
                        print(f"  Region AUC Error: {region_auc_err:.6f}, Tau Avg Error: {tau_avg_err:.6f}")
                        print(f"  Combined Score: {balanced_score:.6f}")

                        # Update best model if this one is better
                        if balanced_score < best_score:
                            best_score = balanced_score
                            best_pair = (t_file, w_file)
                            best_data = {
                                'time_mse': time_mse,
                                'wavelength_mse': wavelength_mse,
                                'region_auc_err': region_auc_err,
                                'tau_avg_err': tau_avg_err
                            }
                            best_arrays = (np.column_stack((t_sim, y_sim_raw)), np.column_stack((wl_sim, I_sim_raw)))
                    except Exception as e:
                        print(f"Error processing files {t_file}, {w_file}: {e}")
                        continue
            
            # Add to results if a best model was found
            if best_pair is not None and best_arrays is not None:
                error_tuple = (
                    run_name,
                    best_pair[0],  # t_file
                    best_pair[1],  # w_file
                    best_data['time_mse'],
                    best_data['wavelength_mse'],
                    best_data['region_auc_err'],
                    best_data['tau_avg_err'],
                    best_score
                )
                all_errors.append(error_tuple)
        
        if not all_errors:
            print("No valid simulation results found. Cannot proceed with analysis.")
            return False
        
        # Sort results by score
        self.all_errors_sorted = sorted(all_errors, key=lambda x: x[7])
        self.best_model = self.all_errors_sorted[0]
        
        # Save numpy data only for the top N_best models
        self._save_top_models_npy_data()
        
        # Create and export summary dataframe
        self._export_summary_dataframe()
        
        return True
    
    def _save_top_models_npy_data(self):
        """
        Save numpy data only for the top N_best models that will be used for plotting.
        """
        print(f"\nSaving numpy data for top {self.N_best} models...")
        
        for idx, (run_name, tfile, wfile, _, _, _, _, _) in enumerate(self.all_errors_sorted[:self.N_best]):
            try:
                # Load the data
                t_data = np.load(tfile)
                wl_data = np.load(wfile)
                
                # Sort the data
                t_data = t_data[np.argsort(t_data[:, 0])]
                wl_data = wl_data[np.argsort(wl_data[:, 0])]
                
                # Extract ratio from filename
                ratio_str = self.extract_ratio_from_filename(tfile)
                
                # Save the numpy data for this top model
                time_path, wavelength_path = self._save_npy_data(
                    run_name, t_data, wl_data, ratio_str
                )
                
                print(f"  Saved data for rank {idx+1}: {run_name} (ratio: {ratio_str})")
                
            except Exception as e:
                print(f"  Error saving data for {run_name}: {e}")
        
        print(f"Completed saving numpy data for top {self.N_best} models.")
    
    def plot_error_space(self):
        """
        Create a scatter plot showing how models perform in both time and wavelength dimensions.
        This helps diagnose whether models are balanced or excel in one dimension at the expense of the other.
        """
        if not self.all_errors_sorted:
            print("No models analyzed yet. Run analyze_simulations() first.")
            return None
        
        # Extract data for plotting
        time_errors = []
        wl_errors = []
        scores = []
        labels = []
        
        for run_name, _, _, time_mse, wl_mse, _, _, score in self.all_errors_sorted[:30]:  # Top 30 models
            time_errors.append(time_mse)
            wl_errors.append(wl_mse)
            scores.append(score)
            # Create shortened label
            short_name = run_name.replace('Run_', '').split('_')[0]  # Just take the first part
            labels.append(short_name)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(time_errors, wl_errors, c=scores, cmap='viridis_r', 
                            s=100, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Score (lower is better)')
        
        # Add labels for selected points
        for i, label in enumerate(labels[:10]):  # Only label top 10
            plt.annotate(label, (time_errors[i], wl_errors[i]), 
                        fontsize=8, ha='right', va='bottom')
        
        # Highlight the best model
        plt.scatter(time_errors[0], wl_errors[0], s=200, color='red', 
                marker='*', edgecolors='k', label='Best Model')
        
        # Add diagonal line representing perfect balance
        max_err = max(max(time_errors), max(wl_errors))
        plt.plot([0, max_err], [0, max_err], 'k--', alpha=0.3, label='Perfect Balance')
        
        # Set labels and title
        plt.xlabel('Time-Resolved Error (MSE)')
        plt.ylabel('Wavelength-Resolved Error (MSE)')
        plt.title('Model Performance in Time vs. Wavelength Domain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'error_space.png')
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        print(f"Error space visualization saved to {output_path}")

    def _export_summary_dataframe(self):
        """Create and export a DataFrame of the top models."""
        df_rows = []
        for i, (run_name, tfile, wfile, tmse, wmse, aucerr, tauerr, score) in enumerate(self.all_errors_sorted[:self.N_best]):
            # Extract model parameters
            params = self.extract_params_from_name(run_name)
            
            # Extract ratio from time file
            time_ratio_str = self.extract_ratio_from_filename(tfile)
            
            df_row = {
                'Rank': i+1, 
                'Run Name': run_name, 
                'Ratio': time_ratio_str,
                'Time File': os.path.basename(tfile),
                'Wavelength File': os.path.basename(wfile),
                'Time MSE': tmse, 
                'Wavelength MSE': wmse,
                'Region AUC Error': aucerr, 
                'Tau Avg Error': tauerr,
                'Score': score
            }
            
            # Add model parameters
            for key, value in params.items():
                df_row[key] = value
                    
            df_rows.append(df_row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(df_rows)
        csv_path = os.path.join(self.output_dir, 'best_model_errors.csv')
        df.to_csv(csv_path, index=False)
        print(f"Exported model summary to {csv_path}")
        
        # Display dataframe
        try:
            import ace_tools as tools
            tools.display_dataframe_to_user(name="Top Ranked Simulation Models", dataframe=df)
        except ImportError:
            print("ace_tools not available, displaying dataframe in console")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)
        
    def display_best_model(self):
        """
        Display detailed information about the best model and create detailed plot.
        """
        if not self.all_errors_sorted:
            print("No models analyzed yet. Run analyze_simulations() first.")
            return None
        
        # Get the best model
        run_name, tfile, wfile, time_mse, wavelength_mse, region_auc_err, tau_avg_err, score = self.best_model
        
        # Extract the specific file names and ratios
        tfile_name = os.path.basename(tfile)
        wfile_name = os.path.basename(wfile)
        time_ratio_str = self.extract_ratio_from_filename(tfile)
        
        print("\n" + "="*80)
        print(f"BEST MODEL: {run_name}")
        print("="*80)
        
        print(f"\nBest Time File: {tfile_name}")
        print(f"Ratio: {time_ratio_str}")
        
        print(f"\nBest Wavelength File: {wfile_name}")
        
        # Extract parameters from the run name
        params = self.extract_params_from_name(run_name)
        print("\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print("\nError Metrics:")
        print(f"  Time MSE: {time_mse:.6f}")
        print(f"  Wavelength MSE: {wavelength_mse:.6f}")
        print(f"  Region AUC Error: {region_auc_err:.6f}")
        print(f"  Tau Average Error: {tau_avg_err:.6f}")
        print(f"  Overall Score: {score:.6f}")
        
        # Load the data
        t_data = np.load(tfile)
        wl_data = np.load(wfile)
        
        t_data = t_data[np.argsort(t_data[:, 0])]
        wl_data = wl_data[np.argsort(wl_data[:, 0])]
        
        t = t_data[:, 0]
        pop = self.normalizer.time_trace(t_data[:, 1])
        wl = wl_data[:, 0]
        inten = self.normalizer.spectrum(wl, wl_data[:, 1])
        
        # Fit the time data
        fit_params = None
        try:
            popt, _ = curve_fit(
                self.bi_exponential_decay_fit_func,
                t, pop,
                p0=[0.5, 1.0, 0.5, 5.0],
                bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]),
                maxfev=5000
            )
            a1_fit, tau1_fit, a2_fit, tau2_fit = popt
            a1_pct = a1_fit * 100
            a2_pct = a2_fit * 100
            tau_avg = (a1_fit * tau1_fit + a2_fit * tau2_fit) / (a1_fit + a2_fit)
            fit_params = {'a1_pct': a1_pct, 'tau1': tau1_fit, 'a2_pct': a2_pct, 'tau2': tau2_fit, 'tau_avg': tau_avg}
            
            print("\nFitted Parameters:")
            print(f"  a1: {a1_pct:.1f}% with tau1: {tau1_fit:.2f} ns")
            print(f"  a2: {a2_pct:.1f}% with tau2: {tau2_fit:.2f} ns")
            print(f"  tau_avg: {tau_avg:.2f} ns")
            print(f"  Experimental tau_avg: {self.exp_time_params['tau_avg']:.2f} ns")
        except Exception as e:
            print(f"\nFitting failed: {e}")
        
        # Create and save detailed plot
        self._create_detailed_plot(run_name, t, pop, wl, inten, time_ratio_str, fit_params)
        
        # Create comparison table
        if fit_params:
            self._print_comparison_table(fit_params)
        
        return self.best_model
    
    def _create_detailed_plot(self, run_name, t, pop, wl, inten, ratio_str, fit_params=None):
        """Create and save a detailed plot for the best model."""
        # Generate a detailed plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time-resolved plot
        ax1.plot(t, pop, 'b-', linewidth=2, label='Simulation')
        exp_t = np.linspace(0, np.max(t), 500)
        exp_pop = self.bi_exponential_decay_plot_func(exp_t, **self.exp_time_params)
        ax1.plot(exp_t, exp_pop, 'k--', linewidth=2, label='Experimental')
        
        if fit_params:
            # Add fitted curve
            fit_curve = self.bi_exponential_decay_plot_func(
                t, 
                fit_params['a1_pct'], 
                fit_params['tau1'], 
                fit_params['a2_pct'], 
                fit_params['tau2']
            )
            ax1.plot(t, fit_curve, 'r-', linewidth=1.5, label='Bi-exp fit')
            
            # Add parameters text box
            text_box = (
                f"a₁ = {fit_params['a1_pct']:.1f}%, τ₁ = {fit_params['tau1']:.2f} ns\n"
                f"a₂ = {fit_params['a2_pct']:.1f}%, τ₂ = {fit_params['tau2']:.2f} ns\n"
                f"τavg = {fit_params['tau_avg']:.2f} ns"
            )
            
            ax1.text(0.55, 0.6, text_box, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
        
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Normalized Population', fontsize=12)
        ax1.set_title('Time-Resolved Comparison', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.2)
        
        # Wavelength-resolved plot
        ax2.plot(wl, inten, 'b-', linewidth=2, label='Simulation')
        exp_wl = self.exp_wl_data[:, 0]
        exp_inten = self.exp_wl_data[:, 1]
        ax2.plot(exp_wl, exp_inten, 'k--', linewidth=2, label='Experimental')
        
        # Highlight the region of interest
        region_min, region_max = 710, 740
        ax2.axvspan(region_min, region_max, color='lightgray', alpha=0.3, label='Region of Interest')
        
        ax2.set_xlim(self.wavelength_range)
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Normalized Intensity', fontsize=12)
        ax2.set_title('Wavelength-Resolved Comparison', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)
        
        # Set the title with model name and ratio
        plt.suptitle(f'Best Model: {run_name}\nRatio: {ratio_str}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'best_model_detailed.png')
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        print(f"\nDetailed plot saved to {output_path}")
    
    def _print_comparison_table(self, fit_params):
        """Print a comparison table between simulated and experimental values."""
        print("\nComparison with Experimental Values:")
        print("{:<20} {:<15} {:<15} {:<15}".format("Parameter", "Simulation", "Experimental", "Relative Error (%)"))
        print("-" * 70)
        
        try:
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                "tau1 (ns)", 
                fit_params['tau1'], 
                self.exp_time_params['tau1'],
                abs(fit_params['tau1'] - self.exp_time_params['tau1']) / self.exp_time_params['tau1'] * 100
            ))
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                "tau2 (ns)", 
                fit_params['tau2'], 
                self.exp_time_params['tau2'],
                abs(fit_params['tau2'] - self.exp_time_params['tau2']) / self.exp_time_params['tau2'] * 100
            ))
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                "tau_avg (ns)", 
                fit_params['tau_avg'], 
                self.exp_time_params['tau_avg'],
                abs(fit_params['tau_avg'] - self.exp_time_params['tau_avg']) / self.exp_time_params['tau_avg'] * 100
            ))
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                "a1 (%)", 
                fit_params['a1_pct'], 
                self.exp_time_params['a1_pct'],
                abs(fit_params['a1_pct'] - self.exp_time_params['a1_pct']) / self.exp_time_params['a1_pct'] * 100
            ))
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                "a2 (%)", 
                fit_params['a2_pct'], 
                self.exp_time_params['a2_pct'],
                abs(fit_params['a2_pct'] - self.exp_time_params['a2_pct']) / self.exp_time_params['a2_pct'] * 100
            ))
        except:
            print("Could not calculate comparison metrics.")
    
    def export_best_model_data(self):
        """Export data files for the best model."""
        if not self.best_model:
            print("No best model to export.")
            return
        
        run_name, tfile, wfile, time_mse, wavelength_mse, region_auc_err, tau_avg_err, score = self.best_model
        
        # Extract ratio from filename
        time_ratio_str = self.extract_ratio_from_filename(tfile)
        
        print(f"\nExporting best model data...")
        print(f"Model: {run_name}")
        print(f"Ratio: {time_ratio_str}")
        
        # Load and process data
        t_data = np.load(tfile)
        wl_data = np.load(wfile)
        
        t_data = t_data[np.argsort(t_data[:, 0])]
        wl_data = wl_data[np.argsort(wl_data[:, 0])]
        
        # Export time-resolved data
        time_df = pd.DataFrame({
            'Time (ns)': t_data[:, 0],
            'Simulation': t_data[:, 1] / np.max(t_data[:, 1])
        })
        
        # Add experimental data
        exp_t = np.linspace(0, np.max(t_data[:, 0]), len(t_data))
        exp_pop = self.bi_exponential_decay_plot_func(exp_t, **self.exp_time_params)
        time_df['Experimental'] = np.interp(time_df['Time (ns)'], exp_t, exp_pop)
        
        # Export wavelength-resolved data
        wl = wl_data[:, 0]
        inten_raw = wl_data[:, 1]
        inten_norm = self.normalizer.spectrum(wl, inten_raw)
        wavelength_df = pd.DataFrame({
            'Wavelength (nm)': wl,
            'Simulation': inten_norm
        })
        
        # Save to CSV
        time_path = os.path.join(self.output_dir, 'best_model_time_data.csv')
        wavelength_path = os.path.join(self.output_dir, 'best_model_wavelength_data.csv')
        info_path = os.path.join(self.output_dir, 'best_model_info.txt')
        
        time_df.to_csv(time_path, index=False)
        wavelength_df.to_csv(wavelength_path, index=False)
        
        # Also save the ratio information to a text file
        with open(info_path, 'w') as f:
            f.write(f"Best Model: {run_name}\n")
            f.write(f"Ratio: {time_ratio_str}\n")
            f.write(f"Time File: {os.path.basename(tfile)}\n")
            f.write(f"Wavelength File: {os.path.basename(wfile)}\n")
            f.write(f"Score: {score:.6f}\n")
        
        print(f"\nExported time-resolved data to {time_path}")
        print(f"Exported wavelength-resolved data to {wavelength_path}")
        print(f"Exported model information to {info_path}")

    def plot_best_models(self):
        """
        Generate plots for all top-ranked models showing time-resolved and wavelength-resolved data.
        """
        for idx, (run_name, tfile, wfile, time_mse, wavelength_mse, region_auc_err, tau_avg_err, score) in enumerate(self.all_errors_sorted[:self.N_best]):
            # Create unique filename with timestamp to avoid caching issues
            timestamp = int(time.time())
            output_filename = f'{self.plots_dir}/combined_plot_{idx+1}_{run_name}_{timestamp}.png'
            
            # Load and process data
            t_data = np.load(tfile)
            wl_data = np.load(wfile)
            
            t_data = t_data[np.argsort(t_data[:, 0])]
            wl_data = wl_data[np.argsort(wl_data[:, 0])]
            
            t = t_data[:, 0]
            pop = self.normalizer.time_trace(t_data[:, 1])
            wl = wl_data[:, 0]
            inten = self.normalizer.spectrum(wl, wl_data[:, 1])
            
            # Extract ratio from time file
            time_ratio_str = self.extract_ratio_from_filename(tfile)
            
            # Create experimental data for comparison
            exp_t = np.linspace(0, np.max(t), 500)
            exp_pop = self.bi_exponential_decay_plot_func(exp_t, **self.exp_time_params)
            exp_wl = self.exp_wl_data[:, 0]
            exp_inten = self.exp_wl_data[:, 1]
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time-resolved plot
            ax1.plot(t, pop, label='Simulation')
            ax1.plot(exp_t, exp_pop, 'k--', label='Experimental')
            
            try:
                # Fit the data
                popt, _ = curve_fit(
                    self.bi_exponential_decay_fit_func,
                    t, pop,
                    p0=[0.5, 1.0, 0.5, 5.0],
                    bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]),
                    maxfev=5000
                )
                a1_fit, tau1_fit, a2_fit, tau2_fit = popt
                a1_pct = a1_fit * 100
                a2_pct = a2_fit * 100
                tau_avg = (a1_fit * tau1_fit + a2_fit * tau2_fit) / (a1_fit + a2_fit)
                
                # Add fitted curve
                fit_curve = self.bi_exponential_decay_fit_func(t, *popt)
                ax1.plot(t, fit_curve, 'r--', label='Bi-exp fit')
                
                # Add parameters text box
                text_box = (
                    f"a₁ = {a1_pct:.1f}%, τ₁ = {tau1_fit:.2f} ns\n"
                    f"a₂ = {a2_pct:.1f}%, τ₂ = {tau2_fit:.2f} ns\n"
                    f"τavg = {tau_avg:.2f} ns"
                )
                
                ax1.text(0.55, 0.6, text_box, transform=ax1.transAxes, fontsize=9,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
            except Exception as e:
                print(f"Fitting failed for {run_name}: {e}")
            
            ax1.set_xlabel('Time (ns)')
            ax1.set_ylabel('Normalized Population')
            ax1.set_title('Time-Resolved')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.2)
            
            # Wavelength-resolved plot
            ax2.plot(wl, inten, label='Simulation')
            ax2.plot(exp_wl, exp_inten, 'k--', label='Experimental')
            ax2.set_xlim(self.wavelength_range)
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Normalized Intensity')
            ax2.set_title('Wavelength-Resolved')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.2)
            
            # Match the format exactly as in the detailed plot
            plt.suptitle(f'Model: {run_name}\nRatio: {time_ratio_str}\nBalanced Score: {score:.4f}', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            
            try:
                plt.savefig(output_filename, dpi=300)
                print(f"Saved plot {idx+1}/{self.N_best} to {output_filename}")
            except Exception as e:
                print(f"Error saving plot to {output_filename}: {e}")
            
            plt.show()
            
    def visual_model_comparison(self, top_n=5):
        """
        Create a comprehensive visual comparison of the top N models.
        Shows both time-resolved and wavelength-resolved data side by side for better comparison.
        """
        if not self.all_errors_sorted or top_n <= 0:
            print("No models analyzed yet or invalid top_n value.")
            return
        
        # Limit to available models
        top_n = min(top_n, len(self.all_errors_sorted))
        
        # Set up the figure - one row per model, two columns (time and wavelength)
        fig, axes = plt.subplots(top_n, 2, figsize=(15, 5*top_n))
        
        # If only one model, convert axes to 2D array
        if top_n == 1:
            axes = np.array([axes])
        
        # Process each model
        for i, (run_name, tfile, wfile, time_mse, wl_mse, auc_err, tau_err, score) in enumerate(self.all_errors_sorted[:top_n]):
            # Get ratio information
            ratio_str = self.extract_ratio_from_filename(tfile)
            
            # Load time data
            t_data = np.load(tfile)
            t_data = t_data[np.argsort(t_data[:, 0])]
            t = t_data[:, 0]
            pop = self.normalizer.time_trace(t_data[:, 1])

            # Load wavelength data
            wl_data = np.load(wfile)
            wl_data = wl_data[np.argsort(wl_data[:, 0])]
            wl = wl_data[:, 0]
            inten = self.normalizer.spectrum(wl, wl_data[:, 1])
            
            # Plot time-resolved data
            ax_time = axes[i, 0]
            ax_time.plot(t, pop, 'b-', label='Simulation')
            
            # Add experimental data
            exp_t = np.linspace(0, np.max(t), 500)
            exp_pop = self.bi_exponential_decay_plot_func(exp_t, **self.exp_time_params)
            ax_time.plot(exp_t, exp_pop, 'k--', label='Experimental')
            
            # Try to fit and plot
            try:
                popt, _ = curve_fit(
                    self.bi_exponential_decay_fit_func,
                    t, pop,
                    p0=[0.5, 1.0, 0.5, 5.0],
                    bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]),
                    maxfev=5000
                )
                a1_fit, tau1_fit, a2_fit, tau2_fit = popt
                fit_curve = self.bi_exponential_decay_fit_func(t, *popt)
                ax_time.plot(t, fit_curve, 'r--', label='Fit')
                
                # Calculate average lifetime
                tau_avg = (a1_fit * tau1_fit + a2_fit * tau2_fit) / (a1_fit + a2_fit)
                
                # Add text with parameters
                param_text = f"a₁={a1_fit*100:.1f}%, τ₁={tau1_fit:.2f}ns\n" \
                            f"a₂={a2_fit*100:.1f}%, τ₂={tau2_fit:.2f}ns\n" \
                            f"τavg={tau_avg:.2f}ns"
                ax_time.text(0.05, 0.05, param_text, transform=ax_time.transAxes,
                            fontsize=8, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            except:
                pass
            
            # Set up time plot
            ax_time.set_xlabel('Time (ns)')
            ax_time.set_ylabel('Normalized Population')
            ax_time.set_title(f'Time-Resolved (MSE: {time_mse:.6f})')
            ax_time.legend(loc='upper right', fontsize=8)
            ax_time.grid(alpha=0.3)
            
            # Plot wavelength-resolved data
            ax_wl = axes[i, 1]
            ax_wl.plot(wl, inten, 'b-', label='Simulation')
            
            # Add experimental data
            exp_wl = self.exp_wl_data[:, 0]
            exp_inten = self.exp_wl_data[:, 1]
            ax_wl.plot(exp_wl, exp_inten, 'k--', label='Experimental')
            
            # Highlight region of interest
            region_min, region_max = 710, 740
            ax_wl.axvspan(region_min, region_max, color='lightgray', alpha=0.3, 
                        label='ROI')
            
            # Set up wavelength plot
            ax_wl.set_xlabel('Wavelength (nm)')
            ax_wl.set_ylabel('Normalized Intensity')
            ax_wl.set_title(f'Wavelength-Resolved (MSE: {wl_mse:.6f})')
            ax_wl.set_xlim(self.wavelength_range)
            ax_wl.legend(loc='upper right', fontsize=8)
            ax_wl.grid(alpha=0.3)
            
            # Add model info as row title
            row_title = f"Rank {i+1}: {run_name}\nRatio: {ratio_str}, Score: {score:.6f}"
            fig.text(0.01, 0.5 - i/top_n*0.9, row_title, fontsize=10, 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout(rect=[0.03, 0, 1, 0.98])
        plt.subplots_adjust(hspace=0.4)
        
        # Add overall title
        plt.suptitle(f'Top {top_n} Model Comparison', fontsize=16, y=0.99)
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'top_{top_n}_model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visual comparison of top {top_n} models saved to {output_path}")

    # New function to add to SimulationAnalyzer
    def export_all_simulation_data(self, output_file=None):
        """
        Export all simulation data to a CSV file suitable for machine learning.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the CSV file. If None, uses default path.
        
        Returns:
        --------
        str
            Path where the data was saved.
        """
        if not self.all_errors_sorted:
            print("No models analyzed yet. Run analyze_simulations() first.")
            return None
        
        # Default output path
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'all_simulation_data.csv')
        
        # Create DataFrame rows
        df_rows = []
        for run_name, tfile, wfile, tmse, wmse, aucerr, tauerr, score in self.all_errors_sorted:
            # Extract model parameters from run name
            params = self.extract_params_from_name(run_name)
            
            # Extract ratio from filename
            ratio_str = self.extract_ratio_from_filename(tfile)
            
            # Create row with all metrics and parameters
            row = {
                'Run_Name': run_name,
                'Ratio': ratio_str,
                'Time_File': os.path.basename(tfile),
                'Wavelength_File': os.path.basename(wfile),
                'Time_MSE': tmse,
                'Wavelength_MSE': wmse,
                'Region_AUC_Error': aucerr,
                'Tau_Avg_Error': tauerr,
                'Combined_Score': score
            }
            
            # Add extracted parameters
            for key, value in params.items():
                row[key] = value
            
            df_rows.append(row)
        
        # Create DataFrame and save to CSV
        import pandas as pd
        df = pd.DataFrame(df_rows)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Exported data for {len(df_rows)} simulations to {output_file}")
        return output_file