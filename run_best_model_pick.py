import os
import time
from Best_model_pick import SimulationAnalyzer
import matplotlib.pyplot as plt
plt.style.use('default')


# Analysis parameters
base_dir = 'Simulation'
exp_wavelength_file = 'EXP_PATH'
exp_time_params = {'a1_pct': 20, 'tau1': 0.1, 'a2_pct': 80.0, 'tau2': 4, 'tau_avg': 4} 
wavelength_range = (650, 760)
N_best = 3

# Create the analyzer
analyzer = SimulationAnalyzer(
    base_dir=base_dir,
    exp_wavelength_file=exp_wavelength_file,
    exp_time_params=exp_time_params,
    wavelength_range=wavelength_range,
    N_best=N_best
)

# Run the analysis
print(f"Starting simulation analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 80)
analyzer.run_analysis()

# Export ALL simulation data for machine learning
print("\nExporting all simulation data for machine learning...")
all_data_path = analyzer.export_all_simulation_data()
print(f"All simulation data exported to: {all_data_path}")

# Generate additional visualizations
print("\nGenerating error space visualization...")
analyzer.plot_error_space()

print("\nGenerating visual model comparison...")
analyzer.visual_model_comparison(top_n=3)

print("-" * 80)
print(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")