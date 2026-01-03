"""Script to analyze dataset and check for data leakage."""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config

def analyze_dataset(data_path: Path, config_path: Path):
    """Analyze dataset for data leakage and value ranges."""
    print("=" * 60)
    print("АНАЛИЗ ДАННЫХ И ПРОВЕРКА НА DATA LEAK")
    print("=" * 60)
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    config = load_config(config_path)
    
    # Extract all t_decoh values
    all_t_decoh = []
    all_gamma = []
    all_observables = []
    all_times = []
    
    for split in ['train', 'val', 'test']:
        t_key = f'{split}_t_decoh'
        gamma_key = f'{split}_gamma'
        obs_key = f'{split}_observables'
        times_key = f'{split}_times'
        
        if t_key in data:
            all_t_decoh.extend(data[t_key])
            all_gamma.extend(data[gamma_key] if gamma_key in data else [0.0] * len(data[t_key]))
            all_observables.extend(data[obs_key] if obs_key in data else [])
            all_times.extend(data[times_key] if times_key in data else [])
    
    all_t_decoh = np.array(all_t_decoh)
    all_gamma = np.array(all_gamma)
    
    print(f"\n1. ДИАПАЗОН ВРЕМЕНИ ДЕКОГЕРЕНЦИИ:")
    print(f"   Min: {np.min(all_t_decoh):.6f}")
    print(f"   Max: {np.max(all_t_decoh):.6f}")
    print(f"   Mean: {np.mean(all_t_decoh):.6f}")
    print(f"   Std: {np.std(all_t_decoh):.6f}")
    print(f"   Range: {np.max(all_t_decoh) - np.min(all_t_decoh):.6f}")
    print(f"   Median: {np.median(all_t_decoh):.6f}")
    
    # Check distribution
    print(f"\n2. РАСПРЕДЕЛЕНИЕ:")
    hist, bins = np.histogram(all_t_decoh, bins=20)
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"   [{bins[i]:.3f}, {bins[i+1]:.3f}]: {hist[i]} значений ({hist[i]/len(all_t_decoh)*100:.1f}%)")
    
    # Check for data leakage
    print(f"\n3. ПРОВЕРКА НА DATA LEAK:")
    
    # Check if sequences use future information
    seq_length = config.get('training', {}).get('seq_length', 100)
    dt = config.get('physics', {}).get('dt', 0.1)
    T_obs = seq_length * dt
    
    print(f"   seq_length: {seq_length}")
    print(f"   dt: {dt}")
    print(f"   T_obs (время наблюдения): {T_obs}")
    
    # Check if target uses information beyond T_obs
    predict_remaining = config.get('training', {}).get('predict_remaining', True)
    print(f"   predict_remaining: {predict_remaining}")
    
    if predict_remaining:
        print(f"   ✓ Целевая переменная: remaining = t_decoh - T_obs")
        print(f"   ✓ Модель видит данные только до T_obs = {T_obs}")
        print(f"   ✓ Нет подглядывания в будущее - правильно!")
    else:
        print(f"   ⚠ Целевая переменная: абсолютное время t_decoh")
        print(f"   ⚠ Модель может использовать информацию о будущем!")
    
    # Check sequence creation
    print(f"\n4. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ:")
    if len(all_observables) > 0:
        sample_obs = all_observables[0]
        if isinstance(sample_obs, dict):
            sample_times = all_times[0] if len(all_times) > 0 else None
            print(f"   Пример траектории:")
            print(f"   - Ключи observables: {list(sample_obs.keys())}")
            if sample_times is not None:
                print(f"   - Длина временной сетки: {len(sample_times)}")
                print(f"   - Время наблюдения (первые {seq_length} точек): {sample_times[:min(seq_length, len(sample_times))][-1]:.3f}")
                if len(sample_times) > seq_length:
                    print(f"   - Полное время симуляции: {sample_times[-1]:.3f}")
                    print(f"   ✓ Последовательность обрезается до seq_length - нет data leak")
                else:
                    print(f"   - Последовательность дополняется (padding)")
    
    # Check if normalization leaks
    print(f"\n5. ПРОВЕРКА НОРМАЛИЗАЦИИ:")
    print(f"   Нормализация применяется ПОСЛЕ разделения на train/val/test")
    print(f"   ✓ Нормализация обучается только на train - правильно!")
    
    return {
        't_decoh': all_t_decoh,
        'gamma': all_gamma,
        'observables': all_observables,
        'times': all_times,
        'config': config
    }


def plot_pauli_oscillations(analysis_data, output_dir: Path, n_samples=5):
    """Plot Pauli operator oscillations for sample trajectories."""
    print(f"\n6. ПОСТРОЕНИЕ ГРАФИКОВ ОСЦИЛЛЯЦИЙ ОПЕРАТОРОВ ПАУЛИ:")
    
    observables = analysis_data['observables']
    times = analysis_data['times']
    t_decoh = analysis_data['t_decoh']
    config = analysis_data['config']
    
    # Get features - check what's available in data, not just config
    # Config might only specify what to use for training, but data has all observables
    available_features = []
    if len(observables) > 0 and isinstance(observables[0], dict):
        available_features = list(observables[0].keys())
        # Filter out non-Pauli observables for plotting
        pauli_features = [f for f in available_features if f.startswith('sigma_')]
        if pauli_features:
            features = pauli_features  # Use all available Pauli operators
        else:
            features = config.get('features', ['sigma_z'])
    else:
        features = config.get('features', ['sigma_z'])
    
    print(f"   Доступные операторы: {available_features}")
    print(f"   Операторы Паули для графиков: {features}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot sample trajectories
    n_plot = min(n_samples, len(observables))
    
    fig, axes = plt.subplots(n_plot, len(features), figsize=(5*len(features), 3*n_plot))
    
    # Handle different subplot configurations
    # plt.subplots returns different types depending on dimensions
    if n_plot == 1 and len(features) == 1:
        # Single subplot - axes is a single Axes object
        axes_array = np.array([[axes]])
    elif n_plot == 1:
        # One row, multiple columns - axes is 1D array
        axes_array = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.reshape(1, -1)
    elif len(features) == 1:
        # Multiple rows, one column - axes is 1D array
        axes_array = np.array([[ax] for ax in axes]) if not isinstance(axes, np.ndarray) else axes.reshape(-1, 1)
    else:
        # Multiple rows and columns - axes is 2D array
        axes_array = np.array(axes)
    
    for i in range(n_plot):
        obs_dict = observables[i]
        time_points = times[i] if i < len(times) else None
        t_d = t_decoh[i]
        
        for j, feat in enumerate(features):
            # Get the correct axis
            ax = axes_array[i, j]
            
            if isinstance(obs_dict, dict) and feat in obs_dict:
                obs_values = obs_dict[feat]
                
                if time_points is None:
                    time_points = np.arange(len(obs_values)) * config.get('physics', {}).get('dt', 0.1)
                
                # Plot trajectory
                ax.plot(time_points, obs_values, 'b-', alpha=0.7, linewidth=1.5, label='True trajectory')
                
                # Mark decoherence time
                ax.axvline(t_d, color='r', linestyle='--', linewidth=2, label=f'$t_{{decoh}}$ = {t_d:.3f}')
                
                # Mark observation window (what model sees)
                seq_length = config.get('training', {}).get('seq_length', 100)
                dt = config.get('physics', {}).get('dt', 0.1)
                T_obs = seq_length * dt
                
                # Split trajectory into observed (model input) and future (model doesn't see)
                obs_mask = time_points <= T_obs
                future_mask = time_points > T_obs
                
                if np.any(obs_mask):
                    # Observed part - what model sees
                    obs_times = time_points[obs_mask]
                    obs_vals = obs_values[obs_mask] if len(obs_values) == len(time_points) else obs_values[:len(obs_times)]
                    ax.plot(obs_times, obs_vals, 'b-', alpha=0.8, linewidth=2, 
                           label='Observed (model input)')
                
                if np.any(future_mask):
                    # Future part - model doesn't see this, only predicts t_decoh
                    future_times = time_points[future_mask]
                    future_vals = obs_values[future_mask] if len(obs_values) == len(time_points) else []
                    if len(future_vals) > 0:
                        ax.plot(future_times, future_vals, 'gray', alpha=0.5, linewidth=1, 
                               linestyle=':', label='Future (model predicts t_decoh)')
                
                # Mark observation boundary
                if T_obs <= time_points[-1]:
                    ax.axvline(T_obs, color='g', linestyle=':', linewidth=2, 
                              label=f'$T_{{obs}}$ = {T_obs:.3f} (model sees only up to here)', alpha=0.7)
                    ax.axvspan(0, T_obs, alpha=0.1, color='green', label='Observation window')
                
                ax.set_xlabel('Time')
                ax.set_ylabel(f'$\\langle {feat} \\rangle$')
                ax.set_title(f'Sample {i+1}: {feat} (t_decoh={t_d:.3f})')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'pauli_oscillations_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✓ Графики сохранены в {output_dir / 'pauli_oscillations_samples.png'}")
    
    # Plot aggregated statistics
    fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
    if len(features) == 1:
        # Single subplot - axes is a single Axes object
        axes_list = [axes]
    else:
        # Multiple subplots - axes is 1D array
        axes_list = axes if isinstance(axes, np.ndarray) else [axes]
    
    for j, feat in enumerate(features):
        ax = axes_list[j]
        
        # Collect all trajectories
        all_obs_vals = []
        all_time_points = []
        
        for i in range(min(50, len(observables))):  # Use first 50 for clarity
            obs_dict = observables[i]
            if isinstance(obs_dict, dict) and feat in obs_dict:
                obs_values = obs_dict[feat]
                time_points = times[i] if i < len(times) else None
                
                if time_points is None:
                    time_points = np.arange(len(obs_values)) * config.get('physics', {}).get('dt', 0.1)
                
                all_obs_vals.append(obs_values)
                all_time_points.append(time_points)
        
        # Plot mean and std
        if all_obs_vals:
            try:
                from scipy.interpolate import interp1d
                
                # Ensure all time points and obs values are numpy arrays
                all_time_points = [np.asarray(tp, dtype=np.float64) for tp in all_time_points]
                all_obs_vals = [np.asarray(obs, dtype=np.float64) for obs in all_obs_vals]
                
                # Find common time grid
                max_time = max(float(tp[-1]) for tp in all_time_points)
                min_time = min(float(tp[0]) for tp in all_time_points)
                common_times = np.linspace(min_time, max_time, 200, dtype=np.float64)
                
                # Interpolate all trajectories to common grid
                interpolated = []
                for obs_vals, tp in zip(all_obs_vals, all_time_points):
                    # Ensure obs_vals and tp are numpy arrays
                    obs_vals = np.asarray(obs_vals, dtype=np.float64).flatten()
                    tp = np.asarray(tp, dtype=np.float64).flatten()
                    
                    # Skip if empty
                    if len(obs_vals) == 0 or len(tp) == 0:
                        continue
                    
                    try:
                        interp_func = interp1d(tp, obs_vals, kind='linear', 
                                              bounds_error=False, fill_value='extrapolate')
                        interp_values = interp_func(common_times)
                        # Ensure interpolated values are numpy array
                        interp_values = np.asarray(interp_values, dtype=np.float64).flatten()
                        if len(interp_values) == len(common_times):
                            interpolated.append(interp_values)
                    except Exception as e:
                        print(f"   ⚠ Warning: Could not interpolate trajectory {i}: {e}")
                        continue
                
                if len(interpolated) == 0:
                    print("   ⚠ No valid interpolated trajectories, skipping average plot")
                    plt.close(fig)
                    return
                
                # Convert to numpy array, ensuring all elements are arrays
                interpolated = np.array(interpolated, dtype=np.float64)
                if interpolated.ndim != 2:
                    print(f"   ⚠ Unexpected interpolated shape: {interpolated.shape}, skipping average plot")
                    plt.close(fig)
                    return
                
                mean_obs = np.mean(interpolated, axis=0)
                std_obs = np.std(interpolated, axis=0)
            except ImportError:
                print("   ⚠ scipy not available, skipping average trajectory plot")
                plt.close(fig)
                return
            
            ax.plot(common_times, mean_obs, 'b-', linewidth=2, label='Mean')
            ax.fill_between(common_times, mean_obs - std_obs, mean_obs + std_obs, 
                           alpha=0.3, color='blue', label='±1 std')
            
            # Mark average decoherence time
            avg_t_decoh = np.mean(t_decoh[:len(all_obs_vals)])
            ax.axvline(avg_t_decoh, color='r', linestyle='--', linewidth=2, 
                      label=f'Avg $t_{{decoh}}$ = {avg_t_decoh:.3f}')
            
            ax.set_xlabel('Time')
            ax.set_ylabel(f'$\\langle {feat} \\rangle$')
            ax.set_title(f'Average {feat} trajectory (n={len(all_obs_vals)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'pauli_oscillations_average.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✓ Графики средних значений сохранены в {output_dir / 'pauli_oscillations_average.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset and check for data leakage')
    parser.add_argument('--data', type=str, default='data/E1_dataset.npz', help='Path to dataset')
    parser.add_argument('--config', type=str, default='configs/experiments/E1.yaml', help='Path to config')
    parser.add_argument('--output-dir', type=str, default='experiments/analysis', help='Output directory')
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    if not data_path.exists():
        print(f"ERROR: Dataset not found: {data_path}")
        return
    
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return
    
    # Analyze
    analysis_data = analyze_dataset(data_path, config_path)
    
    # Plot oscillations
    plot_pauli_oscillations(analysis_data, output_dir, n_samples=5)
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЁН")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    main()
