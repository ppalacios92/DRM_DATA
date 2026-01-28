import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .newmark import NewmarkSpectrumAnalyzer
# ================================================================
# INDEPENDENT FUNCTIONS 
def plot_models_response(models, 
                         node_id, 
                         xlim=[0,40], 
                         data_type='velocity'):
    """
    Plot time history response for multiple DRM models.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    node_id : list of lists
        Node IDs for each model. Can include integers or 'QA' string.
        Example: [[0, 5], ['QA'], [10, 'QA', 20]]
    xlim : list, default=[0, 40]
        Time limits [tmin, tmax]
    data_type : str, default='velocity'
        'accel', 'vel', or 'disp'
    """

    if data_type == 'accel':
        ylabel = 'Acceleration'
    elif data_type == 'vel':
        ylabel = 'Velocity'
    else:
        ylabel = 'Displacement'
    
    model_names = [drm.model_name for drm in models]
    fig = plt.figure(figsize=(8, 8))
    
    for i, drm_obj in enumerate(models):
        nodes = node_id[i] if isinstance(node_id[i], list) else [node_id[i]]
        
        for node_idx in nodes:
            # Handle QA
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Model {i}, Node: QA, Position: {drm_obj.xyz_qa[0]}")
                data = drm_obj.get_qa_data(data_type)
                data_x, data_y, data_z = data[0], data[1], data[2]
                label = f'{model_names[i]}_QA_dt={drm_obj.dt:.4f}s'
            elif node_idx < len(drm_obj.xyz):
                print(f"Model {i}, Node: {node_idx}, Position: {drm_obj.xyz_all[node_idx]}")
                data = drm_obj.get_node_data(node_idx, data_type)
                data_x, data_y, data_z = data[0], data[1], data[2]
                label = f'{model_names[i]}_N{node_idx}_dt={drm_obj.dt:.4f}s'
            else:
                print(f"Model {i}, Node: {node_idx}, Position: {drm_obj.xyz_all[node_idx]}")
                data = drm_obj.get_qa_data(data_type)
                data_x, data_y, data_z = data[0], data[1], data[2]
                label = f'{model_names[i]}_QA_dt={drm_obj.dt:.4f}s'
            
            plt.subplot(3, 1, 1)
            plt.plot(drm_obj.time, data_z, linewidth=1, label=label)
            
            plt.subplot(3, 1, 2)
            plt.plot(drm_obj.time, data_x, linewidth=1, label=label)
            
            plt.subplot(3, 1, 3)
            plt.plot(drm_obj.time, data_y, linewidth=1, label=label)
    
    plt.subplot(3, 1, 1)
    plt.title(f'Vertical (Z) - {ylabel}', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title(f'X - {ylabel}', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title(f'Y - {ylabel}', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ================================================================
def plot_models_gf(models, 
                   node_id, 
                   subfault, 
                   xlim=[0,40]):
    """
    Plot Green's functions for multiple DRM models.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    node_id : list of lists
        Node IDs for each model. Can include integers or 'QA' string.
        Example: [[0], [5], ['QA']]
    subfault : int or list
        Subfault ID(s) to plot
    xlim : list, default=[0, 40]
        Time limits [tmin, tmax]
    """
    subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    model_names = [drm.model_name for drm in models]
    fig = plt.figure(figsize=(8, 10))
    
    for i, drm_obj in enumerate(models):
        nodes = node_id[i] if isinstance(node_id[i], list) else [node_id[i]]
        
        for node_idx in nodes:
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Model {i}: ! GFs not available for QA node")
                continue
            
            for sub_idx in subfault_indices:
                print(f"Model {i}, Node: {node_idx}, Subfault: {sub_idx}, Position: {drm_obj.xyz_all[node_idx]}")
                
                gf_z = drm_obj.get_gf(node_idx, sub_idx, 'z')
                gf_e = drm_obj.get_gf(node_idx, sub_idx, 'e')
                gf_n = drm_obj.get_gf(node_idx, sub_idx, 'n')
                
                label = f'{model_names[i]}_N{node_idx}_dt={drm_obj.dt:.4f}s'
                
                plt.subplot(3, 1, 1)
                plt.plot(drm_obj.gf_time, gf_z, linewidth=1, alpha=1, label=label)
                
                plt.subplot(3, 1, 2)
                plt.plot(drm_obj.gf_time, gf_e, linewidth=1, alpha=1, label=label)
                
                plt.subplot(3, 1, 3)
                plt.plot(drm_obj.gf_time, gf_n, linewidth=1, alpha=1, label=label)
    
    plt.subplot(3, 1, 1)
    plt.title('Vertical (Z) - Green Function', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title('East (E) - Green Function', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title('North (N) - Green Function', fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ================================================================
def plot_models_f_spectrum(models, node_id, subfault, xlim=[0,100]):
    """
    Plot Fourier spectrum for multiple DRM models.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    node_id : list of lists
        Node IDs for each model. Can include integers or 'QA' string.
        Example: [[0], [5], ['QA']]
    subfault : int or list
        Subfault ID(s) to plot
    xlim : list, default=[0, 100]
        Frequency limits [fmin, fmax]
    """
    subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    model_names = [drm.model_name for drm in models]
    fig = plt.figure(figsize=(8, 10))
    
    for i, drm_obj in enumerate(models):
        nodes = node_id[i] if isinstance(node_id[i], list) else [node_id[i]]
        
        for node_idx in nodes:
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Model {i}: ! Spectrum not available for QA node")
                continue
            
            for sub_idx in subfault_indices:
                print(f"Model {i}, Node: {node_idx}, Subfault: {sub_idx}, Position: {drm_obj.xyz_all[node_idx]}")
                
                try:
                    real_z = drm_obj.get_spectrum(node_idx, sub_idx, 'z', 'real')
                    imag_z = drm_obj.get_spectrum(node_idx, sub_idx, 'z', 'imag')
                    real_e = drm_obj.get_spectrum(node_idx, sub_idx, 'e', 'real')
                    imag_e = drm_obj.get_spectrum(node_idx, sub_idx, 'e', 'imag')
                    real_n = drm_obj.get_spectrum(node_idx, sub_idx, 'n', 'real')
                    imag_n = drm_obj.get_spectrum(node_idx, sub_idx, 'n', 'imag')
                    
                    mag_z = np.sqrt(real_z**2 + imag_z**2)
                    mag_e = np.sqrt(real_e**2 + imag_e**2)
                    mag_n = np.sqrt(real_n**2 + imag_n**2)
                    
                    label = f'{model_names[i]}_N{node_idx}_dt={drm_obj.dt:.4f}s'
                    
                    plt.subplot(3, 1, 1)
                    plt.loglog(drm_obj.freqs, mag_z, linewidth=1, alpha=0.7, label=label)
                    
                    plt.subplot(3, 1, 2)
                    plt.loglog(drm_obj.freqs, mag_e, linewidth=1, alpha=0.7, label=label)
                    
                    plt.subplot(3, 1, 3)
                    plt.loglog(drm_obj.freqs, mag_n, linewidth=1, alpha=0.7, label=label)
                    
                except KeyError:
                    print(f"  ! No spectrum")
    
    plt.subplot(3, 1, 1)
    plt.title('Vertical (Z) - Fourier Spectrum', fontweight='bold')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (log scale)')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title('East (E) - Fourier Spectrum', fontweight='bold')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (log scale)')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title('North (N) - Fourier Spectrum', fontweight='bold')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (log scale)')
    plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ================================================================
def compare_models_node_response(models, 
                                 node_id, 
                                 data_type='vel', 
                                 reference_index=0):
    """
    Compare DRM models against a reference model.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects to compare
    node_id : list of lists
        Node IDs for each model [[n1], [n2], ...]
    data_type : str
        'accel', 'vel', or 'disp'
    reference_index : int
        Index of reference (trusted) model
    
    Returns
    -------
    dict
        Comparison results
    """
    
    def compute_metrics(sig_ref, sig_test):
        """Calculate comparison metrics between reference and test signal"""
        diff = sig_ref - sig_test
        
        # GoF (Goodness of Fit)
        num = np.sum(diff**2)
        den = np.sum(sig_ref**2 + sig_test**2)
        gof = 1 - np.sqrt(num / den) if den > 0 else 0
        
        # Peak Error %
        max_ref = np.max(np.abs(sig_ref))
        peak_err = (np.max(np.abs(diff)) / max_ref * 100) if max_ref > 0 else 0
        
        # Correlation
        corr = np.corrcoef(sig_ref, sig_test)[0, 1]
        
        # RMSE
        rmse = np.sqrt(np.mean(diff**2))
        
        return gof, peak_err, corr, rmse
    
    n_models = len(models)
    model_names = [drm.model_name for drm in models]
    components = ['Z', 'E', 'N']
    
    # Load data from all models
    data_all = []
    times_all = []
    
    for i, drm in enumerate(models):
        node_idx = node_id[i][0]
        
        if node_idx < len(drm.xyz):
            data = drm.get_node_data(node_idx, data_type)
        else:
            data = drm.get_qa_data(data_type)
        
        # Components: data[0]=X, data[1]=Y, data[2]=Z
        data_z = data[2]
        data_e = data[0]
        data_n = data[1]
        
        data_all.append([data_z, data_e, data_n])
        times_all.append(drm.time)
    
    # Reference data
    ref_data = data_all[reference_index]
    ref_time = times_all[reference_index]
    ref_name = model_names[reference_index]
    
    # Compute metrics for all models vs reference
    results = {}
    
    print("=" * 70)
    print(f"COMPARISON vs Reference ({ref_name})")
    print("=" * 70)
    print(f"Data type: {data_type}")
    print(f"Reference node: {node_id[reference_index][0]}")
    print("")
    
    for i in range(n_models):
        if i == reference_index:
            continue
        
        model_name = model_names[i]
        test_data = data_all[i]
        test_time = times_all[i]
        
        print(f"Model: {model_name} vs Reference")
        
        model_results = {}
        
        for ic, comp in enumerate(components):
            sig_ref = ref_data[ic]
            sig_test = test_data[ic]
            
            # Interpolate to common time
            t_common = np.linspace(
                max(ref_time[0], test_time[0]), 
                min(ref_time[-1], test_time[-1]), 
                min(len(ref_time), len(test_time))
            )
            
            sig_ref_interp = np.interp(t_common, ref_time, sig_ref)
            sig_test_interp = np.interp(t_common, test_time, sig_test)
            
            gof, peak_err, corr, rmse = compute_metrics(sig_ref_interp, sig_test_interp)
            
            model_results[comp] = {
                'gof': gof,
                'peak_err': peak_err,
                'corr': corr,
                'rmse': rmse
            }
            
            print(f"  {comp}: GoF={gof:.4f}, PeakErr={peak_err:.2f}%, Corr={corr:.4f}, RMSE={rmse:.6f}")
        
        results[model_name] = model_results
        print("")
    
    print("=" * 70)

# ================================================================
def plot_models_newmark_spectra(models, 
                                node_id=None,
                                target_pos=None,
                                xlim=[0, 5], 
                                data_type='accel'):
    """
    Plot Newmark response spectra for multiple DRM models.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    node_id : list of lists or list, optional
        Node IDs for each model. Can include integers or 'QA' string.
        Examples: [[0], ['QA'], [5]] or [0, 'QA', 5]
    target_pos : array-like, optional
        [x, y, z] position to find nearest node in all models
    xlim : list, default=[0, 5]
        Period limits [Tmin, Tmax]
    data_type : str, default='accel'
        'accel', 'vel', or 'disp'
    """
    if node_id is None and target_pos is None:
        raise ValueError("Provide node_id or target_pos")
    
    model_names = [drm.model_name for drm in models]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    for i, drm in enumerate(models):
        # Determine node index
        if target_pos is not None:
            distances = np.linalg.norm(drm.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            print(f"Model {i}: Distance = {distances[node_idx]:.6f} km")
        else:
            # Handle list of lists or simple list
            if isinstance(node_id[0], list):
                node_idx = node_id[i][0]
            else:
                node_idx = node_id[i]
        
        # Get data
        if node_idx == 'QA' or node_idx == 'qa':
            data = drm.get_qa_data(data_type)
            label_node = 'QA'
        elif node_idx < len(drm.xyz):
            data = drm.get_node_data(node_idx, data_type)
            label_node = f'N{node_idx}'
        else:
            data = drm.get_qa_data(data_type)
            label_node = 'QA'
        
        data_x, data_y, data_z = data[0], data[1], data[2]
        dt = drm.time[1] - drm.time[0]
        
        # Compute spectra
        if data_type == 'accel':
            spec_z = NewmarkSpectrumAnalyzer.compute(data_z / 9.81, dt)
            spec_x = NewmarkSpectrumAnalyzer.compute(data_x / 9.81, dt)
            spec_y = NewmarkSpectrumAnalyzer.compute(data_y / 9.81, dt)
        else:
            spec_z = NewmarkSpectrumAnalyzer.compute(data_z, dt)
            spec_x = NewmarkSpectrumAnalyzer.compute(data_x, dt)
            spec_y = NewmarkSpectrumAnalyzer.compute(data_y, dt)
        
        T = spec_z['T']
        PSa_z = spec_z['PSa']
        PSa_x = spec_x['PSa']
        PSa_y = spec_y['PSa']
        
        label = f'{model_names[i]}_{label_node}_dt={dt:.4f}s'
        
        # Plot
        axes[0].plot(T, PSa_z, linewidth=2, label=label)
        axes[1].plot(T, PSa_x, linewidth=2, label=label)
        axes[2].plot(T, PSa_y, linewidth=2, label=label)
    
    # Configure
    ylabel = 'Sa (g)' if data_type == 'accel' else 'Spectral Response'
    
    axes[0].set_title('Vertical (Z) - Newmark Spectrum', fontweight='bold')
    axes[0].set_xlabel('T (s)', fontsize=12)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_xlim(xlim)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title('X - Newmark Spectrum', fontweight='bold')
    axes[1].set_xlabel('T (s)', fontsize=12)
    axes[1].set_ylabel(ylabel, fontsize=12)
    axes[1].set_xlim(xlim)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].set_title('Y - Newmark Spectrum', fontweight='bold')
    axes[2].set_xlabel('T (s)', fontsize=12)
    axes[2].set_ylabel(ylabel, fontsize=12)
    axes[2].set_xlim(xlim)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

# ================================================================
def plot_models_DRM(models, 
                   xlim=None, ylim=None, zlim=None,
                   label_nodes=False,
                   show='all',
                   show_nodes=True,
                   show_cubes=True):
    """
    Plot multiple DRM domains together.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    xlim, ylim, zlim : list, optional
        Axis limits [min, max]
    label_nodes : bool or str, optional
        'corners', 'corners_edges', 'corners_half', or False
    show : str, optional
        'all', 'internal', 'boundary'
    show_nodes : bool, optional
        Show scatter points (default: True)
    show_cubes : bool, optional
        Show cube wireframes (default: True)
    """
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    u_x = np.array([0, 1, 0])
    u_y = np.array([1, 0, 0])
    u_z = np.cross(u_x, u_y)
    R = np.column_stack([u_x, u_y, u_z])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, drm in enumerate(models):
        color = colors[i]
        
        xyz_t = drm.xyz * 1000 @ R
        xyz_qa_t = drm.xyz_qa * 1000 @ R
        
        xyz_int = xyz_t[drm.internal]
        xyz_ext = xyz_t[~drm.internal]
        
        # Plot nodes
        if show_nodes:
            if show in ['all', 'boundary']:
                ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                          c=[color], marker='o', s=50, alpha=0.3)
            
            if show in ['all', 'internal']:
                ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                          c=[color], marker='s', s=30, alpha=0.6)
        
        # QA point
        ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                  c=[color], marker='*', s=300, edgecolors='black', 
                  linewidths=2, label=f'{drm.model_name}')
        
        # Cube
        if show_cubes:
            x_min, x_max = xyz_int[:, 0].min(), xyz_int[:, 0].max()
            y_min, y_max = xyz_int[:, 1].min(), xyz_int[:, 1].max()
            z_min, z_max = xyz_int[:, 2].min(), xyz_int[:, 2].max()
            
            corners = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ])
            
            faces = [
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[0], corners[3], corners[7], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]]
            ]
            
            cube = Poly3DCollection(faces, alpha=0.15, facecolor=color,
                                   edgecolor=color, linewidths=2)
            ax.add_collection3d(cube)
        
        # Labels
        if label_nodes == 'corners':
            x_min = xyz_int[:, 0].min()
            x_max = xyz_int[:, 0].max()
            y_min = xyz_int[:, 1].min()
            y_max = xyz_int[:, 1].max()
            z_min = xyz_int[:, 2].min()
            z_max = xyz_int[:, 2].max()
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for j in range(len(xyz_t)):
                x, y, z = xyz_t[j]
                is_corner = ((abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3) and
                           (abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3) and
                           (abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3)) or \
                          ((abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3) and
                           (abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3) and
                           (abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3))
                
                if is_corner:
                    ax.text(x, y, z, str(j), fontsize=8, color=color)
        
        elif label_nodes == 'corners_edges':
            x_min = xyz_int[:, 0].min()
            x_max = xyz_int[:, 0].max()
            y_min = xyz_int[:, 1].min()
            y_max = xyz_int[:, 1].max()
            z_min = xyz_int[:, 2].min()
            z_max = xyz_int[:, 2].max()
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for j in range(len(xyz_t)):
                x, y, z = xyz_t[j]
                on_edge = (sum([abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3,
                              abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3,
                              abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3]) >= 2) or \
                         (sum([abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3,
                              abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3,
                              abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3]) >= 2)
                
                if on_edge:
                    ax.text(x, y, z, str(j), fontsize=8, color=color)
        
        elif label_nodes == 'corners_half':
            x_min = xyz_int[:, 0].min()
            x_max = xyz_int[:, 0].max()
            y_min = xyz_int[:, 1].min()
            y_max = xyz_int[:, 1].max()
            z_min = xyz_int[:, 2].min()
            z_max = xyz_int[:, 2].max()
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            
            for j in range(len(xyz_t)):
                x, y, z = xyz_t[j]
                on_x = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                
                is_corner = on_x and on_y and on_z
                is_mid = sum([on_x and on_y and abs(z - z_mid) < 1e-3,
                            on_x and abs(y - y_mid) < 1e-3 and on_z,
                            abs(x - x_mid) < 1e-3 and on_y and on_z]) > 0
                
                if is_corner or is_mid:
                    ax.text(x, y, z, str(j), fontsize=8, color=color)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_models_tensor_gf(models, 
                          node_id, 
                          subfault, 
                          xlim=[0, 40]):
    """
    Plot tensor Green's functions (9 components) for multiple DRM models.
    
    Parameters
    ----------
    models : list of DRM
        List of DRM objects
    node_id : list of lists
        Node IDs for each model. Example: [[0], [0, 5], [10]]
    subfault : int or list
        Subfault ID(s) to plot
    xlim : list, default=[0, 40]
        Time limits [tmin, tmax]
    """
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    
    subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    model_names = [drm.model_name for drm in models]
    
    component_labels = [
        'G_11', 'G_12', 'G_13',
        'G_21', 'G_22', 'G_23',
        'G_31', 'G_32', 'G_33'
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    for i, drm in enumerate(models):
        nodes = node_id[i] if isinstance(node_id[i], list) else [node_id[i]]
        
        for node_idx in nodes:
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Model {i}: ! Tensor GFs not available for QA node")
                continue
            
            for sub_idx in subfault_indices:
                print(f"Model {model_names[i]}, Node: {node_idx}, Subfault: {sub_idx}")
                
                # Get source node/subfault (handle mapping)
                if drm.node_mapping is not None:
                    pair_idx = drm.node_mapping[node_idx, sub_idx]
                    if pair_idx == -1:
                        print(f"  ! Node {node_idx}, subfault {sub_idx} not computed")
                        continue
                    source_node, source_subfault = drm.pairs_mapping[pair_idx]
                else:
                    source_node, source_subfault = node_idx, sub_idx
                
                # Load tdata and t0
                with h5py.File(drm.filename, 'r') as f:
                    path_tdata = f'GF/sta_{source_node}/sub_{source_subfault}/tdata'
                    path_t0 = f'GF/sta_{source_node}/sub_{source_subfault}/t0'
                    tdata = f[path_tdata][:]
                    t0 = f[path_t0][()]
                
                # Time vector
                time = np.arange(tdata.shape[0]) * drm.dt + t0
                
                label = f'{model_names[i]}_N{node_idx}_S{sub_idx}'
                
                # Plot 9 components
                for j in range(9):
                    row = j // 3
                    col = j % 3
                    ax = axes[row, col]
                    ax.plot(time, tdata[:, j], linewidth=0.8, label=label)
    
    # Configure axes
    for j in range(9):
        row = j // 3
        col = j % 3
        ax = axes[row, col]
        ax.set_title(f'{component_labels[j]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)
    
    axes[0, 0].legend(fontsize=8)
    plt.suptitle('Tensor Green Functions Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()