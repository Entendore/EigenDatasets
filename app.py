import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QGroupBox, QGridLayout, QPushButton, QRadioButton, 
    QButtonGroup, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

# Helper class for drawing beautiful 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class EigenAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set & Group Theory: Eigenvalue Analyzer (with Data Import)")
        self.resize(1400, 850)

        # State variables
        self.mode = "synthetic"
        self.X_imported = None
        self.feature_names = []

        # Main Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Control Panel
        controls_layout = self._create_controls()
        main_layout.addLayout(controls_layout, 1)

        # Right Matplotlib Canvas
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 4)

        # Initial Draw
        self.update_visualization()

    def _create_controls(self):
        outer_layout = QVBoxLayout()
        
        # --- Data Source Group ---
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout()
        
        self.radio_synthetic = QRadioButton("Synthetic Data (3D)")
        self.radio_imported = QRadioButton("Imported CSV Data")
        self.radio_synthetic.setChecked(True)
        
        self.data_group = QButtonGroup()
        self.data_group.addButton(self.radio_synthetic)
        self.data_group.addButton(self.radio_imported)
        
        self.radio_synthetic.toggled.connect(self.toggle_mode)
        
        self.btn_load_csv = QPushButton("Load CSV Dataset")
        self.btn_load_csv.setEnabled(False)
        self.btn_load_csv.clicked.connect(self.load_csv)
        
        self.lbl_dataset_info = QLabel("No dataset imported.")
        self.lbl_dataset_info.setWordWrap(True)
        
        source_layout.addWidget(self.radio_synthetic)
        source_layout.addWidget(self.radio_imported)
        source_layout.addWidget(self.btn_load_csv)
        source_layout.addWidget(self.lbl_dataset_info)
        source_group.setLayout(source_layout)
        outer_layout.addWidget(source_group)

        # --- Synthetic Controls Group ---
        self.synth_group = QGroupBox("Synthetic Covariance Controls")
        synth_layout = QGridLayout()

        self.var_x_slider, var_x_layout = self._create_slider(1, 50, 25)
        self.var_y_slider, var_y_layout = self._create_slider(1, 50, 25)
        self.var_z_slider, var_z_layout = self._create_slider(1, 50, 5)
        self.corr_xy_slider, corr_xy_layout = self._create_slider(-80, 80, 60)
        self.corr_xz_slider, corr_xz_layout = self._create_slider(-80, 80, 0)
        self.corr_yz_slider, corr_yz_layout = self._create_slider(-80, 80, 10)

        synth_layout.addWidget(QLabel("Var X (Stretch):"), 0, 0)
        synth_layout.addLayout(var_x_layout, 0, 1)
        synth_layout.addWidget(QLabel("Var Y (Stretch):"), 1, 0)
        synth_layout.addLayout(var_y_layout, 1, 1)
        synth_layout.addWidget(QLabel("Var Z (Stretch):"), 2, 0)
        synth_layout.addLayout(var_z_layout, 2, 1)
        synth_layout.addWidget(QLabel("Corr XY (Rotate):"), 3, 0)
        synth_layout.addLayout(corr_xy_layout, 3, 1)
        synth_layout.addWidget(QLabel("Corr XZ (Rotate):"), 4, 0)
        synth_layout.addLayout(corr_xz_layout, 4, 1)
        synth_layout.addWidget(QLabel("Corr YZ (Rotate):"), 5, 0)
        synth_layout.addLayout(corr_yz_layout, 5, 1)
        
        self.synth_group.setLayout(synth_layout)
        outer_layout.addWidget(self.synth_group)

        # --- Math Output Group ---
        math_group = QGroupBox("Group & Set Theory Math Output")
        math_layout = QVBoxLayout()
        self.math_label = QLabel("")
        self.math_label.setWordWrap(True)
        self.math_label.setStyleSheet("font-family: monospace; background-color: #f4f4f4; padding: 10px; border-radius: 5px;")
        math_layout.addWidget(self.math_label)
        math_group.setLayout(math_layout)
        outer_layout.addWidget(math_group)
        
        outer_layout.addStretch(1)
        return outer_layout

    def _create_slider(self, min_val, max_val, init_val):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(self.update_visualization)

        label = QLabel(f"{init_val / 10.0 if min_val >= -10 else init_val / 100.0:.2f}")
        label.setFixedWidth(40)

        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(label)

        slider.valueChanged.connect(lambda v: label.setText(f"{v / 10.0 if min_val >= -10 else v / 100.0:.2f}"))
        return slider, layout

    def toggle_mode(self):
        if self.radio_synthetic.isChecked():
            self.mode = "synthetic"
            self.synth_group.setEnabled(True)
            self.btn_load_csv.setEnabled(False)
        else:
            self.mode = "imported"
            self.synth_group.setEnabled(False)
            self.btn_load_csv.setEnabled(True)
            if self.X_imported is None:
                QMessageBox.warning(self, "No Data", "Please load a CSV file.")
                self.radio_synthetic.setChecked(True)
                
        self.update_visualization()

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            try:
                df = pd.read_csv(file_name)
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=[np.number]).dropna()
                
                if numeric_df.shape[1] < 2:
                    QMessageBox.critical(self, "Data Error", "Dataset needs at least 2 numeric columns.")
                    return
                
                self.feature_names = numeric_df.columns.tolist()
                X = numeric_df.values
                
                # Standardize data (crucial for real datasets)
                self.X_imported = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                
                info_text = f"Loaded: {file_name.split('/')[-1]}\nFeatures: {len(self.feature_names)}\nSamples: {X.shape[0]}"
                self.lbl_dataset_info.setText(info_text)
                self.update_visualization()
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def update_visualization(self):
        if self.mode == "synthetic":
            X_centered = self._generate_synthetic_data()
            title_prefix = "Synthetic 3D"
        else:
            if self.X_imported is None: return
            X_centered = self.X_imported
            title_prefix = f"Imported ({len(self.feature_names)}D)"

        # Eigendecomposition
        cov_calc = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_calc)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Group Theory Operations
        Q = eigenvectors
        Q_T = Q.T
        
        # For whitening, avoid division by zero for tiny eigenvalues
        safe_eigenvalues = np.where(eigenvalues > 1e-8, eigenvalues, 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(safe_eigenvalues))
        X_whitened_full = (X_centered @ Q_T) @ D_inv_sqrt

        # 5. Plotting
        self.fig.clear()
        self.fig.suptitle(f'{title_prefix} Data | Set & Group Theory Analyzer', fontsize=14, fontweight='bold')
        
        # Extract Top 3 components for visualization
        n_components = min(3, X_centered.shape[1])
        X_3d = X_centered @ eigenvectors[:, :n_components]
        X_white_3d = X_whitened_full[:, :n_components]
        
        # Pad with zeros if only 2D
        if n_components == 2:
            X_3d = np.hstack((X_3d, np.zeros((X_3d.shape[0], 1))))
            X_white_3d = np.hstack((X_white_3d, np.zeros((X_white_3d.shape[0], 1))))
            
        # Panel 1
        ax1 = self.fig.add_subplot(131, projection='3d')
        self._plot_forward_action(ax1, X_3d, eigenvalues[:n_components])
        
        # Panel 2
        ax2 = self.fig.add_subplot(132, projection='3d')
        self._plot_inverse_action(ax2, X_3d, X_white_3d)
        
        # Panel 3
        ax3 = self.fig.add_subplot(133)
        self._plot_subspace_projection(ax3, X_centered, eigenvectors, eigenvalues)

        self.canvas.draw()

        # 6. Update Math Text
        ortho_check = (Q @ Q_T).round(2)
        var_pct = (eigenvalues[0]/sum(eigenvalues))*100 if sum(eigenvalues) > 0 else 0
        text = (
            f"Set Λ (Top 3 Eigenvalues):\n"
            f"  λ1: {eigenvalues[0]:.3f}\n  λ2: {eigenvalues[1]:.3f}\n  λ3: {eigenvalues[2] if len(eigenvalues)>2 else 0:.3f}\n"
            f"Variance kept by subset {{v1, v2}}: {(eigenvalues[0]+eigenvalues[1])/sum(eigenvalues)*100:.1f}%\n"
            f"\nGroup Theory Proof:\n"
            f"Q * Q^T = I (Identity):\n{ortho_check}"
        )
        self.math_label.setText(text)

    def _generate_synthetic_data(self):
        var_x = self.var_x_slider.value() / 10.0
        var_y = self.var_y_slider.value() / 10.0
        var_z = self.var_z_slider.value() / 10.0
        corr_xy = self.corr_xy_slider.value() / 100.0
        corr_xz = self.corr_xz_slider.value() / 100.0
        corr_yz = self.corr_yz_slider.value() / 100.0

        cov_matrix = [
            [var_x, corr_xy * np.sqrt(var_x * var_y), corr_xz * np.sqrt(var_x * var_z)],
            [corr_xy * np.sqrt(var_x * var_y), var_y, corr_yz * np.sqrt(var_y * var_z)],
            [corr_xz * np.sqrt(var_x * var_z), corr_yz * np.sqrt(var_y * var_z), var_z]
        ]
        np.random.seed(42)
        X = np.random.multivariate_normal([0, 0, 0], cov_matrix, 500)
        return X - np.mean(X, axis=0)

    def _plot_forward_action(self, ax, X_pc, eigenvalues):
        ax.set_title('1. Data in Top 3 PC Space\n(Scaling Group $D$ stretches axes)', fontsize=10)
        ax.scatter(X_pc[:, 0], X_pc[:, 1], X_pc[:, 2], alpha=0.2, c='steelblue', s=5)
        
        # Draw the PC axes (which are the eigenvectors themselves in this space)
        colors = ['red', 'green', 'purple']
        for i in range(3):
            vec_len = np.sqrt(eigenvalues[i]) * 2 if i < len(eigenvalues) else 0
            arrow = Arrow3D([0, vec_len], [0, 0], [0, 0], mutation_scale=12, lw=3, arrowstyle="-|>", color=colors[i])
            ax.add_artist(arrow)
            
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        ax.view_init(elev=20, azim=45)

    def _plot_inverse_action(self, ax, X_pc, X_white):
        ax.set_title('2. Inverse Action: $D^{-1/2} Q^T X$\nWhitening (Spherical Group)', fontsize=10)
        ax.scatter(X_pc[:, 0], X_pc[:, 1], X_pc[:, 2], alpha=0.05, c='gray', s=3)
        ax.scatter(X_white[:, 0], X_white[:, 1], X_white[:, 2], alpha=0.2, c='darkorange', s=5)
        
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        ax.view_init(elev=20, azim=45)

    def _plot_subspace_projection(self, ax, X, eigenvectors, eigenvalues):
        ax.set_title('3. Set Theory: Subset $\{v_1, v_2\}$\nFeature Correlation Projection', fontsize=10)
        
        # Project data to 2D
        X_2d = X @ eigenvectors[:, :2]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, c='teal', s=5)

        # Project original features into 2D subspace
        n_features = X.shape[1]
        original_axes = np.eye(n_features)
        projected_axes = original_axes @ eigenvectors[:, :2]
        
        # Plot feature arrows (limit text labels if too many features)
        for i in range(n_features):
            name = self.feature_names[i] if self.mode == "imported" else f"Feat {i+1}"
            # Only draw arrows if they have meaningful length
            if np.linalg.norm(projected_axes[i]) > 0.1:
                ax.quiver(0, 0, projected_axes[i, 0], projected_axes[i, 1], 
                          angles='xy', scale_units='xy', scale=1, color='darkred', 
                          width=0.005, headwidth=3, headlength=4)
                # Offset text slightly
                ax.text(projected_axes[i, 0]*1.1, projected_axes[i, 1]*1.1, name, 
                        color='darkred', fontsize=8, ha='center', va='center')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        
        # Dynamic limits based on data spread
        max_val = max(np.max(np.abs(X_2d)), np.max(np.abs(projected_axes))*1.2, 3)
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, linestyle=':', alpha=0.4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EigenAnalyzerApp()
    window.show()
    sys.exit(app.exec())