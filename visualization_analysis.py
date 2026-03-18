"""
Visualization and Analysis Toolkit for Particle Collision Simulator
====================================================================
Real-time plotting, energy analysis, contact dynamics, and performance profiling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import time
from typing import List, Dict, Tuple


class SimulationAnalyzer:
    """Analyze and visualize simulation results."""
    
    def __init__(self, simulator):
        """
        Initialize analyzer with a simulator instance.
        
        Args:
            simulator: ParticleCollisionSimulator instance
        """
        self.sim = simulator
    
    def plot_energy_evolution(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot kinetic, potential, and total energy over time.
        Useful for validating physics (total energy should be ~constant for elastic).
        """
        if not self.sim.energy_history:
            print("No energy history recorded. Run simulation first.")
            return
        
        times = [e['time'] for e in self.sim.energy_history]
        kinetic = [e['kinetic'] for e in self.sim.energy_history]
        potential = [e['potential'] for e in self.sim.energy_history]
        total = [e['total'] for e in self.sim.energy_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Energy components
        ax1.plot(times, kinetic, label='Kinetic', linewidth=2, alpha=0.8)
        ax1.plot(times, potential, label='Potential', linewidth=2, alpha=0.8)
        ax1.plot(times, total, label='Total', linewidth=2.5, color='black', linestyle='--')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Energy (J)', fontsize=11)
        ax1.set_title('Energy Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Energy conservation (% change in total energy)
        initial_energy = total[0]
        energy_change_pct = [(e - initial_energy) / initial_energy * 100 for e in total]
        ax2.plot(times, energy_change_pct, linewidth=2, color='red', alpha=0.7)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Total Energy Change (%)', fontsize=11)
        ax2.set_title('Energy Conservation Check', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_momentum_evolution(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot momentum magnitude and components over time.
        Should remain approximately constant (no external net forces).
        """
        if not self.sim.momentum_history:
            print("No momentum history recorded.")
            return
        
        times = [m['time'] for m in self.sim.momentum_history]
        magnitudes = [m['magnitude'] for m in self.sim.momentum_history]
        px = [m['vector'][0] for m in self.sim.momentum_history]
        py = [m['vector'][1] for m in self.sim.momentum_history]
        pz = [m['vector'][2] for m in self.sim.momentum_history]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(times, magnitudes, label='Magnitude', linewidth=2.5, color='black')
        ax.plot(times, px, label='Px', linewidth=1.5, alpha=0.6)
        ax.plot(times, py, label='Py', linewidth=1.5, alpha=0.6)
        ax.plot(times, pz, label='Pz (gravity)', linewidth=1.5, alpha=0.6, linestyle='--')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Momentum (kg·m/s)', fontsize=11)
        ax.set_title('Linear Momentum Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_collision_rate(self, figsize: Tuple[int, int] = (10, 4)):
        """
        Plot collision detection rate over time.
        Shows dynamics of particle interactions.
        """
        if not self.sim.collision_history:
            print("No collision history recorded.")
            return
        
        steps = np.arange(len(self.sim.collision_history)) * 10  # Every 10 steps
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(steps, self.sim.collision_history, width=8, alpha=0.7, color='steelblue')
        ax.set_xlabel('Step Index', fontsize=11)
        ax.set_ylabel('Collisions Detected', fontsize=11)
        ax.set_title('Collision Detection Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_distribution(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot velocity distribution (magnitude histogram and components).
        """
        v_mag = np.linalg.norm(self.sim.velocities, axis=1)
        v_x = self.sim.velocities[:, 0]
        v_y = self.sim.velocities[:, 1]
        v_z = self.sim.velocities[:, 2]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Velocity magnitude histogram
        axes[0].hist(v_mag, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(np.mean(v_mag), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(v_mag):.2f} m/s')
        axes[0].set_xlabel('Speed (m/s)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Speed Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Component scatter
        axes[1].scatter(v_x, v_y, alpha=0.3, s=10, c=v_z, cmap='viridis')
        axes[1].set_xlabel('Vx (m/s)', fontsize=11)
        axes[1].set_ylabel('Vy (m/s)', fontsize=11)
        axes[1].set_title('Velocity Component Scatter (colored by Vz)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_position_distribution_3d(self, figsize: Tuple[int, int] = (10, 8)):
        """
        3D scatter plot of final particle positions colored by velocity magnitude.
        """
        v_mag = np.linalg.norm(self.sim.velocities, axis=1)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.sim.positions[:, 0],
            self.sim.positions[:, 1],
            self.sim.positions[:, 2],
            c=v_mag,
            cmap='hot',
            s=20,
            alpha=0.6
        )
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title('Final Particle Configuration (colored by speed)', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Speed (m/s)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of simulation statistics.
        """
        report = []
        report.append("=" * 70)
        report.append("PARTICLE COLLISION SIMULATION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Basic info
        report.append("SIMULATION PARAMETERS:")
        report.append(f"  Particles: {self.sim.n_particles}")
        report.append(f"  Domain: [{self.sim.domain_bounds[0]}, {self.sim.domain_bounds[1]}]")
        report.append(f"  Particle radius: {self.sim.config.particle_radius} m")
        report.append(f"  Gravity: {self.sim.config.gravity} m/s²")
        report.append(f"  Coefficient of restitution: {self.sim.config.coefficient_of_restitution}")
        report.append(f"  Damping: {self.sim.config.damping}")
        report.append(f"  Time step: {self.sim.config.time_step} s")
        report.append("")
        
        # Runtime stats
        report.append("SIMULATION EXECUTION:")
        report.append(f"  Total simulation time: {self.sim.time:.3f} s")
        report.append(f"  Total steps: {self.sim.step_count}")
        report.append(f"  Total collisions detected: {sum(self.sim.collision_history)}")
        report.append("")
        
        # Physics validation
        if self.sim.energy_history:
            initial_energy = self.sim.energy_history[0]['total']
            final_energy = self.sim.energy_history[-1]['total']
            energy_error_pct = abs(final_energy - initial_energy) / initial_energy * 100
            
            report.append("ENERGY CONSERVATION:")
            report.append(f"  Initial total energy: {initial_energy:.3e} J")
            report.append(f"  Final total energy: {final_energy:.3e} J")
            report.append(f"  Energy error: {energy_error_pct:.4f}%")
            report.append(f"  Status: {'✓ GOOD' if energy_error_pct < 1.0 else '⚠ WARNING'}")
            report.append("")
        
        # Final state
        v_mag = np.linalg.norm(self.sim.velocities, axis=1)
        report.append("FINAL PARTICLE STATE:")
        report.append(f"  Mean speed: {np.mean(v_mag):.4f} m/s")
        report.append(f"  Max speed: {np.max(v_mag):.4f} m/s")
        report.append(f"  Min speed: {np.min(v_mag):.4f} m/s")
        report.append(f"  Std dev (speed): {np.std(v_mag):.4f} m/s")
        report.append("")
        
        # Position stats
        report.append("SPATIAL DISTRIBUTION:")
        for axis, label in enumerate(['X', 'Y', 'Z']):
            pos_axis = self.sim.positions[:, axis]
            report.append(f"  {label}: [{np.min(pos_axis):.2f}, {np.max(pos_axis):.2f}], "
                        f"mean={np.mean(pos_axis):.2f}, std={np.std(pos_axis):.2f}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


class SimulationVisualizer:
    """Create 3D animation of simulation (requires manual frame capture for high particle counts)."""
    
    @staticmethod
    def plot_single_frame(sim, frame_idx: int = -1, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot a single frame of simulation state.
        
        Args:
            sim: Simulator instance
            frame_idx: Which frame to plot (not used for static sim, just visualization)
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color particles by velocity magnitude
        v_mag = np.linalg.norm(sim.velocities, axis=1)
        
        scatter = ax.scatter(
            sim.positions[:, 0],
            sim.positions[:, 1],
            sim.positions[:, 2],
            c=v_mag,
            cmap='hot',
            s=15,
            alpha=0.6
        )
        
        # Domain boundaries
        d_min, d_max = sim.domain_bounds
        
        ax.set_xlim([d_min, d_max])
        ax.set_ylim([d_min, d_max])
        ax.set_zlim([d_min, d_max])
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'Particle Configuration (t={sim.time:.2f}s, N={sim.n_particles})',
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Speed (m/s)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_trajectory_samples(sim, n_particles_to_plot: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot 3D trajectories for a subset of particles (requires recording history).
        Note: Simulator doesn't record trajectory history by default; this is a placeholder
        for users who want to extend the simulator to store position history.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # For now, just show final positions with velocity vectors
        sample_indices = np.random.choice(sim.n_particles, n_particles_to_plot, replace=False)
        
        for idx in sample_indices:
            pos = sim.positions[idx]
            vel = sim.velocities[idx]
            
            # Draw velocity vector as arrow
            ax.quiver(pos[0], pos[1], pos[2],
                     vel[0]*0.1, vel[1]*0.1, vel[2]*0.1,
                     arrow_length_ratio=0.3, alpha=0.6)
        
        # Plot positions
        ax.scatter(sim.positions[sample_indices, 0],
                  sim.positions[sample_indices, 1],
                  sim.positions[sample_indices, 2],
                  s=100, alpha=0.8, color='red')
        
        d_min, d_max = sim.domain_bounds
        ax.set_xlim([d_min, d_max])
        ax.set_ylim([d_min, d_max])
        ax.set_zlim([d_min, d_max])
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'Velocity Vectors (sample of {n_particles_to_plot} particles)',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig


# ============================================================================
# QUICK ANALYSIS EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Import simulator
    from particle_collisions_3d import ParticleCollisionSimulator, PhysicsConfig
    
    # Quick 10-second simulation
    print("Running quick analysis simulation...")
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.95,
        damping=0.999,
        time_step=0.001
    )
    
    sim = ParticleCollisionSimulator(n_particles=2000, config=config)
    sim.initialize_sphere_packing(num_per_side=13, velocity_scale=2.0)
    sim.run(duration=5.0, verbose=True)
    
    # Analyze and visualize
    print("\nGenerating analysis plots...\n")
    analyzer = SimulationAnalyzer(sim)
    
    # Generate report
    print(analyzer.generate_report())
    
    # Create plots
    fig1 = analyzer.plot_energy_evolution()
    plt.savefig('energy_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: energy_evolution.png")
    
    fig2 = analyzer.plot_momentum_evolution()
    plt.savefig('momentum_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: momentum_evolution.png")
    
    fig3 = analyzer.plot_collision_rate()
    plt.savefig('collision_rate.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: collision_rate.png")
    
    fig4 = analyzer.plot_velocity_distribution()
    plt.savefig('velocity_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: velocity_distribution.png")
    
    fig5 = analyzer.plot_position_distribution_3d()
    plt.savefig('position_3d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: position_3d.png")
    
    # Visualizer
    fig6 = SimulationVisualizer.plot_single_frame(sim)
    plt.savefig('final_state_3d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: final_state_3d.png")
    
    print("\nAll plots generated successfully!")