"""
3D Elastic Particle Collision Simulator with Spatial Hashing
============================================================
Production-grade physics simulation for 10k+ billiard ball collisions in open space with gravity.

Key optimizations:
- Spatial hash grid for O(1) average collision candidate detection
- Numba JIT compilation for hot loops (detection + response)
- Vectorized NumPy operations for physics updates
- Impulse-based collision response (momentum conserving)
- Energy tracking for validation

FIXED: Boundary conditions now use scalar radius instead of array

Author: Computational Physics Co-Pilot
"""

import numpy as np
from numba import jit, prange
import time
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class PhysicsConfig:
    """Physical parameters for the simulation."""
    gravity: float = 9.81
    coefficient_of_restitution: float = 0.99  # Elastic collisions
    damping: float = 0.999  # Velocity damping per step (0.999 = minimal)
    time_step: float = 0.001  # seconds
    particle_radius: float = 0.1  # meters
    particle_mass: float = 1.0  # kg


class SpatialHashGrid:
    """
    3D spatial hash grid for fast collision candidate detection.
    Divides space into uniform cells; particles are hashed to cells.
    """
    
    def __init__(self, cell_size: float, domain_bounds: Tuple[float, float]):
        """
        Args:
            cell_size: Size of each hash cell (should be ~2x particle radius)
            domain_bounds: (min, max) coordinate bounds for the domain
        """
        self.cell_size = cell_size
        self.domain_min, self.domain_max = domain_bounds
        self.grid_cells = {}  # cell_key -> list of particle indices
    
    def _hash_position(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert 3D position to grid cell coordinates."""
        cell_coords = np.floor((pos - self.domain_min) / self.cell_size).astype(int)
        return tuple(cell_coords)
    
    def insert_particles(self, positions: np.ndarray):
        """Insert all particles into the grid."""
        self.grid_cells.clear()
        for i, pos in enumerate(positions):
            cell_key = self._hash_position(pos)
            if cell_key not in self.grid_cells:
                self.grid_cells[cell_key] = []
            self.grid_cells[cell_key].append(i)
    
    def get_collision_candidates(self, particle_idx: int, pos: np.ndarray) -> np.ndarray:
        """
        Get candidate particles for collision with particle_idx.
        Checks 27 neighboring cells (3x3x3 neighborhood).
        """
        cell_key = self._hash_position(pos)
        candidates = set()
        
        # Check 3x3x3 neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (cell_key[0] + dx, cell_key[1] + dy, cell_key[2] + dz)
                    if neighbor_key in self.grid_cells:
                        candidates.update(self.grid_cells[neighbor_key])
        
        candidates.discard(particle_idx)  # Remove self
        return np.array(list(candidates), dtype=np.int32)


@jit(nopython=True, parallel=True)
def detect_and_respond_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    masses: np.ndarray,
    collision_pairs: np.ndarray,
    restitution: float,
    max_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Numba-compiled collision detection and impulse-based response.
    
    Args:
        positions: (N, 3) particle positions
        velocities: (N, 3) particle velocities
        radii: (N,) particle radii
        masses: (N,) particle masses
        collision_pairs: (M, 2) candidate collision pairs
        restitution: coefficient of restitution (1.0 = elastic)
        max_iterations: max iterations for iterative collision resolution
    
    Returns:
        Updated velocities, new positions, and number of collisions detected
    """
    n_particles = positions.shape[0]
    new_velocities = velocities.copy()
    new_positions = positions.copy()
    collision_count = 0
    
    # Collision detection and response loop
    for iteration in range(max_iterations):
        collisions_this_iter = 0
        
        for pair_idx in prange(collision_pairs.shape[0]):
            i, j = collision_pairs[pair_idx, 0], collision_pairs[pair_idx, 1]
            
            if i >= n_particles or j >= n_particles:
                continue
            
            # Vector from j to i
            delta_pos = new_positions[i] - new_positions[j]
            distance = np.sqrt(delta_pos[0]**2 + delta_pos[1]**2 + delta_pos[2]**2)
            min_distance = radii[i] + radii[j]
            
            # Check if collision occurs
            if distance < min_distance and distance > 1e-10:
                collisions_this_iter += 1
                collision_count += 1
                
                # Normal vector
                normal = delta_pos / distance
                
                # Relative velocity
                delta_vel = new_velocities[i] - new_velocities[j]
                relative_speed = delta_vel[0]*normal[0] + delta_vel[1]*normal[1] + delta_vel[2]*normal[2]
                
                # Only resolve if particles are moving toward each other
                if relative_speed < 0:
                    # Impulse calculation (two equal masses assumed)
                    impulse = -(1.0 + restitution) * relative_speed / (1.0/masses[i] + 1.0/masses[j])
                    
                    # Apply impulse to velocities
                    impulse_vec = impulse * normal
                    new_velocities[i] += impulse_vec / masses[i]
                    new_velocities[j] -= impulse_vec / masses[j]
                    
                    # Separation: move particles apart to avoid overlap
                    overlap = min_distance - distance
                    separation = (overlap / 2.0 + 1e-5) * normal
                    new_positions[i] += separation
                    new_positions[j] -= separation
        
        # Exit early if no collisions detected
        if collisions_this_iter == 0:
            break
    
    return new_velocities, new_positions, collision_count


class ParticleCollisionSimulator:
    """
    High-performance 3D elastic particle collision simulator.
    Handles up to 10k+ particles with spatial hashing and Numba optimization.
    """
    
    def __init__(
        self,
        n_particles: int,
        config: PhysicsConfig = PhysicsConfig(),
        domain_bounds: Tuple[float, float] = (-100, 100)
    ):
        """
        Initialize simulator.
        
        Args:
            n_particles: Number of particles to simulate
            config: PhysicsConfig object with physical parameters
            domain_bounds: (min, max) spatial bounds for domain
        """
        self.n_particles = n_particles
        self.config = config
        self.domain_bounds = domain_bounds
        self.time = 0.0
        self.step_count = 0
        
        # Particle state arrays (memory-efficient)
        self.positions = np.zeros((n_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 3), dtype=np.float32)
        self.radii = np.full(n_particles, config.particle_radius, dtype=np.float32)
        self.masses = np.full(n_particles, config.particle_mass, dtype=np.float32)
        
        # Spatial hash grid
        cell_size = 2.5 * config.particle_radius  # ~2.5x particle radius
        self.grid = SpatialHashGrid(cell_size, domain_bounds)
        
        # Statistics tracking
        self.collision_history = []
        self.energy_history = []
        self.momentum_history = []
    
    def initialize_random_configuration(self, seed: int = 42):
        """
        Initialize particles with random positions and velocities in domain.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        d_min, d_max = self.domain_bounds
        domain_size = d_max - d_min
        
        # Random positions within domain
        self.positions = np.random.uniform(
            d_min, d_max - 2*self.config.particle_radius,
            size=(self.n_particles, 3)
        ).astype(np.float32)
        
        # Random velocities (small magnitude for stability)
        self.velocities = np.random.uniform(
            -5, 5,
            size=(self.n_particles, 3)
        ).astype(np.float32)
    
    def initialize_sphere_packing(self, num_per_side: int = 20, velocity_scale: float = 2.0):
        """
        Initialize particles in a regular grid formation (like pool balls).
        
        Args:
            num_per_side: Cuberoot of number of particles (should be ~n_particles^(1/3))
            velocity_scale: Scale of initial velocity magnitudes
        """
        spacing = 2.5 * self.config.particle_radius
        
        idx = 0
        for i in range(num_per_side):
            for j in range(num_per_side):
                for k in range(num_per_side):
                    if idx >= self.n_particles:
                        break
                    
                    x = -50 + i * spacing
                    y = -50 + j * spacing
                    z = k * spacing
                    
                    self.positions[idx] = [x, y, z]
                    
                    # Initial random velocities
                    vx = np.random.uniform(-velocity_scale, velocity_scale)
                    vy = np.random.uniform(-velocity_scale, velocity_scale)
                    vz = np.random.uniform(-velocity_scale, velocity_scale)
                    self.velocities[idx] = [vx, vy, vz]
                    
                    idx += 1
    
    def _get_collision_pairs(self) -> np.ndarray:
        """
        Use spatial hash grid to find collision candidate pairs.
        
        Returns:
            (M, 2) array of particle index pairs to check
        """
        self.grid.insert_particles(self.positions)
        
        pairs = []
        for i in range(self.n_particles):
            candidates = self.grid.get_collision_candidates(i, self.positions[i])
            for j in candidates:
                if i < j:  # Avoid duplicate pairs
                    pairs.append([i, j])
        
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
    
    def step(self):
        """
        Advance simulation by one time step.
        Includes: gravity, damping, collision detection/response, integration.
        """
        dt = self.config.time_step
        
        # Apply gravity
        self.velocities[:, 2] -= self.config.gravity * dt
        
        # Apply damping
        self.velocities *= self.config.damping
        
        # Find collision pairs using spatial hashing
        collision_pairs = self._get_collision_pairs()
        
        # Collision detection and impulse response (Numba-compiled)
        self.velocities, self.positions, collisions = detect_and_respond_collisions(
            self.positions,
            self.velocities,
            self.radii,
            self.masses,
            collision_pairs,
            self.config.coefficient_of_restitution
        )
        
        # Semi-implicit Euler integration
        self.positions += self.velocities * dt
        
        # Boundary conditions: reflect off domain bounds
        self._apply_boundary_conditions()
        
        # Update time and statistics
        self.time += dt
        self.step_count += 1
        
        if self.step_count % 10 == 0:  # Record stats every 10 steps
            self._record_statistics(collisions)
    
    def _apply_boundary_conditions(self):
        """Reflect particles off domain boundaries."""
        d_min, d_max = self.domain_bounds
        radius = self.config.particle_radius  # Use scalar radius (FIXED)
        
        # X boundaries
        out_of_bounds_x_min = self.positions[:, 0] < d_min + radius
        out_of_bounds_x_max = self.positions[:, 0] > d_max - radius
        self.positions[out_of_bounds_x_min, 0] = d_min + radius
        self.positions[out_of_bounds_x_max, 0] = d_max - radius
        self.velocities[out_of_bounds_x_min, 0] = np.abs(self.velocities[out_of_bounds_x_min, 0])
        self.velocities[out_of_bounds_x_max, 0] = -np.abs(self.velocities[out_of_bounds_x_max, 0])
        
        # Y boundaries
        out_of_bounds_y_min = self.positions[:, 1] < d_min + radius
        out_of_bounds_y_max = self.positions[:, 1] > d_max - radius
        self.positions[out_of_bounds_y_min, 1] = d_min + radius
        self.positions[out_of_bounds_y_max, 1] = d_max - radius
        self.velocities[out_of_bounds_y_min, 1] = np.abs(self.velocities[out_of_bounds_y_min, 1])
        self.velocities[out_of_bounds_y_max, 1] = -np.abs(self.velocities[out_of_bounds_y_max, 1])
        
        # Z boundary (floor at z = d_min)
        out_of_bounds_z = self.positions[:, 2] < d_min + radius
        self.positions[out_of_bounds_z, 2] = d_min + radius
        self.velocities[out_of_bounds_z, 2] = np.abs(self.velocities[out_of_bounds_z, 2]) * self.config.coefficient_of_restitution
    
    def _record_statistics(self, collision_count: int):
        """Record energy, momentum, and collision statistics."""
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)
        
        # Momentum
        momentum = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)
        momentum_magnitude = np.linalg.norm(momentum)
        
        # Potential energy (gravitational)
        potential_energy = np.sum(self.masses * self.config.gravity * self.positions[:, 2])
        
        self.collision_history.append(collision_count)
        self.energy_history.append({
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy,
            'time': self.time
        })
        self.momentum_history.append({
            'magnitude': momentum_magnitude,
            'vector': momentum.copy(),
            'time': self.time
        })
    
    def run(self, duration: float, verbose: bool = True) -> Dict:
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation time in seconds
            verbose: Print progress updates
        
        Returns:
            Statistics dictionary
        """
        n_steps = int(duration / self.config.time_step)
        start_time = time.time()
        
        for step_idx in range(n_steps):
            self.step()
            
            if verbose and (step_idx + 1) % max(1, n_steps // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step_idx+1}/{n_steps} | "
                      f"Time: {self.time:.3f}s | "
                      f"Elapsed: {elapsed:.2f}s | "
                      f"Particles: {self.n_particles}")
        
        elapsed_total = time.time() - start_time
        
        stats = {
            'n_steps': n_steps,
            'simulation_time': self.time,
            'wall_time': elapsed_total,
            'steps_per_second': n_steps / elapsed_total,
            'particles': self.n_particles,
            'total_collisions': sum(self.collision_history)
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Simulation complete!")
            print(f"  Particles: {stats['particles']}")
            print(f"  Steps: {stats['n_steps']}")
            print(f"  Sim time: {stats['simulation_time']:.2f}s")
            print(f"  Wall time: {stats['wall_time']:.2f}s")
            print(f"  Performance: {stats['steps_per_second']:.1f} steps/sec")
            print(f"  Total collisions: {stats['total_collisions']}")
            print(f"{'='*60}\n")
        
        return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.95,
        damping=0.999,
        time_step=0.001,
        particle_radius=0.15,
        particle_mass=1.0
    )
    
    # Create simulator with 5000 particles
    print("Initializing simulator with 5000 particles...")
    sim = ParticleCollisionSimulator(
        n_particles=5000,
        config=config,
        domain_bounds=(-50, 50)
    )
    
    # Initialize with grid packing (like stacked billiard balls)
    sim.initialize_sphere_packing(num_per_side=17, velocity_scale=3.0)
    print(f"Particles initialized at random positions in domain.")
    print(f"Domain: [{-50}, {50}]")
    print(f"Particle radius: {config.particle_radius}m")
    print(f"Coefficient of restitution: {config.coefficient_of_restitution}\n")
    
    # Run simulation
    print("Running simulation for 10 seconds...\n")
    stats = sim.run(duration=10.0, verbose=True)
    
    # Print final statistics
    if sim.energy_history:
        final_energy = sim.energy_history[-1]
        print(f"Final energies:")
        print(f"  Kinetic: {final_energy['kinetic']:.2e} J")
        print(f"  Potential: {final_energy['potential']:.2e} J")
        print(f"  Total: {final_energy['total']:.2e} J")
    
    if sim.momentum_history:
        final_momentum = sim.momentum_history[-1]
        print(f"\nFinal momentum magnitude: {final_momentum['magnitude']:.2f} kg·m/s")
    
    print(f"\nParticle state shapes:")
    print(f"  Positions: {sim.positions.shape}")
    print(f"  Velocities: {sim.velocities.shape}")