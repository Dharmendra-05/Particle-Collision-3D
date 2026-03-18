"""
ADVANCED EXAMPLES & PRACTICAL USE CASES
========================================
Demonstrates various scenarios and techniques for the particle collision simulator.
"""

import numpy as np
from particle_collisions_3d import ParticleCollisionSimulator, PhysicsConfig
from visualization_analysis import SimulationAnalyzer, SimulationVisualizer


# ============================================================================
# EXAMPLE 1: Billiard Ball Break Shot
# ============================================================================

def example_billiard_break():
    """
    Simulate a pool/billiard break shot.
    15 balls arranged in triangle + cue ball moving toward them.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: BILLIARD BREAK SHOT")
    print("="*70)
    
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.98,  # Professional billiard balls
        damping=0.995,  # Slight air resistance
        time_step=0.0005,  # Small timestep for accuracy
        particle_radius=0.0285,  # Standard billiard ball: 57mm diameter
        particle_mass=0.17  # kg
    )
    
    sim = ParticleCollisionSimulator(n_particles=16, config=config, domain_bounds=(0, 3))
    
    # Triangle rack for 15 balls (standard pool)
    # 1-5-4-3-2 arrangement
    positions = [
        # Cue ball (white, moving)
        [0.2, 1.5, 0.0285],
    ]
    
    # Triangle arrangement
    spacing = 0.06  # ~2x ball diameter
    apex_x = 2.5
    apex_y = 1.5
    
    for row in range(5):
        for col in range(row + 1):
            x = apex_x + row * spacing * np.cos(np.pi/6)
            y = apex_y - (row/2 - col) * spacing
            positions.append([x, y, 0.0285])
    
    # Set positions
    for i, pos in enumerate(positions):
        sim.positions[i] = pos
    
    # Cue ball has high initial velocity
    sim.velocities[0] = [8.0, 0.0, 0.0]  # 8 m/s toward rack
    
    # Run simulation
    print("Simulating break shot...")
    print("  Cue ball velocity: 8.0 m/s")
    print("  Domain: 0-3 meters")
    sim.run(duration=3.0, verbose=False)
    
    # Analyze
    analyzer = SimulationAnalyzer(sim)
    print("\n" + analyzer.generate_report())
    
    return sim


# ============================================================================
# EXAMPLE 2: Granular Flow (Sand)
# ============================================================================

def example_granular_flow():
    """
    Simulate sand-like particles with higher damping and lower restitution.
    Shows how parameters change behavior from elastic to dissipative.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: GRANULAR FLOW (SAND-LIKE PARTICLES)")
    print("="*70)
    
    # Sand particles: small, dense, inelastic
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.4,  # Very inelastic
        damping=0.96,  # Significant damping (energy loss)
        time_step=0.002,  # Larger timestep OK with damping
        particle_radius=0.02,  # Smaller grains (4cm diameter)
        particle_mass=0.5
    )
    
    sim = ParticleCollisionSimulator(n_particles=3000, config=config, domain_bounds=(-20, 20))
    
    # Random initial positions (falling from above)
    np.random.seed(42)
    sim.positions[:, 0] = np.random.uniform(-15, 15, 3000)  # X
    sim.positions[:, 1] = np.random.uniform(-15, 15, 3000)  # Y
    sim.positions[:, 2] = np.random.uniform(10, 30, 3000)  # Z (high up)
    
    # Small random velocities
    sim.velocities[:, :] = np.random.uniform(-1, 1, (3000, 3))
    
    print("Simulating granular flow...")
    print("  Particles: 3000")
    print("  Coefficient of restitution: 0.4 (inelastic)")
    print("  Damping: 0.96 (significant dissipation)")
    
    sim.run(duration=8.0, verbose=False)
    
    # Analyze
    analyzer = SimulationAnalyzer(sim)
    print("\n" + analyzer.generate_report())
    
    # Plot energy decay (should show exponential decay)
    fig = analyzer.plot_energy_evolution()
    fig.savefig('/tmp/granular_energy.png', dpi=100)
    print("✓ Saved: granular_energy.png")
    
    return sim


# ============================================================================
# EXAMPLE 3: Shaking/Vibration Study
# ============================================================================

def example_shaking_container():
    """
    Simulate particles in a container that shakes.
    Models seismic activity or shaker table experiments.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: PARTICLE DYNAMICS IN SHAKING CONTAINER")
    print("="*70)
    
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.85,
        damping=0.98,
        time_step=0.001,
        particle_radius=0.1,
        particle_mass=1.0
    )
    
    sim = ParticleCollisionSimulator(n_particles=2000, config=config, domain_bounds=(-30, 30))
    
    # Random packing
    np.random.seed(123)
    sim.positions = np.random.uniform(-25, 25, (2000, 3)).astype(np.float32)
    sim.velocities = np.random.uniform(-2, 2, (2000, 3)).astype(np.float32)
    
    # Custom step function to apply shaking
    class ShakedSimulator(ParticleCollisionSimulator):
        def __init__(self, *args, shake_amplitude=2.0, shake_freq=5.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.shake_amplitude = shake_amplitude
            self.shake_freq = shake_freq
        
        def step(self):
            # Apply sinusoidal acceleration (shaking)
            shake_accel = self.shake_amplitude * np.sin(2 * np.pi * self.shake_freq * self.time)
            
            # Apply to all particles
            self.velocities[:, 0] += shake_accel * self.config.time_step
            
            # Call parent step
            super().step()
    
    # Create shaking simulator
    shaking_sim = ShakedSimulator(
        n_particles=2000,
        config=config,
        domain_bounds=(-30, 30),
        shake_amplitude=5.0,
        shake_freq=3.0  # 3 Hz shaking
    )
    shaking_sim.positions = sim.positions.copy()
    shaking_sim.velocities = sim.velocities.copy()
    
    print("Simulating particles in shaking container...")
    print("  Shake frequency: 3 Hz")
    print("  Shake amplitude: 5 m/s²")
    print("  Particles: 2000")
    
    shaking_sim.run(duration=5.0, verbose=False)
    
    # Analyze
    analyzer = SimulationAnalyzer(shaking_sim)
    print("\n" + analyzer.generate_report())
    
    fig = analyzer.plot_energy_evolution()
    fig.savefig('/tmp/shaking_energy.png', dpi=100)
    print("✓ Saved: shaking_energy.png")
    
    return shaking_sim


# ============================================================================
# EXAMPLE 4: Comparative Physics Study
# ============================================================================

def example_comparison_elastic_vs_inelastic():
    """
    Run identical initial conditions with different coefficients of restitution.
    Compare energy decay and settling times.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: ELASTIC vs INELASTIC COLLISIONS")
    print("="*70)
    
    np.random.seed(999)
    
    scenarios = {
        'Elastic (e=1.0)': {
            'coefficient_of_restitution': 1.0,
            'damping': 0.9999,
            'color': 'blue'
        },
        'Nearly Elastic (e=0.95)': {
            'coefficient_of_restitution': 0.95,
            'damping': 0.999,
            'color': 'green'
        },
        'Inelastic (e=0.5)': {
            'coefficient_of_restitution': 0.5,
            'damping': 0.99,
            'color': 'red'
        }
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\nRunning: {scenario_name}")
        
        config = PhysicsConfig(
            gravity=9.81,
            coefficient_of_restitution=params['coefficient_of_restitution'],
            damping=params['damping'],
            time_step=0.001,
            particle_radius=0.1,
            particle_mass=1.0
        )
        
        sim = ParticleCollisionSimulator(n_particles=1000, config=config)
        
        # Same initial conditions for all
        seed_pos = np.random.uniform(-20, 20, (1000, 3))
        seed_vel = np.random.uniform(-3, 3, (1000, 3))
        
        sim.positions = seed_pos.astype(np.float32)
        sim.velocities = seed_vel.astype(np.float32)
        
        sim.run(duration=10.0, verbose=False)
        
        results[scenario_name] = sim
        
        # Extract metrics
        if sim.energy_history:
            initial_energy = sim.energy_history[0]['total']
            final_energy = sim.energy_history[-1]['total']
            energy_retained = final_energy / initial_energy * 100
            print(f"  Energy retained: {energy_retained:.1f}%")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    for scenario_name, sim in results.items():
        analyzer = SimulationAnalyzer(sim)
        print(f"\n{scenario_name}:")
        
        if sim.energy_history:
            initial = sim.energy_history[0]['total']
            final = sim.energy_history[-1]['total']
            print(f"  Total energy change: {(final-initial)/initial*100:.2f}%")
        
        v_mag = np.linalg.norm(sim.velocities, axis=1)
        print(f"  Final mean speed: {np.mean(v_mag):.3f} m/s")
        print(f"  Final max speed: {np.max(v_mag):.3f} m/s")
    
    return results


# ============================================================================
# EXAMPLE 5: Large Scale Simulation (10k+ particles)
# ============================================================================

def example_large_scale_10k():
    """
    Demonstrate high-performance simulation with 10k particles.
    Shows scalability and real-time performance metrics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: LARGE SCALE SIMULATION (10,000 PARTICLES)")
    print("="*70)
    
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.95,
        damping=0.999,
        time_step=0.001,
        particle_radius=0.12,
        particle_mass=1.0
    )
    
    print("Initializing 10,000 particle system...")
    sim = ParticleCollisionSimulator(n_particles=10000, config=config, domain_bounds=(-40, 40))
    sim.initialize_sphere_packing(num_per_side=21, velocity_scale=2.5)
    
    print("Running 5-second simulation (5000 time steps)...")
    stats = sim.run(duration=5.0, verbose=True)
    
    print("\nPerformance Summary:")
    print(f"  Steps per second: {stats['steps_per_second']:.1f}")
    print(f"  Time per step: {stats['wall_time']/stats['n_steps']*1000:.2f} ms")
    print(f"  Particles: {stats['particles']}")
    print(f"  Total collisions: {stats['total_collisions']}")
    
    # Efficiency metric
    collisions_per_step = stats['total_collisions'] / stats['n_steps']
    print(f"  Collisions per step: {collisions_per_step:.1f}")
    
    return sim


# ============================================================================
# EXAMPLE 6: Custom Initialization Patterns
# ============================================================================

def example_custom_initialization():
    """
    Demonstrate various initialization patterns and configurations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: CUSTOM INITIALIZATION PATTERNS")
    print("="*70)
    
    config = PhysicsConfig(
        gravity=9.81,
        coefficient_of_restitution=0.95,
        damping=0.999,
        time_step=0.001,
        particle_radius=0.1,
        particle_mass=1.0
    )
    
    # Pattern 1: Single layer flat
    print("\nPattern 1: Flat Monolayer")
    sim1 = ParticleCollisionSimulator(n_particles=500, config=config)
    z_height = 0.1
    x = np.linspace(-20, 20, 25)
    y = np.linspace(-20, 20, 20)
    X, Y = np.meshgrid(x, y)
    sim1.positions[:500, 0] = X.flatten()[:500]
    sim1.positions[:500, 1] = Y.flatten()[:500]
    sim1.positions[:500, 2] = z_height
    sim1.velocities[:500, :] = np.random.uniform(-1, 1, (500, 3))
    
    # Pattern 2: Gaussian cloud
    print("Pattern 2: Gaussian Cloud")
    sim2 = ParticleCollisionSimulator(n_particles=1000, config=config)
    sim2.positions = np.random.normal(0, 10, (1000, 3)).astype(np.float32)
    sim2.velocities = np.random.normal(0, 2, (1000, 3)).astype(np.float32)
    
    # Pattern 3: Spiral shell
    print("Pattern 3: Spiral Shell")
    sim3 = ParticleCollisionSimulator(n_particles=1000, config=config)
    t = np.linspace(0, 4*np.pi, 1000)
    r = 15
    sim3.positions[:, 0] = r * np.cos(t)
    sim3.positions[:, 1] = r * np.sin(t)
    sim3.positions[:, 2] = t / (4*np.pi) * 20
    sim3.velocities = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)
    
    # Pattern 4: Two colliding spheres
    print("Pattern 4: Two Colliding Spheres")
    sim4 = ParticleCollisionSimulator(n_particles=2000, config=config)
    
    # Sphere 1 (moving left)
    theta1 = np.random.uniform(0, 2*np.pi, 1000)
    phi1 = np.random.uniform(0, np.pi, 1000)
    r_sphere = 8
    sim4.positions[:1000, 0] = -15 + r_sphere * np.sin(phi1) * np.cos(theta1)
    sim4.positions[:1000, 1] = r_sphere * np.sin(phi1) * np.sin(theta1)
    sim4.positions[:1000, 2] = r_sphere * np.cos(phi1)
    sim4.velocities[:1000, 0] = 3.0  # Moving right
    
    # Sphere 2 (moving right)
    sim4.positions[1000:, 0] = 15 + r_sphere * np.sin(phi1) * np.cos(theta1)
    sim4.positions[1000:, 1] = r_sphere * np.sin(phi1) * np.sin(theta1)
    sim4.positions[1000:, 2] = r_sphere * np.cos(phi1)
    sim4.velocities[1000:, 0] = -3.0  # Moving left
    
    print("\n✓ All patterns initialized successfully")
    print("  Available for immediate simulation")
    
    return sim1, sim2, sim3, sim4


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " 3D PARTICLE COLLISION SIMULATOR - ADVANCED EXAMPLES".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Run examples
    try:
        # Example 1: Billiard break
        billiard_sim = example_billiard_break()
        
        # Example 2: Granular flow
        granular_sim = example_granular_flow()
        
        # Example 3: Shaking
        shaking_sim = example_shaking_container()
        
        # Example 4: Comparison
        comparison_sims = example_comparison_elastic_vs_inelastic()
        
        # Example 5: Large scale
        large_scale_sim = example_large_scale_10k()
        
        # Example 6: Custom patterns
        custom_sims = example_custom_initialization()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print("\nGenerated simulations:")
        print("  1. Billiard break (16 particles)")
        print("  2. Granular flow (3000 particles)")
        print("  3. Shaking container (2000 particles)")
        print("  4. Elastic vs inelastic comparison (1000 particles each)")
        print("  5. Large scale (10,000 particles)")
        print("  6. Custom initialization patterns (up to 2000 particles)")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()