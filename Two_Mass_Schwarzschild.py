def schwarzschild_metric(M, r, theta):
    """
    Returns the Schwarzschild metric coefficients for given r and theta.

    Parameters:
    M : float - Mass of the central body
    r : float - Radial distance
    theta : float - Angular coordinate

    Returns:
    g : 4x4 ndarray - Metric tensor
    """
    g = np.zeros((4, 4))
    g[0, 0] = -(1 - 2 * M / r)
    g[1, 1] = 1 / (1 - 2 * M / r)
    g[2, 2] = r ** 2
    g[3, 3] = r ** 2 * np.sin(theta) ** 2
    return g


def christoffel_symbols(M, r, theta):
    """
    Computes the Christoffel symbols for the Schwarzschild metric.

    Parameters:
    M : float - Mass of the central body
    r : float - Radial distance
    theta : float - Angular coordinate

    Returns:
    Gamma : 4x4x4 ndarray - Christoffel symbols
    """
    Gamma = np.zeros((4, 4, 4))
    # Non-zero Christoffel symbols for Schwarzschild metric:
    Gamma[0, 0, 1] = M / (r * (r - 2 * M))
    Gamma[0, 1, 0] = M / (r * (r - 2 * M))
    Gamma[1, 0, 0] = (2 * M) / (r * (r - 2 * M))
    Gamma[1, 1, 1] = -M / (r * (r - 2 * M))
    Gamma[1, 2, 2] = -(r - 2 * M)
    Gamma[1, 3, 3] = -(r - 2 * M) * np.sin(theta) ** 2
    Gamma[2, 1, 2] = 1 / r
    Gamma[2, 2, 1] = 1 / r
    Gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
    Gamma[3, 1, 3] = 1 / r
    Gamma[3, 3, 1] = 1 / r
    Gamma[3, 2, 3] = np.cos(theta) / np.sin(theta)
    Gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
    return Gamma


def geodesic_equations(tau, state, M):
    """
    Defines the geodesic equations for a particle in the Schwarzschild spacetime.

    Parameters:
    tau : float - Proper time
    state : ndarray - Array containing [t, r, theta, phi, v_t, v_r, v_theta, v_phi]
    M : float - Mass of the central body

    Returns:
    dstate_dtau : ndarray - Time derivative of the state
    """
    t, r, theta, phi, v_t, v_r, v_theta, v_phi = state

    # Compute Christoffel symbols at the current position
    Gamma = christoffel_symbols(M, r, theta)

    # Initialize derivatives
    dstate_dtau = np.zeros_like(state)

    # dx/dtau = velocity
    dstate_dtau[0] = v_t
    dstate_dtau[1] = v_r
    dstate_dtau[2] = v_theta
    dstate_dtau[3] = v_phi

    # dv/dtau = acceleration from Christoffel symbols
    dstate_dtau[4] = -Gamma[0, 1, 1] * v_r ** 2 - Gamma[0, 2, 2] * v_theta ** 2 - Gamma[0, 3, 3] * v_phi ** 2
    dstate_dtau[5] = -Gamma[1, 0, 0] * v_t ** 2 - Gamma[1, 2, 2] * v_theta ** 2 - Gamma[1, 3, 3] * v_phi ** 2
    dstate_dtau[6] = -Gamma[2, 2, 1] * v_theta * v_r - Gamma[2, 3, 3] * v_phi ** 2
    dstate_dtau[7] = -Gamma[3, 3, 1] * v_phi * v_r - Gamma[3, 3, 2] * v_phi * v_theta

    return dstate_dtau



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Constants and Functions
M = 1.0  # Mass of the central body
tau_span = (0, 100)  # Proper time interval
initial_state = [0, 10, np.pi / 2, 0, 0, 0, 0.1, 0.05]  # [t, r, theta, phi, v_t, v_r, v_theta, v_phi]

def geodesic_equations(tau, state, M):
    """Geodesic equations."""
    t, r, theta, phi, v_t, v_r, v_theta, v_phi = state
    Gamma = christoffel_symbols(M, r, theta)
    dstate_dtau = np.zeros_like(state)
    dstate_dtau[:4] = [v_t, v_r, v_theta, v_phi]
    dstate_dtau[4] = -Gamma[0, 1, 1] * v_r**2 - Gamma[0, 2, 2] * v_theta**2 - Gamma[0, 3, 3] * v_phi**2
    dstate_dtau[5] = -Gamma[1, 0, 0] * v_t**2 - Gamma[1, 2, 2] * v_theta**2 - Gamma[1, 3, 3] * v_phi**2
    dstate_dtau[6] = -Gamma[2, 1, 2] * v_r * v_theta - Gamma[2, 3, 3] * v_phi**2
    dstate_dtau[7] = -Gamma[3, 1, 3] * v_r * v_phi - Gamma[3, 2, 3] * v_theta * v_phi
    return dstate_dtau

def kretschmann_scalar(M, r):
    """Kretschmann scalar."""
    return 48 * M**2 / r**6

# Numerical integration
solution = solve_ivp(
    geodesic_equations,
    tau_span,
    initial_state,
    args=(M,),
    method='RK45',
    t_eval=np.linspace(tau_span[0], tau_span[1], 1000),
)

# Extract results
t_vals = solution.y[0]
r_vals = solution.y[1]
theta_vals = solution.y[2]
phi_vals = solution.y[3]
x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
z_vals = r_vals * np.cos(theta_vals)

# Generate and save plots
# 1. 2D Trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="Geodesic")
plt.scatter(0, 0, color="red", label="Massive Body (M)")
plt.title("2D Projection of Geodesic in Schwarzschild Spacetime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.grid()
plt.savefig("geodesic_2d_projection.png")

# 2. 3D Trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label="Geodesic")
ax.scatter(0, 0, 0, color="red", label="Massive Body (M)")
ax.set_title("3D Visualization of Geodesic in Schwarzschild Spacetime")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.savefig("geodesic_3d_trajectory.png")

# 3. Kretschmann Scalar
r_vals_field = np.linspace(2 * M + 0.1, 20, 500)
K_vals = kretschmann_scalar(M, r_vals_field)
plt.figure(figsize=(8, 6))
plt.plot(r_vals_field, K_vals, label="Kretschmann Scalar $K$")
plt.title("Kretschmann Scalar in Schwarzschild Spacetime")
plt.xlabel("r")
plt.ylabel("K")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("kretschmann_scalar.png")

# 4. Geodesic with Curvature Field
K_normalized = (kretschmann_scalar(M, r_vals) - min(K_vals)) / (max(K_vals) - min(K_vals))
plt.figure(figsize=(8, 8))
plt.scatter(x_vals, y_vals, c=K_normalized, cmap='viridis', label="Geodesic (Curvature Field)")
plt.scatter(0, 0, color="red", label="Massive Body (M)")
plt.title("Geodesic with Spacetime Curvature Field")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Normalized Kretschmann Scalar")
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig("geodesic_curvature_field.png")
