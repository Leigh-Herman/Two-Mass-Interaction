import numpy as np


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

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
M = 1.0  # Mass of the central body
tau_span = (0, 100)  # Proper time interval
initial_state = [0, 10, np.pi / 2, 0, 0, 0, 0.1, 0.05]  # [t, r, theta, phi, v_t, v_r, v_theta, v_phi]

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

# Convert to Cartesian coordinates for visualization
x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
z_vals = r_vals * np.cos(theta_vals)

# Plot the trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="Geodesic")
plt.scatter(0, 0, color="red", label="Massive Body (M)")
plt.title("Trajectory of a Small Mass in Schwarzschild Spacetime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

# Plot the 2D trajectory (x-y projection)
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="Geodesic")
plt.scatter(0, 0, color="red", label="Massive Body (M)")
plt.title("2D Projection of Geodesic in Schwarzschild Spacetime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

# Plot the 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label="Geodesic")
ax.scatter(0, 0, 0, color="red", label="Massive Body (M)")
ax.set_title("3D Visualization of Geodesic in Schwarzschild Spacetime")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()


def kretschmann_scalar(M, r):
    """
    Computes the Kretschmann scalar for the Schwarzschild spacetime.

    Parameters:
    M : float - Mass of the central body
    r : float or ndarray - Radial distance(s)

    Returns:
    K : float or ndarray - Kretschmann scalar
    """
    return 48 * M ** 2 / r ** 6


# Generate values for r
r_vals_field = np.linspace(2 * M + 0.1, 20, 500)  # Avoid r = 2M (Schwarzschild radius)
K_vals = kretschmann_scalar(M, r_vals_field)

# Plot Kretschmann scalar as a function of r
plt.figure(figsize=(8, 6))
plt.plot(r_vals_field, K_vals, label="Kretschmann Scalar $K$")
plt.title("Kretschmann Scalar in Schwarzschild Spacetime")
plt.xlabel("r")
plt.ylabel("K")
plt.yscale("log")  # Log scale to capture wide range of values
plt.grid()
plt.legend()
plt.show()
