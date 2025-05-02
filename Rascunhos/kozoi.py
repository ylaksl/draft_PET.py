import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def kozai_lidov(t, y, A, B):
    """Defines the differential equations for the Kozai-Lidov mechanism."""
    omega, e, itheta = y
    domega_dt = A / (1 - e**2) * np.sin(2 * omega) * np.cos(itheta)
    de_dt = B * e * np.sin(2 * omega) * np.cos(itheta)
    ditheta_dt = -B * np.sin(2 * omega)
    return [domega_dt, de_dt, ditheta_dt]

def simulate_kozai_lidov(A=0.1, B=0.05, t_span=(0, 100), y0=[0, 0.5, np.pi/4]):
    """Solves the Kozai-Lidov system numerically."""
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(kozai_lidov, t_span, y0, t_eval=t_eval, args=(A, B))
    return sol.t, sol.y

def plot_kozai_lidov():
    """Plots the evolution of the Kozai-Lidov mechanism variables."""
    t, y = simulate_kozai_lidov()
    omega, e, itheta = y
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, omega, label=r'$\omega$')
    axs[0].set_ylabel(r'$\omega$')
    axs[0].legend()
    
    axs[1].plot(t, e, label=r'$e$', color='r')
    axs[1].set_ylabel(r'$e$')
    axs[1].legend()
    
    axs[2].plot(t, itheta, label=r'$\theta$', color='g')
    axs[2].set_ylabel(r'$\theta$')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kozai_lidov()
