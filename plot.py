"""
This file contains the code for all relevant plots.
"""
import numpy as np
import scipy.special as spy
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import os

#Set style and font
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

def set_ax_info(ax, xlabel, ylabel, style='plain', title=None):
    """
    Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        title (str): the desired title on the axis
        xlabel (str): the desired lab on the x-axis
        ylabel (str): the desired lab on the y-axis
    """
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.ticklabel_format(style=style)
    ax.legend(fontsize=15)


r = 0.1
alpha = r/(1+r)
beta = 1-alpha

def E_r(t):
    """
    Re part of energy
    """
    u = np.ones(t.shape)
    u[t < 0] = 0
    u[t > 1] = -1
    u[t > 2] = 0
    return u

def E_i(t):
    """
    Im part of energy
    """
    return np.log(np.abs(2 - t)) - np.log(np.abs(-t))

def exp_u1(t):
    """
    This is the expectd energy density for the squeezed state
    """
    u = alpha*2*(E_r(t)**2 + E_i(t)**2)
    return u


def exp_u_psi(t):
    """
    This is the expectd energy density for the single psi state
    """
    u = beta*2*r*(E_r(t)**2 - E_i(t)**2)
    return u

def exp_u(t):
    """
    This is the expected energy density for the mixed state
    """
    u = 4*r/(r + 1) * E_r(t)**2
    return u

def xi_r(omega):
    """
    Re part of frequency spectrum
    """
    val = np.zeros(omega.shape) 
    omega_ = omega[omega > 0]
    val[omega > 0] = 2*np.sinc(omega_/2) * np.sin(omega_/2)*np.sin(omega_)/(np.pi*np.sqrt(omega_))
    return val

def xi_i(omega):
    """
    Im part of frequency spectrum
    """
    val = np.zeros(omega.shape) 
    omega_ = omega[omega > 0]
    val[omega > 0] = -2*np.sinc(omega_/2) * np.sin(omega_/2)* np.cos(omega_)/(np.pi*np.sqrt(omega_))
    return val

def fidelity_mix(b):
    """
    Fidelity for pure state
    """
    fidelity = spy.erf(b) - 2*b*np.exp(-b**2)/np.sqrt(np.pi)
    return fidelity

def fidelity_pure(b):
    """
    Fidelity for mixed state
    """
    inner_product = spy.erf(b) - 2*b*np.exp(-b**2)/np.sqrt(np.pi)
    fidelity = np.sqrt(inner_product)
    return fidelity

def plot_fid():
    """
    Plot fidelity
    """
    b_array = np.linspace(0,3,1000)
    fid_mix_array = fidelity_mix(b_array)
    fid_pure_array = fidelity_pure(b_array)

    fig, ax = plt.subplots()
    
    title = r'Fidelity Plot'
    xlabel = r'$b$'
    ylabel = r'Fidelity'
    ax.plot(b_array, fid_mix_array, label = r'$F(\rho,|1_f\rangle)$')
    
    ax.plot(b_array, fid_pure_array, label = r'$\sqrt{\langle 1_f(b)|1_f(b)\rangle}$')

    ax.plot([1/np.sqrt(2), 1/np.sqrt(2)], [0, 1], '--',label = r'Maximum possible $b$')
    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)

    fig.set_size_inches(6.4, 4.8)
    fig.tight_layout()
    here = os.path.abspath(".")
    path_plots = 'plots/'
    file = path_plots + "fidelity_plot.pdf"
    print(f'Saving plot: file://{here}/{file}')
    fig.savefig(file)
    # plt.show()

def plot_energy_and_spectrum():
    """
    Plot of energy and spectrum
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)

    title = 'Energy Spectrum'
    xlabel = r'$t$'
    ylabel = r'Energy'
    t_array = np.linspace(-0.5,2.5,100000)
    ax1.set_ylim([-1,1])
    ax1.plot(t_array, exp_u1(t_array), label = r'$\alpha \langle 1| {:}\!\, E^2{:} \!\, |1 \rangle$')
    ax1.plot(t_array, exp_u_psi(t_array), label = r'$\beta \langle \psi| {:}\!\, E^2{:} \!\, |\psi \rangle$')
    ax1.plot(t_array, exp_u(t_array), label = r'$\langle u \rangle$')

    set_ax_info(ax1, xlabel, ylabel, style='sci', title=title)


    title = 'Frequency Spectrum'
    xlabel = r'$\omega$'
    ylabel = r'Amplitude'
    omega_array = np.linspace(-2,5,100000)
    ax2.plot(omega_array, xi_r(omega_array), label = r'$\xi_r$')
    ax2.plot(omega_array, xi_i(omega_array), label = r'$\xi_i$')
    ax2.set_ylim([-0.2,0.2])


    set_ax_info(ax2, xlabel, ylabel, style='sci', title=title)


    #Saving
    fig.set_size_inches(6.4, 4.8*2)
    fig.tight_layout()
    here = os.path.abspath(".")
    path_plots = 'plots/'
    print('This is for r =', r)
    file = path_plots + "energy_plot.pdf"
    print(f'Saving plot: file://{here}/{file}')
    fig.savefig(file)

    # plt.show()

plot_fid()
plot_energy_and_spectrum()