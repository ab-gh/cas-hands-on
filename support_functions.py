'''
Support functions for Longitudinal hands on exercises on Tracking

The list of function to use is

- plot_phase_space_trajectory
- plot_phase_space_distribution
- synchrotron_tune
- separatrix
- run_animation
- oscillation_spectrum
- synchrotron_tune

'''

import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML


def plot_phase_space_trajectory(phase_trajectory,
                                energy_trajectory,
                                phase_sep=None,
                                separatrix_array=None,
                                figname=None,
                                draw_line=True,
                                xlim=None, ylim=None):
    """
    This support function can be used to plot the trajectory of a particle
    in the longitudinal phase space. The o marker is the starting coordinate,
    the * marker is the end coordinate.

    Parameters
    ----------
    phase_trajectory : np.array or list
        The phase coordinates of a distribution of particles in [rad]
    energy_trajectory : np.array or list
        The energy coordinates of a distribution of particles in [eV]
    phase_sep : np.array
        The phase of the separatrix array in [rad], as output by the separatrix function
    separatrix_array : np.array
        The separatrix array in [eV], as output by the separatrix function
    figname : str, optional
        The name of your nice figure, by default None
    draw_line : bool, optional
        Hide the trajectory to plot only the start/end coordinates, by default True
    xlim : tuple, optional
        The limits in phase for your nice plot in [rad], e.g. (-np.pi, np.pi), by default None
    ylim : tuple, optional
        The limits in energy for your nice plot [eV], e.g. (-1e6, 1e6), by default None
    """

    phase_trajectory = np.array(phase_trajectory)
    energy_trajectory = np.array(energy_trajectory) / 1e6
    plt.figure(figname, figsize=(8, 8))
    if figname is None:
        plt.clf()
    if draw_line:
        alpha = 1
    else:
        alpha = 0
    if phase_trajectory.ndim == 1:
        p = plt.plot(phase_trajectory, energy_trajectory,
                     alpha=alpha)
        plt.plot(phase_trajectory[0], energy_trajectory[0],
                 'o', color=p[0].get_color())
        plt.plot(phase_trajectory[-1], energy_trajectory[-1],
                 '*', color=p[0].get_color())
    else:
        for idx_part in range(phase_trajectory.shape[1]):
            p = plt.plot(phase_trajectory[:, idx_part],
                         energy_trajectory[:, idx_part],
                         alpha=alpha)
            plt.plot(phase_trajectory[0, idx_part],
                     energy_trajectory[0, idx_part],
                     'o', color=p[0].get_color())
            plt.plot(phase_trajectory[-1, idx_part],
                     energy_trajectory[-1, idx_part],
                     '*', color=p[0].get_color())

    if (phase_sep is not None) and (separatrix_array is not None):
        plt.plot(phase_sep, separatrix_array / 1e6, 'g:')

    plt.xlabel('Phase $\\phi$ [rad]')
    plt.ylabel('Energy $\\Delta E$ [MeV]')
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_phase_space_distribution(
        phase_coordinates,
        energy_coordinates,
        phase_sep=None,
        separatrix_array=None,
        figname=None,
        xbins=50, ybins=50,
        xlim=None, ylim=None):
    """
    Plot a distribution of particles
    in the longitudinal phase space, and see the longitudinal profiles
    in phase and energy.

    Parameters
    ----------
    phase_coordinates : np.array or list
        The phase coordinates of a distribution of particles in [rad]
    energy_coordinates : np.array or list
        The energy coordinates of a distribution of particles in [eV]
    phase_sep : np.array
        The phase of the separatrix array in [rad], as output by the separatrix function
    separatrix_array : np.array
        The separatrix array in [eV], as output by the separatrix function
    figname : str, optional
        The name of your nice figure, by default None
    xbins : int, optional
        The number of bins to generate a nice longitudinal profile in phase, by default 50
    ybins : int, optional
        The number of bins to generate a nice longitudinal profile in energy, by default 50
    xlim : tuple, optional
        The limits in phase for your nice plot in [rad], e.g. (-np.pi, np.pi), by default None
    ylim : tuple, optional
        The limits in energy for your nice plot [eV], e.g. (-1, 1), by default None
    """

    plt.figure(figname, figsize=(8, 8))
    plt.clf()
    # Definitions for placing the axes
    left, width = 0.115, 0.63
    bottom, height = 0.115, 0.63
    bottom_h = left_h = left + width + 0.03

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    # rect_txtBox= [left_h, bottom_h, 0.2, 0.2]

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axScatter = plt.axes(rect_scatter)

    # txtBox = plt.axes(rect_txtBox)

    global distri_plot
    distri_plot, = axScatter.plot(
        phase_coordinates, energy_coordinates, 'o', alpha=0.5)

    if (phase_sep is not None) and (separatrix_array is not None):
        axScatter.plot(phase_sep, separatrix_array, 'r')

    start_xlim = axScatter.get_xlim()
    start_ylim = axScatter.get_ylim()

    hist_phase = np.histogram(phase_coordinates, xbins, range=xlim)
    global line_phase
    line_phase, = axHistx.plot(hist_phase[1][0:-1] + (
        hist_phase[1][1] - hist_phase[1][0]) / 2, hist_phase[0] / np.max(hist_phase[0]))
    axHistx.axes.get_xaxis().set_ticklabels([])
    axHistx.axes.get_yaxis().set_ticklabels([])
    axHistx.set_xlim(start_xlim)
    # axHistx.set_ylim(start_ylim)
    axHistx.set_ylabel('Bunch profile $\\lambda_{\\phi}$')

    hist_energy = np.histogram(energy_coordinates, ybins, range=ylim)
    global line_energy
    line_energy, = axHisty.plot(hist_energy[0] / np.max(
        hist_energy[0]), hist_energy[1][0:-1] + (hist_energy[1][1] - hist_energy[1][0]) / 2)
    axHisty.axes.get_xaxis().set_ticklabels([])
    axHisty.axes.get_yaxis().set_ticklabels([])
    axHisty.set_ylim(start_ylim)
    axHisty.set_xlabel('Energy spread $\\lambda_{\\Delta E}$')

    axScatter.set_xlabel('Phase $\\phi$ [rad]')
    axScatter.set_ylabel('Energy $\\Delta E$ [eV]')
    plt.xlim(xlim)
    plt.ylim(ylim)


def separatrix(phase_array, f_rev, eta, beta, energy, charge, voltage, harmonic, acceleration=0):
    """Return the separatrix as an array for plotting purposes (together with the corresponding
    phase values).

    Parameters
    ----------
    phase_array : np.array
        The input phase array in [rad]
    f_rev : float
        The revolution frequency in [Hz]
    eta : float
        THe phase slippage factor
    beta : float
        The relativistic beta
    energy : float
        The beam total energy in [eV]
    charge : float
        The particle charge in [e]
    voltage : float
        The rf voltage in [V]
    harmonic : float
        The rf harmonic number
    acceleration : float
        The beam energy gain per turn in [eV]

    Returns
    -------
    phase_sep, separatrix_array : np.array
        The corresponding phase and sepatrix values
    """

    warnings.filterwarnings("once")

    if eta > 0:
        phi_s = np.pi - np.arcsin(acceleration / charge / voltage)
        if acceleration > 0:
            phi_ufp = (np.pi - phi_s)
        else:
            phi_ufp = 2 * np.pi + (np.pi - phi_s)
    else:
        phi_s = np.arcsin(acceleration / charge / voltage)
        if acceleration > 0:
            phi_ufp = np.pi - phi_s
        else:
            phi_ufp = -np.pi - phi_s

    def pot_well(phi):
        return -(np.cos(phi) + phi * np.sin(phi_s)) / np.cos(phi_s)

    f_s0 = np.sqrt(
        -(2 * np.pi * f_rev)**2 * harmonic * eta * charge * voltage * np.cos(phi_s) /
        (2 * np.pi * beta**2 * energy)) / (2 * np.pi)

    sync_tune = f_s0 / f_rev

    sep_fac = sync_tune * beta**2 / (harmonic * np.abs(eta)) * energy
    separatrix_array = np.sqrt(
        2 * (pot_well(phi_ufp) - pot_well(phase_array))) * sep_fac

    separatrix_array = np.append(separatrix_array, -separatrix_array[::-1])
    phase_sep = np.append(phase_array, phase_array[::-1])

    return phase_sep[np.isfinite(separatrix_array)], separatrix_array[np.isfinite(separatrix_array)]


def generate_bunch(bunch_position, bunch_length,
                   bunch_energy, energy_spread,
                   n_macroparticles):
    """Generate a nice bunch of particles distributed as a parabola in phase space.

    Parameters
    ----------
    bunch_position : float
        The position in phase [rad] of the center of mass of the bunch
    bunch_length : float
        The length in phase [rad] of the bunch
    bunch_energy : float
        The position in energy [eV] of the center of mass of the bunch
        (relative to the synchronous energy)
    energy_spread : float
        The spread in energy [eV] of the bunch
    n_macroparticles : int
        The number of macroparticles to generate.

    Returns
    -------
    particle_phase : np.array
        The distribution of macroparticles in phase (rad)
    particle_energy : np.array
        The distribution of particles in energy (eV)
    """

    # Generating phase and energy arrays
    phase_array = np.linspace(bunch_position - bunch_length / 2,
                              bunch_position + bunch_length / 2,
                              100)
    energy_array = np.linspace(bunch_energy - energy_spread / 2,
                               bunch_energy + energy_spread / 2,
                               100)

    # Getting Hamiltonian on a grid
    phase_grid, deltaE_grid = np.meshgrid(
        phase_array, energy_array)

    # Bin sizes
    bin_phase = phase_array[1] - phase_array[0]
    bin_energy = energy_array[1] - energy_array[0]

    # Density grid
    isodensity_lines = ((phase_grid - bunch_position) / bunch_length * 2)**2. + \
        ((deltaE_grid - bunch_energy) / energy_spread * 2)**2.
    density_grid = 1 - isodensity_lines**2.
    density_grid[density_grid < 0] = 0
    density_grid /= np.sum(density_grid)

    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(np.arange(0, np.size(density_grid)),
                               n_macroparticles, p=density_grid.flatten())

    # Randomize particles inside each grid cell (uniform distribution)
    particle_phase = (np.ascontiguousarray(
        phase_grid.flatten()[indexes] + (np.random.rand(n_macroparticles) - 0.5) * bin_phase))
    particle_energy = (np.ascontiguousarray(
        deltaE_grid.flatten()[indexes] + (np.random.rand(n_macroparticles) - 0.5) * bin_energy))

    return particle_phase, particle_energy


def oscillation_spectrum(phase_track, fft_zero_padding=0):
    """Compute the spectrum of a particle phase oscillation by applying an fft.

    Parameters
    ----------
    phase_track : np.array or list
        The phase oscillations of a particle in rad, over a few synchrotron periods.
    fft_zero_padding : int, optional
        The number of points for zero padding, to get a nice spectrum, by default 0

    Returns
    -------
    freq_array : np.array
        The frequency array of the spectrum
    fft_osc : np.array
        The amplitude of the phase oscillation spectrum
    """
    n_turns = len(phase_track)

    freq_array = np.fft.rfftfreq(n_turns + fft_zero_padding)
    fft_osc = np.abs(
        np.fft.rfft(
            phase_track - np.mean(phase_track),
            n_turns + fft_zero_padding) * 2 / (n_turns))

    return freq_array, fft_osc


def synchrotron_tune(phase_track, fft_zero_padding=0):
    """Compute the synchrotron tune from a particle phase oscillations.

    Parameters
    ----------
    phase_track : np.array or list
        The phase oscillations of a particle in rad, over a few synchrotron periods.
    fft_zero_padding : int, optional
        The number of points for zero padding, to get a nice spectrum, by default 0

    Returns
    -------
    oscillation_amplitude : float
        The amplitude of the particle phase oscillation in rad
    sync_tune : float
        The synchrotron tune of the particle
    """
    freq_array, spectrum_array = oscillation_spectrum(
        phase_track, fft_zero_padding=fft_zero_padding)

    oscillation_amplitude = np.max(spectrum_array)
    sync_tune = float(
        np.mean(freq_array[spectrum_array == oscillation_amplitude]))

    return oscillation_amplitude, sync_tune


class _TrackAnimation(object):

    def __init__(
            self, phase_coordinates, energy_coordinates,
            drift_function, rf_kick_function,
            drift_args, rf_kick_args,
            figname, iterations, framerate,
            xbins=50, ybins=50, xlim=None, ylim=None,
            phase_sep=None, separatrix_array=None):

        self.phase_coordinates = phase_coordinates
        self.energy_coordinates = energy_coordinates
        self.drift_function = drift_function
        self.rf_kick_function = rf_kick_function
        self.drift_args = drift_args
        self.rf_kick_args = rf_kick_args
        self.figname = figname
        self.iterations = iterations
        self.framerate = framerate
        self.xbins = xbins
        self.ybins = ybins
        self.xlim = xlim
        self.ylim = ylim
        self.phase_sep = phase_sep
        self.separatrix_array = separatrix_array

    def run_animation(self):

        self._init()
        anim = animation.FuncAnimation(
            self.anim_fig, self._animate, init_func=self._init,
            frames=self.iterations, interval=1000 / self.framerate, blit=True)
        return HTML(anim.to_jshtml())

    def _init(self):

        plot_phase_space_distribution(self.phase_coordinates,
                                      self.energy_coordinates,
                                      figname=self.figname,
                                      xbins=self.xbins, ybins=self.ybins,
                                      xlim=self.xlim, ylim=self.ylim,
                                      phase_sep=self.phase_sep,
                                      separatrix_array=self.separatrix_array)

        self.anim_fig = plt.gcf()

        return (line_phase, line_energy, distri_plot)

    def _animate(self, i):

        self.phase_coordinates = self.drift_function(
            self.phase_coordinates,
            self.energy_coordinates, *self.drift_args)
        self.energy_coordinates = self.rf_kick_function(
            self.energy_coordinates,
            self.phase_coordinates, *self.rf_kick_args)

        hist_phase = np.histogram(
            self.phase_coordinates, self.xbins, range=self.xlim)
        line_phase.set_data(hist_phase[1][0:-1] + (hist_phase[1][1] -
                                                   hist_phase[1][0]) / 2, hist_phase[0] / np.max(hist_phase[0]))

        hist_energy = np.histogram(
            self.energy_coordinates, self.ybins, range=self.ylim)
        line_energy.set_data(hist_energy[0] / np.max(hist_energy[0]),
                             hist_energy[1][0:-1] + (hist_energy[1][1] - hist_energy[1][0]) / 2)

        distri_plot.set_data(self.phase_coordinates, self.energy_coordinates)

        return (line_phase, line_energy, distri_plot)


def run_animation(phase_coordinates,
                  energy_coordinates,
                  drift_function, rf_kick_function,
                  drift_args, rf_kick_args,
                  figname, iterations, framerate,
                  phase_sep=None, separatrix_array=None,
                  xbins=50, ybins=50, xlim=None, ylim=None):
    """A routine to animate the motion of particles in the
    longitudinal phase space based on your own tracking equations.

    Parameters
    ----------
    phase_trajectory : np.array or list
        The phase coordinates of a distribution of particles in [rad]
    energy_trajectory : np.array or list
        The energy coordinates of a distribution of particles in [eV]
    drift_function : function
        Your drift equation of motion with the syntax
        phase_coordinates = drift_function(phase_coordinates, energy_coordinates, *drift_args)
        where drift_args is a list of arguments for the drift equation of motion (e.g. slippage)
    rf_kick_function : function
        Your kick equation of motion with the syntax
        energy_coordinates = kick_function(energy_coordinates, phase_coordinates, *kick_args)
        where kick_args is a list of arguments for the drift equation of motion (e.g. rf voltage)
    figname : str
        The name of your nice animation
    iterations : int
        The number of iterations for the tracking (i.e. number of turns)
    framerate : float
        The framerate of the animation (e.g. 30 fps)
    phase_sep : np.array
        The phase of the separatrix array in [rad], as output by the separatrix function
    separatrix_array : np.array
        The separatrix array in [eV], as output by the separatrix function
    xbins : int, optional
        The number of bins to generate a nice longitudinal profile in phase, by default 50
    ybins : int, optional
        The number of bins to generate a nice longitudinal profile in energy, by default 50
    xlim : tuple, optional
        The limits in phase for your nice plot in [rad], e.g. (-np.pi, np.pi), by default None
    ylim : tuple, optional
        The limits in energy for your nice plot [eV], e.g. (-1e6, 1e6), by default None
    """
    trackanim = _TrackAnimation(
        phase_coordinates,
        energy_coordinates,
        drift_function, rf_kick_function,
        drift_args, rf_kick_args,
        figname, iterations, framerate,
        phase_sep=phase_sep, separatrix_array=separatrix_array,
        xbins=xbins, ybins=ybins,
        xlim=xlim, ylim=ylim)

    return trackanim.run_animation()
