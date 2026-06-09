import matplotlib.pyplot as plt
import numpy as np
import arrayGeometry as ag
import beamforming_calculations as bf

# ============================================================
#        First declare your mic array, then call functions.
#        You can either write your custom array (as I've done here), or use a generator.
# ============================================================

mic_distances_mm = [(0, -20), (14.14, -14.14), (20, 0), (14.14, 14.14),
                   (0, 20), (-14.14, 14.14), (-20, 0), (-14.14, -14.14),
                    (-5, -8.66), (5, -8.66), (10, 0), (5, 8.66), (-5, 8.66), (-10, 0), (0,0)]

# Or use a generator...:
# mic_distances_mm = ag.circular_array(48, 25) # this is a circular array with 8 mics, radius 25mm, centered at (0,0), starting at 0 degrees
# mic_distances_mm = ag.linear_array(6, 10) # this is a linear array with 6 mics, spaced 10 mm apart
mic_distances_mm = ag.rectangular_array(4, 4, 22, 22) # this is a 3-row, 4-column rectangular array with 10 mm spacing in both directions
# mic_distances_mm = ag.concentric_rings([(1, 0, 0), (13, 100, 0), (14, 50, 0)]) # this is a concentric ring array with 1 mic at center, 8 mics in a ring of radius 20mm, and 6 mics in a ring of radius 10mm, all centered at (0,0)
# mic_distances_mm = ag.cross_array(20, 4, 5) #this is a cross array with arms of length 20mm, 4 mics per arm, spaced 5 mm apart, centered at (0,0)
# mic_distances_mm = ag.spiral_array(48, 4, 300) # this is a spiral array with 48 mics, 4 arms, and diameter of 300mm
# mic_distances_mm = ag.sunflower_array(48, 50) # this is a sunflower array with 48 mics, element spacing of 50mm    
# print(mic_distances_mm)

# Actual function calls. Remember to change both theta_degs in calculate_and_print and plot_microphone_array! 
# Otherwise, what you input into the DSP will not be what you see in the plot which can be confusing. 

bf.plot_beam_pattern(mic_distances_mm, speed_of_sound=343, sampling_rate=48000, theta_deg=60, phi_deg=32, freq=8000, resolution=100, method='DAS', h=16, w=6) # this will plot the beam pattern for the given array and parameters
# bf.plot_microphone_array_and_beam(mic_distances_mm, theta_deg=0, phi_deg=45, freq=4000)
# bf.calculate_and_print(mic_distances_mm, theta_deg=0, phi_deg=45)