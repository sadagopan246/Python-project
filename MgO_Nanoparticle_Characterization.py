# ============================================================
# XRD Analysis of MgO Nanoparticles
# Uses: numpy, pandas, matplotlib, scipy only
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: Load the CSV file
# ============================================================

# Read the CSV file into a pandas DataFrame
# Change this path to wherever your CSV file is saved
file_path = r"D:\SRM classes\7th Sem\Python programming\Project\MgO_XRD_sample.csv"

df = pd.read_csv(file_path)

# Show the first few rows so we can check the data loaded correctly
print("First 5 rows of the data:")
print(df.head())
print()
print("Column names found in the file:", df.columns.tolist())
print()

# ============================================================
# STEP 2: Extract the 2-theta and Intensity columns
# ============================================================

# The CSV has columns named "2Theta" and "Intensity"
# If your file has different column names, change these two lines:
angle_column = "2Theta"
intensity_column = "Intensity"

# ============================================================
# STEP 3: Clean the data
# Remove any rows that have missing or non-numeric values
# ============================================================

# Keep only rows where both columns have valid numbers
df = df[[angle_column, intensity_column]].dropna()

# Convert both columns to numbers (any bad values become NaN, then we drop them)
df[angle_column]    = pd.to_numeric(df[angle_column],    errors="coerce")
df[intensity_column] = pd.to_numeric(df[intensity_column], errors="coerce")
df = df.dropna()

# Extract as simple numpy arrays (easier to work with)
two_theta = df[angle_column].values
intensity  = df[intensity_column].values

print(f"Data loaded: {len(two_theta)} data points")
print(f"2-theta range: {two_theta.min():.2f}° to {two_theta.max():.2f}°")
print()

# ============================================================
# STEP 4: Smooth the data a little so peak-finding is more reliable
# We use a simple moving average (rolling mean)
# ============================================================

# Window size for smoothing – bigger = smoother but may merge close peaks
smooth_window = 5

# Smooth using a numpy convolution (same as rolling average)
smooth_intensity = np.convolve(
    intensity,
    np.ones(smooth_window) / smooth_window,
    mode="same"
)

# ============================================================
# STEP 5: Detect peaks using scipy find_peaks
#
# You can adjust these three numbers to tune peak detection:
#   height      – minimum intensity a peak must have (absolute)
#   prominence  – how much a peak must stand out above its surroundings
#   distance    – minimum gap (in data points) between two peaks
# ============================================================

height_threshold = np.mean(smooth_intensity) + 1.5 * np.std(smooth_intensity)
prominence_value = 30        # adjust if you miss peaks or get too many
distance_value   = 50        # minimum separation between peaks (data points)

def find_peaks_numpy(y, height=None, prominence=None, distance=1):
    """Simple NumPy replacement for scipy.signal.find_peaks."""
    y = np.asarray(y)
    n = len(y)

    # Candidate peaks: higher than both neighbours
    # (>= on the right handles flat tops)
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

    # Height filter
    if height is not None:
        cand = cand[y[cand] >= height]

    # Prominence filter
    if prominence is not None:
        keep = []
        for p in cand:
            # walk left until a higher point is found
            left_min = y[p]
            i = p - 1
            while i >= 0 and y[i] <= y[p]:
                left_min = min(left_min, y[i])
                i -= 1
            # walk right until a higher point is found
            right_min = y[p]
            j = p + 1
            while j < n and y[j] <= y[p]:
                right_min = min(right_min, y[j])
                j += 1
            prom = y[p] - max(left_min, right_min)
            if prom >= prominence:
                keep.append(p)
        cand = np.array(keep, dtype=int)

    # Distance filter: keep the tallest peaks first, drop close neighbours
    if distance is not None and distance > 1 and len(cand) > 0:
        order = cand[np.argsort(y[cand])[::-1]]
        selected = []
        for p in order:
            if all(abs(p - s) >= distance for s in selected):
                selected.append(p)
        cand = np.array(sorted(selected), dtype=int)

    return cand, {}

# find_peaks returns the INDEX positions of the peaks
peak_indices, peak_properties = find_peaks_numpy(
    smooth_intensity,
    height=height_threshold,
    prominence=prominence_value,
    distance=distance_value
)

# Get the actual 2-theta angles and intensities at those peak positions
peak_angles     = two_theta[peak_indices]
peak_intensities = intensity[peak_indices]

print(f"Number of peaks detected: {len(peak_indices)}")
print("Detected peak positions (2θ):", np.round(peak_angles, 2))
print()

# ============================================================
# STEP 6: Plot the full XRD diffraction pattern
#          and mark the detected peaks on it
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the raw XRD pattern
ax.plot(two_theta, intensity, color="steelblue", linewidth=0.8,
        label="XRD Pattern (raw)", alpha=0.7)

# Overlay the smoothed curve so we can see what peak-finding used
ax.plot(two_theta, smooth_intensity, color="navy", linewidth=1.2,
        label="Smoothed pattern", alpha=0.5)

# Mark each detected peak with a red triangle
ax.plot(peak_angles, peak_intensities + 20,
        "v", color="red", markersize=8, label="Detected peaks")

# Add a dashed vertical line and angle label at each peak
for angle, intens in zip(peak_angles, peak_intensities):
    ax.axvline(x=angle, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(angle + 0.2, intens + 25, f"{angle:.2f}°",
            rotation=90, fontsize=7, color="red")

# Labels, title, grid, legend
ax.set_xlabel("2θ (degrees)", fontsize=13)
ax.set_ylabel("Intensity (a.u.)", fontsize=13)
ax.set_title("XRD Diffraction Pattern of MgO Nanoparticles", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("MgO_XRD_pattern.png", dpi=150)
plt.show()
print("XRD plot saved as 'MgO_XRD_pattern.png'")
print()

# ============================================================
# STEP 7: Calculate FWHM for each peak from the raw data
#
# Method: for each peak we look at a small window around it,
# find the half-maximum level, and interpolate to find the
# two angles where intensity crosses that level (left and right).
# FWHM = right_angle – left_angle
# ============================================================

def calculate_fwhm(two_theta, intensity, peak_index, window=80):
    """
    Estimate the FWHM of a single peak.

    Parameters
    ----------
    two_theta   : 1-D array of 2-theta angles
    intensity   : 1-D array of raw intensities
    peak_index  : index of the peak in the arrays
    window      : number of data points to look at on each side

    Returns
    -------
    fwhm        : full width at half maximum in degrees
                  (returns None if the calculation fails)
    """

    # Define the region around the peak
    left_idx  = max(0, peak_index - window)
    right_idx = min(len(intensity) - 1, peak_index + window)

    # Slice out the local region
    local_angles    = two_theta[left_idx : right_idx + 1]
    local_intensity = intensity[left_idx : right_idx + 1]

    # The peak value is the maximum inside this window
    peak_value = local_intensity.max()

    # The baseline is the minimum intensity in the window
    baseline = local_intensity.min()

    # Half-maximum level (measured above the baseline)
    half_max = baseline + (peak_value - baseline) / 2.0

    # Find indices where intensity crosses the half-max level
    # We look for sign changes in (intensity – half_max)
    above_half = local_intensity >= half_max

    # Find the LEFT crossing (first True→False or start)
    left_cross = None
    for i in range(1, len(above_half)):
        if not above_half[i - 1] and above_half[i]:   # crosses upward
            # Linear interpolation between points i-1 and i
            x1, x2 = local_angles[i - 1], local_angles[i]
            y1, y2 = local_intensity[i - 1], local_intensity[i]
            left_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            break

    # Find the RIGHT crossing
    right_cross = None
    for i in range(len(above_half) - 1, 0, -1):
        if not above_half[i] and above_half[i - 1]:   # crosses downward
            x1, x2 = local_angles[i - 1], local_angles[i]
            y1, y2 = local_intensity[i - 1], local_intensity[i]
            right_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            break

    # Both crossings must be found for a valid FWHM
    if left_cross is not None and right_cross is not None:
        fwhm = right_cross - left_cross
        if fwhm > 0:
            return fwhm

    return None   # could not calculate

# Calculate FWHM for every detected peak
fwhm_list = []
for idx in peak_indices:
    fwhm = calculate_fwhm(two_theta, intensity, idx, window=80)
    fwhm_list.append(fwhm)

# ============================================================
# STEP 8: Scherrer equation to get crystallite size
#
#   D = (K * lambda) / (beta * cos(theta))
#
#   K      = 0.9  (Scherrer shape factor)
#   lambda = 0.15406 nm  (Cu Kα wavelength)
#   beta   = FWHM converted from degrees to radians
#   theta  = half of the measured 2θ angle, in radians
# ============================================================

K      = 0.9          # Scherrer shape factor
lam    = 0.15406      # Cu Kα wavelength in nm

# Lists to collect results for the summary table
results = []

print("Calculating crystallite size with the Scherrer equation ...")
print()

for i, (angle, intens, fwhm_deg) in enumerate(
        zip(peak_angles, peak_intensities, fwhm_list)):

    if fwhm_deg is None or fwhm_deg <= 0:
        # Skip this peak if FWHM could not be determined
        results.append({
            "Peak #"         : i + 1,
            "2θ (deg)"       : round(angle, 3),
            "Intensity"      : round(intens, 1),
            "FWHM (deg)"     : "N/A",
            "FWHM (rad)"     : "N/A",
            "θ (deg)"        : round(angle / 2, 3),
            "Size D (nm)"    : "N/A"
        })
        continue

    # Convert FWHM from degrees to radians
    fwhm_rad = np.deg2rad(fwhm_deg)

    # Bragg angle θ = half of 2θ, converted to radians
    theta_deg = angle / 2.0
    theta_rad = np.deg2rad(theta_deg)

    # Apply Scherrer equation
    D_nm = (K * lam) / (fwhm_rad * np.cos(theta_rad))

    results.append({
        "Peak #"         : i + 1,
        "2θ (deg)"       : round(angle, 3),
        "Intensity"      : round(intens, 1),
        "FWHM (deg)"     : round(fwhm_deg, 4),
        "FWHM (rad)"     : round(fwhm_rad, 6),
        "θ (deg)"        : round(theta_deg, 3),
        "Size D (nm)"    : round(D_nm, 2)
    })

# ============================================================
# STEP 9: Display results table
# ============================================================

results_df = pd.DataFrame(results)

print("=" * 75)
print("        Scherrer Crystallite Size Results for MgO Nanoparticles")
print("=" * 75)
print(results_df.to_string(index=False))
print()

# Calculate the average crystallite size (ignoring peaks where it is "N/A")
valid_sizes = [
    r["Size D (nm)"] for r in results
    if isinstance(r["Size D (nm)"], float)
]

if valid_sizes:
    average_size = np.mean(valid_sizes)
    print(f"Number of valid peaks used   : {len(valid_sizes)}")
    print(f"Average crystallite size (D) : {average_size:.2f} nm")
else:
    print("No valid crystallite sizes could be calculated.")

print()
print("Parameters used:")
print(f"  Shape factor K  = {K}")
print(f"  Wavelength λ    = {lam} nm  (Cu Kα)")
print()

# ============================================================
# STEP 10: Save the results table to a CSV file
# ============================================================

results_df.to_csv("MgO_Scherrer_results.csv", index=False)
print("Results table saved as 'MgO_Scherrer_results.csv'")



# ============================================================
# FTIR Analysis of MgO Nanoparticles
# Uses: numpy, pandas, matplotlib only
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: Load the CSV file
# ============================================================

# Change this to the name of your FTIR CSV file
file_path = r"D:\SRM classes\7th Sem\Python programming\Project\MgO_FTIR_digitized.csv"

# Folder where the graph and results will be saved
output_folder = r"D:\SRM classes\7th Sem\Python programming\Project"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Show the first few rows so we can check the data loaded correctly
print("First 5 rows of the data:")
print(df.head())
print()
print("Column names found in the file:", df.columns.tolist())
print()

# ============================================================
# STEP 2: Extract the wavenumber and transmittance columns
# ============================================================

# The first column is taken as wavenumber, the second as transmittance.
# If your columns are in a different order, change these two lines:
wave_column = df.columns[0]
trans_column = df.columns[1]

# ============================================================
# STEP 3: Clean the data
# ============================================================

# Keep only the two columns we need and remove empty rows
df = df[[wave_column, trans_column]].dropna()

# Convert both columns to numbers (bad values become NaN, then we drop them)
df[wave_column] = pd.to_numeric(df[wave_column], errors="coerce")
df[trans_column] = pd.to_numeric(df[trans_column], errors="coerce")
df = df.dropna()

# Transmittance must be greater than zero (needed for the log calculation)
df = df[df[trans_column] > 0]

# Extract as simple numpy arrays
wavenumber = df[wave_column].values
transmittance = df[trans_column].values

# If transmittance is in percent (0-100), convert it to a fraction (0-1)
if transmittance.max() > 1.5:
    transmittance = transmittance / 100

# Sort the data by wavenumber
order = np.argsort(wavenumber)
wavenumber = wavenumber[order]
transmittance = transmittance[order]

print(f"Data loaded: {len(wavenumber)} data points")
print(f"Wavenumber range: {wavenumber.min():.2f} to {wavenumber.max():.2f} cm-1")
print()

# ============================================================
# STEP 4: Calculate percent transmittance and absorbance
#   Absorbance A = -log10(T)
# ============================================================

percent_T = transmittance * 100
absorbance = -np.log10(transmittance)

# ============================================================
# STEP 5: Define the bands we want to analyse
# ============================================================

# Search window (cm-1) on each side of the expected band position
window = 60

# (band position, assignment)
bands = [
    (3640, "O-H stretching (adsorbed water)"),
    (2350, "Atmospheric CO2"),
    (1438, "Adsorbed carbonates"),
    (859,  "Mg-O-Mg stretching"),
]

# The broad Mg-O stretching region
mgo_low = 500
mgo_high = 650

# ============================================================
# STEP 6: Find the band values from the data
# For each band we look inside the window and find the point
# with the lowest transmittance (the strongest absorption)
# ============================================================

results = []

for position, name in bands:
    # Index of the data point closest to the expected band position
    near_index = np.argmin(np.abs(wavenumber - position))

    # Select the data inside the search window
    mask = (wavenumber >= position - window) & (wavenumber <= position + window)
    local_wave = wavenumber[mask]
    local_T = transmittance[mask]

    # If there is no data in this window, skip the band
    if len(local_wave) == 0:
        print(f"No data found near {position} cm-1, skipping this band.")
        continue

    # Lowest transmittance inside the window = band minimum
    min_index = np.argmin(local_T)
    min_wave = local_wave[min_index]
    min_T = local_T[min_index]

    results.append({
        "Assignment": name,
        "Band (cm-1)": position,
        "Transmittance (%)": round(percent_T[near_index], 2),
        "Absorbance": round(absorbance[near_index], 4),
        "Minimum at (cm-1)": round(min_wave, 2),
        "Absorbance at Min": round(-np.log10(min_T), 4)
    })

# Now the broad Mg-O stretching region (500-650 cm-1)
mask = (wavenumber >= mgo_low) & (wavenumber <= mgo_high)
if mask.sum() > 0:
    local_wave = wavenumber[mask]
    local_T = transmittance[mask]
    min_index = np.argmin(local_T)

    results.append({
        "Assignment": "Mg-O stretching",
        "Band (cm-1)": "500-650",
        "Transmittance (%)": round(local_T[min_index] * 100, 2),
        "Absorbance": round(-np.log10(local_T[min_index]), 4),
        "Minimum at (cm-1)": round(local_wave[min_index], 2),
        "Absorbance at Min": round(-np.log10(local_T[min_index]), 4)
    })

# ============================================================
# STEP 7: Display the results table
# ============================================================

results_df = pd.DataFrame(results)

print("=" * 95)
print("                 FTIR Results for MgO Nanoparticles")
print("=" * 95)
print(results_df.to_string(index=False))
print()
print(f"Number of bands analysed : {len(results_df)}")
print()
print("Parameters used:")
print(f"  Search window = ±{window} cm-1")
print(f"  Mg-O region   = {mgo_low}-{mgo_high} cm-1")
print()

# ============================================================
# STEP 8: Plot the FTIR spectrum
# Top graph: Transmittance    Bottom graph: Absorbance
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Plot the spectra
ax1.plot(wavenumber, transmittance, color="dimgray", linewidth=1.2,
         label="MgO Transmittance")
ax2.plot(wavenumber, absorbance, color="darkred", linewidth=1.2,
         label="MgO Absorbance")

# Mark each band with a dashed line and a label
for position, name in bands:
    for ax in (ax1, ax2):
        ax.axvline(x=position, color="red", linestyle="--",
                   linewidth=0.7, alpha=0.6)
    ax1.text(position + 20, transmittance.min(), f"{name}\n{position} cm-1",
             rotation=90, fontsize=7, color="red", va="bottom")

# Shade the Mg-O stretching region
for ax in (ax1, ax2):
    ax.axvspan(mgo_low, mgo_high, color="violet", alpha=0.3)
ax1.text(575, transmittance.max(), "Mg-O stretching", rotation=90,
         ha="center", va="top", fontsize=9)

# FTIR spectra are normally drawn with high wavenumber on the left
ax1.set_xlim(4000, 500)

# Labels, title, grid, legend
ax1.set_ylabel("Transmittance", fontsize=12)
ax1.set_title("FTIR Spectrum of MgO Nanoparticles", fontsize=14)
ax2.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
ax2.set_ylabel("Absorbance  A = -log$_{10}$(T)", fontsize=12)
ax1.legend(fontsize=10)
ax2.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.4)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

# Save the graph first, then show it on screen
plt.savefig(output_folder + r"\MgO_FTIR_Spectrum.png", dpi=150)
plt.show()
print("FTIR plot saved as 'MgO_FTIR_Spectrum.png'")
print()

# ============================================================
# STEP 9: Save the results table to a CSV file
# ============================================================

results_df.to_csv(output_folder + r"\MgO_FTIR_Results.csv", index=False)
print("Results table saved as 'MgO_FTIR_Results.csv'")
print(f"Both files are in: {output_folder}")
