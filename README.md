## malaria-resistance-mapping
#Insecticide resistance mapping using MalariaGEN Ag3 data with climate correlation analysis

##Step 1:Installing malaraiagen_data

!pip install malariagen_data -q


 ##Step 2: Importing the data
 
 import malariagen_data
 
ag3 = malariagen_data.Ag3()

print("Connected ✓")
print(f"Version: {malariagen_data.__version__}")


##STEP 3: Load Sample Metadata

import pandas as pd

# Load all sample metadata from MalariaGEN Ag3
samples = ag3.sample_metadata()

print(f"Total samples: {len(samples):,}")
print(f"Columns: {list(samples.columns)}")
print(f"\nCountries: {sorted(samples['country'].unique())}")
print(f"\nYears: {samples['year'].min()} - {samples['year'].max()}")
print(f"\nFirst 5 rows:")
samples.head()


# STEP 4: Clean Data + Load kdr Resistance Mutations

import numpy as np

# --- Clean sample metadata ---
# Remove lab crosses and invalid years
df = samples.copy()
df = df[df['country'] != 'Lab Cross']
df = df[df['year'] > 0]
df = df.dropna(subset=['latitude', 'longitude'])

print(f"Clean samples: {len(df):,}")
print(f"Countries: {df['country'].nunique()}")
print(f"Years: {df['year'].min()} - {df['year'].max()}")

# --- Species breakdown ---
print(f"\nSpecies (taxon):")
print(df['taxon'].value_counts())

# --- Sample counts per country ---
print(f"\nSamples per country:")
print(df['country'].value_counts().to_string())


##STEP 5: Load kdr Resistance Allele Frequencies


# Load Ag3 cohort diversity - this gives us resistance allele data
# kdr mutations: Vgsc L995F (kdr-west) and L995S (kdr-east)
# These are the primary pyrethroid/DDT target-site resistance mutations

print("Loading kdr SNP data...")
print("This may take 2-3 minutes on first load...")

# Get SNP allele frequencies for the kdr locus
# Vgsc gene is on chromosome 2L at position ~2,358,158
snp_data = ag3.snp_allele_frequencies(
    transcript="AGAP004707-RD",  # Vgsc / kdr gene
    cohorts="admin1_year",
    min_cohort_size=10,
    site_mask="gamb_colu",
    sample_sets="3.0"
)

print("kdr data loaded ✓")
print(f"Shape: {snp_data.shape}")
snp_data.head()


##STEP 6: Extract kdr Mutations + Build Mapping Dataset


# Reset index to work with the data easily
snp_df = snp_data.reset_index()

# Extract the two key kdr mutations
# L995F = kdr-west (pyrethroid/DDT resistance, dominant West Africa)
# L995S = kdr-east (pyrethroid/DDT resistance, dominant East Africa)
kdr_west = snp_df[snp_df['aa_change'] == 'L995F'].copy()
kdr_east = snp_df[snp_df['aa_change'] == 'L995S'].copy()

print(f"kdr-west (L995F) variants found: {len(kdr_west)}")
print(f"kdr-east (L995S) variants found: {len(kdr_east)}")

# Get frequency columns (one per cohort)
freq_cols = [c for c in snp_df.columns if c.startswith('frq_')]
print(f"\nNumber of cohorts: {len(freq_cols)}")
print(f"Example cohorts: {freq_cols[:5]}")


##STEP 7: Reshape into Mappable Format


def extract_kdr_by_cohort(kdr_row, mutation_name):
    """Extract frequency per cohort and parse country/year from cohort name"""
    records = []
    for col in freq_cols:
        freq = kdr_row[col].values[0] if len(kdr_row) > 0 else None
        if freq is not None and not pd.isna(freq):
            # Parse cohort name: frq_GH-01_gamb_2014 → country=GH, year=2014
            parts = col.replace('frq_', '').split('_')
            country_iso = parts[0]         # e.g. GH
            year = parts[-1]               # e.g. 2014
            taxon = '_'.join(parts[1:-1])  # e.g. gamb
            records.append({
                'cohort': col.replace('frq_', ''),
                'country_iso': country_iso,
                'year': int(year),
                'taxon': taxon,
                'mutation': mutation_name,
                'kdr_frequency': float(freq)
            })
    return records

# Extract records for both mutations
records = []
if len(kdr_west) > 0:
    records += extract_kdr_by_cohort(kdr_west, 'kdr-west (L995F)')
if len(kdr_east) > 0:
    records += extract_kdr_by_cohort(kdr_east, 'kdr-east (L995S)')

kdr_map = pd.DataFrame(records)

# Map ISO codes to full country names
iso_to_country = {
    'GH': 'Ghana', 'BF': 'Burkina Faso', 'ML': 'Mali', 'GM': 'Gambia',
    'UG': 'Uganda', 'TZ': 'Tanzania', 'ET': 'Ethiopia', 'CM': 'Cameroon',
    'NG': 'Nigeria', 'CD': 'DR Congo', 'BJ': 'Benin', 'SN': 'Senegal',
    'KE': 'Kenya', 'ZA': 'South Africa', 'CI': 'Cote d\'Ivoire',
    'GW': 'Guinea-Bissau', 'GN': 'Guinea', 'GA': 'Gabon', 'ZW': 'Zimbabwe',
    'ZM': 'Zambia', 'TG': 'Togo', 'MW': 'Malawi', 'SS': 'South Sudan',
    'AO': 'Angola', 'MZ': 'Mozambique', 'CF': 'Central African Republic'
}
kdr_map['country'] = kdr_map['country_iso'].map(iso_to_country)

print(f"Total cohort-mutation records: {len(kdr_map)}")
print(f"\nkdr frequency summary:")
print(kdr_map.groupby('mutation')['kdr_frequency'].describe().round(3))
print(f"\nSample records:")
print(kdr_map[kdr_map['kdr_frequency'] > 0].head(10).to_string())

# ============================================================
# STEP 8: Fix Country Mapping + Add Coordinates
# ============================================================

# Fix ISO parsing - country code is first 2 chars only
kdr_map['country_iso2'] = kdr_map['country_iso'].str[:2]

iso_to_country = {
    'GH': 'Ghana', 'BF': 'Burkina Faso', 'ML': 'Mali', 'GM': 'Gambia',
    'UG': 'Uganda', 'TZ': 'Tanzania', 'ET': 'Ethiopia', 'CM': 'Cameroon',
    'NG': 'Nigeria', 'CD': 'DR Congo', 'BJ': 'Benin', 'SN': 'Senegal',
    'KE': 'Kenya', 'ZA': 'South Africa', 'CI': "Cote d'Ivoire",
    'GW': 'Guinea-Bissau', 'GN': 'Guinea', 'GA': 'Gabon', 'ZW': 'Zimbabwe',
    'ZM': 'Zambia', 'TG': 'Togo', 'MW': 'Malawi', 'SS': 'South Sudan',
    'AO': 'Angola', 'MZ': 'Mozambique', 'CF': 'Central African Republic',
    'GQ': 'Equatorial Guinea', 'ST': 'Sao Tome', 'KM': 'Comoros'
}

kdr_map['country'] = kdr_map['country_iso2'].map(iso_to_country)

# Add country centroid coordinates for mapping
country_coords = {
    'Ghana': (7.95, -1.02), 'Burkina Faso': (12.36, -1.53),
    'Mali': (17.57, -3.99), 'Gambia': (13.44, -15.31),
    'Uganda': (1.37, 32.29), 'Tanzania': (-6.37, 34.89),
    'Ethiopia': (9.15, 40.49), 'Cameroon': (3.85, 11.50),
    'Nigeria': (9.08, 8.67), 'DR Congo': (-4.03, 21.76),
    'Benin': (9.31, 2.32), 'Senegal': (14.50, -14.45),
    'Kenya': (-0.02, 37.91), 'South Africa': (-30.56, 22.94),
    "Cote d'Ivoire": (7.54, -5.55), 'Guinea-Bissau': (11.80, -15.18),
    'Guinea': (11.74, -15.68), 'Gabon': (-0.80, 11.61),
    'Zimbabwe': (-19.02, 29.15), 'Zambia': (-13.13, 27.85),
    'Togo': (8.62, 0.82), 'Malawi': (-13.25, 34.30),
    'South Sudan': (6.88, 31.57), 'Angola': (-11.20, 17.87),
    'Mozambique': (-18.67, 35.53), 'Central African Republic': (6.61, 20.94),
    'Equatorial Guinea': (1.65, 10.27), 'Sao Tome': (0.19, 6.61),
    'Comoros': (-11.64, 43.33)
}

kdr_map['latitude'] = kdr_map['country'].map(lambda x: country_coords.get(x, (np.nan, np.nan))[0])
kdr_map['longitude'] = kdr_map['country'].map(lambda x: country_coords.get(x, (np.nan, np.nan))[1])

# Also merge with sample-level coordinates for precise locations
sample_coords = df.groupby('country')[['latitude','longitude']].mean().reset_index()
sample_coords.columns = ['country', 'lat_precise', 'lon_precise']
kdr_map = kdr_map.merge(sample_coords, on='country', how='left')

# Use precise coords where available
kdr_map['latitude'] = kdr_map['lat_precise'].fillna(kdr_map['latitude'])
kdr_map['longitude'] = kdr_map['lon_precise'].fillna(kdr_map['longitude'])
kdr_map = kdr_map.drop(columns=['lat_precise','lon_precise'])

print(f"Records with country mapped: {kdr_map['country'].notna().sum()} / {len(kdr_map)}")
print(f"\nkdr-west frequency by country:")
west = kdr_map[kdr_map['mutation']=='kdr-west (L995F)'][['country','year','kdr_frequency']].dropna()
print(west.sort_values('kdr_frequency', ascending=False).to_string())

# ============================================================
# PHASE 2: Download WorldClim Climate Data
# ============================================================
import requests
import zipfile
import os
import numpy as np
import pandas as pd

os.makedirs('/content/climate', exist_ok=True)

# Download WorldClim 2.1 mean temperature (10 min resolution - small file)
print("Downloading WorldClim temperature data...")
url_temp = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip"
r = requests.get(url_temp, stream=True)
total = 0
with open('/content/climate/tavg.zip', 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
        total += len(chunk)
print(f"Downloaded {total/1024/1024:.1f} MB ✓")

# Download WorldClim 2.1 precipitation
print("Downloading WorldClim rainfall data...")
url_prec = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_prec.zip"
r2 = requests.get(url_prec, stream=True)
total2 = 0
with open('/content/climate/prec.zip', 'wb') as f:
    for chunk in r2.iter_content(chunk_size=8192):
        f.write(chunk)
        total2 += len(chunk)
print(f"Downloaded {total2/1024/1024:.1f} MB ✓")

print("\nExtracting...")
with zipfile.ZipFile('/content/climate/tavg.zip', 'r') as z:
    z.extractall('/content/climate/tavg/')
with zipfile.ZipFile('/content/climate/prec.zip', 'r') as z:
    z.extractall('/content/climate/prec/')

print("Climate data ready ✓")
print(f"Temperature files: {os.listdir('/content/climate/tavg/')[:3]}")
print(f"Rainfall files: {os.listdir('/content/climate/prec/')[:3]}")

# ============================================================
# PHASE 2B: Extract Climate Values at Sample Locations
# ============================================================
import rasterio
import numpy as np
import pandas as pd

# Install rasterio if needed
try:
    import rasterio
except:
    !pip install rasterio -q
    import rasterio

# Load all 12 monthly temperature rasters and compute annual mean
print("Extracting temperature values at sample locations...")

tavg_files = sorted([f'/content/climate/tavg/{f}'
                     for f in os.listdir('/content/climate/tavg/')
                     if f.endswith('.tif')])
prec_files = sorted([f'/content/climate/prec/{f}'
                     for f in os.listdir('/content/climate/prec/')
                     if f.endswith('.tif')])

print(f"Temperature rasters: {len(tavg_files)}")
print(f"Rainfall rasters: {len(prec_files)}")

def extract_raster_values(raster_path, lats, lons):
    """Extract raster values at given lat/lon coordinates"""
    values = []
    with rasterio.open(raster_path) as src:
        for lat, lon in zip(lats, lons):
            try:
                row, col = src.index(lon, lat)
                val = src.read(1)[row, col]
                # NoData check
                if val == src.nodata or val < -100:
                    values.append(np.nan)
                else:
                    values.append(float(val))
            except:
                values.append(np.nan)
    return values

# Use kdr_map coordinates
lats = kdr_map['latitude'].values
lons = kdr_map['longitude'].values

# Extract monthly temps and average
monthly_temps = []
for f in tavg_files:
    vals = extract_raster_values(f, lats, lons)
    monthly_temps.append(vals)

kdr_map['mean_temp_c'] = np.nanmean(monthly_temps, axis=0)

# Extract monthly rainfall and sum for annual total
monthly_prec = []
for f in prec_files:
    vals = extract_raster_values(f, lats, lons)
    monthly_prec.append(vals)

kdr_map['annual_rainfall_mm'] = np.nansum(monthly_prec, axis=0)

print("\nClimate extraction complete ✓")
print(f"\nTemperature range: {kdr_map['mean_temp_c'].min():.1f}°C - {kdr_map['mean_temp_c'].max():.1f}°C")
print(f"Rainfall range: {kdr_map['annual_rainfall_mm'].min():.0f}mm - {kdr_map['annual_rainfall_mm'].max():.0f}mm")
print(f"\nSample of enriched dataset:")
print(kdr_map[['country','year','mutation','kdr_frequency','mean_temp_c','annual_rainfall_mm']].dropna().head(10).to_string())


# ============================================================
# PHASE 3: ANALYSIS CHARTS
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import numpy as np
import os

os.makedirs('/content/malaria-resistance-mapping/outputs/charts', exist_ok=True)

# Clean data for plotting
plot_df = kdr_map.dropna(subset=['kdr_frequency','mean_temp_c','annual_rainfall_mm','country'])
west = plot_df[plot_df['mutation'] == 'kdr-west (L995F)']
east = plot_df[plot_df['mutation'] == 'kdr-east (L995S)']

# Color palette
colors = {'kdr-west (L995F)': '#E74C3C', 'kdr-east (L995S)': '#2E86AB'}

# ── Figure 1: kdr Frequency by Country ──────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(14, 12))
fig1.suptitle('kdr Insecticide Resistance Allele Frequencies\nby Country — MalariaGEN Ag3',
              fontsize=15, fontweight='bold', y=0.98)

for ax, (mut, mdf), title in zip(
    axes,
    [('kdr-west (L995F)', west), ('kdr-east (L995S)', east)],
    ['kdr-west (L995F) — Pyrethroid/DDT Resistance (West Africa dominant)',
     'kdr-east (L995S) — Pyrethroid/DDT Resistance (East Africa dominant)']
):
    country_mean = mdf.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False)
    bars = ax.bar(country_mean.index, country_mean.values,
                  color=colors[mut], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.axhline(0.25, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='25% threshold')
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel('Mean Allele Frequency', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(country_mean.index, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add value labels on bars
    for bar, val in zip(bars, country_mean.values):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
fig1.savefig('/content/malaria-resistance-mapping/outputs/charts/01_kdr_by_country.png',
             dpi=150, bbox_inches='tight')
plt.show()
print("Chart 1 saved ✓")

# ── Figure 2: Resistance vs Temperature ─────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('kdr Resistance Frequency vs Mean Annual Temperature\nWorldClim 2.1 Data',
              fontsize=13, fontweight='bold')

for ax, (mut, mdf) in zip(axes, [('kdr-west (L995F)', west), ('kdr-east (L995S)', east)]):
    x = mdf['mean_temp_c'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    ax.scatter(x, y, c=colors[mut], alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

    # Regression line
    if len(x) > 2:
        slope, intercept, r, p, se = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, slope * xline + intercept, 'k--', linewidth=1.5,
                label=f'r={r:.2f}, p={p:.3f}')
        ax.legend(fontsize=9)

    ax.set_xlabel('Mean Annual Temperature (°C)', fontsize=10)
    ax.set_ylabel('kdr Allele Frequency', fontsize=10)
    ax.set_title(mut, fontsize=10, color=colors[mut], fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig2.savefig('/content/malaria-resistance-mapping/outputs/charts/02_resistance_vs_temperature.png',
             dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved ✓")

# ── Figure 3: Resistance vs Rainfall ────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('kdr Resistance Frequency vs Annual Rainfall\nWorldClim 2.1 Data',
              fontsize=13, fontweight='bold')

for ax, (mut, mdf) in zip(axes, [('kdr-west (L995F)', west), ('kdr-east (L995S)', east)]):
    x = mdf['annual_rainfall_mm'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    sc = ax.scatter(x, y, c=colors[mut], alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

    if len(x) > 2:
        slope, intercept, r, p, se = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, slope * xline + intercept, 'k--', linewidth=1.5,
                label=f'r={r:.2f}, p={p:.3f}')
        ax.legend(fontsize=9)

    ax.set_xlabel('Annual Rainfall (mm)', fontsize=10)
    ax.set_ylabel('kdr Allele Frequency', fontsize=10)
    ax.set_title(mut, fontsize=10, color=colors[mut], fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig3.savefig('/content/malaria-resistance-mapping/outputs/charts/03_resistance_vs_rainfall.png',
             dpi=150, bbox_inches='tight')
plt.show()
print("Chart 3 saved ✓")

# ── Figure 4: Resistance Trend Over Time ────────────────────
fig4, ax = plt.subplots(figsize=(14, 6))
fig4.suptitle('kdr Resistance Allele Frequency Over Time\nAll African Countries Combined',
              fontsize=13, fontweight='bold')

for mut, mdf in [('kdr-west (L995F)', west), ('kdr-east (L995S)', east)]:
    yearly = mdf.groupby('year')['kdr_frequency'].agg(['mean','std','count']).reset_index()
    yearly = yearly[yearly['count'] >= 2]
    ax.plot(yearly['year'], yearly['mean'], 'o-',
            color=colors[mut], linewidth=2, markersize=6, label=mut)
    ax.fill_between(yearly['year'],
                    yearly['mean'] - yearly['std'],
                    yearly['mean'] + yearly['std'],
                    alpha=0.15, color=colors[mut])

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Mean kdr Allele Frequency', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='50% threshold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig4.savefig('/content/malaria-resistance-mapping/outputs/charts/04_resistance_over_time.png',
             dpi=150, bbox_inches='tight')
plt.show()
print("Chart 4 saved ✓")

print("\n✅ All 4 charts complete and saved!")

# ============================================================
# PHASE 4: Statistical Summary + Interpretation
# ============================================================
from scipy import stats

print("=" * 55)
print("RESISTANCE VS CLIMATE — STATISTICAL SUMMARY")
print("=" * 55)

plot_df = kdr_map.dropna(subset=['kdr_frequency','mean_temp_c','annual_rainfall_mm','country'])
west = plot_df[plot_df['mutation'] == 'kdr-west (L995F)']
east = plot_df[plot_df['mutation'] == 'kdr-east (L995S)']

for name, mdf in [('kdr-west (L995F)', west), ('kdr-east (L995S)', east)]:
    print(f"\n── {name} ──")
    print(f"  Cohorts: {len(mdf)}")
    print(f"  Mean frequency: {mdf['kdr_frequency'].mean():.3f}")
    print(f"  Max frequency:  {mdf['kdr_frequency'].max():.3f} ({mdf.loc[mdf['kdr_frequency'].idxmax(),'country']})")

    # Temp correlation
    x = mdf['mean_temp_c'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    r_t, p_t = stats.spearmanr(x[mask], y[mask])
    print(f"  vs Temperature:  r={r_t:.3f}, p={p_t:.3f} {'✓ significant' if p_t < 0.05 else '✗ not significant'}")

    # Rainfall correlation
    x2 = mdf['annual_rainfall_mm'].values
    mask2 = ~np.isnan(x2) & ~np.isnan(y)
    r_r, p_r = stats.spearmanr(x2[mask2], y[mask2])
    print(f"  vs Rainfall:     r={r_r:.3f}, p={p_r:.3f} {'✓ significant' if p_r < 0.05 else '✗ not significant'}")

    # Year trend
    r_y, p_y = stats.spearmanr(mdf['year'], mdf['kdr_frequency'])
    print(f"  vs Year (trend): r={r_y:.3f}, p={p_y:.3f} {'✓ significant' if p_y < 0.05 else '✗ not significant'}")

print("\n" + "=" * 55)
print("TOP 5 HIGHEST RESISTANCE COUNTRIES (kdr-west)")
print("=" * 55)
top5 = west.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False).head(5)
for country, freq in top5.items():
    temp = west[west['country']==country]['mean_temp_c'].mean()
    rain = west[west['country']==country]['annual_rainfall_mm'].mean()
    print(f"  {country:<25} freq={freq:.3f}  temp={temp:.1f}°C  rain={rain:.0f}mm")

# ============================================================
# CHART 5: Correlation Heatmap + Summary Figure
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import numpy as np
import os

os.makedirs('/content/malaria-resistance-mapping/outputs/charts', exist_ok=True)

plot_df = kdr_map.dropna(subset=['kdr_frequency','mean_temp_c','annual_rainfall_mm','country'])
west = plot_df[plot_df['mutation'] == 'kdr-west (L995F)']
east = plot_df[plot_df['mutation'] == 'kdr-east (L995S)']
colors = {'kdr-west (L995F)': '#E74C3C', 'kdr-east (L995S)': '#2E86AB'}

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Insecticide Resistance Analysis — MalariaGEN Ag3\nSub-Saharan Africa (2004–2024)',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: kdr by Country (West) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
country_mean = west.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False)
bar_colors = ['#C0392B' if v > 0.5 else '#E74C3C' if v > 0.25 else '#F1948A'
              for v in country_mean.values]
bars = ax1.bar(country_mean.index, country_mean.values,
               color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.4, label='50% threshold')
ax1.set_title('kdr-west (L995F) Mean Frequency by Country', fontweight='bold', fontsize=10)
ax1.set_ylabel('Allele Frequency')
ax1.set_ylim(0, 1.1)
ax1.set_xticklabels(country_mean.index, rotation=45, ha='right', fontsize=7.5)
ax1.legend(fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, country_mean.values):
    if val > 0.05:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5)

# ── Panel 2: kdr by Country (East) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
country_mean_e = east.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False).head(8)
ax2.barh(country_mean_e.index, country_mean_e.values,
         color='#2E86AB', edgecolor='white', linewidth=0.5)
ax2.set_title('kdr-east (L995S)\nTop Countries', fontweight='bold', fontsize=10)
ax2.set_xlabel('Allele Frequency')
ax2.set_xlim(0, 1.1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel 3: Resistance vs Temperature ──────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    x = mdf['mean_temp_c'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    ax3.scatter(x[mask], y[mask], c=c, alpha=0.6, s=50, label=mut)
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax3.plot(xl, slope*xl + intercept, color=c, linewidth=1.5, linestyle='--')
ax3.set_xlabel('Mean Temperature (°C)', fontsize=9)
ax3.set_ylabel('kdr Frequency', fontsize=9)
ax3.set_title('Resistance vs Temperature', fontweight='bold', fontsize=10)
ax3.legend(fontsize=7)
ax3.text(0.05, 0.92, 'West: r=0.562***', transform=ax3.transAxes, fontsize=7, color='#E74C3C')
ax3.text(0.05, 0.84, 'East: r=-0.344*', transform=ax3.transAxes, fontsize=7, color='#2E86AB')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel 4: Resistance vs Rainfall ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    x = mdf['annual_rainfall_mm'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    ax4.scatter(x[mask], y[mask], c=c, alpha=0.6, s=50, label=mut)
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax4.plot(xl, slope*xl + intercept, color=c, linewidth=1.5, linestyle='--')
ax4.set_xlabel('Annual Rainfall (mm)', fontsize=9)
ax4.set_ylabel('kdr Frequency', fontsize=9)
ax4.set_title('Resistance vs Rainfall', fontweight='bold', fontsize=10)
ax4.legend(fontsize=7)
ax4.text(0.05, 0.92, 'West: r=0.097 ns', transform=ax4.transAxes, fontsize=7, color='#E74C3C')
ax4.text(0.05, 0.84, 'East: r=0.284*', transform=ax4.transAxes, fontsize=7, color='#2E86AB')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel 5: Resistance Over Time ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    yearly = mdf.groupby('year')['kdr_frequency'].agg(['mean','std']).reset_index()
    ax5.plot(yearly['year'], yearly['mean'], 'o-', color=c,
             linewidth=2, markersize=5, label=mut)
    ax5.fill_between(yearly['year'],
                     (yearly['mean']-yearly['std']).clip(0),
                     (yearly['mean']+yearly['std']).clip(0,1),
                     alpha=0.15, color=c)
ax5.set_xlabel('Year', fontsize=9)
ax5.set_ylabel('Mean kdr Frequency', fontsize=9)
ax5.set_title('Resistance Trend Over Time', fontweight='bold', fontsize=10)
ax5.legend(fontsize=7)
ax5.set_ylim(0, 1.05)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel 6: Correlation Heatmap ────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
corr_data = []
variables = ['kdr_frequency', 'mean_temp_c', 'annual_rainfall_mm', 'year']
var_labels = ['kdr Frequency', 'Temperature (°C)', 'Rainfall (mm)', 'Year']

for mut, mdf, label in [('West', west, 'kdr-west'), ('East', east, 'kdr-east')]:
    row = {'Mutation': label}
    for v, vl in zip(variables[1:], var_labels[1:]):
        clean = mdf[['kdr_frequency', v]].dropna()
        r, p = stats.spearmanr(clean['kdr_frequency'], clean[v])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        row[vl] = f'{r:.2f}{stars}'
    corr_data.append(row)

corr_df = pd.DataFrame(corr_data).set_index('Mutation')

# Numeric version for heatmap colors
corr_num = []
for mut, mdf in [('kdr-west', west), ('kdr-east', east)]:
    row = []
    for v in variables[1:]:
        clean = mdf[['kdr_frequency', v]].dropna()
        r, p = stats.spearmanr(clean['kdr_frequency'], clean[v])
        row.append(r)
    corr_num.append(row)

corr_num_df = pd.DataFrame(corr_num,
                            index=['kdr-west (L995F)', 'kdr-east (L995S)'],
                            columns=var_labels[1:])

sns.heatmap(corr_num_df, annot=corr_df.values, fmt='',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax6, linewidths=0.5, linecolor='white',
            annot_kws={'size': 12, 'weight': 'bold'},
            cbar_kws={'label': 'Spearman r', 'shrink': 0.6})
ax6.set_title('Spearman Correlation: kdr Resistance vs Climate & Time Variables',
              fontweight='bold', fontsize=11, pad=10)
ax6.set_xticklabels(ax6.get_xticklabels(), fontsize=11)
ax6.set_yticklabels(ax6.get_yticklabels(), fontsize=10, rotation=0)

plt.savefig('/content/malaria-resistance-mapping/outputs/charts/05_summary_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Summary chart saved!")
print("\nKey findings:")
print("  • kdr-west significantly driven by temperature (r=0.562, p<0.001)")
print("  • kdr-west resistance increasing over time (r=0.331, p=0.016)")
print("  • kdr-east negatively associated with temperature (r=-0.344, p=0.012)")
print("  • Rainfall weakly associated with kdr-east only (r=0.284, p=0.039)")
print("  • Guinea, Côte d'Ivoire, Ghana at critical resistance levels (>90%)")

# ============================================================
# CHART 5: Correlation Heatmap + Summary Figure
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import numpy as np
import os

os.makedirs('/content/malaria-resistance-mapping/outputs/charts', exist_ok=True)

plot_df = kdr_map.dropna(subset=['kdr_frequency','mean_temp_c','annual_rainfall_mm','country'])
west = plot_df[plot_df['mutation'] == 'kdr-west (L995F)']
east = plot_df[plot_df['mutation'] == 'kdr-east (L995S)']
colors = {'kdr-west (L995F)': '#E74C3C', 'kdr-east (L995S)': '#2E86AB'}

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Insecticide Resistance Analysis — MalariaGEN Ag3\nSub-Saharan Africa (2004–2024)',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: kdr by Country (West) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
country_mean = west.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False)
bar_colors = ['#C0392B' if v > 0.5 else '#E74C3C' if v > 0.25 else '#F1948A'
              for v in country_mean.values]
bars = ax1.bar(country_mean.index, country_mean.values,
               color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.4, label='50% threshold')
ax1.set_title('kdr-west (L995F) Mean Frequency by Country', fontweight='bold', fontsize=10)
ax1.set_ylabel('Allele Frequency')
ax1.set_ylim(0, 1.1)
ax1.set_xticklabels(country_mean.index, rotation=45, ha='right', fontsize=7.5)
ax1.legend(fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, country_mean.values):
    if val > 0.05:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5)

# ── Panel 2: kdr by Country (East) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
country_mean_e = east.groupby('country')['kdr_frequency'].mean().sort_values(ascending=False).head(8)
ax2.barh(country_mean_e.index, country_mean_e.values,
         color='#2E86AB', edgecolor='white', linewidth=0.5)
ax2.set_title('kdr-east (L995S)\nTop Countries', fontweight='bold', fontsize=10)
ax2.set_xlabel('Allele Frequency')
ax2.set_xlim(0, 1.1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel 3: Resistance vs Temperature ──────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    x = mdf['mean_temp_c'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    ax3.scatter(x[mask], y[mask], c=c, alpha=0.6, s=50, label=mut)
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax3.plot(xl, slope*xl + intercept, color=c, linewidth=1.5, linestyle='--')
ax3.set_xlabel('Mean Temperature (°C)', fontsize=9)
ax3.set_ylabel('kdr Frequency', fontsize=9)
ax3.set_title('Resistance vs Temperature', fontweight='bold', fontsize=10)
ax3.legend(fontsize=7)
ax3.text(0.05, 0.92, 'West: r=0.562***', transform=ax3.transAxes, fontsize=7, color='#E74C3C')
ax3.text(0.05, 0.84, 'East: r=-0.344*', transform=ax3.transAxes, fontsize=7, color='#2E86AB')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel 4: Resistance vs Rainfall ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    x = mdf['annual_rainfall_mm'].values
    y = mdf['kdr_frequency'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    ax4.scatter(x[mask], y[mask], c=c, alpha=0.6, s=50, label=mut)
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax4.plot(xl, slope*xl + intercept, color=c, linewidth=1.5, linestyle='--')
ax4.set_xlabel('Annual Rainfall (mm)', fontsize=9)
ax4.set_ylabel('kdr Frequency', fontsize=9)
ax4.set_title('Resistance vs Rainfall', fontweight='bold', fontsize=10)
ax4.legend(fontsize=7)
ax4.text(0.05, 0.92, 'West: r=0.097 ns', transform=ax4.transAxes, fontsize=7, color='#E74C3C')
ax4.text(0.05, 0.84, 'East: r=0.284*', transform=ax4.transAxes, fontsize=7, color='#2E86AB')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel 5: Resistance Over Time ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
for mut, mdf, c in [('kdr-west', west, '#E74C3C'), ('kdr-east', east, '#2E86AB')]:
    yearly = mdf.groupby('year')['kdr_frequency'].agg(['mean','std']).reset_index()
    ax5.plot(yearly['year'], yearly['mean'], 'o-', color=c,
             linewidth=2, markersize=5, label=mut)
    ax5.fill_between(yearly['year'],
                     (yearly['mean']-yearly['std']).clip(0),
                     (yearly['mean']+yearly['std']).clip(0,1),
                     alpha=0.15, color=c)
ax5.set_xlabel('Year', fontsize=9)
ax5.set_ylabel('Mean kdr Frequency', fontsize=9)
ax5.set_title('Resistance Trend Over Time', fontweight='bold', fontsize=10)
ax5.legend(fontsize=7)
ax5.set_ylim(0, 1.05)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel 6: Correlation Heatmap ────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
corr_data = []
variables = ['kdr_frequency', 'mean_temp_c', 'annual_rainfall_mm', 'year']
var_labels = ['kdr Frequency', 'Temperature (°C)', 'Rainfall (mm)', 'Year']

for mut, mdf, label in [('West', west, 'kdr-west'), ('East', east, 'kdr-east')]:
    row = {'Mutation': label}
    for v, vl in zip(variables[1:], var_labels[1:]):
        clean = mdf[['kdr_frequency', v]].dropna()
        r, p = stats.spearmanr(clean['kdr_frequency'], clean[v])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        row[vl] = f'{r:.2f}{stars}'
    corr_data.append(row)

corr_df = pd.DataFrame(corr_data).set_index('Mutation')

# Numeric version for heatmap colors
corr_num = []
for mut, mdf in [('kdr-west', west), ('kdr-east', east)]:
    row = []
    for v in variables[1:]:
        clean = mdf[['kdr_frequency', v]].dropna()
        r, p = stats.spearmanr(clean['kdr_frequency'], clean[v])
        row.append(r)
    corr_num.append(row)

corr_num_df = pd.DataFrame(corr_num,
                            index=['kdr-west (L995F)', 'kdr-east (L995S)'],
                            columns=var_labels[1:])

sns.heatmap(corr_num_df, annot=corr_df.values, fmt='',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax6, linewidths=0.5, linecolor='white',
            annot_kws={'size': 12, 'weight': 'bold'},
            cbar_kws={'label': 'Spearman r', 'shrink': 0.6})
ax6.set_title('Spearman Correlation: kdr Resistance vs Climate & Time Variables',
              fontweight='bold', fontsize=11, pad=10)
ax6.set_xticklabels(ax6.get_xticklabels(), fontsize=11)
ax6.set_yticklabels(ax6.get_yticklabels(), fontsize=10, rotation=0)

plt.savefig('/content/malaria-resistance-mapping/outputs/charts/05_summary_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Summary chart saved!")
print("\nKey findings:")
print("  • kdr-west significantly driven by temperature (r=0.562, p<0.001)")
print("  • kdr-west resistance increasing over time (r=0.331, p=0.016)")
print("  • kdr-east negatively associated with temperature (r=-0.344, p=0.012)")
print("  • Rainfall weakly associated with kdr-east only (r=0.284, p=0.039)")
print("  • Guinea, Côte d'Ivoire, Ghana at critical resistance levels (>90%)")

from google.colab import files
import os

# Download all charts
chart_dir = '/content/malaria-resistance-mapping/outputs/charts/'
for f in sorted(os.listdir(chart_dir)):
    if f.endswith('.png'):
        print(f"Downloading {f}...")
        files.download(chart_dir + f)

# Download enriched dataset
files.download('/content/malaria-resistance-mapping/data/processed/kdr_climate_enriched.csv')

# Download the notebook
print("\nAlso go to File → Download → Download .ipynb to save the notebook")
print("\nAll files downloaded ✓")

from google.colab import files
import os

# Save enriched dataset to content root
kdr_map.to_csv('/content/kdr_climate_enriched.csv', index=False)
print(f"Saved: {len(kdr_map)} rows ✓")

# Download it
files.download('/content/kdr_climate_enriched.csv')

# Also download the notebook reminder
print("\nRemember: File → Download → Download .ipynb to save notebook")

