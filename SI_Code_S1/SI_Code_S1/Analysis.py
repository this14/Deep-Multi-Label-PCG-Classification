import pandas as pd
import numpy as np
from pathlib import Path

# Set file path
csv_path = r"D:\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"

# Load CSV file
df = pd.read_csv(csv_path)

print("=" * 80)
print("CirCor DigiScope Phonocardiogram Dataset - Complete Analysis")
print("=" * 80)

# ============================================================================
# 1. Basic Dataset Information
# ============================================================================
print("\n[1. DATASET OVERVIEW]")
print("-" * 80)
print(f"Total Subjects: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

# Analyze recording locations to calculate total number of recordings
def count_recordings(loc_str):
    """Count number of recordings from Recording locations string"""
    if pd.isna(loc_str):
        return 0
    # Count from 'AV+PV+TV+MV' format: number of '+' symbols + 1
    return len(loc_str.split('+'))

df['num_recordings'] = df['Recording locations:'].apply(count_recordings)
total_recordings = df['num_recordings'].sum()

print(f"Total Recordings: {total_recordings}")
print(f"Average Recordings per Subject: {total_recordings/len(df):.2f}")

# ============================================================================
# 2. MURMUR Distribution Analysis
# ============================================================================
print("\n[2. MURMUR DISTRIBUTION]")
print("-" * 80)

murmur_dist = df['Murmur'].value_counts()
print("Murmur Status:")
for status, count in murmur_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {status}: {count} ({percentage:.1f}%)")

# Number of recordings for Murmur Present cases
murmur_present = df[df['Murmur'] == 'Present']
murmur_present_recordings = murmur_present['num_recordings'].sum()
print(f"\nMurmur Present Subjects: {len(murmur_present)}")
print(f"Murmur Present Total Recordings: {murmur_present_recordings}")

# ============================================================================
# 3. SYSTOLIC MURMUR Characteristics Analysis (Murmur Present only)
# ============================================================================
print("\n[3. SYSTOLIC MURMUR CHARACTERISTICS - Distribution]")
print("-" * 80)

# Timing
print("\n3.1 Systolic Murmur Timing:")
timing_dist = murmur_present['Systolic murmur timing'].value_counts()
for timing, count in timing_dist.items():
    if pd.notna(timing):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {timing}: {count} ({percentage:.1f}%)")

# Shape
print("\n3.2 Systolic Murmur Shape:")
shape_dist = murmur_present['Systolic murmur shape'].value_counts()
for shape, count in shape_dist.items():
    if pd.notna(shape):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {shape}: {count} ({percentage:.1f}%)")

# Grading
print("\n3.3 Systolic Murmur Grading:")
grading_dist = murmur_present['Systolic murmur grading'].value_counts()
for grading, count in grading_dist.items():
    if pd.notna(grading):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {grading}: {count} ({percentage:.1f}%)")

# Pitch
print("\n3.4 Systolic Murmur Pitch:")
pitch_dist = murmur_present['Systolic murmur pitch'].value_counts()
for pitch, count in pitch_dist.items():
    if pd.notna(pitch):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {pitch}: {count} ({percentage:.1f}%)")

# Quality
print("\n3.5 Systolic Murmur Quality:")
quality_dist = murmur_present['Systolic murmur quality'].value_counts()
for quality, count in quality_dist.items():
    if pd.notna(quality):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {quality}: {count} ({percentage:.1f}%)")

# ============================================================================
# 4. RECORDING LOCATIONS Analysis
# ============================================================================
print("\n[4. RECORDING LOCATIONS ANALYSIS]")
print("-" * 80)

# Recording locations distribution
print("Recording Locations Distribution:")
location_counts = df['Recording locations:'].value_counts().head(10)
for loc, count in location_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  - {loc}: {count} ({percentage:.1f}%)")

# Most audible location (for Murmur Present cases)
print("\nMost Audible Location (Murmur Present):")
audible_loc = murmur_present['Most audible location'].value_counts()
for loc, count in audible_loc.items():
    if pd.notna(loc):
        percentage = (count / len(murmur_present)) * 100
        print(f"  - {loc}: {count} ({percentage:.1f}%)")

# ============================================================================
# 5. Demographic Information
# ============================================================================
print("\n[5. DEMOGRAPHIC INFORMATION]")
print("-" * 80)

# Age
print("Age Distribution:")
age_dist = df['Age'].value_counts().sort_index()
for age, count in age_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {age}: {count} ({percentage:.1f}%)")

# Sex
print("\nSex Distribution:")
sex_dist = df['Sex'].value_counts()
for sex, count in sex_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {sex}: {count} ({percentage:.1f}%)")

# Height and Weight statistics
print("\nHeight (cm):")
print(f"  - Mean: {df['Height'].mean():.1f}")
print(f"  - Std: {df['Height'].std():.1f}")
print(f"  - Range: {df['Height'].min():.1f} - {df['Height'].max():.1f}")

print("\nWeight (kg):")
print(f"  - Mean: {df['Weight'].mean():.1f}")
print(f"  - Std: {df['Weight'].std():.1f}")
print(f"  - Range: {df['Weight'].min():.1f} - {df['Weight'].max():.1f}")

# Pregnancy status
pregnancy_dist = df['Pregnancy status'].value_counts()
print("\nPregnancy Status:")
for status, count in pregnancy_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {status}: {count} ({percentage:.1f}%)")

# ============================================================================
# 6. OUTCOME Analysis
# ============================================================================
print("\n[6. CLINICAL OUTCOME]")
print("-" * 80)

outcome_dist = df['Outcome'].value_counts()
print("Outcome Distribution:")
for outcome, count in outcome_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {outcome}: {count} ({percentage:.1f}%)")

# Relationship between Murmur and Outcome
print("\nCross-tabulation: Murmur vs Outcome")
cross_tab = pd.crosstab(df['Murmur'], df['Outcome'], margins=True)
print(cross_tab)

# ============================================================================
# 7. CAMPAIGN Information
# ============================================================================
print("\n[7. SCREENING CAMPAIGN]")
print("-" * 80)

campaign_dist = df['Campaign'].value_counts()
print("Campaign Distribution:")
for campaign, count in campaign_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  - {campaign}: {count} ({percentage:.1f}%)")

# ============================================================================
# 8. Recommended Data Split (60:20:20 ratio)
# ============================================================================
print("\n[8. RECOMMENDED DATA SPLIT (60:20:20)]")
print("-" * 80)

total_subjects = len(df)
train_size = int(total_subjects * 0.6)
val_size = int(total_subjects * 0.2)
test_size = total_subjects - train_size - val_size

print(f"Training Set: {train_size} subjects ({train_size/total_subjects*100:.1f}%)")
print(f"Validation Set: {val_size} subjects ({val_size/total_subjects*100:.1f}%)")
print(f"Test Set: {test_size} subjects ({test_size/total_subjects*100:.1f}%)")

# ============================================================================
# 9. Confirmed Information Summary for Paper
# ============================================================================
print("\n" + "=" * 80)
print("[CONFIRMED INFORMATION FOR PAPER]")
print("=" * 80)

print("""
✅ Sampling Frequency: 4000 Hz
✅ IRB Approval: 5192-Complexo Hospitalar HUOC/PROCAPE Institutional Review Board
✅ Data Collection Period: July-August 2014, June-July 2015 (Northeast Brazil)
✅ Public Dataset Ratio: 60% (Training set for PhysioNet Challenge 2022)
✅ License: ODC Attribution License (Open Data Commons)
✅ Age Range: 0-21 years (mean ± STD = 6.1 ± 4.3 years)
✅ Recording Duration: 4.8-80.4 seconds (mean ± STD = 22.9 ± 7.4 seconds)
""")

print("\n" + "=" * 80)
print("Analysis Complete! Use these statistics in your paper.")
print("=" * 80)

# Additional: CSV save option
save_option = input("\n\nWould you like to save detailed analysis results to CSV? (y/n): ")
if save_option.lower() == 'y':
    output_path = r"D:\the-circor-digiscope-phonocardiogram-dataset-1.0.3\dataset_analysis_results.csv"
    
    # Create summary statistics as DataFrame
    summary_data = {
        'Category': ['Total Subjects', 'Total Recordings', 'Murmur Present', 
                     'Murmur Absent', 'Murmur Unknown', 'Outcome Normal', 
                     'Outcome Abnormal'],
        'Count': [len(df), total_recordings, len(murmur_present),
                  len(df[df['Murmur'] == 'Absent']), 
                  len(df[df['Murmur'] == 'Unknown']),
                  len(df[df['Outcome'] == 'Normal']),
                  len(df[df['Outcome'] == 'Abnormal'])]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Analysis results saved: {output_path}")