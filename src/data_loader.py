import pandas as pd

def load_data():
    vitals = pd.read_csv("data/raw/vitals.csv")
    labs = pd.read_csv("data/raw/labs.csv")
    icu = pd.read_csv("data/raw/icustays.csv")
    patients = pd.read_csv("data/raw/patients.csv")
    notes = pd.read_csv("data/raw/notes.csv")

    print("\n========== DATASET SUMMARY ==========")

    print(f"\nVitals Shape: {vitals.shape}")
    print(vitals.columns)
    print(vitals.head())

    print(f"\nLabs Shape: {labs.shape}")
    print(labs.columns)
    print(labs.head())

    print(f"\nICU Shape: {icu.shape}")
    print(icu.columns)
    print(icu.head())

    print(f"\nPatients Shape: {patients.shape}")
    print(patients.columns)
    print(patients.head())

    print(f"\nNotes Shape: {notes.shape}")
    print(notes.columns)
    print(notes.head())

    return vitals, labs, icu, patients, notes

if __name__ == "__main__":
    load_data()