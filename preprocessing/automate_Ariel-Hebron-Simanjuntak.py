import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def run_preprocessing(input_path, output_dir):
    """
    Melakukan preprocessing data Credit Risk Dataset secara otomatis.
    Tahapan sesuai dengan notebook eksperimen:
      1. Load data
      2. Hapus duplikat
      3. Penanganan missing values (per kolom dengan median)
      4. Penghapusan outlier
      5. Encoding fitur kategorikal
      6. Standarisasi fitur numerik
      7. Train-test split
      8. Simpan hasil preprocessing
    """

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset berhasil dimuat dengan {df.shape[0]} baris dan {df.shape[1]} kolom.")

    # -------------------------------------------------------------------------
    # 2. Hapus Duplikat
    # -------------------------------------------------------------------------
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Duplikat dihapus: {before - df.shape[0]} baris. Sisa: {df.shape[0]} baris.")

    # -------------------------------------------------------------------------
    # 3. Penanganan Missing Values
    # -------------------------------------------------------------------------
    df_clean = df.copy()

    # Mengisi missing values pada person_emp_length dengan median
    df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(
        df_clean['person_emp_length'].median()
    )

    # Mengisi missing values pada loan_int_rate dengan median
    df_clean['loan_int_rate'] = df_clean['loan_int_rate'].fillna(
        df_clean['loan_int_rate'].median()
    )

    print("Missing values setelah penanganan:")
    print(df_clean.isnull().sum())

    # -------------------------------------------------------------------------
    # 4. Penghapusan Outlier
    # -------------------------------------------------------------------------
    print(f"Jumlah data sebelum menghapus outlier: {df_clean.shape[0]}")

    # Menghapus outlier pada person_age (> 100 tahun tidak realistis)
    df_clean = df_clean[df_clean['person_age'] <= 100]

    # Menghapus outlier pada person_emp_length (> 60 tahun tidak realistis)
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]

    print(f"Jumlah data setelah menghapus outlier: {df_clean.shape[0]}")
    print(f"Data yang dihapus: {df.shape[0] - df_clean.shape[0]} baris")

    # -------------------------------------------------------------------------
    # 5. Encoding Fitur Kategorikal
    # -------------------------------------------------------------------------

    # Binary encoding untuk cb_person_default_on_file
    df_clean['cb_person_default_on_file'] = df_clean['cb_person_default_on_file'].map(
        {'Y': 1, 'N': 0}
    )

    # Ordinal encoding untuk loan_grade (A=1 terbaik, G=7 terburuk)
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df_clean['loan_grade'] = df_clean['loan_grade'].map(grade_mapping)

    # One-hot encoding untuk person_home_ownership dan loan_intent
    df_clean = pd.get_dummies(df_clean, columns=['person_home_ownership', 'loan_intent'])

    print(f"Encoding selesai. Dimensi data: {df_clean.shape}")

    # -------------------------------------------------------------------------
    # 6. Standarisasi Fitur Numerik
    # -------------------------------------------------------------------------
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status']

    numerical_features = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ]

    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    print(f"Standarisasi selesai. Dimensi fitur (X): {X.shape}")
    print(f"Dimensi target (y): {y.shape}")

    # -------------------------------------------------------------------------
    # 7. Train-Test Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Ukuran data latih: {X_train.shape}")
    print(f"Ukuran data uji  : {X_test.shape}")
    print(f"\nProporsi target pada data latih:")
    print(y_train.value_counts(normalize=True).round(4))
    print(f"\nProporsi target pada data uji:")
    print(y_test.value_counts(normalize=True).round(4))

    # -------------------------------------------------------------------------
    # 8. Simpan Hasil Preprocessing
    # -------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Simpan full preprocessed dataset (sesuai notebook)
    preprocessed_data = pd.concat([X, y.reset_index(drop=True)], axis=1)
    output_path = os.path.join(output_dir, 'credit_risk_dataset_processed.csv')
    preprocessed_data.to_csv(output_path, index=False)
    print(f"\nData preprocessing berhasil disimpan ke {output_path}")
    print(f"Jumlah data: {preprocessed_data.shape[0]} baris, {preprocessed_data.shape[1]} kolom")

    # Simpan juga train/test split secara terpisah
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1)

    train_path = os.path.join(output_dir, 'credit_risk_train.csv')
    test_path  = os.path.join(output_dir, 'credit_risk_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"Data latih disimpan ke {train_path}")
    print(f"Data uji   disimpan ke {test_path}")

    return preprocessed_data, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automate preprocessing Credit Risk Dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../credit_risk_dataset_raw/credit_risk_dataset.csv',
        help='Path ke file CSV raw dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./credit_risk_dataset_preprocessing',
        help='Direktori output hasil preprocessing'
    )
    args = parser.parse_args()

    run_preprocessing(args.input, args.output_dir)
