import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def run_preprocessing(input_path, output_dir):
    print(f"Loading data from {input_path}")
    data = pd.read_csv(input_path)
    
    # Menghapus duplikat misal ada
    data = data.drop_duplicates()
    
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Preprocessing data...")
    X_preprocessed = preprocessor.fit_transform(X)
    
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_feature_names)
    
    X_prep_df = pd.DataFrame(X_preprocessed, columns=all_feature_names)
    df_final = pd.concat([X_prep_df, y.reset_index(drop=True)], axis=1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'credit_risk_dataset_processed.csv')
    df_final.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../credit_risk_dataset_raw/credit_risk_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./credit_risk_dataset_preprocessing')
    args = parser.parse_args()
    
    run_preprocessing(args.input, args.output_dir)
