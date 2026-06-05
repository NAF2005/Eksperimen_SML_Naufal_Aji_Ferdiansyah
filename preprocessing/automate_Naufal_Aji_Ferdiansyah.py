import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Mulai proses preprocessing...")
    # 1. Bersihkan nama kolom
    df.columns = df.columns.str.replace(' ', '_')
    
    # 2. Ubah target jadi biner (1 = Good, 0 = Bad)
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
    
    # 3. Pisahkan fitur dan target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 5. Gabungkan kembali
    df_processed = pd.concat([X_scaled, y], axis=1)
    return df_processed

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data preprocessing berhasil tersimpan di {output_path}")

if __name__ == "__main__":
    # 1. Dapatkan lokasi direktori
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    
    # 2. Definisikan path folder raw dan path file
    RAW_DIR = os.path.join(ROOT_DIR, "dataset_raw")
    RAW_DATA_PATH = os.path.join(RAW_DIR, "winequality-red.csv")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "winequality_preprocessing.csv")
    
    print("-" * 50)
    print(f"🔍 INVESTIGASI FOLDER RAW")
    print(f"Mencari folder di: {RAW_DIR}")
    
    # Cek apakah foldernya ada
    if not os.path.exists(RAW_DIR):
        print("❌ KESIMPULAN: Folder 'data_raw' TIDAK ADA atau salah nama.")
    else:
        print("✅ Folder 'dataset_raw' DITEMUKAN.")
        print(f"Isi file di dalam folder tersebut adalah: {os.listdir(RAW_DIR)}")
        print("-" * 50)
        
        # Cek file CSV-nya
        if os.path.exists(RAW_DATA_PATH):
            print("Mengeksekusi preprocessing...")
            df_raw = load_data(RAW_DATA_PATH)
            df_clean = preprocess_data(df_raw)
            save_data(df_clean, PROCESSED_DATA_PATH)
        else:
            print("❌ KESIMPULAN: File CSV tidak cocok. Silakan samakan nama file di atas dengan 'winequality-red.csv'")