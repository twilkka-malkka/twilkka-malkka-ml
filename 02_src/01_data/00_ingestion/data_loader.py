import pandas as pd
import os

class DataLoader:
    def __init__(self, config=None):
        self.config = config

    def load_csv(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 경로에 파일이 없습니다: {file_path}")
        df = pd.read_csv(file_path)
        return df