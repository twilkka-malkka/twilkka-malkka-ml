import pandas as pd
import json
from pathlib import Path

class MetricWriter:
    def __init__(self, base_path: Path):
        """
        :param base_path: metric이 저장될 디렉토리
        """
        self.base_path = base_path

    def save_performance(self, metrics_dict: dict, model_name: str) -> None:
        """
        학습된 모델의 Accuracy, F1-score 등의 수치 지표를 JSON으로 저장
        """

        file_name = f"{model_name}_metrics.json"
        save_path = self.base_path / file_name

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"✅ {file_name} 저장 완료: {save_path}")

    def save_csv(self, df: pd.DataFrame, file_name: str) -> None:
        """
        전달받은 dataframe을 CSV로 저장
        """

        save_path = self.base_path / file_name
        df.to_csv(save_path, index=False)
        print(f"✅ {file_name} 저장 완료: {save_path}")