from typing import Any

import pandas as pd
import sys
from pathlib import Path

# 경로 설정
ROOT = Path(__file__).resolve().parents[3]
INGESTION_PATH = ROOT / "02_src" / "01_data" / "00_ingestion"
PREPROCESS_PATH = ROOT / "02_src" / "01_data" / "01_preprocessing"
REGISTRY_PATH = ROOT / "02_src" / "02_model" / "03_registry"

sys.path.insert(0, str(INGESTION_PATH))
sys.path.insert(1, str(PREPROCESS_PATH))
sys.path.insert(2, str(REGISTRY_PATH))


from data_loader import DataLoader
from data_processor import DataProcessor
from model_manager import ModelManager


class ModelPredictor:
    def __init__(self, processor_config = None):
        self.processor = DataProcessor(processor_config)
        self._model_package_cache = {} #모델 객체 캐싱

    def _get_model(self, model_type: str) -> dict:
        """
        model package 불러오기
        """
        if model_type not in self._model_package_cache:
            self._model_package_cache[model_type] = ModelManager.load_model_package(model_type)
        return self._model_package_cache[model_type]

    def _label_risk(self, prob: float) -> str:
        """
        churn probability에 따른 위험 분류
        """
        if prob >= 0.75: return "높은 위험"
        if prob >= 0.50: return "중간 위험"
        if prob >= 0.25: return "낮은 위험"
        return "안전"

    def _prepare_input_data(self, input_df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """
        데이터 파이프라인을 통해 유저 데이터를 모델 입력 데이터로 변환
        """

        # dataprocessor로 모델 입력 데이터 생성
        processed_df = self.processor.run_full_pipeline(input_df)

        # 피쳐 순서 정렬
        X = DataLoader.reorder_columns(processed_df, mode = 'feature')

        # 필요시 스케일링 적용
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        return X_scaled

    def predict(self, model_type: str, user_data: dict) -> dict:
        """
        [단건 예측] 외부 API로부터 유저 1명의 데이터를 받아 결과를 Dict로 반환합니다.
        """
        # 입력을 DataFrame으로 변환
        input_df = pd.DataFrame([user_data])

        # 입력 데이터 검증
        is_ok, missing = DataLoader.validate_df(input_df, mode = 'predict')
        if not is_ok:
            return {"status": "error", "message": f"Missing input: {missing}"}

        # 모델 로드
        model_package = self._get_model(model_type)
        model = model_package['model']
        scaler = model_package['scaler']
        threshold = model_package['config'].get('threshold', 0.5)

        # 모델 입력 데이터 준비
        X_scaled = self._prepare_input_data(input_df, scaler)

        # 이탈 확률 계산
        prob = float(model.predict_proba(X_scaled)[0, 1])

        return {
            "status": "success",
            "result": {
            "user_id": user_data.get("user_id"),
            "churn_probability": round(prob, 4),
            "is_churn": int(prob >= threshold),
            "risk_label": self._label_risk(prob),
            "threshold": threshold,
            "risk_score": round(prob * 100, 1)
            }
        }

    def batch_predict(self, model_type: str, users_data_list: list[dict]) -> dict:
        """
        [배치 예측] 유저 리스트를 받아 결과 리스트를 반환합니다.
        """

        # 전체 리스트를 한 번에 DataFrame으로 변환
        input_df = pd.DataFrame(users_data_list)

        # 입력 데이터 검증
        is_ok, missing = DataLoader.validate_df(input_df, mode='predict')
        if not is_ok:
            return {"status": "error", "message": f"Missing input: {missing}"}

        model_package = self._get_model(model_type)
        model = model_package['model']
        scaler = model_package['scaler']
        threshold = model_package['config'].get('threshold', 0.5)

        # 입력 데이터 준비
        X_scaled = self._prepare_input_data(input_df, scaler)

        #이탈 확률 계산
        probs = model.predict_proba(X_scaled)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            prob_val = float(prob)
            results.append({
                "user_id": users_data_list[i].get("user_id"),
                "churn_probability": round(prob_val, 4),
                "is_churn": int(prob_val >= threshold),
                "risk_label": self._label_risk(prob_val),
                "threshold": threshold,
                "risk_score": round(prob_val * 100, 1)
            })

        return {"status": "success", "results":results}