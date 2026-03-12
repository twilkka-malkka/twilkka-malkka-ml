import joblib
import json
import os
from pathlib import Path

import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
SAVE_FOLDER = ROOT / "05_artifacts"

class ModelManager:
    @staticmethod
    def save_model_package(package_dict):
        model = package_dict['model']
        model_type = type(model).__name__

        #모델별 폴더 생성
        model_folder = SAVE_FOLDER / "00_models" / model_type
        os.makedirs(model_folder, exist_ok=True)

        # 모델 저장 (XGBoost만 예외)
        if "XGB" in model_type:
            model_path = model_folder / 'model.json'
            model.save_model(model_path)
            save_method = "native_json"
        else:
            model_path = model_folder / 'model.joblib'
            joblib.dump(model, model_path)
            save_method = "joblib"

        # 전처리기 저장
        if package_dict['scaler'] is not None:
            scaler_path = SAVE_FOLDER / "01_preprocessors" / (model_type+"_scaler.joblib")
            joblib.dump(package_dict['scaler'], scaler_path)

        # 3. 설정 파일 저장
        config_path = model_folder / "config.json"
        config_data = {
            "model_type": model_type,
            "save_method": save_method,
            "threshold": package_dict['threshold']
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)

        print(f"✅ [{model_folder}에 모델 저장 완료 ({save_method} 방식)")

    @staticmethod
    def load_model_package(model_type):
        model_folder = SAVE_FOLDER / "00_models" / model_type

        # Config 먼저 읽기
        config_path = model_folder / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 2. 모델 로드
        if config['save_method'] == "native_json":
            # XGBoost 객체 생성 후 로드
            model = xgb.XGBClassifier()
            model.load_model(model_folder/ 'model.json')
        else:
            model = joblib.load(model_folder/ 'model.joblib')

        # 3. 스케일러 및 나머지 로드
        scaler_path = SAVE_FOLDER / "01_preprocessors" / (model_type+"_scaler.joblib")
        scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None

        return {
            'model': model,
            'scaler': scaler,
            'config': config
        }