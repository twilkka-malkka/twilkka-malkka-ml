import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src" / "00_common"))
sys.path.insert(1, str(ROOT / "02_src" /"01_data" / "00_ingestion"))
sys.path.insert(2, str(ROOT / "02_src" /"01_data" / "01_preprocessing"))
sys.path.insert(3, str(ROOT / "02_src" / "02_model"/ "00_architectures"))
sys.path.insert(4, str(ROOT / "02_src" / "02_model" / "01_training"))
sys.path.insert(5, str(ROOT / "02_src" / "02_model" / "03_registry"))


from config_loader import load_config

from data_loader import DataLoader
from data_processor import DataProcessor

from factory import ModelFactory
from trainer import Trainer
from model_manager import ModelManager

def parse_args():
    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="훈련할 모델 이름 약어 (rf, xgb, lr / 기본값: all)"
    )
    return parser.parse_args()

def run_model_pipeline(model_name, model_config, raw_df):
    processor = DataProcessor(model_config)
    # user_id 제거
    X_train, X_test, y_train, y_test = processor.load_train_data(raw_df)

    if model_config["use_scaler"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = ModelFactory.create_model(model_name, model_config['hyperparameters'])

    trainer = Trainer(model, model_config)

    #Cross validaiton
    metrics = trainer.validate(X_train, y_train)

    for metric_name, metric in zip(model_config['train_config']['scoring'], metrics):
        print(f'Train {metric_name} : {metrics[metric]}')

    #final train
    trained_model = trainer.fit_final(X_train, y_train)

    #test
    best_threshold = trainer.get_best_threshold(X_test, y_test)

    test_report = trainer.evaluate(X_test, y_test, threshold = best_threshold)
    #print(test_report)
    print("\n[ Test Classification Report ]")
    report_df = pd.DataFrame(test_report).transpose()
    print(report_df.round(3).to_string())
    #model save
    rf_pacakge = {
        'model' : trained_model,
        'scaler' : scaler if scaler is not None else None,
        'threshold' : best_threshold,
    }
    ModelManager.save_model_package(rf_pacakge)


def main():
    args = parse_args()
    full_config = load_config('model_config.json')
    loader = DataLoader()
    raw_data_path = ROOT / "00_data" / "01_interim" / "netflix_users_train.csv"
    raw_df = loader.load_csv(raw_data_path)
    # 훈련 대상 모델 결정
    if args.model.lower() == "all":
        target_models = full_config.keys()
    else:
        if args.model.lower() == "rf":
            model_name = "RandomForest"
        elif args.model.lower() == "xgb":
            model_name = "XGBoost"
        elif args.model.lower() == "lr":
            model_name = "LogisticRegression"
        else:
            print(f"❌ Error: {args.model}은 설정 파일에 존재하지 않습니다.")
            return
        target_models = [model_name]

    # 모델별 순차 훈련 진행
    for m_name in target_models:
        try:
            run_model_pipeline(m_name, full_config[m_name], raw_df)
        except Exception as e:
            print(f"⚠️ {m_name} 훈련 중 오류 발생: {e}")
            continue

    print("\n" + "✨" * 3 + " 모든 훈련 프로세스가 종료되었습니다 " + "✨" * 3)
if __name__  == "__main__":
    main()