import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src" / "00_common"))
sys.path.insert(1, str(ROOT / "02_src" / "01_data" / "00_ingestion"))
sys.path.insert(2, str(ROOT / "02_src" / "01_data" / "01_preprocessing"))
sys.path.insert(3, str(ROOT / "02_src" / "01_data" / "02_io"))
sys.path.insert(4, str(ROOT / "02_src" / "02_model" / "00_architectures"))
sys.path.insert(5, str(ROOT / "02_src" / "02_model" / "01_training"))
sys.path.insert(6, str(ROOT / "02_src" / "02_model" / "03_registry"))

from config_loader import load_config

from data_loader import DataLoader
from data_processor import DataProcessor
from writer import MetricWriter

from factory import ModelFactory
from trainer import ModelTrainer
from model_manager import ModelManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument("filename", type=str, help="02_processed 폴더 내 입력 파일명 (예: train.csv)")
    parser.add_argument(
        "-m", "--model",
        nargs='+',
        default=["all"],
        choices=["rf", "xgb", "lr", "all"],
        help="훈련할 모델 목록 (예 --model rf xgb / 기본값: all)"
    )
    return parser.parse_args()


def run_model_pipeline(model_name: str, model_config: dict, raw_df: pd.DataFrame):
    processor = DataProcessor(model_config)
    # user_id 제거
    X_train, X_test, y_train, y_test = processor.load_train_data(raw_df)

    if model_config["use_scaler"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    model = ModelFactory.create_model(model_name, model_config['hyperparameters'])
    trainer = ModelTrainer(model, model_config)

    #Cross validaiton
    metrics = trainer.validate(X_train, y_train)

    for metric_name, metric in zip(model_config['train_config']['scoring'], metrics):
        print(f'Train {metric_name} : {metrics[metric]}')

    #final train
    trained_model = trainer.fit_final(X_train, y_train)

    #test
    best_threshold = trainer.get_best_threshold(X_test, y_test)
    test_report = trainer.evaluate(X_test, y_test, threshold=best_threshold)

    print("\n[ Test Classification Report ]")
    report_df = pd.DataFrame(test_report).transpose()
    print(report_df.round(3).to_string())
    print('Applied Threshold: ', best_threshold, end='\n\n')

    metric_path = ROOT / "05_artifacts" / "02_metrics"
    writer = MetricWriter(base_path = metric_path)

    # save metrics
    writer.save_performance(test_report, model_name)

    # save pr curve data
    y_probs = trained_model.predict_proba(X_test)[:, 1]
    pr_df = trainer.get_pr_curve_df(y_test, y_probs)
    writer.save_csv(pr_df, f"{model_name}_pr_curve_data.csv")

    # save feature importances
    feature_names = raw_df.drop(columns=['user_id', 'is_active']).columns.tolist()
    importance_df = trainer.get_feature_importance_df(trained_model, feature_names)

    if not importance_df.empty:
        writer.save_csv(importance_df, f"{model_name}_feature_importance.csv")

    #save model package
    model_pacakge = {
        'model': trained_model,
        'scaler': scaler,
        'threshold': best_threshold,
    }
    ModelManager.save_model_package(model_pacakge)


def main():
    args = parse_args()
    full_config = load_config('model_config.json')
    loader = DataLoader()

    raw_data_path = ROOT / "00_data" / "02_processed" / args.filename

    if not raw_data_path.exists():
        print(f"❌ Error: 파일을 찾을 수 없습니다: {raw_data_path}")
        return

    raw_df = loader.load_csv(raw_data_path)

    # 훈련 대상 모델 결정
    if "all" in args.model:
        target_models = list(full_config.keys())
    else:
        mapping = {"rf": "RandomForest", "xgb": "XGBoost", "lr": "LogisticRegression"}
        target_models = [mapping[m] for m in args.model if m in mapping]

    # 모델별 순차 훈련 진행
    for m_name in target_models:
        try:
            print(f'\n{m_name} 모델 훈련 시작...')
            run_model_pipeline(m_name, full_config[m_name], raw_df)
        except Exception as e:
            print(f"⚠️ {m_name} 훈련 중 오류 발생: {e}")
            continue

    print("\n" + "✨" * 3 + " 모든 훈련 프로세스가 종료되었습니다 " + "✨" * 3)


if __name__ == "__main__":
    main()
