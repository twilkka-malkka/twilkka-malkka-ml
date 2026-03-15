from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import numpy as np
import pandas as pd
from typing import Any

class ModelTrainer:
    def __init__(self, model: Any, config: dict):
        self.model = model
        self.train_cfg = config.get('train_config', {})

    def validate(self, X: Any, y: Any) -> dict:
        """
        train data에 대해 교차 검증 수행 후 평가 지표별 평균값을 반환합니다.
        """
        cv = StratifiedKFold(
            n_splits=self.train_cfg.get('cv_folds', 5),
            shuffle=self.train_cfg.get('shuffle', True),
            random_state=self.train_cfg.get('random_state', True),
        )

        cv_results = cross_validate(
            self.model, X, y,
            cv=self.train_cfg.get('cv_folds', 5),
            scoring=self.train_cfg.get('scoring'),
            n_jobs=-1
        )

        # 각 지표별 평균값 계산 (test_precision, test_f1 등)
        avg_metrics = {k: np.mean(v) for k, v in cv_results.items() if k.startswith('test_')}
        return avg_metrics

    def fit_final(self, X: Any, y: Any) -> Any:
        """
        전체 train data로 모델을 최종 학습합니다.
        """
        self.model.fit(X, y)
        return self.model

    def get_best_threshold(self, X: Any, y: Any) -> float:
        """
        모델의 f1-score를 최대로 하는 threshold 값을 찾습니다.
        """
        probs = self.model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = 0
        best_threshold = 0.5

        for tau in thresholds:
            preds = (probs >= tau).astype(int)
            score = f1_score(y, preds, average='macro')
            if score > best_f1:
                best_f1 = score
                best_threshold = tau
        return best_threshold

    def evaluate(self, X_test: Any, y_test:Any, threshold: float=0.5) -> dict:
        """
        평가 데이터에 대한 classification report를 반환합니다.
        """
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        # 결과 리포트 반환
        target_names = ['active', 'churned']
        report = classification_report(y_test, preds, target_names=target_names , output_dict=True)
        return report

    def get_pr_curve_df(self, y_test: Any, y_probs: Any) -> pd.DataFrame:
        """
        PR-Curve 시각화를 위한 데이터를 생성하여 DataFrame으로 반환합니다.
        """
        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

        # DataFrame 구성 (길이 매칭을 위해 마지막 포인트 제외)
        return pd.DataFrame({
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds
        })


    def get_feature_importance_df(self, model: Any, feature_names: list[str]) -> pd.DataFrame:
        """
        모델 유형에 따라 피처 중요도를 추출하여 중요도 내림차순으로 정렬된 DataFrame으로 반환합니다.
        """
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) #중요도이므로 절댓값 적용

        if importances is None: #중요도 정보가 없으면 빈 DataFrame 반환
            return pd.DataFrame()

        # 백분율 정규화
        total_importance = np.sum(importances)
        if total_importance > 0:
            importance_normalized = (importances / total_importance) * 100
        else:
            importance_normalized = importances #학습이 되지 않았거나 모든 데이터가 동일해 변별력이 없는 경우

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_normalized
        }).sort_values(by='importance', ascending=False)

        importance_df['importance'] = importance_df['importance'].round(3)

        return importance_df