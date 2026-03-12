from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import numpy as np

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.train_cfg = config.get('train_config', {})

    def validate(self, X, y):

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

    def fit_final(self, X, y):
        #전체 학습 데이터로 최종 학습
        self.model.fit(X, y)
        return self.model

    def get_best_threshold(self, X, y):
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

    def evaluate(self, X_test, y_test, threshold=0.5):
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        # 지표 계산
        return classification_report(y_test, preds, output_dict=True)