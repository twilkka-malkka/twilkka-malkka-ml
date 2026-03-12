import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, config):
        self.model_cfg = config
        self.target = "is_active"

        # 경로 설정 (ROOT는 utils 등에서 정의한 것을 가져오거나 직접 계산)
        self.root = Path(__file__).parent.parent.parent
        self.interim_dir = self.root / "00_data" / "01_interim"
        self.watch_features_path = self.interim_dir / "watch_features.csv"

    def _select_columns(self, df):
        user_columns = ['user_id', 'age', 'plan_tier', 'subscription_start_date', 'monthly_spend']
        existing_cols = [c for c in user_columns if c in df.columns]
        return df[existing_cols].copy()

    def _clean_age(self, df):
        if 'age' in df.columns:
            df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan
            df['age'] = df['age'].fillna(df['age'].median())
        return df

    def _add_age_group(self, df):
        if 'age' in df.columns:
            df['age_group'] = (df['age'] // 10).fillna(0).astype(int)
        return df

    def _fill_monthly_spend_nan(self, df):
        if 'monthly_spend' in df.columns:
            df['monthly_spend'] = df['monthly_spend'].fillna(0)
        return df

    def _process_dates(self, df, reference_date):
        if "subscription_start_date" in df.columns:
            df["subscription_start_date"] = pd.to_datetime(df["subscription_start_date"], errors='coerce')
            df["subscription_start_date"] = df["subscription_start_date"].fillna(reference_date)

            tenure = (reference_date - df["subscription_start_date"])
            df["subscription_tenure_days"] = tenure.dt.days
            return df.drop(columns=['subscription_start_date'])
        return df

    def _load_watch_features(self):
        """기계산된 시청 기록 특징 로드 (없으면 생성 로직 유도 가능)"""
        if not self.watch_features_path.exists():
            # 여기서 외부 generate 함수를 부르거나 에러를 낼 수 있음
            raise FileNotFoundError(f"❌ Watch features file not found: {self.watch_features_path}")

        watch_features = pd.read_csv(self.watch_features_path)

        # 기준 날짜 추출
        ref_date = pd.Timestamp.now()
        if 'last_watch_date_ref' in watch_features.columns:
            ref_date = pd.to_datetime(watch_features['last_watch_date_ref'].iloc[0])

        return watch_features, ref_date

    # --- [Main Pipeline] ---

    def run_full_pipeline(self, raw_df):
        """Raw 데이터로부터 최종 모델 입력(X) 제작"""

        # 1. 시청 기록 데이터 및 기준일 로드
        watch_features, reference_date = self._load_watch_features()

        # 2. 유저 정보 전처리 (Pipe 체이닝 활용)
        processed_df = (
            raw_df.pipe(self._select_columns)
            .pipe(self._clean_age)
            .pipe(self._add_age_group)
            .pipe(self._fill_monthly_spend_nan)
            .pipe(self._process_dates, reference_date)
        )

        # 3. 시청 기록 결합 (Merge)
        watch_cols = [c for c in watch_features.columns if c != 'last_watch_date_ref']
        inference_data = (
            processed_df
            .merge(watch_features[watch_cols], on="user_id", how="left")
            .fillna(0)
        )

        # 4. 결측값 및 특수 케이스 처리 (예: 시청 기록 없는 유저)
        if "days_since_last_watch" in inference_data.columns:
            inference_data["days_since_last_watch"] = inference_data["days_since_last_watch"].replace(0, 999)

        return inference_data

    def load_train_data(self, raw_df):
        if 'user_id' in raw_df.columns:
            train_df = raw_df.drop(columns=['user_id'])

        # is_churn을 사용하는 경우
        #if config["inverse_target"]:

        # churn 변수 생성
        train_df['is_churned'] = 1 - train_df['is_active']
        # 기존 변수 제거
        final_train_df = train_df.drop(columns=['is_active'])

        X = final_train_df.drop(columns=['is_churned'])
        y = final_train_df['is_churned']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_cfg.get('test_size', 0.2),
            random_state=self.model_cfg.get('random_state', 42),
            shuffle=self.model_cfg.get('shuffle', True)
        )

        return X_train, X_test, y_train, y_test



