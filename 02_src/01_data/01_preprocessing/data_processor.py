import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, config = None):
        self.model_cfg = config
        self.target = "is_active"

        # 경로 설정 (ROOT는 utils 등에서 정의한 것을 가져오거나 직접 계산)
        self.root = Path(__file__).parents[3]
        self.interim_dir = self.root / "00_data" / "01_interim"
        self.processed_dir = self.root / "00_data" / "02_processed"
        self.watch_features_path = self.processed_dir / "watch_features.csv"

    def _select_columns(self, df):
        user_columns = ['user_id', 'age', 'plan_tier', 'subscription_start_date', 'monthly_spend']
        existing_cols = [c for c in user_columns if c in df.columns]
        if self.target in df.columns: #학습 or 모델 테스트 데이터인 경우 포함시킴
            existing_cols.append(self.target)
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
        if not self.watch_features_path.exists():
            # 집계한 csv 파일 생성
            self.generate_and_save_watch_features()

        watch_features = pd.read_csv(self.watch_features_path)

        # 기준 날짜 추출
        ref_date = pd.Timestamp.now()
        if 'last_watch_date_ref' in watch_features.columns:
            ref_date = pd.to_datetime(watch_features['last_watch_date_ref'].iloc[0])

        return watch_features, ref_date

    # --- [Main Pipeline] ---

    # user data 전처리
    def clean_user_data(self, df):
        return (df.pipe(self._select_columns)
            .pipe(self._clean_age)
            .pipe(self._add_age_group)
            .pipe(self._fill_monthly_spend_nan))

    def build_features(self, df):

        watch_features, reference_date = self._load_watch_features()
        df = df.pipe(self._process_dates, reference_date)

        watch_cols = [c for c in watch_features.columns if c != 'last_watch_date_ref']
        final_df = (
            df
            .merge(watch_features[watch_cols], on="user_id", how="left")
            .fillna(0)
        )
        if "days_since_last_watch" in final_df.columns:
            final_df["days_since_last_watch"] = final_df["days_since_last_watch"].replace(0, 999)

        return final_df

    def run_full_pipeline(self, raw_df):
        """Raw 데이터로부터 최종 모델 입력(X) 제작"""
        interim_df = self.clean_user_data(raw_df)
        final_df = self.build_features(interim_df)
        return final_df

    def load_train_data(self, raw_df):
        if 'user_id' in raw_df.columns:
            train_df = raw_df.drop(columns=['user_id'])
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

    def generate_and_save_watch_features(self):

        raw_path = self.root / "00_data" / "00_raw" / "netflix_watch_history.csv"  # 수정 예정
        if not raw_path.exists():
            print(f"Error: {raw_path} not found.")
            return

        print("Pre-calculating watch features...")
        history_df = pd.read_csv(raw_path)

        feature_orders =  [
            "user_id", "watch_count", "unique_movies", "total_watch_time", "avg_watch_time", "watch_days",
            "recent_watch_count", "days_since_last_watch", "avg_progress",
            "completion_rate", "download_ratio", "avg_rating"
        ]
        # 전처리
        df = self._process_watch_history(history_df)

        last_date = df['watch_date'].max()

        # 집계
        basic_stats = self._agg_watch_basic_stats(df)
        time_stats = self._calculate_watch_time_features(df, last_date)

        watch_features = pd.concat([basic_stats, time_stats], axis = 1).reset_index()

        # 순서 정렬
        watch_features = watch_features[feature_orders]
        # 기준 날짜 저장 (나중에 추론 시 tenure 계산용)
        watch_features['last_watch_date_ref'] = last_date

        watch_features.to_csv(self.watch_features_path, index=False)
        print(f"Saved pre-calculated features to {self.watch_features_path}")

    def _process_watch_history(self, history_df):
        df = history_df.copy()
        df['watch_date'] = pd.to_datetime(history_df['watch_date'], errors='coerce')
        df["watch_duration_minutes"] = history_df["watch_duration_minutes"].fillna(0)
        df["progress_percentage"] = history_df["progress_percentage"].fillna(0)
        df['completed'] = history_df['progress_percentage'] >= 90
        return df

    def _agg_watch_basic_stats(self, df):
        basic_stats = df.groupby("user_id").agg(
            watch_count=("movie_id", "size"),
            unique_movies=("movie_id", "nunique"),
            total_watch_time=("watch_duration_minutes", "sum"),
            avg_watch_time=("watch_duration_minutes", "mean"),
            watch_days=("watch_date", "nunique"),
            avg_progress=("progress_percentage", "mean"),
            completion_rate=("completed", "mean"),
            download_ratio=("is_download", "mean"),
            avg_rating=("user_rating", "mean")
        )
        return basic_stats

    def _calculate_watch_time_features(self, df, last_date, recent_threshold = 31):
        recent_cutoff = last_date - pd.Timedelta(days = recent_threshold)
        recent_count = (
            df[df["watch_date"] >= recent_cutoff]
            .groupby("user_id")
            .size()
            .rename('recent_watch_count')
        )

        last_watch = df.groupby("user_id")["watch_date"].max()
        days_since_last_watch = (last_date - last_watch).dt.days.rename('days_since_last_watch')

        return pd.concat([recent_count, days_since_last_watch], axis = 1)

if __name__ == '__main__':
    DataProcessor(None).generate_and_save_watch_features()