import argparse
import sys
from pathlib import Path

# 프로젝트 루트 설정 및 모듈 경로 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src" / "01_data" / "00_ingestion"))
sys.path.insert(1, str(ROOT / "02_src" / "01_data" / "01_preprocessing"))

from data_loader import DataLoader
from data_processor import DataProcessor
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Raw 데이터 정제 및 Train/Test 자동 분리")
    parser.add_argument("filename", type=str, help="00_raw 폴더 내 입력 파일명 (예: user.csv)")
    parser.add_argument(
        "-s", "--test_size",
        type=float,
        default=0.2,
        help="테스트 셋 비율 (0.0 ~ 1.0 사이 소수점, 기본값: 0.2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0 < args.test_size < 1):
        print(f"❌ Error: test_size는 0과 1 사이여야 합니다. 입력값: {args.test_size}")
        return

    # 입력 파일명에서 확장자 제거
    input_path = Path(args.filename)
    prefix = input_path.stem

    # 경로 설정
    raw_path = ROOT / "00_data" / "00_raw" / args.filename
    interim_dir = ROOT / "00_data" / "01_interim"

    # 파일 존재 여부 확인
    if not raw_path.exists():
        print(f"❌ Error: 파일을 찾을 수 없습니다: {raw_path}")
        return

    # 4. 데이터 로드
    raw_df = DataLoader.load_csv(raw_path)

    # 데이터 정제
    cleaned_df = DataProcessor().clean_user_data(raw_df)

    # Train / Test 분리
    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=args.test_size,
        stratify=cleaned_df['is_active'],
        random_state=42,
        shuffle=True
    )
    # 분리된 dataset 저장
    train_out = interim_dir / f"{prefix}_train.csv"
    test_out = interim_dir / f"{prefix}_test.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"✅ [Stage 02] 작업 완료: '{args.filename}' -> 분리 및 저장 성공")
    print(f"   - 훈련용: {train_out.name} ({len(train_df)} rows)")
    print(f"   - 테스트용: {test_out.name} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()