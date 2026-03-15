import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "00_data" / "00_raw"

sys.path.insert(0, str(ROOT / "02_src" / "01_data" / "00_ingestion"))

from data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Raw Data Validation Step")
    parser.add_argument("filename", type=str, help="검증할 raw 데이터 파일명")
    args = parser.parse_args()

    #데이터 로드
    raw_path = RAW_DATA_DIR / args.filename
    try:
        raw_df = DataLoader.load_csv(raw_path)
        print(f"🔍 Validating raw data: {args.filename}...")

        #데이터 검증
        is_ok, missing = DataLoader.validate_df(raw_df, mode = 'raw')

        if not is_ok:
            print(f"🛑 [Stage 01] Raw 데이터 검증 실패! 누락 컬럼: {missing}")
            sys.exit(1)

        print("\n✅ [Stage 01] Raw 데이터 검증 성공!")
        return

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)  # 파이프라인 중단을 위한 에러 코드 반환

if __name__ == "__main__":
    main()