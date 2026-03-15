import argparse
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src" / "01_data" / "00_ingestion"))
sys.path.insert(0, str(ROOT / "02_src" / "01_data" / "01_preprocessing"))

from data_loader import DataLoader
from data_processor import DataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="중간 데이터로부터 모델 입력 데이터 생성")
    parser.add_argument("input", type=str, help="01_interim 폴더 내 입력 파일명")
    parser.add_argument("-o", "--output", type=str, help="02_processed 폴더 내 저장할 파일명 (기본값: input명과 동일)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_name = args.output if args.output else args.input

    # 파일 경로 설정
    interim_path = ROOT / "00_data" / "01_interim" / args.input
    processed_dir = ROOT / "00_data" / "02_processed"

    # 데이터 로드
    interim_df = DataLoader.load_csv(interim_path)

    # 피쳐 생성
    final_df = DataProcessor().build_features(interim_df)

    # 5. 저장
    output_path = processed_dir / output_name
    final_df.to_csv(output_path, index=False)

    print(f"✅ [Stage 03] 모델 학습 데이터 생성 완료: {output_path} (Shape: {final_df.shape})")


if __name__ == "__main__":
    main()