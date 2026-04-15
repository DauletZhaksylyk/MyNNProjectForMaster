import argparse
import json

from FraudDetector import FraudDetector


def main():
    parser = argparse.ArgumentParser(description="Quick local test for post-call fraud analysis.")
    parser.add_argument("--model-dir", required=True, help="Path to the trained model directory.")
    parser.add_argument("--text", default="", help="Single text or transcript to analyze.")
    parser.add_argument("--file", default="", help="Path to a text file with a transcript.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as file:
            transcript = file.read()
    else:
        transcript = args.text

    if not transcript.strip():
        raise ValueError("Provide --text or --file with a transcript.")

    detector = FraudDetector(args.model_dir)
    result = detector.analyze_call(transcript)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
