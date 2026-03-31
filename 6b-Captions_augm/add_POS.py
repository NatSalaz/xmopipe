import os
import yaml
import argparse
import spacy
from pathlib import Path

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["captions_augm"]

nlp = spacy.load("en_core_web_sm")


def pos_tag_format(sentence):
    doc = nlp(sentence)
    tags = " ".join([f"{token.text}/{token.pos_}" for token in doc])
    return f"#{tags}#0.0#0.0"


def main():
    parser = argparse.ArgumentParser(description="Adds POS for text captions")
    parser.add_argument("--input", required=True, help="Input path to dataset's text")
    parser.add_argument("--output", default=_cfg["pos_output_dir"], help="Output path")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r") as f:
                lines = f.read().splitlines()

            processed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    processed_lines.append(f"{line} {pos_tag_format(line)}")
                else:
                    processed_lines.append("")
            with open(output_path, "w") as f:
                f.write("\n".join(processed_lines))

            print(f"Processed: {filename}")

    print("Done.")


if __name__ == "__main__":
    main()
