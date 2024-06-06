import json
import pandas as pd
import os
import argparse
import os, sys
import glob
from tqdm import tqdm
import csv


def process(args):
    jsonl_files = [os.path.splitext(f)[0] for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    for jsonl_file in jsonl_files:
        input_file = os.path.join(args.input_dir, f"{jsonl_file}.jsonl")
        output_file = os.path.join(args.input_dir, f"{jsonl_file}_clean.jsonl")
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in fin:
                data = json.loads(line.strip())
                output = data["output"]
                if output.startswith("I'm sorry, but I cannot"):
                    continue
                if output.startswith("I'm sorry, but I'm not sure"):
                    continue
                if "as an ai language model" in output.lower():
                    continue
                if "i'm sorry, but i" in output.lower():
                    continue
                if "I'm not sure what you" in output:
                    continue
                json_string = json.dumps(data, ensure_ascii=False)
                fout.write(json_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to the input directory", required=True)
    
    args = parser.parse_args()

    process(args)