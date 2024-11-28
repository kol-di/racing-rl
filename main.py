from racingrl.train import train
from racingrl.inference import inference
import argparse


def launch():
    parser = argparse.ArgumentParser(description="Run the script in train or inference mode.")
    parser.add_argument(
        "mode", 
        choices=["train", "inference"], 
        help="Specify the mode to run: 'train' or 'inference'."
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "inference":
        inference()
        

if __name__ == '__main__':
    launch()