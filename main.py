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

    # Parse the mode argument first
    args, _ = parser.parse_known_args()

    # Add optional argument only for inference mode
    if args.mode == "inference":
        parser.add_argument(
            "--weights_path", 
            type=str, 
            help="Path to trained model weights."
        )

    # Re-parse arguments now with mode-specific arguments
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "inference":
        inference(model_weights_path=args.weights_path)
        

if __name__ == '__main__':
    launch()