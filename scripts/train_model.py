# scripts/train_model.py
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the dataset file")
    parser.add_argument("--save_path", required=True, help="Path to save the trained model")
    args = parser.parse_args()

    positions, forces = load_data(args.data_path)
    positions, forces = preprocess_data(positions, forces)
    train_model(positions, forces, args.save_path)
