"""
train the sentiment model
    - Load the training data from `data/train.csv`.
    - Train the model using the trainer module.
    - Save the trained model to `src/sentiment_model.pkl`.

@goge052215
"""

import os
import pandas as pd
import trainer

def train():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    model_path = os.path.join(base_dir, 'src', 'sentiment_model.pkl')

    print(f"Loading data from {data_dir}...")
    try:
        train_df = pd.read_csv(train_path, encoding='latin1')
        test_df = pd.read_csv(test_path, encoding='latin1')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Initializing trainer...")
    t = trainer.Trainer(train_df)
    
    print("Training model...")
    metrics = t.train(test_df)
    print("Training complete. Metrics:")
    print(metrics)
    
    print(f"Saving model to {model_path}...")
    t.save_model(model_path)
    print("Done.")

if __name__ == "__main__":
    train()
