import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path, test_size=0.1):
    data = pd.read_csv(file_path)
    train, test = train_test_split(data, test_size=test_size, random_state=42) 

    # Save the data
    train.to_csv('q_pair_train.csv', index=False)
    test.to_csv('q_pair_test.csv', index=False)

if __name__ == "__main__":
    file_path = 'question_pairs.csv'
    split_data(file_path)