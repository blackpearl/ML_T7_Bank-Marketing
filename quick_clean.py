import pandas as pd

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file, sep=";")

    df = df[df['job'] != 'unknown']
    df = df[df['marital'] != 'unknown']
    df['education'] = df['education'].replace('unknown', df['education'].mode()[0])
    df['default'] = df['default'].replace('unknown', 'Default_Unknown')

    df.to_csv(output_file, index=False, sep=";")

if __name__ == "__main__":
    clean_dataset("bank-additional-full.csv", "bank-additional-full-cleaned.csv")
