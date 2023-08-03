import pandas as pd

def main():
        
    names_labels = pd.read_csv('./data/raw/NR-ER-train/names_labels.csv', header=None)
    names_labels.sort_values(by=0, inplace=True)

    names_smiles = pd.read_csv('./data/raw/NR-ER-train/names_smiles.csv', header=None)
    names_smiles.sort_values(by=0, inplace=True)

    data = pd.DataFrame()
    data['id'] = names_labels[0]
    data['smiles'] = names_smiles[1]
    data['label'] = names_labels[1]
    data.to_csv('./data/raw/train.csv')
    print(data.shape)

    names_labels = pd.read_csv('./data/raw/NR-ER-test/names_labels.csv', header=None)
    names_labels.sort_values(by=0, inplace=True)

    names_smiles = pd.read_csv('./data/raw/NR-ER-test/names_smiles.csv', header=None)
    names_smiles.sort_values(by=0, inplace=True)

    data = pd.DataFrame()
    data['id'] = names_labels[0]
    data['smiles'] = names_smiles[1]
    data['label'] = names_labels[1]
    data.to_csv('./data/raw/test.csv')
    print(data.shape)

if __name__ == "__main__":
    main()