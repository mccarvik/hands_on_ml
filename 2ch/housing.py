import os
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HOUSING_PATH = os.path.join("datasets", "housing")
PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/2ch/'

def run():
    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())
    
    housing.hist(bins=50, figsize=(20,15))
    plt.savefig(PNG_PATH + "attribute_histogram_plots.png", dpi=300)
    plt.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    

if __name__ == '__main__':
    run()