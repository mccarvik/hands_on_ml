import os, hashlib, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
# from sklearn.preprocessing import CategoricalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression


HOUSING_PATH = os.path.join("datasets", "housing")
PNG_PATH = '/home/ubuntu/workspace/hands_on_ml/png/2ch/'

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


def select_and_train():
    housing = load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    housing_num = housing.drop('ocean_proximity', axis=1)
    
    num_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    lin_reg = LinearRegression()
    lin_reg.fit(housing_num_tr, housing_labels)
    
    some_data = housing_num.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = num_pipeline.transform(some_data)

    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))
    print(some_data_prepared)


def run():
    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())
    
    # housing.hist(bins=50, figsize=(20,15))
    # plt.savefig(PNG_PATH + "attribute_histogram_plots.png", dpi=300)
    # plt.close()
    
    # train_set, test_set = split_train_test(housing, 0.2)
    # print(len(train_set), "train +", len(test_set), "test")

    # housing_with_id = housing.reset_index()   # adds an `index` column
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    
    # housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    # print(len(train_set), "train +", len(test_set), "test")
    
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # Divide by 1.5 to limit the number of income categories
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    # Label those above 5 as 5
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    print(housing["income_cat"].value_counts())

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    print(housing["income_cat"].value_counts() / len(housing))


def prepare_data():
    housing = load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    # print(sample_incomplete_rows)
    
    # sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1 - get rid of rows missing data
    # sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2 - get rid of attribute
    # median = housing["total_bedrooms"].median()
    # sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3 - fill miss data with some value

    imputer = Imputer(strategy="median")
    housing_num = housing.drop('ocean_proximity', axis=1)
    imputer.fit(housing_num)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)
    # print(imputer.statistics_)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index = list(housing.index.values))
    # print(housing_tr.loc[sample_incomplete_rows.index.values])
    
    housing_cat = housing['ocean_proximity']
    housing_cat_encoded, housing_categories = housing_cat.factorize()
    # housing_cat_encoded[:10]
    # print(housing_categories)

    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
    print(housing_cat_1hot.toarray())
    
    # Not released until sklearn 0.20
    # cat_encoder = CategoricalEncoder()
    # housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
    # housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
    # print(housing_cat_1hot)
    

def explore_data():
    housing = load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    housing = strat_train_set.copy()
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # plt.savefig(PNG_PATH + "bad_visualization_plot", dpi=300)
    
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #     s=housing["population"]/100, label="population", figsize=(10,7),
    #     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    # plt.legend()
    # plt.savefig(PNG_PATH + "housing_prices_scatterplot", dpi=300)
    
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # attributes = ["median_house_value", "median_income", "total_rooms",
    #           "housing_median_age"]
    # scatter_matrix(housing[attributes], figsize=(12, 8))
    # plt.savefig(PNG_PATH + "scatter_matrix_plot", dpi=300)
    
    # housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    # plt.axis([0, 16, 0, 550000])
    # plt.savefig(PNG_PATH + "income_vs_house_value_scatterplot", dpi=300)
    
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def transformers():
    housing = load_housing_data()
    housing_num = housing.drop('ocean_proximity', axis=1)
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)
    # housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
    # print(housing_extra_attribs.head())
    
    num_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr)
    
    # None of this will work without CategoricalEncoder part of sklearn yet
    # num_attribs = list(housing_num)
    # cat_attribs = ["ocean_proximity"]

    # num_pipeline = Pipeline([
    #     ('selector', DataFrameSelector(num_attribs)),
    #     ('imputer', Imputer(strategy="median")),
    #     ('attribs_adder', CombinedAttributesAdder()),
    #     ('std_scaler', StandardScaler()),
    # ])

    # cat_pipeline = Pipeline([
    #     ('selector', DataFrameSelector(cat_attribs)),
    #     ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    # ])
    
    # full_pipeline = FeatureUnion(transformer_list=[
    #     ("num_pipeline", num_pipeline),
    #     ("cat_pipeline", cat_pipeline),
    # ])
    # housing_prepared = full_pipeline.fit_transform(housing)
    # print(housing_prepared)


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

    
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self  # nothing else to do
        
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.attribute_names].values


if __name__ == '__main__':
    # run()
    # explore_data()
    # prepare_data()
    # transformers()
    select_and_train()