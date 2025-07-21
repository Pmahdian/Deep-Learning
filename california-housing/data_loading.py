import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    california_housing = fetch_california_housing(as_frame=True)
    print(california_housing.DESCR)
    print(california_housing.frame.head())
    print(california_housing.frame.info())
    return california_housing

if __name__ == "__main__":
    data = load_data()