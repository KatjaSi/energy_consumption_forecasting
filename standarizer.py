import numpy as np
import pandas as pd
from typing import List
from pandas.core.frame import DataFrame

class Standarizer:
    
    def __init__(self):
        #self.means = list() # List of means corresponding to the values
        #self.stds =list() # all standard deviations corresponding to the values
        self.column_values = dict() #key is columns name, value is a tuple : (mean, std) for the column
    
    def fit(self, df:DataFrame, columns:List[str]): #define which columns to be standarized
        self.columns = columns # The names of all the calues to be standarized
        for col in columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            #self.means.append(col_mean)
            #self.stds.append(col_std)
            self.column_values[col] = (col_mean, col_std)

    def transform(self, df, inplace=False):
        # returns dataframe where the columns with the names which were fited on are transformed
        if inplace:
            df_ = df
        else:
            df_ = df.copy()
        for col in self.column_values:
            col_mean, col_std = self.column_values[col]
            df_[col] = (df_[col]-col_mean)/col_std
        
        return df_
    
    def inverse(self, df, inplace=False):
        if inplace:
            df_ = df
        else:
            df_ = df.copy()
        for col in self.column_values:
            col_mean, col_std = self.column_values[col]
            df_[col] = df_[col]*col_std+col_mean
        return df_
    
    def inverse_one_column(self, col_name, y): 
        """
        to be used on target value
        """
        col_mean, col_std = self.column_values[col_name]
        return np.array(y)*col_std + col_mean


    
def main():
    columns = ["A", "B", "C"]
    rows = ["D", "E", "F"]
    data = np.array([[1, 2, 2], [3, 3, 5],[5, 4, 4]])
    data1 = np.array([[1, 3, 2], [3, 3, 5],[5, 4, 5]])
    df = pd.DataFrame(data=data1, index=rows, columns=columns)
    stz = Standarizer()
    stz.fit(df, columns=["A", "C"])
    print(df)
    print("transforming..")
    stz.transform(df, inplace=True)
    print(df)
    print("inverse.")
    stz.inverse(df, inplace=True)
    print(df)
if __name__ == "__main__":
    main()