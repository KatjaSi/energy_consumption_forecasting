import numpy as np
import pandas as pd

def get_X_samples(X_sequence: pd.DataFrame, n_steps_in = 24):
    """
    Funtion to prepare the time series data for the LSTM layer input (to be used for training)
    sequence is a 2 D array of sumples,
    each row is a row of features corresponding to one time step
    """
    try:
        X = X_sequence.values
    except:
        X = X_sequence
    X = np.array([X[i:i+n_steps_in] for i in range(len(X)-n_steps_in+1)])
    X = np.asarray(X).astype('float32')
    return X


def main():
    #df_consumption, df_weather, df = read_consumption_and_weather()
    #df = df_consumption.NO1[:4]
    columns = ["A", "B", "C"]
    data = np.array([[1, 3, 2], [3, 3, 5],[5, 4, 5], [7, 6, 7]])
    n_steps_in = 2
    arr =np.array([data[i:i+n_steps_in] for i in range(len(data)-n_steps_in+1)])
    print(arr.shape)
    #df = pd.DataFrame(data=data, columns=columns)
    #print(data)
# Using the special variable
# __name__
if __name__=="__main__":
    main()