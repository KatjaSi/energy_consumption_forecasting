from pathlib import Path
import numpy as np
import warnings
import pandas as pd
from keras.models import load_model

from preprocess import get_X_samples

warnings.filterwarnings('ignore')


data_path = Path("data")

#function to get one 24 hrs prediction from the x_input
# x_input is a sequence of lenr
# for example 24 or 168 H back in time
# x_slice is the slice of X with the length is 24 or 168
def predict_one_sequence(model, X, start_index, sequence_length=24, horizont=24):
    predictions = list()
    x_copy = X[start_index:start_index+sequence_length]
    x_samples = get_X_samples(x_copy, n_steps_in = 24)
    prediction = model.predict(x_samples, verbose=0)
    predictions.append(prediction[0][0])
    for i in range(horizont-1):
        # shift > one day
        timereference = X.index[start_index+sequence_length+i]
        x_copy.loc[timereference] = X.loc[timereference]
        x_copy = x_copy[1:]
        x_copy.at[timereference, 'consumption_lag_1H'] =  predictions[-1] #TODO: nothing happens here
        old_mean_24H =  x_copy.loc[timereference].consumption_mean_24H
        x_copy.at[timereference, 'consumption_mean_24H'] = np.mean(x_copy.iloc[-24:].consumption_lag_1H)
       # x_copy.at[timereference, 'consumption_mean_168H'] =x_copy.loc[timereference].consumption_mean_168H - \
        #                        24*(old_mean_24H -x_copy.loc[timereference].consumption_mean_24H)/168
        x_samples = get_X_samples(x_copy, n_steps_in = 24)   
        prediction = model.predict(x_samples, verbose=0)
        predictions.append(prediction[0][0])
    return predictions

def main():
    df = pd.read_csv(data_path / "df_NO1_train.csv")
    df.set_index("reference_time", inplace = True)
    #print(df.head())
    X = df.drop(columns=['consumption'])
    y = df['consumption']

    model = load_model('models\model1')
    predictions = predict_one_sequence(model, X, 0, 24, 24)
    print(predictions)

if __name__ == "__main__":
    main()