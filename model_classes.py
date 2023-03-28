import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

import seaborn as sns
import pickle
import os

from sklearn.pipeline import Pipeline
from keras.models import load_model

from standarizer import Standarizer
from pipeline_classes import LagsAdder, RowRemover, WindowFeatureAdder
from pipeline_classes import ColumnRemover,  X_y_splitter, X_y_sampler

from preprocess import get_X_samples

def create_base_model():
    base_model = Sequential()
    #model.add(Dropout(0.2))
    base_model.add(Input(shape=(34,)))
    base_model.add(Dense(64, 'relu' ))
    #base_model.add(Dropout(0.2))
    #base_model.add(Dense(256, 'relu'))
    #base_model.add(Dropout(0.2))
    base_model.add(Dense(32, 'relu'))
    base_model.add(Dense(1, 'linear'))
    optimizer = keras.optimizers.Adam(lr=0.001)
    base_model.compile(optimizer=optimizer, loss='mse')
    return base_model


class ModelWrapper:

    def __init__(self, model): #train and valid set are not splittet
        self.model = model
        #self.train_set = train_set.copy()
        #self.valid_set = valid_set.copy()
        self.stz = Standarizer()

    def scale_and_fit(self, epochs=80, verbose=1):
        pass

    def plot_learning_history(self):
        history = self.history
        plt.rc('figure',figsize=(10,6))
        plt.plot(history['loss'], 'r', label='Training loss')
        plt.plot(history['val_loss'], 'g', label='Validation loss')
        plt.title('Training VS Validation loss')
        plt.xlabel('No. of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def get_standarizer(self):
        return self.stz
    
    def plot_statistics(self, X_test, y_test, graph_1=True, graph_2=True, save=True):
        """
        # graph_1 is ordinal pyplotlib plot, graph_2 is seaborn plot illustrating mean and std
        """ 
        rand_indeces= np.random.randint(0,len(X_test)-48,100)
        all_preds = [self.predict_one_sequence(X_test, at_index=i) for i in rand_indeces]
        all_preds = np.array(all_preds)
        predictions = [self.stz.inverse_one_column('consumption', all_preds[i]) for i in range(len(all_preds))]
        real_consumptions = np.array([y_test[i:i+24] for i in rand_indeces])
        predictions = np.array(predictions)
        real_consumptions = [self.stz.inverse_one_column('consumption', real_consumptions[i]) for i in range(len(real_consumptions))]
        errors = np.array([np.abs(real_consumptions[i]-predictions[i]) for i in range(len(real_consumptions))])
        mean_errors = np.mean(errors, axis = 0)
        stds = np.std(errors, axis = 0)
        if graph_1:
            plt.plot(mean_errors, 'r', label='Mean errors')
            plt.plot(stds, 'g', label='Standard deviations')
            plt.title('Mean errors && Standard deviations')
            plt.xlabel('Hr')
            plt.ylabel('Error measurers')
            plt.legend()
            plt.show()
        if graph_2:
            fig, ax = plt.subplots(figsize=(16,8))
            sns.boxplot(data=errors, orient='v', showfliers=False)
        if save:
            self.mean_errors = mean_errors
            self.stds = stds
            self.errors = errors

    def plot_saved_statistics(self, graph_1=True, graph_2=True):
        if graph_1:
            plt.plot(self.mean_errors, 'r', label='Mean errors')
            plt.plot(self.stds, 'g', label='Standard deviations')
            plt.title('Mean errors && Standard deviations')
            plt.xlabel('Hr')
            plt.ylabel('Error measurers')
            plt.legend()
            plt.show()
        if graph_2:
            fig, ax = plt.subplots(figsize=(16,8))
            sns.boxplot(data=self.errors, orient='v', showfliers=False)


    def save(self, path): 
        os.makedirs(path, exist_ok=True) 
        learning_history_path = path + '/history.pkl'
        model_path = path + '/model'
        self.model.save(model_path)
        stz_path = path +'/stz.pkl'
        with open(learning_history_path, 'wb') as f:
            pickle.dump(self.history, f)
        with open(stz_path, 'wb') as f:
            pickle.dump(self.stz, f)
        try:
            with open(path+'/mean_errors.pkl', 'wb') as f:
                pickle.dump(self.mean_errors, f)
            with open(path+'/stds.pkl', 'wb') as f:
                pickle.dump(self.stds, f)
            with open(path+'/errors.pkl', 'wb') as f:
                pickle.dump(self.errors, f)
            Warning("Statistics saved")
        except:
            Warning("No statistics saved")

    def get_set_mean_errors(self, X, y):
        rand_indeces= np.random.randint(0,len(X)-48,10)
        all_preds = [self.predict_one_sequence(X, at_index=i) for i in rand_indeces]
        all_preds = np.array(all_preds)
        predictions = [self.stz.inverse_one_column('consumption', all_preds[i]) for i in range(len(all_preds))]
        real_consumptions = np.array([y[i:i+24] for i in rand_indeces])
        predictions = np.array(predictions)
        real_consumptions = [self.stz.inverse_one_column('consumption', real_consumptions[i]) for i in range(len(real_consumptions))]
        errors = np.array([np.abs(real_consumptions[i]-predictions[i]) for i in range(len(real_consumptions))])
        mean_errors = np.mean(errors, axis = 0)
        return mean_errors

class BaseModelWrapper(ModelWrapper):

    def __init__(self, model):
        super().__init__(model)
        self.base_pipe = Pipeline([
            ('add_lags', LagsAdder(freqs=[f'{i}H' for i in range(1,25)]) ),
           # ('add_window_features', WindowFeatureAdder(24, ['mean'])),
            ('remove_rows',  RowRemover(168)),
            ('remove_cols', ColumnRemover()),
            ('X_y_split',     X_y_splitter())
        ])

    def scale_and_fit(self, train_set, valid_set, epochs=80, verbose=1):
        self.stz.fit(train_set, columns=['consumption', 'temperature'])
        train_set = self.stz.transform(train_set)
        valid_set = self.stz.transform(valid_set)  
        X_train, y_train = self.base_pipe.fit_transform(train_set)
        X_val, y_val = self.base_pipe.transform(valid_set)
        
        history = self.model.fit(X_train.values.astype('float32'), y_train.values.astype('float32'), 
            validation_data = (X_val.values.astype('float32'), y_val.values.astype('float32')), epochs=epochs, verbose=verbose)
        
        self.history = history.history # dictionary, keys = 'loss', 'val_loss'
        
    def scale_and_transform(self, test_set):
        self.stz.transform(test_set)
        test_set = self.stz.transform(test_set)
        X_test, y_test = self.base_pipe.transform(test_set)
        
        return X_test, y_test
        
    def predict_one_sequence(self, X, at_index, horizont=24):
        # Predicts one sequence of consumptions for standarized and pipelined X
        predictions = list()
        x_input = X.iloc[at_index:at_index+1]
        x_copy = x_input #.values.astype('float32').reshape((1,-1))
        prediction = self.model.predict(x_copy.values.astype('float32').reshape((1,-1)), verbose=0)
        predictions.append(prediction[0][0])
        for i in range(1, horizont):
            # shift > one day
            timereference = X.index[at_index+i]
            x_copy.loc[timereference] = X.loc[timereference]
            for j in range(1, i+1):
                x_copy.at[timereference,f'consumption_lag_{j}H'] =  predictions[-j]
            prediction = self.model.predict(x_copy.loc[timereference].values.astype('float32').reshape((1,-1)), verbose=0)
            predictions.append(prediction[0][0])
        return predictions
        

    def draw_predictions(self, X, y, start_time):
        # X and y are scaled
        start_index = X.index.get_loc(start_time)
        real_consumptions = np.array(y[start_index:start_index+48])
        real_consumptions = self.stz.inverse_one_column('consumption', real_consumptions)
        predictions = self.predict_one_sequence(X, at_index=start_index+24)
        predictions = self.stz.inverse_one_column('consumption', predictions)
        predictions = np.concatenate((np.array([None]*24), predictions), axis=0)
        plt.rc('figure',figsize=(12,8))
        plt.plot(predictions, 'g', label='Predictions')
        plt.plot(real_consumptions, 'r', label='Real consumptions')
        plt.title('Real consumptions VS Predictions')
        plt.xlabel('Hr')
        plt.ylabel('Consumption')
        plt.legend()
        plt.show()


    @staticmethod
    def load(path):
        stz_path = path +'/stz.pkl'
        model_path = path +'/model'
        loaded_model = load_model(model_path)
        mr = BaseModelWrapper(loaded_model)
        with open(stz_path, 'rb') as f:
            stz = pickle.load(f)
            mr.stz = stz
        if os.path.exists(path+'/mean_errors.pkl'):
            with open(path+'/mean_errors.pkl', 'rb') as f:
                mean_errors = pickle.load(f)
                mr.mean_errors = mean_errors
        if os.path.exists(path+'/stds.pkl'):
            with open(path+'/stds.pkl', 'rb') as f:
                stds = pickle.load(f)
                mr.stds = stds
        if os.path.exists(path+'/errors.pkl'):
            with open(path+'/errors.pkl', 'rb') as f:
                errors = pickle.load(f)
                mr.errors = errors
        if os.path.exists(path+'/history.pkl'):
            with open(path+'/history.pkl', 'rb') as f:
                history = pickle.load(f)
                mr.history = history
        return mr



class LSTMModelWrapper(ModelWrapper):

    def __init__(self, model):
        super().__init__(model)
        self.lstm_pipe = Pipeline([
            ('add_lags', LagsAdder(freqs=['1H', '24H', '168H']) ),
            ('add_window_features', WindowFeatureAdder(24, ['mean'])),
            ('remove_rows',  RowRemover(168)),
            ('remove_cols', ColumnRemover()),
            ('X_y_split',     X_y_splitter())
        ])

    def scale_and_fit(self, train_set, valid_set, epochs=80, verbose=1):
        self.stz.fit(train_set, columns=['consumption', 'temperature'])
        train_set = self.stz.transform(train_set)
        valid_set = self.stz.transform(valid_set)  

        X, y = self.lstm_pipe.fit_transform(train_set)
        X = get_X_samples(X)
        y = y[23:]

        X_val, y_val = self.lstm_pipe.transform(valid_set)
        X_val = get_X_samples(X_val)
        y_val = y_val[23:]

        
        history = self.model.fit(X,  y, 
            validation_data = (X_val, y_val), epochs=epochs, verbose=verbose)
        
        self.history = history.history # dictionary, keys = 'loss', 'val_loss'

    def scale_and_transform(self, test_set):
        test_set = self.stz.transform(test_set)
        X_test, y_test = self.lstm_pipe.transform(test_set)
        #X_test = get_X_samples(X_test)
        #y_test = y_test[23:]
        return X_test, y_test


    def predict_one_sequence(self, X, at_index, horizont=24): # at_index=start_index
        predictions = list()
        x_copy = X[at_index:at_index+24]
        x_samples = get_X_samples(x_copy, n_steps_in = 24)
        prediction = self.model.predict(x_samples, verbose=0)
        predictions.append(prediction[0][0])
        for i in range(horizont-1):
            # shift > one day
            timereference = X.index[at_index+24+i]
            x_copy.loc[timereference] = X.loc[timereference]
            x_copy = x_copy[1:]
            x_copy.at[timereference, 'consumption_lag_1H'] =  predictions[-1] 
            x_copy.at[timereference, 'consumption_mean_24H'] = np.mean(x_copy.iloc[-24:].consumption_lag_1H)
            x_samples = get_X_samples(x_copy, n_steps_in = 24)   
            prediction = self.model.predict(x_samples, verbose=0)
            predictions.append(prediction[0][0])
        return predictions
    
    def draw_predictions(self, X, y, start_time):
        start_index = X.index.get_loc(start_time)
        real_consumptions = np.array(y[start_index:start_index+48])
        real_consumptions = self.stz.inverse_one_column('consumption', real_consumptions)
        predictions = self.predict_one_sequence(X, at_index=start_index)
        predictions = self.stz.inverse_one_column('consumption', predictions)
        predictions = np.concatenate((np.array([None]*24), predictions), axis=0)
        #predictions = np.concatenate((real_consumptions[:24], predictions), axis=0)
        plt.plot(predictions, 'g', label='Predictions')
        plt.plot(real_consumptions, 'r', label='Real consumptions')
        plt.title('Real consumptions VS Predictions')
        plt.xlabel('Hr')
        plt.ylabel('Consumption')
        plt.legend()
        plt.show()


    @staticmethod
    def load(path):
        stz_path = path +'/stz.pkl'
        model_path = path +'/model'
        loaded_model = load_model(model_path)
        mr = LSTMModelWrapper(loaded_model)
        with open(stz_path, 'rb') as f:
            stz = pickle.load(f)
            mr.stz = stz
        if os.path.exists(path+'/mean_errors.pkl'):
            with open(path+'/mean_errors.pkl', 'rb') as f:
                mean_errors = pickle.load(f)
                mr.mean_errors = mean_errors
        if os.path.exists(path+'/stds.pkl'):
            with open(path+'/stds.pkl', 'rb') as f:
                stds = pickle.load(f)
                mr.stds = stds
        if os.path.exists(path+'/errors.pkl'):
            with open(path+'/errors.pkl', 'rb') as f:
                errors = pickle.load(f)
                mr.errors = errors
        if os.path.exists(path+'/history.pkl'):
            with open(path+'/history.pkl', 'rb') as f:
                history = pickle.load(f)
                mr.history = history
        return mr




class CNNModelWrapper(ModelWrapper):

    def __init__(self, model):
        super().__init__(model)
        self.cnn_pipe = Pipeline([
            ('add_lags', LagsAdder(freqs=['1H', '24H', '168H']) ),
            ('add_window_features', WindowFeatureAdder(24, ['mean'])),
            ('remove_rows',  RowRemover(168)),
            ('remove_cols', ColumnRemover()),
            ('X_y_split',     X_y_splitter())
        ])

    def scale_and_fit(self, train_set, valid_set, epochs=80, verbose=1):
        self.stz.fit(train_set, columns=['consumption', 'temperature'])
        train_set = self.stz.transform(train_set)
        valid_set = self.stz.transform(valid_set)  

        X, y = self.cnn_pipe.fit_transform(train_set)
        X = get_X_samples(X)
        y = y[23:]

        X_val, y_val = self.cnn_pipe.transform(valid_set)
        X_val = get_X_samples(X_val)
        y_val = y_val[23:]


        ##### difference from LSTM #####
        x_tempreature = X[:, :, 0].reshape(X.shape[0], X.shape[1], 1)
        lag_1H_x = X[:,:,10].reshape(X.shape[0], X.shape[1], 1)
        static_features_x = X[:, :, 1:10]
        lags_x =  X[:, :, 11:14].reshape(X.shape[0], X.shape[1], 3)

        x_tempreature_val = X_val[:, :, 0].reshape(X_val.shape[0], X_val.shape[1], 1)
        lag_1H_x_val = X_val[:,:,10].reshape(X_val.shape[0], X_val.shape[1], 1)
        static_features_x_val = X_val[:, :, 1:10] 
        lags_x_val =  X_val[:, :, 11:14].reshape(X_val.shape[0], X_val.shape[1], 3)
        
        history = self.model.fit([x_tempreature, lag_1H_x, lags_x, static_features_x], \
                                        y,validation_data=([x_tempreature_val, lag_1H_x_val, lags_x_val, static_features_x_val], y_val),\
                                        epochs=epochs, verbose=verbose)
        
        self.history = history.history # dictionary, keys = 'loss', 'val_loss'

    def scale_and_transform(self, test_set):
        test_set = self.stz.transform(test_set)
        X_test, y_test = self.cnn_pipe.transform(test_set)
        #X_test = get_X_samples(X_test)
        #y_test = y_test[23:]
        return X_test, y_test


    def predict_one_sequence(self, X, at_index, horizont=24): # at_index=start_index
        predictions = list()
        x_copy = X[at_index:at_index+24]
        x_samples = get_X_samples(x_copy, n_steps_in = 24)

        x_tempreature = x_samples[:, :, 0].reshape(x_samples.shape[0], x_samples.shape[1], 1)
        lag_1H_x = x_samples[:,:,10].reshape(x_samples.shape[0], x_samples.shape[1], 1)
        static_features_x = x_samples[:, :, 1:10]
        lags_x =  x_samples[:, :, 11:14].reshape(x_samples.shape[0], x_samples.shape[1], 3)


        prediction = self.model.predict([x_tempreature, lag_1H_x, lags_x, static_features_x], verbose=0)
        predictions.append(prediction[0][0])
        for i in range(horizont-1):
            # shift > one day
            timereference = X.index[at_index+24+i]
            x_copy.loc[timereference] = X.loc[timereference]
            x_copy = x_copy[1:]
            x_copy.at[timereference, 'consumption_lag_1H'] =  predictions[-1] 
            x_copy.at[timereference, 'consumption_mean_24H'] = np.mean(x_copy.iloc[-24:].consumption_lag_1H)
            x_samples = get_X_samples(x_copy, n_steps_in = 24)   

            x_tempreature = x_samples[:, :, 0].reshape(x_samples.shape[0], x_samples.shape[1], 1)
            lag_1H_x = x_samples[:,:,10].reshape(x_samples.shape[0], x_samples.shape[1], 1)
            static_features_x = x_samples[:, :, 1:10]
            lags_x =  x_samples[:, :, 11:14].reshape(x_samples.shape[0], x_samples.shape[1], 3)

            prediction = self.model.predict([x_tempreature, lag_1H_x, lags_x, static_features_x], verbose=0)
            predictions.append(prediction[0][0])
        return predictions
    
    def draw_predictions(self, X, y, start_time):
        start_index = X.index.get_loc(start_time)
        real_consumptions = np.array(y[start_index:start_index+48])
        real_consumptions = self.stz.inverse_one_column('consumption', real_consumptions)
        predictions = self.predict_one_sequence(X, at_index=start_index)
        predictions = self.stz.inverse_one_column('consumption', predictions)
        predictions = np.concatenate((np.array([None]*24), predictions), axis=0)
        #predictions = np.concatenate((real_consumptions[:24], predictions), axis=0)
        plt.plot(predictions, 'g', label='Predictions')
        plt.plot(real_consumptions, 'r', label='Real consumptions')
        plt.title('Real consumptions VS Predictions')
        plt.xlabel('Hr')
        plt.ylabel('Consumption')
        plt.legend()
        plt.show()


    @staticmethod
    def load(path):
        stz_path = path +'/stz.pkl'
        model_path = path +'/model'
        loaded_model = load_model(model_path)
        mr = CNNModelWrapper(loaded_model)
        with open(stz_path, 'rb') as f:
            stz = pickle.load(f)
            mr.stz = stz
        if os.path.exists(path+'/mean_errors.pkl'):
            with open(path+'/mean_errors.pkl', 'rb') as f:
                mean_errors = pickle.load(f)
                mr.mean_errors = mean_errors
        if os.path.exists(path+'/stds.pkl'):
            with open(path+'/stds.pkl', 'rb') as f:
                stds = pickle.load(f)
                mr.stds = stds
        if os.path.exists(path+'/errors.pkl'):
            with open(path+'/errors.pkl', 'rb') as f:
                errors = pickle.load(f)
                mr.errors = errors
        if os.path.exists(path+'/history.pkl'):
            with open(path+'/history.pkl', 'rb') as f:
                history = pickle.load(f)
                mr.history = history
        return mr