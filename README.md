# energy_consumption_forecasting
 Energy Consumption Prediction using Feed-Forward Network, Multichannel CNN, and 2-layer LSTM Models
 
 # Introduction
 
 The electric power grid is a complex network of power lines, substations and transformers that connects produces to consumers.
 The grid does not store energy, which presents a significant challenge for energy management, as predicting energy consumption accurately is essential 
 for balancing the grid and ensuring a stable power supply.

AI models can help to address this challenge by providing energy consumption predictions. By analyzing historical energy consumption data and other relevant variables,
such as weather forecasts, time of day, and day of the week, AI models can forecast future energy consumption patterns with quite high accuracy. 
These predictions can help utilities to optimize their energy resources, reduce energy waste, and prevent blackouts or other disruptions to the power supply.

Furthermore, with the increasing adoption of renewable energy sources, such as solar and wind power, accurate energy consumption predictions become even more important.
Renewable energy sources are inherently variable and intermittent, which means that utilities must be able to forecast energy consumption patterns in order 
to balance the grid and avoid overloading or underutilizing renewable energy resources.

In this project, I will use three AI models to predict energy consumption 24 hours ahead: a feed-forward 3 layer network, a multichannel CNN, and a 2-layer LSTM.

# Methodology

The dataset used for this project will consist of historical energy consumption data for each of the 5 regions in Norway. 
The dataset will be preprocessed and standarized. The dataset will be split into training, validation and testing sets.

I will start by implementing a feed-forward 3 layer network model. The model will be trained on the training set. This model is considered baseline model.

Next, a 2-layer LSTM model will be implemented.  

Finally, a multichannel CNN model will be implemented. The model will have multiple input channels, each representing a different feature of the energy consumption data.
The model will be trained on the training set using backpropagation with Adam as the optimizer. 

# Prediction

All the models will be evaluated using MSE as the evaluation metric for the one step-ahead forecasting. 
All three models will forecast one step ahead and then the results will be used as a feature for the next step forecasting.  
This process is repeated until the end of the 24H forecasting window is reached.


