### BTC Price Predictor

Given the coinbase and bitstamp datasets, write a script, forecast_btc.py, that creates,  
trains, and validates a keras model for the forecasting of BTC.  

Your model should use the past 24 hours of BTC data to predict the value of BTC at the  
close of the following hour (approximately how long the average transaction takes).  

Model 1 from the article is in the folder "1h + .5"   
Model 2 from the article is in the folder "1h"  
Model 3 from the article is in the folder "5m + .5"  

preprocess_data.py contains the preprocessing done with numpy.  
forecast_btc.py contains the tensorflow dataset, training and validation code.  
exploration.py is basically a scratch pad for examining the data.  
