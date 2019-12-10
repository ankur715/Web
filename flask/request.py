#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Request predictions
 
# Predictions are made by passing a POST JSON request to the created Flask web server 
# which is on port 5000 by default. In app.py this request is received and a prediction 
# is based on the already loaded prediction function of our model. 
# It returns the prediction in JSON format.

import requests
import json

url = 'http://0.0.0.0:5000/api/'

data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)

# Now, all you need to do is call the web server with the correct syntax of data points. 
# This corresponds with the format of the original dataset to get this JSON response of your predictions. 

# For example:
#    python request.py -> <Response[200]> “[1.]"

# For the data we sent we got a prediction of class 1 as output of our model. 
# Actually all you are doing is sending data in an array to an endpoint, which is transformed to JSON format. 
# The endpoint reads the JSON post and transforms it back to the original array.