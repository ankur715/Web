#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Run Flask
 
# Flask runs on a server. This can be in the environment of the client or a different server 
# depending on the client’s requirements. When running python app.py it first loads the 
# created pickle file. Once this is loaded you can start making predictions.


# In[ ]:


from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = 'models/final_prediction.pickle'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')

