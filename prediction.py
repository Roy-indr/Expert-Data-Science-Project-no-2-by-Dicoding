import joblib
import numpy as np

model = joblib.load('model/rdf_model.joblib')
encoder_label = joblib.load('model/encoder_label.joblib')

def prediction(data):
    result = model.predict(data)
    if result == 0:
        return 'Possibility of dropout is high'
    elif result == 1:
        return 'Possibly will not drop out but will still enrolled'
    else:
        return 'Possibly graduated soon'
    return final_result