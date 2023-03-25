# loading the trained model 
import pickle



try:
   with open("Ts_model.sav", "rb") as f:
        model = pickle.load(f)
except IndexError as e:
    print(e)