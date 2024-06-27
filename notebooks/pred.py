import pickle
import warnings

warnings.filterwarnings("ignore")
#Load the trained model 
def load_model(model_path):
    '''
    Load a pre trained model from a pickle file ..

    parameters -> model path 

    Returns -> Loaded pre-trained model 
    '''
    with open (model_path , 'rb') as sp:
        return pickle.load(sp)

model = load_model('/Users/navi/Desktop/space-project/notebooks/starPredictor')
print(model)


# Making Prediction 
inputFeatures = [[2637.00000 , 0.00073 ,0.12700 , 17.22000]]

def make_pred(model , inputFeatures):

    '''
    making predictions  from  pre-trained model from a pickle file ..  

    parameters -> 
    model : path of pretrained model ..
    inputFeatures : List[List(float)]] 

    Returns -> tuple[str , List[float] , List[str]]  .  tuple of prediction Classes , probabilities ,classes to wich they belongs 
    '''

    predicClass = model.predict(inputFeatures)[0]
    probabilities = model.predict_proba(inputFeatures)
    classes = model.classes_
    return predicClass , probabilities ,classes

print()
print("------------------ \n" , make_pred(model , inputFeatures))
print("------------------")