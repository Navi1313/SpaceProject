import pickle
import warnings


warnings.filterwarnings("ignore")

# Load the trained model 
def load_model(model_path):
    '''
    Load a pre trained model from a pickle file ..

    parameters -> model path 

    Returns -> Loaded pre-trained model 
    '''
    with open(model_path, 'rb') as sp:
        return pickle.load(sp)
    
model_path = load_model('StarPredictor1')
inputfeatures = [[2234 , 0.00072  , 0.231 , 17.4]]

def make_pred(model , inputFeature):

    '''
    making predictions  from  pre-trained model from a pickle file ..  

    parameters -> 
    model : path of pretrained model ..
    inputFeatures : List[List(float)]] 

    Returns -> tuple[str , List[float] , List[str]]  .  tuple of prediction Classes , probabilities ,classes to wich they belongs 
    '''

    pred_class = model.predict(inputFeature)
    probabilities = model.predict_proba(inputFeature)
    classes = model.classes_
    return pred_class, probabilities, classes


print(make_pred(model_path , inputfeatures))