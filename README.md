<div align="center">
  <h1 align="center"> ⚡️ Stars Classes Predictor 🐍  <h1> 
</div>

## 📋 <a name="table">Table of Contents</a>

1. 🤖 [Introduction](#introduction)
2. ⚙️ [Tech Stack](#tech-stack)
3. 🤸 [Quick Start](#quick-start)
4. 🔗 [Links](#links)
5. 🚀 [Output](#Output)


## <a name="introduction">🤖 Introduction</a>
This will predict the ['class of star', 'Color of star' ] by taking an input of ['Tempreature' , 'Luminosity' ,'Radius', 'Absolute Magnitude'] .
This projet is based on Machine Learning gives Multiple output from trained model using Multioutput Classifiers in Logistic Regression. 


## <a name="tech-stack">⚙️ Tech Stack</a>

- Python
- Machine Learning (linear + Logestic Regrerssion)
- Fast Api 
- Uvicorn server
- Scikit-learn
- Jupyter (Anaconda)
- virutal environment using conda python 


## <a name="quick-start">🤸 Quick Start</a>

Follow these steps to set up the project locally on your machine.

**Prerequisites**

Make sure you have the following installed on your machine:

###  1) For Model Building and Training ###

- [Anaconda](https://www.anaconda.com/download)
- [python](https://www.python.org/downloads/)
- [jupyter Notebook] 
- [test environment]
  
### 2) For Model Deployment 

- [VSCode]
- [Git](https://git-scm.com/)
  
-----------------------------------------------------------------

**Model  Building setup**

0. Must be created an test environment.
   
```bash
 conda create --name myenv python==lastestVersion
```

```bash
 conda activate myenv
```

...................

1. import numpy and pandas for data manipulation and cleaning.
2. import necessary scikit-learn libraries
3. Create the Model
4. train the model
5. test,train and split using inbuild methods
6. check the accuracy using j-score and h-loss.
7. then dump the model; 
8. Now Predict the class of star and color from this model .


**Depoloyment Setup for Project**

1. import fast api and uvicorn in VScode after creating new Project inside same directory .
```bash
!pip install fastapi 
```
```bash
!pip install uvicorn 
```
2. Crate an application.py file write an code to deploy on web and add the model pikle file.
3. follow Repo for more.


**Running the Project**

```bash
uvicorn api:app --reload 
```
Output :-> 

"Uvicorn running on http://127.0.0.1:8000"

-----------------------------------------------------------------------------



## <a name="links">🔗 Links</a>

### Read the documentation for more details
1. <a href="https://portal.thirdweb.com/"> thirdweb.com </a>
2. <a href="https://portal.thirdweb.com/"> fastapi.com </a>
3. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html"> scikit-learn Mutioutput Classifier </a>
4. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeling</a>


**Developer**


<a href="www.linkedin.com/in/navjot-singh-407025256"> Navi Sabharwal </a>


## <a name="output">⚡️ Output</a>

<img width="1440" alt="Screenshot 2024-07-06 at 6 24 21 PM" src="https://github.com/Navi1313/SpaceProject/assets/121182901/e7e5a26f-2f80-4ad6-9229-fa7c2e296a7c">
