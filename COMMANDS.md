
### make sure to add this function insdide the api.py file ##

```
@app.post('/predict')
def prediction(temperature, luminosity, radius, abs_mag):
    input_features = [[temperature, luminosity, radius, abs_mag]]
    pred_class, probs, classes = make_pred(model_path, input_features)
    
  
    # Convert numpy arrays to lists
    return {
         "Predicted_class": pred_class,
    }

```





Commands I used to do create all these stuff with git :->

S1 -> Go to GitHub create one empty project with project name. 

This is my home directory 
(base) navi@MacAir % 

(base) navi@MacAir % cd Desktop

(base) navi@MacAir Desktop % mkdir space-project

(base) navi@MacAir Desktop % cd space-project 

(base) navi@MacAir space-project % git init 

If main branch does not exists :->
(base) navi@MacAir space-project % git checkout -b main  

(base) navi@MacAir space-project % git checkout main 

Check in which branch we are present :->
(base) navi@MacAir space-project % git branch 

The result must be :->

* main

(base) navi@MacAir space-project % ls
commandsToPush.txt	notebooks

(base) navi@MacAir Desktop % cd space-project


NOW WRITE CODE IN THE NOTEBOOK :->

Main commands on tips are :->

(base) navi@MacAir space-project % git remote add origin https://github.com/Navi1313/Blockchain_Assignments/tree/main/eth-Proff-Assessment-3.git

(base) navi@MacAir space-project % git branch -M main   

(base) navi@MacAir space-project % git add .

(base) navi@MacAir space-project % git status 

(base) navi@MacAir space-project % git commit -m  " any msg "

(base) navi@MacAir space-project % git push 

Now see your git hub repo to see update changes 

