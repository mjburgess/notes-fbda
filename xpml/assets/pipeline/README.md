# Illustrative Machine Learning Project
### with ONNX

### Installation Requirements

* RUN THIS FROM "ANACONDA PROMPT"
	* cd to the project directory 
	* first run
	
```
		pip install onnx skl2onnx onnxruntime
```

### An Example Run
```
python online/A_injest.py 
python offline/A_prepare.py
python offline/B_model.py
python offline/C_evaluate.py 1581687160 
python online/C_deploy.py 1581687259 4.6 3.4 1.4 0.3
python analyst_monitor.py

```

#### --- ONLINE --- 
> python online/A_injest.py

* injest data
    * simulate iris flower data being received every second
    * with a given probability, simulate missing data (drop the target)
    * this is dumped into the data/ folder 

### --- OFFLINE ---
> python offline/A_prepare.py

* prepare raw injest for modelling
    * performs data preparation steps
        * here: removes NAs
        * may eg., PCA
    * splits into test/train


> python offline/B_model.py

* try various models
    * do not save: only trained on the training set
    * save: best parameters for later use



> offline/C_evaluate.py VALIDATION

* take best parameters and eval against test set
    * save best model trained on all data
    * save score of model against test set



### --- ONLINE --- 
> online/C_deploy.py MODEL X1 X2 X3 X4

* prepare features & predict target
    * prepare unseen incoming data for prediction 
        * (online/prepare.py) is run as an import to
    * predict using saved model
 
> analyst_monitor.py
> devops_monitor.py

* monitor running prediction systems
    * eg., quality metrics over time


## Folders

### Code
offline/
online/

### Data
online_db/
offline_db/

### Models & Config
models/
validations/
predictions/

## Exercise
### Part 1
* Obtain your own copy of the project and execute each stage as above
* Inspect the python files, output data set and output configuration files
* Convince yourself you understand the motivation behind each stage
    * Input, Process, and Ouput


### Part 2
* Create a similar project for a dense NN training the fashion MNIST data set

* Key Steps:
    * Injest the image data from keras as image files
    * PCA 
    * Train small dense neural network (eg., 2-layer)
    * Save predictions along with image files