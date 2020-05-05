# Practical Big Data Analytics
### Data Analytics, Science and Big Data

## Monday: Data Analytics 
### Intro to Data Science, Big Data & Analytics: Skills & Concepts
* Describe data, data insight and software-based data projects
	* prediction in-system, vs. data storytelling
	
* Describe what a data analyst does

* Describe what a data scientist does
* Describe what skills are needed for data science
	

* Describe key features of the scientific method
	* ie., that the analyst and scientist are disagreeable
	* esp., hypotheses and refutation
	
### Intro to Python for Data Science & Analytics
* Write a program which uses builtin types to create:
    - a regression data set -- lists, floats
    - a classification data set -- lists, sets

* Write a program which uses loops and comprehensions to:
    * apply a transformation to every element of a data set

* Write a program which uses functions to:
    - group reusable statistical code, eg., mean(), weighted_sum(), etc.
    
* Write a program which imports libraries to:
    - visualize your data sets

### Data Analytics in Python 
* Write a program which:
	- Uses NumPy for Numerical Computing
	- Creates a numpy array 
	- Obtains descriptive statistics (eg., .max())
	- Index using rows and columns
	- Index asking for *all* rows, with some columns
	- Use a mask index
		- eg., `x[ x < 3 ]`
			
* Write a program which:
	- Uses Pandas for ETL
	- Creates a DataFrames
	- Access columns by named index
	- Access rows 
		- `df.loc[ row  ,  col ]`
	- Obtains descriptive statistics
		- `.describe()`
		
	- Uses Sql-like operations on a dataframe
		- in either pandas or spark:
		- .groupby(), .filter(), ...
		
* Write a program which:
	- Uses seaborn to visualize data
	- Displays a scatter plot
	- Displays a line plot
	- Displays a regression plot

	- Displays a bar plot
	- Displays a distribution plot, eg., violinplot


## Tuesday: Machine Learning
### Intro to Machine Learning: Concepts with Python
* Describe the key phases in the history of AI:
    - 40s/50s logic programming
    - 80s expert systems
    - 00s machine learning
    - 10s deep neural networks

* Describe the different between explanatory and associative models
    * and hence the difference between scientific modelling and machine learning

* Describe the difference between
	- Supervised and Unsupervised Learning
	- Supervised: we have historical (x, y)
	- Unsupervised: only historical (x)
	
* Describe problems in:
	- Regression
	- Classification
	- Clustering

* Write program which:
	- Uses sklearn to learn a relationship between variables
	- Reports the coefficient and intercept of a linear regression line

* Write and run a example program in:
	- Regression (diabetes set)
	- Classification 
		* either (knn, own set)
		* or, logisticregression over example data


		
### Machine Learning Algorithms: Neural Networks
* Describe the purpose of a machine learning algorithm
	* Fitting a (statistical) model == finding a useful line

* Describe the advantages of a neural network 
	* Easy expansion of parameters 
		* eg., add a hidden layer = adding many more parameters
	* Adding parameters = more detailed line = more accurate on complex data
	
* Use https://playground.tensorflow.org/
	* Fit the network to example data with different numbers of:
		* Hidden Layers
		* Neurones/Layer
	* Describe the effect of changes to the architecture on the loss/fit
		

### Machine Learning: Modelling & Evaluation
* Describe the difference between in-sample, out-sample 
* Explain why the performance on the out-sample is the goal 
	* Describe the role of the test set in estimating this performance 
* Describe model selection
	* ie., splitting the training data to into train/validate
	*  .... evaluating model on validation data
	* selecting best model on validation data
		* by trying lots of splits 

* Write a program which performs a test-train split 
	* and evaluates model performance on the test set 	

* Run a program to cross validate hyperparameters for KNN
	* Find the best k number of neighbors 
		

## Wednesday:  Big Data Engineering
### Machine Learning: DevOps Pipelines and Workflow with Python
* Describe the key phases in a data science project
* Describe the key python libraries for each phase

* Describe key elements of the data flow pipeline in a simple machine learning project:
	* online vs. offline
	* injest, process, tune, train, predict...

* Run an example project 
	* Describe the input, output and role of each stage

		

### Introduction to Big Data: Concepts & Tools
* Describe the multiple notions of Big Data
	* 5Vs
	* "Non-Tradtional Methods"

* Describe main NoSQL data models:
	* Graph, KeyValue, Document, Columnar, ... 
	* Neo4j, Redis, MongoDB, Cassandra

* Describe purpose/advantages of a distributed file system
	* Performance - local data access 
	* Availability - data always available
	* Redundance - fail-well
	* Schemaless - takes any kind of data 
	
* Describe disadvantage of distributed file system
	* Nework IO => very poor compute performance
	* Querying is hard
		* Schemaless -- need to impose structure 
		
* The two roles/areas of interest to big data
	* Engineers
	* Analysts
	
* Describe the main trade-off theorum for Engineers
	* CAP Theorum
	* Consistency vs. Availability vs. Partition Tolerance
	
* Describe main "non-traditional" methods/data for analytics
	* Audio, Video, Images, Non-Transactional Data, Realtime Analytics
	* Graphs


### Functional Programming for Big Data
* Describe functional programming 
	* passing functions as arguments 
	* phrasing as map(f, x)
	
* Describe role of map/flatmap/reduce/filter

* Use map/flatMap/fold/reduce/exists/forall with a list of data to
	* transform, filter and aggregate information

## Thursday: Big Data Processing

### Graphs Analytics
* NetworkX
	* Nodes, Edges, Weights
	* Asked questions of importance about real-world data sets		
		* eg., infectious disease set, degree centrality (number of neighbors)
	* distribution of importance (ie., a few people had most of the "interesting measure")

### NoSQL Databases
* Neo4j
* MongoDB
* Install, run, query Neo4j
	

### Introduction to Hadoop & Spark
* HDFS
* MapReduce
* Spark
* Describe execute model of spark	
	* Driver Program -> {Resource Manager} -> Execute

* Describe key libraries for spark:
	* pyspark


## Friday: Spark

### Spark RDDs and DataFrames
* Use pyspark to create RDDs and:
	* map, flatMap, reduce, fold, 
	* min, max, count, collect
	
* Use pyspark to create DataFrames and:
	* describe, select, groupby, orderby, filter, ...
	* use SparkSQL to run a query against a dataframe	
		* use .explain() to see how that query is executed


### Appendix
- Spark Machine Learning
	* Run example spark machine learning project 
	* Describe correspondance (in method) to sklearn 
	
- Python and Data Serialization 
- Visual Science
 

## Follow-On Courses
* Python - QAPYTH3 -- 4d Python SE
* Practical Machine Learning (5D) -- QA

* Next Year: c. May
	* New Curriculum 
	
## Next Steps

* Books
	- Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
	- Building Machine Learning Powered Applications
	- Data Science and Big Data Analytics: Discovering, Analyzing, Visualizing and Presenting Data#
	- The Truthful Art 
	- Designing Data-Intensive Applications
	
		
* Videos
	* practical conference talks:
		- scipy conf
		- pycon
		- goto
		
	* MIT, Stanford, 
		* ML/DS/AS: Machine Learning, Calculus, Statistics, Linear Algebra, Probability
		* Big Data: Introduction to computational thinking, ... 
		
	* Videos by Andreas Mueller
	
	* exploratory data analysis, bokeh, visualization, 
	* machine learning workflow
	* neo4j, cassandra, mongo, spark
	
* Articles
	* InfoQ -- CAP 12 Years Later, Eric
	
* Notebooks
	* kaggle search (Apache 2)
	* github + notebook search (+ MIT/Apache 2) 
		* try to look for as open licence as possible
		
* Projects
	- do something interesting
	
## Contact

* michael.burgess@qa.com

