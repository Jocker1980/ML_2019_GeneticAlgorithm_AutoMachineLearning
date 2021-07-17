# Genetic Algorithm for hyperparameter optimization
Exploring AutoML with genetic algorithm. TPOT for hyperparameter optimization.

TPOT (Tree-based Pipeline Optimisation Technique) is a library for autoML. I particularly like the fact that it could use genetic programming to optimize hyperparameters. When learning about grid search I was wondering how to includ more than the hyperparameters in the search space as the other aspect like the feature engineering will also influence the score. I think I found it in TPOT.

Moreover I am learning in a contimuous manner through other couses and internet and I learnt that Grid Search is now obsolete regarding the performance. Despite it is a good start to learn or for algorithms with a few hyperparameter, there are better technics such as Random Search or Bayesian optimization and recently a novel bandit-based approach.

ML or DL are fast changing/progressing fields and it is part of the game to follow new techniques to not be late. I know I still have everything to learn even several basics, to not say: "I neither know nor think that I know". But in // to fulfilling the basic gaps I want to stay in touch with the last public ML/DL progresses.

Here is a paper "Toward the automated analysis of complex diseases in genome-wide association studies using genetic programming" to better explain where it comes from and how it works.

ABSTRACT 
https://arxiv.org/abs/1702.01780

Full paper
https://arxiv.org/pdf/1702.01780.pdf


## TPOT Review
TPOT uses an evolutionary algorithm to automatically design and optimize a series of standard machine learning operations (i.e., a pipeline) that maximize the final classifier’s accuracy on a supervised classification dataset. It achieves this task using a combination of genetic programming (GP) [1] and
Pareto optimization (specifically, NSGA2 [7]), which optimizes over the trade-off between the number of operations in the pipeline and the accuracy achieved by the pipeline.

TPOT implements four main types of pipeline operators: (1) preprocessors, (2) decomposition, (3) feature selection, and finally (4) models. All the pipeline operators make use of existing implementations in the Python scikit-learn library [33]. Preprocessors consist of two scaling operators to scale the features and an operator that generates new features via polynomial combinations of numerical features.

Decomposition consists of a variant of the principal component analysis (RandomizedPCA). Feature selection implements various strategies that serve to filter down the features by
some criteria, such as the linear correlation between the feature and the outcome. Models consist of supervised machine learning models, such as tree-based methods, probabilistic and non-probabilistic models, and k-nearest neighbors.

TPOT combines all the operators described above and assembles machine learning pipelines from them. When a pipeline is evaluated, the entire dataset is passed through the pipeline operations in a sequential manner—scaling the data, performing feature selection, generating predictions from the features, etc.—until the final pipeline operation is reached. Once the dataset has fully traversed the pipeline, the final predictions are used to evaluate the overall classification accuracy of the pipeline. This accuracy score is used as part of the pipeline’s fitness criteria in the GP algorithm.

To automatically generate and optimize these machine learning pipelines, TPOT uses a GP algorithm as implemented in DEAP [9], which is a Python package for evolutionary algorithms. Oftentimes, GP builds trees of mathematical functions that seek to optimize toward a specified criteria. In TPOT, GP is used to optimize the number and order of pipeline operators as well as each operator’s parameters. TPOT follows a standard GP process for 100 generations: random initialization of the initial population (default population size of 100), evaluation of the population on a supervised classification dataset, selection of the most fit individuals on the Pareto front via NSGA2, and variation through uniform mutation (90% of all individuals per generation) and one-point crossover (5% of all individuals per generation). For more information on the TPOT optimization process, see [29].

### REFERENCES extract
[1] Wolfgang Banzhaf, Peter Nordin, Robert E Keller, and Frank D Francone. 1998. Genetic programming: an introduction. Morgan Kaufmann Publishers San Francisco

[7] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T Meyarivan. 2002. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6 (2002), 182–197.

[9] F´elix-Antoine Fortin, Fran¸cois-Michel De Rainville, Marc-André Gardner, Marc Parizeau, and Christian Gagn´e. 2012. DEAP: Evolutionary algorithms made easy. Journal of Machine Learning Research 13 (2012), 2171–2175.

[29] Randal S Olson, Nathan Bartley, Ryan J Urbanowicz, and Jason H Moore. 2016. Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. In Proceedings of the Genetic and Evolutionary Computation Conference 2016 (GECCO ’16). ACM, New York, NY, USA, 485–492. DOI:http://dx.doi.org/10.1145/2908812.2908918

[33] F Pedregosa, G Varoquaux, A Gramfort, V Michel, B Thirion, O Grisel, M Blondel, P Prettenhofer, R Weiss, V Dubourg, J Vanderplas, A Passos, D Cournapeau, M Brucher, M Perrot, and E Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12 (2011), 2825–2830.


## Particularities of TPOT

### Hyperparameter optimization

For automated machine learning in general, approaches have mainly focused on optimizing subsets of a machine learning pipeline [18], which is otherwise known as hyperparameter optimization. One readily accessible approach is grid search, which applies brute force search within a search space of all possible model parameters to find the best model configuration. Relatively recently, randomized search [2] and Bayesian optimization [35] techniques have entered into the foray and have offered more intelligently derived solutions— by adaptively choosing new configurations to train—to the hyperparameter optimization task. Much more recently, a novel bandit-based approach to hyperparameter optimization have outperformed state-of-the-art Bayesian optimization algorithms by 5x to more than an order of magnitude for various deep learning and kernel-based learning problems [21].


### REFERENCES extract
[2] James Bergstra and Yoshua Bengio. 2012. Random search for hyper-parameter optimization. Journal of Machine Learning Research 13, Feb (2012), 281–305.

[18] Frank Hutter, J¨org L¨ucke, and Lars Schmidt-Thieme. 2015. Beyond manual tuning of hyperparameters. KI-K¨unstliche Intelligenz 29, 4 (2015), 329–337.

[21] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. 2016. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. arXiv preprint arXiv:1603.06560 (2016).

[35] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. 2012. Practical bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems. 2951–2959.

## Black box, white box model
XGBoost can sometimes model higher-order interactions when the dataset is heavily filtered beforehand. However, the resulting XGBoost model is not nearly as interpretable as with TPOT-MDR. TPOT-MDR produces a model that we can inspect to study the pattern of feature interactions within the dataset, whereas XGBoost provides only a complex ensemble of decision trees.

This is an important consideration when building machine learning tools for bioinformatics: More often than not, bioinformaticians do not need a black box model that achieves high prediction accuracy on a real-world dataset. Instead, bioinformaticians seek to build a model that can be used as a microscope for understanding the underlying biology of the system they are modeling. In this regard, the models generated by TPOT-MDR can be invaluable for elucidating the higher-order interactions that are often present in complex biological systems.

In conclusion, TPOT-MDR is a promising step forward in using evolutionary algorithms to automate the design of machine learning workflows for bioinformaticians. We believe that evolutionary algorithms (EAs) are poised to excel in the automated machine learning domain, and specialized tools such as TPOT-MDR highlight the strengths of EAs by showing how easily EA solution representations can be adapted to a particular domain.
