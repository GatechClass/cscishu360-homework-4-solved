# cscishu360-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [CSCISHU360 Homework 4 Solved](https://mantutor.com/product/cscishu360-instructions-solved-3/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;114902&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCISHU360 Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
• Online submission: You must submit your solutions online on the course Gradescope site (you can find a link on the course Brightspace site). You need to submit (1) a PDF that contains the solutions to all questions to the Gradescope HW4 Paperwork assignment (including the questions asked in the programming problems), (2) x.py or x.ipynb files for the programming questions to the Gradescope HW4 Code Files assignment. We recommend that you type the solution (e.g., using LATEX or Word), but we will accept scanned/pictured solutions as well (clarity matters).

• Generative AI Policy: You are free to use any generative AI, but you are required to document the usage: which AI do you use, and what’s the query to the AI. You are responsible for checking the correctness.

Before you start: this homework only has programming problems. You should still have all questions answered in the write-up pdf. Also note that some sub-problems are still essentially math problems and you need to show detailed derivations.

Problems 1 and 2 are two parts of one problem, so we suggest you read the description of both problems in tandem.

This homework could be challenging and hope the following tips help:

• Understanding of the gradient boosting tree concept. In particular, we use every single tree to compute a “gradient step”, and the sum of the gradient steps gives us the final predictions.

• Understanding of the python notebook we provide. The code we provide aims to share implementations between the random forests and the GBDTs. Try to think about how different parts of the code could be re-utilized by the two models.

• Debugging your code: always try to debug a small case. For example, use very few data points and build a tree with a depth of 2 or 3. Then, you can look at all the decision rules and data point assignments and check if they are reasonable.

1 Programming Problem: Random Forests [40 points]

Random forests (RF) build an ensemble of trees independently. It uses bootstrap sampling (sampling with replacement, as discussed in class) to randomly generate B datasets from the original training dataset, each with the same size as the original one but might contain some duplicated (or missing) samples. Each sample has a multiplicity which is greater than or equal to zero. In your python implementation, you can use numpy.random.choice for the sampling procedure.

. (1)

The optimization problem for training each tree in RF is

, (2)

where ˆyi is the prediction produced by tree-b fb(·;θb) for data point xi, ℓ(·,·) is a loss function (detailed in Problem 2.5.3), and Ω(θb) is a regularizer applied to the parameters θb of model-b (that is, Ω(θb) measures the complexity of model-k). Most descriptions of ensemble learning in Problem 2.1 of the homework (below) can be also applied to RF, such as the definitions of fk(·;θk) and θk, except Eq. (3) and Eq. (4).

Different methods can be used to find the decision rule on each node during the optimization of a single tree. A core difference between random forests and GBDTs (which we will describe in Problem 2) is the tree growing methods. Specifically, in the case of GBDT, we use the standard greedy tree-splitting algorithm; in the case of random forests, we greedily learn each tree using a bootstrapped data sample and random feature selection as described in class. That is, the key difference is the data that is being used (always original data in the case of GBDT or bootstrap sample for each tree in the case of RFs), and in the case of RFs we choose a random subset of features each time we grow a node. The underlying algorithm, however, is very similar. Therefore, to facilitate code reuse between this and the next problem, and also to make more fair the comparison between RFs and GBDTs, we ask you to use the same code base between this and the next problem (detailed in Problem 2.4 below).

Each tree in the RF method is like the first tree in GBDT, as the RF method does not consider any previously produced trees when it grows a new tree (the trees are independent with RFs). With RFs, we simply start with ˆyi0. You need to notice this fact when re-using the code from GBDT, because Gj and Hj for tree-k in RF only depend on ˆyi0, not ˆ . Instructions 2-5 in Problem 2.5, however, can still be applied to RF tree building here.

In this problem, you will implement RFs for both regression and binary classification problems. Please read Problem 2.1, 2.4, and 2.5 below before you start.

1. [20 points] Implement RF for regression task, and test its performance on Boston house price dataset used in Homework 2. Report the training and test RMSE. How is the performance of RF compared to least square regression and ridge regression?

2. [20 points] Implement RF for binary classification task, and test its performance on Credit-g dataset. It is a dataset classifying people described by 20 attributes as good or bad credit risks. The full description of the attributes can be found at https://www.openml.org/d/31. Report the training and test accuracy. Try your implementation on breast cancer diagnostic dataset , and report the training and test accuracy.

2 Programming Problem: Gradient Boosting Decision Trees [60 points]

2.1 Problem of Ensemble Learning in GBDTs

Gradient Boosting Decision Trees (GBDT) is a class of methods that use an ensemble of K models (decision trees) . It produces predictions by adding together the outputs of the K models as follows:

. (3)

The resulting ˆy can be used for the predicted response for regression problems, or can correspond to the class logits (i.e., the inputs to a logistic or softmax function to generate class probabilities) when used for classification problems.

The optimization problem for training an ensemble of models is

, (4)

where θk is the parameters for the kth model, and where ˆyi is the prediction produced by GBDT for data point xi, ℓ(·,·) is a loss function (detailed definition of losses are given in Problem 2.5.3), and Ω(θk) is a regularizer applied to the parameters θk of model-k (that is, Ω(θk) measures the complexity of model-k).

, (5)

Therefore, to define a tree fk(·), we need to determine the structure of the tree T ≜ (Nk ∪ Lk,E) (E is the set of tree edges), the feature dimension pj and threshold τj associated with each non-leaf node j ∈ Nk,

Figure 1: An example of GBDT: Does the person like computer games?

and the weight associated with each leaf node j ∈ Lk. These comprise the learnable parameters θk of fk(·;θk), i.e.,

. (6)

To define an ensemble of multiple trees, we also need to know the number of trees K.

We cannot directly apply gradient descent to learn the above parameters of GBDT because: (1) some of the above variables are discrete and some could have an exponential number of choices, including for example, the number of trees, the structure of each tree, the feature dimension choice associated with each non-leaf node, the weights at the leaf nodes; and 2) the overall decision tree process is not differentiable meaning straightforward naive gradient descent seems inapplicable.

2.2 Overview of the GBDT algorithm

The basic idea of GBDT training is additive training, or boosting. As mentioned above, Boosting is a metaalgorithm that trains multiple models one after another, and in the end combines additively to produce a prediction. Boosting often aims to convert multiple weak learners (which might be only slightly better than a random guess) into a strong learner that can achieve error arbitrarily close to zero. Boosting has many forms and instantiations, including AdaBoost [6], random forests [5, 2], gradient boosting [3, 4], etc. Note that Bagging [1] is not boosting since there is no interdependence between the trainings of the different models, rather each model in Bagging is trained on a separate bootstrap data sample.

GBDT training shares ideas similar to coordinate descent in that only one part of the model is optimized at a time, while the other parts are fixed. In the coordinate descent algorithm (you implemented it for Lasso), each outer loop iteration requires one pass of all the feature dimensions. In each iteration of its inner loop, it starts from the first dimension, and optimizes only one dimension of the weight vector by fixing all the other dimensions and conditioning on all the other dimensions. Each tree in GBDT is analogous to a dimension in coordinate descent, but the optimization process is different. GBDT starts from the first tree, and optimizes only one tree per time by fixing all the previous trees and we condition on all the previously produced trees. One core difference GBDT training has from coordinate descent is that GBDT training does not have the outer loop associated with coordinate descent, i.e., it only does one pass over all the trees. If it was coordinate descent, we would optimize each coordinate only once in succession.

In particular, we start from one tree, and only optimize one tree at a time conditioned on all previously produced trees. This, therefore, is a greedy strategy as we spoke about in class. After the training of one tree finished, we add this new tree to our growing ensemble and then repeat the above process. The algorithm stops when we have added tmax trees to the ensemble.

In the optimization of each tree, we start from a root node, find the best decision rule (a feature dimension and a threshold) and split the node into two children, go to each of the children, and recursively find the best decision rule on each of the child nodes, and continue until some stopping criterion is fulfilled (as will be explained very shortly below).

In the following, we will first elaborate how to add trees one after the other, and then provide details regarding how to optimize a single tree based on a set of previous trees (which might be empty, and so this also explains how to start with the first tree).

2.3 Growing the forest: How to add a new tree?

Assume that there will be K trees in the end. Therefore, we will get a sequence of (partially) aggregated predictions , from the K trees as follows:

yˆi0 = 0,

,

According to Eq. (4), fixing all the previous k − 1 trees, the objective used to optimize tree-k is

, (7)

Let’s simplify the first term using Taylor’s expansion, ignoring higher-order terms:

. (8)

After applying Taylor’s expansion to ℓ(yi,yˆik−1 + fk(xi)), we have

, (9)

where gi and hi denote the first-order and second-order derivatives of ℓ(yi,yˆik−1) w.r.t. ˆyik−1, i.e.,

. (10)

The second term Ω(θk) in Eq. (7) is a regularization term aiming to penalize the degree of complexity of tree-k. It depends on the number of leaf nodes, and the L2 regularization of wk. With GBDTs, it is defined as:

. (11)

We plugin Eq. (9) and Eq. (11) into Eq. (7), and after ignoring constants, we get

Fk(θk) + const.

, (12)

2.4 Growing a tree: How to optimize a single tree?

Now we can start to optimize a single tree fk(·;θk). Look at the objective function in Eq. (12): it is a sum of |Lk| independent simple scalar quadratic functions of wjk for all the j ∈ Lk! How to minimize a quadratic function? This we know is easy, and similar to least square regression, the solution has a nice closed form. Hence, wjk minimizing Fk(θk) is

. (13)

We can plug the above optimal wjk into Eq. (12) and obtain an updated objective

(14)

However, there are still two groups of unknown parameters in θk, which are the tree structure T and the decision rules {(pj,τj)}j∈Nk. In the following, we will elaborate how to learn these parameters by additive training of a single tree.

We will start from the root node and determine the associated decision rule (pj,τj); this rule should minimize the updated objective in Eq. (14), where Lk contains the left and right child of the root node. Then, the same process of determining (pj,τj) will be recursively applied to the left and right nodes, until a stopping criteria (as described below) is fulfilled.

For each candidate decision rule (pj,τj), we can compute the improvement it brings to the objective Eq. (14). Before splitting node j to a left child j(L) and a right child j(R), the objective is

(15)

After splitting, the leaf nodes change to j(L) and j(R), and the objective becomes

Hence, the improvement (we usually call it the “gain”) is

(17)

(18)

(19)

Therefore, the best decision rule (pj,τj) on node j ∈ Nk is the one (out of the m × n possible rules) maximizing the gain, which corresponds to the decision rule that minimizes the updated objective in Eq. (14). That is, we wish to perform the following optimization:

(20)

We start from the root node, apply the above criterion to find the best decision rule, split the root into two child nodes, and recursively apply the above criterion to find the decision rules on the child nodes, the grandchildren, and so on. We stop splitting according to a stopping criterion is satisfied. In particular, we stop to split a node if either of the following events happens:

1. the tree has reached a maximal depth dmax;

2. the improvement achieved by the best decision rule for the node (Eq. (20)) goes negative (or is still positive but falls below a small positive threshold, in the following experiments, you can try this, but please report results based on the “goes negative” criterion);

2.5 Details of Practical Implementation

1. Learning rate η: You might notice that the tree growing in GBDT is a greedy process. In practice, to avoid overfitting on a single tree, and to give more chances to new trees, we will make the process less greedy. In particular, we usually assign a weight 0 ≤ η ≤ 1 to each newly added tree when aggregating its output with the outputs of previously added trees. Hence, the sequence at the beginning of Problem 2.3 becomes

yˆi0 = 0,

yˆi1 = ηf1(xi) = yˆi0 + ηf1(xi), yˆi2 = ηf1(xi) + ηf2(xi) = yˆi1 + ηf2(xi),

··· k

yˆik = η X fk′(xi) = yˆik−1 + ηfk(xi),

k′=1

···

K yˆiK = η Xfk(xi) = yˆiK−1 + ηfK(xi).

k=1

Note that this change must be applied in both training and during testing/inference. 0 ≤ η ≤ 1 is usually called the “learning rate” of GBDT, but it is not exactly the same as the variable we usually call the learning rate in gradient descent.

2. Initial prediction yˆi0: GBDT does not have bias term b like linear model y = wx+b. Fortunately, ˆyi0 plays a similar role as b. Hence, instead of starting from ˆyi0 = 0, we start from ˆ , i.e., the average of ground truth on the training set. For classification, it is also fine to use this initialization (the average of lots of 1s and 0s), but do not forget to transfer the data type of label from “int” to “float” when computing the average in this case.

3. Choices of loss function ℓ(·,·): ℓ(·,·) is a sample-wise loss. In the experiments, you should use least square loss ℓ(y,yˆ) = (y−yˆ)2 for regression problems. For binary classification problems, we use one-hot (0/1) encoding of labels y (y is either 0 or 1), and logistic regression (the GBDT output ˆy is the logit in this case, which is a real number and the input to logistic function producing class probability), i.e.,

ℓ(y,yˆ) = y log(1 + exp(−yˆ)) + (1 − y)log(1 + exp(ˆy)). (21)

The prediction of binary logistic regression, which is the class probabilities, is

Pr( , Pr(class = 0) = 1 − Pr(class = 1). (22)

To produce a one-hot (0/1) prediction, we apply a threshold of 0.5 to the probability, i.e.,

1, Pr(class = 1) &gt; 0.5

(23)

0, Pr(class = 1) ≤ 0.5

4. Hyper-parameters: There are six hyper-parameters in GBDT, i.e., λ and γ in regularization Ω(·), dmax and nmin in stopping criterion for optimizing single tree, maximal number of trees tmax in stopping criterion for growing forests, and learning rate η. We will not give you exact values for these hyper-parameters, since tuning them is an important skill in machine learning. Instead, we will give you ranges of them for you to tune. Note larger dmax and tmax require more computations. Their ranges are: λ ∈ [0,10], γ ∈ [0,1], dmax ∈ [2,10], nmin ∈ [1,50], tmax ∈ [5,50], η ∈ [0.1,1.0].

In RFs, we do not have the learning rate, but there is another hyper-parameter, which is the size m′ of the random subset of features, from which you need to find the best feature and the associated decision rule for a node. You can use m′ ∈ [0.2m,0.5m].

5. Stopping criteria: There are two types of stopping criteria needed to be used in GBDT/RFs training: 1) we stop to add new trees once we get tmax trees; and 2) we stop to grow a single tree once either of the three criteria given at the end of Problem 2.4 fulfills.

6. Acceleration: We encourage you to apply different acceleration methods after you make sure the code works correctly. You can use multiprocessing for acceleration, and it is effective. However, do not increase the number of threads to be too large. It will make it even slower. You can also try numba (a python compiler) with care.

2.6 Questions

1. [4 points] What is the computational complexity of optimizing a tree of depth d in terms of m and n?

2. [4 points] What operation requires the most expensive computation in GBDT training? Can you suggest a method to improve the efficiency (please do not suggest parallel or distributed computing here since we will discuss it in the next question)? Please give a short description of your method.

3. [8 points] Which parts of GBDT training can be computed in parallel? Briefly describe your solution, and use it in your implementation. (Hint: you might need to use “from multiprocessing import Pool” and “from functools import partial”. We also talked about multiprocessing in the recitation session.)

4. [20 points] Implement GBDT for the regression task, and test its performance on Boston house price dataset used in Homework 2. Report the training and test RMSE. How is the performance of GBDT compared to least square regression and ridge regression?

5. [20 points] Implement GBDT for the binary classification task, and test its performance on Creditg dataset. Report the training and test accuracy. Try your implementation on the breast cancer diagnostic dataset, and report the training and test accuracy.

6. [4 points] According to the results on the three experiments, how is the performance of random forests compared to GBDT? Can you give some explanations?

References

[1] Leo Breiman. Bagging predictors. Machine Learning, 24(2):123–140, 1996.

[2] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[3] Jerome H. Friedman. Stochastic gradient boosting. Computational Statistics and Data Analysis, 38:367– 378, 1999.

[4] Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29:1189–1232, 2000.

[5] Tin Kam Ho. Random decision forests. In Proceedings of 3rd International Conference on Document Analysis and Recognition, volume 1, pages 278–282, 1995.

[6] Robert E. Schapire. The strength of weak learnability. Machine Learning, 5(2):197–227, 1990.
