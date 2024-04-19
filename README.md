<img src="docs/img/rmachines-hex-01.png" width="120" align="right" />

# Random Machines

A novel ensemble method employing Support Vector Machines (SVMs) as base learners. This powerful ensemble model offers versatility and robust performance across different datasets and compared with other consolidated methods as Random Forests (Maia M, et. al, 2021) <doi:10.6339/21-JDS1025>.

# Random Machines Classifier - Conceptual Overview

The Random Machines Classifier is an ensemble method that leverages multiple kernel functions within Support Vector Machine (SVM) models to perform classification tasks. Here's a breakdown of the key concepts and the algorithmic process described in the provided article:

## Conceptual Components:

### Kernel Functions:
Kernel functions transform the input data into a higher-dimensional space where a hyperplane can be used to separate classes. The Random Machines method evaluates multiple kernel functions, such as linear, polynomial, and radial basis functions (RBF), to determine which provides the best performance during training.

### Model Weights:
Each base SVM model within the ensemble is weighted based on its out-of-bag (OOB) prediction accuracy. The weight $w_i$ for model $i$ is calculated as:

```math
w_i = \frac{1}{(1 - \Omega_i)^2},
```

where $\Omega_i$ is the accuracy of the $i$-th model's predictions $g_i$ calculated on Out of Bag Sample $\(OOBG_i\)$ obtained from $i$-th bootstrap sample $\forall i = 1, . . . , B$ as test sample.

Final Classification:
The final classification decision is made by aggregating the predictions of all the base SVM models, each weighted by their respective weight $w_i$, according to:

$$G(x) = \text{sign} \left( \sum_{j=1}^B w_j g_j(x) \right),$$

Where $B$ is the number of bootstrap samples, $g_j(x)$ is the prediction from $j$-th model, and $G(x)$ the aggregated prediction.

## Bibliography

To learn more about the methods and how they are work please check.

**Ara, Anderson, et al. "Random Machines Regression Approach: an ensemble support vector regression model with free kernel choice." arXiv preprint arXiv:2003.12643 (2020).** [ArXiV Link](https://arxiv.org/abs/2003.12643)

**Ara, Anderson, et al. "Random Machines: A bagged-weighted support vector model with free kernel choice." arXiv preprint arXiv:1911.09411 (2019).** [ArXiV Link](https://arxiv.org/abs/1911.09411)
