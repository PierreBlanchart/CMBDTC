## Install instructions:

To install the R dependencies, run (if needed) from the R command line:
```{r}
install.packages(c("xgboost", "data.table", "Rcpp", "RcppArmadillo", "cpp11"))
```

The C++ dependencies are Boost and TBB. Check the paths for these two libraries in the file "./src/Makevars".

Once you have correct paths specified, install the package from the console:
```console
cd PACKAGE_SOURCE_FOLDER
R CMD INSTALL -l /INSTALL_PATH ./ --preclean
```

## Results

### Example 1

MNIST dataset: image from class 5 missclassified in class 6. We find the CF example of this image that is classified in class 5 by the model.

Visually, we see that the original missclasified image is transformed into something which looks more like a 5.

![Alt text](./results/Query_Class_6.png?raw=true "Query image from class 5 (missclassified in class 6)")

![Alt text](./results/CF_Class_5.png?raw=true "Corresponding CF example rightly classified in class 5")

### Beam search

Making the decision threshold decrease from 0.5 to 0.2 with a step of 0.01. We get closer and closer to the class center, with a CF example that resembles more and more a "5". To be read row-wise.

![Alt text](./results/BeamSearch_6To5.png?raw=true "Beam search (5 miss-classified as a 6)")

![Alt text](./results/BeamSearch_9To4.png?raw=true "Beam search (4 miss-classified as a 9)")

## Demos

Demos scripts are in the folder ./demos.
There are five demo scripts which illustrate the main functionnalities of the software:

1. ./demos/demo_surrogate_binary.R: this script illustrates the search for the closest CF example on a binary classification problem between two classes of the MNIST dataset. The search range is initialized using an approximate CF search using a derivable surrogate of the tree ensemble model / XGBoost model.
2. ./demos/demo_surrogate_multiclass.R: same as before but on a multiclass problem with all the 10 classes of MNIST.
3. ./demos/demo_CF_MNIST_beam_search.R: this script illustrates the search for the closest CF examples on a binary classification problem between two classes of the MNIST dataset. We make vary the decision threshold, which forces the algorithm to search for CF examples which are classified with more and more confidence inside the targeted class for the CF example. For each value of the decision threshold, we plot the corresponding CF example and mention the induced distortion (distance to the original query). We also assess visually the changes brought to the initial query in the successive CF examples to see if the ambiguities in the first query image are "resolved" to make it look more like an image belonging to the targeted class for the CF example.
4. ./demos/demo_restricted_CF.R: example of fixing the values of some input variables over which the user has no control. The CF example is computed on the remaining variables. Here, a dataset representing consumer credits approval/denial based on a set of 20 exogenous variables describing the credit applicant situation is used. We diagnosed the cases of credit denial: we determine which minimal changes in the input characteristics would allow the user to obtain the credit he asked for, knowing that there are characteristics on which he has no control (and which are forced to stay fixed by using a restricted CF query).
5. ./demos/demo_restricted_CF_regression.R: extension of the CF algorithm to a regression problem. We use a dataset describing the sale of individual residential property in Ames, Iowa from 2006 to 2010. The data set contains 2930 observations and a large number of explanatory variables (23 nominal, 23 ordinal, 14 discrete, and 20 continuous) involved in assessing home values. These variables describe the characteristics of the house, as well as that of the neighborhood. We apply our algorithm to a XGBoost regression model trained to predict the sale price of a house given a set of explanatory variables, and we try to answer the question from the seller point of view: what would be the changes/repairs to perform a minima in the house to increase the sale price to a target price ? We try to answer this question by considering only the variables that it is possible to change. For instance, it is not possible to change variables such as the total area, the date of construction, the quality of the neighborhood ... So, we keep these variables fixed, applying the CF approach to the remaining set of variables.

