# Forest fires
Predicting forest fires before they happen.

## Aims
This repo aims to predict forest fires in the northeast of Portugal before they happen. It is based on a dataset provided by UC Irvine, hosted at https://archive.ics.uci.edu/ml/datasets/forest+fires.

## Preprocessing
The data was inputted into an SQLite database, and categorical data was turned into numerical. One extra feature was generated from the data obtained: the current season. This was all placed in a database using the `make_db.py` and `add_to_db.py` scripts.

## Results

### Oct 17, 2017
After performing in-depth analysis of the data and features, the ones with the biggest impact on results were kept. A support vector machine was used as a predictive model, and after some analysis, it was found that the optimal parameters for it were a value of C=0.1 and a 2nd degree polynomial. It achieved a 60% accuracy in predicting fires, as opposed to the naive dummy classifier, which achieved a 52% accuracy (random guessing).

Of course, the improvement in accuracy is small, but it is still better than random guessing. A higher confidence threshold could be used to improve performance at the cost of detecting a smaller number of fires.
