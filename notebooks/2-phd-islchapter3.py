# Importing the necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize

# Loading the Boston dataset
Boston = load_data("Boston")
Boston.columns

# Creating the model specification
X = pd.DataFrame({"intercept": np.ones(Boston.shape[0]), "lstat": Boston["lstat"]})
X
y = Boston["medv"]
model = sm.OLS(y, X)
results = model.fit()
summarize(results)
# Creating the model specification using the ModelSpec class
design = MS(["lstat"])
design.fit(Boston)
X = design.transform(Boston)
X[:4]

design = MS(["lstat"])
X = design.fit_transform(Boston)
X[:4]

results.summary()
# Creating a prediction model
new_df = pd.DataFrame({"lstat": [5, 10, 15]})
new_X = design.transform(new_df)
new_X
# Making predictions
new_predictions = results.get_prediction(new_X)
new_predictions.predicted_mean
# Confidence and Prediction intervals
new_predictions.conf_int(obs=True, alpha=0.05)


def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to the axes ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)


ax = Boston.plot.scatter("lstat", "medv")
ax.axline(
    (ax.get_xlim()[0], results.params.iloc[0]),
    slope=results.params.iloc[1],
    color="r",
    linewidth=3,
    linestyle="--",
)
abline(ax, results.params[0], results.params[1], "r--", linewidth=3)
