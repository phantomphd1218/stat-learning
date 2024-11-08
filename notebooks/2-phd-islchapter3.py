# Importing the necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP.models import ModelSpec as MS, summarize, poly
from matplotlib.pyplot import subplots

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
# Creating a model with multiple predictors
X = MS(["lstat", "age"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

terms = Boston.columns.drop(["medv"])
terms
X = MS(terms).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

minus_age = Boston.columns.drop(["medv", "age"])
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
results1 = model1.fit()
summarize(results1)
# List comprehension to calculate the VIF
vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({"vif": vals}, index=X.columns[1:])
vif
# Interaction terms
X = MS(["lstat", "age", ("lstat", "age")]).fit_transform(Boston)
model2 = sm.OLS(y, X)
summarize(model2.fit())
# Quadratic terms
X = MS([poly("lstat", degree=2), "age"]).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)
anova_lm(results1, results3)

ax = subplots(figsize=(8, 8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel("Fitted value")
ax.set_ylabel("Residual")
ax.axhline(0, c="k", ls="--")

# Qualitative predictors
Carseats = load_data("Carseats")
Carseats.columns

allvars = list(Carseats.columns.drop("Sales"))
y = Carseats["Sales"]
final = allvars + [("Income", "Advertising"), ("Price", "Age")]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
summarize(model.fit())
