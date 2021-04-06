# Introduction
# http://peopleanalytics-regression-book.org/index.html

# install peopleanalyticsdata package
install.packages("peopleanalyticsdata")
library(peopleanalyticsdata)

# see a list of data sets
data(package = "peopleanalyticsdata")

# find out more about a specific data set ('managers' example)
help(managers)


# Overview of inferential modelling ----
# http://peopleanalytics-regression-book.org/inf-model.html#the-process-of-inferential-modeling

# Statistical Foundations ----
# http://peopleanalytics-regression-book.org/found-stats.html

# Linear regression for continuous outcomes ----
# http://peopleanalytics-regression-book.org/linear-reg-ols.html

# Walkthrough example {#walkthrough-ols}

  # You are working as an analyst for the Biology department of a large academic institution which offers a four year undergraduate degree program.  The academic leaders of the department are interested in understanding how student performance in the final year examination of the degree program relates to performance in the prior three years. 
  # To help with this, you have been provided with data for 975 individuals graduating in the past three years, and you have been asked to create a model to explain each individual's final examination score based on their examination scores for the first three years of their program.  The Year 1 examination scores are awarded on a scale of 0--100, Years 2 and 3 on a scale of 0--200, and the Final year is awarded on a scale of 0--300.

# if needed, download ugtests data
url <- "http://peopleanalytics-regression-book.org/data/ugtests.csv"
ugtests <- read.csv(url)

# Preview
head(ugtests)
str(ugtests)
summary(ugtests)

# display a pairplot of all four columns of data
library(GGally)
GGally::ggpairs(ugtests)

# Figure example with writing annotation and lines
# ``` {r sample-plot-residuals, fig.cap = "Residuals of $y=1.2x + 5$ for our ten observations", fig.align = "center", echo = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
#   ggplot2::ggplot(data = d, aes(x = Yr3, y = Final)) + 
#   ggplot2::geom_point() +
#   ggplot2::geom_function(fun = function(x) {1.2*x + 5}, colour = "red", linetype = "dashed") +
#   ggplot2::annotate("text", x = 137, y = 152, label = "y = 1.2x + 5", colour = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[1], y = 1.2*Yr3[1] + 5, xend = Yr3[1], yend = Final[1]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[2], y = 1.2*Yr3[2] + 5, xend = Yr3[2], yend = Final[2]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[3], y = 1.2*Yr3[3] + 5, xend = Yr3[3], yend = Final[3]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[4], y = 1.2*Yr3[4] + 5, xend = Yr3[4], yend = Final[4]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[5], y = 1.2*Yr3[5] + 5, xend = Yr3[5], yend = Final[5]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[6], y = 1.2*Yr3[6] + 5, xend = Yr3[6], yend = Final[6]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[7], y = 1.2*Yr3[7] + 5, xend = Yr3[7], yend = Final[7]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[8], y = 1.2*Yr3[8] + 5, xend = Yr3[8], yend = Final[8]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[9], y = 1.2*Yr3[9] + 5, xend = Yr3[9], yend = Final[9]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[10], y = 1.2*Yr3[10] + 5, xend = Yr3[10], yend = Final[10]), color = "red")
# ```

# Determine model fit
# calculate model
model <- lm(formula = Final ~ Yr3, data = d)
# view the names of the objects in the model
names(model) 
model$coefficients


# Showing what "EXPLAINES X% of the VARIANCE" really means
# Its the difference between error from the Null model and Error from the Best-Fit model.

# Therefore before we fit our model, we have an error of 1574.21, and after we fit it, we have an error of 398.35. So we have reduced the error of our model by 1175.86 or, expressed as a proportion, by 0.75. In other words, we can say that our model explains 0.75 (or 75%) of the variance of our outcome.
# This metric is known as the R2 of our model and is the primary metric used in measuring the fit of a linear regression model17.

# ```{r model-overlay, fig.cap = "Comparison of residuals of fitted model (red) against random variable (blue)", fig.align = "center", echo = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
# ggplot2::ggplot(data = d, aes(x = Yr3, y = Final)) +
#   ggplot2::geom_point() +
#   ggplot2::geom_function(fun = function(x) mean(d$Final), color = "blue", linetype = "dashed") +
#   ggplot2::annotate("text", x = 65, y = 137, label = "y = 133.70", colour = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[1], y = round(mean(Final), 2), xend = Yr3[1], yend = Final[1]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[2], y = round(mean(Final), 2), xend = Yr3[2], yend = Final[2]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[3], y = round(mean(Final), 2), xend = Yr3[3], yend = Final[3]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[4], y = round(mean(Final), 2), xend = Yr3[4], yend = Final[4]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[5], y = round(mean(Final), 2), xend = Yr3[5], yend = Final[5]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[6], y = round(mean(Final), 2), xend = Yr3[6], yend = Final[6]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[7], y = round(mean(Final), 2), xend = Yr3[7], yend = Final[7]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[8], y = round(mean(Final), 2), xend = Yr3[8], yend = Final[8]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[9], y = round(mean(Final), 2), xend = Yr3[9], yend = Final[9]), color = "blue") +
#   ggplot2::geom_segment(aes(x = Yr3[10], y = round(mean(Final), 2), xend = Yr3[10], yend = Final[10]), color = "blue") +
#   ggplot2::geom_function(fun = function(x) {model$coefficients[2]*x + model$coefficients[1]}, colour = "red", 
#                          linetype = "dashed") +
#   ggplot2::annotate("text", x = 100, y = 110, label = "y = 1.14x + 16.63", colour = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[1], y = model$fitted.values[1], xend = Yr3[1], yend = Final[1]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[2], y = model$fitted.values[2], xend = Yr3[2], yend = Final[2]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[3], y = model$fitted.values[3], xend = Yr3[3], yend = Final[3]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[4], y = model$fitted.values[4], xend = Yr3[4], yend = Final[4]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[5], y = model$fitted.values[5], xend = Yr3[5], yend = Final[5]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[6], y = model$fitted.values[6], xend = Yr3[6], yend = Final[6]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[7], y = model$fitted.values[7], xend = Yr3[7], yend = Final[7]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[8], y = model$fitted.values[8], xend = Yr3[8], yend = Final[8]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[9], y = model$fitted.values[9], xend = Yr3[9], yend = Final[9]), color = "red") +
#   ggplot2::geom_segment(aes(x = Yr3[10], y = model$fitted.values[10], xend = Yr3[10], yend = Final[10]), color = "red")
# ```

# predict from model
# get a prediction interval 
predict(newmodel, new_students, interval = "prediction")


# Transforming categorical inputs to dummy variables
# example
(vehicle_data <- data.frame(
  make = c("Ford", "Toyota", "Audi"), 
  manufacturing_cost = c(15000, 19000, 28000)
))
library(dummies)
(dummy_vehicle <- dummies::dummy("make", data = vehicle_data))
(vehicle_data_dummies <- cbind(
  manufacturing_cost = vehicle_data$manufacturing_cost,
  dummy_vehicle
))

# interpreting dummy variables
  # the model will assume a ‘reference value’ for the categorical variable—often this is the first value in alphabetical or numerical order. In this case, Audi would be the reference dummy variable. The model then calculates the effect on the outcome variable of a ‘switch’ from Audi to one of the other dummies18. If we were to try to use the data in our vehicle_data_dummies data set to explain the retail price of a vehicle, we would interpret coefficients like this:
  #   
  # Comparing two cars of the same make, we would expect each extra dollar spent on manufacturing to change the retail price by …
  # Comparing a Ford with an Audi of the same manufacturing cost, we would expect a difference in retail price of …
  # Comparing a Toyota with an Audi of the same manufacturing cost, we would expect a difference in retail price of …

# Testing model assumptions
# http://peopleanalytics-regression-book.org/linear-reg-ols.html#testing-your-model-assumptions

# Extending linear regression with interaction and exponential terms

# interpretation: e can conclude that the effect of an extra point in the examination in Year 3 will be different depending on how the student performed in Year 2.
# By examining the shape of this curved plane, we can observe that the model considers trajectories in the Year 2 and Year 3 examination scores. 



# Binomial Logistic Regression for Binary Outcomes ----

# The math
# Hence, binomial logistic regression is said to be in a class of generalized linear models or GLMs

# ```{r norm-log-curves, fig.align = "center", fig.cap = "The logistic function (blue dashed line) is very similar to a cumulative normal distribution (red solid line), but easier to interpret", echo = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
# library(ggplot2)
# ggplot2::ggplot() +
#   ggplot2::xlim(-5, 5) +
#   xlab("x") +
#   ylab("P (cumulative)") +
#   ggplot2::geom_function(fun = pnorm, color = "red") +
#   ggplot2::geom_function(fun = plogis, color = "blue", linetype = "dashed")
# ```

# Use Cases
#  Given a set of data about sales managers in an organization, including performance against targets, team size, tenure in the organization and other factors, what influence do these factors have on the likelihood of the individual receiving a high performance rating?
#  Given a set of demographic, income and location data, what influence does each have on the likelihood of an individual voting in an election?
#  Given a set of statistics about the in-game activity of soccer players, what relationship does each statistic have with the likelihood of a player scoring a goal?

# Example - likelihood of promotion 

# if needed, download salespeople data
url <- "http://peopleanalytics-regression-book.org/data/salespeople.csv"
salespeople <- read.csv(url)

# Inspect
# look at the first few rows of data
head(salespeople)
summary(salespeople)

library(GGally)

# remove NAs
salespeople <- salespeople[complete.cases(salespeople), ]

# convert performance to ordered factor and promoted to categorical
salespeople$performance <- ordered(salespeople$performance, 
                                   levels = 1:4)
salespeople$promoted <- as.factor(salespeople$promoted)

# generate pairplot
GGally::ggpairs(salespeople)

# ```{r prom-with-logistic, fig.cap = "Overlaying logistic functions with various gradients onto previous plot", fig.align="center", echo = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
# ggplot2::ggplot(data = salespeople, aes(x = sales, y = as.numeric(as.character(promoted)))) +
#   ggplot2::geom_point() +
#   ggplot2::geom_function(fun = function(x) plogis(x, mean(salespeople$sales), 0.75*sd(salespeople$sales)), color = "black", linetype = "solid") +
#   ggplot2::geom_function(fun = function(x) plogis(x, mean(salespeople$sales), 0.5*sd(salespeople$sales)), color = "blue", linetype = "dashed") +
#   ggplot2::geom_function(fun = function(x) plogis(x, mean(salespeople$sales), 0.25*sd(salespeople$sales)), color = "red", linetype = "dotdash") +
#   ylab("promoted")
# ```

# Modeling the log odds and interpreting the coefficients

# run a binomial model 
sales_model <- glm(formula = promoted ~ sales, 
                   data = salespeople, family = "binomial")

# view the coefficients
sales_model$coefficients

# We can interpret the coefficients as follows:
  # The (Intercept) coefficient is the value of the log odds with zero input value of  x — it is the log odds of promotion if you made no sales
  # The sales coefficient represents the increase in the log odds of promotion associated with each unit increase in sales. We can convert these coefficients from log odds to odds by applying the exponent function, to return to the identity we had previously

# convert log odds to base odds and odds ratio
exp(sales_model$coefficients)
# So we can see that the base odds of promotion with zero sales is very close to zero, which makes sense. Note that odds can only be precisely zero in a situation where it is impossible to be in the positive class (that is, nobody gets promoted). We can also see that each unit (that is, every $1000) of sales multiplies the base odds by approximately 1.04—in other words it increases the odds of promotion by 4%.

# Odds versus probability
  # If a certain event has a probability of 0.1, then this means that its odds are 1:9, or 0.111.
  # If a certain event has a probability of 0.1, then this means that its odds are 1:9, or 0.111. If the probability is 0.5, then the odds are 1, if the probability is 0.9, then the odds are 9, and if the probability is 0.99, the odds are 99. As we approach a probability of 1, the odds become exponentially large

  # * important note, a given increase in odds can have a different effect on probability depending on what the original probability was in the first place.
  # the closer the base probability is to zero, the similar the effect of the increase on both odds and on probability. However, the higher the probability of the event, the less impact the increase in odds has

# probability is # of favorable outcomes/ total outcomes --> "IN / OUT OF"
# odds is # of favorable : (to) : # of unfavorable outcomes --> "TO"

# ```{r odds-prob, fig.cap="Odds plotted against probability", fig.align="center", echo = FALSE, message = FALSE, warning = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
# library(latex2exp)
# ggplot2::ggplot() +
#   xlim(0, 1) +
#   ylim(0, 100) +
#   geom_function(fun = function (x) {x/(1-x)}, colour = "black") +
#   xlab("Probability") +
#   ylab("Odds")
# ```


# Multivariate Logistic regression
# Each coefficient (exponentiaed) represents the odds ration associated with a unit increase in Xi assuming no change in other inputs 

# with ordinal converted to dummy variable
library(dummies)

# convert performance to dummy
perf_dummies <- dummies::dummy("performance", data = salespeople)


# replace in salespeople dataframe
salespeople_dummies <- cbind(
  salespeople[c("promoted", "sales", "customer_rate")], 
  perf_dummies
)

# check it worked
head(salespeople_dummies)

# run binomial glm
full_model <- glm(formula = "promoted ~ .",
                  family = "binomial",
                  data = salespeople_dummies)

# get coefficient summary 
(coefs <- summary(full_model)$coefficients)
# model is using performance4 as the reference case. 

# create coefficient table with estimates, p-values and odds ratios
(full_coefs <- cbind(coefs[ ,c("Estimate", "Pr(>|z|)")], 
                     odds_ratio = exp(full_model$coefficients))) 


# Now we can interpret our model as follows:
  # All else being equal, sales have a significant positive effect on the likelihood of promotion, with each additional thousand dollars of sales increasing the odds of promotion by 4%
  # All else being equal, customer ratings have a significant negative effect on the likelihood of promotion, with one full rating higher associated with 67% lower odds of promotion
  # All else being equal, performance ratings have no significant effect on the likelihood of promotion

# add confidance intervals
exp(confint(full_model))
# interpretation
# all else being equal—every additional unit of sales increases the odds of promotion by between 3.0% and 5.7%, and every additional point in customer rating decreases the odds of promotion by between 22% and 89%.



# Understanding the fit and goodness-of-fit of a binomial logistic regression model
  # there is no clear unified point of view in the statistics community on a single appropriate measure for model fit in the case of logistic regression. 

# simplify model
simpler_model <- glm(formula = promoted ~ sales + customer_rate,
                     family = "binomial",
                     data = salespeople)

# pesudo R2 Models: 
  # * McFadden's $R^2$ works by comparing the likelihood function of the fitted model with that of a random model and using this to estimate the explained variance in the outcome
  # * Cox and Snell's $R^2$ works by applying a 'sum of squares' analogy to the likelihood functions to align more closely with the precise methodology for calculating $R^2$ in linear regression.  However, this usually means that the maximum value is less than 1, and in certain circumstances substantially less than 1, which can be problematic and unintuitive for an $R^2$
  # * Nagelkerke's $R^2$ resolves the issue with the upper bound for Cox and Snell by dividing Cox and Snell's $R^2$ by its upper bound.  This restores an intuitive scale with a maximum of 1, but is considered somewhat arbitrary with limited theoretical foundation
  # * Tjur's $R^2$ is a more recent and simpler concept.  It is defined as simply the absolute difference between the predicted probabilities of the positive observations and those of the negative observations

library(DescTools)
DescTools::PseudoR2(
  simpler_model, 
  which = c("McFadden", "CoxSnell", "Nagelkerke", "Tjur")
)

# Goodness-of-fit tests for logistic regression models compare the predictions to the observed outcome and test the null hypothesis that they are similar. This means that, unlike in linear regression, a low p-value indicates a poor fit. 
library(LogisticDx)

# get range of goodness-of-fit diagnostics
simpler_model_diagnostics <- LogisticDx::gof(simpler_model, 
                                             plotROC = TRUE)

# returns a list
names(simpler_model_diagnostics)

# in our case we are interested in goodness-of-fit statistics
simpler_model_diagnostics$gof


# Model parsimony
# Parsimony describes the concept of being careful with resources or with information. A model could be described as more parsimonious if it can achieve the same (or very close to the same) fit with a smaller number of inputs. The Akaike Information Criterion or AIC is a measure of model parsimony that is computed for log-likelihood models like logistic regression models, with a lower AIC indicating a more parsimonious model.

# Predict

# define new observations
(new_data <- data.frame(sales = c(420, 510, 710), 
                        customer_rate = c(3.4, 2.3, 4.2)))

# predict probability of promotion
predict(simpler_model, new_data, type = "response")

# examining outliers
  # residuals of logistic regression models are rarely examined, but they can be useful in identifying outliers or particularly influential observations and in assessing goodness-of-fit.
  # When residuals are examined, they need to be transformed in order to be analyzed appropriately. For example, the Pearson residual is a standardized form of residual from logistic regression which can be expected to have a normal distribution over large enough samples. 

d <- density(residuals(simpler_model, "pearson"))
plot(d, main= "")


# Multinomial Logistic Regression for Nominal Category Outcomes ----
  #  where a limited number of outcome categories (more than two) are being modeled and where those outcome categories have no order. 

# Examples of typical situations that might be modeled by multinomial logistic regression include:
#   
#   Modeling voting choice in elections with multiple candidates
#   Modeling choice of career options by students
#   Modeling choice of benefit options by employees

# if needed, download health_insurance data
url <- "http://peopleanalytics-regression-book.org/data/health_insurance.csv"
health_insurance <- read.csv(url)

# inspect 
head(health_insurance)
str(health_insurance)

library(GGally)

# convert product and gender to factors
health_insurance$product <- as.factor(health_insurance$product)
health_insurance$gender <- as.factor(health_insurance$gender)

GGally::ggpairs(health_insurance)

#  stratified binomial models
  # Treat options and independent binomial regressions

library(dummies)

# create dummies for product choice outcome
dummy_product <- dummies::dummy("product", data = health_insurance)

# combine to original set
health_insurance <- cbind(health_insurance, dummy_product)

# run a binomial model for the Product A dummy against 
# all input variables (let glm() handle dummy input variables)
A_model <- glm(
  formula = productA ~ age + gender + household + 
    position_level + absent, 
  data = health_insurance, 
  family = "binomial"
)


# summary
summary(A_model)

# simpler model
A_simple <- glm(
  formula = productA ~ age + household + gender + position_level, 
  data = health_insurance
)

# view odds ratio as a data frame
as.data.frame(exp(A_simple$coefficients))
# As an example, and as a reminder from our previous chapter, we interpret the odds ratio for age as follows: all else being equal, every additional year of age is associated with an approximately 2.2% decrease in the odds of choosing Product A over the other products.


# A multinomial logistic model
  # A multinomial logistic model will base itself from a defined reference category, and run a generalized linear model on the log-odds of membership of each of the other categories versus the reference category.
  # often known as the relative risk of one category compared to the reference category.

# define reference by ensuring it is the first level of the factor
health_insurance$product <- relevel(health_insurance$product, ref = "A")

# check that A is now our reference
levels(health_insurance$product)

library(nnet)

multi_model <- multinom(
  formula = product ~ age + gender + household + 
    position_level + absent, 
  data = health_insurance
)
summary(multi_model)

# interpreting

# calculate z-statistics of coefficients
z_stats <- summary(multi_model)$coefficients/
  summary(multi_model)$standard.errors

# convert to p-values
p_values <- (1 - pnorm(abs(z_stats)))*2

# display p-values in transposed data frame
data.frame(t(p_values))

# display odds ratios in transposed data frame
odds_ratios <- exp(summary(multi_model)$coefficients)
data.frame(t(odds_ratios))

# # Here are some examples of how these odds ratios can be interpreted in the multinomial context (used in combination with the p-values above):
# 
  # All else being equal, every additional year of age increases the relative odds of selecting Product B versus Product A by approximately 28%, and increases the relative odds of selecting Product C versus Product A by approximately 31%
  # All else being equal, being Male reduces the relative odds of selecting Product B relative to Product A by 91%.
  # All else being equal, each additional household member deceases the odds of selecting product B relative to Product A by 62%, and increases the odds of selecting Product C relative to Product A by 23%.

# model fit & variable reduction
# http://peopleanalytics-regression-book.org/multinomial-logistic-regression-for-nominal-category-outcomes.html#elim
  # outline process for selecting which variables to keep, must consider effect of multiple reference categories

DescTools::PseudoR2(simpler_multi_model, 
                    which = c("McFadden", "CoxSnell", "Nagelkerke"))



# Proportional Odds Logistic Regression for Ordered Category Outcomes ----

  # we can imagine a latent version of our outcome variable that takes a continuous form, and where the categories are formed at specific cutoff points on that continuous variable. 
  # For example, we can say that each unit increase in input variable x increases the odds of y being in a higher category by a certain ratio.


  # An important underlying assumption is that no input variable has a disproportionate effect on a specific level of the outcome variable. This is known as the proportional odds assumption.
  # means that the ‘slope’ of the logistic function is the same for all category cutoffs32.

# theory
# ```{r prop-odds-int, fig.cap=if (knitr::is_latex_output()) {"Proportional odds model illustration for a 5-point Likert survey scale outcome greater than 3 on a single input variable.  Each cutoff point $\\tau_k$ in the latent continuous outcome variable $y'$ gives rise to a binomial logistic function."} else {"Proportional odds model illustration for a 5-point Likert survey scale outcome greater than 3 on a single input variable.  Each cutoff point in the latent continuous outcome variable gives rise to a binomial logistic function."}, fig.align = "center", echo = FALSE, out.width = if (knitr::is_latex_output()) {"90%"}}
# library(ggplot2)
# p1 <- ggplot() +
#   xlim(0, 5) +
#   ylim(0, 5) +
#   geom_function(fun = function(x) x, color = "blue") +
#   geom_hline(yintercept = 3, color = "red", linetype = "dashed") +
#   xlab("x") +
#   ylab("y'") +
#   theme(axis.text.x=element_blank(),
#         axis.ticks.x=element_blank()) +
#   annotate("text", x = 0.5, y = 1.5, label = "y = 1,2,3", color = "red") +
#   annotate("text", x = 0.5, y = 4, label = "y = 4,5", color = "red") +
#   annotate("text", x = 3.7, y = 4.8, label = "y' = bx + c", color = "blue") +
#   annotate("text", x = 0.5, y = 3.2, label = expression(paste("y' = ", tau[3])), color = "red") +
#   theme(axis.text.y=element_blank(),
#         axis.ticks.y=element_blank())
# p2 <- ggplot() +
#   xlim(-5, 5) +
#   ylim(0, 1) +
#   geom_function(fun = plogis, color = "blue") +
#   xlab("x") +
#   ylab(expression(paste("P(y' > ", tau[3], ")"))) +
#   theme(axis.text.x=element_blank(),
#         axis.ticks.x=element_blank() )
# gridExtra::grid.arrange(p1, p2, nrow = 1)
# ```


# example
# if needed, download data
url <- "http://peopleanalytics-regression-book.org/data/soccer.csv"
soccer <- read.csv(url)

# convert discipline to ordered factor
soccer$discipline <- ordered(soccer$discipline, 
                             levels = c("None", "Yellow", "Red"))

# apply as.factor to four columns
cats <- c("position", "country", "result", "level")
soccer[ ,cats] <- lapply(soccer[ ,cats], as.factor)

# check again
str(soccer)

# run proportional odds model
library(MASS)
model <- polr(
  formula = discipline ~ n_yellow_25 + n_red_25 + position + 
    country + level + result, 
  data = soccer
)

# get summary
summary(model)

# get coefficients (it's in matrix form)
coefficients <- summary(model)$coefficients

# calculate p-values
p_value <- (1 - pnorm(abs(coefficients[ ,"t value"]), 0, 1))*2

# bind back to coefficients
(coefficients <- cbind(coefficients, p_value))

# calculate odds ratios
odds_ratio <- exp(coefficients[ ,"Value"])

# combine with coefficient and p_value
(coefficients <- cbind(
  coefficients[ ,c("Value", "p_value")],
  odds_ratio
))


# interpretation
  # Each additional yellow card received in the prior 25 games is associated with an approximately 38% higher odds of greater disciplinary action by the referee
  # Strikers have approximately 50% lower odds of greater disciplinary action from referees compared to Defenders



#  Calculating the likelihood of an observation being in a specific ordinal category
head(fitted(model))
# can be used in predictive analytics by classifying new observations into the ordinal category with the highest fitted probability. 


# diagnostics
# diagnostics of simpler model
DescTools::PseudoR2(
  model, 
  which = c("McFadden", "CoxSnell", "Nagelkerke", "AIC")
)

# lipsitz test 
generalhoslem::lipsitz.test(model)
# pulkstenis-robinson test 
# (requires the vector of categorical input variables as an argument)
generalhoslem::pulkrob.chisq(model, catvars = cats)

# Testing the proportional odds assumption
  #  proportional odds logistic regression model depends on the assumption that each input variable has a similar effect on the different levels of the ordinal outcome variable. It is very important to check

# Options to check the assumption
# Sighting the coefficients of stratified binomial models

# create binary variable for "Yellow" or "Red" versus "None"
soccer$yellow_plus <- ifelse(soccer$discipline == "None", 0, 1)

# create binary variable for "Red" versus "Yellow" or "None"
soccer$red <- ifelse(soccer$discipline == "Red", 1, 0)

# model for at least a yellow card
yellowplus_model <- glm(
  yellow_plus ~ n_yellow_25 + n_red_25 + position + 
    result + country + level, 
  data = soccer, 
  family = "binomial"
)

# model for a red card
red_model <- glm(
  red ~ n_yellow_25 + n_red_25 + position + 
    result + country + level,
  data = soccer, 
  family = "binomial"
)

# display the coefficients of both models and examine the difference between them.
(coefficient_comparison <- data.frame(
  yellowplus = summary(yellowplus_model)$coefficients[ , "Estimate"],
  red = summary(red_model)$coefficients[ ,"Estimate"],
  diff = summary(red_model)$coefficients[ ,"Estimate"] - 
    summary(yellowplus_model)$coefficients[ , "Estimate"]
))

# he differences appear relatively small. Large differences in coefficients would indicate that the proportional odds assumption is likely violated and alternative approaches to the problem should be considered.


# Option --The Brant-Wald test
  # generalized ordinal logistic regression model is approximated and compared to the calculated proportional odds model. A generalized ordinal logistic regression model is simply a relaxing of the proportional odds model to allow for different coefficients at each level of the ordinal outcome variable.
  # The Wald test is conducted on the comparison of the proportional odds and generalized models. A Wald test is a hypothesis test of the significance of the difference in model coefficients, producing a chi-square statistic. A low p-value in a Brant-Wald test is an indicator that the coefficient does not satisfy the proportional odds assumption.

library(brant)
brant::brant(model)

# A p-value of less than 0.05 on this test—particularly on the Omnibus plus at least one of the variables—should be interpreted as a failure of the proportional odds assumption.


# Mixed models for explicit hierarchy in data ----
  # When data are group-based and time-based. e.g. students were actually a mix of students on different degree programs, then we may wish to take this into account in how we model the problem—that is, we would want to assume that each student observation is only independent and identically distributed within each degree program.
  # Similarly time-based hierarchy can occer where we are modeling how answers to some questions might depend on answers to others, we may wish to consider the effect of the person on this model.

# Fixed and random effects
  # Fixed (not accounting for hierarcy / groupings): ignoring the group data. In this model, we assume that the coefficients all have a fixed effect on the input variables—that is, they act on every observation in the same way. This may be fine if there is trust that group membership is unlikely to have any impact on the relationship being modeled, or if we are comfortable making inferences about variables at the observation level only.
  # Random (interpreting at GROUP and Observation level): group membership may have an effect on the relationship being modeled, and if we are interested in interpreting our model at the group and observation leve --> e.g. changes the random intercept by group. Can also see group level effects 

# Example
# if needed, get data
url <- "http://peopleanalytics-regression-book.org/data/speed_dating.csv"
speed_dating <- read.csv(url)

# Fixed EFFECT only
# run standard binomial model
model <- glm(dec ~ agediff + samerace + attr + intel + prob, 
             data = speed_dating, 
             family = "binomial")

summary(model)


# Mixed model
# if we think each individual has a different ingoing base likelihood of making a positive decision on a speed date.
# then there is ‘a random effect for iid on the intercept of the model.’

# run binomial mixed effects model
library(lme4)

iid_intercept_model <- lme4:::glmer(
  dec ~ agediff + samerace + attr + intel + prob + (1 | iid),
  data = speed_dating,
  family = "binomial"
)

# view summary without correlation table of fixed effects
summary(iid_intercept_model, 
        correlation = FALSE)

# interpretation
# We see that there is considerable variance in the intercept from individual to individual
# Here, we can see that different individuals process the two inputs in their decision making in different ways, leading to different individual formulas which determine the likelihood of a positive decision

# CAN MAKE ATTRIBUTION / PREDICTIONS SPECIFIC TO THE INDIVIDUAL BY INCLUDING RANDOM EFFECTS

#  we can extend our random effects to the slope coefficients of our model. 
  # (1 + agediff | iid) to model a random effect of iid on the intercept and the agediff coefficient. 
# Similarly, if we wanted to consider two grouping variables—like iid and goal—on the intercept, 
  # we could add both (1 | iid) and (1 | goal) to our model formula.



# Structural Equation Models ----
# https://github.com/armacintosh/Regression_Notes_Examples.git







