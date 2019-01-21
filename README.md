# machinelearning
reproduce some interesting machine learning algos, especially about bayesian
the structure here should be managed by Sklearn structure, i.e. fit() and predict()
variance _inference is a powerful weapon recently found to do bayesian inference
the idea is transforming finding posterior dist to an optimization problem (choosing best approximation distribution from a family)

The key point is to use variational analysis tech in maths to solve the optimization problem, i.e. the derivative about the function
or distribution itself.

To solve optimization problem, with mean field assumption, we can use acent coordinate. 
Convergence is guranteed by convexity and seperable. See LASSSO regression for more info.


here we use this idea in simple bayesian linear regression model
