# spikelearn
[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()

This project is part of my Master of Sciences graduate studies at Federal University of ABC.

# Objectives
## Timing
Time perception is discussed since Aristotle, yet its underlying neural basis remains elusive. By studying neural activity measured not only after but also _during_ learning, we hope to make original contributions to the field.

## Machine learning
If we had interest only in single variable correlations, like the famous _Time Cells_ or _Ramping Neurons_, we could get along with traditional analysis. To go further into _population coding_, we apply machine learning techniques such as Logistic Regression, Support Vector Machines, XGBoost, and I hope to soon use LSTMs.

## Open science
It is impossible to overstate the Python community contributions to my work, from the libraries ([sklearn](http://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/), [numpy](http://www.numpy.org/)), to the foruns ([stack overflow](https://stackoverflow.com/)). Even without strong principles, one should feel obliged to contribute back in the possible means.

I strongly believe in the principles of open science, and in the power of collaboration for the greater good, which's the reason why I am learning conventions and best practices for documentation and trying to make it all as organized as possible, even though organization was never my strength.

I intend to integrate my functions with *pip* sometime before ending my Master's (scheduled to end September/2019).

# Directory Organization
We mostly followed the organization proposed by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/). The specific usage for each of the main folders in this project follows:

## Data
Unfortunately our data is not available yet, and is being ignored by GitHub for the moment. We separate it by the level of preprocessing, aka raw, interim and processed. We also have a folder for external data, and one for results, the latter ignored mainly for storage purposes.

## spikelearn
All functions, classes and analysis procedures will be stored in the library folder _spikelearn_. It is divided into data, measures, models, tools and visuals. TODO explain each subfolder

## src
All code that will be run, the scrips, is located in the _src_ directory, where it is divided into analysis, data, EFO, models, and visualization. More information about the src folder on the README inside it.

# TODO
- Make 'sample' datasets in csv format for increasing fluency.
- Start logging
- Add pip integration to the library
