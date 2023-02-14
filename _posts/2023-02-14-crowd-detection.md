---
layout: post
title: "Convolutional Neural Networks for crowd numerosity estimation"
subtitle: ""
background: '/img/posts/crowd-detection/crowd.jpg'
---

# Introduction

Hello there! 

This publication is a code layout of my [project](https://github.com/PashaIanko/Kaggle.CrowdCounting), available on github. We will use SOTA architectures, to estimate crowd numerosity in a busy shopping mall. This problem originates from a popular [Kaggle discussion](https://www.kaggle.com/datasets/fmena14/crowd-counting), devoted to crowd detection.


In this article, we will cover following topics:

- Alternative approaches to crowd counting, and why Deep Learning is a competitive approach

- Famous "Shopping Mall" dataset

- Experiments, performed with six SOTA architectures, to approach a crowd numerosity estimation

- Results of the best model on the test set, and concluding with the best transfer learning techniques, that helped to achieve 1.63 MAE performance

Enjoy!




# Alternative approaches to crowd counting

Alternative approaches are based on the idea of local visual features. The main concept, is that the features, extracted from the raw image, constitute an input to a regression model, predicting the crowdedness.

For example, the work of [Chen et al.](http://personal.ie.cuhk.edu.hk/~ccloy/files/bmvc_2012b.pdf) splits an original image into a grid of cells. From each cell, local visual features are extracted. In summary, each input image is transformed into a cell-ordered numerical vector, that encodes the extracted features. This vector represents an input for a multi-output regression model, predicting number of pedestrians in each cell independently. Such approach achieves 3.15 Mean Absolute Error (MAE) on a "Shopping Mall Dataset".





