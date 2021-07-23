---
layout: archive
title: "Software"
permalink: /software/
author_profile: true
redirect_from:
  - /software
---

The software used in the paper  <a href="/files/RegularizedCGMforML_v1.pdf" target="_blank">Regularized Step Directions in Conjugate Gradient Minimization for Machine Learning</a> is available here for open source download.

We have reimplemented conjugate gradient method of Conmin in C and connected it to AMPL.  In our C implementation, we have omitted the BFGS method also implemented in the original Conmin distribution, and we will call our new code *Conmin-CG*. The cubic regularization scheme proposed in our paper as Algorithm 2 was implemented and tested by modifying this software, and is called *Conmin-CG with Cubic Regularization*.

- <a href="/files/Conmin-CG.zip" target="_blank">Click to download Conmin-CG </a>
- <a href="/files/Conmin-CG with Cubic Regularization.zip" target="_blank">Click to download Conmin-CG with Cubic Regularization</a>

In addition, we implemented the same code in MATLAB to compare to ADMM. 

- <a href="/files/Conmin-CG_MatlabVersion.zip" target="_blank">Click to download Conmin-CG with and without Cubic Regularization in MATLAB </a>

 
---