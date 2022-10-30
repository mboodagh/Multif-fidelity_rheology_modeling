# Multif-fidelity_rheology_modeling
# General description
In this work, the goal is to determine the rheological properties of suspension of fibers using multi-fidelity machine learning models, i.e., Neural networks and Gaussian process. The notebook file **"Multifidelity_rheology_fiber_suspensions"** provides a detailed description of how the high-fidelity rheological data can be combined with low-fidelity data in multi-fidelity surrogates to make rheological precitions of the suspension of fibers. 
Here, we will demonstrate how the apperent viscosity as a function of starin rate can be determined for the suspension of fibers with different concetrations. The machine learning anlysis in conducted in Python using PyTorch and PyMC. For further information, please visit our publication at https://aip.scitation.org/doi/full/10.1063/5.0087449.

# Motivation

Determining the rheological properties of fluids containing fibers is vital to many industrial applications such as chemical processing and material reinforcement, which involve the transportation and mixing of fiber suspensions. Despite being highly informative, computations in the realm of rheology are often time-consuming and expensive, particularly for fiber simulations with many degrees of freedom. Examining the role of an individual input parameter in the range of interest could require long hours of simulations, let alone the different combinations of the input parameters. Recently, there has been a growing interest in using machine learning methods to develop models that can learn the relationships between multiple input and output parameters and can give an accurate prediction for the unknown inputs, such as neural networks (NN). NNs, even though flexible, require a relatively large amount of data for training to produce accurate predictions. Given the expensive nature of the computational data, it is of paramount importance to come up with a procedure that can generate a sufficient amount of non-expensive data, i.e., the low fidelity data for the training of the machine learning algorithms. Combining the low fidelity data with the high-fidelity data—the data obtained from expensive computations or experiments—leads to an algorithm in the form of multi-fidelity modeling (MFM). Multi-fidelity modeling (MFM) is a promising and robust platform for extending the range of predictions when there is insufficient data

# Method
## Obtaining data
We use numerical simulations to generate the high-fidelity dataset. The Immersed Boundary Method is used to solved the coupled fluid dynamics-solid mehanics equations. For single-fidelity NN and GP surrogates, we focus only on the high-fidelity data as the source for training and testing the surrogates. As shown in below figure for the single-fidelity NN, we have four inputs and one output, which is the viscosity.

![DNN](https://user-images.githubusercontent.com/60017299/198901572-f232c2e1-1d34-4b64-b1f3-e0ce09619e7b.jpg)

As we mentioned, the high-fidelity data points that describe the behavior of a
three-dimensional (3D) system with large degrees of freedom are typically expensive. The numerical simulation of the suspension of fibers
in the current study is no exception as the 3D simulations with a sufficient level of mesh-independency are time-consuming. The cornerstone idea behind MFNNs is to include more data and possibly more
physics in the form of lower-fidelity data in training the network. The
challenging part is how to map the data points from lower fidelity
models to the ones from higher fidelity models to rectify the response
of the low-fidelity model. It is possible to have data with multiple levels
of fidelity, where all the lower levels must be sequentially tuned to
match the highest fidelity. In the present work, we focus only on two
levels of fidelity. One of the most widely used models to relate the lowfidelity predictions $y_LF$ to the high-fidelity predictions $y_HF$ is the linear
correlation as follows:

![image](https://user-images.githubusercontent.com/60017299/198902332-09186dcc-2938-4565-9187-c7b2a132eccf.png)


where $\delta_c(\mathbf{x})$ and $\rho_c(\mathbf{x})$ are the additive and multiplicative correlation surrogates, respectively, and $\rho_c$ is set to a constant value in most models \cite{fernandez2016review}. 
Linear correlations, even with non-constant parameters, cannot capture the correlation between the low and high-fidelity data in specific problems. As a result, to capture the non-linearities, a generalized autoregressive scheme has been introduced as follows:

![image](https://user-images.githubusercontent.com/60017299/198902456-8c82d4a4-1ce4-4ce1-8bb0-70f9a0b4b030.png)

where $F$ is an unknown function and is determined iteratively. 
Furthermore, the effect of $F$ and $\delta_c$ can be expressed as a combination of a linear $F_l$ and non-linear $F_{nl}$ correlations as follows:

![image](https://user-images.githubusercontent.com/60017299/198902429-22092646-ce31-4431-b016-cca7f9925d7f.png)


The schematic representation of multi-fidelity neuran networks (MFNN) is shown in the below figure.

![MFNN_page-0001](https://user-images.githubusercontent.com/60017299/198902801-ceed0978-ada4-4241-80fa-df6b8fbdb191.jpg)


In order to compare DNN and GP predictions for different training data sizes, we vary $M_t$ between 53 and 93 for both DNN and GP. The $R^2$ and MSE values vary significantly for different hyperparameters. Consequently, in order to make a comparison, we pick the optimized value, i.e., the highest $R^2$ values generally corresponding to minimum NRMSE values for both the GP and DNN. As shown in the below table, the $R^2$ values for the single-fidelity models varies in the range $0.71-0.94$. However, it doesn't exceed $0.8$ when the data size is smaller than $70$. To increase the accuracy beyond $0.9$, one would need to add more data points like in the case of $M_t=93$. However, further increase of the high-fidelity dataset is very expensive. Additionally, even if it is possible to add to the training dataset, such as the case $M_t=93$, as we will show, the range of predictions remains limited to the range of the training datasets. 
