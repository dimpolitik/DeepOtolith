# DeepOtolith
Automating fish ageing from otolith images using deep learning (the role of multitask learning)


## Short description

Knowledge on the age of fish is vital for assessing the status of fish stocks and proposing management actions to ensure their sustainability. Prevalent methods of fish
ageing are based on the readings of otolith images by experts, a process that is often time-consuming and costly. This suggests the need for automatic and cost-effective
approaches. 

Our goal was to investigate the feasibility of using deep learning to provide an automatic estimation of fish age from otolith images through a convolutional neural
network designed for image analysis. On top of this network, we propose an enhanced - with multitask learning - network to better estimate fish age by introducing as an
auxiliary training task the prediction of fish length from otolith images. 

The proposed approach is applied on a collection of 5027 otolith images of red mullet (Mullus barbatus), considering fish age estimation as a multi-class classification task with six age groups (Age-0, Age-1, Age-2, Age-3, Age-4, Age-5+). Results showed that the network without multitask learning predicted fish age correctly by 64.4%, attaining high performance for younger age groups (Age-0 and Age-1, F1 score > 0.8) and moderate performance for older age groups (Age-2 to Age-5+, F1 score: 0.50-0.54). The network with multitask learning increased correctness in age prediction reaching 69.2% and proved efficient to leverage its predictive performance for older age groups (Age-2 to Age-5+, F1 score: 0.57-0.64). 

Our findings suggest that deep learning has the potential to support the automation of fish age reading, though further research is required to build an operational tool useful in routine fish aging protocols for age reading experts.

## References

* Politikos, D.V., Petasis G., Chatzispyrou A., Mytilineou C., Anastasopoulou A. 2021. Automating fish age estimation combining otolith images and deep learning: 
the role of multitask learning. Fisheries Research (under review).
