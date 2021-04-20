# Projects-of-Neural-Network-and-Deep-Learning-DATA130011.01
This is a report including all projects in my 2021 Spring Neural Network and Deep Learning course (DATA130011.01) in [School of Data Science](https://sds.fudan.edu.cn/)  of [Fudan University](https://www.fudan.edu.cn/) .
The project is mainly about the construction of neural networks.
## Project
   * [Lab1_Warmup](./Lab1)
   * [Project1_Handwritten digit classification](./Project1)
   * [Project2_CIFAR-10](./Project2)
   * [Project3_3D Object Classification](./Project3)
   * [FinalProject_Scene Text Recognition](./FinalProject_GOMOKU)
   
## Details
**Lab1_Warmup**
* This lab aims to help the students refresh the basics of python, particularly, NumPy
* You can see the detail of project [here](./Lab1/Lab1.pdf) and my report [here](./Lab1/Report.pdf)
    
**Project1_Handwritten digit classification**
* In this project, we use MATLAB to achieve the basic methods and settings in neural network
* You can see the detail of project [here](./Project1/project_1.pdf) and my report [here](./Project1/Report_钟诚_16307110259.pdf)

**Project2_CIFAR-10**
* Network needs to contain a lot of components. (e.g. 2D pooling layer, Drop out, Residual Connection) So it will improve the performance of the network on the CIFAR-10 dataset based on the ResNet-18 network.
* You can see the detail of project [here](./Project2/Reference/project_2.pdf) and my report [here](./Project2/Report.pdf)
    
**Project3_3D Object Classification**
* In this project, I will try to use neural network to classify 3D point clouds, and initially understand the use of 3D data and the application of deep learning in the 3D field.

* You can see the detail of project [here](./Project3/project_3.pdf) and my report [here](./Project3/Project3-Report.pdf)    

**FinalProject_Scene Text Recognition**
* I do this project with [Ruipu Luo](https://rupertluo.github.io/)
* In this project, I try to solve the task of scene text recognition. Scene Text Recognition (STR) refers to the recognition of text information in pictures of natural scenes. One related concept is Optical Character Recognition (OCR), which refers to analyzing and processing the scanned document image to identify the text information.
* In this final project, we used a pipeline model for OCR tasks, that is, first detect the position of the text box, and then do the recognition of text in the box.For the text
detection stage, we compared the YOLOv3 model and the EAST model, which are widely used in object and text recognition. Due to the high resolution of the picture in the training set, we replaced the backbone of the original EAST model with resnet18, and achieved a certain performance improvement. In the text recognition stage, we adopted two models that is CRNN and Seq2Seq with Attention, which are the most widely used. We finally used the model combination of YOLOv3+CRNN and obtained the best f-measure:(0.3055) on the validation set, which is split from the training set.
* You can see the detail of project [here](./Final-Project/final_project.pdf) and my report [here](./Final-Project/final_report_16307130247-16307110259.pdf)
