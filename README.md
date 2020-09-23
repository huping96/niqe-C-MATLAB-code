# niqe-C-MATLAB-code

Paper name: Making a "Completely Blind" Image Quality Analyzer
Author: Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik
Year: 2013

It contains the C++ code and MATLAB code of the niqe algorithm. The MATLAB code contains the training code and the test code, and the C++ code only contains the test code. Therefore, when running the C++ test code, you need to import the mu.txt and cov.txt obtained by MATLAB.

The included mu.txt and cov.txt files are obtained by MATLAB training my private data set. Because the algorithm needs to be in the same environment and the same lighting conditions for the training set and test image, it could get higher accuracy, so the two txt files are of little use to you. So, please train your own data set on MATLAB code in order to get correct results.

My highest accuracy rate is 97.15%.

C++ code: Open NIQE.sln to run

If you have any questions, or have any suggestions for improvement of my code, you can contact me at any time. 
My email: huping199609@163.com
