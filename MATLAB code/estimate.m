clear all;
clc;
folderpath ='D:\低质量图像数据集\image quality data\train_clear\';
% folderpath ='D:\低质量图像数据集\image quality data\phone_误差分析2\phone_train_clear';
% folderpath ='D:\0901\roberts\动态模糊_清晰\';

[mu_prisparam cov_prisparam] = estimatemodelparam(folderpath,96,96,0,0,0.75);