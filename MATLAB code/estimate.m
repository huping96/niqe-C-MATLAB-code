clear all;
clc;
folderpath ='D:\������ͼ�����ݼ�\image quality data\train_clear\';
% folderpath ='D:\������ͼ�����ݼ�\image quality data\phone_������2\phone_train_clear';
% folderpath ='D:\0901\roberts\��̬ģ��_����\';

[mu_prisparam cov_prisparam] = estimatemodelparam(folderpath,96,96,0,0,0.75);