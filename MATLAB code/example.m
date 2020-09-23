clear all
close all
clc

load modelparameters_new.mat
blocksizerow    = 96;    %patch行数
blocksizecol    = 96;    %patch列数
blockrowoverlap = 0;     %无重叠 
blockcoloverlap = 0;     %无重叠

% file_path ='D:\低质量图像数据集\image quality data\选取阈值数据集\数据集1\';
% file = dir(strcat(file_path,'\*.bmp'));
% file_path2='D:\0901\niqe\离焦score\';
% file_path3='D:\0901\niqe\离焦_模糊\';
% file_path4='D:\0901\niqe\离焦_清晰\';
% num_file = length(file);
% quality = 1:num_file;
% % threshold = 45.76;
% select = 0;
% tic
% for L = 1:num_file;
%     img_name = strcat(file_path,'\',file(L).name);
%     image=imread(img_name);
% %     image=im2double(image); 
% %   figure;imshow(I,[]);
%     [M,N]=size(image);
%     quality(L) = computequality(image,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
%     
% %     FI = quality(L);
% %     ti1 = vision.TextInserter(num2str(FI), 'Location', [30 30],'FontSize', 30);
% %     img = step(ti1,image);
% %     image = im2uint8(img);
% %     imwrite(image,strcat(file_path2,file(L).name));
% %     
% %     if(quality(L) >= threshold)
% %         imwrite(image,strcat(file_path3,file(L).name));
% %     else
% %         imwrite(image,strcat(file_path4,file(L).name));
% %     end
% 
% end
% time = toc;
% 
% % quality = mapminmax(quality, 0, 100);
% x=1:num_file;
% plot(x,quality,'marker','o');
% xlabel('figure number');ylabel('quality');
% % axis([0 150 0 1]);
% hold on
% 
% [b,i] = sort(quality);

file_path ='D:\低质量图像数据集\image quality data\选取阈值数据集\子集1\';
file_path2='D:\0901\niqe1\动态模糊score\';
file_path3='D:\0901\niqe1\动态模糊_模糊\';
file_path4='D:\0901\niqe1\动态模糊_清晰\';
img_path_list1 = dir(strcat(file_path,'*.bmp'));
Len= length(img_path_list1);
quality = 1:Len;
threshold = 4.52;
select = 0;
tic
for k=1:Len
    image = imread([file_path,num2str(k),'.bmp'],'bmp');
    %disp(['k = ', num2str(k)]);
%     [row,col] = size(image);
%     image = imcrop(image,[row/2 col/2 128 128]);
%     figure;
%     imshow(image);
    quality(k) = computequality(image,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    
%     ti1 = vision.TextInserter(num2str(quality(k)), 'Location', [30 30],'FontSize', 30);
%     img = step(ti1,image);
%     image = im2uint8(img);
%     image_name_new = strcat(num2str(k),'.bmp');
%     imwrite(image,strcat(file_path2,image_name_new));
%     if(quality(k) >= threshold)
%         imwrite(image,strcat(file_path3,image_name_new));
%     else
%         imwrite(image,strcat(file_path4,image_name_new));
%     end
end
time = toc;
% quality = mapminmax(quality, 0, 100);
x=1:Len;
plot(x,quality,'marker','o');
xlabel('figure number');ylabel('quality');
% axis([0 150 0 1]);
hold on

[b,i] = sort(quality)
% 
% for iii=1:Len
%     if(quality(iii) > threshold)
%         select = select+1;
%     end
% end
% 
% error1 = 0;     
% for i = 1:20
%     if(quality(i) > threshold)
%         error1 = error1 + 1;
%     end
% end
% error2 = 0;
% for ii = 21:Len
%     if(quality(ii) < threshold)
%         error2 = error2 + 1;
%     end
% end
% accuracy = (Len-error1-error2)/Len;