#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>    
#include "niqe.h"

using namespace std;
using namespace cv;

#define  NUM  1     //读取image的个数

int main()
{
	int n = 1;
	char tmp[10];
	double total_time = 0;
	while (n <= NUM)
	{
		_itoa_s(n, tmp, 10);
		string ImgName = tmp;
		ImgName = "E:\\data\\GASE\\Image\\0b32c64301a14029906dd67f97748c1f_L51.bmp";
		Mat orig = imread(ImgName, 0);
		if (orig.data == 0)
		{
			printf("[error] 没有图片\n");
		}

		double score = 0;
		double Time = (double)cv::getTickCount();
		computescore(orig, &score);
		Time = ((double)cv::getTickCount() - Time) / 10000;
		total_time += Time;
		cout << "=> Result = " << score << endl;
		n++;
	}

	cout << "=> Execute time = " << total_time / NUM << " ms" << endl;

	waitKey(0);
	return 0;
}