#include "niqe.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp> 
//#include "cvaux.h" //��������ͷ�ļ� 
#include<fstream>

using namespace std;
#define  blocksizerow  96    // patch����
#define  blocksizecol  96    // patch����

// function to compute best fit parameters from AGGDfit
// ��Ϸǹ���ֲ�
Mat AGGDfit(Mat structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
{
	// create a copy of an image using BwImage constructor (brisque.h - more info)
	BwImage ImArr(structdis);

	long int poscount = 0, negcount = 0;
	double possqsum = 0, negsqsum = 0, abssum = 0;
	for (int i = 0; i < structdis.rows; i++)
	{
		for (int j = 0; j < structdis.cols; j++)
		{
			// BwImage provides [][] access
			double pt = ImArr[i][j];
			if (pt > 0)
			{
				poscount++;
				possqsum += pt * pt;
				abssum += pt;
			}
			else if (pt < 0)
			{
				negcount++;
				negsqsum += pt * pt;
				abssum -= pt;
			}
		}
	}

	lsigma_best = cv::pow(negsqsum / negcount, 0.5);
	rsigma_best = cv::pow(possqsum / poscount, 0.5);

	double gammahat = lsigma_best / rsigma_best;
	long int totalcount = (structdis.cols) * (structdis.rows);
	double rhat = cv::pow(abssum / totalcount, static_cast<double>(2)) / ((negsqsum + possqsum) / totalcount);
	double rhatnorm = rhat * (cv::pow(gammahat, 3) + 1) * (gammahat + 1) / pow(pow(gammahat, 2) + 1, 2);

	double prevgamma = 0;
	double prevdiff = 1e10;
	float sampling = 0.001f;

	// possible to coarsen sampling to quicken the code, with some loss of accuracy
	for (float gam = 0.2; gam < 10; gam += sampling)
	{
		double r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam));
		double diff = abs(r_gam - rhatnorm);
		if (diff > prevdiff) break;
		prevdiff = diff;
		prevgamma = gam;
	}
	gamma_best = prevgamma;

	return structdis.clone();
}


// function to compute brisque features
// feature dimensions : 1 x 36
// ��ȡ����
void ComputeniqeFeature(Mat& orig_bw, vector<double>& featurevector)
{
	if (orig_bw.empty())
	{
		cout << "input image err!" << endl;
		return;
	}

	// number of times to scale the image
	int scalenum = 2;
	for (int itr_scale = 1; itr_scale <= scalenum; itr_scale++)
	{
		// �ı�ͼ���С
		Size dst_size(orig_bw.cols / pow((double)2, itr_scale - 1), orig_bw.rows / pow((double)2, itr_scale - 1));
		Mat imdist_scaled;
		resize(orig_bw, imdist_scaled, dst_size, 0, 0, INTER_CUBIC);

		// calculating MSCN coefficients
		// ����MSCNϵ��

		// compute mu (local mean)
		// ����ֲ���ֵ
		Mat mu(imdist_scaled.size(), CV_64FC1, 1);          // ������ԭͼ��С��ͬ�Ŀ�ͼ�� mu
		GaussianBlur(imdist_scaled, mu, Size(7, 7), 1.167); // mu = imfilter(im, window, 'replicate');  ԭͼ����˹�˲���� mu
		Mat mu_sq;
		pow(mu, double(2), mu_sq);                          // mu_sq = mu.*mu;

														    // compute sigma (local sigma)
														    // ����ֲ�����
		Mat sigma1(imdist_scaled.size(), CV_64FC1, 1);      // ԭͼ���
		Mat sigma2(imdist_scaled.size(), CV_64FC1, 1);      // ԭͼ������˲�
		Mat sigma3(imdist_scaled.size(), CV_64FC1, 1);      // ԭͼ������˲� - ��ֵ��ƽ��
		Mat sigma(imdist_scaled.size(), CV_64FC1, 1);       // ���� sigma
		multiply(imdist_scaled, imdist_scaled, sigma1);     // im.*im
		GaussianBlur(sigma1, sigma2, Size(7, 7), 1.167);    // imfilter(im.*im,window,'replicate')
		subtract(sigma2, mu_sq, sigma3);                    // imfilter(im.*im,window,'replicate') - mu_sq
		pow(sigma3, double(0.5), sigma);                    // sigma = sqrt(abs(imfilter(im.*im,window,'replicate') - mu_sq));
															// to avoid DivideByZero Error
															// ����ֲ�����Ϊ0����Ϊ�������MSCNϵ��Ҫ���Ծֲ�����
															//add(sigma, Scalar(1.0 / 255), sigma);

															// structdis is MSCN image
															// ����MSCN
		Mat structdis1(imdist_scaled.size(), CV_64FC1, 1);  // ԭͼ - ��ֵ
		Mat structdis(imdist_scaled.size(), CV_64FC1, 1);   // ���� MSCN
		subtract(imdist_scaled, mu, structdis1);            // (im-mu)
		divide(structdis1, sigma + 1, structdis);           // structdis = (im-mu)./(sigma+1);

															// Compute AGGD fit to MSCN image
															// lsgima_best �󷽲rsigma_best�ҷ��gamma��ֵ
		double lsigma_best, rsigma_best, gamma_best;

		// �ǶԳƹ����˹�ֲ����
		structdis = AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);

		// �ȷŽ�ȥ�������� 
		// ��״����
		featurevector.push_back(gamma_best);                // GGD:gamma
		// �������
		featurevector.push_back((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2);        //GGD:sigma^2

		// Compute paired product images
		// indices for orientations (H, V, D1, D2)
		// ���������ԳƲ���
		int shifts[4][2] = { { 0,1 },{ 1,0 },{ 1,1 },{ 1,-1 } };   // �иĶ�

		for (int itr_shift = 1; itr_shift <= 4; itr_shift++)
		{
			// select the shifting index from the 2D array
			// ѡ����
			int* reqshift = shifts[itr_shift - 1];

			// declare shifted_structdis as pairwise image
			// ������ԭͼ�����ͼ���С��ͬ�Ŀ�ͼ��
			Mat shifted_structdis(imdist_scaled.size(), CV_64FC1, 1);

			// create copies of the images using BwImage constructor
			// ����BwImage�õ�ԭͼMSCN�������˻�ͼ��MSCN����
			// utility constructor for better subscript access (for pixels)
			BwImage OrigArr(structdis);
			BwImage ShiftArr(shifted_structdis);

			// create pair-wise product for the given orientation (reqshift)
			// ������������ĳɶԳ˻�
			for (int i = 0; i < structdis.rows; i++)
			{
				for (int j = 0; j < structdis.cols; j++)
				{
					if (i + reqshift[0] >= 0 && i + reqshift[0] < structdis.rows && j + reqshift[1] >= 0 && j + reqshift[1] < structdis.cols)
					{
						ShiftArr[i][j] = OrigArr[i + reqshift[0]][j + reqshift[1]];
					}
					else
					{
						ShiftArr[i][j] = 0;
					}
				}
			}

			// Mat structdis_pairwise;
			shifted_structdis = ShiftArr.equate(shifted_structdis);

			// calculate the products of the pairs
			// �õ��÷���˻����
			multiply(structdis, shifted_structdis, shifted_structdis);

			// fit the pairwise product to AGGD
			shifted_structdis = AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);

			double constant = sqrt(tgamma(1 / gamma_best)) / sqrt(tgamma(3 / gamma_best));
			double meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant;//4th params

			// push the calculated parameters from AGGD fit to pair-wise products
			// ���AGGD�ĸ�����
			featurevector.push_back(gamma_best);
			featurevector.push_back(meanparam);
			featurevector.push_back(pow(lsigma_best, 2));
			featurevector.push_back(pow(rsigma_best, 2));
		}
	}
}


// �����������۵÷֣�����Խ��ͼ������Խ��
double computescore(Mat orig, double* score)
{
	// ��һ��
	Mat orig_bw;
	orig.convertTo(orig_bw, CV_64FC1, 1 / 255.0);

	double block_rownum = floor(orig_bw.rows / blocksizerow); // 5
	double block_colnum = floor(orig_bw.cols / blocksizecol); // 6

	vector<vector<double>> all_niqeFeatures;    //���ı����Ƕ�����������Ǿ��󣿣�
	// all_niqeFeatures.resize(30, vector<double>(36));

	for (int j = 0; j < blocksizerow * block_rownum; j += blocksizerow)
	{
		for (int i = 0; i <= blocksizecol * block_colnum; i += blocksizecol)
		{
			vector<double> brisqueFeatures;
			if (i + blocksizecol < orig_bw.cols - 1)
			{
				rectangle(orig, Rect(i, j, blocksizecol, blocksizerow), Scalar(255), 1);
				Mat imageROI = orig_bw(Rect(i, j, blocksizecol, blocksizerow));
				ComputeniqeFeature(imageROI, brisqueFeatures);
				all_niqeFeatures.push_back(brisqueFeatures);
			}
		}
	}

	vector<double> meanv;
	int j = 0;
	for (int j = 0; j < 36; j++)
	{
		double total = 0;
		for (int i = 0; i < all_niqeFeatures.size(); i++)
		{
			total += all_niqeFeatures[i][j];
		}
		meanv.push_back(total/ all_niqeFeatures.size());
	}

	//for (int i = 0; i < all_niqeFeatures.size(); i++)  // ���30
	//{
	//	for (int j = 0; j < all_niqeFeatures[i].size(); i++)  // �ڲ�36
	//	{
	//		
	//	}
	//}

	Mat_<double> ft(all_niqeFeatures.size(), all_niqeFeatures[0].size());  // ��ά����ת����Mat�������Mat�������ֱ�Ϊ����С���ڲ��С
	for (size_t i = 0; i < all_niqeFeatures.size(); i++)
	{
		for (size_t j = 0; j < all_niqeFeatures[i].size(); j++)
		{
			ft(i, j) = all_niqeFeatures[i][j];
		}
	}

	Mat_<double> covMat, meanMat;
	calcCovarMatrix(ft, covMat, meanMat, cv::COVAR_NORMAL | cv::COVAR_ROWS);  //????
	covMat = covMat / (all_niqeFeatures.size() - 1);     // �����õ�Э�������ÿ��Ԫ�س��ԣ�n-1���󣬲�����Matlab����Ľ����ͬ����n�Ǿ����������

	fstream file1, file2;//�����ļ�������
	file1.open("./mu.txt");
	file2.open("./cov.txt");
	Mat mu_prisparam = Mat::zeros(1, 36, CV_32FC1);//����Mat����󣬶����ʼ��ֵȫ����0�������С��txtһ��
	Mat cov_prisparam = Mat::zeros(36, 36, CV_32FC1);
	for (int j = 0; j < 36; j++)
	{
		file1 >> mu_prisparam.at<float>(j);
	}
	for (int i = 0; i < 36; i++)
	{
		for (int j = 0; j < 36; j++)
		{
			file2 >> cov_prisparam.at<float>(i, j);
		}
	}
	mu_prisparam.convertTo(mu_prisparam, CV_64FC1);
	cov_prisparam.convertTo(cov_prisparam, CV_64FC1);
	Mat aa = mu_prisparam - meanMat;
	Mat bb = ((cov_prisparam + covMat) / 2).inv();
	Mat cc = aa.t();
	Mat aabb = aa*bb;
	Mat aabbcc = aabb*cc;
	aabbcc.convertTo(aabbcc, CV_32FC1);
	double score0 = aabbcc.at<float>(0, 0);
	*score = pow(score0, 0.5);

	return 0;
}