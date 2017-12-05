#include "symNet.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

void symNet::symSURFDetect(string& root, string& file)
{
	readSymSURFPair(root, file);
	_mkdir("symmetry");
	_chdir("symmetry");

	_mkdir("result");
	_mkdir("result_ROI");
	_mkdir("result_NMS");
	_chdir("..");
	for (int m = 0; m < list.size(); m++)
	{
		vector<struct bbox_t> boxScores;
		int count = 1;
		std::cout << m + 1 << "/" << list.size() << std::endl;

		cv::Mat img = cv::imread(root + "/image/" + list[m].filename + ".bmp");
		cv::Mat drawedImg = img.clone();

		for (int p = 0; p < list[m].symSURFPairNum; p++)
		{
			cv::Point center = list[m].center[p];

			if (center.x - ROI_width / 2 < 0 || center.y - ROI_height / 2 < 0 || center.x + ROI_width / 2 > img.cols || center.y + ROI_height / 2 > img.rows)
			{
				continue;
			}

			cv::Rect ROI(cv::Point(center.x - ROI_width / 2, center.y - ROI_height / 2), cv::Point(center.x + ROI_width / 2, center.y + ROI_height / 2));

			cv::Mat interest = img(ROI);
			cv::Mat interest_mirror;

			flip(interest, interest_mirror, 1);

			int patch_height = interest.rows / patch_h;
			int patch_width = interest.cols / patch_w;

			float** features, **features_mirror;

			features = new float*[patchNum];
			features_mirror = new float*[patchNum];

			for (int i = 0; i < patchNum; i++)
			{
				features[i] = new float[4096];
				features_mirror[i] = new float[4096];
			}

			float score = 0;

			bool jump = false;

			for (int i = 0; i < patch_h; i++)
			{
				for (int j = 0; j < patch_w; j++)
				{
					cv::Rect patch_ROI(0 + j*patch_width, 0 + j*patch_height, patch_width, patch_height);
					cv::Mat patch = interest(patch_ROI);
					cv::Mat patch_mirror = interest_mirror(patch_ROI);

					extracter.featureExtract(patch, features[i*patch_w + j]);
					extracter.featureExtract(patch_mirror, features_mirror[i*patch_w + j]);

					score += classifier.featureCompare(features[i*patch_w + j], features_mirror[i*patch_w + j]);

					if (score + patch_h*patch_w - i*patch_w + j - 1 < threshold * patchNum)
					{
						jump = true;
						break;
					}
				}
				if (jump)
					break;
			}

			if (jump)
			{
				for (int i = 0; i < patchNum; i++)
				{
					delete[] features[i];
					delete[] features_mirror[i];
				}
				delete[] features;
				delete[] features_mirror;
				continue;
			}

			if (createROI)
			{
				cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
				string filename = list[m].filename;
				filename += "_";
				filename += std::to_string(count++);
				filename += ".png";
				cv::imwrite("symmetry/result_ROI/" + filename, interest);
			}
			struct bbox_t temp;
			temp.Point[0] = center.x - ROI_width / 2;
			temp.Point[1] = center.y - ROI_height / 2;
			temp.Point[2] = center.x + ROI_width / 2;
			temp.Point[3] = center.y + ROI_height / 2;
			temp.score = score;

			boxScores.push_back(temp);

			for (int i = 0; i < patchNum; i++)
			{
				delete[] features[i];
				delete[] features_mirror[i];
			}
			delete[] features;
			delete[] features_mirror;
		}

		//output << testImages[m] << "," << score * 100 / patchNum << std::endl;
		cv::imwrite("symmetry/result/" + list[m].filename + ".png", drawedImg);
		//output << std::endl;

		if (useNMS)
		{
			boxScores = NMS_bbox(boxScores, 0.7);

			cv::Mat img_NMS = img.clone();
			for (int i = 0; i < boxScores.size(); i++)
			{
				const float *one_bbox_point = boxScores[i].Point;
				/*float width = one_bbox_point[2] - one_bbox_point[0];
				float height = one_bbox_point[3] - one_bbox_point[1];
				float left_top_x = one_bbox_point[0];
				float left_top_y = one_bbox_point[1];
				Rect one_rect(left_top_x, left_top_x, width, height);*/

				cv::rectangle(img_NMS, cv::Point(one_bbox_point[0], one_bbox_point[1]), cv::Point(one_bbox_point[2], one_bbox_point[3]), cv::Scalar(0, 0, 255), 1);

				//rectangle(img, one_rect, Scalar(255, 0, 0), 2);
			}

			cv::imwrite("symmetry/result_NMS/" + list[m].filename + ".png", img_NMS);
		}
	}
}

void symNet::readSymSURFPair(string& root, string& filename)
{
	_chdir(root.c_str());

	std::ifstream input(filename);

	string tempString;

	std::getline(input, tempString, '\n');
	int pairNum = atoi(tempString.c_str());

	int x, y;
	for (int m = 0; m < pairNum; m++)
	{
		struct symSURFImage temp;
		string line;
		std::getline(input, line, '\n');
		std::istringstream tempStream(line);
		std::getline(tempStream, temp.filename, ',');
		//testImages[m].filename += ".bmp";
		std::getline(tempStream, tempString, ',');
		temp.symSURFPairNum = atoi(tempString.c_str());

		temp.center = new cv::Point[temp.symSURFPairNum];
		temp.left = new cv::Point[temp.symSURFPairNum];
		temp.right = new cv::Point[temp.symSURFPairNum];

		for (int j = 0; j < temp.symSURFPairNum; j++)
		{
			std::getline(tempStream, tempString, ',');
			x = atoi(tempString.c_str());
			std::getline(tempStream, tempString, ',');
			y = atoi(tempString.c_str());
			temp.left[j] = cv::Point(x, y);

			std::getline(tempStream, tempString, ',');
			x = atoi(tempString.c_str());
			std::getline(tempStream, tempString, ',');
			y = atoi(tempString.c_str());
			temp.right[j] = cv::Point(x, y);

			std::getline(tempStream, tempString, ',');
			x = atoi(tempString.c_str());
			std::getline(tempStream, tempString, ',');
			y = atoi(tempString.c_str());
			temp.center[j] = cv::Point(x, y);
		}
		list.push_back(temp);
	}
}

void symNet::slidingWindowDetect(string& root, string& folder)
{
	readDirectory(root, folder);
	_mkdir("slidingWindow");
	_chdir("slidingWindow");

	_mkdir("result");
	if ( createROI)
		_mkdir("result_ROI");
	if ( useNMS)
		_mkdir("result_NMS");
	_chdir("..");
	for (int m = 0; m < list.size(); m++)
	{
		vector<struct bbox_t> boxScores;
		int count = 1;
		std::cout << m + 1 << "/" << list.size() << std::endl;

		cv::Mat img = cv::imread(root + "/" + folder + "/" + list[m].filename);
		cv::Mat drawedImg = img.clone();

		cv::Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;
		cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(grad_x, abs_grad_x);  //Âà¦¨CV_8U
		cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(grad_y, abs_grad_y);
		

		cv::Mat dst1, dst2;
		cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
		cv::threshold(dst1, dst2, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		int* integralImage;
		integralImage = new int[img.rows*img.cols];
		memset(integralImage, 0, sizeof(int)*img.rows*img.cols);

		/*cv::Ptr<cv::xfeatures2d::SURF> surf;

		surf = cv::xfeatures2d::SURF::create(800);
		std::vector<cv::KeyPoint> keyPoint;
		cv::Mat surfFeature;

		surf->detectAndCompute(gray, cv::Mat(), keyPoint, surfFeature);

		int* SURFIntegralImage;
		SURFIntegralImage = new int[img.rows*img.cols];
		memset(SURFIntegralImage, 0, sizeof(int)*img.rows*img.cols);

		for (int i = 0; i < keyPoint.size(); i++)
		{
			SURFIntegralImage[int(keyPoint[i].pt.y)*img.cols + int(keyPoint[i].pt.x)] = 1;
		}*/

		uchar* p_edge = dst2.ptr<uchar>();
		for (int y = 0; y < img.rows; y++, p_edge += dst2.step)
		{
			for (int x = 0; x < img.cols; x++)
			{
				if (p_edge[x] == 255)
				{
					integralImage[y*img.cols + x] = 1;
				}

				if (x == 0 && y == 0)
				{
					continue;
				}

				else if (y == 0)
				{
					integralImage[x] += integralImage[x - 1];

					//SURFIntegralImage[x] += SURFIntegralImage[x - 1];
				}
				else if (x == 0)
				{
					integralImage[y*img.cols] += integralImage[(y - 1)*img.cols];

					//SURFIntegralImage[y*img.cols] += SURFIntegralImage[(y - 1)*img.cols];
				}
				else
				{
					integralImage[y*img.cols + x] -= integralImage[(y - 1)*img.cols + x - 1];
					integralImage[y*img.cols + x] += integralImage[(y - 1)*img.cols + x];
					integralImage[y*img.cols + x] += integralImage[(y)*img.cols + x - 1];

					//SURFIntegralImage[y*img.cols + x] -= SURFIntegralImage[(y - 1)*img.cols + x - 1];
					//SURFIntegralImage[y*img.cols + x] += SURFIntegralImage[(y - 1)*img.cols + x];
					//SURFIntegralImage[y*img.cols + x] += SURFIntegralImage[(y)*img.cols + x - 1];
				}
			}
		}
		
		


		string filename = list[m].filename.substr(0, list[m].filename.size() - 4);
		for (int y = 0; y < img.rows; y += step)
		{
			for (int x = 0; x < img.cols; x += step)
			{
				if (x - ROI_width / 2 < 0 || y - ROI_height / 2 < 0 || x + ROI_width / 2 > img.cols || y + ROI_height / 2 > img.rows)
				{
					continue;
				}

				std::cout << x << "," << y << std::endl;

				int edgeNum;
				edgeNum = integralImage[(y + ROI_height / 2)*img.cols + x + ROI_width / 2];
				edgeNum += integralImage[(y - ROI_height / 2 - 1)*img.cols + x - ROI_width / 2 - 1];
				edgeNum -= integralImage[(y + ROI_height / 2)*img.cols + x - ROI_width / 2 - 1];
				edgeNum -= integralImage[(y - ROI_height / 2 - 1)*img.cols + x + ROI_width / 2];

				if (edgeNum < ROI_width * ROI_height * 0.3)
				{
					continue;
				}

				cv::Rect ROI(cv::Point(x - ROI_width / 2, y - ROI_height / 2), cv::Point(x + ROI_width / 2, y + ROI_height / 2));

				cv::Mat interest = img(ROI);
				cv::Mat interest_mirror;

				cv::flip(interest, interest_mirror, 1);

				int patch_height = interest.rows / patch_h;
				int patch_width = interest.cols / patch_w;

				float** features, **features_mirror;

				features = new float*[patchNum];
				features_mirror = new float*[patchNum];

				for (int i = 0; i < patchNum; i++)
				{
					features[i] = new float[4096];
					features_mirror[i] = new float[4096];
				}	

				float score = 0;

				bool jump = false;

				for (int i = 0; i < patch_h; i++)
				{
					for (int j = 0; j < patch_w; j++)
					{
						cv::Rect patch_ROI(0 + j*patch_width, 0 + i*patch_height, patch_width, patch_height);
						cv::Mat patch = interest(patch_ROI);
						cv::Mat patch_mirror = interest_mirror(patch_ROI);

						extracter.featureExtract(patch, features[i*patch_w + j]);
						extracter.featureExtract(patch_mirror, features_mirror[i*patch_w + j]);

						score += classifier.featureCompare(features[i*patch_w + j], features_mirror[i*patch_w + j]);

						if (score + patch_h*patch_w - i*patch_w + j - 1 < threshold * patchNum)
						{
							jump = true;
							break;
						}
					}
					if (jump)
						break;
				}

				if (jump)
				{
					for (int i = 0; i < patchNum; i++)
					{
						delete[] features[i];
						delete[] features_mirror[i];
					}
					delete[] features;
					delete[] features_mirror;
					continue;
				}
				//matchNet Score
				//std::cout << score << std::endl;
				
				/*int SURFNum;
				SURFNum = SURFIntegralImage[(y + ROI_height / 2)*img.cols + x + ROI_width / 2];
				SURFNum += SURFIntegralImage[(y - ROI_height / 2 - 1)*img.cols + x - ROI_width / 2 - 1];
				SURFNum -= SURFIntegralImage[(y + ROI_height / 2)*img.cols + x - ROI_width / 2 - 1];
				SURFNum -= SURFIntegralImage[(y - ROI_height / 2 - 1)*img.cols + x + ROI_width / 2];

				if (SURFNum < 10)
				{
					continue;
				}*/
				
				cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
				if (createROI)
				{	
					string filename_ROI = filename;
					filename_ROI += "_";
					filename_ROI += std::to_string(count++);
					filename_ROI += ".png";
					cv::imwrite("slidingWindow/result_ROI/" + filename_ROI, interest);
				}

				if (useNMS)
				{
					struct bbox_t temp;
					temp.Point[0] = x - ROI_width / 2;
					temp.Point[1] = y - ROI_height / 2;
					temp.Point[2] = x + ROI_width / 2;
					temp.Point[3] = y + ROI_height / 2;
					temp.score = score;

					boxScores.push_back(temp);
				}


				for (int i = 0; i < patchNum; i++)
				{
					delete[] features[i];
					delete[] features_mirror[i];
				}
				delete[] features;
				delete[] features_mirror;
				
			}
		}

		cv::imwrite("slidingWindow/result/" + filename + ".png", drawedImg);

		boxScores = NMS_bbox(boxScores, 0.7);

		if (useNMS)
		{
			cv::Mat img_NMS = img.clone();
			for (int i = 0; i < boxScores.size(); i++)
			{
				const float *one_bbox_point = boxScores[i].Point;
				/*float width = one_bbox_point[2] - one_bbox_point[0];
				float height = one_bbox_point[3] - one_bbox_point[1];
				float left_top_x = one_bbox_point[0];
				float left_top_y = one_bbox_point[1];
				Rect one_rect(left_top_x, left_top_x, width, height);*/

				cv::rectangle(img_NMS, cv::Point(one_bbox_point[0], one_bbox_point[1]), cv::Point(one_bbox_point[2], one_bbox_point[3]), cv::Scalar(0, 0, 255), 1);

				//rectangle(img, one_rect, Scalar(255, 0, 0), 2);
			}

			cv::imwrite("slidingWindow/result_NMS/" + filename + ".png", img_NMS);
		}

		delete[] integralImage;
	}
}

void symNet::readDirectory(string& root, string& folder)
{
	_chdir(root.c_str());

	struct dirent *drnt;
	DIR *dir;
	dir = opendir(folder.c_str());
	while (drnt = readdir(dir))
	{
		if (drnt->d_type == DT_REG)
		{
			struct symSURFImage temp;
			temp.filename = drnt->d_name;
			list.push_back(temp);
		}
	}
}

void symNet::singleImage(string& root, string& folder)
{
	readDirectory(root, folder);

	for (int m = 0; m < list.size(); m++)
	{
		cv::Mat img = cv::imread(root + '/' + folder + '/' + list[m].filename);
		cv::Mat mirror;
		cv::flip(img, mirror, 1);

		int patch_height = img.rows / patch_h;
		int patch_width = img.cols / patch_w;

		float** features, **features_mirror;

		features = new float*[patchNum];
		features_mirror = new float*[patchNum];

		for (int i = 0; i < patchNum; i++)
		{
			features[i] = new float[4096];
			features_mirror[i] = new float[4096];
		}

		float score = 0;

		for (int i = 0; i < patch_h; i++)
		{
			for (int j = 0; j < patch_w; j++)
			{
				cv::Rect patch_ROI(0 + j*patch_width, 0 + i*patch_height, patch_width, patch_height);
				cv::Mat patch = img(patch_ROI);
				cv::Mat patch_mirror = mirror(patch_ROI);

				extracter.featureExtract(patch, features[i*patch_w + j]);
				extracter.featureExtract(patch_mirror, features_mirror[i*patch_w + j]);

				score += classifier.featureCompare(features[i*patch_w + j], features_mirror[i*patch_w + j]);
			}
			
		}

		std::cout << score / patchNum << std::endl;

		for (int i = 0; i < patchNum; i++)
		{
			delete[] features[i];
			delete[] features_mirror[i];
		}
		delete[] features;
		delete[] features_mirror;
	}
}