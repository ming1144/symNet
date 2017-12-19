#define _CRT_SECURE_NO_WARNINGS
#include "symNet.h"

#include <ctime>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>

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

	std::time_t t = std::time(0);
	struct tm * now = localtime(&t);

	string resultFolder = "slidingWindow-";
	resultFolder += to_string(now->tm_year + 1900);
	resultFolder += to_string(now->tm_mon + 1);
	resultFolder += to_string(now->tm_mday);
	resultFolder += '-';
	resultFolder += modelDate;
	if (useCrossEntropy)
		resultFolder += "-CrossEntropy";

	_mkdir(resultFolder.c_str());
	_chdir(resultFolder.c_str());

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


		//edge Detetct
		cv::Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;
		cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U
		cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(grad_y, abs_grad_y);
		
		gray.release();
		grad_x.release();
		grad_y.release();

		cv::Mat dst1, dst2;
		cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
		cv::threshold(dst1, dst2, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		int* integralImage;
		integralImage = new int[img.rows*img.cols];
		memset(integralImage, 0, sizeof(int)*img.rows*img.cols);

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
				}
				else if (x == 0)
				{
					integralImage[y*img.cols] += integralImage[(y - 1)*img.cols];
				}
				else
				{
					integralImage[y*img.cols + x] -= integralImage[(y - 1)*img.cols + x - 1];
					integralImage[y*img.cols + x] += integralImage[(y - 1)*img.cols + x];
					integralImage[y*img.cols + x] += integralImage[(y)*img.cols + x - 1];
				}
			}
		}
		
		abs_grad_x.release();
		abs_grad_y.release();
		dst1.release();
		dst2.release();


		string filename = list[m].filename.substr(0, list[m].filename.size() - 4);
		for (int y = 0; y < img.rows; y += step_h)
		{
			for (int x = 0; x < img.cols; x += step_w)
			{
				if (x - ROI_width / 2 < 0 || y - ROI_height / 2 < 0 || x + ROI_width / 2 > img.cols || y + ROI_height / 2 > img.rows)
				{
					continue;
				}

				std::cout << x << "," << y << std::endl;

				int edgenum;
				edgenum = integralImage[(y + ROI_height / 2)*img.cols + x + ROI_width / 2];
				edgenum += integralImage[(y - ROI_height / 2 - 1)*img.cols + x - ROI_width / 2 - 1];
				edgenum -= integralImage[(y + ROI_height / 2)*img.cols + x - ROI_width / 2 - 1];
				edgenum -= integralImage[(y - ROI_height / 2 - 1)*img.cols + x + ROI_width / 2];

				if (edgenum < ROI_width * ROI_height * 0.3)
				{
					continue;
				}

				cv::Rect ROI(cv::Point(x - ROI_width / 2, y - ROI_height / 2), cv::Point(x + ROI_width / 2, y + ROI_height / 2));

				cv::Mat interest = img(ROI);
				cv::Mat interest_mirror;

				cv::flip(interest, interest_mirror, 1);


				//MatchNet
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
							patch.release();
							patch_mirror.release();
							break;
						}

						patch.release();
						patch_mirror.release();
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
					interest.release();
					interest_mirror.release();
					continue;
				}
				//matchNet Score
				//std::cout << score << std::endl;

				if (!onlyMatchNet)
				{
					//SymNet
					std::vector<Prediction> predict = symClassifier.Classify(interest);

					if (!useCrossEntropy)
					{
						if (predict[0].first != "1")
						{
							for (int i = 0; i < patchNum; i++)
							{
								delete[] features[i];
								delete[] features_mirror[i];
							}
							delete[] features;
							delete[] features_mirror;
							interest.release();
							interest_mirror.release();
							continue;
						}
					}
					else
					{
						if (predict[0].second >= CrossEntropyThreshold)
						{
							for (int i = 0; i < patchNum; i++)
							{
								delete[] features[i];
								delete[] features_mirror[i];
							}
							delete[] features;
							delete[] features_mirror;
							interest.release();
							interest_mirror.release();
							continue;
						}
					}
				}

				cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
				if (createROI)
				{	
					string filename_ROI = filename;
					filename_ROI += "_";
					filename_ROI += std::to_string(count++);
					filename_ROI += ".png";
					cv::imwrite(resultFolder + "/result_ROI/" + filename_ROI, interest);
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

				interest.release();
				interest_mirror.release();
				
			}
		}

		cv::imwrite(resultFolder + "/result/" + filename + ".png", drawedImg);

		if (useNMS)
		{
			boxScores = NMS_bbox(boxScores, NMSThreshold);
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

			cv::imwrite(resultFolder + "/result_NMS/" + filename + ".png", img_NMS);
			img_NMS.release();
		}

		delete[] integralImage;
		img.release();
		drawedImg.release();
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

void symNet::singleImage(string& root, string& imagePath)
{
	cv::Mat img = cv::imread(root + '/' + imagePath );
	cv::Mat drawedImg = img.clone();

	vector<struct bbox_t> boxScores;
	int count = 1;

	//edge Detetct
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U
	cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	gray.release();
	grad_x.release();
	grad_y.release();

	cv::Mat dst1, dst2;
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
	cv::threshold(dst1, dst2, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	int* integralImage;
	integralImage = new int[img.rows*img.cols];
	memset(integralImage, 0, sizeof(int)*img.rows*img.cols);

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
			}
			else if (x == 0)
			{
				integralImage[y*img.cols] += integralImage[(y - 1)*img.cols];
			}
			else
			{
				integralImage[y*img.cols + x] -= integralImage[(y - 1)*img.cols + x - 1];
				integralImage[y*img.cols + x] += integralImage[(y - 1)*img.cols + x];
				integralImage[y*img.cols + x] += integralImage[(y)*img.cols + x - 1];
			}
		}
	}

	abs_grad_x.release();
	abs_grad_y.release();
	dst1.release();
	dst2.release();

	string filename = imagePath.substr(0, imagePath.size() - 4);
	_mkdir(filename.c_str());
	_chdir(filename.c_str());
	_mkdir("ROI");
	_chdir("..");
	for (int y = 0; y < img.rows; y += step_h)
	{
		for (int x = 0; x < img.cols; x += step_w)
		{
			if (x - ROI_width / 2 < 0 || y - ROI_height / 2 < 0 || x + ROI_width / 2 > img.cols || y + ROI_height / 2 > img.rows)
			{
				continue;
			}

			std::cout << x << "," << y << std::endl;

			int edgenum;
			edgenum = integralImage[(y + ROI_height / 2)*img.cols + x + ROI_width / 2];
			edgenum += integralImage[(y - ROI_height / 2 - 1)*img.cols + x - ROI_width / 2 - 1];
			edgenum -= integralImage[(y + ROI_height / 2)*img.cols + x - ROI_width / 2 - 1];
			edgenum -= integralImage[(y - ROI_height / 2 - 1)*img.cols + x + ROI_width / 2];

			if (edgenum < ROI_width * ROI_height * 0.3)
			{
				continue;
			}

			cv::Rect ROI(cv::Point(x - ROI_width / 2, y - ROI_height / 2), cv::Point(x + ROI_width / 2, y + ROI_height / 2));

			cv::Mat interest = img(ROI);
			cv::Mat interest_mirror;

			cv::flip(interest, interest_mirror, 1);

			//MatchNet
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
						patch.release();
						patch_mirror.release();
						break;
					}

					patch.release();
					patch_mirror.release();
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
				interest.release();
				interest_mirror.release();
				continue;
			}
			//matchNet Score
			//std::cout << score << std::endl;

			if (!onlyMatchNet)
			{
				//SymNet
				std::vector<Prediction> predict = symClassifier.Classify(interest);

				if (!useCrossEntropy)
				{
					if (predict[0].first != "1")
					{
						for (int i = 0; i < patchNum; i++)
						{
							delete[] features[i];
							delete[] features_mirror[i];
						}
						delete[] features;
						delete[] features_mirror;
						interest.release();
						interest_mirror.release();
						continue;
					}
				}
				else
				{
					if (predict[0].second >= CrossEntropyThreshold)
					{
						for (int i = 0; i < patchNum; i++)
						{
							delete[] features[i];
							delete[] features_mirror[i];
						}
						delete[] features;
						delete[] features_mirror;
						interest.release();
						interest_mirror.release();
						continue;
					}
				}
			}

			cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
			string filename_ROI = filename;
			filename_ROI += "_";
			filename_ROI += std::to_string(count++);
			filename_ROI += ".png";
			cv::imwrite(filename + "/ROI/" + filename_ROI, interest);

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

			interest.release();
			interest_mirror.release();

		}
	}

	cv::imwrite(filename + "/" + filename + ".png", drawedImg);

	if (useNMS)
	{
		boxScores = NMS_bbox(boxScores, NMSThreshold);
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

		cv::imwrite(filename + "/" + filename + "_NMS.png", img_NMS);
		img_NMS.release();
	}

	int times = 0, i, j;
    vector<bbox_t> updateBoxes;
	bool* used;
	do
	{
		times = 0;
		used = new bool[boxScores.size()];
		memset(used, 0, sizeof(bool)*boxScores.size());

		for (i = 0; i < boxScores.size(); i++)
		{
			if (used[i])
				continue;
			for (j = i + 1; j < boxScores.size(); j++)
			{
				if (used[j])
					continue;

				int border_x_min, border_x_max, border_y_min, border_y_max;

				border_x_min = boxScores[i].Point[0];
				border_y_min = boxScores[i].Point[1];
				border_x_max = boxScores[i].Point[2];
				border_y_max = boxScores[i].Point[3];

				int pos;

				cv::Point leftAbove, leftBelow, rightAbove, rightBelow;

				leftBelow.x = boxScores[j].Point[0];
				leftBelow.y = boxScores[j].Point[1];
				leftAbove.x = boxScores[j].Point[0];
				leftAbove.y = boxScores[j].Point[3];
				rightBelow.x = boxScores[j].Point[2];
				rightBelow.y = boxScores[j].Point[1];
				rightAbove.x = boxScores[j].Point[2];
				rightAbove.y = boxScores[j].Point[3];

				if (leftBelow.x >= border_x_min && leftBelow.x <= border_x_max && leftBelow.y >= border_y_min && leftBelow.y <= border_y_max)
				{
					pos = 1;
				}
				else if (leftAbove.x >= border_x_min && leftAbove.x <= border_x_max && leftAbove.y >= border_y_min && leftAbove.y <= border_y_max)
				{
					pos = 2;
				}
				else if (rightBelow.x >= border_x_min && rightBelow.x <= border_x_max && rightBelow.y >= border_y_min && rightBelow.y <= border_y_max)
				{
					pos = 3;
				}
				else if (rightAbove.x >= border_x_min && rightAbove.x <= border_x_max && rightAbove.y >= border_y_min && rightAbove.y <= border_y_max)
				{
					pos = 4;
				}
				else
				{
					continue;
				}

				cv::Point newLeftBelow, newRightAbove;
				switch (pos)
				{
				case 1:
					newLeftBelow.x = boxScores[i].Point[0];
					newLeftBelow.y = boxScores[i].Point[1];
					newRightAbove.x = boxScores[j].Point[2];
					newRightAbove.y = boxScores[j].Point[3];
					break;
				case 2:
					newLeftBelow.x = boxScores[i].Point[0];
					newLeftBelow.y = boxScores[j].Point[1];
					newRightAbove.x = boxScores[j].Point[2];
					newRightAbove.y = boxScores[i].Point[3];
					break;
				case 3:
					newLeftBelow.x = boxScores[j].Point[0];
					newLeftBelow.y = boxScores[i].Point[1];
					newRightAbove.x = boxScores[i].Point[2];
					newRightAbove.y = boxScores[j].Point[3];
					break;
				case 4:
					newLeftBelow.x = boxScores[j].Point[0];
					newLeftBelow.y = boxScores[j].Point[1];
					newRightAbove.x = boxScores[i].Point[2];
					newRightAbove.y = boxScores[i].Point[3];
					break;
				}

				int edgenum;
				edgenum = integralImage[newRightAbove.y*img.cols + newRightAbove.x];
				edgenum += integralImage[(newLeftBelow.y - 1)*img.cols + newLeftBelow.x - 1];
				edgenum -= integralImage[newRightAbove.y*img.cols + newLeftBelow.x - 1];
				edgenum -= integralImage[(newLeftBelow.y - 1)*img.cols + newRightAbove.x];

				if (edgenum < (newRightAbove.x - newLeftBelow.x) * (newRightAbove.y - newLeftBelow.y) * 0.3)
				{
					continue;
				}

				cv::Rect ROI(newLeftBelow, newRightAbove);

				cv::Mat interest = img(ROI);
				cv::Mat interest_mirror;

				cv::flip(interest, interest_mirror, 1);

				

				int new_patch_w = ROI.width / ROI_width;
				if (ROI.width % ROI_width >= ROI_width/2)
					new_patch_w++;
				int new_patch_h = ROI.height / ROI_height;
				if (ROI.height % ROI_height >= ROI_height/2)
					new_patch_h++;

				int new_patch_width = ROI.width / new_patch_w;
				int new_patch_height = ROI.height / new_patch_h;
				int new_patchNum = new_patch_w * new_patch_h;
				
				float** features, **features_mirror;

				features = new float*[new_patchNum];
				features_mirror = new float*[new_patchNum];

				for (int i = 0; i < new_patchNum; i++)
				{
					features[i] = new float[4096];
					features_mirror[i] = new float[4096];
				}

				float score = 0;

				bool jump = false;

				for (int i = 0; i < new_patch_h; i++)
				{
					for (int j = 0; j < new_patch_w; j++)
					{
						cv::Rect patch_ROI(0 + j*new_patch_width, 0 + i*new_patch_height, new_patch_width, new_patch_height);
						cv::Mat patch = interest(patch_ROI);
						cv::Mat patch_mirror = interest_mirror(patch_ROI);

						extracter.featureExtract(patch, features[i*new_patch_w + j]);
						extracter.featureExtract(patch_mirror, features_mirror[i*new_patch_w + j]);

						score += classifier.featureCompare(features[i*new_patch_w + j], features_mirror[i*new_patch_w + j]);

						if (score + new_patch_h*new_patch_w - i*new_patch_w + j - 1 < threshold * new_patchNum)
						{
							jump = true;
							patch.release();
							patch_mirror.release();
							break;
						}

						patch.release();
						patch_mirror.release();
					}
					if (jump)
						break;
				}

				if (jump)
				{
					for (int i = 0; i < new_patchNum; i++)
					{
						delete[] features[i];
						delete[] features_mirror[i];
					}
					delete[] features;
					delete[] features_mirror;
					interest.release();
					interest_mirror.release();
					continue;
				}
				//matchNet Score
				//std::cout << score << std::endl;

				if (!onlyMatchNet)
				{
					//SymNet
					std::vector<Prediction> predict = symClassifier.Classify(interest);

					if (!useCrossEntropy)
					{
						if (predict[0].first != "1")
						{
							for (int i = 0; i < new_patchNum; i++)
							{
								delete[] features[i];
								delete[] features_mirror[i];
							}
							delete[] features;
							delete[] features_mirror;
							interest.release();
							interest_mirror.release();
							continue;
						}
					}
					else
					{
						if (predict[0].second >= CrossEntropyThreshold)
						{
							for (int i = 0; i < new_patchNum; i++)
							{
								delete[] features[i];
								delete[] features_mirror[i];
							}
							delete[] features;
							delete[] features_mirror;
							interest.release();
							interest_mirror.release();
							continue;
						}
					}
				}

				bbox_t temp;
				temp.score = score;
				temp.Point[0] = newLeftBelow.x;
				temp.Point[1] = newLeftBelow.y;
				temp.Point[2] = newRightAbove.x;
				temp.Point[3] = newRightAbove.y;
				updateBoxes.push_back(temp);

				used[j] = true;
				times++;
				break;
			}
			if (j != boxScores.size())
				used[i] = true;

		}

		for (i = 0; i < boxScores.size(); i++)
		{
			if (!used[i])
			{
				updateBoxes.push_back(boxScores[i]);
			}
		}

		boxScores.clear();
		boxScores = updateBoxes;
		updateBoxes.clear();
		delete used;
	} while (times != 0);

	_chdir(filename.c_str());
	_mkdir("2");
	_chdir("..");
	
	cv::Mat drawedImg2 = img.clone();
	boxScores = NMS_bbox(boxScores, NMSThreshold);
	for (i = 0; i < boxScores.size(); i++)
	{
		const float *one_bbox_point = boxScores[i].Point;
		cv::Rect temp(cv::Point(one_bbox_point[0], one_bbox_point[1]), cv::Point(one_bbox_point[2], one_bbox_point[3]));
		cv::Mat tempMat = img(temp);
		cv::imwrite(filename + "/2/" + filename + "_" + to_string(i) +".png", tempMat);

		cv::rectangle(drawedImg2, temp, cv::Scalar(0, 0, 255), 1);
	}
	cv::imwrite(filename + "/" + filename + "_2.png", drawedImg2);
	drawedImg2.release();

	delete[] integralImage;
	img.release();
	drawedImg.release();
}