#include "symNet.h"

symNet::symNet(string& feature_trained_filename, string& feature_model_filename, string& classifier_trained_filename, string& classifier_model_filename)
{
	extracter = Extracter::Extracter(feature_trained_filename, feature_model_filename);
	classifier = Classifier::Classifier(classifier_trained_filename, classifier_model_filename);
}

symNet::~symNet()
{
}

void symNet::setPatch(int width, int height)
{
	patch_w = width;
	patch_h = height;
	patchNum = width*height;
}

void symNet::setROI(int width, int height)
{
	ROI_width = width;
	ROI_height = height;
}

void symNet::setStep(int step_new)
{
	step = step_new;
}

void symNet::setThreshold(float threshold_new)
{
	threshold = threshold_new;
}

void symNet::symSURFDetect(string& root, string& file)
{
	readSymSURFPair(root, file);

	_mkdir("result");
	_mkdir("result_ROI");
	_mkdir("result_NMS");
	for (int m = 0; m < list.size(); m++)
	{
		vector<struct bbox_t> boxScores;
		int count = 1;
		std::cout << m + 1 << "/" << list.size() << std::endl;

		cv::Mat img = cv::imread(root + "/image/" + list[m].filename + ".jpg");
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
				continue;

			cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
			string filename = list[m].filename;
			filename += "_";
			filename += std::to_string(count++);
			filename += ".png";
			cv::imwrite("result_ROI/" + filename, interest);
			
			struct bbox_t temp;
			temp.Point[0] = center.x - ROI_width / 2;
			temp.Point[1] = center.y - ROI_height / 2;
			temp.Point[2] = center.x + ROI_width / 2;
			temp.Point[3] = center.y + ROI_height / 2;
			temp.score = score;

			boxScores.push_back(temp);
		}

		//output << testImages[m] << "," << score * 100 / patchNum << std::endl;
		cv::imwrite("result/" + list[m].filename + ".png", drawedImg);
		//output << std::endl;

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

		cv::imwrite("result_NMS/" + list[m].filename + ".png", img_NMS);
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

	_mkdir("result");
	_mkdir("result_ROI");
	_mkdir("result_NMS");
	for (int m = 0; m < list.size(); m++)
	{
		vector<struct bbox_t> boxScores;
		int count = 1;
		std::cout << m + 1 << "/" << list.size() << std::endl;

		cv::Mat img = cv::imread(root + "/" + folder + "/" + list[m].filename);
		cv::Mat drawedImg = img.clone();

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
				cv::Rect ROI(cv::Point(x - ROI_width / 2, y - ROI_height / 2), cv::Point(x + ROI_width / 2, y + ROI_height / 2));

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
					continue;

				std::cout << score << std::endl;

				cv::rectangle(drawedImg, ROI, cv::Scalar(0, 0, 255));
				string filename_ROI = filename;
				filename_ROI += "_";
				filename_ROI += std::to_string(count++);
				filename_ROI += ".png";
				imwrite("result_ROI/" + filename_ROI, interest);

				struct bbox_t temp;
				temp.Point[0] = x - ROI_width / 2;
				temp.Point[1] = y - ROI_height / 2;
				temp.Point[2] = x + ROI_width / 2;
				temp.Point[3] = y + ROI_height / 2;
				temp.score = score;

				boxScores.push_back(temp);
			}
		}

		imwrite("result/" + filename + ".png", drawedImg);

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

		cv::imwrite("result_NMS/" + filename + ".png", img_NMS);
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