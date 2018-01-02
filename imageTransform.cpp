#include "imageTransform.h"

cv::Mat imageTransform::localBinaryPattern(const cv::Mat& input)
{
	int i, j;

	cv::Mat grayscale;
	if (input.type() == CV_8UC3)
	{
		cv::cvtColor(input, grayscale, CV_BGR2GRAY);
	}
	else
	{
		grayscale = input.clone();
	}

	cv::Mat output(input.rows, input.cols, CV_8UC1);
	uchar* p_grayscale = grayscale.ptr<uchar>();
	for (i = 1; i < input.rows - 1; i++)
	{
		for (j = 1; j < input.cols - 1; j++)
		{
			uchar temp = grayscale.at<uchar>(i, j);
			uchar num = 0;
			if (temp > grayscale.at<uchar>(i - 1, j - 1)) num = num | 1;
			if (temp > grayscale.at<uchar>(i, j - 1)) num = num | 2;
			if (temp > grayscale.at<uchar>(i + 1, j - 1)) num = num | 4;
			if (temp > grayscale.at<uchar>(i + 1, j)) num = num | 8;
			if (temp > grayscale.at<uchar>(i + 1, j + 1)) num = num | 16;
			if (temp > grayscale.at<uchar>(i, j + 1)) num = num | 32;
			if (temp > grayscale.at<uchar>(i - 1, j + 1)) num = num | 64;
			if (temp > grayscale.at<uchar>(i - 1, j)) num = num | 128;

			output.at<uchar>(i, j) = num;
		}
	}

	return output;
}

void imageTransform::nakagami(const cv::Mat* input, cv::Mat*output, int maskSize)
{
	int i, j, x, y, mean_element;
	double mean, variance;
	cv::Vec3b temp;
	uchar tempUchar;
	cv::Mat tempImg(input->rows + maskSize - 1, input->cols + maskSize - 1, input->type(), cv::Scalar(0));
	input->copyTo(tempImg.rowRange(maskSize / 2, maskSize / 2 + input->rows).colRange(maskSize / 2, maskSize / 2 + input->cols));

	for (i = maskSize / 2; i < maskSize / 2 + input->rows; i++)
	{
		for (j = maskSize / 2; j < maskSize / 2 + input->cols; j++)
		{
			mean = 0;
			mean_element = 0;
			for (y = i - maskSize / 2; y <= i + maskSize / 2; y++)
			{
				for (x = j - maskSize / 2; x <= j + maskSize / 2; x++)
				{
					temp = tempImg.at<cv::Vec3b>(y, x);
					if (temp.val[0] == temp.val[1] && temp.val[0] == temp.val[2] && temp.val[0] != 0 && temp.val[1] != 0 && temp.val[2] != 0)
					{
						tempUchar = (temp.val[0] + temp.val[1] + temp.val[2]) / 3;
						mean += tempUchar * tempUchar;
						mean_element++;
					}
				}
			}
			if (mean == 0 || mean_element == 0)
			{
				output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = 0;
			}

			mean /= (double)mean_element;
			variance = 0;
			for (y = i - maskSize / 2; y <= i + maskSize / 2; y++)
			{
				for (x = j - maskSize / 2; x <= j + maskSize / 2; x++)
				{
					temp = tempImg.at<cv::Vec3b>(y, x);
					if (temp.val[0] == temp.val[1] && temp.val[0] == temp.val[2] && temp.val[0] != 0 && temp.val[1] != 0 && temp.val[2] != 0)
					{
						tempUchar = (temp.val[0] + temp.val[1] + temp.val[2]) / 3;
						variance += (tempUchar * tempUchar - mean) * (tempUchar * tempUchar - mean);
					}
				}
			}
			variance /= (double)mean_element;

			if (variance == 0)
			{
				output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = 0;
			}
			output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = mean * mean / variance;
		}
	}
}

cv::Mat imageTransform::imageRotate(const cv::Mat& img, float angle)
{
	//0 : leftupper, 1 : rightupper, 2 : leftbelow, 3 : rightbelow
	int width = img.cols, height = img.rows;

	float radius = angle * M_PI / 180;

	while (angle > 360)
	{
		angle -= 360;
	}

	cv::Mat rotatedImg;
	if (angle == 0)
	{
		rotatedImg = img.clone();
	}
	else
	{
		cv::Point corner[4];
		cv::Point center = cv::Point(width / 2, height / 2);

		corner[0] = cv::Point(0, 0);
		corner[1] = cv::Point(width - 1, 0);
		corner[2] = cv::Point(0, height - 1);
		corner[3] = cv::Point(width - 1, height - 1);

		corner[0].y = height - 1 - corner[0].y;
		corner[1].y = height - 1 - corner[1].y;
		corner[2].y = height - 1 - corner[2].y;
		corner[3].y = height - 1 - corner[3].y;

		corner[0] -= center;
		corner[1] -= center;
		corner[2] -= center;
		corner[3] -= center;

		cv::Vec2f rotate = cv::Vec2f(sin(radius), cos(radius));
		cv::Vec2f inverseRotate = cv::Vec2f(sin(-radius), cos(-radius));

		cv::Point rotatedCorner[4];

		rotatedCorner[0] = cv::Point(rotate[1] * corner[0].x - rotate[0] * corner[0].y, rotate[0] * corner[0].x + rotate[1] * corner[0].y);
		rotatedCorner[1] = cv::Point(rotate[1] * corner[1].x - rotate[0] * corner[1].y, rotate[0] * corner[1].x + rotate[1] * corner[1].y);
		rotatedCorner[2] = cv::Point(rotate[1] * corner[2].x - rotate[0] * corner[2].y, rotate[0] * corner[2].x + rotate[1] * corner[2].y);
		rotatedCorner[3] = cv::Point(rotate[1] * corner[3].x - rotate[0] * corner[3].y, rotate[0] * corner[3].x + rotate[1] * corner[3].y);

		int newHeight, newWidth;
		if (angle == 180)
		{
			newHeight = height;
			newWidth = width;
		}
		else if (angle == 90 || angle == 270)
		{
			newHeight = width;
			newWidth = height;
		}
		else if ((angle < 90 && angle > 0) || (angle < 270 && angle > 180))
		{
			newHeight = abs(rotatedCorner[1].y - rotatedCorner[2].y);
			newWidth = abs(rotatedCorner[0].x - rotatedCorner[3].x);
		}
		else
		{
			newHeight = abs(rotatedCorner[0].y - rotatedCorner[3].y);
			newWidth = abs(rotatedCorner[1].x - rotatedCorner[2].x);
		}

		rotatedImg = cv::Mat(newHeight, newWidth, img.type());
		cv::Point rotatedImgCenter = cv::Point(newWidth / 2, newHeight / 2);
		for (int i = 0; i < newHeight; i++)
		{
			for (int j = 0; j < newWidth; j++)
			{
				cv::Point originP(j, newHeight - 1 - i);
				originP -= rotatedImgCenter;
				cv::Point rotatedP(inverseRotate[1] * originP.x - inverseRotate[0] * originP.y, inverseRotate[0] * originP.x + inverseRotate[1] * originP.y);

				if ((rotatedP.x <= corner[3].x && rotatedP.x >= corner[0].x) && (rotatedP.y >= corner[3].y && rotatedP.y <= corner[0].y))
				{
					rotatedP += center;
					rotatedP.y = height - 1 - rotatedP.y;
					rotatedImg.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(rotatedP);
				}
				else
				{
					rotatedImg.at<cv::Vec3b>(i, j) = cv::Vec3b::all(0);
				}
			}
		}
	}

	return rotatedImg;
};

void imageTransform::gradientAndAngle(cv::Mat& img, cv::Mat& gradient, cv::Mat& angle)
{
	int i, j;
	int rows = img.rows;
	int cols = img.cols;

	float max_gradient = 0;

	cv::Mat grayscale(rows, cols, CV_8UC1);

	uchar* p_grayscale;
	if (img.type() == CV_8UC3)
	{
		uchar* p_img = img.ptr<uchar>();
		p_grayscale = grayscale.ptr<uchar>();
		for (i = 0; i < rows * cols; i++)
		{
			p_grayscale[i] = (p_img[i * 3] + p_img[i * 3 + 1] + p_img[i * 3 + 2]) / 3;
		}
	}
	else
		grayscale = img;

	cv::Mat diffX(rows, cols, CV_32FC1);
	cv::Mat diffY(rows, cols, CV_32FC1);
	cv::Mat gradient_F = cv::Mat(rows, cols, CV_32FC1);
	cv::Mat angle_F = cv::Mat(rows, cols, CV_32FC1);

	p_grayscale = grayscale.ptr<uchar>();

	for (int i = 0; i < rows; i++, p_grayscale += grayscale.step)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1)
			{
				diffX.at<float>(i, j) = 0;
				diffY.at<float>(i, j) = 0;
				gradient_F.at<float>(i, j) = 0;
				angle_F.at<float>(i, j) = 0;
			}
			else
			{
				diffX.at<float>(i, j) = p_grayscale[j + 1] - p_grayscale[j - 1];
				diffY.at<float>(i, j) = p_grayscale[j + cols] - p_grayscale[j - cols];
				gradient_F.at<float>(i, j) = sqrt(pow(diffX.at<float>(i, j), 2) + pow(diffY.at<float>(i, j), 2));

				if (gradient_F.at<float>(i, j) > max_gradient)
					max_gradient = gradient_F.at<float>(i, j);

				if (diffX.at<float>(i, j) >= 0 && diffY.at<float>(i, j) == 0)
				{
					angle_F.at<float>(i, j) = 0;
				}
				else if (diffX.at<float>(i, j) == 0 && diffY.at<float>(i, j) >= 0)
				{
					angle_F.at<float>(i, j) = 90;
				}
				else if (diffX.at<float>(i, j) < 0 && diffY.at<float>(i, j) == 0)
				{
					angle_F.at<float>(i, j) = 180;
				}
				else if (diffX.at<float>(i, j) == 0 && diffY.at<float>(i, j) >= 0)
				{
					angle_F.at<float>(i, j) = 270;
				}
				//area 1
				else if (diffX.at<float>(i, j) >= 0 && diffY.at<float>(i, j) >= 0)
				{
					angle_F.at<float>(i, j) = atan(diffY.at<float>(i, j) / diffX.at<float>(i, j)) * (180 / M_PI);
				}
				//area 2
				else if (diffX.at<float>(i, j) < 0 && diffY.at<float>(i, j) >= 0)
				{
					angle_F.at<float>(i, j) = atan(diffY.at<float>(i, j) / diffX.at<float>(i, j)) * (180 / M_PI) + 90;
				}
				//area 3
				else if (diffX.at<float>(i, j) < 0 && diffY.at<float>(i, j) < 0)
				{
					angle_F.at<float>(i, j) = atan(diffY.at<float>(i, j) / diffX.at<float>(i, j)) * (180 / M_PI) + 180;
				}
				//area 4
				else if (diffX.at<float>(i, j) >= 0 && diffY.at<float>(i, j) < 0)
				{
					angle_F.at<float>(i, j) = atan(diffY.at<float>(i, j) / diffX.at<float>(i, j)) * (180 / M_PI) + 270;
				}
			}
		}
	}

	gradient = cv::Mat(rows, cols, CV_8UC1);
	angle = cv::Mat(rows, cols, CV_8UC1);

	int binClassNum = 36;
	int binClassRnage = 360 / binClassNum;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int binClass = angle_F.at<float>(i, j) / binClassRnage;

			angle.at<uchar>(i, j) = binClassRnage * binClass;

			gradient.at<uchar>(i, j) = gradient_F.at<float>(i, j) * 255 / max_gradient;
		}
	}
};