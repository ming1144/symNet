#pragma once
#ifndef _NMS_H_
#define _NMS_H_

#include<vector>
using namespace std;

struct bbox_t {
	float Point[4];//x1, y1, x2, y2 -->左上角x,y 右下角x,y
	float score;
};


/*----濾掉同類別重複的框框----*/
inline float compute_bboxArea(float *A)
{
	float width = A[2] - A[0];
	float height = A[3] - A[1];

	width = (width > 0.0) ? width : 0;
	height = (height > 0.0) ? height : 0;

	return width*height;
}
float compute_interArea(bbox_t &A, bbox_t &B);
vector<bbox_t> NMS_bbox(std::vector<bbox_t> &one_cls_bbox_results, float area_thresh);
bool sort_bbox_according_score(bbox_t &A, bbox_t &B);

#endif