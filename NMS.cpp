#include"NMS.h"
#include<algorithm>


bool sort_bbox_according_score(bbox_t &A, bbox_t &B)
{
	return A.score > B.score;
}

float compute_interArea(bbox_t &A, bbox_t &B)
{
	float area_A, area_B, area_inter;
	float inter_bbox[4];
	float smallest_bbox_area;
	area_A = compute_bboxArea(A.Point);
	area_B = compute_bboxArea(B.Point);

	inter_bbox[0] = std::max(A.Point[0], B.Point[0]);
	inter_bbox[1] = std::max(A.Point[1], B.Point[1]);
	inter_bbox[2] = std::min(A.Point[2], B.Point[2]);
	inter_bbox[3] = std::min(A.Point[3], B.Point[3]);

	area_inter = compute_bboxArea(inter_bbox);

	smallest_bbox_area = std::min(area_A, area_B);
	return area_inter / smallest_bbox_area;
}

vector<bbox_t> NMS_bbox(std::vector<bbox_t> &one_cls_bbox_results, float area_thresh)
{
	vector<bool> obj_been_used(one_cls_bbox_results.size(), false);
	vector<bbox_t> results_bbox;

	//依照分數由高分排到低分
	sort(one_cls_bbox_results.begin(), one_cls_bbox_results.end(), sort_bbox_according_score);

	for (int i = 0; i < one_cls_bbox_results.size(); i++)
	{
		if (obj_been_used[i] == true) {
			continue;
		}

		for (int j = i + 1; j < one_cls_bbox_results.size(); j++)
		{
			if (obj_been_used[j] == true) {
				continue;
			}
			//Only known the bbox, but don't know which one is small;
			float inter_area = compute_interArea(one_cls_bbox_results[i], one_cls_bbox_results[j]);


			if (inter_area >= area_thresh)//相交範圍太大,且分數較bbox_i低分將他濾掉
			{
				obj_been_used[j] = true;
			}

		}
		results_bbox.push_back(one_cls_bbox_results[i]);
	}

	return results_bbox;
}