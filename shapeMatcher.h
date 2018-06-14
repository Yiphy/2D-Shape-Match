#ifndef SHAPE_MATCHER_2D_H_
#define SHAPE_MATCHER_2D_H_

#include"headers.h"

#define MAX_DETECT_NUM 10000

typedef struct _Target_Info
{
	double x;
	double y;
	double angle;

	double similarity;

}TargetInfo;

typedef struct _Shape_Targets
{
	int nTargetsNumber;
	TargetInfo tarInfo[MAX_DETECT_NUM];

}ShapeTargets;

class ShapeMatcher2d
{
public:
	ShapeMatcher2d();
	~ShapeMatcher2d();

	bool creatModel(Mat tmpSrc, double angleStep, double angleStart = 0, double angleEnd = 360);

	bool match(Mat dst, double similarityThres = 0.5);

	inline int getTargetsNumber(){ return m_stTargets.nTargetsNumber; }

	inline ShapeTargets getTargets(){ return m_stTargets; }

	void drawRes(Mat& dst);

private:

	Mat rotateImg(Mat src, double angle, bool isOriginalSize = true);

	Mat getEdgeImg(Mat src);

	bool calMaxSimilarity(Mat src, double& maxSimilarity, double& angle);

	double calSimilarity(Mat src1, Mat src2);

	bool clusterAnalyze(vector<pair<Point, double>>points, vector<pair<Point, double>>& peaks, int disThres = 5, int numberThres = 3);

	void drawRes(Mat& dst, int xIdx, int yIdx, double angle, double value);

	void ShapeMatcher2d::drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType);

private:
	
	double m_dAngleStart;
	double m_dAngleEnd;
	double m_dAngleStep;

	int m_iModelWidth;
	int m_iModelHeight;

	int m_iModelActualWidth;
	int m_iModelActualHeight;
	Mat m_mSrc;
	
	vector<pair<Mat, int>> m_mMutiAngleModel;  //

	ShapeTargets m_stTargets;
};

#endif
