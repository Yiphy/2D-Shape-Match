
/********************************************************************************************
# 2D - Shape - Match
# shape matching for translation and rotation cases(Zoom cases is not included by now)
# Zhang Yifei(yiphyzhang@126.com)

1. Introduction
This program implements the 2d shapes matching algothrim for translation
and rotation cases, based on OpenCV.Please sent questions, comments or
bugs to Zhang Yifei(yiphyzhang@126.com)

This software was developed / tested on Windows7 & visualStudio 2013.

2. License & Disclaimer
Copyright @ 2018   Zhang Yifei(yiphyzhang@126.com)
This software may be used for research purposes only.

3. Build and Run
(1) Install OpenCV2.4.XX
(2) Compile this project with visual Studio 2013.
********************************************************************************************/

#ifndef SHAPE_MATCHER_2D_H_
#define SHAPE_MATCHER_2D_H_

#include"headers.h"

#define MAX_DETECT_NUM 100

/**
* Brief: target information
* @param[0/1]: x,y: coordinate of target in 2d image
* @param[2]: angle: rotation angle of the target in counterclockwise.
* @param[3]: similarity: similarity of current coordinate and rotation angle.
* Details:
*/
typedef struct _Target_Info
{
	double x;
	double y;
	double angle;
	double similarity;
}TargetInfo;

/**
* Brief: target information
* @param[0]: nTargetsNumber: number of current targets.
* @param[1]: tarInfo[MAX_DETECT_NUM]: detailed information of each target. MAX_DETECT_NUM is default set to be 100.
* Details:
*/
typedef struct _Shape_Targets
{
	int nTargetsNumber;
	TargetInfo tarInfo[MAX_DETECT_NUM];
}ShapeTargets;


/**
* Brief: target location and rotation detection for 2d images.
  
  This class may be used as below:

     0. setEdgeParam();
     1. creatModel();
     2. match(); or matchOverlap();  
     3. getTargets(); or drawRes();
  
* Details:
*/
class ShapeMatcher2d
{
public:
	ShapeMatcher2d();

	/**
	* Brief: default parameters to extract edges.
	* @param[in] edgeParam0: bilateralFilter blur parameter, default set to be 15
	* @param[in] edgeParam1: canny edge parameter, default set to be 30
	* @param[in] edgeParam2: canny edge parameter, default set to be 50
	* @param[in] edgeParam3: dilate parameter in matching module, default set to be 5
	* Details: default parameters to extract edges of model and target image.
	*/
	ShapeMatcher2d(int edgeParam0, int edgeParam1, int edgeParam2, int edgeParam3)
	{
		m_iEdgeParam0 = edgeParam0;
		m_iEdgeParam1 = edgeParam1;
		m_iEdgeParam2 = edgeParam2;
		m_iEdgeParam3 = edgeParam3;
	}

	~ShapeMatcher2d();
	
	/**
	* Brief: default parameters to extract edges.
	* @param[in] edgeParam0: bilateralFilter blur parameter, default set to be 15
	* @param[in] edgeParam1: canny edge parameter, default set to be 30
	* @param[in] edgeParam2: canny edge parameter, default set to be 50
	* @param[in] edgeParam3: dilate parameter in matching module, default set to be 5
	* Details: default parameters to extract edges of model and target image.
	*/
	inline void setEdgeParam(int edgeParam0, int edgeParam1, int edgeParam2, int edgeParam3)
	{
		m_iEdgeParam0 = edgeParam0;
		m_iEdgeParam1 = edgeParam1;
		m_iEdgeParam2 = edgeParam2;
		m_iEdgeParam3 = edgeParam3;
	}

	/**
	* Brief: default parameters to creat models.
	* @param[in] src: original image to creat models.
	* @param[in] angleStep: step of angle, from start angle to end angle
	* @param[in] angleStart: start angle
	* @param[in] angleEnd: end angle
	* Details: angleStart must be smaller than angleEnd. But both paramters can be smaller than 0 or bigger than 360.
	*           For example, (angleStart = -10 , angleEnd = 10) is OK. It's same as (angleStart = 350 , angleEnd = 370)
	*/
	bool creatModel(Mat src, double angleStep, double angleStart = 0, double angleEnd = 360);

	/**
	* Brief: match models within the dst image. Overlap cases are applicable.
	* @param[in] src: source images. CV_8UC1
	* @param[in] similarityThres: similarity biggers than this threshold will be recognized as target.
	* Details: This is recommended in overlap cases. But this may be sensitive to noises.
	*/
	bool matchOverlap(Mat src, double similarityThres = 0.5);

	/**
	* Brief: match models within the dst image. Overlap cases may not be applicable.
	* @param[in] src: source images. CV_8UC1
	* @param[in] similarityThres: similarity biggers than this threshold will be recognized as target.
	* Details: This is recommended in most cases.
	*/
	bool match(Mat src, double similarityThres = 0.5);
	
	/**
	* Brief: get targets number.
	* @param[out] return value.
	* Details:
	*/
	inline int getTargetsNumber(){ return m_stTargets.nTargetsNumber; }
	
	/**
	* Brief: get detection targets.
	* @param[out] return value.
	* Details:
	*/
	inline ShapeTargets getTargets(){ return m_stTargets; }
	
	/**
	* Brief: draw detection results on destination image.
	* @param[in&out] dst: destination images. will be translated into color SOURCE image. CV_8UC3
	* Details:
	*/
	void drawRes(Mat& dst);

private:
	/**
	* Brief: rotate image in counterclockwise
	* Details: if src.rows==src.cols, isOriginalSize = true; else isOriginalSize = false.
	*/
	Mat rotateImg(Mat src, double angle, bool isOriginalSize = true);

	/**
	* Brief: only reserve non-zero pixel of src. 
	* Details: cut pixels of zero intensity. only keep non-zero pixels of src and saved in dst. all CV_8UC1
	*/
	void cutMinImg(Mat src, Mat& dst);

	/**
	* Brief: get edge image of src.
	* Details: bilateralFilter -> canny -> threshold -> Exaggerate to diagonal size
	*/
	Mat getEdgeImg(Mat src);

	/**
	* Brief: cluster candidate targets to get peaks point.
	* Details:
	*/
	bool clusterAnalyze(vector<pair<Point, double>>points, vector<pair<Point, double>>& peaks, int disThres = 5, int numberThres = 1);

	void drawRes(Mat& dst, int xIdx, int yIdx, double angle, double value);

	void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType);

	/**
	* Brief: set nTatgetNumber to be 0.
	* Details:
	*/
	void initialMatchedTargets();

	/**
	* Brief shrink the angle to [0,360)
	* @param[in ]: angle
	* @param[out]: return angle
	* Details: if input 1.2  , return 1.2; if input -1.2 , return 358.8;  if input 365.2, return 5.2; if input 723.4, return 3.4.
	*/
	double shrinkAngle(double angle);

private:

	int m_iEdgeParam0;
	int m_iEdgeParam1;
	int m_iEdgeParam2;
	int m_iEdgeParam3;

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
