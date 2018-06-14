#ifndef SHAPE_MATCHER_2D_CPP_
#define SHAPE_MATCHER_2D_CPP_

#include"shapeMatcher.h"

ShapeMatcher2d::ShapeMatcher2d()
{
	m_stTargets.nTargetsNumber = 0;
	for (int i = 0; i < MAX_DETECT_NUM; i++)
	{
		m_stTargets.tarInfo[i].x = 0;
		m_stTargets.tarInfo[i].y = 0;
		m_stTargets.tarInfo[i].similarity = 0;
		m_stTargets.tarInfo[i].angle = 0;
	}
}

ShapeMatcher2d::~ShapeMatcher2d()
{
	//delete[] m_stTargets.tarInfo;
}

Mat ShapeMatcher2d::rotateImg(Mat src, double degree, bool isOriginalSize)
{
	if (isOriginalSize)
	{
		CvPoint2D32f center;
		center.x = float(src.rows/ 2.0 + 0.5);
		center.y = float(src.cols / 2.0 + 0.5);

		Mat tranMatrix = getRotationMatrix2D(center, degree, 1);
		
		Mat src_rotate = Mat::zeros(src.size(), src.type());
		warpAffine(src, src_rotate, tranMatrix, src.size(), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		
		return src_rotate;
	}
	else
	{
		double angle = degree  * CV_PI / 180.; // 弧度    
		double a = sin(angle), b = cos(angle);
		int width = src.cols;
		int height = src.rows;
		int width_rotate = int(height * fabs(a) + width * fabs(b));
		int height_rotate = int(width * fabs(a) + height * fabs(b));

		//旋转数组map  
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]  
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]  
		// 旋转中心  
		float map[6];
		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
		Mat tranMatrix = getRotationMatrix2D(center, degree, 1);
		map[2] += (width_rotate - width) / 2;
		map[5] += (height_rotate - height) / 2;

		Mat src_rotate = Mat::zeros(width_rotate, height_rotate, src.type());
		warpAffine(src, src_rotate, tranMatrix, Size(width_rotate, height_rotate), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		return src_rotate;
	}
}

Mat ShapeMatcher2d::getEdgeImg(Mat src)
{
	Mat edgeMat = Mat::zeros(src.cols, src.rows, CV_8UC1);
	
	int edgeParam0 = 30, edgeParam1 = 80;
	Canny(src, edgeMat, edgeParam0, edgeParam1, 3);
	threshold(edgeMat, edgeMat, 10, 255, CV_THRESH_BINARY);
	
	int minX(edgeMat.cols), maxX(0);
	int minY(edgeMat.rows), maxY(0);
	for (int i = 0; i < edgeMat.rows; i++)
	{
		for (int j = 0; j < edgeMat.cols; j++)
		{
			if (edgeMat.at<uchar>(i, j) == 255)
			{
				minX = j < minX ? j : minX;
				maxX = j > maxX ? j : maxX;
				minY = i < minY ? i : minY;
				maxY = i > maxY ? i : maxY;
			}
		}
	}
	
	int edgeWidth = (int)(sqrt((maxX - minX + 1)*(maxX - minX + 1) + (maxY - minY + 1)*(maxY - minY + 1)));
	int edgeHeight = (int)(sqrt((maxX - minX + 1)*(maxX - minX + 1) + (maxY - minY + 1)*(maxY - minY + 1)));

	Rect roi(minX, minY, maxX - minX + 1, maxY - minY + 1);
	
	Mat edge = Mat::zeros(edgeWidth, edgeHeight, CV_8UC1);
	Rect roiEdge((edgeWidth - maxX + minX - 1) / 2, (edgeHeight - maxY + minY - 1) / 2, maxX - minX + 1, maxY - minY + 1);

	edgeMat(roi).copyTo(edge(roiEdge));	

	m_iModelWidth = edgeWidth;
	m_iModelHeight = edgeHeight;

	m_iModelActualWidth = maxX - minX + 1;
	m_iModelActualHeight = maxY - minY + 1;

	return edge;
}

bool ShapeMatcher2d::creatModel(Mat tmpSrc, double angleStep, double angleStart, double angleEnd)
{
	if (tmpSrc.empty() || tmpSrc.type() != CV_8UC1)
		return false;

	if (angleStart < 0 || angleEnd >= 360 || angleEnd <= angleStart)
		return false;

	if (angleStep < 0.01)
		return false;

	Mat edgeMat = getEdgeImg(tmpSrc);

	m_dAngleStart = angleStart;
	m_dAngleEnd = angleEnd;
	m_dAngleStep = angleStep;

	for (double angle = angleStart; angle < angleEnd; angle += angleStep)
	{
		Mat modelRotate = rotateImg(edgeMat, angle, true);
		threshold(modelRotate, modelRotate, 10, 1, CV_THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		//dilate(modelRotate, modelRotate, element);
		int npixel = countNonZero(modelRotate);
		m_mMutiAngleModel.push_back(make_pair(modelRotate, npixel));
	}		

	return true;
}

double ShapeMatcher2d::calSimilarity(Mat model, Mat src)
{
	if (src.size() != model.size())
		return 0;

	if (src.type() != CV_8UC1 || model.type() != CV_8UC1)
		return 0;

	Mat andMat;
	bitwise_and(model, src, andMat);

	int modelPixel = countNonZero(model);
	int andPixel = countNonZero(andMat);

	return 1.0*andPixel / modelPixel;

}

bool ShapeMatcher2d::calMaxSimilarity(Mat src, double& maxSimilarity, double& angle)
{
	if (src.empty())
		return false;

	maxSimilarity = 0;
	angle = -1;

	for (size_t i = 0; i < m_mMutiAngleModel.size(); i++)
	{
		Mat model = m_mMutiAngleModel[i].first;

		double curSimilarity = calSimilarity(model, src);

		if (curSimilarity>maxSimilarity)
		{
			maxSimilarity = curSimilarity;
			angle = m_mMutiAngleModel[i].second;
		}
	}

	return true;
}

bool compare_pair(pair<Point, double> p1, pair<Point, double> p2)
{
	return p1.second > p2.second;
}

bool ShapeMatcher2d::clusterAnalyze(vector<pair<Point, double>>points, vector<pair<Point, double>>& peaks, int disThres, int numberThres)
{
	if ((int)points.size() < numberThres)
		return false;

	//cluster of the graspPoints
	vector<int> labels;
	int th2 = disThres* disThres;
	int n_labels = cv::partition(points, labels, [th2](const pair<Point, double>& lhs, const pair<Point, double>& rhs)
	{
		return ((lhs.first.x - rhs.first.x)*(lhs.first.x - rhs.first.x) + (lhs.first.y - rhs.first.y)*(lhs.first.y - rhs.first.y)) < th2;
	});


	peaks.resize(n_labels);
	for (int i = 0; i < n_labels; i++)
	{
		peaks[i].first = Point(-1, -1);
		peaks[i].second = -1;
	}

	for (size_t i = 0; i < points.size(); i++)
	{
		Point cpoint = points[i].first;
		double cvalue = points[i].second;

		if (cvalue>peaks[labels[i]].second)
		{
			peaks[labels[i]].second = cvalue;
			peaks[labels[i]].first = cpoint;
		}		
	}
	
	sort(peaks.begin(), peaks.end(), compare_pair);

	return true;
}

void ShapeMatcher2d::drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面） 

	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置）

	arrow.x = (int)(pEnd.x + len * cos(angle + PI * alpha / 180));
	arrow.y = (int)(pEnd.y + len * sin(angle + PI * alpha / 180));
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = (int)(pEnd.x + len * cos(angle - PI * alpha / 180));
	arrow.y = (int)(pEnd.y + len * sin(angle - PI * alpha / 180));
	line(img, pEnd, arrow, color, thickness, lineType);
}

void ShapeMatcher2d::drawRes(Mat&dst, int x, int y, double angle, double value)
{
	if (dst.empty() || dst.type() != CV_8UC3)
		return;

	if (x < 0 || x >= dst.cols || y < 0 || y >= dst.rows)
		return;

	int cH = m_iModelActualHeight;
	int cW = m_iModelActualWidth;

	//DrawResults of recognition	
	int arrowLen = 20, arrowAngle = 30, arrowThick = 1;
	int fontStyle = FONT_HERSHEY_COMPLEX;// FONT_HERSHEY_TRIPLEX;
	double fontScale = 0.6;
	int fontThick = 1;
	
	double ratio = 1.5;
	Point ps, pe;
	pe.x = x - cH*sin(CV_PI*angle / 180) / 2 * ratio;
	pe.y = y - cH*cos(CV_PI*angle / 180) / 2 * ratio;
	ps.x = x;// +cH*sin(CV_PI*angle / 180) / 2 * ratio;
	ps.y = y;// +cH*cos(CV_PI*angle / 180) / 2 * ratio;
	drawArrow(dst, ps, pe, arrowLen, arrowAngle, Scalar(0, 0, 255), 1, 8);
	
	stringstream angleStrS;	angleStrS << angle;
	stringstream valueStrS;	valueStrS << value;

	string angleStr = angleStrS.str(); 
	//angleStr = "A:" + angleStr;
	string valueStr = valueStrS.str();
	valueStr = valueStr.substr(0, valueStr.find_first_of('.') + 4);
	////valueStr = "V:" + valueStr;

	//cv::putText(dst, angleStr, ps, fontStyle, fontScale, Scalar(0, 0, 255), fontThick, 8, false);
	cv::putText(dst, valueStr, Point(x, y), fontStyle, fontScale, Scalar(0, 0, 255), fontThick, 8, false);
	circle(dst, Point(x, y), 3, Scalar(0, 0, 255), -1, 8);

	Point p0, p1, p2, p3;
	p0.x = x - cH*sin(CV_PI*angle / 180) / 2 - cW*cos(CV_PI*angle / 180) / 2;
	p0.y = y - cH*cos(CV_PI*angle / 180) / 2 + cW*sin(CV_PI*angle / 180) / 2;

	p1.x = x - cH*sin(CV_PI*angle / 180) / 2 + cW*cos(CV_PI*angle / 180) / 2;
	p1.y = y - cH*cos(CV_PI*angle / 180) / 2 - cW*sin(CV_PI*angle / 180) / 2;

	p2.x = x + cH*sin(CV_PI*angle / 180) / 2 + cW*cos(CV_PI*angle / 180) / 2;
	p2.y = y + cH*cos(CV_PI*angle / 180) / 2 - cW*sin(CV_PI*angle / 180) / 2;
	
	p3.x = x + cH*sin(CV_PI*angle / 180) / 2 - cW*cos(CV_PI*angle / 180) / 2;
	p3.y = y + cH*cos(CV_PI*angle / 180) / 2 + cW*sin(CV_PI*angle / 180) / 2;

	line(dst, p0, p1, Scalar(0, 255, 0), 1, 8, 0);
	line(dst, p1, p2, Scalar(0, 255, 0), 1, 8, 0);
	line(dst, p2, p3, Scalar(0, 255, 0), 1, 8, 0);
	line(dst, p3, p0, Scalar(0, 255, 0), 1, 8, 0);

}

void ShapeMatcher2d::drawRes(Mat& dst)
{
	cvtColor(m_mSrc, dst, CV_GRAY2BGR);

	int ntargets = m_stTargets.nTargetsNumber;
	for (int i = 0; i < ntargets; i++)
	{
		double angle = m_stTargets.tarInfo[i].angle;
		int xIdx = m_stTargets.tarInfo[i].x;
		int yIdx = m_stTargets.tarInfo[i].y;
		double similarity = m_stTargets.tarInfo[i].similarity;

		drawRes(dst, xIdx, yIdx, angle, similarity);

	}

}

bool ShapeMatcher2d::match(Mat dst, double similarityThres)
{
	if (dst.empty() || dst.type() != CV_8UC1)
		return false;

	m_mSrc = dst.clone();

	Mat edgeMat = dst.clone();
	int edgeParam0 = 11, edgeParam1 = 50;
	Canny(edgeMat, edgeMat, edgeParam0, edgeParam1, 3);
	
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
	dilate(edgeMat, edgeMat, element);
	threshold(edgeMat, edgeMat, 10, 1, CV_THRESH_BINARY);

	int resH = dst.rows - m_iModelHeight + 1;
	int resW = dst.cols - m_iModelWidth + 1;

	Mat maxSimilarityAngle = Mat::zeros(resH, resW, CV_32FC2);
	Mat maxSimilarity = Mat::zeros(resH, resW, CV_32FC1);
	for (size_t i = 0; i < m_mMutiAngleModel.size(); i++)
	{
		Mat model = m_mMutiAngleModel[i].first.clone();
		int npixel = m_mMutiAngleModel[i].second;
		double angle = m_dAngleStart + i*m_dAngleStep;

		Mat resMatch;
		matchTemplate(edgeMat, model, resMatch, CV_TM_CCORR);
		
		for (int row = 0; row < resMatch.rows; row++)
		{
			for (int col = 0; col < resMatch.cols; col++)
			{
				if (resMatch.at<float>(row, col) / npixel > maxSimilarityAngle.at<Vec2f>(row, col)[0])
				{
					maxSimilarityAngle.at<Vec2f>(row, col)[0] = resMatch.at<float>(row, col) / npixel;
					maxSimilarityAngle.at<Vec2f>(row, col)[1] = (float)angle;
					maxSimilarity.at<float>(row, col) = resMatch.at<float>(row, col) / npixel;
				}
			}
		}
	}

	vector<pair<Point,double>> candidatePoints;
	for (int row = 0; row < maxSimilarityAngle.rows; row++)
	{
		for (int col = 0; col < maxSimilarityAngle.cols; col++)
		{
			double cSimilarity = maxSimilarityAngle.at<Vec2f>(row, col)[0];
			if (cSimilarity>similarityThres)
				candidatePoints.push_back(make_pair(Point(col, row), cSimilarity));
		}
	}

	vector<pair<Point, double>> peaks;
	clusterAnalyze(candidatePoints, peaks);

	for (size_t i = 0; i < peaks.size(); i++)
	{
		int cn = m_stTargets.nTargetsNumber;
		if (cn >= MAX_DETECT_NUM)
			break;

		Point cpoint = peaks[i].first;
		double csimilarity = peaks[i].second;
		double angle = maxSimilarityAngle.at<Vec2f>(cpoint.y, cpoint.x)[1];

		m_stTargets.nTargetsNumber++;
		m_stTargets.tarInfo[cn].x = cpoint.x + m_iModelWidth / 2;
		m_stTargets.tarInfo[cn].y = cpoint.y + m_iModelHeight / 2;
		m_stTargets.tarInfo[cn].angle = angle;
		m_stTargets.tarInfo[cn].similarity = csimilarity;
	}

	return true;
}

#endif
