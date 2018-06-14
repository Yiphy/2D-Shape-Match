#include"shapeMatcher.h"

void testMatch()
{
	Mat model = imread("model2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src = imread("src.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	string dstPath = "dst2.bmp";

	Mat dstMat; cvtColor(src, dstMat, CV_GRAY2BGR);
	
	clock_t ts, te;

	ts = clock();

	ShapeMatcher2d targetFinder;
	targetFinder.creatModel(model, 1, 0, 45);

	ts = clock();
	
	targetFinder.match(src, 0.7);
	
	te = clock();

	cout << te - ts << " ms" << endl;
	
	Mat dst;
	targetFinder.drawRes(dst);
	imwrite(dstPath.c_str(),dst);

	return;
}



void main()
{	

	testMatch();

	return;
}
