#include"shapeMatcher.h"

void testMatchMultiModel1()
{
	Mat model = imread("model1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src1 = imread("model1_src1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = imread("model1_src2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat dst1, dst2, dst3, dst4;
	string dstPath1 = "model1_dst1.bmp";
	string dstPath2 = "model1_dst2.bmp";
		
	int param0 = 15, param1 = 30, param2 = 50, param3 = 5;
	ShapeMatcher2d targetFinder; 
	targetFinder.setEdgeParam(param0, param1, param2, param3);

	cout << endl << "Model2" << endl;

	clock_t ts, te;
	cout << "Image 1...";	ts = clock();
	targetFinder.creatModel(model, 3, 0, 359);
	targetFinder.matchOverlap(src1, 0.7);
	targetFinder.drawRes(dst1);
	imwrite(dstPath1.c_str(), dst1);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 2...";	ts = clock();
	targetFinder.creatModel(model, 1, -10, 10);
	targetFinder.match(src2, 0.7);
	targetFinder.drawRes(dst2);
	imwrite(dstPath2.c_str(), dst2);
	te = clock();	cout << " in " << te - ts << " ms" << endl;
		
	return;
}

void testMatchMultiModel2()
{
	Mat model = imread("model2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src1 = imread("model2_src1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = imread("model2_src2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat dst1, dst2, dst3, dst4;
	string dstPath1 = "model2_dst1.bmp";
	string dstPath2 = "model2_dst2.bmp";

	int param0 = 15, param1 = 30, param2 = 50, param3 = 5;
	ShapeMatcher2d targetFinder;
	targetFinder.setEdgeParam(param0, param1, param2, param3);

	cout << endl << "Model2" << endl;

	clock_t ts, te;
	cout << "Image 1...";	ts = clock();
	targetFinder.creatModel(model, 1, 345, 359);
	targetFinder.matchOverlap(src1, 0.7);
	targetFinder.drawRes(dst1);
	imwrite(dstPath1.c_str(), dst1);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 2...";	ts = clock();
	targetFinder.creatModel(model, 1, 345, 359);
	targetFinder.matchOverlap(src2, 0.7);
	targetFinder.drawRes(dst2);
	imwrite(dstPath2.c_str(), dst2);
	te = clock();	cout << " in " << te - ts << " ms" << endl;
	
	return;
}

void testMatchMultiModel3()
{
	Mat model = imread("model3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src1 = imread("model3_src1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = imread("model3_src2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src3 = imread("model3_src3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src4 = imread("model3_src4.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src5 = imread("model3_src5.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src6 = imread("model3_src6.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat dst1, dst2, dst3, dst4, dst5, dst6;
	string dstPath1 = "model3_dst1.bmp";
	string dstPath2 = "model3_dst2.bmp";
	string dstPath3 = "model3_dst3.bmp";
	string dstPath4 = "model3_dst4.bmp";
	string dstPath5 = "model3_dst5.bmp";
	string dstPath6 = "model3_dst6.bmp";

	int param0 = 15, param1 = 30, param2 = 50, param3 = 3;
	ShapeMatcher2d targetFinder;
	targetFinder.setEdgeParam(param0, param1, param2, param3);

	cout << endl << "Model3" << endl;

	clock_t ts, te;
	cout << "Image 1...";	ts = clock();
	targetFinder.creatModel(model, 1, 0, 10);
	targetFinder.match(src1, 0.7);
	targetFinder.drawRes(dst1);
	imwrite(dstPath1.c_str(), dst1);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 2...";	ts = clock();
	targetFinder.creatModel(model, 1, 0, 30);
	targetFinder.match(src2, 0.7);
	targetFinder.drawRes(dst2);
	imwrite(dstPath2.c_str(), dst2);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 3...";	ts = clock();
	targetFinder.creatModel(model, 1, 350, 360);
	targetFinder.match(src3, 0.7);
	targetFinder.drawRes(dst3);
	imwrite(dstPath3.c_str(), dst3);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 4...";	ts = clock();
	targetFinder.creatModel(model, 1, 270, 280);
	targetFinder.match(src4, 0.7);
	targetFinder.drawRes(dst4);
	imwrite(dstPath4.c_str(), dst4);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 5...";	ts = clock();
	targetFinder.creatModel(model, 1, 90, 120);
	targetFinder.match(src5, 0.7);
	targetFinder.drawRes(dst5);
	imwrite(dstPath5.c_str(), dst5);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	cout << "Image 6...";	ts = clock();
	targetFinder.creatModel(model, 1, 180, 210);
	targetFinder.match(src6, 0.7);
	targetFinder.drawRes(dst6);
	imwrite(dstPath6.c_str(), dst6);
	te = clock();	cout << " in " << te - ts << " ms" << endl;

	return;
}

void main()
{
	testMatchMultiModel1();

	testMatchMultiModel2();

	testMatchMultiModel3();

	return;
}
