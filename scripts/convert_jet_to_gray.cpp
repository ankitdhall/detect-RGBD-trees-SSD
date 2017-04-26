#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
using namespace cv;

int main()
{
	string line, line_orig;
	Mat img;

	Mat img_rgb, img_d, fin_img;
	ifstream list_file("VOC2007_disp_filelist.txt");
	ifstream list_file_orig("JPEGImages_VOC2007.txt");
	if(list_file.is_open())
	{
		while(getline(list_file, line))
		{
			/*img = imread(line, 0);
			bitwise_not(img, img);
			imwrite(line, img);*/
			cout << line << endl;
			

			getline(list_file_orig, line_orig);

			//assert(line.compare(line_orig) == 0);

			img_rgb = imread(line_orig);
			img_d = imread(line, CV_LOAD_IMAGE_GRAYSCALE);
			vector<Mat> channels;
			Mat bgr[3];
			split(img_rgb, bgr);

			channels.push_back(img_d);
			channels.push_back(bgr[0]);
			channels.push_back(bgr[1]);
			channels.push_back(bgr[2]);
	
	
			merge(channels, fin_img);
			/*imshow("img", fin_img);
			waitKey(0);*/
			//Mat abgr[4];
			//split(fin_img, abgr);
			//imshow("disparity", abgr[0]);
			imwrite(line_orig.substr(0, line_orig.size()-3) + "png", fin_img);
			//waitKey(0);
		}
	}
	list_file.close();
	
	return 0;
}
