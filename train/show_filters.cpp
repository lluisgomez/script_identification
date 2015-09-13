#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "../utils.h"

using namespace cv;
using namespace std;

/*This is part of the implementation of the paper "Text Detection and Character Recognition with
  Unsupervised Feature Learning" by A. Coates et al. in ICDAR2011*/


// Train first layer filters with kmeans
int main (int argc, char* argv[])
{



  Mat filters, M, P;
  FileStorage fs("first_layer_filters.xml", FileStorage::READ);
  fs["D"] >> filters;
  fs["M"] >> M;
  fs["P"] >> P;
  fs.release();


  /*Visualize the filter bank*/
  visualizeNatwork(filters);

}
