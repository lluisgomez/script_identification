// Include Opencv
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <limits>

#include "utils.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

  // KNN neighbours
  int k=1;
  // KdTree with 5 random trees
  flann::KDTreeIndexParams indexParams(5);

  const char *scripcrs[] = {"Lat", "Chi", "Kan", "Kor"};
  vector<string> scripts(scripcrs,scripcrs+4);

  cout << "Load data and Create the Index for Class 1... "; cout.flush();
  //Load data for Latin script
  Ptr<ml::TrainData> data_Latin = ml::TrainData::loadFromCSV(
                                    "Latin_features.csv",0,0);
  Mat feats_Latin = data_Latin->getTrainSamples();
  //cout << "Latin: Loaded " << feats_Latin.rows << " samples, " << feats_Latin.cols 
  //                                  << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Latin(feats_Latin, indexParams);
  // Save the index
  //kdtree_Latin.save("train/trained_kdtree_index_Latin.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 2... "; cout.flush();
  //Load data for Chinese script
  Ptr<ml::TrainData> data_Chinese = ml::TrainData::loadFromCSV(
                                   "Chinese_features.csv",0,0);
  Mat feats_Chinese = data_Chinese->getTrainSamples();
  //cout << "Chinese: Loaded " << feats_Chinese.rows << " samples, " << feats_Chinese.cols 
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Chinese(feats_Chinese, indexParams);
  // Save the index
  //kdtree_Chinese.save("train/trained_kdtree_index_Chinese.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 3... "; cout.flush();
  //Load data for Kannada script
  Ptr<ml::TrainData> data_Kannada = ml::TrainData::loadFromCSV(
                                    "Kannada_features.csv",0,0);
  Mat feats_Kannada = data_Kannada->getTrainSamples();
  //cout << "Kannada: Loaded " << feats_Kannada.rows << " samples, " << feats_Kannada.cols 
  //                                  << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Kannada(feats_Kannada, indexParams);
  // Save the index
  //kdtree_Kannada.save("train/trained_kdtree_index_Kannada.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 4... "; cout.flush();
  //Load data for Korean script
  Ptr<ml::TrainData> data_Korean = ml::TrainData::loadFromCSV(
                                   "Korean_features.csv",0,0);
  Mat feats_Korean = data_Korean->getTrainSamples();
  //cout << "Korean: Loaded " << feats_Korean.rows << " samples, " << feats_Korean.cols 
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Korean(feats_Korean, indexParams);
  // Save the index
  //kdtree_Korean.save("train/trained_kdtree_index_Korean.fln");
  cout << "done!" << endl; cout.flush();



  /////////////////////////////////////////////////////////////////////////////
  // Compute weights for each feature on each class
  Mat indices_1;
  Mat dists_1_2;
  Mat dists_1_3;
  Mat dists_1_4;

  kdtree_Chinese.knnSearch(feats_Latin, indices_1, dists_1_2, 10, flann::SearchParams(64));
  kdtree_Kannada.knnSearch(feats_Latin, indices_1, dists_1_3, 10, flann::SearchParams(64));
  kdtree_Korean.knnSearch(feats_Latin, indices_1, dists_1_4, 10, flann::SearchParams(64));

  reduce(dists_1_2, dists_1_2, -1, CV_REDUCE_AVG);
  reduce(dists_1_3, dists_1_3, -1, CV_REDUCE_AVG);
  reduce(dists_1_4, dists_1_4, -1, CV_REDUCE_AVG);

  Mat weights_Latin;
  add(dists_1_2,dists_1_3,weights_Latin);
  add(dists_1_4,weights_Latin,weights_Latin);
  weights_Latin = weights_Latin/3;

  kdtree_Latin.knnSearch(feats_Chinese, indices_1, dists_1_2, 10, flann::SearchParams(64));
  kdtree_Kannada.knnSearch(feats_Chinese, indices_1, dists_1_3, 10, flann::SearchParams(64));
  kdtree_Korean.knnSearch(feats_Chinese, indices_1, dists_1_4, 10, flann::SearchParams(64));

  reduce(dists_1_2, dists_1_2, -1, CV_REDUCE_AVG);
  reduce(dists_1_3, dists_1_3, -1, CV_REDUCE_AVG);
  reduce(dists_1_4, dists_1_4, -1, CV_REDUCE_AVG);

  Mat weights_Chinese;
  add(dists_1_2,dists_1_3,weights_Chinese);
  add(dists_1_4,weights_Chinese,weights_Chinese);
  weights_Chinese = weights_Chinese/3;

  kdtree_Chinese.knnSearch(feats_Kannada, indices_1, dists_1_2, 10, flann::SearchParams(64));
  kdtree_Latin.knnSearch(feats_Kannada, indices_1, dists_1_3, 10, flann::SearchParams(64));
  kdtree_Korean.knnSearch(feats_Kannada, indices_1, dists_1_4, 10, flann::SearchParams(64));

  reduce(dists_1_2, dists_1_2, -1, CV_REDUCE_AVG);
  reduce(dists_1_3, dists_1_3, -1, CV_REDUCE_AVG);
  reduce(dists_1_4, dists_1_4, -1, CV_REDUCE_AVG);

  Mat weights_Kannada;
  add(dists_1_2,dists_1_3,weights_Kannada);
  add(dists_1_4,weights_Kannada,weights_Kannada);
  weights_Kannada = weights_Kannada/3;

  kdtree_Chinese.knnSearch(feats_Korean, indices_1, dists_1_2, 10, flann::SearchParams(64));
  kdtree_Kannada.knnSearch(feats_Korean, indices_1, dists_1_3, 10, flann::SearchParams(64));
  kdtree_Latin.knnSearch(feats_Korean, indices_1, dists_1_4, 10, flann::SearchParams(64));

  reduce(dists_1_2, dists_1_2, -1, CV_REDUCE_AVG);
  reduce(dists_1_3, dists_1_3, -1, CV_REDUCE_AVG);
  reduce(dists_1_4, dists_1_4, -1, CV_REDUCE_AVG);

  Mat weights_Korean;
  add(dists_1_2,dists_1_3,weights_Korean);
  add(dists_1_4,weights_Korean,weights_Korean);
  weights_Korean = weights_Korean/3;

  double min_dist, max_dist, minVal, maxVal;
  minMaxLoc(weights_Latin, &minVal, &maxVal);
  min_dist = minVal; max_dist = maxVal;
  minMaxLoc(weights_Chinese, &minVal, &maxVal);
  min_dist = min(min_dist,minVal); max_dist = max(max_dist,maxVal);
  minMaxLoc(weights_Kannada, &minVal, &maxVal);
  min_dist = min(min_dist,minVal); max_dist = max(max_dist,maxVal);
  minMaxLoc(weights_Korean, &minVal, &maxVal);
  min_dist = min(min_dist,minVal); max_dist = max(max_dist,maxVal);

  weights_Latin = (weights_Latin - min_dist) / (max_dist - min_dist);
  weights_Chinese = (weights_Chinese - min_dist) / (max_dist - min_dist);
  weights_Kannada = (weights_Kannada - min_dist) / (max_dist - min_dist);
  weights_Korean = (weights_Korean - min_dist) / (max_dist - min_dist);

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////


  //If we have image(s) passed as an argument do to classification using NBNN
  if (argc>2)
  {
    //First extract features
    //Load filters bank and withenning params
    Mat filters, M, P;
    FileStorage fs("first_layer_filters.xml", FileStorage::READ);
    fs["D"] >> filters;
    fs["M"] >> M;
    fs["P"] >> P;
    fs.release();
  
    int src_height  = 64;
    int image_size  = 32;
    int quad_size   = 12;
    int patch_size  = 8;
    int num_quads   = 25; //extract 25 quads (12x12) from each image
    int num_tiles   = 25; //extract 25 patches (8x8) from each quad 
  
    double alpha    = 0.5; //used for feature representation: 
                           //scalar non-linear function z = max(0, |D*a| - alpha)
  
    Mat quad;
    Mat tmp;

    ofstream outfile;
    outfile.open (argv[argc-1]);

    for (int f=1; f<argc-1; f++)
    {
      cout << "Extracting features for image " << argv[f] << " ... "; cout.flush();
      Mat src  = imread(argv[f]);
      if(src.channels() != 3)
        return 0;
      cvtColor(src,src,COLOR_RGB2GRAY);
      int src_width = (src.cols*src_height)/src.rows;
      resize(src,src,Size(src_width,src_height));
  
      Mat query = Mat::zeros(0,1737,CV_64FC1);
  
      // Do sliding window from x=0 to src_width-image_size in three rows (top,middle,bottom)
      for (int y=0; y<=src_height-image_size; y=y+8)
      { 
        for (int x=0; x<=src_width-image_size; x=x+8)
        { 
  
          Mat img;
          src(Rect(x,y,image_size,image_size)).copyTo(img); // img must be 32x32 pixels
  
          vector< vector<double> > data_pool(9); 
          int quad_id = 1;
          for (int q_x=0; q_x<=image_size-quad_size; q_x=q_x+(quad_size/2-1))
          {
            for (int q_y=0; q_y<=image_size-quad_size; q_y=q_y+(quad_size/2-1))
            {
              Rect quad_rect = Rect(q_x,q_y,quad_size,quad_size); 
              img(quad_rect).copyTo(quad);
      
              //start sliding window (8x8) in each tile and store the patch as row in data_pool
              for (int w_x=0; w_x<=quad_size-patch_size; w_x++)
              {
                for (int w_y=0; w_y<=quad_size-patch_size; w_y++)
                {
                  quad(Rect(w_x,w_y,patch_size,patch_size)).copyTo(tmp);
                  tmp = tmp.reshape(0,1);
                  tmp.convertTo(tmp, CV_64F);
                  normalizeAndZCA(tmp,M,P);
                  vector<double> patch;
                  tmp.copyTo(patch);
                  if ((quad_id == 1)||(quad_id == 2)||(quad_id == 6)||(quad_id == 7))
                    data_pool[0].insert(data_pool[0].end(),patch.begin(),patch.end());
                  if ((quad_id == 2)||(quad_id == 7)||(quad_id == 3)||(quad_id == 8)||(quad_id == 4)||(quad_id == 9))
                    data_pool[1].insert(data_pool[1].end(),patch.begin(),patch.end());
                  if ((quad_id == 4)||(quad_id == 9)||(quad_id == 5)||(quad_id == 10))
                    data_pool[2].insert(data_pool[2].end(),patch.begin(),patch.end());
                  if ((quad_id == 6)||(quad_id == 11)||(quad_id == 16)||(quad_id == 7)||(quad_id == 12)||(quad_id == 17))
                    data_pool[3].insert(data_pool[3].end(),patch.begin(),patch.end());
                  if ((quad_id == 7)||(quad_id == 12)||(quad_id == 17)||(quad_id == 8)||(quad_id == 13)||(quad_id == 18)||(quad_id == 9)||(quad_id == 14)||(quad_id == 19))
                    data_pool[4].insert(data_pool[4].end(),patch.begin(),patch.end());
                  if ((quad_id == 9)||(quad_id == 14)||(quad_id == 19)||(quad_id == 10)||(quad_id == 15)||(quad_id == 20))
                    data_pool[5].insert(data_pool[5].end(),patch.begin(),patch.end());
                  if ((quad_id == 16)||(quad_id == 21)||(quad_id == 17)||(quad_id == 22))
                    data_pool[6].insert(data_pool[6].end(),patch.begin(),patch.end());
                  if ((quad_id == 17)||(quad_id == 22)||(quad_id == 18)||(quad_id == 23)||(quad_id == 19)||(quad_id == 24))
                    data_pool[7].insert(data_pool[7].end(),patch.begin(),patch.end());
                  if ((quad_id == 19)||(quad_id == 24)||(quad_id == 20)||(quad_id == 25))
                    data_pool[8].insert(data_pool[8].end(),patch.begin(),patch.end());
                }
              }
      
              quad_id++;
            }
          }
      
          //do dot product of each normalized and whitened patch 
          //each pool is averaged and this yields a representation of 9xD 
          Mat feature = Mat::zeros(9,filters.rows,CV_64FC1);
          for (int i=0; i<9; i++)
          {
            Mat pool = Mat(data_pool[i]);
            pool = pool.reshape(0,data_pool[i].size()/filters.cols);
            for (int p=0; p<pool.rows; p++)
            {
              for (int f=0; f<filters.rows; f++)
              {
                feature.row(i).at<double>(0,f) = feature.row(i).at<double>(0,f) + max(0.0,std::abs(pool.row(p).dot(filters.row(f)))-alpha);
              }
            }
          }
          feature = feature.reshape(0,1);
          query.push_back(feature);
  
        }
      }
      //cout << "Extracted " << query.rows << " samples, " << query.cols << "-D"<<endl;
   
      query.convertTo(query, feats_Latin.type());
      //cout << feats_Latin.type() << endl;
      //cout << query.type() << endl;
   

      vector<double> I2Cdistances(4,0);
   
      // Batch: Call knnSearch
      Mat indices;
      Mat dists;
     
      kdtree_Latin.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[0] += dists.at<float>(row,col) * 
                             (1 - weights_Latin.at<float>(indices.at<float>(row,col),0));
      //cout << "Image To Class (Latin) Distance:: "<< I2Cdistances[0] << endl;
     
      kdtree_Chinese.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[1] += dists.at<float>(row,col) *
                             (1 - weights_Chinese.at<float>(indices.at<float>(row,col),0));
      //cout << "Image To Class (Chinese) Distance:: "<< I2Cdistances[1] << endl;
     
      kdtree_Kannada.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[2] += dists.at<float>(row,col) *
                             (1 - weights_Kannada.at<float>(indices.at<float>(row,col),0));
      //cout << "Image To Class (Kannada) Distance:: "<< I2Cdistances[2] << endl;
     
      kdtree_Korean.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[3] += dists.at<float>(row,col) * 
                             (1 - weights_Korean.at<float>(indices.at<float>(row,col),0));
      //cout << "Image To Class (Korean) Distance:: "<< I2Cdistances[3] << endl;
     

      //Classify image
      double minVal,maxVal;
      Point minLoc,maxLoc;
      minMaxLoc(I2Cdistances, &minVal, &maxVal, &minLoc, &maxLoc);
      //cout << minLoc << endl;
      //cout << "Predicted Script: " << scripts[minLoc.x] << endl;
      //cout << minLoc.x << endl;
      //outfile << argv[f] << "|" << scripts[minLoc.x] << endl;
      outfile << argv[f] << " " << minLoc.x << endl;
     

      cout << "done!" << endl; cout.flush();
    } // end foreach input image

  } //fi if argc>1

  return 0;
}
