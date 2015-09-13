// Include Opencv
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

#include "utils.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

  // KNN neighbours
  int k=1;
  // KdTree with 5 random trees
  flann::KDTreeIndexParams indexParams(5);

  const char *scripcrs[] = {"Eng", "Hin", "Ben", "Ori", "Guj", "Pun", 
                            "Kan", "Tam", "Tel", "Arb"};
  vector<string> scripts(scripcrs,scripcrs+10);

  cout << "Load data and Create the Index for Class 1... "; cout.flush();
  //Load data for English script
  Ptr<ml::TrainData> data_English = ml::TrainData::loadFromCSV(
                                    "English_features.csv",0,0);
  Mat feats_English = data_English->getTrainSamples();
  //cout << "English: Loaded " << feats_English.rows << " samples, " << feats_English.cols  cout.flush();
  //                                  << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_English(feats_English, indexParams);
  // Save the index
  //kdtree_English.save("train/trained_kdtree_index_English.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 2... "; cout.flush();
  //Load data for Hindi script
  Ptr<ml::TrainData> data_Hindi = ml::TrainData::loadFromCSV(
                                   "Hindi_features.csv",0,0);
  Mat feats_Hindi = data_Hindi->getTrainSamples();
  //cout << "Hindi: Loaded " << feats_Hindi.rows << " samples, " << feats_Hindi.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Hindi(feats_Hindi, indexParams);
  // Save the index
  //kdtree_Hindi.save("train/trained_kdtree_index_Hindi.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 3... "; cout.flush();
  //Load data for Bengali script
  Ptr<ml::TrainData> data_Bengali = ml::TrainData::loadFromCSV(
                                    "Bengali_features.csv",0,0);
  Mat feats_Bengali = data_Bengali->getTrainSamples();
  //cout << "Bengali: Loaded " << feats_Bengali.rows << " samples, " << feats_Bengali.cols  cout.flush();
  //                                  << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Bengali(feats_Bengali, indexParams);
  // Save the index
  //kdtree_Bengali.save("train/trained_kdtree_index_Bengali.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 4... "; cout.flush();
  //Load data for Oriya script
  Ptr<ml::TrainData> data_Oriya = ml::TrainData::loadFromCSV(
                                   "Oriya_features.csv",0,0);
  Mat feats_Oriya = data_Oriya->getTrainSamples();
  //cout << "Oriya: Loaded " << feats_Oriya.rows << " samples, " << feats_Oriya.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Oriya(feats_Oriya, indexParams);
  // Save the index
  //kdtree_Oriya.save("train/trained_kdtree_index_Oriya.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 5... "; cout.flush();
  //Load data for Gujrathi script
  Ptr<ml::TrainData> data_Gujrathi = ml::TrainData::loadFromCSV(
                                   "Gujrathi_features.csv",0,0);
  Mat feats_Gujrathi = data_Gujrathi->getTrainSamples();
  //cout << "Gujrathi: Loaded " << feats_Gujrathi.rows << " samples, " << feats_Gujrathi.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Gujrathi(feats_Gujrathi, indexParams);
  // Save the index
  //kdtree_Gujrathi.save("train/trained_kdtree_index_Gujrathi.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 6... "; cout.flush();
  //Load data for Punjabi script
  Ptr<ml::TrainData> data_Punjabi = ml::TrainData::loadFromCSV(
                                   "Punjabi_features.csv",0,0);
  Mat feats_Punjabi = data_Punjabi->getTrainSamples();
  //cout << "Punjabi: Loaded " << feats_Punjabi.rows << " samples, " << feats_Punjabi.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Punjabi(feats_Punjabi, indexParams);
  // Save the index
  //kdtree_Punjabi.save("train/trained_kdtree_index_Punjabi.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 7... "; cout.flush();
  //Load data for Kannada script
  Ptr<ml::TrainData> data_Kannada = ml::TrainData::loadFromCSV(
                                   "Kannada_features.csv",0,0);
  Mat feats_Kannada = data_Kannada->getTrainSamples();
  //cout << "Kannada: Loaded " << feats_Kannada.rows << " samples, " << feats_Kannada.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Kannada(feats_Kannada, indexParams);
  // Save the index
  //kdtree_Kannada.save("train/trained_kdtree_index_Kannada.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 8... "; cout.flush();
  //Load data for Tamil script
  Ptr<ml::TrainData> data_Tamil = ml::TrainData::loadFromCSV(
                                   "Tamil_features.csv",0,0);
  Mat feats_Tamil = data_Tamil->getTrainSamples();
  //cout << "Tamil: Loaded " << feats_Tamil.rows << " samples, " << feats_Tamil.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Tamil(feats_Tamil, indexParams);
  // Save the index
  //kdtree_Tamil.save("train/trained_kdtree_index_Tamil.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 9... "; cout.flush();
  //Load data for Telegu script
  Ptr<ml::TrainData> data_Telegu = ml::TrainData::loadFromCSV(
                                   "Telegu_features.csv",0,0);
  Mat feats_Telegu = data_Telegu->getTrainSamples();
  //cout << "Telegu: Loaded " << feats_Telegu.rows << " samples, " << feats_Telegu.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Telegu(feats_Telegu, indexParams);
  // Save the index
  //kdtree_Telegu.save("train/trained_kdtree_index_Telegu.fln");
  cout << "done!" << endl; cout.flush();


  cout << "Load data and Create the Index for Class 10... "; cout.flush();
  //Load data for Arabic script
  Ptr<ml::TrainData> data_Arabic = ml::TrainData::loadFromCSV(
                                   "Arabic_features.csv",0,0);
  Mat feats_Arabic = data_Arabic->getTrainSamples();
  //cout << "Arabic: Loaded " << feats_Arabic.rows << " samples, " << feats_Arabic.cols  cout.flush();
  //                                 << "-D"<<endl;
  // Create the Index
  flann::Index kdtree_Arabic(feats_Arabic, indexParams);
  // Save the index
  //kdtree_Arabic.save("train/trained_kdtree_index_Arabic.fln");
  cout << "done!" << endl; cout.flush();






  //If we have image(s) passed as an argument do to classification using NBNN
  if (argc>2)
  {
    //First extract features
    //Load filters bank and withenning params
    Mat filters, M, P;
    FileStorage fs("first_layer_centroids.xml", FileStorage::READ);
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
  
      Mat query = Mat::zeros(0,9*filters.rows,CV_64FC1);
  
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
      //cout << "Extracted " << query.rows << " samples, " << query.cols << "-D"<<endl; cout.flush();
   
      query.convertTo(query, feats_English.type());
      //cout << feats_English.type() << endl; cout.flush();
      //cout << query.type() << endl; cout.flush();
   

      vector<double> I2Cdistances(10,0);
   
      // Batch: Call knnSearch
      Mat indices;
      Mat dists;
     
      kdtree_English.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[0] += dists.at<float>(row,col);
      //cout << "Image To Class (English) Distance:: "<< I2Cdistances[0] << endl; cout.flush();
     
      kdtree_Hindi.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[1] += dists.at<float>(row,col);
      //cout << "Image To Class (Hindi) Distance:: "<< I2Cdistances[1] << endl; cout.flush();
     
      kdtree_Bengali.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[2] += dists.at<float>(row,col);
      //cout << "Image To Class (Bengali) Distance:: "<< I2Cdistances[2] << endl; cout.flush();
     
      kdtree_Oriya.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[3] += dists.at<float>(row,col);
      //cout << "Image To Class (Oriya) Distance:: "<< I2Cdistances[3] << endl; cout.flush();
     
      kdtree_Gujrathi.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[4] += dists.at<float>(row,col);
      //cout << "Image To Class (Gujrathi) Distance:: "<< I2Cdistances[4] << endl; cout.flush();
     
      kdtree_Punjabi.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[5] += dists.at<float>(row,col);
      //cout << "Image To Class (Punjabi) Distance:: "<< I2Cdistances[5] << endl; cout.flush();
     
      kdtree_Kannada.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[6] += dists.at<float>(row,col);
      //cout << "Image To Class (Kannada) Distance:: "<< I2Cdistances[6] << endl; cout.flush();
     
      kdtree_Tamil.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[7] += dists.at<float>(row,col);
      //cout << "Image To Class (Tamil) Distance:: "<< I2Cdistances[7] << endl; cout.flush();
     
      kdtree_Telegu.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[8] += dists.at<float>(row,col);
      //cout << "Image To Class (Telegu) Distance:: "<< I2Cdistances[8] << endl; cout.flush();
     
      kdtree_Arabic.knnSearch(query, indices, dists, k, flann::SearchParams(64));
      for(int row = 0 ; row < indices.rows ; row++)
        for(int col = 0 ; col < indices.cols ; col++)
          I2Cdistances[9] += dists.at<float>(row,col);
      //cout << "Image To Class (Arabic) Distance:: "<< I2Cdistances[9] << endl; cout.flush();

      //Classify image
      double minVal,maxVal;
      Point minLoc,maxLoc;
      minMaxLoc(I2Cdistances, &minVal, &maxVal, &minLoc, &maxLoc);
      //cout << minLoc << endl; cout.flush();
      //cout << "Predicted Script: " << scripts[minLoc.x] << endl; cout.flush();
      //cout << minLoc.x << endl; cout.flush();
      outfile << argv[f] << "|" << scripts[minLoc.x] << endl;
     

      cout << "done!" << endl; cout.flush();
    } // end foreach input image

  } //fi if argc>1

  return 0;
}
