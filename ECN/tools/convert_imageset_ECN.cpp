// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/lexical_cast.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"


#define NUM_CHANNELS 10 //number of patches grouped as single image

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::vector<std::string>, int> > lines;
  std::string filename_1;
  std::string filename_2;
  std::string filename_3;
  std::string filename_4;
  std::string filename_5;
  std::string filename_6;
  std::string filename_7;
  std::string filename_8;
  std::string filename_9;
  std::string filename_10;
  int label;
  while (infile >> filename_1 >> filename_2 >> filename_3 >> filename_4 >> filename_5 >> filename_6 >> filename_7 >> filename_8 >> filename_9 >> filename_10 >> label) {
    std::vector<std::string> patches_filenames;
    patches_filenames.push_back(filename_1);
    patches_filenames.push_back(filename_2);
    patches_filenames.push_back(filename_3);
    patches_filenames.push_back(filename_4);
    patches_filenames.push_back(filename_5);
    patches_filenames.push_back(filename_6);
    patches_filenames.push_back(filename_7);
    patches_filenames.push_back(filename_8);
    patches_filenames.push_back(filename_9);
    patches_filenames.push_back(filename_10);
    lines.push_back(std::make_pair(patches_filenames, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  Datum datum_aux;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;


  for (int line_id = 0; line_id < lines.size(); ++line_id) 
  {

    //LOG(INFO) << "Processing image " << line_id << " with " << lines[line_id].first.size() << " patches.";
    //LOG(INFO) << "                 (we have " << (lines[line_id].first.size() / NUM_CHANNELS) << " groups";


      //LOG(INFO) << "      got " << NUM_CHANNELS << " patches from " << group_id;

      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
        // Guess the encoding type from the file name
        string fn = lines[line_id].first[0];
        size_t p = fn.rfind('.');
        if ( p == fn.npos )
          LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
        enc = fn.substr(p);
        std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      }
  
      datum.set_channels(NUM_CHANNELS);  // one channel for each image in the group
      std::string buffer;

      for (int patch_id = 0; patch_id < NUM_CHANNELS; patch_id++)
      {
        //LOG(INFO) << "          channel " << patch_id << ": " << root_folder + lines[line_id].first[group_id+patch_id];
        status = ReadImageToDatum(root_folder + lines[line_id].first[patch_id],
            lines[line_id].second, resize_height, resize_width, is_color,
            enc, &datum_aux);
        if (status == false) continue;
        if (patch_id == 0)
        {
          datum.set_height(datum_aux.height());
          datum.set_width(datum_aux.width());
        }
        if (check_size) {
          if (!data_size_initialized) {
            data_size = datum_aux.channels() * datum_aux.height() * datum_aux.width();
            data_size_initialized = true;
          } else {
            const std::string& data = datum_aux.data();
            CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                << data.size();
          }
        }

        int datum_channels = NUM_CHANNELS;
        int datum_height = datum.height();
        int datum_width = datum.width();
        int datum_size = datum_channels * datum_height * datum_width;
        buffer.insert(datum_height * datum_width * patch_id, datum_aux.data());
        //LOG(INFO) << "          channel " << patch_id << " inserted!";
      }
  
      datum.set_data(buffer);
      datum.set_label(lines[line_id].second);

      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
          (lines[line_id].first[0]+lines[line_id].first[1]).c_str());



      //LOG(INFO) << "      put group " << group_id << " in the db.";

      // Put in db
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(string(key_cstr, length), out);

      if (++count % 1000 == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db->NewTransaction());
        LOG(INFO) << "Processed " << count << " files.";
      }

  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return 0;
}
