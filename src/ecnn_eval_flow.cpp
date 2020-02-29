#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <HDFql.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#define foreach BOOST_FOREACH


bool parse_arguments(int argc, char* argv[],
                     std::string* path_to_input_rosbag,
                     std::string* path_to_input_flow,
                     int* num_frames_skip,
                     std::string* events_topic,
                     std::string* frames_topic
					 )
{
  if(argc < 6)
  {
    std::cerr << "Not enough arguments" << std::endl;
    std::cerr << "Usage: rosrun ecnn_eval_flow ecnn_eval_flow path_to_bag.bag";
    return false;
  }

  *path_to_input_rosbag = std::string(argv[1]);
  *path_to_input_flow  = std::string(argv[2]);
  std::istringstream ss(argv[3]);
  ss >> *num_frames_skip;
  *events_topic  = std::string(argv[4]);
  *frames_topic  = std::string(argv[5]);
  std::cout << "path to input: " << *path_to_input_rosbag << ", path_to_flow: " << *path_to_input_flow <<
		  ", num frames to skip: " << *num_frames_skip << std::endl;

  return true;
}

struct Event {
	int x;
	int y;
	double t;
	signed char s;
	Event(int x, int y, double t, signed char s) :
		x(x), y(y), t(t), s(s) {}
};

std::vector<cv::Mat> load_flow_dir(std::string flow_base_path, int image_idx)
{
	std::stringstream ss;
	ss << "frame_" << std::setw(10) << std::setfill('0') << image_idx << ".yml";
	std::string s = ss.str();
	boost::filesystem::path dir(flow_base_path);
	boost::filesystem::path file(s);
	boost::filesystem::path flow_path = dir/file;
	std::cout << flow_path.string() << std::endl;
	if(boost::filesystem::exists(flow_path))
	{
		std::cout << flow_path << " exists!" << std::endl;
		cv::FileStorage fs(flow_path.string(), cv::FileStorage::READ);
		cv::Mat flow_x;
		cv::Mat flow_y;
		fs["flow_x"] >> flow_x;
		fs["flow_y"] >> flow_y;
		std::vector<cv::Mat> ret = {flow_x, flow_y};
		std::cout << flow_x.size() << std::endl;
	}
	std::vector<cv::Mat> ret;
	return ret;
}

std::vector<cv::Mat> load_flow_hdf(int image_idx)
{

	char script[1024];
	std::stringstream ss;
	ss << "frame_" << std::setw(9) << std::setfill('0') << image_idx;
	std::string s = ss.str();
	sprintf(script, ("SHOW DATASET " + s + "/flow/flow_x").c_str());
	int success = HDFql::execute(script);
	if(success == 0)
	{
		double hdf_ts = 0;
		sprintf(script, ("SELECT FROM " + s +"/timestamp INTO MEMORY %d").c_str(),
				HDFql::variableTransientRegister(&hdf_ts));
		std::cout << script << ": timestamp = " << hdf_ts << std::endl;
		long long image_size[2] = {0, 0};
		sprintf(script, ("SHOW DIMENSION " + s + "/flow/flow_x INTO MEMORY %d")
				.c_str(), HDFql::variableTransientRegister(&image_size));
		HDFql::execute(script);
		cv::Mat flow_x = cv::Mat::zeros(image_size[0], image_size[1], CV_32FC1);
		cv::Mat flow_y = cv::Mat::zeros(image_size[0], image_size[1], CV_32FC1);
		sprintf(script, ("SELECT FROM " + s + "/flow/flow_x INTO MEMORY %d").c_str(),
				HDFql::variableTransientRegister(flow_x.data));
		int sx = HDFql::execute(script);
		sprintf(script, ("SELECT FROM " + s + "/flow/flow_y INTO MEMORY %d").c_str(),
				HDFql::variableTransientRegister(flow_y.data));
		int sy = HDFql::execute(script);
		std::cout << s << ": Flow size = " << image_size[0] << "x" << image_size[1] << std::endl;
		std::cout << "Loading flow: " << sx << ", " << sy << std::endl;
		std::vector<cv::Mat> flow = {flow_x, flow_y};
		return flow;
	}
	std::vector<cv::Mat> ret;
	return ret;
}

bool warp_events(const std::string path_to_input_rosbag,
		const std::string path_to_input_flow,
		const int num_frames_skip,
		const std::string event_topic,
		const std::string image_topic
		)
{
  std::cout << "Processing: " << path_to_input_rosbag << std::endl;

  auto const pos = path_to_input_rosbag.find_last_of('/');
  const std::string output_dir = path_to_input_rosbag.substr(0, pos + 1) + "stats/";
  const std::string output_filename = path_to_input_rosbag.substr(
      pos + 1, path_to_input_rosbag.length() - (pos + 1) - 4) + ".txt";
  const std::string path_to_output = output_dir + output_filename;
  boost::filesystem::create_directories(output_dir);

  std::string flow_ext = boost::filesystem::extension(path_to_input_flow);
  bool flow_h5 = flow_ext.compare(".h5") == 0?true:false;
  std::cout << "extension = " << flow_ext << ", flow_h5 = " << flow_h5 << std::endl;

  rosbag::Bag input_bag;
  try
  {
    input_bag.open(path_to_input_rosbag, rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    std::cerr << "Error: could not open rosbag: " << path_to_input_rosbag << std::endl;
    return false;
  }

  rosbag::View view(input_bag);

  std::unordered_map<std::string, std::vector<dvs_msgs::Event>> event_packets_for_each_event_topic;

  const uint32_t num_messages = view.size();
  uint32_t message_index = 0;

  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat prev_image;
  bool first_msg = true;
  bool first_frame = true;
  double start_time = 0;
  double end_time = 0;
  int image_idx = 0;
  char script[1024];
  std::vector<std::vector<cv::Mat>> flow_arr;
  std::vector<double> flow_ts;
  std::vector<Event> event_arr;

  if(flow_h5)
  {
	  sprintf(script, "USE FILE %s", path_to_input_flow.c_str());
	  HDFql::execute(script);
  }

  foreach(rosbag::MessageInstance const m, view)
  {
    if (m.getDataType() == "dvs_msgs/EventArray")
    {

      std::vector<dvs_msgs::Event>& events = event_packets_for_each_event_topic[m.getTopic()];
      dvs_msgs::EventArrayConstPtr s = m.instantiate<dvs_msgs::EventArray>();
      int num_events_tmp = s->events.size();
      if (first_msg)
      {
        start_time = s->events.front().ts.toSec();
        first_msg = false;
      }
      end_time = std::max(s->events.back().ts.toSec(), end_time);

	  for (auto e : s->events) {
		  Event event(e.x, e.y, e.ts.toSec(), e.polarity?1:-1);
		  event_arr.push_back(event);
		  //Warp events
	  }
	   
    }
	else if (m.getDataType() == "sensor_msgs/Image" && m.getTopic()==image_topic)
	{
		sensor_msgs::ImageConstPtr icp =
				m.instantiate<sensor_msgs::Image>();
		double timestamp = icp->header.stamp.toSec();
		if(first_frame)
		{
			continue;
		}
		try
		{
			cv_ptr = cv_bridge::toCvCopy(icp,
					sensor_msgs::image_encodings::TYPE_8UC1);
		} catch (cv_bridge::Exception& e)
		{
			return false;
		}
		if(flow_h5)
		{
			std::vector<cv::Mat> flow = load_flow_hdf(image_idx);
			flow_arr.push_back(flow);
			flow_ts.push_back(timestamp);
			std::cout << flow.size() << std::endl;
		} else
		{
			std::vector<cv::Mat> flow = load_flow_dir(path_to_input_flow, image_idx);
			flow_arr.push_back(flow);
			flow_ts.push_back(timestamp);
			std::cout << flow.size() << std::endl;
		}
		image_idx ++;
	}
  }

  input_bag.close();

  return true;
}

bool write_stats(const std::string path_to_output,
                 const int& num_events,
                 const int& num_frames,
				 const double& num_normed_pos_events, 
				 const double& num_normed_neg_events, 
                 const double& duration)
{

  std::ofstream stats_file;
  stats_file.open(path_to_output);
//  stats_file << "Number of events: " << num_events << '\n';
//  stats_file << "Number of frames: " << num_frames << '\n';
//  stats_file << "Total duration (s): " << duration << '\n';
  stats_file << num_events << ", " << num_frames << ", " << duration << ", " << num_normed_pos_events << ", " << num_normed_neg_events << std::endl;
  stats_file.close();
  return true;
}

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int main(int argc, char* argv[])
{
  std::string path_to_input_rosbag;
  std::string path_to_input_flow;
  std::string events_topic;
  std::string frames_topic;
  int num_frames_skip;

  if (!parse_arguments(argc, argv, &path_to_input_rosbag, &path_to_input_flow,
		  &num_frames_skip, &events_topic, &frames_topic))
  {
    return -1;
  }

  if (!warp_events(path_to_input_rosbag,
		  path_to_input_flow,
		  num_frames_skip,
		  events_topic,
		  frames_topic))
  {
	  return -1;
  }

//  auto const pos = path_to_input_rosbag.find_last_of('/');
//  const std::string output_dir = path_to_input_rosbag.substr(0, pos + 1) + "stats/";
//  const std::string output_filename = path_to_input_rosbag.substr(
//      pos + 1, path_to_input_rosbag.length() - (pos + 1) - 4) + ".txt";
//  const std::string path_to_output = output_dir + output_filename;
//  boost::filesystem::create_directories(output_dir);
//  write_stats(path_to_output,
//              num_events,
//              num_frames,
//			  num_normed_pos_events,
//			  num_normed_neg_events,
//              duration);
  return 0;
}
