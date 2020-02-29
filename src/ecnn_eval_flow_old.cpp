#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#define foreach BOOST_FOREACH

bool parse_arguments(int argc, char* argv[],
                     std::string* events_source,
                     std::string* flow_source,
                     std::string* skip_frames)
{
  if(argc < 3)
  {
    std::cerr << "Not enough arguments" << std::endl;
    std::cerr << "Usage: rosrun ecnn_eval_flow ecnn_eval_flow events_source.bag flow_source(dir or h5) frames_to_skip";
    return false;
  }

  	  *events_source = std::string(argv[1]);
  	  *flow_source = std::string(argv[2]);
  	  *skip_frames = std::string(argv[3]);

  return true;
}

bool compute_stats(const std::string events_source,
                   std::string* flow_source,
                   std::string* skip_frames,
                   int& num_events,
                   int& num_frames,
				   double& num_normed_pos_events, 
				   double& num_normed_neg_events, 
                   double& duration)
{
  std::cout << "Processing: " << events_source << std::endl;

  auto const pos = events_source.find_last_of('/');
  const std::string output_dir = events_source.substr(0, pos + 1) + "stats/";
  const std::string output_filename = events_source.substr(
      pos + 1, events_source.length() - (pos + 1) - 4) + ".txt";
  const std::string path_to_output = output_dir + output_filename;
  boost::filesystem::create_directories(output_dir);

  rosbag::Bag input_bag;
  try
  {
    input_bag.open(events_source, rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    std::cerr << "Error: could not open rosbag: " << events_source << std::endl;
    return false;
  }

  rosbag::View view(input_bag);

  std::unordered_map<std::string, std::vector<dvs_msgs::Event>> event_packets_for_each_event_topic;

  const uint32_t num_messages = view.size();
  uint32_t message_index = 0;

  int num_events_tmp = 0;
  int num_frames_tmp = 0;
  double start_time;
  double end_time = 0;
  bool first_msg = true;

  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat prev_image;
  double pos_normed_event_sum = 0;
  int pos_events_since_last_frame = 0;
  double neg_normed_event_sum = 0;
  int neg_events_since_last_frame = 0;
  std::vector<double> pos_normed_events;
  std::vector<double> neg_normed_events;
  std::string image_topic = "/cam0/image_raw";
  std::string event_topic = "/cam0/events";
//  std::string image_topic = "/dvs/image_raw";
//  std::string event_topic = "/dvs/events";

  foreach(rosbag::MessageInstance const m, view)
  {
    if (m.getDataType() == "dvs_msgs/EventArray")
    {

      std::vector<dvs_msgs::Event>& events = event_packets_for_each_event_topic[m.getTopic()];
      dvs_msgs::EventArrayConstPtr s = m.instantiate<dvs_msgs::EventArray>();
      num_events_tmp += s->events.size();
      if (first_msg)
      {
        start_time = s->events.front().ts.toSec();
        first_msg = false;
      }
      end_time = std::max(s->events.back().ts.toSec(), end_time);

	  for (auto e : s->events) {
		if(e.polarity) {pos_events_since_last_frame++;}
		else{neg_events_since_last_frame++;}
	  }
	   
    }
	else if (m.getDataType() == "sensor_msgs/Image" && m.getTopic()==image_topic) {
		sensor_msgs::ImageConstPtr icp =
				m.instantiate<sensor_msgs::Image>();
		double timestamp = icp->header.stamp.toSec();
		try {
			cv_ptr = cv_bridge::toCvCopy(icp,
					sensor_msgs::image_encodings::TYPE_8UC1);
		} catch (cv_bridge::Exception& e) {
			return false;
		}
		if(num_frames_tmp > 0)
		{
			cv::Mat diff_img;
			cv::subtract(cv_ptr->image, prev_image, diff_img, cv::noArray(), CV_32FC1);
			cv::Mat pos_img;
			cv::Mat neg_img;
			cv::threshold(diff_img, pos_img, 0, 255, cv::THRESH_TOZERO);	
			cv::threshold(diff_img, neg_img, 0, 255, cv::THRESH_TOZERO_INV);	
			double pos_sum = cv::sum(pos_img)[0];
			double neg_sum = -cv::sum(neg_img)[0];
//			std::cout << "pos: " << pos_events_since_last_frame << "/" << pos_sum
//				<< ", neg: " << neg_events_since_last_frame << "/" << neg_sum << ", sum: " << cv::sum(diff_img)[0] << "\n";
			pos_normed_events.push_back(pos_events_since_last_frame/pos_sum);
			neg_normed_events.push_back(neg_events_since_last_frame/neg_sum);
//			std::cout << "pos: " << pos_normed_events.back() << ", neg: " << neg_normed_events.back() << "\n";
			pos_events_since_last_frame = 0; 
			neg_events_since_last_frame = 0; 
		}
		prev_image = cv_ptr->image;
		num_frames_tmp += 1;
	}
  }

  input_bag.close();

  num_events = num_events_tmp;
  num_normed_pos_events = accumulate(pos_normed_events.begin(), pos_normed_events.end(), 0.0)/pos_normed_events.size();
  num_normed_neg_events = accumulate(neg_normed_events.begin(), neg_normed_events.end(), 0.0)/neg_normed_events.size();
  num_frames = num_frames_tmp;
  duration = end_time - start_time;

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
  std::string events_source;
  std::string* flow_source;
  std::string* skip_frames;
  int max_num_events_per_packet;
  ros::Duration max_duration_event_packet;

  if (!parse_arguments(argc, argv, &events_source))
  {
    return -1;
  }

  double num_normed_pos_events; 
  double num_normed_neg_events; 
  int num_events;
  int num_frames;
  double duration;

  if (!compute_stats(events_source,
                    num_events,
                    num_frames,
					num_normed_pos_events,
					num_normed_neg_events,
                    duration))
  {
    return -1;
  }

  auto const pos = events_source.find_last_of('/');
  const std::string output_dir = events_source.substr(0, pos + 1) + "stats/";
  const std::string output_filename = events_source.substr(
      pos + 1, events_source.length() - (pos + 1) - 4) + ".txt";
  const std::string path_to_output = output_dir + output_filename;
  boost::filesystem::create_directories(output_dir);
  write_stats(path_to_output,
              num_events,
              num_frames,
			  num_normed_pos_events,
			  num_normed_neg_events,
              duration);
  return 0;
}
