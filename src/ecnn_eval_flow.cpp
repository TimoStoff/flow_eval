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

typedef std::vector<cv::Mat> Flow;

bool parse_arguments(int argc, char* argv[],
                     std::string* path_to_input_rosbag,
                     std::string* path_to_input_flow,
                     int* num_frames_skip,
                     std::string* events_topic,
                     std::string* frames_topic,
                     std::string* output_path,
					 int* ours
					 )
{
  if(argc < 8)
  {
    std::cerr << argc << ": Not enough arguments" << std::endl;
    std::cerr << "Usage: rosrun ecnn_eval_flow ecnn_eval_flow path_to_bag.bag" << std::endl;
    return false;
  }

  *path_to_input_rosbag = std::string(argv[1]);
  *path_to_input_flow  = std::string(argv[2]);
  std::istringstream ss(argv[3]);
  ss >> *num_frames_skip;
  *events_topic  = std::string(argv[4]);
  *frames_topic  = std::string(argv[5]);
  *output_path  = std::string(argv[6]);
  std::istringstream ss1(argv[7]);
  ss1 >> *ours;
  std::cout << "path to input: " << *path_to_input_rosbag << ", path_to_flow: " << *path_to_input_flow <<
		  ", num frames to skip: " << *num_frames_skip << std::endl;

  return true;
}

struct Event {
	double x;
	double y;
	double t;
	int s;
	Event(int x, int y, double t, signed char s) :
		x(x), y(y), t(t), s(s) {}
};

int centercrop(cv::Mat & in, cv::Size & new_size)
{
	const int offsetW = (in.cols - new_size.width) / 2;
	const int offsetH = (in.rows - new_size.height) / 2;
	const cv::Rect roi(offsetW, offsetH, new_size.width, new_size.height);
	in = in(roi).clone();
	return 1;
}

Flow load_flow_dir(std::string flow_base_path, int image_idx)
{
	std::stringstream ss;
	ss << "frame_" << std::setw(10) << std::setfill('0') << image_idx << ".yml";
	std::string s = ss.str();
	boost::filesystem::path dir(flow_base_path);
	boost::filesystem::path file(s);
	boost::filesystem::path flow_path = dir/file;
	if(boost::filesystem::exists(flow_path))
	{
		cv::FileStorage fs(flow_path.string(), cv::FileStorage::READ);
		cv::Mat flow_x;
		cv::Mat flow_y;
		fs["flow_x"] >> flow_x;
		fs["flow_y"] >> flow_y;
		Flow ret = {flow_x, flow_y};
		return ret;
	}
	Flow ret;
	return ret;
}

Flow load_flow_hdf(int image_idx)
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
		Flow flow = {flow_x, flow_y};
		return flow;
	}
	Flow ret;
	return ret;
}

int warp_events_to_image_reverse(std::vector<Flow> & flow_arr,
		std::vector<double> & flow_ts,
		std::vector<Event>& events,
		cv::Mat & iwe,
		int skip_frames,
		int start_event_idx)
{
	bool no_warp = false;
	int final_flow_idx = skip_frames-1;
	if(flow_ts.size() < skip_frames )
	{
		std::cout << "Insufficient frames:" << flow_ts.size() << "<" << skip_frames << std::endl;
		return -1;
	}
	double end_ts = flow_ts.at(final_flow_idx);
	int last_event_idx = 0;
	std::vector<double> timestamps;
	{
	for(int flow_idx = final_flow_idx-1; flow_idx >= 0; flow_idx--)
		timestamps.push_back(flow_ts.at(flow_idx));
	}
	timestamps.push_back(events.front().t);
	for(auto&t:timestamps){std::cout << t << ", "; }std::cout << std::endl;
	std::cout << "Reverse warping events, flow_ts=" << (flow_ts.size()<skip_frames) << std::endl;
	for(int flow_idx = 0; flow_idx <= final_flow_idx; flow_idx++)
	{
		last_event_idx = 0;
		std::cout << "Frame " << timestamps.at(flow_idx) << std::endl;
		for(int e_idx=start_event_idx; e_idx>=0; e_idx--)
		{
			Event & e = events.at(e_idx);
			double t_ref = timestamps.at(flow_idx);
			const Flow & flow = flow_arr.at(flow_idx);
			if(e.t<t_ref) {
				std::cout << "event " << e_idx << "=" << e.t << "<=" << t_ref << std::endl;
				break;
			}
			if(!no_warp)
			{
				double dt = t_ref - e.t;
				const int ex = int(e.x);
				const int ey = int(e.y);
				const cv::Size & flow_sz = flow.at(0).size();
				if(ex<0 || ex>flow_sz.width || ey<0 || ey>flow_sz.height) {continue;}
				double flowx = flow.at(0).at<float>(ey,ex);
				double flowy = flow.at(1).at<float>(ey,ex);
				e.x = e.x-flowx*dt;
				e.y = e.y-flowy*dt;
				e.t = t_ref;
			}
			last_event_idx++;
		}
	}
	for(int e_idx=0; e_idx<last_event_idx; e_idx++)
	{
		Event & e = events.at(e_idx);
		const int px = int(e.x);
		const int py = int(e.y);
		const double dx = e.x-px;
		const double dy = e.y-py;
		if(px<0 || px>=iwe.size().width-1 || py<0 || py>=iwe.size().height-1) {
			continue;
		}
		iwe.at<float>(cv::Point(px, py)) += e.s * (1.0 - dx) * (1.0 - dy);
		iwe.at<float>(cv::Point(px + 1, py)) += e.s * dx * (1.0 - dy);
		iwe.at<float>(cv::Point(px, py + 1)) += e.s * dy * (1.0 - dx);
		iwe.at<float>(cv::Point(px + 1, py + 1)) += e.s * dx * dy;
	}
	return last_event_idx;
}

int warp_events_to_avg_ts_image(std::vector<Flow> & flow_arr,
		std::vector<double> & flow_ts,
		std::vector<Event>& events,
		cv::Size & imgsz,
		int skip_frames,
		cv::Scalar & loss)
{
	cv::Mat pos_img(imgsz, CV_32FC1);
	cv::Mat pos_sum(imgsz, CV_32FC1);
	cv::Mat neg_img(imgsz, CV_32FC1);
	cv::Mat neg_sum(imgsz, CV_32FC1);
	bool no_warp = false;
	int final_flow_idx = skip_frames-1;
	const double first_ts = flow_ts.front();
	if(flow_ts.size() < skip_frames )
	{
		return -1;
	}
	if(events.back().t < flow_ts.at(final_flow_idx))
	{
		return -1;
	}
	std::cout << "Warping events, flow_ts=" << (flow_ts.size()<skip_frames) << std::endl;
	double end_ts = flow_ts.at(final_flow_idx);
	int last_event_idx = 0;
	for(int flow_idx = 0; flow_idx <= final_flow_idx; flow_idx++)
	{
		last_event_idx = 0;
		std::cout << "Frame " << flow_idx << std::endl;
		for(Event & e: events)
		{
			const Flow & flow = flow_arr.at(flow_idx);
			double t_ref = flow_ts.at(flow_idx);
			if(e.t >= end_ts || e.t >= t_ref) {break;}
			if(!no_warp)
			{
				double dt = t_ref - e.t;
				const int ex = int(e.x);
				const int ey = int(e.y);
				const cv::Size & flow_sz = flow.at(0).size();
				if(ex<0 || ex>flow_sz.width || ey<0 || ey>flow_sz.height) {continue;}
				double flowx = flow.at(0).at<float>(ey, ex);
				double flowy = flow.at(1).at<float>(ey, ex);
				e.x = e.x+flowx*dt;
				e.y = e.y+flowy*dt;
				e.t = t_ref;
			}
			last_event_idx++;
		}
	}
	for(int e_idx=0; e_idx<last_event_idx; e_idx++)
	{
		Event & e = events.at(e_idx);
		const int px = int(e.x);
		const int py = int(e.y);
		const double dx = e.x-px;
		const double dy = e.y-py;
		if(px<0 || px>=imgsz.width-1 || py<0 || py>=imgsz.height-1) {
			continue;
		}
		double ts = e.t-first_ts;
		if(e.s>0)
		{
			pos_img.at<float>(cv::Point(px, py)) += ts * (1.0 - dx) * (1.0 - dy);
			pos_img.at<float>(cv::Point(px + 1, py)) += ts * dx * (1.0 - dy);
			pos_img.at<float>(cv::Point(px, py + 1)) += ts * dy * (1.0 - dx);
			pos_img.at<float>(cv::Point(px + 1, py + 1)) += ts * dx * dy;
			pos_sum.at<float>(cv::Point(px, py)) +=  (1.0 - dx) * (1.0 - dy);
			pos_sum.at<float>(cv::Point(px + 1, py)) +=  dx * (1.0 - dy);
			pos_sum.at<float>(cv::Point(px, py + 1)) +=  dy * (1.0 - dx);
			pos_sum.at<float>(cv::Point(px + 1, py + 1)) += dx * dy;
		} else
		{
			neg_img.at<float>(cv::Point(px, py)) += ts * (1.0 - dx) * (1.0 - dy);
			neg_img.at<float>(cv::Point(px + 1, py)) += ts * dx * (1.0 - dy);
			neg_img.at<float>(cv::Point(px, py + 1)) += ts * dy * (1.0 - dx);
			neg_img.at<float>(cv::Point(px + 1, py + 1)) += ts * dx * dy;
			neg_sum.at<float>(cv::Point(px, py)) +=  (1.0 - dx) * (1.0 - dy);
			neg_sum.at<float>(cv::Point(px + 1, py)) +=  dx * (1.0 - dy);
			neg_sum.at<float>(cv::Point(px, py + 1)) +=  dy * (1.0 - dx);
			neg_sum.at<float>(cv::Point(px + 1, py + 1)) += dx * dy;
		}
	}
	pos_img = pos_img.mul(1.0/pos_sum);
	neg_img = neg_img.mul(1.0/neg_sum);
	cv::Mat pos_prod = pos_img.mul(pos_img);
	cv::Mat neg_prod = neg_img.mul(neg_img);
	loss = cv::sum(pos_prod) + cv::sum(neg_prod);
	return last_event_idx;
}
int warp_events_to_image(std::vector<Flow> & flow_arr,
		std::vector<double> & flow_ts,
		std::vector<Event>& events,
		cv::Mat & iwe,
		int skip_frames)
{
	bool no_warp = false;
	int final_flow_idx = skip_frames-1;
	if(flow_ts.size() < skip_frames )
	{
//		std::cout << "Insufficient frames:" << flow_ts.size() << "<" << skip_frames << std::endl;
		return -1;
	}
	if(events.back().t < flow_ts.at(final_flow_idx))
	{
//		std::cout << "Insufficient events:" << flow_ts.size() << "<" << skip_frames
//				<< ", " << events.back().t << "<" << flow_ts.at(final_flow_idx)<< std::endl;
		return -1;
	}
	std::cout << "Warping events, flow_ts=" << (flow_ts.size()<skip_frames) << std::endl;
	double end_ts = flow_ts.at(final_flow_idx);
	int last_event_idx = 0;
	for(int flow_idx = 0; flow_idx <= final_flow_idx; flow_idx++)
	{
		last_event_idx = 0;
		std::cout << "Frame " << flow_idx << std::endl;
		for(Event & e: events)
		{
			const Flow & flow = flow_arr.at(flow_idx);
			double t_ref = flow_ts.at(flow_idx);
			if(e.t >= end_ts || e.t >= t_ref) {break;}
			if(!no_warp)
			{
				double dt = t_ref - e.t;
				const int ex = int(e.x);
				const int ey = int(e.y);
				const cv::Size & flow_sz = flow.at(0).size();
				if(ex<0 || ex>flow_sz.width || ey<0 || ey>flow_sz.height) {continue;}
				double flowx = flow.at(0).at<float>(ey, ex);
				double flowy = flow.at(1).at<float>(ey, ex);
				e.x = e.x+flowx*dt;
				e.y = e.y+flowy*dt;
				e.t = t_ref;
			}
			last_event_idx++;
		}
	}
	for(int e_idx=0; e_idx<last_event_idx; e_idx++)
	{
		Event & e = events.at(e_idx);
		const int px = int(e.x);
		const int py = int(e.y);
		const double dx = e.x-px;
		const double dy = e.y-py;
		if(px<0 || px>=iwe.size().width-1 || py<0 || py>=iwe.size().height-1) {
			continue;
		}
		iwe.at<float>(cv::Point(px, py)) += e.s * (1.0 - dx) * (1.0 - dy);
		iwe.at<float>(cv::Point(px + 1, py)) += e.s * dx * (1.0 - dy);
		iwe.at<float>(cv::Point(px, py + 1)) += e.s * dy * (1.0 - dx);
		iwe.at<float>(cv::Point(px + 1, py + 1)) += e.s * dx * dy;
	}
	return last_event_idx;
}

std::vector<double> warp_events(const std::string path_to_input_rosbag,
		const std::string path_to_input_flow,
		const int num_frames_skip,
		const std::string event_topic,
		const std::string image_topic,
		const bool & our_flow
		)
{
  std::cout << "Processing: " << path_to_input_rosbag << std::endl;
  std::vector<double> variances;

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
    return variances;
  }

  rosbag::View view(input_bag);

  std::unordered_map<std::string, std::vector<dvs_msgs::Event>> event_packets_for_each_event_topic;

  const uint32_t num_messages = view.size();
  uint32_t message_index = 0;

  cv_bridge::CvImagePtr cv_ptr;
  cv::Mat prev_image;
  double prev_image_ts;
  bool reverse_warp = false;
  bool first_msg = true;
  bool first_frame = true;
  bool has_offset = false;
  double start_time = 0;
  double end_time = 0;
  int image_idx = 0;
  char script[1024];
  int w_offset = 0;
  int h_offset = 0;
  std::vector<Flow> flow_arr;
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
     		Event event(e.x-w_offset, e.y-h_offset, e.ts.toSec(), e.polarity?1:-1);
     		if(event.x>=0 && event.y>=0) {event_arr.push_back(event);}
     	}
    }
	else if (m.getDataType() == "sensor_msgs/Image" && m.getTopic()==image_topic)
	{
		sensor_msgs::ImageConstPtr icp =
				m.instantiate<sensor_msgs::Image>();
		double timestamp = icp->header.stamp.toSec();
		if(first_frame)
		{
			prev_image_ts = timestamp;
			first_frame = false;
			continue;
		}
		try
		{
			cv_ptr = cv_bridge::toCvCopy(icp,
					sensor_msgs::image_encodings::TYPE_8UC1);
		} catch (cv_bridge::Exception& e)
		{
			return variances;
		}
		if(flow_h5)
		{
			Flow flow = load_flow_hdf(image_idx);
			flow_arr.push_back(flow);
			flow_ts.push_back(timestamp);
		} else
		{
			double dt = timestamp-prev_image_ts;
			Flow flow = load_flow_dir(path_to_input_flow, image_idx);
			for(auto & f_c:flow)
			{
				f_c /= dt;
			}
			flow_arr.push_back(flow);
			flow_ts.push_back(timestamp);
		}
		if(flow_arr.back().size() == 0)
		{
			input_bag.close();
			return variances;
		}
		if(!has_offset)
		{
			w_offset = (cv_ptr->image.cols-flow_arr.at(0).at(0).cols)/2;
			h_offset = (cv_ptr->image.rows-flow_arr.at(0).at(0).rows)/2;
			for(Event & e : event_arr)
			{
				e.x -= w_offset;
				e.y -= h_offset;
			}
			has_offset = true;
			std::cout << "OFFSET = " << w_offset << ", " << h_offset << std::endl;
		}
		std::cout << "Image " << image_idx << std::endl;
		cv::Mat iwe = cv::Mat::zeros(flow_arr.at(0).at(0).size(), CV_32FC1);
		int last_event_idx = warp_events_to_image(flow_arr, flow_ts, event_arr, iwe, num_frames_skip);

		if(last_event_idx > -1)
		{
			//Need to do this, since EVFlowNet has 160x160 crops
			//for fair comparison
			if(our_flow)
			{
				const cv::Rect roi(40, 10, 160, 160);
				iwe = iwe(roi).clone();
			}
			cv::Scalar mean, stdev;
			cv::meanStdDev(iwe, mean, stdev);
			iwe -= mean[0];
			cv::meanStdDev(iwe, mean, stdev);
			double var = stdev[0]*stdev[0];

			if(reverse_warp)
			{
				iwe = cv::Mat::zeros(flow_arr.at(0).at(0).size(), CV_32FC1);
				warp_events_to_image_reverse(flow_arr, flow_ts, event_arr, iwe,
						num_frames_skip, last_event_idx);
				if(our_flow)
				{
					const cv::Rect roi(40, 10, 160, 160);
					iwe = iwe(roi).clone();
				}
				cv::meanStdDev(iwe, mean, stdev);
				iwe -= mean[0];
				cv::meanStdDev(iwe, mean, stdev);
				var += stdev[0]*stdev[0];
			}

			std::stringstream ss;
			ss << "/tmp/warps/frame_" << std::setw(9) << std::setfill('0') << image_idx << ".png";
			std::string s = ss.str();
			cv::Mat normed;
			cv::normalize(iwe, normed, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::imwrite(s, normed);

			variances.push_back(var);
			event_arr.erase(event_arr.begin(), event_arr.begin()+last_event_idx);
			flow_arr.erase(flow_arr.begin(), flow_arr.begin()+num_frames_skip);
			flow_ts.erase(flow_ts.begin(), flow_ts.begin()+num_frames_skip);
		}
		image_idx ++;
		prev_image_ts = timestamp;
	}
  }

  input_bag.close();

  return variances;
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
  std::string output_path;
  int num_frames_skip;
  int ours;

  if (!parse_arguments(argc, argv, &path_to_input_rosbag, &path_to_input_flow,
		  &num_frames_skip, &events_topic, &frames_topic, &output_path, &ours))
  {
    return -1;
  }
  bool our_flow = ours==1?true:false;
  boost::filesystem::create_directories(output_path);
  boost::filesystem::path outpath(output_path);
  boost::filesystem::path outfile(path_to_input_rosbag);
  boost::filesystem::path rootdir = outfile.stem();
  boost::filesystem::path file_path = output_path/rootdir;

  std::vector<double> variances = warp_events(path_to_input_rosbag,
		  path_to_input_flow,
		  num_frames_skip,
		  events_topic,
		  frames_topic,
		  our_flow);

  std::string variance_file_name = file_path.string()+".txt";
  std::cout << "Saving results to " << variance_file_name << std::endl;
  std::ofstream myfile;
  double total_var = 0;
  myfile.open(variance_file_name);
  //EVFlowNet seems to end ~5 frames earlier
  int offset = our_flow?5:0;
  for(int i=0; i<variances.size()-offset; i++)
  {
	  total_var += variances.at(i);
  }
  myfile << rootdir.string() << "\n";
  myfile << "variance " << total_var/(1.0*variances.size());
  myfile.close();

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
