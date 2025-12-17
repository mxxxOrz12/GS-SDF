#pragma once
#include "data_loader/data_parsers/rosbag_parser.hpp"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"
#include "utils/sensor_utils/sensors.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Fastlivo : Rosbag {
  explicit Fastlivo(const std::filesystem::path &_bag_path,
                    const torch::Device &_device = torch::kCPU,
                    const bool &_preload = true, const float &_res_scale = 1.0,
                    const sensor::Sensors &_sensor = sensor::Sensors(),
                    const int &_ds_pt_num = 1e5)
      : Rosbag(_bag_path, _device, _preload, _res_scale,
               coords::SystemType::OpenCV, _sensor, _ds_pt_num) {
    depth_type_ = DepthType::PLY;

    dataset_name_ = bag_path_.filename();
    dataset_name_ = dataset_name_.replace_extension();

    color_pose_topic = "/aft_mapped_to_init";
    color_topic = "/origin_img";
    depth_pose_topic = "/aft_mapped_to_init";
    depth_topic = "/cloud_registered_body";

    load_intrinsics();

    load_data();
    if (std::filesystem::exists(color_pose_path_)) {
    } else {
      std::cout << "pose_path_ does not exist: " << color_pose_path_ << std::endl;
    }
  }

  void load_intrinsics() override {
    // auto scale = 0.5f; // output image from Fastlivo is 640x512
    auto scale = 1.0f; // output image from Fastlivo is 1280x1024
    sensor_.camera.width = scale * sensor_.camera.width;
    sensor_.camera.height = scale * sensor_.camera.height;
    // HKU
    sensor_.camera.fx = scale * sensor_.camera.fx;
    sensor_.camera.fy = scale * sensor_.camera.fy;
    sensor_.camera.cx = scale * sensor_.camera.cx;
    sensor_.camera.cy = scale * sensor_.camera.cy;
  }


  void load_normals() {
    std::filesystem::path normal_dir = bag_path_ / "normals"; // 你的法线文件夹
    if (!std::filesystem::exists(normal_dir)) return;

    std::vector<torch::Tensor> normal_list;
    // 遍历训练用的 RGB 图片索引
    size_t num_imgs = train_to_raw_map_ids_.empty() ? raw_color_filelists_.size() : train_to_raw_map_ids_.size();

    for (size_t i = 0; i < num_imgs; ++i) {
        int idx = train_to_raw_map_ids_.empty() ? i : train_to_raw_map_ids_[i];
        std::filesystem::path rgb_path = raw_color_filelists_[idx];
        
        // 拼凑法线路径 (假设文件名相同，后缀为 .png)
        std::filesystem::path normal_path = normal_dir / rgb_path.stem(); 
        normal_path.replace_extension(".png");

        cv::Mat n_img = cv::imread(normal_path.string());
        if (n_img.empty()) {
            n_img = cv::Mat::zeros(sensor_.camera.height, sensor_.camera.width, CV_8UC3);
        } else {
             // 强制 Resize 到和 RGB 一样大
             if (n_img.rows != sensor_.camera.height || n_img.cols != sensor_.camera.width) {
                 cv::resize(n_img, n_img, cv::Size(sensor_.camera.width, sensor_.camera.height));
             }
        }

        // 转 Tensor 并归一化 [-1, 1]
        torch::Tensor n_tensor = torch::from_blob(n_img.data, {n_img.rows, n_img.cols, 3}, torch::kByte);
        n_tensor = n_tensor.permute({2, 0, 1}).to(torch::kFloat32); // [3, H, W]
        n_tensor = (n_tensor / 127.5f) - 1.0f;
        normal_list.push_back(n_tensor);
    }
    
    if(!normal_list.empty()) {
        train_normal_ = torch::stack(normal_list).to(device_); // [N, 3, H, W]
    }
  }


};
} // namespace dataparser