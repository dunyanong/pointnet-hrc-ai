#include <sl/Camera.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

class PointNetInference {
private:
    sl::Camera zed;
    torch::jit::script::Module model;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    
public:
    PointNetInference(const std::string& model_path) {
        // Initialize ZED camera
        sl::InitParameters init_params;
        init_params.camera_resolution = sl::RESOLUTION::HD720;
        init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
        init_params.coordinate_units = sl::UNIT::METER;
        
        sl::ERROR_CODE zed_error = zed.open(init_params);
        if (zed_error != sl::ERROR_CODE::SUCCESS) {
            throw std::runtime_error("Failed to open ZED camera: " + std::to_string((int)zed_error));
        }
        
        // Load TorchScript model
        try {
            model = torch::jit::load(model_path);
            model.eval();
            std::cout << "Model loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model: " + std::string(e.what()));
        }
        
        // Configure voxel grid filter
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
    }
    
    ~PointNetInference() {
        zed.close();
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr capturePointCloud() {
        sl::Mat point_cloud_mat;
        sl::RuntimeParameters runtime_params;
        
        // Grab frame
        if (zed.grab(runtime_params) == sl::ERROR_CODE::SUCCESS) {
            zed.retrieveMeasure(point_cloud_mat, sl::MEASURE::XYZRGBA);
            
            // Convert ZED point cloud to PCL format
            auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            
            int width = point_cloud_mat.getWidth();
            int height = point_cloud_mat.getHeight();
            
            pcl_cloud->width = width;
            pcl_cloud->height = height;
            pcl_cloud->is_dense = false;
            pcl_cloud->points.resize(width * height);
            
            sl::float4* point_cloud_ptr = point_cloud_mat.getPtr<sl::float4>();
            
            for (int i = 0; i < width * height; ++i) {
                sl::float4 point = point_cloud_ptr[i];
                
                if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                    pcl_cloud->points[i].x = point.x;
                    pcl_cloud->points[i].y = point.y;
                    pcl_cloud->points[i].z = point.z;
                } else {
                    pcl_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
                    pcl_cloud->points[i].y = std::numeric_limits<float>::quiet_NaN();
                    pcl_cloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
                }
            }
            
            return pcl_cloud;
        }
        
        return nullptr;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
        
        auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        
        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*input_cloud, *filtered_cloud, indices);
        
        // Apply voxel grid filter for downsampling
        auto downsampled_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        voxel_filter.setInputCloud(filtered_cloud);
        voxel_filter.filter(*downsampled_cloud);
        
        return downsampled_cloud;
    }
    
    torch::Tensor pointCloudToTensor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        std::vector<float> points_data;
        points_data.reserve(cloud->size() * 3);
        
        for (const auto& point : cloud->points) {
            points_data.push_back(point.x);
            points_data.push_back(point.y);
            points_data.push_back(point.z);
        }
        
        // Create tensor with shape [1, N, 3] where N is number of points
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor tensor = torch::from_blob(
            points_data.data(), 
            {1, static_cast<long>(cloud->size()), 3}, 
            options
        ).clone();
        
        return tensor;
    }
    
    std::string runInference(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        if (cloud->empty()) {
            return "Empty point cloud";
        }
        
        // Convert point cloud to tensor
        torch::Tensor input_tensor = pointCloudToTensor(cloud);
        
        // Prepare input for model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Run inference
        try {
            torch::NoGradGuard no_grad;
            at::Tensor output = model.forward(inputs).toTensor();
            
            // Apply softmax to get probabilities
            auto probabilities = torch::softmax(output, 1);
            
            // Get predicted class
            auto max_result = torch::max(probabilities, 1);
            int predicted_class = std::get<1>(max_result).item<int>();
            float confidence = std::get<0>(max_result).item<float>();
            
            // Map class index to label (adjust based on your model)
            std::vector<std::string> class_labels = {"background", "laptop", "other_object"};
            
            if (predicted_class < class_labels.size()) {
                return class_labels[predicted_class] + " (confidence: " + 
                       std::to_string(confidence) + ")";
            } else {
                return "unknown_class";
            }
            
        } catch (const std::exception& e) {
            return "Inference error: " + std::string(e.what());
        }
    }
    
    void run() {
        std::cout << "Starting point cloud capture and inference..." << std::endl;
        
        while (true) {
            // Capture point cloud
            auto raw_cloud = capturePointCloud();
            if (!raw_cloud) {
                std::cout << "Failed to capture point cloud" << std::endl;
                continue;
            }
            
            // Preprocess point cloud
            auto processed_cloud = preprocessPointCloud(raw_cloud);
            
            std::cout << "Captured " << processed_cloud->size() << " points" << std::endl;
            
            // Run inference
            std::string result = runInference(processed_cloud);
            std::cout << "Classification result: " << result << std::endl;
            
            // Check for exit condition
            char key = cv::waitKey(1);
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
            
            // Small delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.pt>" << std::endl;
        return -1;
    }
    
    try {
        PointNetInference inference(argv[1]);
        inference.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}