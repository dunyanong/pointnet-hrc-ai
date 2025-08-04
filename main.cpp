#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <sl/Camera.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include "pointnet_inference.h"

class ZEDPointNetApp {
public:
    ZEDPointNetApp() : 
        viewer_(new pcl::visualization::PCLVisualizer("PointNet Laptop Segmentation")),
        inference_engine_(nullptr),
        running_(false) {
        
        // Initialize viewer
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();
    }
    
    ~ZEDPointNetApp() {
        cleanup();
    }
    
    bool initialize() {
        // Initialize ZED camera
        if (!initializeZED()) {
            std::cerr << "Failed to initialize ZED camera" << std::endl;
            return false;
        }
        
        // Initialize PointNet model
        if (!initializePointNet()) {
            std::cerr << "Failed to initialize PointNet model" << std::endl;
            return false;
        }
        
        std::cout << "Initialization completed successfully!" << std::endl;
        return true;
    }
    
    void run() {
        if (!zed_.isOpened()) {
            std::cerr << "ZED camera not opened!" << std::endl;
            return;
        }
        
        running_ = true;
        
        // Main processing loop
        sl::Mat zed_cloud;
        sl::Mat zed_image;
        
        while (running_ && !viewer_->wasStopped()) {
            // Capture from ZED
            sl::ERROR_CODE err = zed_.grab();
            if (err == sl::ERROR_CODE::SUCCESS) {
                // Get point cloud and image
                zed_.retrieveMeasure(zed_cloud, sl::MEASURE::XYZRGBA);
                zed_.retrieveImage(zed_image, sl::VIEW::LEFT);
                
                // Convert to PCL format
                auto pcl_cloud = convertZEDtoPCL(zed_cloud);
                
                if (pcl_cloud && !pcl_cloud->empty()) {
                    // Preprocess point cloud
                    auto filtered_cloud = preprocessPointCloud(pcl_cloud);
                    
                    // Perform PointNet inference
                    std::vector<int> labels;
                    std::vector<std::vector<float>> probabilities;
                    
                    if (inference_engine_->segmentPointCloud(filtered_cloud, labels, probabilities)) {
                        // Visualize results
                        visualizeSegmentation(filtered_cloud, labels, probabilities);
                        
                        // Display ZED image
                        displayZEDImage(zed_image);
                    }
                }
            }
            
            // Update viewer
            viewer_->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
    
    void stop() {
        running_ = false;
    }

private:
    bool initializeZED() {
        sl::InitParameters init_params;
        init_params.camera_resolution = sl::RESOLUTION::HD720;
        init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
        init_params.coordinate_units = sl::UNIT::METER;
        init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
        
        sl::ERROR_CODE err = zed_.open(init_params);
        if (err != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "ZED initialization failed: " << err << std::endl;
            return false;
        }
        
        std::cout << "ZED camera initialized successfully" << std::endl;
        return true;
    }
    
    bool initializePointNet() {
        std::string model_path = "DeployModel/laptop_classifier_traced.pt";
        inference_engine_ = std::make_unique<PointNetInference>(model_path, "cuda");
        
        if (!inference_engine_->initialize()) {
            std::cerr << "Failed to initialize PointNet inference engine" << std::endl;
            return false;
        }
        
        std::cout << "PointNet model loaded successfully" << std::endl;
        return true;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr convertZEDtoPCL(const sl::Mat& zed_cloud) {
        auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        
        int width = zed_cloud.getWidth();
        int height = zed_cloud.getHeight();
        
        pcl_cloud->width = width;
        pcl_cloud->height = height;
        pcl_cloud->is_dense = false;
        pcl_cloud->points.resize(width * height);
        
        float* zed_ptr = zed_cloud.getPtr<float>();
        
        for (int i = 0; i < width * height; ++i) {
            float x = zed_ptr[i * 4 + 0];
            float y = zed_ptr[i * 4 + 1];
            float z = zed_ptr[i * 4 + 2];
            
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                pcl_cloud->points[i].x = x;
                pcl_cloud->points[i].y = y;
                pcl_cloud->points[i].z = z;
            } else {
                pcl_cloud->points[i].x = pcl_cloud->points[i].y = pcl_cloud->points[i].z = NAN;
            }
        }
        
        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*pcl_cloud, *pcl_cloud, indices);
        
        return pcl_cloud;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        
        // Statistical outlier removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        
        auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        sor.filter(*filtered_cloud);
        
        // Voxel grid downsampling
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(filtered_cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f); // 1cm voxel size
        
        auto downsampled_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        vg.filter(*downsampled_cloud);
        
        return downsampled_cloud;
    }
    
    void visualizeSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             const std::vector<int>& labels,
                             const std::vector<std::vector<float>>& probabilities) {
        
        // Create colored point cloud for visualization
        auto colored_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        colored_cloud->width = cloud->width;
        colored_cloud->height = cloud->height;
        colored_cloud->is_dense = cloud->is_dense;
        colored_cloud->points.resize(cloud->points.size());
        
        for (size_t i = 0; i < cloud->points.size() && i < labels.size(); ++i) {
            colored_cloud->points[i].x = cloud->points[i].x;
            colored_cloud->points[i].y = cloud->points[i].y;
            colored_cloud->points[i].z = cloud->points[i].z;
            
            // Color based on segmentation result
            if (labels[i] == 1) { // Laptop
                // Red for laptop points
                colored_cloud->points[i].r = 255;
                colored_cloud->points[i].g = 0;
                colored_cloud->points[i].b = 0;
            } else { // Background
                // Blue for background points
                colored_cloud->points[i].r = 0;
                colored_cloud->points[i].g = 0;
                colored_cloud->points[i].b = 255;
            }
        }
        
        // Update visualization
        if (!viewer_->updatePointCloud(colored_cloud, "segmentation")) {
            viewer_->addPointCloud(colored_cloud, "segmentation");
        }
        
        // Add text with statistics
        int laptop_points = std::count(labels.begin(), labels.end(), 1);
        int total_points = labels.size();
        float laptop_percentage = 100.0f * laptop_points / total_points;
        
        std::string text = "Laptop: " + std::to_string(laptop_points) + "/" + 
                          std::to_string(total_points) + " (" + 
                          std::to_string(laptop_percentage) + "%)";
        
        viewer_->removeText3D("stats");
        viewer_->addText(text, 10, 10, 16, 1.0, 1.0, 1.0, "stats");
    }
    
    void displayZEDImage(const sl::Mat& zed_image) {
        cv::Mat cv_image(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>());
        cv::Mat display_image;
        cv::cvtColor(cv_image, display_image, cv::COLOR_BGRA2BGR);
        
        cv::imshow("ZED Camera", display_image);
        cv::waitKey(1);
    }
    
    void cleanup() {
        if (zed_.isOpened()) {
            zed_.close();
        }
        cv::destroyAllWindows();
    }
    
    sl::Camera zed_;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
    std::unique_ptr<PointNetInference> inference_engine_;
    bool running_;
};

int main() {
    std::cout << "PointNet-HRC Vision System" << std::endl;
    std::cout << "===========================" << std::endl;
    
    ZEDPointNetApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    std::cout << "Starting main loop..." << std::endl;
    std::cout << "Press 'q' in the viewer window to quit" << std::endl;
    
    app.run();
    
    std::cout << "Application finished" << std::endl;
    return 0;
}