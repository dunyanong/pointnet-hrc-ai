#include "pointnet_inference.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>

PointNetInference::PointNetInference(const std::string& model_path, const std::string& device)
    : model_path_(model_path), device_str_(device), device_(torch::kCPU), model_loaded_(false) {
    
    // Set device
    if (device == "cuda" && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA, 0);
        std::cout << "Using CUDA device" << std::endl;
    } else {
        device_ = torch::Device(torch::kCPU);
        std::cout << "Using CPU device" << std::endl;
    }
}

PointNetInference::~PointNetInference() {
    // Cleanup handled by RAII
}

bool PointNetInference::initialize() {
    try {
        std::cout << "Loading model from: " << model_path_ << std::endl;
        
        // Load the TorchScript model
        model_ = torch::jit::load(model_path_, device_);
        model_.eval();
        
        model_loaded_ = true;
        std::cout << "Model loaded successfully!" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

bool PointNetInference::segmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                         std::vector<int>& labels,
                                         std::vector<std::vector<float>>& probabilities) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded!" << std::endl;
        return false;
    }
    
    if (cloud->empty()) {
        std::cerr << "Empty point cloud!" << std::endl;
        return false;
    }
    
    try {
        // Preprocess the point cloud
        torch::Tensor input_tensor = preprocessPointCloud(cloud);
        input_tensor = input_tensor.to(device_);
        
        // Add batch dimension
        input_tensor = input_tensor.unsqueeze(0); // [1, 3, N]
        
        std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;
        
        // Perform inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::NoGradGuard no_grad;
        auto result = model_.forward(inputs);
        
        // Handle tuple output from PointNet model (output, feature_transform)
        at::Tensor output;
        if (result.isTuple()) {
            auto tuple_result = result.toTuple();
            output = tuple_result->elements()[0].toTensor();
        } else {
            output = result.toTensor();
        }
        
        std::cout << "Output tensor shape: " << output.sizes() << std::endl;
        
        // Apply softmax to get probabilities
        output = torch::softmax(output, /*dim=*/1);
        
        // Move to CPU for processing
        output = output.to(torch::kCPU);
        
        // Apply confidence threshold to reduce false positives
        const float LAPTOP_CONFIDENCE_THRESHOLD = 0.85f;  // Require 85% confidence for laptop classification
        
        // Convert to vectors
        auto output_accessor = output.accessor<float, 2>();
        
        int num_points = output.size(0);
        labels.resize(num_points);
        probabilities.resize(num_points);
        
        for (int i = 0; i < num_points; ++i) {
            probabilities[i].resize(NUM_CLASSES);
            for (int j = 0; j < NUM_CLASSES; ++j) {
                probabilities[i][j] = output_accessor[i][j];
            }
            
            // Apply threshold: only classify as laptop if confidence > threshold
            if (probabilities[i][1] > LAPTOP_CONFIDENCE_THRESHOLD) {
                labels[i] = 1;  // Laptop
            } else {
                labels[i] = 0;  // Background
            }
        }
        
        // Print statistics
        int laptop_points = std::count(labels.begin(), labels.end(), 1);
        int background_points = std::count(labels.begin(), labels.end(), 0);
        
        // Calculate average confidence for debugging
        float avg_laptop_confidence = 0.0f;
        float max_laptop_confidence = 0.0f;
        for (int i = 0; i < num_points; ++i) {
            avg_laptop_confidence += probabilities[i][1];
            max_laptop_confidence = std::max(max_laptop_confidence, probabilities[i][1]);
        }
        avg_laptop_confidence /= num_points;
        
        std::cout << "Segmentation results:" << std::endl;
        std::cout << "  Background points: " << background_points 
                  << " (" << (100.0 * background_points / num_points) << "%)" << std::endl;
        std::cout << "  Laptop points: " << laptop_points 
                  << " (" << (100.0 * laptop_points / num_points) << "%)" << std::endl;
        std::cout << "  Avg laptop confidence: " << avg_laptop_confidence << std::endl;
        std::cout << "  Max laptop confidence: " << max_laptop_confidence << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor PointNetInference::preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                     int target_points) {
    // Convert PCL point cloud to tensor
    std::vector<std::vector<float>> points_vec;
    
    if (static_cast<int>(cloud->size()) > target_points) {
        // Random sampling if too many points
        pcl::RandomSample<pcl::PointXYZ> sampler;
        sampler.setInputCloud(cloud);
        sampler.setSample(target_points);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        sampler.filter(*sampled_cloud);
        
        for (const auto& point : *sampled_cloud) {
            points_vec.push_back({point.x, point.y, point.z});
        }
    } else {
        // Copy all points
        for (const auto& point : *cloud) {
            points_vec.push_back({point.x, point.y, point.z});
        }
        
        // Pad with repeated points if needed
        while (static_cast<int>(points_vec.size()) < target_points) {
            int random_idx = rand() % cloud->size();
            const auto& point = (*cloud)[random_idx];
            points_vec.push_back({point.x, point.y, point.z});
        }
    }
    
    // Convert to tensor [N, 3]
    torch::Tensor points_tensor = torch::zeros({target_points, 3});
    for (int i = 0; i < target_points; ++i) {
        points_tensor[i][0] = points_vec[i][0];
        points_tensor[i][1] = points_vec[i][1];
        points_tensor[i][2] = points_vec[i][2];
    }
    
    // Normalize to unit sphere
    points_tensor = normalizePointCloud(points_tensor);
    
    // Transpose to [3, N] format expected by PointNet
    points_tensor = points_tensor.transpose(0, 1);
    
    return points_tensor;
}

torch::Tensor PointNetInference::normalizePointCloud(const torch::Tensor& points) {
    // Center the point cloud
    torch::Tensor centroid = torch::mean(points, /*dim=*/0, /*keepdim=*/true);
    torch::Tensor centered = points - centroid;
    
    // Scale to unit sphere
    torch::Tensor distances = torch::norm(centered, /*p=*/2, /*dim=*/1);
    torch::Tensor max_dist = torch::max(distances);
    
    if (max_dist.item<float>() > 0) {
        centered = centered / max_dist;
    }
    
    return centered;
}
