#ifndef POINTNET_INFERENCE_H
#define POINTNET_INFERENCE_H

#include <torch/torch.h>
#include <torch/script.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <memory>

/**
 * @brief Class for PointNet inference on point clouds
 */
class PointNetInference {
public:
    /**
     * @brief Constructor
     * @param model_path Path to the TorchScript model file
     * @param device Device to run inference on ("cpu" or "cuda")
     */
    PointNetInference(const std::string& model_path, const std::string& device = "cuda");
    
    /**
     * @brief Destructor
     */
    ~PointNetInference();
    
    /**
     * @brief Initialize the model
     * @return True if successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Perform semantic segmentation on a point cloud
     * @param cloud Input point cloud
     * @param labels Output segmentation labels (0=background, 1=laptop)
     * @param probabilities Output class probabilities
     * @return True if successful, false otherwise
     */
    bool segmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          std::vector<int>& labels,
                          std::vector<std::vector<float>>& probabilities);
    
    /**
     * @brief Preprocess point cloud for inference
     * @param cloud Input point cloud
     * @param target_points Target number of points (default: 1024)
     * @return Preprocessed point cloud as tensor
     */
    torch::Tensor preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                     int target_points = 1024);
    
    /**
     * @brief Normalize point cloud to unit sphere
     * @param points Input points tensor
     * @return Normalized points tensor
     */
    torch::Tensor normalizePointCloud(const torch::Tensor& points);
    
    /**
     * @brief Check if model is loaded
     * @return True if model is loaded, false otherwise
     */
    bool isModelLoaded() const { return model_loaded_; }
    
private:
    std::string model_path_;
    std::string device_str_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    
    static constexpr int NUM_CLASSES = 2;
    static constexpr int DEFAULT_NUM_POINTS = 1024;
};

#endif // POINTNET_INFERENCE_H
