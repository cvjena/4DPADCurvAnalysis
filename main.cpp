// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

// Literature:
// [1] Thümmel, Martin, and Sickert, Sven, and Joachim Denzler. "Facial Behavior Analysis using 4D Curvature Statistics for Presentation Attack Detection." arXiv preprint https://arxiv.org/pdf/1910.06056.pdf.
// [2] Dmytro Derkach, et al. "Head Pose Estimation Based on 3-D Facial Landmarks Localization and Regression". FG'2017

#include "visualization.hpp"
#include "matplotlibcpp.h"
#include <chrono>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <fstream>
#include <igl/procrustes.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/impl/search.hpp>
#include <stack>
#include <pcl/io/ply_io.h>
#include "rs_frame_image.h" // wrapper from rs2::video_frame to dlib color image.

using namespace cv;
using namespace std;
using namespace rs2;
namespace plt = matplotlibcpp;

#define numStripes 128 // number of stripes and elements along each stripe
#define windowSize 720

bool stopRecord = false;
vector<double> correlationSeries; // contains the max. cross-correlation values over time
vector<double> timeSeries; // contains the corresponding time points
int numFrame   = 0;
float duration = 0;

// Struct for managing the rotation of the pointcloud view
struct state {
  state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0), ml(false), mr(false), offset_z(0.15), offset_x(0), offset_y(0) {}
  double yaw, pitch, last_x, last_y, offset_x, offset_y, offset_z;
  bool ml, mr;
};

// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window &app, state &app_state) {
  app.on_left_mouse  = [&](bool pressed) { app_state.ml = pressed; };
  app.on_right_mouse = [&](bool pressed) { app_state.mr = pressed; };

  app.on_mouse_scroll = [&](double xoffset, double yoffset) { app_state.offset_z += yoffset * 0.05; };

  app.on_mouse_move = [&](double x, double y) {
    if (app_state.ml) {
      app_state.yaw -= (x - app_state.last_x);
      app_state.yaw = max(app_state.yaw, -120.0);
      app_state.yaw = min(app_state.yaw, +120.0);
      app_state.pitch += (y - app_state.last_y);
      app_state.pitch = max(app_state.pitch, -80.0);
      app_state.pitch = min(app_state.pitch, +80.0);
    }
    if (app_state.mr) {
      app_state.offset_y += (y - app_state.last_y) * 0.001;
      app_state.offset_x += (x - app_state.last_x) * 0.001;
    }
    app_state.last_x = x;
    app_state.last_y = y;
  };

  app.on_key_release = [&](int key) {
    if (key == 256) { // Escape
      stopRecord = true;
    } else if (key == 82) { //'r' = reset all time series and begin with the first question
      app_state.yaw = app_state.pitch = 0;
      app_state.offset_x = app_state.offset_y = 0;
      app_state.offset_z                      = 0.15;
      correlationSeries.clear();
      timeSeries.clear();
      numFrame = 0;
      duration = 0;
    }
  };
}

// colorbar for visualizing the curvature values in the image representation from GoogleAI (https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html)
float turbo_srgb_floats[256][3] = {{0.18995, 0.07176, 0.23217}, {0.19483, 0.08339, 0.26149}, {0.19956, 0.09498, 0.29024}, {0.20415, 0.10652, 0.31844}, {0.20860, 0.11802, 0.34607}, {0.21291, 0.12947, 0.37314}, {0.21708, 0.14087, 0.39964}, {0.22111, 0.15223, 0.42558}, {0.22500, 0.16354, 0.45096}, {0.22875, 0.17481, 0.47578}, {0.23236, 0.18603, 0.50004}, {0.23582, 0.19720, 0.52373}, {0.23915, 0.20833, 0.54686}, {0.24234, 0.21941, 0.56942}, {0.24539, 0.23044, 0.59142}, {0.24830, 0.24143, 0.61286}, {0.25107, 0.25237, 0.63374}, {0.25369, 0.26327, 0.65406}, {0.25618, 0.27412, 0.67381}, {0.25853, 0.28492, 0.69300}, {0.26074, 0.29568, 0.71162}, {0.26280, 0.30639, 0.72968}, {0.26473, 0.31706, 0.74718}, {0.26652, 0.32768, 0.76412}, {0.26816, 0.33825, 0.78050}, {0.26967, 0.34878, 0.79631}, {0.27103, 0.35926, 0.81156}, {0.27226, 0.36970, 0.82624}, {0.27334, 0.38008, 0.84037},
                                   {0.27429, 0.39043, 0.85393}, {0.27509, 0.40072, 0.86692}, {0.27576, 0.41097, 0.87936}, {0.27628, 0.42118, 0.89123}, {0.27667, 0.43134, 0.90254}, {0.27691, 0.44145, 0.91328}, {0.27701, 0.45152, 0.92347}, {0.27698, 0.46153, 0.93309}, {0.27680, 0.47151, 0.94214}, {0.27648, 0.48144, 0.95064}, {0.27603, 0.49132, 0.95857}, {0.27543, 0.50115, 0.96594}, {0.27469, 0.51094, 0.97275}, {0.27381, 0.52069, 0.97899}, {0.27273, 0.53040, 0.98461}, {0.27106, 0.54015, 0.98930}, {0.26878, 0.54995, 0.99303}, {0.26592, 0.55979, 0.99583}, {0.26252, 0.56967, 0.99773}, {0.25862, 0.57958, 0.99876}, {0.25425, 0.58950, 0.99896}, {0.24946, 0.59943, 0.99835}, {0.24427, 0.60937, 0.99697}, {0.23874, 0.61931, 0.99485}, {0.23288, 0.62923, 0.99202}, {0.22676, 0.63913, 0.98851}, {0.22039, 0.64901, 0.98436}, {0.21382, 0.65886, 0.97959}, {0.20708, 0.66866, 0.97423},
                                   {0.20021, 0.67842, 0.96833}, {0.19326, 0.68812, 0.96190}, {0.18625, 0.69775, 0.95498}, {0.17923, 0.70732, 0.94761}, {0.17223, 0.71680, 0.93981}, {0.16529, 0.72620, 0.93161}, {0.15844, 0.73551, 0.92305}, {0.15173, 0.74472, 0.91416}, {0.14519, 0.75381, 0.90496}, {0.13886, 0.76279, 0.89550}, {0.13278, 0.77165, 0.88580}, {0.12698, 0.78037, 0.87590}, {0.12151, 0.78896, 0.86581}, {0.11639, 0.79740, 0.85559}, {0.11167, 0.80569, 0.84525}, {0.10738, 0.81381, 0.83484}, {0.10357, 0.82177, 0.82437}, {0.10026, 0.82955, 0.81389}, {0.09750, 0.83714, 0.80342}, {0.09532, 0.84455, 0.79299}, {0.09377, 0.85175, 0.78264}, {0.09287, 0.85875, 0.77240}, {0.09267, 0.86554, 0.76230}, {0.09320, 0.87211, 0.75237}, {0.09451, 0.87844, 0.74265}, {0.09662, 0.88454, 0.73316}, {0.09958, 0.89040, 0.72393}, {0.10342, 0.89600, 0.71500}, {0.10815, 0.90142, 0.70599},
                                   {0.11374, 0.90673, 0.69651}, {0.12014, 0.91193, 0.68660}, {0.12733, 0.91701, 0.67627}, {0.13526, 0.92197, 0.66556}, {0.14391, 0.92680, 0.65448}, {0.15323, 0.93151, 0.64308}, {0.16319, 0.93609, 0.63137}, {0.17377, 0.94053, 0.61938}, {0.18491, 0.94484, 0.60713}, {0.19659, 0.94901, 0.59466}, {0.20877, 0.95304, 0.58199}, {0.22142, 0.95692, 0.56914}, {0.23449, 0.96065, 0.55614}, {0.24797, 0.96423, 0.54303}, {0.26180, 0.96765, 0.52981}, {0.27597, 0.97092, 0.51653}, {0.29042, 0.97403, 0.50321}, {0.30513, 0.97697, 0.48987}, {0.32006, 0.97974, 0.47654}, {0.33517, 0.98234, 0.46325}, {0.35043, 0.98477, 0.45002}, {0.36581, 0.98702, 0.43688}, {0.38127, 0.98909, 0.42386}, {0.39678, 0.99098, 0.41098}, {0.41229, 0.99268, 0.39826}, {0.42778, 0.99419, 0.38575}, {0.44321, 0.99551, 0.37345}, {0.45854, 0.99663, 0.36140}, {0.47375, 0.99755, 0.34963},
                                   {0.48879, 0.99828, 0.33816}, {0.50362, 0.99879, 0.32701}, {0.51822, 0.99910, 0.31622}, {0.53255, 0.99919, 0.30581}, {0.54658, 0.99907, 0.29581}, {0.56026, 0.99873, 0.28623}, {0.57357, 0.99817, 0.27712}, {0.58646, 0.99739, 0.26849}, {0.59891, 0.99638, 0.26038}, {0.61088, 0.99514, 0.25280}, {0.62233, 0.99366, 0.24579}, {0.63323, 0.99195, 0.23937}, {0.64362, 0.98999, 0.23356}, {0.65394, 0.98775, 0.22835}, {0.66428, 0.98524, 0.22370}, {0.67462, 0.98246, 0.21960}, {0.68494, 0.97941, 0.21602}, {0.69525, 0.97610, 0.21294}, {0.70553, 0.97255, 0.21032}, {0.71577, 0.96875, 0.20815}, {0.72596, 0.96470, 0.20640}, {0.73610, 0.96043, 0.20504}, {0.74617, 0.95593, 0.20406}, {0.75617, 0.95121, 0.20343}, {0.76608, 0.94627, 0.20311}, {0.77591, 0.94113, 0.20310}, {0.78563, 0.93579, 0.20336}, {0.79524, 0.93025, 0.20386}, {0.80473, 0.92452, 0.20459},
                                   {0.81410, 0.91861, 0.20552}, {0.82333, 0.91253, 0.20663}, {0.83241, 0.90627, 0.20788}, {0.84133, 0.89986, 0.20926}, {0.85010, 0.89328, 0.21074}, {0.85868, 0.88655, 0.21230}, {0.86709, 0.87968, 0.21391}, {0.87530, 0.87267, 0.21555}, {0.88331, 0.86553, 0.21719}, {0.89112, 0.85826, 0.21880}, {0.89870, 0.85087, 0.22038}, {0.90605, 0.84337, 0.22188}, {0.91317, 0.83576, 0.22328}, {0.92004, 0.82806, 0.22456}, {0.92666, 0.82025, 0.22570}, {0.93301, 0.81236, 0.22667}, {0.93909, 0.80439, 0.22744}, {0.94489, 0.79634, 0.22800}, {0.95039, 0.78823, 0.22831}, {0.95560, 0.78005, 0.22836}, {0.96049, 0.77181, 0.22811}, {0.96507, 0.76352, 0.22754}, {0.96931, 0.75519, 0.22663}, {0.97323, 0.74682, 0.22536}, {0.97679, 0.73842, 0.22369}, {0.98000, 0.73000, 0.22161}, {0.98289, 0.72140, 0.21918}, {0.98549, 0.71250, 0.21650}, {0.98781, 0.70330, 0.21358},
                                   {0.98986, 0.69382, 0.21043}, {0.99163, 0.68408, 0.20706}, {0.99314, 0.67408, 0.20348}, {0.99438, 0.66386, 0.19971}, {0.99535, 0.65341, 0.19577}, {0.99607, 0.64277, 0.19165}, {0.99654, 0.63193, 0.18738}, {0.99675, 0.62093, 0.18297}, {0.99672, 0.60977, 0.17842}, {0.99644, 0.59846, 0.17376}, {0.99593, 0.58703, 0.16899}, {0.99517, 0.57549, 0.16412}, {0.99419, 0.56386, 0.15918}, {0.99297, 0.55214, 0.15417}, {0.99153, 0.54036, 0.14910}, {0.98987, 0.52854, 0.14398}, {0.98799, 0.51667, 0.13883}, {0.98590, 0.50479, 0.13367}, {0.98360, 0.49291, 0.12849}, {0.98108, 0.48104, 0.12332}, {0.97837, 0.46920, 0.11817}, {0.97545, 0.45740, 0.11305}, {0.97234, 0.44565, 0.10797}, {0.96904, 0.43399, 0.10294}, {0.96555, 0.42241, 0.09798}, {0.96187, 0.41093, 0.09310}, {0.95801, 0.39958, 0.08831}, {0.95398, 0.38836, 0.08362}, {0.94977, 0.37729, 0.07905},
                                   {0.94538, 0.36638, 0.07461}, {0.94084, 0.35566, 0.07031}, {0.93612, 0.34513, 0.06616}, {0.93125, 0.33482, 0.06218}, {0.92623, 0.32473, 0.05837}, {0.92105, 0.31489, 0.05475}, {0.91572, 0.30530, 0.05134}, {0.91024, 0.29599, 0.04814}, {0.90463, 0.28696, 0.04516}, {0.89888, 0.27824, 0.04243}, {0.89298, 0.26981, 0.03993}, {0.88691, 0.26152, 0.03753}, {0.88066, 0.25334, 0.03521}, {0.87422, 0.24526, 0.03297}, {0.86760, 0.23730, 0.03082}, {0.86079, 0.22945, 0.02875}, {0.85380, 0.22170, 0.02677}, {0.84662, 0.21407, 0.02487}, {0.83926, 0.20654, 0.02305}, {0.83172, 0.19912, 0.02131}, {0.82399, 0.19182, 0.01966}, {0.81608, 0.18462, 0.01809}, {0.80799, 0.17753, 0.01660}, {0.79971, 0.17055, 0.01520}, {0.79125, 0.16368, 0.01387}, {0.78260, 0.15693, 0.01264}, {0.77377, 0.15028, 0.01148}, {0.76476, 0.14374, 0.01041}, {0.75556, 0.13731, 0.00942},
                                   {0.74617, 0.13098, 0.00851}, {0.73661, 0.12477, 0.00769}, {0.72686, 0.11867, 0.00695}, {0.71692, 0.11268, 0.00629}, {0.70680, 0.10680, 0.00571}, {0.69650, 0.10102, 0.00522}, {0.68602, 0.09536, 0.00481}, {0.67535, 0.08980, 0.00449}, {0.66449, 0.08436, 0.00424}, {0.65345, 0.07902, 0.00408}, {0.64223, 0.07380, 0.00401}, {0.63082, 0.06868, 0.00401}, {0.61923, 0.06367, 0.00410}, {0.60746, 0.05878, 0.00427}, {0.59550, 0.05399, 0.00453}, {0.58336, 0.04931, 0.00486}, {0.57103, 0.04474, 0.00529}, {0.55852, 0.04028, 0.00579}, {0.54583, 0.03593, 0.00638}, {0.53295, 0.03169, 0.00705}, {0.51989, 0.02756, 0.00780}, {0.50664, 0.02354, 0.00863}, {0.49321, 0.01963, 0.00955}, {0.47960, 0.01583, 0.01055}};

// compute the intersection over union by taking the intersection area and dividing it by the sum of the bbox areas minus the interesection area
float bbox_iou(dlib::rectangle box_a, dlib::rectangle box_b) {
  float interArea = box_a.intersect(box_b).area();
  return interArea / (box_a.area() + box_b.area() - interArea);
}

// take the previous bbox if no face was detected, take the first bbox for only one detection and take the bbox with the highest IOU with the previous bbox in case of multiple face detections
dlib::rectangle filterFaceDetections(vector<dlib::rectangle> face_detections, dlib::rectangle last_bbox) {
  if (face_detections.size() == 1 || last_bbox.area() == 0)
    return face_detections[0];
  else if (face_detections.size() == 0 && last_bbox.area() > 0)
    return last_bbox;
  else {
    int max_iou_index = 0;
    float max_iou     = 0;
    for (int i = 1; i < face_detections.size(); i++) {
      float iou = bbox_iou(face_detections[max_iou_index], face_detections[i]);
      if (iou > max_iou) {
        max_iou       = iou;
        max_iou_index = i;
      }
    }
    return face_detections[max_iou_index];
  }
}

// perform relativ padding of the bbox wrt its center
dlib::rectangle applyRelativePadding(dlib::rectangle bbox, float padding, int rows, int cols) {
  int paddingHeight = padding * bbox.height() * 2; // * 2 to preserve the forehead and the necklace
  int paddingWidth  = padding * bbox.width();
  dlib::point tl(max(bbox.left() - paddingWidth, 0l), max(bbox.top() - paddingHeight, 0l));
  dlib::point br(min(bbox.right() + paddingWidth, cols - 1l), min(bbox.bottom() + paddingHeight, rows - 1l));
  return dlib::rectangle(tl, br);
}

// Euler angle structure which is only used for temporal head pose smoothing
struct Pose {
  float yaw   = NAN;
  float pitch = NAN;
  float roll  = NAN;
};

// Estimate the head pose based on a subset of 3D landmark locations
// this function is only used for temporal head pose smoothing
// the code was converted from matlab and is taken from [2]
Pose estimateHeadPose(Matx<float, 68, 3> landmarks) {
  Mat points(0, 3, CV_32F);
  // consider only the inner/outer eye landmark indices, the outer mouth landmarks and the nasal bridge landmark for head pose estimation
  char indices[] = {36, 39, 42, 45, 27, 48, 54};
  // extract all not nan points of the subset
  for (float index : indices)
    if (!isnan(landmarks(index, 0)))
      points.push_back(Mat(landmarks.row(index)));
  Pose pose;
  if (points.rows > 0) {
    Mat mean;
    // calculate the center of the landmark subset and subtract the center from all landmarks
    reduce(points, mean, 0, REDUCE_AVG);
    for (int i = 0; i < points.rows; ++i)
      points.row(i) = points.row(i) - mean;
    // estimate the head pose from the direction of the first two eigen vectors
    // this is equivalent to fitting a plane into this subset as described in Fig. 2 in [1]
    PCA pca(points, Mat(), PCA::DATA_AS_ROW, 2);
    // calculate the normal vector of the fitted plane
    Matx13f n = pca.eigenvectors.row(0).cross(pca.eigenvectors.row(1));
    if (n(2) < 0)
      n *= -1;
    // obtain the yaw and pitch angles from the normal vector
    pose.yaw   = asin(n(0, 0));
    pose.pitch = -asin(n(0, 1));

    // calculate the direction of a line through both mean eye positions or the remaining outer/inner eye landmarks if some of them are nan
    Matx13f leftEye  = isnan(landmarks(36, 0)) ? landmarks.row(39) : (isnan(landmarks(39, 0)) ? landmarks.row(36) : (landmarks.row(36) + landmarks.row(39)) * 0.5);
    Matx13f rightEye = isnan(landmarks(42, 0)) ? landmarks.row(45) : (isnan(landmarks(45, 0)) ? landmarks.row(42) : (landmarks.row(42) + landmarks.row(45)) * 0.5);
    Matx13f line     = leftEye - rightEye;
    pose.roll        = atan(line(1) / line(0)); // obtain the roll angle from the line connecting both eyes according to Sec. A.2 in [1].
  }
  return pose;
}

// transform a rs2::points object to a PCL point cloud, center the nose tip, crop the face to all points inside a sphere with r=cropRadius and add all not nan indices to a list
pcl::PointCloud<pcl::PointXYZI>::Ptr extractCloud(const rs2::points &points, Eigen::MatrixXd &landmarks, float cropRadius, vector<int> &notNanIndices) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  auto sp         = points.get_profile().as<rs2::video_stream_profile>();
  // if the width and height are set, an PCL point cloud is called structured or organized and many algorithms for point cloud processing are highly parallized for this case
  cloud->width    = sp.width();
  cloud->height   = sp.height();
  cloud->is_dense = false; // means that it can contain nan's
  cloud->points.resize(points.size());
  auto ptr = points.get_vertices();
  Eigen::RowVectorXd noseTip3D(landmarks.row(30)); // the 3D landmark with index 30 corresponds to the nose tip
  // iterate over all points in both structured point clouds with the same width and height
  // each point in the PCL cloud is given by the pointer p and in the rs2 cloud by the pointer ptr
  for (auto &p : cloud->points) {
    // place the nose tip to the origin of the point cloud
    p.x                     = ptr->x - noseTip3D(0);
    p.y                     = ptr->y - noseTip3D(1);
    p.z                     = ptr->z - noseTip3D(2);
    // filter out all points whose distance to the nose tip exceeds a threshold
    float distanceToNoseTip = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
    if (ptr->z == 0 || distanceToNoseTip > cropRadius) { // ptr->z == 0 if undefined
      p.x = p.y = p.z = NAN;
    } else
      notNanIndices.push_back(ptr - points.get_vertices()); // store the index of the last added point
    ptr++;
  }
  // similary subtract the nose tip position from all landmark locations
  for (int i = 0; i < 68; i++)
    landmarks.row(i) -= noseTip3D;
  return cloud;
}

// for calculating the corresponding rotation matrix if the head pose was estimated using estimateHeadPose()
Matx33f eulerAnglesToRotationMatrix(Pose pose) {
  Matx33f R_x(1, 0, 0, 0, cos(pose.pitch), -sin(pose.pitch), 0, sin(pose.pitch), cos(pose.pitch));
  Matx33f R_y(cos(pose.yaw), 0, sin(pose.yaw), 0, 1, 0, -sin(pose.yaw), 0, cos(pose.yaw));
  Matx33f R_z(cos(pose.roll), -sin(pose.roll), 0, sin(pose.roll), cos(pose.roll), 0, 0, 0, 1);
  return R_y * R_x * R_z;
}

// calculate the 2D curvatures according to (3) in [1] based on the original point cloud but only for the points of the extracted stripes
void calcCurvature(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int *indices) {
  pcl::search::OrganizedNeighbor<pcl::PointXYZI>::Ptr tree(new pcl::search::OrganizedNeighbor<pcl::PointXYZI>);
  tree->setInputCloud(cloud); // set the input point cloud for calculation to the original point cloud
#pragma omp parallel for
  for (int stripeElement = 0; stripeElement < numStripes * numStripes; stripeElement++) {
    // iterate over all elements of all stripes
    int i = indices[stripeElement]; // i is the point index of an extracted point for the current stripe element
    if (i >= 0) {
      // find the 100 closest neighbors
      vector<int> neighbors;
      vector<float> distances;
      tree->nearestKSearch(cloud->points[i], 100, neighbors, distances);
      // calculate the mean and covariance over all neighbors
      Eigen::Vector4d xyz_centroid;
      Eigen::Matrix3d covariance_matrix;
      computeMeanAndCovarianceMatrix(*cloud, neighbors, covariance_matrix, xyz_centroid);
      // calculate the 2D curvature based on the trace of the covariance matrix according to (3) in [1]
      float curvature = covariance_matrix.trace();
      // normalize the range of the curvature values to [0, 1] based on the eigen value
      // this equals to fitting a plane to the neighbors using a PCA and measuring the curvature based on the length of the first principal vector or normal vector of the plane.
      // the length of this (unnormalized) normal vector is at maximum 1 for equally distributed points inside a unit cube
      Eigen::Vector3d::Scalar eigen_value;
      Eigen::Vector3d eigen_vector;
      pcl::eigen33(covariance_matrix, eigen_value, eigen_vector);
      if (curvature != 0)
        cloud->points[i].intensity = eigen_value / curvature;
    }
  }
}

// this code is taken from the pointcloud example of the librealsense
void draw_pointcloud(window &app, state &app_state, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int *indices) {
  glPopMatrix();
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  gluPerspective(60, app.width() / app.height(), 0.01f, 10.0f);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
  glTranslatef(app_state.offset_x, app_state.offset_y, (float)app_state.offset_z);
  glRotated(app_state.pitch, 1, 0, 0);
  glRotated(app_state.yaw, 0, 1, 0);
  glPointSize(3);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_POINTS);
  for (int stripeElement = 0; stripeElement < numStripes * numStripes; stripeElement++) {
    int i = indices[stripeElement];
    if (i >= 0) {
      pcl::PointXYZI &p = cloud->points[i];
      glVertex3fv(p.data);
      int index = std::max(0, std::min((int)(p.intensity * 30 * 255), 255));
      glColor3f(turbo_srgb_floats[index][0], turbo_srgb_floats[index][1], turbo_srgb_floats[index][2]);
    }
  }
  glEnd();
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glPopAttrib();
  glPushMatrix();
}

// the plotted window containing the color image, curvature image representation and correlation time series can be stored using this function
void screenshot(string filename) {
  GLubyte pixels[3 * windowSize * windowSize];
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, windowSize, windowSize, GL_BGR, GL_UNSIGNED_BYTE, pixels);
  Mat screenshot(windowSize, windowSize, CV_8UC3, pixels);
  flip(screenshot, screenshot, 0);
  imwrite(filename, screenshot);
}

// ask 8 questions for PAD with 3000 ms pause between each question
void addText(float duration) {
  if (std::round(duration / 3000) <= 1) {
    plt::text(1.0, 1.9, "Ihnen werden gleich einige Fragen gestellt, die Sie bitte mit Sprache beantworten.");
  } else if (std::round(duration / 3000) == 2) {
    plt::text(1.0, 1.9, "Befinden Sie sich in Deutschland?");
  } else if (std::round(duration / 3000) == 3) {
    plt::text(1.0, 1.9, "Wo befinden Sie sich gerade?");
  } else if (std::round(duration / 3000) == 4) {
    plt::text(1.0, 1.9, "Welcher Wochentag ist heute?");
  } else if (std::round(duration / 3000) == 5) {
    plt::text(1.0, 1.9, "Haben Sie heute schon Kaffee getrunken?");
  } else if (std::round(duration / 3000) == 6) {
    plt::text(1.0, 1.9, "Wie viele E-Mails haben Sie heute schon gelesen?");
  } else if (std::round(duration / 3000) == 7) {
    plt::text(1.0, 1.9, "Was ist der Grund Ihrer Reise?");
  } else if (std::round(duration / 3000) == 8) {
    plt::text(1.0, 1.9, "Wie lange bleiben Sie in der USA?");
  } else if (std::round(duration / 3000) == 9) {
    plt::text(1.0, 1.9, "Fuehren Sie Fruechte oder Obst mit sich?");
  } else if (std::round(duration / 3000) == 10) {
    plt::text(1.0, 1.9, "Fuehren Sie Fluessigkeiten mit sich?");
  }
}

// normalize the head pose by aligning the current 3D landmark locations with the landmark locations of the first 3D scan
void normalizeHeadPose(const Eigen::MatrixXd &sourceLandmarks, const Eigen::MatrixXd &targetLandmarks, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
  // calculate the distance between each landmark and the nose tip (landmark with index 30)
  float distancesSource[68], distancesTarget[68];
  for (int i = 0; i < sourceLandmarks.rows(); i++) {
    distancesSource[i] = (sourceLandmarks.row(i) - sourceLandmarks.row(30)).norm();
    distancesTarget[i] = (targetLandmarks.row(i) - targetLandmarks.row(30)).norm();
  }
  // remove all oulier landmarks whose distance is either nan or to far from the nose tip (> 11 cm) for both, the current and the last localized landmarks
  Eigen::Array<bool, Eigen::Dynamic, 1> notNanRows = !(sourceLandmarks.col(0).array().isNaN() || targetLandmarks.col(0).array().isNaN());
  for (int i = 0; i < sourceLandmarks.rows(); i++) {
    notNanRows(i) &= distancesSource[i] < 0.11 && distancesTarget[i] < 0.11;
  }
  // consider all landmarks without the outlier, the last 6 landmarks for the inner mouth, and the first 17 landmarks for the jaw
  // these landmarks are usually not accuratly localized
  int numNotNans = notNanRows.bottomRows<51>().topRows<51 - 6>().cast<int>().sum();
  Eigen::MatrixXd sourceLandmarksNanFree(numNotNans, 3);
  Eigen::MatrixXd targetLandmarksNanFree(numNotNans, 3);
  for (int i = 17, j = 0; i < sourceLandmarks.rows() - 6; i++) { // remove 6 inner mouth landmarks as they are very noisy
    if (notNanRows(i)) {
      sourceLandmarksNanFree.row(j) = sourceLandmarks.row(i);
      targetLandmarksNanFree.row(j) = targetLandmarks.row(i);
      j++;
    }
  }
  // apply Procrustes analysis to roughly align the detected landmarks with the last landmarks without scaling and reflections
  Eigen::MatrixXd R;
  Eigen::VectorXd t;
  double scale;
  igl::procrustes(sourceLandmarksNanFree, targetLandmarksNanFree, false, false, scale, R, t);

  // transform the source cloud using the obtained rotation matrix and translation vector
  // the requirement for transposing in place is due to different conventions from which expected side the matrix multiplication should be aplied and if the translation has to be applied before or after the rotation
  R.transposeInPlace();
  Eigen::Matrix4d T;
  T << R(0, 0), R(0, 1), R(0, 2), t(0), R(1, 0), R(1, 1), R(1, 2), t(1), R(2, 0), R(2, 1), R(2, 2), t(2), 0, 0, 0, 1;
  transformPointCloud(*cloud, *cloud, T);
}

int main(int argc, char *argv[]) {
  // initialize OpenGL visualization (see sample source code from librealsense for visualization)
  bool visualization = true;
  window app(windowSize, windowSize, "Radial Curves");
  state app_state;
  register_glfw_callbacks(app, app_state);

  // check if a RealSense is connected
  context ctx;
  device_list devices             = ctx.query_devices();
  device dev;
  config cfg;
  decimation_filter decimationFilter(2); // n x n median filter
  if (argc == 1 && devices.size() > 0) {
    // activate the advanced mode for the first connected RealSense, load a config-file and enable high spatiotemporal resolution
    dev = devices.front();
    string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
    if (!advanced_mode_dev.is_enabled())
      advanced_mode_dev.toggle_advanced_mode(true);

    auto sensor = dev.first<depth_sensor>();
    sensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET,rs2_rs400_visual_preset::RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
    ifstream t("../faceConfigOpt.json");
    string preset_json((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
    advanced_mode_dev.load_json(preset_json);
    cfg.enable_device(serial);
    // to allow for such a high spatial and temporal resolution, at least usb 3.0 must be used.
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  } else if (argc == 2) {
    // if no RealSense is connected,
    string inputFilePath = argv[1];
    cfg.enable_device_from_file(inputFilePath,true);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  } else {
    cerr << "Either connect a RealSense D435/D415 or pass the file path to a bag-file as an argument" << endl;
    return -1;
  }

  // either start the playback or the recording with the connected RealSense
  pipeline pipe;
  pipeline_profile profile;
  try {
    profile = pipe.start(cfg);
  } catch (Exception e) { // if the bag file was recorded with low resulution, the enabling of a high resolution would fail like for the file joachimSmiling.bag
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
    decimationFilter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 1); // equivalent to no filtering to obtain the same image resulution as for the upper case
    profile = pipe.start(cfg);
  }

  // a playback of a bag file runs automatically after loading and must be paused during background processing
  rs2::playback playback = profile.get_device().as<rs2::playback>();
  if (devices.size() == 0) {
    playback.pause();
  }

  // load the tools for face detection and landmark localization
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("/home/thuemmel/pretrainedModels/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  // create a binary mask with the same size as the curvature image to ignore all curvature values outside the mouth region later on
  // the mask contains the first and last 15 stripes around the vertical stripe through the mouth and ignores the first and last 32 curvature values along each stripe outside the mouth region
  // (realsense tweeks to allow for PAD)
  Mat mouthMask          = Mat::zeros(numStripes, numStripes, CV_8U);
  const int mouthStripes = 15;
  int mouthStripeIndices[mouthStripes * 2]; // store the selected stripe indices into a list
  for (int i = 0; i < mouthStripes; i++) {
    mouthMask.row(i).colRange(32 + i, numStripes - 32 - i)                 = 1;
    mouthMask.row(numStripes - 1 - i).colRange(32 + i, numStripes - 32 - i) = 1;
    mouthStripeIndices[i]                                                   = i;
    mouthStripeIndices[i + mouthStripes]                                    = numStripes - 1 - i;
  }

  // initialization for visualization and temporal filtering purposes
  bool firstFrame  = true;
  rs2::align align_to_depth(RS2_STREAM_DEPTH);
  dlib::rectangle last_bbox_cimg = dlib::rectangle();
  Matx<float, numStripes, numStripes> lastCurvatureImg;
  plt::figure_size(1000,1000);
  Eigen::MatrixXd firstLandmarks(68, 3);
  firstLandmarks.setZero();
  while (app && !stopRecord) {
    // extract the next frame either from the playback or connected device
    frameset frames;
    try {
      if (devices.size() == 0)
        playback.resume();
      frames = pipe.wait_for_frames();
      if (devices.size() == 0)
        playback.pause();
    } catch (const rs2::error &e) {
      cerr << e.get_failed_args() << e.get_failed_function() << e.get_type() << endl << e.what() << endl;
      break;
    }
    auto start                              = std::chrono::system_clock::now();
    // halve the image resolution to allow for realtime processing
    frames = decimationFilter.process(frames.get());
    //    frames            = temp_filter.process(frames); // uncomment for temporal depth map smoothing
    frames            = align_to_depth.process(frames); // transform the color image to the depth image
    video_frame color = frames.get_color_frame(); // obtain the registered color images
    rs_frame_image<dlib::rgb_pixel, RS2_FORMAT_RGB8> cimgDlib(color); // convert an rs2::video_frame to an dlib image without copying the data
    Mat cimg(color.get_height(), color.get_width(), CV_8UC3, (void *)color.get_data()); // convert an rs2::video_frame to an OpenCV image for visualization purposes (without copying)
    vector<dlib::rectangle> face_detections = detector(cimgDlib); // detect all faces
    if (face_detections.size() > 0) {
      // apply temporal filtering using bbox IOU if multiple or no faces were detected
      dlib::rectangle bbox_cimg         = filterFaceDetections(face_detections, last_bbox_cimg);
      last_bbox_cimg                    = dlib::rectangle(bbox_cimg);
      // locate the landmarks and pad the bboxes by 10 % to preserve the forehead and chin region
      dlib::full_object_detection shape = landmarkDetector(cimgDlib, bbox_cimg);
      bbox_cimg                         = applyRelativePadding(bbox_cimg, 0.1, cimg.rows, cimg.cols);

      // transform the depth map to a point cloud and extract the 3D coordinates for each 2D landmark
      pointcloud pc;
      depth_frame depth = frames.get_depth_frame();
      points points     = pc.calculate(depth);
      auto vertices     = points.get_vertices();
      // extract the 3D coordinates for each 2D landmark
      Eigen::MatrixXd landmarks(shape.num_parts(), 3);
      for (int i = 0; i < shape.num_parts(); i++) {
        dlib::point p     = shape.part(i);
        vertex landmark3D = vertices[p.y() * depth.get_width() + p.x()];
        landmarks(i, 0)   = landmark3D.x;
        landmarks(i, 1)   = landmark3D.y;
        landmarks(i, 2)   = landmark3D.z;
      }

      // uncomment for temporal filtering of the estimated head pose
      //      Pose pose                 = estimateHeadPose(landmarks);
      //      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      //        noseTip3D = 0.5 * noseTip3D + 0.5 * Point3f(landmarks(29, 0), landmarks(29, 1), landmarks(29, 2));
      //        q         = q.slerp(0.5, Eigen::AngleAxisf(-pose.roll, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(-pose.yaw, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(-pose.pitch, Eigen::Vector3f::UnitX()));

      // transform rs2::points to a PCL point cloud, center the nose tip, crop the face to all points inside a sphere with r=cropRadius and add all indices of points which are not nan to a list
      vector<int> notNanIndices;
      float cropRadius = 0.1;
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = extractCloud(points, landmarks, cropRadius, notNanIndices);

      // align the current 3D landmark locations with the landmark locations of the first 3D scan
      if (firstLandmarks.isZero(0)) {
        firstLandmarks = Eigen::MatrixXd(landmarks);
      } else
        normalizeHeadPose(landmarks, firstLandmarks, cloud);

      //######################### Calculation of Curvature of Radial Stripes (CORS) ################################
      // extract the radial stripes by storing the point indices to the 2D array indices
      // each row corresponds to one radial stripe and each column to one extracted point along each stripe
      // -1 is initialized for all elements and a -1 at [r, c] later on means that no point felt into the grid element given by c along the stripe given by r
      int indices[numStripes * numStripes];
      fill_n(indices, numStripes * numStripes, -1);
      Eigen::Matrix3f R_cumulated = Eigen::Matrix3f::Identity(3, 3); // the normal vector of each plane is obtained by repeatedly applying R to R_cumulated
      Eigen::Matrix3f R(Eigen::AngleAxisf(2 * M_PI / numStripes, Eigen::Vector3f::UnitZ())); // 2 * M_PI / numStripes is the angle between each stripe
      // maxPlaneDistance is used for creating an equidistant grid along the orthogonal projection of each radial stripe to the xy-plane
      // maxPlaneDistance contains the maximum distance between the orthogonal projection of each point and the nose tip
      // maxPlaneDistance is calculated from the maximum distance of any point to the nose tip r=cropRadius and 0.04 as an estimate of the maximum z-distance of the normalized face scan
      float maxPlaneDistance = sqrt(pow(cropRadius, 2) - pow(0.04, 2));
      for (int numStripe = 0; numStripe < numStripes; numStripe++) {
        vector<float> remainders(numStripes, INFINITY);
        // number of elements of the equidistant grid
        float delta = maxPlaneDistance / (numStripes - 1);
        // iterate over all points which are not nan
        for (int index : notNanIndices) {
          pcl::PointXYZI &p  = cloud->at(index);
          // intersect the point cloud with a plane whose normal vector is given by R_cumulated * Eigen::Vector3f::UnitX()
          // according to (1) in [1], the initial plane equals to the yz-plane where the y-axis points downwards and z-axis points into the cameras viewing directing
          float x_projection = p.getVector3fMap().dot(R_cumulated * Eigen::Vector3f::UnitX());
          float y_projection = p.getVector3fMap().dot(R_cumulated * Eigen::Vector3f::UnitY());
          // extract all points which are closer than \delta=1.5 mm according to (1) in [1]
          if (abs(x_projection) < 0.0015 && y_projection >= 0) {
            // assign a point to an element of the equidistant grid using y_projection which contains the distance between the nose tip and the orthogonal projection of the current point to the xy-plane
            int stripeIndex = ceil(y_projection / delta);
            if (stripeIndex < numStripes) {
              // if the point does not exceed the grid boundaries
              // assign the point to both neighboring grid elements if the point is closer to the equidistant grid position than the current assigned point
              float remainder = abs(x_projection);
              if (remainder < remainders[stripeIndex]) {
                remainders[stripeIndex]                     = remainder;
                indices[numStripe * numStripes + stripeIndex] = index;
              }
              stripeIndex = floor(y_projection / delta);
              if (remainder < remainders[stripeIndex]) {
                remainders[stripeIndex]                     = remainder;
                indices[numStripe * numStripes + stripeIndex] = index;
              }
            }
          }
        }
        // cumulate the rotations by a fraction of the full angle to obtain the rotation matrix for the next intersecting plane
        R_cumulated = R * R_cumulated;
      }
      // calculate the 2D curvatures according to (3) in [1] based on the original point cloud but only for the points of the extracted stripes
      calcCurvature(cloud, indices);
      
      // assign the curvature values to an image structure for visualization purposes
      Matx<float, numStripes, numStripes> curvatureImg;
      for (int i = 0; i < numStripes; i++) {
        for (int j = 0; j < numStripes; j++) {
          int index = indices[i * numStripes + j];
          if (index >= 0)
            curvatureImg(i, j) = cloud->points[index].intensity;
          else
            curvatureImg(i, j) = 0;
        }
      }
      if (!firstFrame) {
        if (visualization) {
          // plot the color image with the overlayed landmark locations and the curvature image representation
          plt::clf();
          plt::subplot(2, 2, 1);
          for (int i = 0; i < shape.num_parts(); i++)
            circle(cimg, Point2f(shape.part(i).x(), shape.part(i).y()), 1, Scalar(255, 0, 0), 1);
          plt::imshow(cimg(Rect(bbox_cimg.left(), bbox_cimg.top(), bbox_cimg.width(), bbox_cimg.height())).clone().data, bbox_cimg.height(), bbox_cimg.width(), 3);
          plt::axis("off");
          plt::subplot(2, 2, 2);
          plt::imshow((float *)Mat(curvatureImg).data, numStripes, numStripes, 1, {{"vmax", "0.1"}});
          plt::axis("off");
        }
        // apply the binary mouth mask for PAD
        curvatureImg               = curvatureImg.mul(mouthMask);
        double maxMouthCorrelation = 0;
        // only consider the stripes in the mouth region for PAD
        for (int i : mouthStripeIndices) {
          // calculate the temporal cross-correlation between the curvatures of each radial stripe according to (4) in [1]
          Mat correlation, curvatures;
          copyMakeBorder(curvatureImg.row(i), curvatures, 0, 0, numStripes / 2 - 1, numStripes / 2, BORDER_CONSTANT); // zero padding to remain the stripe size after correlation
          matchTemplate(curvatures / numStripes, lastCurvatureImg.row(i), correlation, TM_CCORR); // correlate the last curvature values with the current curvature values in the same stripe
          double maxCorrelation;
          minMaxLoc(correlation * 1000, NULL, &maxCorrelation); // extract the maximum cross-correlation for the current stripe
          if (maxCorrelation > maxMouthCorrelation)
            maxMouthCorrelation = maxCorrelation; // extract the maximum cross-correlation over all stripes
        }
        // store all max. cross-correlations to an vector for visualization purposes for 8 questions and 3000 ms pause between each question
        if (std::round(duration / 3000) >= 2 && std::round(duration / 3000) <= 10) {
          correlationSeries.push_back(maxMouthCorrelation);
          timeSeries.push_back(duration / 1000 - 4.5);
        }
        // plot the cross-correlation time series
        if (visualization) {
          plt::subplot(2, 1, 2);
          if (std::round(duration / 3000) >= 2)
            plt::plot(timeSeries, correlationSeries);
          addText(duration);
          //          plt::axis("off");
          plt::xlim(0, 27);
          plt::ylim(0.0, 2.0);
          //                plt::ylim(0.75, 2.25);
          plt::xlabel("Zeit / s");
          plt::ylabel("Kreuzkorrelation");
          plt::tight_layout();
          plt::subplots_adjust({{"bottom", 0.1}, {"left", 0.1}, {"right", 0.98}, {"top", 1}, {"wspace", 0}, {"hspace", 0}});
        }
        //        stringstream ss("");
        //        ss << "correlation_" << setw(3) << setfill('0') << numFrame << ".png";
        //        plt::save(ss.str());
      }
      lastCurvatureImg         = curvatureImg;
      firstFrame               = false;
      int elapsed_milliseconds = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
      //      cout << "elapsed time: " << elapsed_seconds << " ms\n";
      duration += elapsed_milliseconds;
      //      stringstream ss("");
      //      ss << "radialStripes_" << setw(3) << setfill('0') << numFrame << ".png";
      //      screenshot(ss.str());

      // perform PAD after all questions were answered
      if (std::round(duration / 3000) > 10) {
        //        cout << "elapsed time in avg: " << duration / numFrame << " ms\n";
        // calculate the variance of the time series
        double sq_sum = inner_product(correlationSeries.begin(), correlationSeries.end(), correlationSeries.begin(), 0.0);
        double sigma  = sqrt(sq_sum / correlationSeries.size() - pow(accumulate(correlationSeries.begin(), correlationSeries.end(), 0.0) / correlationSeries.size(), 2));
        stringstream ss("");
        // and decide between mask and genuine face based on the value of the variance
        if (sigma > 0.2)
          ss << "Echtes Gesicht";
        else
          ss << "Kein echtes Gesicht";
        ss << " (sigma = " << sigma << ")";
        plt::text(1.0, 1.9, ss.str());
      }
      numFrame++;
      // visualize the original point cloud using OpenGL
      if (visualization) {
        draw_pointcloud(app, app_state, cloud, indices);
        plt::draw();
        plt::pause(0.01);
      }
    }
  }
  return 0;
}
