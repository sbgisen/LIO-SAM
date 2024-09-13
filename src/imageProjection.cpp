#include "utility.h"
#include "lio_sam/cloud_info.h"

struct EIGEN_ALIGN16 VelodynePointXYZIRT
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time,
                                                                                                       time))

struct EIGEN_ALIGN16 VelodynePointXYZRGBIRT
{
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZRGBIRT, (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(
                                                              float, intensity, intensity)(float, time, time))

struct EIGEN_ALIGN16 OusterPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(
                                      uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, noise,
                                                                                                 noise)(uint32_t, range,
                                                                                                        range))

struct EIGEN_ALIGN16 OusterPointXYZRGBIRT
{
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZRGBIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, intensity, intensity)(
                                      uint32_t, t, t)(uint16_t, reflectivity,
                                                      reflectivity)(uint8_t, ring, ring)(uint16_t, noise,
                                                                                         noise)(uint32_t, range, range))

const int queueLength = 2000;

template <typename VELODYNE_POINT_TYPE, typename OUSTER_POINT_TYPE, typename POINT_TYPE>
class ImageProjection : public ParamServer
{
private:
  std::mutex imuLock;
  std::mutex odoLock;

  ros::Subscriber subLaserCloud;
  ros::Publisher pubLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;

  ros::Subscriber subImu;
  std::deque<sensor_msgs::Imu> imuQueue;

  ros::Subscriber subOdom;
  std::deque<nav_msgs::Odometry> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;
  sensor_msgs::PointCloud2 currentCloudMsg;

  double* imuTime = new double[queueLength];
  double* imuRotX = new double[queueLength];
  double* imuRotY = new double[queueLength];
  double* imuRotZ = new double[queueLength];

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  typename pcl::PointCloud<VELODYNE_POINT_TYPE>::Ptr laserCloudIn;
  typename pcl::PointCloud<OUSTER_POINT_TYPE>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;

  int deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  lio_sam::cloud_info cloudInfo;
  double timeScanCur;
  double timeScanEnd;
  std_msgs::Header cloudHeader;

  vector<int> columnIdnCountVec;

public:
  ImageProjection() : deskewFlag(0)
  {
    subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this,
                                            ros::TransportHints().tcpNoDelay());
    subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &ImageProjection::odometryHandler,
                                               this, ros::TransportHints().tcpNoDelay());
    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this,
                                                           ros::TransportHints().tcpNoDelay());

    pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
    pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);

    allocateMemory();
    resetParameters();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  void allocateMemory()
  {
    laserCloudIn.reset(new pcl::PointCloud<VELODYNE_POINT_TYPE>());
    tmpOusterCloudIn.reset(new pcl::PointCloud<OUSTER_POINT_TYPE>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN * Horizon_SCAN);

    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    resetParameters();
  }

  void resetParameters()
  {
    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    imuPointerCur = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;

    for (int i = 0; i < queueLength; ++i)
    {
      imuTime[i] = 0;
      imuRotX[i] = 0;
      imuRotY[i] = 0;
      imuRotZ[i] = 0;
    }

    columnIdnCountVec.assign(N_SCAN, 0);
  }

  ~ImageProjection()
  {
  }

  void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
  {
    sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

    std::lock_guard<std::mutex> lock1(imuLock);
    imuQueue.push_back(thisImu);

    // debug IMU data
    // cout << std::setprecision(6);
    // cout << "IMU acc: " << endl;
    // cout << "x: " << thisImu.linear_acceleration.x <<
    //       ", y: " << thisImu.linear_acceleration.y <<
    //       ", z: " << thisImu.linear_acceleration.z << endl;
    // cout << "IMU gyro: " << endl;
    // cout << "x: " << thisImu.angular_velocity.x <<
    //       ", y: " << thisImu.angular_velocity.y <<
    //       ", z: " << thisImu.angular_velocity.z << endl;
    // double imuRoll, imuPitch, imuYaw;
    // tf::Quaternion orientation;
    // tf::quaternionMsgToTF(thisImu.orientation, orientation);
    // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
    // cout << "IMU roll pitch yaw: " << endl;
    // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
  }

  void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
  {
    std::lock_guard<std::mutex> lock2(odoLock);
    odomQueue.push_back(*odometryMsg);
  }

  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
  {
    if (!cachePointCloud(laserCloudMsg))
      return;

    if (!deskewInfo())
      return;

    projectPointCloud();

    cloudExtraction();

    publishClouds();

    resetParameters();
  }

  template <typename OUSTER_POINT_TYPE_>
  typename std::enable_if<std::is_same<OUSTER_POINT_TYPE_, OusterPointXYZRGBIRT>::value, void>::type
  copyTmpOusterCloudIn(int i)
  {
    laserCloudIn->points[i].x = tmpOusterCloudIn->points[i].x;
    laserCloudIn->points[i].y = tmpOusterCloudIn->points[i].y;
    laserCloudIn->points[i].z = tmpOusterCloudIn->points[i].z;
    laserCloudIn->points[i].b = tmpOusterCloudIn->points[i].b;
    laserCloudIn->points[i].g = tmpOusterCloudIn->points[i].g;
    laserCloudIn->points[i].r = tmpOusterCloudIn->points[i].r;
    laserCloudIn->points[i].intensity = tmpOusterCloudIn->points[i].intensity;
    laserCloudIn->points[i].ring = tmpOusterCloudIn->points[i].ring;
    laserCloudIn->points[i].time = tmpOusterCloudIn->points[i].t * 1e-9f;
  }

  template <typename OUSTER_POINT_TYPE_>
  typename std::enable_if<std::is_same<OUSTER_POINT_TYPE_, OusterPointXYZIRT>::value, void>::type
  copyTmpOusterCloudIn(int i)
  {
    laserCloudIn->points[i].x = tmpOusterCloudIn->points[i].x;
    laserCloudIn->points[i].y = tmpOusterCloudIn->points[i].y;
    laserCloudIn->points[i].z = tmpOusterCloudIn->points[i].z;
    laserCloudIn->points[i].intensity = tmpOusterCloudIn->points[i].intensity;
    laserCloudIn->points[i].ring = tmpOusterCloudIn->points[i].ring;
    laserCloudIn->points[i].time = tmpOusterCloudIn->points[i].t * 1e-9f;
  }

  bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
  {
    // cache point cloud
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2)
      return false;

    // convert cloud
    currentCloudMsg = std::move(cloudQueue.front());
    cloudQueue.pop_front();
    if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
    {
      pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
    }
    else if (sensor == SensorType::OUSTER)
    {
      // Convert to Velodyne format
      pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
      laserCloudIn->points.resize(tmpOusterCloudIn->size());
      laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
      for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
      {
        copyTmpOusterCloudIn<OUSTER_POINT_TYPE>(i);
      }
    }
    else
    {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
    }

    // get timestamp
    cloudHeader = currentCloudMsg.header;
    timeScanCur = cloudHeader.stamp.toSec();
    timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    // check dense flag
    if (laserCloudIn->is_dense == false)
    {
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }

    // check ring channel
    // static int ringFlag = 0;
    // if (ringFlag == 0)
    // {
    //   ringFlag = -1;
    //   for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
    //   {
    //     if (currentCloudMsg.fields[i].name == "ring")
    //     {
    //       ringFlag = 1;
    //       break;
    //     }
    //   }
    //   if (ringFlag == -1)
    //   {
    //     ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
    //     ros::shutdown();
    //   }
    // }

    // check point time
    if (deskewFlag == 0)
    {
      deskewFlag = -1;
      for (auto& field : currentCloudMsg.fields)
      {
        if (field.name == "time" || field.name == "t")
        {
          deskewFlag = 1;
          break;
        }
      }
      if (deskewFlag == -1)
        ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
    }

    return true;
  }

  bool deskewInfo()
  {
    std::lock_guard<std::mutex> lock1(imuLock);
    std::lock_guard<std::mutex> lock2(odoLock);

    // make sure IMU data available for the scan
    if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur ||
        imuQueue.back().header.stamp.toSec() < timeScanEnd)
    {
      ROS_DEBUG("Waiting for IMU data ...");
      return false;
    }

    imuDeskewInfo();

    odomDeskewInfo();

    return true;
  }

  void imuDeskewInfo()
  {
    cloudInfo.imuAvailable = false;

    while (!imuQueue.empty())
    {
      if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
        imuQueue.pop_front();
      else
        break;
    }

    if (imuQueue.empty())
      return;

    imuPointerCur = 0;

    for (int i = 0; i < (int)imuQueue.size(); ++i)
    {
      sensor_msgs::Imu thisImuMsg = imuQueue[i];
      double currentImuTime = thisImuMsg.header.stamp.toSec();

      // get roll, pitch, and yaw estimation for this scan
      if (currentImuTime <= timeScanCur)
        imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

      if (currentImuTime > timeScanEnd + 0.01)
        break;

      if (imuPointerCur == 0)
      {
        imuRotX[0] = 0;
        imuRotY[0] = 0;
        imuRotZ[0] = 0;
        imuTime[0] = currentImuTime;
        ++imuPointerCur;
        continue;
      }

      // get angular velocity
      double angular_x, angular_y, angular_z;
      imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

      // integrate rotation
      double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
      imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
      imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
      imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
      imuTime[imuPointerCur] = currentImuTime;
      ++imuPointerCur;
    }

    --imuPointerCur;

    if (imuPointerCur <= 0)
      return;

    cloudInfo.imuAvailable = true;
  }

  void odomDeskewInfo()
  {
    cloudInfo.odomAvailable = false;

    while (!odomQueue.empty())
    {
      if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
        odomQueue.pop_front();
      else
        break;
    }

    if (odomQueue.empty())
      return;

    if (odomQueue.front().header.stamp.toSec() > timeScanCur)
      return;

    // get start odometry at the beinning of the scan
    nav_msgs::Odometry startOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i)
    {
      startOdomMsg = odomQueue[i];

      if (ROS_TIME(&startOdomMsg) < timeScanCur)
        continue;
      else
        break;
    }

    tf::Quaternion orientation;
    tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

    double roll, pitch, yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // Initial guess used in mapOptimization
    cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
    cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
    cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
    cloudInfo.initialGuessRoll = roll;
    cloudInfo.initialGuessPitch = pitch;
    cloudInfo.initialGuessYaw = yaw;

    cloudInfo.odomAvailable = true;

    // get end odometry at the end of the scan
    odomDeskewFlag = false;

    if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
      return;

    nav_msgs::Odometry endOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i)
    {
      endOdomMsg = odomQueue[i];

      if (ROS_TIME(&endOdomMsg) < timeScanEnd)
        continue;
      else
        break;
    }

    if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
      return;

    Eigen::Affine3f transBegin =
        pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
                               startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
                                                      endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

    float rollIncre, pitchIncre, yawIncre;
    pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

    odomDeskewFlag = true;
  }

  void findRotation(double pointTime, float* rotXCur, float* rotYCur, float* rotZCur)
  {
    *rotXCur = 0;
    *rotYCur = 0;
    *rotZCur = 0;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur)
    {
      if (pointTime < imuTime[imuPointerFront])
        break;
      ++imuPointerFront;
    }

    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
    {
      *rotXCur = imuRotX[imuPointerFront];
      *rotYCur = imuRotY[imuPointerFront];
      *rotZCur = imuRotZ[imuPointerFront];
    }
    else
    {
      int imuPointerBack = imuPointerFront - 1;
      double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
      *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
      *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
    }
  }

  void findPosition(double relTime, float* posXCur, float* posYCur, float* posZCur)
  {
    *posXCur = 0;
    *posYCur = 0;
    *posZCur = 0;

    // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus
    // code below is commented.

    // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
    //     return;

    // float ratio = relTime / (timeScanEnd - timeScanCur);

    // *posXCur = ratio * odomIncreX;
    // *posYCur = ratio * odomIncreY;
    // *posZCur = ratio * odomIncreZ;
  }

  PointType deskewPoint(PointType* point, double relTime)
  {
    if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
      return *point;

    double pointTime = timeScanCur + relTime;

    float rotXCur, rotYCur, rotZCur;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur, posYCur, posZCur;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    if (firstPointFlag == true)
    {
      transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
      firstPointFlag = false;
    }

    // transform points to start
    Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBt = transStartInverse * transFinal;

    PointType newPoint;
    newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
    newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
    newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
    newPoint.intensity = point->intensity;
    newPoint.b = point->b;
    newPoint.g = point->g;
    newPoint.r = point->r;

    return newPoint;
  }

  template <typename POINT_TYPE_>
  typename std::enable_if<std::is_same<POINT_TYPE_, PointType>::value, pcl::PointCloud<PointType>::Ptr>::type
  convertPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
      const auto& pointFrom = cloudIn->points[i];
      cloudOut->points[i].x = pointFrom.x;
      cloudOut->points[i].y = pointFrom.y;
      cloudOut->points[i].z = pointFrom.z;
      cloudOut->points[i].b = pointFrom.b;
      cloudOut->points[i].g = pointFrom.g;
      cloudOut->points[i].r = pointFrom.r;
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  template <typename POINT_TYPE_>
  typename std::enable_if<std::is_same<POINT_TYPE_, pcl::PointXYZI>::value, pcl::PointCloud<pcl::PointXYZI>::Ptr>::type
  convertPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZI>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
      const auto& pointFrom = cloudIn->points[i];
      cloudOut->points[i].x = pointFrom.x;
      cloudOut->points[i].y = pointFrom.y;
      cloudOut->points[i].z = pointFrom.z;
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  template <typename VELODYNE_POINT_TYPE_>
  typename std::enable_if<std::is_same<VELODYNE_POINT_TYPE_, VelodynePointXYZRGBIRT>::value, PointType>::type
  copyLaserCloudIn(PointType thisPoint, int i)
  {
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;
    thisPoint.b = laserCloudIn->points[i].b;
    thisPoint.g = laserCloudIn->points[i].g;
    thisPoint.r = laserCloudIn->points[i].r;
    thisPoint.intensity = laserCloudIn->points[i].intensity;

    return thisPoint;
  }

  template <typename VELODYNE_POINT_TYPE_>
  typename std::enable_if<std::is_same<VELODYNE_POINT_TYPE_, VelodynePointXYZIRT>::value, PointType>::type
  copyLaserCloudIn(PointType thisPoint, int i)
  {
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;
    thisPoint.intensity = laserCloudIn->points[i].intensity;

    return thisPoint;
  }

  void projectPointCloud()
  {
    int cloudSize = laserCloudIn->points.size();
    // range image projection
    for (int i = 0; i < cloudSize; ++i)
    {
      PointType thisPoint;
      thisPoint = copyLaserCloudIn<VELODYNE_POINT_TYPE>(thisPoint, i);

      float range = pointDistance(thisPoint);
      if (range < lidarMinRange || range > lidarMaxRange)
        continue;

      int rowIdn = laserCloudIn->points[i].ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN)
        continue;

      if (rowIdn % downsampleRate != 0)
        continue;

      int columnIdn = -1;
      if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
      {
        float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
        static float ang_res_x = 360.0 / float(Horizon_SCAN);
        columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
        if (columnIdn >= Horizon_SCAN)
          columnIdn -= Horizon_SCAN;
      }
      else if (sensor == SensorType::LIVOX)
      {
        columnIdn = columnIdnCountVec[rowIdn];
        columnIdnCountVec[rowIdn] += 1;
      }

      if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
        continue;

      if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
        continue;

      thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

      rangeMat.at<float>(rowIdn, columnIdn) = range;

      int index = columnIdn + rowIdn * Horizon_SCAN;
      fullCloud->points[index] = thisPoint;
    }
  }

  void cloudExtraction()
  {
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i)
    {
      cloudInfo.startRingIndex[i] = count - 1 + 5;

      for (int j = 0; j < Horizon_SCAN; ++j)
      {
        if (rangeMat.at<float>(i, j) != FLT_MAX)
        {
          // mark the points' column index for marking occlusion later
          cloudInfo.pointColInd[count] = j;
          // save range info
          cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
          // save extracted cloud
          extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
          // size of extracted cloud
          ++count;
        }
      }
      cloudInfo.endRingIndex[i] = count - 1 - 5;
    }
  }

  void publishClouds()
  {
    cloudInfo.header = cloudHeader;
    typename pcl::PointCloud<POINT_TYPE>::Ptr extractedCloud_(new pcl::PointCloud<POINT_TYPE>());
    *extractedCloud_ = *convertPointCloud<POINT_TYPE>(extractedCloud);
    publishCloud(pubExtractedCloud, extractedCloud_, cloudHeader.stamp, lidarFrame);
    sensor_msgs::PointCloud2 tempDeskewed;
    pcl::toROSMsg(*extractedCloud, tempDeskewed);
    tempDeskewed.header.stamp = cloudHeader.stamp;
    tempDeskewed.header.frame_id = lidarFrame;
    cloudInfo.cloud_deskewed = tempDeskewed;
    pubLaserCloudInfo.publish(cloudInfo);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lio_sam");

  ParamServer PS;

  if (PS.useRGB)
  {
    ImageProjection<VelodynePointXYZRGBIRT, OusterPointXYZRGBIRT, PointType> IP;
    ROS_INFO("\033[1;32m----> Image Projection Started with RGB.\033[0m");
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
  }
  else
  {
    ImageProjection<VelodynePointXYZIRT, OusterPointXYZIRT, pcl::PointXYZI> IP;
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
  }

  return 0;
}
