
#ifndef FIDUCIALCALIBSYSTEM_H_
#define FIDUCIALCALIBSYSTEM_H_

#include <memory>
#include <future>
#include "stdutils.h"
#include "Map.h"
#include "CameraModel.h"
#include "FiducialTracker.h"

struct Matches
{
    int fromId;
    int toId;
    std::vector<std::pair<std::pair<int,int>,std::pair<Eigen::Vector2f,Eigen::Vector2f>>> matches;
};

namespace planecalib
{
//class FiducialTracker;
class HomographyCalibration;
class FiducialCalibSystem
{
public:
	FiducialCalibSystem():mExpectedPixelNoiseStd(1), mUse3DGroundTruth(false), mFix3DPoints(false), mUseNormalizedConstraints(true), mSuccesfulTrackCount(0)
	{}
	~FiducialCalibSystem();
    void processImage(double timestamp, cv::Mat &imgColor, std::map<std::pair<int,int>,Eigen::Vector2f> &features);
    void processMeasurements(std::map<std::pair<int,int>,Eigen::Vector2f> &frameFeatures, cv::Mat &img);
    void init();
    bool init(double timestamp, cv::Mat &imgColor, std::map<std::pair<int,int>,Eigen::Vector2f> &features);
    void calibrate();
    void doFullBA();
    void doHomographyCalib(bool fixP0);
    void doHomographyBA();
    void doValidationBA();

protected:
	////////////////////////////////////////////////////////
	// Members

    Eigen::Vector2i mImageSize;
    std::vector<std::map<std::pair<int,int>,Eigen::Vector2f>> features;
    std::vector<Matches> frameMatches;
    void matchFeatures();
    void estimateHomographyFromMatches(cv::Mat &img);
    void createKeyframe();
    std::vector<Eigen::Matrix3fr> homographies;

    CameraModel mCamera;
	Eigen::Vector3f mNormal;
	std::unique_ptr<HomographyCalibration> mCalib;
    std::unique_ptr<Map> mMap;
    std::unique_ptr<FiducialTracker> mTracker;

    float mExpectedPixelNoiseStd;
    bool mUse3DGroundTruth;
    bool mFix3DPoints;
    bool mUseNormalizedConstraints;
    int mSuccesfulTrackCount;
};

}

#endif /* SLAMSYSTEM_H_ */
