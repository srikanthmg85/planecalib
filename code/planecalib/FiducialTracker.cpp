/*
 * PoseTracker.cpp
 *
 *  Created on: 9.2.2014
 *      Author: dan
 */

#include "FiducialTracker.h"

#include <limits>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Profiler.h"
#include "cvutils.h"
#include "log.h"
#include "HomographyEstimation.h"
#include "PnpEstimation.h"
#include "flags.h"

namespace planecalib {

FiducialTracker::~FiducialTracker()
{

}

void FiducialTracker::init(const Eigen::Vector2i &imageSize, int octaveCount)
{
	mImageSize = imageSize;
	mOctaveCount = octaveCount;

	//Model refiners
	mHomographyEstimator.reset(new HomographyEstimation());
}

const Eigen::Matrix3fr &FiducialTracker::getCurrentPose2D() const
{
	return mPose2D;
}

void FiducialTracker::resetTracking(Map *map, const Eigen::Matrix3fr &initialPose)
{
	mMap = map;
	mLastPose2D = mPose2D = initialPose;
	mIsLost = true;

	//Forget all previous matches
	mLastFrame.reset(NULL);
	mFrame.reset(NULL);

}

bool FiducialTracker::trackFrame(double timestamp, const cv::Mat &imageColor, const std::map<std::pair<int,int>,Eigen::Vector2f> &features)
{
	ProfileSection s("trackFrame");
    mImageColor = imageColor;

	//Save old data
	mLastPose2D = mPose2D;
	mLastFrame.reset(NULL);
	if (mFrame)
	{
		mLastFrame = std::move(mFrame);

		//Remove outliers
		std::remove_if(mLastFrame->getMatches().begin(), mLastFrame->getMatches().end(),
			[](FeatureMatch &match){ return !match.getReprojectionErrors().isInlier; });
	}

	//Reset new frame data
	mFrame.reset(new TrackingFrame());
	mFrame->initImageData(imageColor, mCamera, features);
	mFrame->setTimestamp(timestamp);

	//Matches
	findMatches();

	//Estimate pose
	mIsLost = trackFrameHomography();

	//Build match map
	mFrame->createMatchMap();

	return !mIsLost;
}

void FiducialTracker::findMatches()
{
	//Get features in view
	auto &features = mMap->getFeatures();

	if (features.empty())
		return;

	const int octave = 0;
	const int scale = 1 << octave;

	auto &imgKeypoints = mFrame->getWarpedKeypoints(octave);
	auto &fiducialCoords = mFrame->getFiducialCoords(octave);

    MYAPP_LOG << "features.size() = " << features.size() << std::endl;
    MYAPP_LOG << "imgKeypoints.size() = " << imgKeypoints.size() << std::endl;

	for (int i=0; i<features.size() ; i++)
	{
        auto &feature = *features[i];
        auto &refMeasurement = *feature.getMeasurements()[0];
        auto refFiducialPos = refMeasurement.getFiducialCoords();
        auto refImgPos = refMeasurement.getPosition();

        for(int j=0;j<imgKeypoints.size();j++)
        {
            auto currFiducialPos = fiducialCoords[j];
            if(refFiducialPos == currFiducialPos)
            {
                mFrame->getMatches().push_back(FeatureMatch(&refMeasurement, octave, imgKeypoints[j], Eigen::Vector2f{imgKeypoints[j].pt.x,imgKeypoints[j].pt.y}, 0));
            }
        }
	}

	MYAPP_LOG << "Matches = " << mFrame->getMatches().size() << "\n";
}

bool FiducialTracker::trackFrameHomography()
{
	const int kMinInlierCount = 20;
	ProfileSection s("trackFrameHomography");

	int matchCount = mFrame->getMatches().size();

	//Homography
	if (matchCount < kMinInlierCount)
	{
		MYAPP_LOG << "Lost in 2D, only " << matchCount << " matches\n";
		return true;
	}

	//Create cv vectors
	std::vector<Eigen::Vector2f> refPoints, imgPoints;
	std::vector<float> scales;
	for (auto &match : mFrame->getMatches())
	{
		refPoints.push_back(match.getFeature().getPosition());
		// imgPoints.push_back(mCamera.unprojectToScaleSpace(match.getPosition()));
        imgPoints.push_back(match.getPosition());
        // MYAPP_LOG << "Tracking : " << refPoints.back() << " " << imgPoints.back() << std::endl;
		scales.push_back((float)(1<<match.getOctave()));
	}

	//Eigen::Matrix<uchar,Eigen::Dynamic,1> mask(refPoints.size());
	//cv::Mat1b mask_cv(refPoints.size(), 1, mask.data());

	Eigen::Matrix3fr H;

	{
		ProfileSection s("homographyRansac");

		HomographyRansac ransac;
		ransac.setParams(3, 10, 100, (int)(0.99f * matchCount));
		ransac.setData(&refPoints, &imgPoints, &scales);
		ransac.doRansac();
		H = ransac.getBestModel().cast<float>();
		MYAPP_LOG << "Homography ransac: inliers=" << ransac.getBestInlierCount() << "/" << matchCount << "\n";
	}

	//Refine
	HomographyEstimation hest;
	std::vector<bool> inliersVec;
	{
		ProfileSection s("refineHomography");
		H = hest.estimateCeres(H, imgPoints, refPoints, scales, 2.5, inliersVec);
	}

    //std::vector<FeatureMatch> goodMatches;
	//int inlierCountBefore = mask.sum();
	mMatchInlierCount = 0;
	mMatchInlierCountByOctave.resize(mOctaveCount);
    auto principalPoint = mCamera.getPrincipalPoint();
	for (int i = 0; i<(int)matchCount; i++)
	{
		if (inliersVec[i])
		{
            cv::circle(mImageColor,cv::Point(imgPoints[i][0]+principalPoint[0],imgPoints[i][1]+principalPoint[1]),5,cv::Scalar(0,255,0),-1);
			mMatchInlierCount++;
			mMatchInlierCountByOctave[mFrame->getMatches()[i].getOctave()]++;
		}
        else
        {
            cv::circle(mImageColor,cv::Point(imgPoints[i][0]+principalPoint[0],imgPoints[i][1]+principalPoint[1]),5,cv::Scalar(0,0,255),-1);
        }
				//goodMatches.push_back(mMatches[i]);
	}

    cv::imshow("Tracking Inliers",mImageColor);
    cv::waitKey(100);

	//inlierCountAfter = goodMatches.size();
	//mMatches = std::move(goodMatches);
	//MYAPP_LOG << "Inliers before=" << inlierCountBefore << ", inliers after=" << inlierCountAfter << "\n";

	if (mMatchInlierCount > kMinInlierCount)
	{
		mPose2D = H;
		mIsLost = false;
	}
	else
	{
		MYAPP_LOG << "Lost in 2D, only " << mMatchInlierCount << " inliers\n";
		mIsLost = true;
	}

	//Eval
	for (auto &match : mFrame->getMatches())
	{
		match.getReprojectionErrors().isInlier = true;
	}

    MYAPP_LOG << "H = " << mPose2D << std::endl;

	//Use only optical if lost
	return mIsLost;
}

} /* namespace dtslam */
