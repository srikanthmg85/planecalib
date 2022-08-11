
#include "FiducialCalibSystem.h"
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "Keyframe.h"
#include "Profiler.h"
#include "FiducialTracker.h"
#include "HomographyCalibration.h"
#include "HomographyEstimation.h"
#include "BundleAdjuster.h"
#include "CalibratedBundleAdjuster.h"
#include "PnpEstimation.h"
//#include "CeresUtils.h"
#include "FeatureIndexer.h"
#include "flags.h"

#include <random>
using namespace cv;

namespace planecalib
{

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

FiducialCalibSystem::~FiducialCalibSystem()
{

}

void FiducialCalibSystem::processImage(double timestamp, cv::Mat &imgColor, std::map<std::pair<int,int>,Eigen::Vector2f> &features)
{
    MYAPP_LOG << "Track frame " << timestamp << " features.size() =  " << features.size() << std::endl;

	bool trackingSuccessfull = mTracker->trackFrame(timestamp, imgColor, features);

    // Every successfully tracked frame is a keyframe

    if(trackingSuccessfull)
    {
        createKeyframe();
    }
    else
    {
        MYAPP_LOG << "Tracking failed " << std::endl;
    }

}

void FiducialCalibSystem::createKeyframe()
{
	std::unique_ptr<Keyframe> frame_(new Keyframe());
	Keyframe *frame = frame_.get();

	const TrackingFrame *trackerFrame = mTracker->getFrame();

	frame->init(*mTracker->getFrame());
	frame->setPose(mTracker->getCurrentPose2D());

	//Add keyframe to map
	mMap->addKeyframe(std::move(frame_));

	//Separate matches by octave
	std::vector<std::vector<const FeatureMatch *>> matchesByOctave;
	std::vector<std::vector<cv::KeyPoint>> keypointsByOctave;
	matchesByOctave.resize(trackerFrame->getOriginalPyramid().getOctaveCount());
	keypointsByOctave.resize(trackerFrame->getOriginalPyramid().getOctaveCount());
	for (auto &match : mTracker->getFrame()->getMatches())
	{
		const int scale = 1 << match.getOctave();

		cv::KeyPoint kp = match.getKeypoint();
		kp.octave = 0;
		kp.pt = eutils::ToCVPoint((match.getPosition() / scale).eval());
		kp.size /= scale;

		matchesByOctave[match.getOctave()].push_back(&match);
		keypointsByOctave[match.getOctave()].push_back(kp);
	}


	for (int octave = 0; octave < (int)matchesByOctave.size(); octave++)
	{
		if (keypointsByOctave.empty())
			continue;

		//Create measurement
		for (int i = 0; i < (int)matchesByOctave[octave].size(); i++)
		{
			auto &match = *matchesByOctave[octave][i];
            auto &refMeasurement = *match.getSourceMeasurement();
            auto refFiducialPos = refMeasurement.getFiducialCoords();
			auto m = make_unique<FeatureMeasurement>(const_cast<Feature*>(&match.getFeature()), frame, match.getPosition(), octave, refFiducialPos);

			//Save measurement
			frame->getMeasurements().push_back(m.get());
			m->getFeature().getMeasurements().push_back(std::move(m));
		}
	}

	//frame->freeSpace();
}


bool FiducialCalibSystem::init(double timestamp, cv::Mat &imgColor, std::map<std::pair<int,int>,Eigen::Vector2f> &features)
{

	//Create first key frame
    std::unique_ptr<Keyframe> keyframe(new Keyframe());

    keyframe->init(imgColor, features);
    keyframe->setTimestamp(timestamp);
    keyframe->setPose(Eigen::Matrix3fr::Identity());

	mImageSize = keyframe->getImageSize();

	//Init camera
	mCamera.init(mImageSize.cast<float>() / 2, Eigen::Vector2f(1,1), mImageSize);
	mCamera.getDistortionModel().init();

	//Normal
	mNormal = Eigen::Vector3f::Zero();

	//Reset map
	mMap.reset(new Map());

    //Reset tracker
	mTracker.reset(new FiducialTracker());
	mTracker->init(mImageSize, keyframe->getOctaveCount());
	mTracker->setCamera(mCamera);

	mCalib.reset(new HomographyCalibration());

	//Start map
	//Add keyframe
	Keyframe *pkeyframe = keyframe.get();
	mMap->addKeyframe(std::move(keyframe));

	Eigen::Matrix3fr poseInv = pkeyframe->getPose().inverse();

	//Create 2D features

	for (auto it=features.begin(); it!=features.end(); it++)
	{
        Eigen::Vector2f featurePos = it->second;
        Eigen::Vector2d fiducialPos{it->first.first, it->first.second};
		auto *pfeature = mMap->createFiducial(*pkeyframe, poseInv, featurePos, 0, fiducialPos);
		pfeature->setPosition(featurePos - mCamera.getPrincipalPoint());
	}

	MYAPP_LOG << "Created " << features.size() << " features " << "\n";
    mTracker->resetTracking(mMap.get(), mMap->getKeyframes().back()->getPose());
	return true;
}

void FiducialCalibSystem::matchFeatures()
{
    std::map<std::pair<int,int>,Eigen::Vector2f> refFeatures = features.front();
    std::map<std::pair<int,int>,Eigen::Vector2f> currFeatures = features.back();

    Matches currMatches;
    currMatches.fromId = features.size() - 1;
    currMatches.toId = 0;

    for(auto it=currFeatures.begin() ; it !=currFeatures.end(); it++)
    {
        auto fiducialId = it->first;
        auto featurePos = it->second;

        auto it_pos = refFeatures.find(fiducialId);
        if(it_pos != refFeatures.end())
        {
            auto refPos = it_pos->second;
            std::pair<Eigen::Vector2f,Eigen::Vector2f> pos(featurePos,refPos);
            currMatches.matches.push_back(std::make_pair(fiducialId,pos));
        }
    }
    frameMatches.push_back(currMatches);
    MYAPP_LOG << "# of features ref = " << refFeatures.size() << " curr = " << currFeatures.size() << std::endl;
    MYAPP_LOG << "Found " << currMatches.matches.size() << " matches between " << currMatches.fromId << " to " << currMatches.toId << "\n";


}

void FiducialCalibSystem::estimateHomographyFromMatches(cv::Mat &img)
{
    //Create cv vectors
	std::vector<Eigen::Vector2f> refPoints, imgPoints;
	std::vector<float> scales;
    auto currMatches = frameMatches.back().matches;
    int matchCount = currMatches.size();
    int frameIdx = frameMatches.size();

	for (auto &match : currMatches)
	{
        auto matchPair = match.second;
		refPoints.push_back(matchPair.second);
		imgPoints.push_back(matchPair.first);
        // MYAPP_LOG << "Direct : " << refPoints.back() << " " << imgPoints.back() << std::endl;
		scales.push_back((float)(1));
	}

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

    int matchInlierCount = 0;

	for (int i = 0; i<(int)matchCount; i++)
	{
		if (inliersVec[i])
		{
            cv::circle(img,cv::Point(imgPoints[i][0],imgPoints[i][1]),5,cv::Scalar(0,255,0),-1);
			matchInlierCount++;
		}
        else
        {
            cv::circle(img,cv::Point(imgPoints[i][0],imgPoints[i][1]),5,cv::Scalar(0,0,255),-1);
        }
	}

    cv::imshow("Inliers",img);
    cv::waitKey(100);

    MYAPP_LOG << "Homography Nonlinear: inliers=" << matchInlierCount << "/" << matchCount << "\n";
    MYAPP_LOG << "Homography " << frameIdx << " : " << H << std::endl;

    if(matchInlierCount > 20){
        // homographiesMap[frameIdx] = H;
        homographies.push_back(H);

    }

}


void FiducialCalibSystem::processMeasurements(std::map<std::pair<int,int>,Eigen::Vector2f> &frameFeatures, cv::Mat &img)
{
    // features.push_back(std::move(frameFeatures));
    features.push_back(frameFeatures);
    if(features.size() == 1)
    {
        return;
    }
    matchFeatures();
    estimateHomographyFromMatches(img);
}

void FiducialCalibSystem::calibrate()
{
    mCalib->setVerbose(true);
    mCalib->setUseNormalizedConstraints(true);
	mCalib->setFixPrincipalPoint(false);
	mCalib->initFromCamera(mCamera);
	mCalib->calibrate(homographies);

	mCalib->updateCamera(mCamera);
	mNormal = mCalib->getNormal().cast<float>();

    auto focalLengths = mCamera.getFocalLength();
    auto principalPoint = mCamera.getPrincipalPoint();

    MYAPP_LOG << "Focal length = " << focalLengths[0] << std::endl;
    MYAPP_LOG << "Principal point = " << principalPoint << std::endl;

}

void FiducialCalibSystem::doHomographyBA()
{
	BundleAdjuster ba;

	ba.setUseLocks(false);
	ba.setMap(mMap.get());
	ba.setOutlierThreshold(3*mExpectedPixelNoiseStd);
	for (auto &framep : mMap->getKeyframes())
	{
		auto &frame = *framep;
		ba.addFrameToAdjust(frame);
	}

	ba.setOnlyDistortion(false);
	ba.setCamera(&mCamera);
	ba.bundleAdjust();

	//Calib
	//doHomographyCalib();

	//Update homography distortion with the new principal point
	//ba.setOnlyDistortion(true);
	//ba.setP0(Eigen::Vector2d(mK(0, 2), mK(1, 2)));
	//ba.bundleAdjust();

	//mHomographyP0 = ba.getP0().cast<float>();
	//mHomographyDistortion.setCoefficients(ba.getDistortion().cast<float>());
	//mActiveDistortion = &mHomographyDistortion;
}

void FiducialCalibSystem::doHomographyCalib(bool fixP0)
{
	std::vector<Eigen::Matrix3fr> allPoses;
	for (auto &frame : mMap->getKeyframes())
	{
		allPoses.push_back(frame->getPose());
	}

    mCalib->setVerbose(true);
	//Calibrate
	mCalib->setUseNormalizedConstraints(mUseNormalizedConstraints);
	mCalib->setFixPrincipalPoint(fixP0);
	mCalib->initFromCamera(mCamera);
	mCalib->calibrate(allPoses);

	mCalib->updateCamera(mCamera);
	mNormal = mCalib->getNormal().cast<float>();
	//mK << 600, 0, 320, 0, 600, 240, 0, 0, 1;
	//mNormal << 0, 0, 1;
    auto focalLengths = mCamera.getFocalLength();
    auto principalPoint = mCamera.getPrincipalPoint();
    MYAPP_LOG << "Focal length = " << focalLengths[0] << std::endl;
    MYAPP_LOG << "Principal point = " << principalPoint << std::endl;
}

void FiducialCalibSystem::doFullBA()
{
	if (!mMap->getIs3DValid())
	{
		MYAPP_LOG << "Initializing metric reconstruction...\n";

		if (mUse3DGroundTruth)
		{
			mCamera = *mMap->mGroundTruthCamera;

			for (auto &pfeature : mMap->getFeatures())
			{
				auto &feature = *pfeature;
				feature.mPosition3D = feature.mGroundTruthPosition3D;
			}
			for (auto &framep : mMap->getKeyframes())
			{
				auto &frame = *framep;
				frame.mPose3DR = frame.mGroundTruthPose3DR;
				frame.mPose3DT = frame.mGroundTruthPose3DT;
			}
		}
		else
		{
			//Set pose for reference frame
			Keyframe &refFrame = *mMap->getKeyframes()[0];

			Eigen::Vector3f basis1, basis2, basis3;
			basis3 = mNormal;
			eutils::GetBasis(basis3, basis1, basis2);
			basis1.normalize();
			basis2.normalize();

			refFrame.mPose3DR(0, 0) = basis1[0];
			refFrame.mPose3DR(1, 0) = basis1[1];
			refFrame.mPose3DR(2, 0) = basis1[2];

			refFrame.mPose3DR(0, 1) = basis2[0];
			refFrame.mPose3DR(1, 1) = basis2[1];
			refFrame.mPose3DR(2, 1) = basis2[2];

			refFrame.mPose3DR(0, 2) = basis3[0];
			refFrame.mPose3DR(1, 2) = basis3[1];
			refFrame.mPose3DR(2, 2) = basis3[2];

			Eigen::Vector3f refCenter = -basis3;
			MYAPP_LOG << "Center of ref camera: " << refCenter.transpose() << "\n";
			refFrame.mPose3DT = -refFrame.mPose3DR*refCenter; //RefCenter = -R'*t = Normal (exactly one unit away from plane center) => t = -R*normal
			//refFrame.mPose3DR = refFrame.mGroundTruthPose3DR;
			//refFrame.mPose3DT = refFrame.mGroundTruthPose3DT;
			//refCenter = -refFrame.mPose3DR.transpose() * refFrame.mPose3DT;

			//auto k = mCamera.getK();
			//cv::Mat1f cvK(3, 3, const_cast<float*>(k.data()));

			//Triangulate all features
			for (auto &pfeature : mMap->getFeatures())
			{
				auto &feature = *pfeature;

				Eigen::Vector3f xn = mCamera.unprojectToWorld(feature.getPosition());
				Eigen::Vector3f xdir = refFrame.mPose3DR.transpose()*xn;

				//Intersect with plane
				//If point in line is x=a*t + b
				//and point in plane is dot(x,n)-d = 0
				//then t=(d-dot(b,n))/dot(a,n)
				//and x = a*(d-dot(b,n))/dot(a,n) + b
				//
				//Here b=refCenter, a=xdir, n=[0,0,1]', d=0
				feature.mPosition3D = refCenter - (refCenter[2]/xdir[2])*xdir;
				feature.mPosition3D[2] = 0; //Just in case

				//Eigen::Vector3f xnt = refFrame.mPose3DR * feature.mPosition3D + refFrame.mPose3DT;
				//Eigen::Vector2f imagePosClean = mCamera.projectFromWorld(xnt);

				//Eigen::Vector3f xnt2 = refFrame.mPose3DR * feature.mGroundTruthPosition3D + refFrame.mPose3DT;
				//Eigen::Vector2f imagePosClean2 = mCamera.projectFromWorld(xnt2);
			}

			//Estimate frame positions
			for (auto &framep : mMap->getKeyframes())
			{
				auto &frame = *framep;

				//Skip ref frame
				if (&frame == &refFrame)
					continue;

				//Build constraints
				std::vector<Eigen::Vector3f> refPoints;
				std::vector<Eigen::Vector2f> imgPoints;
				std::vector<float> scales;
				for (auto &mp : frame.getMeasurements())
				{
					auto &m = *mp;

					refPoints.push_back(m.getFeature().mPosition3D);
					imgPoints.push_back(m.getPosition());
					scales.push_back((float)(1<<m.getOctave()));
				}

				//PnP
				PnPRansac ransac;
				ransac.setParams(3 * mExpectedPixelNoiseStd, 10, 100, (int)(0.99f * frame.getMeasurements().size()));
				ransac.setData(&refPoints, &imgPoints, &scales, &mCamera);
				ransac.doRansac();
				//MYAPP_LOG << "Frame pnp inlier count: " << ransac.getBestInlierCount() << "/" << matches.size() << "\n";
				frame.mPose3DR = ransac.getBestModel().first.cast<float>();
				frame.mPose3DT = ransac.getBestModel().second.cast<float>();

				//Refine
				int inlierCount;
				std::vector<MatchReprojectionErrors> errors;
				PnPRefiner refiner;
				refiner.setCamera(&mCamera);
				refiner.setOutlierThreshold(3 * mExpectedPixelNoiseStd);
				refiner.refinePose(refPoints, imgPoints, scales, frame.mPose3DR, frame.mPose3DT, inlierCount, errors);
				//Save
				//cv::Rodrigues(rvec, cvR);

				//frame.mPose3DR = mapR.cast<float>();
				//frame.mPose3DT[0] = (float)tvec(0, 0);
				//frame.mPose3DT[1] = (float)tvec(1, 0);
				//frame.mPose3DT[2] = (float)tvec(2, 0);
			}
		}
		mMap->setIs3DValid(true);
		auto poseH = mTracker->getCurrentPose2D();
		mTracker->resetTracking(mMap.get(), poseH);
	}

	//Camera params
	//distortion[1] += 0.03;
	//BAAAAA!!!
	CalibratedBundleAdjuster ba;
	ba.setUseLocks(false);
	ba.setFix3DPoints(mFix3DPoints);
	ba.setOutlierThreshold(3 * mExpectedPixelNoiseStd);
	ba.setCamera(&mCamera);
	//ba.setFixDistortion(true);
	//ba.setDistortion(Eigen::Vector2d(0.0935491, -0.157975));
	ba.setMap(mMap.get());
	for (auto &framep : mMap->getKeyframes())
	{
		ba.addFrameToAdjust(*framep);
	}
	ba.bundleAdjust();

	mMap->mCamera.reset(new CameraModel(mCamera));

	//Log
	//MatlabDataLog::Instance().AddValue("K", ba.getK());
	//MatlabDataLog::Instance().AddValue("Kold", mCalib->getK());
	//MatlabDataLog::Instance().AddValue("Nold", mCalib->getNormal());
	//for (auto &framep : mMap->getKeyframes())
	//{
	//	auto &frame = *framep;
	//	MatlabDataLog::Instance().AddCell("poseR", frame.mPose3DR);
	//	MatlabDataLog::Instance().AddCell("poseT", frame.mPose3DT);
	//	MatlabDataLog::Instance().AddCell("poseH", frame.getPose());

	//	MatlabDataLog::Instance().AddCell("posM");
	//	MatlabDataLog::Instance().AddCell("posRt");
	//	MatlabDataLog::Instance().AddCell("posH");
	//	for (auto &mp : frame.getMeasurements())
	//	{
	//		auto &m = *mp;

	//		MatlabDataLog::Instance().AddValueToCell("posM", m.getPosition());

	//		Eigen::Vector3f mm;
	//		Eigen::Vector2f m2;

	//		mm = ba.getK().cast<float>() * (frame.mPose3DR*m.getFeature().mPosition3D + frame.mPose3DT);
	//		m2 << mm[0] / mm[2], mm[1] / mm[2];

	//		MatlabDataLog::Instance().AddValueToCell("posRt", m2);

	//		Eigen::Vector3f p; p << m.getFeature().getPosition()[0], m.getFeature().getPosition()[1], 1;
	//		mm = frame.getPose() * p;
	//		m2 << mm[0] / mm[2], mm[1] / mm[2];

	//		MatlabDataLog::Instance().AddValueToCell("posH", m2);

	//	}
	//}
}

void FiducialCalibSystem::doValidationBA()
{
	//BAAAAA!!!
	CalibratedBundleAdjuster ba;
	ba.setUseLocks(false);
	ba.setFixPrincipalPoint(true);
	ba.setFixDistortion(true);
	ba.setFixFocalLengths(true);
	ba.setFix3DPoints(true);
	ba.setOutlierThreshold(3 * mExpectedPixelNoiseStd);
	ba.setCamera(&mCamera);
	ba.setMap(mMap.get());
	for (auto &framep : mMap->getKeyframes())
	{
		ba.addFrameToAdjust(*framep);
	}
	ba.bundleAdjust();
}


}
