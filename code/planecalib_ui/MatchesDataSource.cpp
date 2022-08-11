/*
 * MatchesDataSource.cpp
 *
 *  Created on: 22.3.2014
 *      Author: dan
 */

#include "MatchesDataSource.h"
#include <opencv2/imgproc.hpp>
#include "planecalib/Profiler.h"
#include "planecalib/log.h"
#include <string>

namespace planecalib
{

MatchesDataSource::MatchesDataSource(void)
{
}

MatchesDataSource::~MatchesDataSource(void)
{
    close();
}

cv::Mat MatchesDataSource::readImage(int idx)
{
	char buffer[1024];
	sprintf(buffer, mSequenceFormat.c_str(), idx);

    std::string fileName = mSequenceFormat + "/" + std::to_string(idx) + ".png";
    MYAPP_LOG << fileName << std::endl;

    std::vector<std::pair<int,Eigen::Vector2f>> features;
    // readFeatures(idx,features);

	try
	{
	return cv::imread(fileName, cv::IMREAD_COLOR);
	}
	catch(cv::Exception &ex)
	{
		MYAPP_LOG << "\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n\n";
		throw ex;
		return cv::Mat();
	}
}

bool MatchesDataSource::open(const std::string &sequenceFormat, int startIdx)
{
	mSequenceFormat = sequenceFormat;
	mCurrentFrameIdx = startIdx;

	cv::Mat sampleImg = readImage(startIdx);
	if(sampleImg.empty())
	{
		MYAPP_LOG << "Error opening image sequence, format=" << sequenceFormat << ", startIdx=" << startIdx << "\n";
		return false;
	}
    MYAPP_LOG << "Image read done\n";
    setSourceSize(sampleImg.size());
    return true;
}

void MatchesDataSource::close(void)
{
	releaseGl();
}

void MatchesDataSource::dropFrames(int count)
{
	mCurrentFrameIdx += count;
}

bool MatchesDataSource::update(void)
{
    cv::Mat frame = readImage(mCurrentFrameIdx);

    if(frame.empty())
    {
    	return false;
    }

    assert(frame.channels()==3);

    ImageDataSource::update(cv::Mat3b(frame), mCurrentFrameIdx);

    mCurrentFrameIdx++;

    return true;
}

bool MatchesDataSource::readFeatures(int idx, std::map<std::pair<int,int>,Eigen::Vector2f> &features)
{
    std::string fileName = mSequenceFormat + "/" + std::to_string(idx) + ".txt";
    MYAPP_LOG << fileName << std::endl;

    std::ifstream inputMatches(fileName);

    if(inputMatches.fail())
        return false;

    std::string line;
    std::vector<double> vals;

    while (std::getline(inputMatches, line)) {
    std::stringstream sstr(line);
    std::string floatString;
    while (std::getline(sstr, floatString, ' ')) {
      double val = std::stod(floatString.c_str());
      vals.push_back(val);
    }

    int gridPosX = int(vals[0]);
    int gridPosY = int(vals[1]);
    double featurePosX = vals[2];
    double featurePosY = vals[3];

    // MYAPP_LOG << gridPosX << " " << gridPosY << " " << featurePosX << " " << featurePosY << std::endl;

    std::pair<int,int> gridPos(gridPosX,gridPosY);
    Eigen::Vector2f featurePos;
    featurePos << featurePosX,featurePosY;

    features[gridPos] = featurePos;
    vals.clear();

  }

    return true;
}



void MatchesDataSource::setSourceSize (const cv::Size &sz)
{
	mSourceSize = sz;
}

} /* namespace planecalib */
