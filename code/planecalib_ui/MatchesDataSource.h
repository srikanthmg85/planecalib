/*
 * MatchesDataSource.h
 *
 *  Created on: 22.3.2014
 *      Author: dan
 */

#ifndef MATCHESDATASOURCE_H_
#define MATCHESDATASOURCE_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glew.h>
#include "ImageDataSource.h"
#include <Eigen/Dense>
#include <Map>

namespace planecalib {

class MatchesDataSource: public ImageDataSource
{
public:
	MatchesDataSource(void);
    ~MatchesDataSource(void);

    bool open(const std::string &sequenceFormat, int startIdx);
    void close(void);

    void dropFrames(int count);
    bool update(void);
    bool readFeatures(int idx, std::map<std::pair<int,int>,Eigen::Vector2f> &features);
    cv::Mat readImage(int idx);

private:
    std::string mSequenceFormat;
    int mCurrentFrameIdx;



    void setSourceSize(const cv::Size &sz);
};

} /* namespace planecalib */

#endif /* MATCHESDATASOURCE_H_ */
