#ifndef IMAGEDATASOURCE_H_
#define IMAGEDATASOURCE_H_

#include <GL/glew.h>
#include <opencv2/core.hpp>

namespace planecalib
{

class ImageDataSource
{
public:
	ImageDataSource():
		mDownsampleCount(0), mSourceTextureId(0)
	{
	}

    virtual ~ImageDataSource(void)
    {
    	releaseGl();
    }

    virtual void dropFrames(int count) = 0;
    virtual bool update(void) = 0;

    unsigned int getTextureId(void) const {return mSourceTextureId;}
    unsigned int getTextureTarget(void) const {return GL_TEXTURE_2D;}
    double getCaptureTime(void) const {return mCaptureTime;}

    void setDownsample(int count) {mDownsampleCount=count; createBuffers();}

    const cv::Size &getSourceSize() const {return mSourceSize;}

    cv::Size getSize() const {return mImgGray.size();}

    const cv::Mat1b &getImgGray() const {return mImgGray;}
    const cv::Mat3b &getImgColor() const {return mImgColor;}

protected:
    cv::Size mSourceSize;
    int mDownsampleCount;

    GLuint mSourceTextureId;

    double mCaptureTime;
    cv::Mat1b mImgGray;
    cv::Mat3b mImgColor;

    virtual void setSourceSize(const cv::Size &sz);
    void createBuffers();
    void releaseGl();
    void update(const cv::Mat3b &source, double captureTime);
};

}

#endif /* IMAGEDATASOURCE_H_ */
