
#ifndef PLANECALIBAPP_H_
#define PLANECALIBAPP_H_

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <Eigen/Dense>

#include "planecalib/Profiler.h"

#include "Application.h"
#include "shaders/Shaders.h"
#include "windows/BaseWindow.h"

namespace planecalib
{

class PlaneCalibSystem;
class FiducialCalibSystem;
class ImageDataSource;
class MatchesDataSource;
class OpenCVDataSource;

class PlaneCalibApp: public Application
{
private:
	bool mInitialized;

    int mFrameCount;

    bool mUsingCamera;
    bool mUsingFiducials;
    std::unique_ptr<ImageDataSource> mImageSrc;
    std::unique_ptr<MatchesDataSource> mFiducialSrc;
    int mDownsampleInputCount;
    Eigen::Vector2i mImageSize;

    Shaders mShaders;

    volatile bool mQuit;

    bool mFrameByFrame;
    bool mAdvanceFrame;

	bool mRecordOneFrame;
	int mRecordId;
	std::string mRecordFileFormat;

	bool mRecordVideo;
	std::string mRecordVideoFilename;
	cv::VideoWriter mRecordVideoWriter;

    bool mShowProfiler;
    bool mShowProfilerTotals;
	
	float mFPS;
	std::chrono::high_resolution_clock::time_point mLastFPSCheck;
	std::chrono::high_resolution_clock::duration mFPSUpdateDuration;
	std::chrono::high_resolution_clock::duration mFPSSampleAccum;
	int mFPSSampleCount;

	std::unique_ptr<PlaneCalibSystem> mSystem;
    std::unique_ptr<FiducialCalibSystem> mFiducialSystem;

    std::vector<std::unique_ptr<BaseWindow>> mWindows;
    BaseWindow *mActiveWindow;

public:
	static const float kDefaultFontHeight;

	PlaneCalibApp();
	~PlaneCalibApp();

    bool getFinished() {return mQuit;}

    Shaders &getShaders() {return mShaders;}
	PlaneCalibSystem &getSystem() { return *mSystem; }

    bool init();
    bool init(bool fiducials);
    void resize();
    void exit();
    bool loop() {return mQuit;}

    void keyDown(bool isSpecial, uchar key);
    void keyUp(bool isSpecial, uchar key);
    void touchDown(int id, int x, int y);
    void touchMove(int x, int y);
    void touchUp(int id, int x, int y);

    void draw(void);
    void processFiducialMatches();

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    bool initImageSrc();

	KeyBindingHandler<PlaneCalibApp> mKeyBindings;
    void runVideo() {mFrameByFrame = !mFrameByFrame; mAdvanceFrame = false;}
    void stepVideo(){mFrameByFrame = true; mAdvanceFrame = true;}
    void toggleProfilerMode();
    void resetProfiler() {Profiler::Instance().reset();}
    void escapePressed() {mQuit=true;}
    void changeWindowKey(bool isSpecial, unsigned char key);
    void resetSystem();
    void toggleRecording();
	void recordOneFrame() { mRecordOneFrame = true; }
	void recordInputFrame(cv::Mat3b &im);
	void recordOutputFrame();

	void saveMap();
	void loadMap();

    void setActiveWindow(BaseWindow *window);
};

}

#endif /* SLAMDRIVER_H_ */
