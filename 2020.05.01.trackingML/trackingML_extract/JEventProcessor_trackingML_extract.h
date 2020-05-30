// $Id$
//
//    File: JEventProcessor_trackingML_extract.h
// Created: Thu Apr 30 17:04:48 EDT 2020
// Creator: davidl (on Linux ifarm1901.jlab.org 3.10.0-1062.4.1.el7.x86_64 x86_64)
//

#ifndef _JEventProcessor_trackingML_extract_
#define _JEventProcessor_trackingML_extract_

#include <JANA/JEventProcessor.h>
#include <fstream>
#include <mutex>
using namespace std;

class JEventProcessor_trackingML_extract:public jana::JEventProcessor{
	public:
		JEventProcessor_trackingML_extract();
		~JEventProcessor_trackingML_extract();
		const char* className(void){return "JEventProcessor_trackingML_extract";}
		
		ofstream *ofs_features = nullptr;
		ofstream *ofs_labels = nullptr;
		mutex output_mutex;
		
		uint64_t first_straw[28]; // number of first straw in a ring so we can number them 1-3522

	private:
		jerror_t init(void);						///< Called once at program start.
		jerror_t brun(jana::JEventLoop *eventLoop, int32_t runnumber);	///< Called everytime a new run number is detected.
		jerror_t evnt(jana::JEventLoop *eventLoop, uint64_t eventnumber);	///< Called every event.
		jerror_t erun(void);						///< Called everytime run number changes, provided brun has been called.
		jerror_t fini(void);						///< Called after last event of last event source has been processed.
};

#endif // _JEventProcessor_trackingML_extract_

