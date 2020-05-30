// $Id$
//
//    File: JEventProcessor_trackingML_extract.cc
// Created: Thu Apr 30 17:04:48 EDT 2020
// Creator: davidl (on Linux ifarm1901.jlab.org 3.10.0-1062.4.1.el7.x86_64 x86_64)
//

#include "JEventProcessor_trackingML_extract.h"
using namespace jana;

#include <sstream>


#include <TRACKING/DTrackTimeBased.h>
#include <particleType.h>


// Routine used to create our JEventProcessor
#include <JANA/JApplication.h>
#include <JANA/JFactory.h>
extern "C"{
void InitPlugin(JApplication *app){
	InitJANAPlugin(app);
	app->AddProcessor(new JEventProcessor_trackingML_extract());
}
} // "C"


//------------------
// JEventProcessor_trackingML_extract (Constructor)
//------------------
JEventProcessor_trackingML_extract::JEventProcessor_trackingML_extract()
{

}

//------------------
// ~JEventProcessor_trackingML_extract (Destructor)
//------------------
JEventProcessor_trackingML_extract::~JEventProcessor_trackingML_extract()
{

}

//------------------
// init
//------------------
jerror_t JEventProcessor_trackingML_extract::init(void)
{

	// Set number of wires in each CDC layer
	vector<uint64_t> cdc_nwires;
	cdc_nwires.push_back(42);
	cdc_nwires.push_back(42);
	cdc_nwires.push_back(54);
	cdc_nwires.push_back(54);
	cdc_nwires.push_back(66);
	cdc_nwires.push_back(66);
	cdc_nwires.push_back(80);
	cdc_nwires.push_back(80);
	cdc_nwires.push_back(93);
	cdc_nwires.push_back(93);
	cdc_nwires.push_back(106);
	cdc_nwires.push_back(106);
	cdc_nwires.push_back(123);
	cdc_nwires.push_back(123);
	cdc_nwires.push_back(135);
	cdc_nwires.push_back(135);
	cdc_nwires.push_back(146);
	cdc_nwires.push_back(146);
	cdc_nwires.push_back(158);
	cdc_nwires.push_back(158);
	cdc_nwires.push_back(170);
	cdc_nwires.push_back(170);
	cdc_nwires.push_back(182);
	cdc_nwires.push_back(182);
	cdc_nwires.push_back(197);
	cdc_nwires.push_back(197);
	cdc_nwires.push_back(209);
	cdc_nwires.push_back(209);

	first_straw[0] = 0;
	for(int i=1; i<28; i++) first_straw[i] = first_straw[i-1] + cdc_nwires[i-1];
	cout << "Total CDC wires: " << (first_straw[27]+cdc_nwires[27]) << endl;

	// Open output files
	ofs_features = new ofstream("trackingML_features6.csv");
	ofs_labels = new ofstream("trackingML_labels6.csv");

	// Write header to features file
	(*ofs_features) << "event,t_start_cntr,t_start_cntr_valid,t_tof,t_tof_valid,t_bcal,t_bcal_valid,t_fcal,t_fcal_valid";
	int iring=1;
	for( auto &nstraws : cdc_nwires ) {
		for(uint64_t istraw=1; istraw<=nstraws ; istraw++){
			(*ofs_features) << ",CDC_ring"<<iring<<"_straw"<<istraw;
		}
		iring++;
	}
	for( int ilayer=1; ilayer<=24; ilayer++){
		for(uint64_t iwire=1; iwire<=96 ; iwire++){
			(*ofs_features) << ",FDC_layer"<<ilayer<<"_wire"<<iwire;
		}
	}
	(*ofs_features) << "\n";
	
	// Write header to labels file
	(*ofs_labels) << "event,q_over_pt,phi,tanl,D,z";
	for( int i=0; i<5; i++)
			for( int j=i; j<5; j++ ) (*ofs_labels) << ",cov_"<<i<<j;
	for( int i=0; i<5; i++)
			for( int j=0; j<5; j++ ) (*ofs_labels) << ",invcov_"<<i<<j;

	(*ofs_labels) << ",chisq,Ndof,rms";
	(*ofs_labels) << "\n";

	return NOERROR;
}

//------------------
// brun
//------------------
jerror_t JEventProcessor_trackingML_extract::brun(JEventLoop *eventLoop, int32_t runnumber)
{
	// This is called whenever the run number changes
	return NOERROR;
}

//------------------
// evnt
//------------------
jerror_t JEventProcessor_trackingML_extract::evnt(JEventLoop *loop, uint64_t eventnumber)
{
	vector<const DTrackTimeBased*> tbts;
	loop->Get( tbts );
	for( auto tbt : tbts ){
		if( tbt->forwardParmFlag() != 0 ) continue; // ensure standard parameterization (see below)
		if( tbt->PID() != PiMinus       ) continue; // only extract piMinus tracks
		
		
		//-------------------------
		// Write labels (state vector and covariance matrix, etc..)
		
		// It seems that the tracking state vector is never reported using the
		// forward parameterization. The values in the state vector are thus:
		// q/p_t , phi, tanl, D, z
		double aVec[5];
		tbt->TrackingStateVector( aVec );
		auto cov = tbt->TrackingErrorMatrix();
		
		stringstream ss_labels;
		
		// event number and tracking covariance matrix (15 values)
		ss_labels << eventnumber;
		ss_labels << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		for(auto a : aVec ) ss_labels << "," << a;
		for( int i=0; i<5; i++)
			for( int j=i; j<5; j++) ss_labels << "," << (*cov)(i,j);

		// inverse covariance matrix (25 values)
		Double_t det;
		std::shared_ptr<TMatrixTSym<float>> invcov((TMatrixTSym<float>*)cov->Clone());
		std::shared_ptr<TMatrixTSym<float>> unit((TMatrixTSym<float>*)cov->Clone());
		invcov->Invert(&det);
		unit->UnitMatrix();
		auto rms = sqrt((((*invcov)*(*cov))-(*unit)).Sqr().Sum());
// 		cout << "-------------------------------------------------------------" << endl;
// 		cout << "event: " << eventnumber << "   sum=" << ((*invcov)*(*cov)-(*unit)).Sqr().Sum() << "  rms=" << rms << endl;
// 		cout <<" covariance:" << endl;
// 		(*cov).Print();
// 		cout <<" inverse:" << endl;
// 		(*invcov).Print();
// 		cout <<" product:" << endl;
// 		((*invcov)*(*cov)).Print();
		
		// Ignore tracks with bad inverse covariance matrices.
		// The "rms" value calculated above quantifies how well
		// the inversion of cov succeeded. A value of 0 means
		// it was perfect. Larger values are worse. Earlier tests
		// showed > 12% of the tracks had an rms>1. We want to ignore
		// those since the larger values in the invcov matrix will
		// lead to the model giving stronger preference for those
		// tracks. Ideally, we would just add the rms to the output
		// and defer cutting on it to the training script. Since we
		// have 2 output files though (features and labels) we would
		// either need to put it in both, or index one and apply the
		// index to the other. I'm avoiding that complication for
		// now by simply hardcoding a cut right here.
		if( rms > 1.0E-2 ) continue;
		
		
		for( int i=0; i<5; i++)
			for( int j=0; j<5; j++) ss_labels << "," << (*invcov)(i,j);
		
		// chisq and Ndof
		ss_labels << "," << tbt->chisq << "," << tbt->Ndof << "," << rms;
		ss_labels << "\n";

		vector<const DFDCPseudo*  > fdchits;
		vector<const DCDCTrackHit*> cdchits;
		tbt->Get( fdchits );
		tbt->Get( cdchits );
		
		// initialize vector with default values
		double vals[ 8 + 3522 + 2304 ]; // 8=timing counters+valid flags, 3522=CDC, 2304=FDC
		for( auto &v : vals ) v = 1E3;
		vals[1] = vals[3] = vals[5] = vals[7] = 0; // default values for valid flags for all timing detectors

		//-------------------------
		// Write features (hits and times)
		
		// We don't seem to have easy access to the full list of start times
		// given to the track fitter. For now, we pick just the one reported 
		// in the DTrackTimeBased object as having been used.
		switch( tbt->t0_detector() ){
			case SYS_START:
				vals[0] = tbt->t0();
				vals[1] = 1;
				break;
			case SYS_TOF:
				vals[2] = tbt->t0();
				vals[3] = 1;
				break;
			case SYS_BCAL:
				vals[4] = tbt->t0();
				vals[5] = 1;
				break;
			case SYS_FCAL:
				vals[6] = tbt->t0();
				vals[7] = 1;
				break;
			default:
				break;		
		}
		

		// Add CDC values
		for( auto h : cdchits ){
			uint64_t idx = 8 + first_straw[h->wire->ring-1] + (h->wire->straw-1);
			assert( idx>=8 );
			assert( idx<(8+3522) );		
			vals[idx] = h->tdrift;
		}

		// Add FDC values
		for( auto h : fdchits ){
			uint64_t idx = 8 + 3522 + (96*(h->wire->layer-1)) + (h->wire->wire-1);			
			assert( idx>=(8+3522) );
			assert( idx<(8+3522+2304) );		
			vals[idx] = h->time;
		}

		// Write all feature values to output string (defer writing to file for when mutex is locked)
		stringstream ss_features;
		ss_features << eventnumber;
		for( auto v : vals ) ss_features << "," << v;
		ss_features << "\n";
		
		// Get exclusive writing rights to output streams to make sure they
		// stay in sync when running multi-threaded
		lock_guard<mutex> lck(output_mutex);
		
		(*ofs_features) << ss_features.str();
		(*ofs_labels  ) << ss_labels.str();
	}


	return NOERROR;
}

//------------------
// erun
//------------------
jerror_t JEventProcessor_trackingML_extract::erun(void)
{
	// This is called whenever the run number changes, before it is
	// changed to give you a chance to clean up before processing
	// events from the next run number.
	return NOERROR;
}

//------------------
// fini
//------------------
jerror_t JEventProcessor_trackingML_extract::fini(void)
{

	if( ofs_features ){
		ofs_features->close();
		delete ofs_features;
	}

	if( ofs_labels ){
		ofs_labels->close();
		delete ofs_labels;
	}

	return NOERROR;
}

