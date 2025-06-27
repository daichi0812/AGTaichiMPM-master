#ifndef __COLOR_GRID_H__
#define __COLOR_GRID_H__

#include "smoothingGrid.h"
#include "slVector.h"
#include "slArray.h"
#include <vector>

class ColorGrid
{
	ColorGrid();
public:
	~ColorGrid();
	ColorGrid(double h, int nx, int ny, int nz, const SlVector3& bbMin, const SlVector3& bbMax, unsigned int flags, 
								const std::vector<SlVector3> &particles, const std::vector<double> &radii, const std::vector<SlVector3> &colors);
	
	void computeColor( const SlVector3& x, SlVector3& color ) const;
	
	//for debug
	void saveGridDataAsResumeData( const std::string& file_name ) const;
	
	double h;
	int nx, ny, nz;
	SlVector3 bbMin, bbMax;
	SlArray3D<SlVector3> colorField;
  							
private:
    // Compute the value of the signed distance field at the point x using trilinear interpolation
    void trilinearInterpolation( const SlVector3& v000, const SlVector3& v100, const SlVector3& v010, const SlVector3& v110,
                                                    const SlVector3& v001, const SlVector3& v101, const SlVector3& v011, const SlVector3& v111,
                                                    const double bc[3], SlVector3& v ) const;
                                                    
    void trilinearInterpolation( const SlVector3& v000, const double w000, const SlVector3& v100, const double w100,
													const SlVector3& v010, const double w010, const SlVector3& v110, const double w110,
                                                    const SlVector3& v001, const double w001, const SlVector3& v101, const double w101, 
													const SlVector3& v011, const double w011, const SlVector3& v111, const double w111, 
                                                    const double bc[3], SlVector3& v ) const;

    // Linear interpolation of v0 and v1 parameterized by alpha
    void linearInterpolation( const SlVector3& v0, const SlVector3& v1, const double& alpha, SlVector3& v ) const;
    
    void linearInterpolation( const SlVector3& v0, const double w0, const SlVector3& v1, const double w1, const double& alpha, SlVector3& v, double& w ) const;
    
    double rmax;
	SlArray3D<double> weights;
};

#endif
