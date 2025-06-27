#include "colorGrid.h"
#include <float.h>
#include "slUtil.h"

ColorGrid::ColorGrid(double h, int nx, int ny, int nz, const SlVector3& bbMin, const SlVector3& bbMax, unsigned int flags, 
	const std::vector<SlVector3> &particles, const std::vector<double> &radii, const std::vector<SlVector3> &colors)
{
	this->nx = nx; this->ny = ny; this->nz = nz;
	this->bbMin = bbMin; this->bbMax = bbMax;
	this->h = h;
	
	/*
	// compute bounding box and grid dimensions
	bbMin[0] = bbMin[1] = bbMin[2] = DBL_MAX;
	bbMax[0] = bbMax[1] = bbMax[2] = -DBL_MAX;
	
	for (std::vector<SlVector3>::const_iterator i=particles.begin(); i!=particles.end(); i++) {
		bbMin[0] = std::min(bbMin[0], (*i)[0]);
		bbMin[1] = std::min(bbMin[1], (*i)[1]);
		bbMin[2] = std::min(bbMin[2], (*i)[2]);
		bbMax[0] = std::max(bbMax[0], (*i)[0]);
		bbMax[1] = std::max(bbMax[1], (*i)[1]);
		bbMax[2] = std::max(bbMax[2], (*i)[2]);
	}
	// increase the bounding box by rmax + something a little bigger than the stencil size
	double maxFactor = 1;
	if (flags & SmoothingGrid::NEIGHBOR_ANISOTROPY) {
		maxFactor = sqrt(maxStretch);
	} else if (flags & SmoothingGrid::VELOCITY_ANISOTROPY) {
		maxFactor *= cbrt(maxStretch); 
	}
	bbMin -= 5*h + maxFactor*rmax + 10*h;
	bbMax += 5*h + maxFactor*rmax + 10*h;
	
	
	int nx_min = floor((bbMin[0]/h));
	int ny_min = floor((bbMin[1]/h));
	int nz_min = floor((bbMin[2]/h));
	int nx_max = ceil((bbMax[0]/h));
	int ny_max = ceil((bbMax[1]/h));
	int nz_max = ceil((bbMax[2]/h));
	
	nx = nx_max - nx_min;
	ny = ny_max - ny_min;
	nz = nz_max - nz_min;
	
	bbMin[0]=nx_min*h;
	bbMin[1]=ny_min*h;
	bbMin[2]=nz_min*h;
	bbMax[0]=nx_max*h;
	bbMax[1]=ny_max*h;
	bbMax[2]=nz_max*h;
	//*/
	
	//if (flags & VERBOSE) 
		std::cout<<"h = "<<h<<" nx = "<<nx<<" ny = "<<ny<<" nz = "<<nz<<std::endl;
	//if (flags & VERBOSE) 
		std::cout<<"Bounding box is "<<bbMin<<" X "<<bbMax<<std::endl;
	
	colorField.allocate(nx,ny,nz);
	weights.allocate(nx,ny,nz);
	
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				weights(i,j,k) = -10.0;
				colorField(i,j,k) = SlVector3(0.0, 0.0, 0.0);
			}
		}
	}
	
	for(unsigned int p = 0; p < particles.size(); p++) 
	{
		double r = radii[p];
		int width = ((int) ceil(rmax / h) + 1) * 2; 
		
		SlInt3 bin;
		bin[0] = (int) floor((particles[p][0] - bbMin[0]) / h);
		bin[1] = (int) floor((particles[p][1] - bbMin[1]) / h);
		bin[2] = (int) floor((particles[p][2] - bbMin[2]) / h);
		
		int imax = std::min(bin[0] + width + 2, nx);
		int jmax = std::min(bin[1] + width + 2, ny);
		int kmax = std::min(bin[2] + width + 2, nz);
		
		for(int i = std::max(bin[0] - width, 0); i < imax; i++) {
			for(int j = std::max(bin[1] - width, 0); j < jmax; j++) {
				for(int k = std::max(bin[2] - width, 0); k < kmax; k++) {
					double gx = i*h + bbMin[0]; 
					double gy = j*h + bbMin[1]; 
					double gz = k*h + bbMin[2];
					double dist2 = (gx-particles[p][0])*(gx-particles[p][0]) 
						+ (gy-particles[p][1])*(gy-particles[p][1]) 
						+ (gz-particles[p][2])*(gz-particles[p][2]);
					double w = r * r * r / std::max(0.00001, dist2 * dist2);
					
					if(weights(i,j,k) < 0.0)
						weights(i,j,k) = w;
					else
						weights(i,j,k) += w;

					colorField(i,j,k) += w * colors[p];
				}
			}
		}
	}
	
	//std::cout << std::endl;
	//std::cout << "###Color Grid Content: " << std::endl;
	
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				if(weights(i,j,k) <= 0.0)
				{
					colorField(i,j,k) = SlVector3(1.0, 1.0, 1.0);
					weights(i,j,k) = 0.0;
				}
				else
				{
					colorField(i,j,k) /= weights(i,j,k);
					weights(i,j,k) = 1.0;
				}
					
				//std::cout << "[(" << i << "," << j << "," << k << "), <" << colorField(i,j,k)[0] << "," << colorField(i,j,k)[1] << "," << colorField(i,j,k)[2] << ">], ";
			}
			//std::cout << std::endl;
		}
	}
	//std::cout << "###" << std::endl;
	//std::cout << std::endl;
}

ColorGrid::~ColorGrid()
{
	
}

void ColorGrid::saveGridDataAsResumeData( const std::string& file_name ) const
{
	FILE* f = fopen(file_name.c_str(), "wb");
	
	int32_t tick = 0;
	fwrite(&tick, sizeof(int32_t), 1, f);
	int32_t num_points = 0;
	
	for(int i = 0; i < nx; i++) { for(int j = 0; j < ny; j++) { for(int k = 0; k < nz; k++) {
		if(weights(i,j,k) > 0) num_points++;
	}}}
	
	fwrite(&num_points, sizeof(int32_t), 1, f);
	
	for(int i = 0; i < num_points; i++) {
		double mass = 1.0;
		fwrite(&mass, sizeof(double), 1, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double density = 1.0;
		fwrite(&density, sizeof(double), 1, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double volume = 1.0;
		fwrite(&volume, sizeof(double), 1, f);
	}
	
	for(int i = 0; i < nx; i++) { for(int j = 0; j < ny; j++) { for(int k = 0; k < nz; k++) {
		if(weights(i,j,k) > 0) {
			double x[3] = {bbMin[0]+i*h, bbMin[1]+j*h, bbMin[2]+k*h};
			fwrite(x, sizeof(double), 3, f);
		}
	}}}
	
	for(int i = 0; i < num_points; i++) {
		double r = 0.001;
		fwrite(&r, sizeof(double), 1, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double distToSurface = 0.001;
		fwrite(&distToSurface, sizeof(double), 1, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double v[3] = {0.0, 0.0, 0.0};
		fwrite(v, sizeof(double), 3, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double F[9] = {
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		};
		fwrite(F, sizeof(double), 9, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double be_bar[9] = {
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		};
		fwrite(be_bar, sizeof(double), 9, f);
	}
	
	for(int i = 0; i < num_points; i++) {
		double sigma[9] = {
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		};
		fwrite(sigma, sizeof(double), 9, f);
	}
	
	/*
	if(state.mat_params.size() > 1)
	{
		for(int i=0; i<num_points; i++)
		{
			int32_t matID = state.points.matID(i);
			fwrite(&matID, sizeof(int32_t), 1, f);
		}
	}
	//*/
	
	for(int i = 0; i < nx; i++) { for(int j = 0; j < ny; j++) { for(int k = 0; k < nz; k++) {
		if(weights(i,j,k) > 0) {
			double color[3] = {colorField(i,j,k)[0], colorField(i,j,k)[1], colorField(i,j,k)[2]};
			fwrite(color, sizeof(double), 3, f);
		}
	}}}
	
	fclose(f);
}

void ColorGrid::computeColor( const SlVector3& x, SlVector3& color ) const
{
  // Determine which cell this point lies in
  unsigned indices[3] = {
	  unsigned(floor((x[0] - bbMin[0]) / h)),
	  unsigned(floor((x[1] - bbMin[1]) / h)),
	  unsigned(floor((x[2] - bbMin[2]) / h))
  };
  
  // Handle points on the 'right' boundary of the grid
  if( indices[0] + 1 >= (unsigned) nx ) color = SlVector3(1.0, 1.0, 1.0);
  if( indices[1] + 1 >= (unsigned) ny ) color = SlVector3(1.0, 1.0, 1.0);
  if( indices[2] + 1 >= (unsigned) nz ) color = SlVector3(1.0, 1.0, 1.0);

  // Compute the 'barycentric' coordinates of the point in the cell
  double bc[3] = {
	  (x[0] - (bbMin[0]+indices[0]*h)) / h,
	  (x[1] - (bbMin[1]+indices[1]*h)) / h,
	  (x[2] - (bbMin[2]+indices[2]*h)) / h
  };
  
  //std::cout << "computeColor(" << x << "[" << indices[0] << "," << indices[1] << "," << indices[2] << "]:" << std::endl;
  
  //std::cout << "[" << indices[0] << "," << indices[1] << "," << indices[2] << "], " << std::endl;
  
  // Grab the value of the distance field at each grid point
  const SlVector3 v000 = colorField( indices[0],     indices[1],     indices[2] );
  const SlVector3 v100 = colorField( indices[0] + 1, indices[1],     indices[2] );
  const SlVector3 v010 = colorField( indices[0],     indices[1] + 1, indices[2] );
  const SlVector3 v110 = colorField( indices[0] + 1, indices[1] + 1, indices[2] );
  const SlVector3 v001 = colorField( indices[0],     indices[1],     indices[2] + 1 );
  const SlVector3 v101 = colorField( indices[0] + 1, indices[1],     indices[2] + 1 );
  const SlVector3 v011 = colorField( indices[0],     indices[1] + 1, indices[2] + 1 );
  const SlVector3 v111 = colorField( indices[0] + 1, indices[1] + 1, indices[2] + 1 );
  
  const double w000 = weights( indices[0],     indices[1],     indices[2] );
  const double w100 = weights( indices[0] + 1, indices[1],     indices[2] );
  const double w010 = weights( indices[0],     indices[1] + 1, indices[2] );
  const double w110 = weights( indices[0] + 1, indices[1] + 1, indices[2] );
  const double w001 = weights( indices[0],     indices[1],     indices[2] + 1 );
  const double w101 = weights( indices[0] + 1, indices[1],     indices[2] + 1 );
  const double w011 = weights( indices[0],     indices[1] + 1, indices[2] + 1 );
  const double w111 = weights( indices[0] + 1, indices[1] + 1, indices[2] + 1 );
  
  //std::cout << "  " << v000 << ", " << v100 << ", " << v010 << ", " << v110 << ", " << v001 << ", " << v101 << ", " << v011 << ", " << v111 << std::endl; 
  
  // Compute the value of the distance field at the sample point with trilinear interpolation
  //trilinearInterpolation( v000, v100, v010, v110, v001, v101, v011, v111, bc, color );
  trilinearInterpolation( v000, w000, v100, w100, v010, w010, v110, w110, v001, w001, v101, w101, v011, w011, v111, w111, bc, color);
}

void ColorGrid::trilinearInterpolation( const SlVector3& v000, const SlVector3& v100, const SlVector3& v010, const SlVector3& v110,
                                                    const SlVector3& v001, const SlVector3& v101, const SlVector3& v011, const SlVector3& v111,
                                                    const double bc[3], SlVector3& v ) const
{  
  SlVector3 v00, v10, v01, v11, v0, v1;
  linearInterpolation( v000, v100, bc[0], v00 );
  linearInterpolation( v010, v110, bc[0], v10 );
  linearInterpolation( v001, v101, bc[0], v01 );
  linearInterpolation( v011, v111, bc[0], v11 );
  
  linearInterpolation( v00, v10, bc[1], v0 );
  linearInterpolation( v01, v11, bc[1], v1 );
  
  linearInterpolation( v0, v1, bc[2], v );
}

void ColorGrid::trilinearInterpolation( const SlVector3& v000, const double w000, const SlVector3& v100, const double w100,
													const SlVector3& v010, const double w010, const SlVector3& v110, const double w110,
                                                    const SlVector3& v001, const double w001, const SlVector3& v101, const double w101, 
													const SlVector3& v011, const double w011, const SlVector3& v111, const double w111, 
                                                    const double bc[3], SlVector3& v ) const
{  
  SlVector3 v00, v10, v01, v11, v0, v1;
  double w00, w10, w01, w11, w0, w1, w;
  linearInterpolation( v000, w000, v100, w100, bc[0], v00, w00 );
  linearInterpolation( v010, w010, v110, w110, bc[0], v10, w10 );
  linearInterpolation( v001, w001, v101, w101, bc[0], v01, w01 );
  linearInterpolation( v011, w011, v111, w111, bc[0], v11, w11 );
  
  linearInterpolation( v00, w00, v10, w10, bc[1], v0, w0 );
  linearInterpolation( v01, w01, v11, w11, bc[1], v1, w1 );
  
  linearInterpolation( v0, w0, v1, w1, bc[2], v, w );
}

void ColorGrid::linearInterpolation( const SlVector3& v0, const SlVector3& v1, const double& alpha, SlVector3& v ) const
{
  v = ( 1.0 - alpha ) * v0 + alpha * v1;
}

void ColorGrid::linearInterpolation( const SlVector3& v0, const double w0, const SlVector3& v1, const double w1, const double& alpha, SlVector3& v, double& w ) const
{
  double w_tot = w0*( 1.0 - alpha ) + w1*alpha;
  
  if(w_tot <= 0.0)
  {
	v = 0.0;
	w = 0.0;
  }
  else
  {
    v = (( 1.0 - alpha ) * w0 * v0 + alpha * w1 * v1) / w_tot;
    w = w0 * (1.0 - alpha) + w1 * alpha;
  }
}

