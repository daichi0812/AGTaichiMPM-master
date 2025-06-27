// Copyright (c) 2011, Regents of the University of Utah
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <getopt.h>
#include "smoothingGrid.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <float.h>
#include <stdint.h>

#include <cassert>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "MarchingCubes.h"

bool loadMPMFile( const std::string& file_name, std::string& load_status, std::vector<SlVector3>& points, std::vector<SlVector3>& velocities, std::vector<double>& radii )
{
  // Attempt to open the user specified file
  std::ifstream obj_file( file_name.c_str() );
  if( !obj_file.good() )
  {
    load_status = "Failed to open file";
    return false;
  }
  
  // Read the number of points
  int32_t num_points;
  obj_file.read( (char*) &num_points, sizeof( int32_t ) );
  if( num_points < 0 )
  {
    load_status = "Data contains negative number of points";
    return false;
  }

  // Read the points
  std::vector<Eigen::Array3f> parsed_points( num_points );
  for( int32_t point_num = 0; point_num < num_points; ++point_num )
  {
    if( !obj_file.good() )
    {
      // TODO: Add point number here
      load_status = "Failed to read MPM particle position";
      return false;
    }
    obj_file.read( (char*) parsed_points[point_num].data(), 3 * sizeof( float ) );
    //std::cout << point_num << ": " << parsed_points[point_num].transpose() << std::endl;
  }

  // Read the radii
  std::vector<float> parsed_radii( num_points );
  for( int32_t point_num = 0; point_num < num_points; ++point_num )
  {
    if( !obj_file.good() )
    {
      // TODO: Add point number here
      load_status = "Failed to read MPM particle radius";
      return false;
    }
    obj_file.read( (char*) &parsed_radii[point_num], sizeof( float ) );
    if( parsed_radii[point_num] < 0 )
    {
      load_status = "Data contains negative radius";
      return false;
    }
    //std::cout << point_num << ": " << parsed_radii[point_num] << std::endl;
  }
  
  // Read the velocities
  std::vector<Eigen::Array3f> parsed_velocities( num_points );
  for( int32_t point_num = 0; point_num < num_points; ++point_num )
  {
    if( !obj_file.good() )
    {
      // TODO: Add point number here
      load_status = "Failed to read MPM particle velocity";
      return false;
    }
    obj_file.read( (char*) parsed_velocities[point_num].data(), 3 * sizeof( float ) );
    parsed_velocities[point_num].setConstant( 0.0 );
  }
  
  // Copy the data over to the double storage
  points.resize( num_points );
  velocities.resize( num_points );
  radii.resize( num_points );
  for( int32_t point_num = 0; point_num < num_points; ++point_num )
  {
    points[point_num] =  SlVector3( (double) parsed_points[point_num].x(), (double) parsed_points[point_num].y(), (double) parsed_points[point_num].z() );
    radii[point_num] = (double) parsed_radii[point_num];
    velocities[point_num] =  SlVector3( (double) parsed_velocities[point_num].x(), (double) parsed_velocities[point_num].y(), (double) parsed_velocities[point_num].z() );
  }
  
  std::cout << "MPM file loaded" << std::endl;
  
  return true;
}



template<class T>
std::string convertToString( const T& tostring )
{
	std::string out_string;
	std::stringstream ss;
	
	ss << tostring;
	ss >> out_string;
	
	return out_string;
}

// TODO: Move obj and stl export functions to their own file

enum MeshFormat
{
  OBJ,
  STL
};

bool exportOBJMesh( const std::string& output_mesh_name, const std::vector<Eigen::Array3i>& triangles, const std::vector<Eigen::Vector3d>& vertices, std::string& save_status )
{
  // Attempt to open the output file
  std::ofstream obj_file( output_mesh_name.c_str() );
  assert( obj_file.good() );
  if( !obj_file.good() )
  {
    save_status = "Failed to open file";
    return false;
  }

  for( std::vector<Eigen::Vector3d>::size_type vrt_idx = 0; vrt_idx < vertices.size(); ++vrt_idx )
  {
    obj_file << "v " << std::setprecision(200) << vertices[vrt_idx].x() << " " << std::setprecision(200) << vertices[vrt_idx].y() << " " << std::setprecision(200) << vertices[vrt_idx].z() << std::endl;
    if( !obj_file.good() )
    {
      save_status = "Failed to save vertex " + convertToString(vrt_idx);
      return false;
    }
  }

  for( std::vector<Eigen::Vector3i>::size_type tri_idx = 0; tri_idx < triangles.size(); ++tri_idx )
  {
    obj_file << "f " << (triangles[tri_idx]+1).transpose() << std::endl;
    if( !obj_file.good() )
    {
      save_status = "Failed to save triangle " + convertToString(tri_idx);
      return false;
    }
  }

  obj_file.close();

  return true;
}

bool exportSurface( const std::string& output_mesh_name, const MeshFormat& output_format, const SmoothingGrid& grid, std::string& save_status )
{
  // Marching cubes visualization of the signed distance field
  std::vector<Eigen::Array3i> triangles;
  std::vector<Eigen::Vector3d> vertices;

  {
    // Grab vertex data from the buffer
    Eigen::VectorXd phi = Eigen::VectorXd::Zero( grid.nx * grid.ny * grid.nz );
    for( unsigned int i = 0; i < grid.phi.nx(); ++i )
    {
      for( unsigned int j = 0; j < grid.phi.ny(); ++j )
      {
        for( unsigned int k = 0; k < grid.phi.nz(); ++k )
        {
          phi( i * grid.phi.ny() * grid.phi.nz() + j * grid.phi.nz() + k ) = grid.phi( i, j, k );
        }
      }
    }

    // Extract a mesh from the signed distance field
    MarchingCubes mc( grid.nx, grid.ny, grid.nz );
    // Use the original marching cubes algorithm
    mc.set_method( true );
    // Use the points we loaded
    mc.set_ext_data( phi.data() );
    // Initialize some storage before the solve
    mc.init_all();
    // Run marching cubes
    {
      const double level_set_value = 0;
      mc.run( level_set_value );
    }

    // Allocate space for the mesh
    triangles.resize( mc.ntrigs() );
    vertices.resize( mc.nverts() );

    SlVector3 bbMin, bbMax;
    bbMin[0] = bbMin[1] = bbMin[2] = DBL_MAX;
    bbMax[0] = bbMax[1] = bbMax[2] = -DBL_MAX;
	
    // Extract the vertices
    for( int i = 0; i < mc.nverts(); ++i )
    {
      const Vertex* cur_vert = mc.vert( i );
      assert( cur_vert != NULL );
      vertices[i] << cur_vert->x, cur_vert->y, cur_vert->z;
//      std::swap( vertices[i].x(), vertices[i].z() );
//      vertices[i].z() *= -1.0;
      // Rescale to be same size as mesh and re-center
      vertices[i] = vertices[i].array() * grid.h + Eigen::Array3d( grid.bbMin.x(), grid.bbMin.y(), grid.bbMin.z() );
      
      double pos[3] = { vertices[i][0], vertices[i][1], vertices[i][2] };
      
      bbMin[0] = std::min(bbMin[0], vertices[i][0]);
      bbMin[1] = std::min(bbMin[1], vertices[i][1]);
      bbMin[2] = std::min(bbMin[2], vertices[i][2]);
      bbMax[0] = std::max(bbMax[0], vertices[i][0]);
      bbMax[1] = std::max(bbMax[1], vertices[i][1]);
      bbMax[2] = std::max(bbMax[2], vertices[i][2]);
    }
    std::cout << std::endl;
    
    std::cout<<"Bounding box (after mc) is "<<bbMin<<" X "<<bbMax<<std::endl;
    
    // Extract the triangles
    for( int i = 0; i < mc.ntrigs(); ++i )
    {
      const Triangle* cur_tri = mc.trig(i);
      assert( cur_tri != NULL );
      triangles[i] << cur_tri->v1, cur_tri->v2, cur_tri->v3;
    }

    // Clean up
    mc.clean_all();
  }

  bool mesh_saved;
  switch( output_format )
  {
    case OBJ:
      mesh_saved = exportOBJMesh( output_mesh_name, triangles, vertices, save_status );
      break;
    default:
      mesh_saved = false;
      save_status = "Invalid mesh type specified";
  }
  
  return mesh_saved;
}



int main( int argc, char** argv )
{
	unsigned int flags = 0;
	static int verboseFlag = 0;
	bool helpFlag = false;
	int iterLaplace = 15;
  int iterBiharmonic = 500;
  int redistanceFrequency = 50;
	double rmin = -DBL_MAX;
  double rmax = -DBL_MAX;
  double rinit = -DBL_MAX;
  double velGain = 1.0;
  double dtLaplace = -DBL_MAX;
  double dtBiharmonic = -DBL_MAX;
  double dtBiharmonicGain = 1.0;
	double maxStretch = 4;
  double rratio = 4;
  double radius_scale = 1.0;
  
  std::string output_dir_name;


	static struct option long_options[] =
  {
		{"help", no_argument, 0, 'h'},
		{"verbose", no_argument, &verboseFlag, 1},
		{"lapiter", required_argument, 0, 'l'},
		{"bihiter", required_argument, 0, 'b'},
		{"laptime", required_argument, 0, 't'},
		{"bihtime", required_argument, 0, 'T'},
		{"rmax", required_argument, 0, 'M'},
		{"rmin", required_argument, 0, 'm'},
		{"redist", required_argument, 0, 'f'},
		{"variable_radius", no_argument, 0, 'V'},
		{"neighbor_anisotropy", no_argument, 0, 'n'},
		{"velocity_anisotropy", required_argument, 0, 'v'},
		{"timestepConst", required_argument, 0, 'B'},
		{"maxStretch", required_argument, 0, 's'},
		{"rratio", required_argument, 0, 'r'},
		{"rinit", required_argument, 0, 'i'},
		{"rscale", required_argument, 0, 'x'},
		{0, 0, 0, 0} 
	};


	while( 1 )
  {
		int option_index = 0;
		int c = getopt_long( argc, argv, "hl:b:t:T:M:m:f:Vnv:B:s:r:g:i:x:", long_options, &option_index );
		if( c == -1 ) break;
		switch( c )
    {
		case 0:
			if( long_options[option_index].flag != 0 )
				break;
			break;

		case 'h':
			helpFlag = true;
			break;
			
		case 'V':
			flags |= SmoothingGrid::VARIABLE_RADIUS;
			break;
			
		case 'n':
			flags |= SmoothingGrid::NEIGHBOR_ANISOTROPY;
			break;
			
		case 'v':
			flags |= SmoothingGrid::VELOCITY_ANISOTROPY;
			velGain = atof(optarg);
			break;

		case 't':
			dtLaplace = atof(optarg);
			break;
			
		case 'T':
			dtBiharmonic = atof(optarg);
			break;
			
		case 'M':
			rmax = atof(optarg);
			break;
			
		case 'm':
			rmin = atof(optarg);
			break;
			
		case 'f':
			redistanceFrequency = atoi(optarg);
			break;
			
		case 'l':
			iterLaplace = atoi(optarg);
			break;
			
		case 'b':
			iterBiharmonic = atoi(optarg);
			break;

		case 'r':
			rratio = atof(optarg);
			break;
				
		case 'B':
			dtBiharmonicGain = atof(optarg);
			break;

		case 's':
			maxStretch = atof(optarg);
			break;
			
		case 'i':
			rinit = atof(optarg);
			break;
		
    case 'x':
      radius_scale = atof(optarg);
      break;

    case '?':
			break;
			
		default:
			abort ();
		}
	}		

	if( verboseFlag ) flags |= SmoothingGrid::VERBOSE;

	if( helpFlag || argc == 1 )
  {
		std::cout << "Welcome! " << std::endl;
		std::cout << "Usage: " << argv[0] << " grid_spacing inputFile outputFile meshFormat" << std::endl;
		std::cout<<"Available switches are......."<<std::endl;
		std::cout<<"-h/--help -> display this message "<<std::endl;
		std::cout<<"--verbose -> output information in command line "<<std::endl;
		std::cout<<"-r/--rratio number -> r_max =  number * r_min, Default value is 4\n\t(over-riden by -M)"<<std::endl;
		std::cout<<"-i/--rinit number -> r_init = number * r_min, otherwise\n\tr_init = 0.5 * (r_min + r_max) "<<std::endl;

		std::cout<<"-V/--variable_radius min max -> The particles have variable radiuses,\n\tin this case r_min, r_max, and r_init are multipliers for the individual particle radii, you will want to use -m and -M with this option"<<std::endl;
		std::cout<<"-n/--neighbor_anisotropy -> Turn on neighborhood based anisotropy "<<std::endl;
		std::cout<<"-v/--velocity_anisotropy number -> Turn on velocity-based anisotropy,\n\tthe number is a gain on the amount of anisotropy,\n\tlarger values lead to more anisotropy "<<std::endl;
		std::cout<<"-s/--maxStretch number -> maximum amount of anisotropy (condition number of G),\n\tDefault value is 4 "<<std::endl;

		std::cout<<"-l/--lapiter number -> Number of laplacian smoothing passes,\n\tDefault value is 15"<<std::endl;
		std::cout<<"-b/--bihiter number -> Number of biharmonic smoothing passes,\n\tDefault value is 500"<<std::endl;
		std::cout<<"-B/--timestepConst number -> A multiplier for the biharmonic timestep "<<std::endl;
		std::cout<<"-t/--laptime number -> Timestep for laplacian smoothing\n\t Deafult value is 0.1*h^2"<<std::endl;
		std::cout<<"-T/--bihtime number -> Timestep for biharmonic smoothing,\n\tDefault value is 0.01*h^4"<<std::endl;
		std::cout<<"-r/--redist number -> Frequency of redistancing, Default value is 50"<<std::endl;

		std::cout<<"-m/--rmin number -> Minimum radius,\n\tDefault value is (0.5 * sqrt(3) * grid_spacing)"<<std::endl;
		std::cout<<"-M/--rmax number -> Maximum radius, Default value is (4 * rmin)"<<std::endl;
		
    std::cout<<"-x/--rscale number -> Multiple the default radius by amount"<<std::endl;
    
		return EXIT_SUCCESS;
	}

  //std::cout << "Radius scale: " << radius_scale << std::endl;

  assert( optind < argc );
  double h = atof(argv[optind++]);
  assert( optind < argc );
  char* infname =  argv[optind++];
  assert( optind < argc );
  char* outfname = argv[optind++];
  assert( optind < argc );
  MeshFormat output_mesh_format;
  {
    const std::string output_mesh_format_name = argv[optind++];
    if( "obj" == output_mesh_format_name )
    {
      output_mesh_format = OBJ;
    }
    else if( "stl" == output_mesh_format_name )
    {
      output_mesh_format = STL;
      std::cout << "stl support is omitted. please use 'obj' instead." << std::endl;
      return EXIT_FAILURE;
    }
    else
    {
      std::cerr << "Invalid mesh format specified. Valid options are: obj, stl" << std::endl;
      return EXIT_FAILURE;
    }
  }
  
//--------------------get name---------------------------------------//
  static std::string g_output_dir_name;


//--------------------export mesh by file name-----------------------//
	if( rmin == -DBL_MAX )
  {
		rmin = 0.86603 * h; // 0.5*sqrt(3)*h
		// there could be a stretch up to this amount, we want to make sure that particles
		// all touch at least one grid point, so make the radius bigger...
		if (flags & SmoothingGrid::NEIGHBOR_ANISOTROPY)
    {
			rmin *= sqrt(maxStretch); 
		}
    else if (flags & SmoothingGrid::VELOCITY_ANISOTROPY)
    {
			rmin *= cbrt(maxStretch); 
		}
	}
	if (rmax == -DBL_MAX) rmax = rratio*rmin;
	if (rinit == -DBL_MAX) rinit = 0.5*(rmin+rmax);
	else if (!(flags & SmoothingGrid::VARIABLE_RADIUS)) rinit *= rmin;
	if (dtLaplace == -DBL_MAX) dtLaplace = 0.1*h*h;
	if (dtBiharmonic == -DBL_MAX) dtBiharmonic = 0.01*dtBiharmonicGain*h*h*h*h;

	timeval startTime, endTime;
	
	std::vector<SlVector3> particles, velocities, colors;
	std::vector<double> radii;
	gettimeofday(&startTime, NULL);
	//readfile(infname, particles, radii, velocities);
  {
    std::string load_status;
    const bool mpm_file_loaded = loadMPMFile( infname, load_status, particles, velocities, radii );
    if( !mpm_file_loaded )
    {
      std::cerr << "Failed to load MPM file " << infname << ": " << load_status << std::endl;
      std::cerr << "Exiting." << std::endl;
      return EXIT_FAILURE;
    }
  }
	gettimeofday(&endTime, NULL);
	if (verboseFlag) std::cout<<"Reading the file took "<<(endTime.tv_sec-startTime.tv_sec)+
		(endTime.tv_usec-startTime.tv_usec)*1.0e-6<<std::endl;

  // Rescale radii
  for( std::vector<double>::size_type prt_num = 0; prt_num < radii.size(); ++prt_num )
  {
    radii[prt_num] *= radius_scale;
    //std::cout << rmin << " " << radii[prt_num] << " " << rmax << std::endl;
  }
  std::cout << "r_bracket: " << rmin << " " << radii[0] << " " << rmax << std::endl;


	gettimeofday(&startTime, NULL);
	SmoothingGrid grid(h, rmin, rmax, rinit, velGain, maxStretch, flags, particles, radii, velocities);
	gettimeofday(&endTime, NULL);
	if (verboseFlag) std::cout<<"Initialization took "<<(endTime.tv_sec-startTime.tv_sec)+
		(endTime.tv_usec-startTime.tv_usec)*1.0e-6<<std::endl;

	gettimeofday(&startTime, NULL);
	grid.doLaplacianSmoothing(iterLaplace, dtLaplace, redistanceFrequency);
	gettimeofday(&endTime, NULL);
	if (verboseFlag) std::cout<<"Laplacian Smoothing took "<<(endTime.tv_sec-startTime.tv_sec)+
		(endTime.tv_usec-startTime.tv_usec)*1.0e-6<<std::endl;
	
	gettimeofday(&startTime, NULL);
	grid.doBiharmonicSmoothing(iterBiharmonic, dtBiharmonic, redistanceFrequency);
	gettimeofday(&endTime, NULL);
	if (verboseFlag) std::cout<<"Biharmonic Smoothing took "<<(endTime.tv_sec-startTime.tv_sec)+
		(endTime.tv_usec-startTime.tv_usec)*1.0e-6<<std::endl;
	
	gettimeofday(&startTime, NULL);
  // TODO: Use this return status
  std::string status;
	//const bool saved_successful = exportSurface( outfname, output_mesh_format, grid, colorGrid, status );
	const bool saved_successful = exportSurface( outfname, output_mesh_format, grid, status );
	gettimeofday(&endTime, NULL);
	if (verboseFlag) std::cout<<"Dumping file with marching tet took "<<(endTime.tv_sec-startTime.tv_sec)+ 
		(endTime.tv_usec-startTime.tv_usec)*1.0e-6<<std::endl;

	return EXIT_SUCCESS;
  
  //------------------export-file-ended-------------------------
  
}
