//
//  main.cpp
//  InclusionTracker
//
//  Created by Yonghao Yue on 2022/01/15.
//

#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <Eigen/Core>
#include "UniformGrid.h"
#include <cmath>

struct Particles
{
  int num_points;
  std::vector<Eigen::Vector3f> x;
  std::vector<Eigen::Matrix3f> rot;
};

struct Inclusions
{
  int num_inclusions;
  std::vector<int> idx;
  std::vector<Eigen::Vector3d> x;
  std::vector<Eigen::Matrix3d> rot;
  std::vector<Eigen::Matrix3d> initial_rot;
};

struct IndependentTriangle
{
  Eigen::Vector3d vertices[3];
};

std::random_device rnd;
std::mt19937 mt( rnd() );
std::uniform_real_distribution<> dist( 0.0, 1.0 );

void clearParticleData( Particles& out_Particles )
{
  out_Particles.num_points = 0;
  out_Particles.x.clear();
  out_Particles.rot.clear();
}

void resizeParticleData( int in_num_p, Particles& out_Particles )
{
  out_Particles.num_points = in_num_p;
  out_Particles.x.resize( in_num_p );
  out_Particles.rot.resize( in_num_p );
}

void clearInclusionData( Inclusions& out_Inclusions )
{
  out_Inclusions.num_inclusions = 0;
  out_Inclusions.idx.clear();
  out_Inclusions.x.clear();
  out_Inclusions.rot.clear();
  out_Inclusions.initial_rot.clear();
}

Eigen::Matrix3d randomRotationMatrix()
{
  Eigen::Matrix3d mat;
  
  const double theta = acos( 2.0 * dist(mt) - 1.0 );
  const double phi = 2.0 * M_PI * dist(mt);
  const double psi = 2.0 * M_PI * dist(mt);
  
  mat.col(0) = Eigen::Vector3d { sin( theta ) * cos( phi ), cos( theta ), sin( theta ) * sin( phi ) };
  
  Eigen::Vector3d plane_y;
  if( fabs( mat.col(0).x() ) > fabs( mat.col(0).y() ) && fabs( mat.col(0).x() ) > fabs( mat.col(0).z() ) )
  {
    plane_y << 0.0, 1.0, 0.0;
  }
  else
  {
    plane_y << 1.0, 0.0, 0.0;
  }
  
  plane_y = ( plane_y - mat.col(0).dot(plane_y) * mat.col(0) ).normalized();
  const Eigen::Vector3d plane_z = ( mat.col(0).cross( plane_y ) ).normalized();
  
  mat.col(1) = ( cos( psi ) * plane_y + sin( psi ) * plane_z ).normalized();
  mat.col(2) = ( mat.col(0).cross( mat.col(1) ) ).normalized();
  
  assert( fabs( mat.col(0).norm() - 1.0 ) < 1.0e-6 );
  assert( fabs( mat.col(1).norm() - 1.0 ) < 1.0e-6 );
  assert( fabs( mat.col(2).norm() - 1.0 ) < 1.0e-6 );
  assert( fabs( mat.col(0).dot( mat.col(1) ) ) < 1.0e-6 );
  assert( fabs( mat.col(1).dot( mat.col(2) ) ) < 1.0e-6 );
  assert( fabs( mat.col(2).dot( mat.col(0) ) ) < 1.0e-6 );
  assert( ( mat.col(2) - mat.col(0).cross( mat.col(1) ) ).norm() < 1.0e-6 );
    
  return mat;
}

void addInclusion( Inclusions& io_Inclusions, const Particles& in_Particles, const int in_pidx )
{
  io_Inclusions.num_inclusions++;
  io_Inclusions.idx.push_back( in_pidx );
  io_Inclusions.x.push_back( in_Particles.x[in_pidx].cast<double>() );
  
  Eigen::Matrix3d random_rotation = randomRotationMatrix();
  
  io_Inclusions.initial_rot.push_back( random_rotation );
  io_Inclusions.rot.push_back( in_Particles.rot[in_pidx].cast<double>() * random_rotation );
}

void updateInclusions( Inclusions& io_Inclusions, const Particles& in_Particles )
{
  for( int i=0; i<io_Inclusions.num_inclusions; i++ )
  {
    const int idx = io_Inclusions.idx[i];
    io_Inclusions.x[i] = in_Particles.x[idx].cast<double>();
    io_Inclusions.rot[i] = in_Particles.rot[idx].cast<double>() * io_Inclusions.initial_rot[i];
  }
}

void generateInclusionMarkers( const Inclusions& in_Inclusions, std::vector<IndependentTriangle>& out_Markers, const double in_gap )
{
  out_Markers.clear();
  // parameters; can be used to encode sizes, types, ...
  const double s = in_gap * 0.5; // factor (0.5) can be replaced by a scaling factor
  const double t = s * sqrt( 3.0 ) * 0.5; //factor (0.5*sqrt(3)) can be replaced by a one encoding the type
  const double kappa = 0.5; // can be chosen to encode other information
  
  for( int i=0; i<in_Inclusions.num_inclusions; i++ )
  {
    // the varycenter of the generated triangle encodes the center position of the inclusions
    // v[1] - v[0] of the generated triangle encodes the x axis;
    // the normal of the generated triangle encodes the y axis;
    // z axis = z axis cross y axis
    IndependentTriangle tri;
    tri.vertices[0] = in_Inclusions.x[i] - ( ( 1.0 + kappa ) * s / 3.0 ) * in_Inclusions.rot[i].col(0) + t / 3.0 * in_Inclusions.rot[i].col(2);
    tri.vertices[1] = tri.vertices[0] + s * in_Inclusions.rot[i].col(0);
    tri.vertices[2] = tri.vertices[0] + kappa * s * in_Inclusions.rot[i].col(0) - t * in_Inclusions.rot[i].col(2);
    
    assert( ( ( tri.vertices[0] + tri.vertices[1] + tri.vertices[2] ) / 3.0 - in_Inclusions.x[i] ).norm() < 1.0e-6 );
    assert( ( ( tri.vertices[1] - tri.vertices[0] ).normalized() - in_Inclusions.rot[i].col(0) ).norm() < 1.0e-6 );
    assert( ( ( ( tri.vertices[1] - tri.vertices[0] ).cross( tri.vertices[2] - tri.vertices[0] ) ).normalized() - in_Inclusions.rot[i].col(1) ).norm() < 1.0e-6 );
    
    out_Markers.push_back( tri );
  }
}

void saveMakers( const char* in_FileName, const std::vector<IndependentTriangle>& in_Markers )
{
  FILE* f = fopen( in_FileName, "wt" );
  
  for( int i=0; i<in_Markers.size(); i++ )
  {
    for( int k=0; k<3; k++ )
    {
      fprintf( f, "v %.10f %.10f %.10f\n", in_Markers[i].vertices[k].x(), in_Markers[i].vertices[k].y(), in_Markers[i].vertices[k].z() );
    }
  }
  
  for( int i=0; i<in_Markers.size(); i++ )
  {
    fprintf( f, "f %d %d %d\n", i*3+1, i*3+2, i*3+3 );
  }
  
  fclose( f );
}

void loadVector( FILE* f, Eigen::Vector3f& out_v )
{
  for( int i=0; i<3; i++ )
  {
    float val = 0.0;
    fread( &val, sizeof( float ), 1, f );
    out_v(i) = val;
  }
}

void loadMatrix( FILE* f, Eigen::Matrix3f& out_mat )
{
  for( int j=0; j<3; j++ )
  {
    for( int i=0; i<3; i++ )
    {
      float val = 0.0;
      fread( &val, sizeof( float ), 1, f );
      out_mat.col(i)(j) = val;
    }
  }
}

void loadData( const char* in_FileName, Particles& out_Particles )
{
  clearParticleData( out_Particles );
  
  FILE *f = fopen( in_FileName, "rb" );
  
  int32_t num_p = 0;
  fread( &num_p, sizeof( int32_t ), 1, f );
  resizeParticleData( num_p, out_Particles );
  
  for( int i=0; i<num_p; i++ )
  {
    Eigen::Vector3f v;
    loadVector( f, v );
    out_Particles.x[i] = v;
  }
  
  for( int i=0; i<num_p; i++ )
  {
    float val = 0.0;
    fread( &val, sizeof( float ), 1, f );
  }
  
  for( int i=0; i<num_p; i++ )
  {
    Eigen::Vector3f v;
    loadVector( f, v );
  }
  
  for( int i=0; i<num_p; i++ )
  {
    int32_t val = 0.0;
    fread( &val, sizeof( int32_t ), 1, f );
  }
  
  for( int i=0; i<num_p; i++ )
  {
    Eigen::Matrix3f m;
    loadMatrix( f, m );
    out_Particles.rot[i] = m;
  }
}

UniformGrid3D* setupUniformGrid( const Particles& in_Particles, const double inclusion_gap )
{
  Eigen::Vector3d _min_coord {  1.0e33,  1.0e33,  1.0e33 };
  Eigen::Vector3d _max_coord { -1.0e33, -1.0e33, -1.0e33 };
  
  for( int i=0; i<in_Particles.num_points; i++ )
  {
    _min_coord = _min_coord.cwiseMin( in_Particles.x[i].cast<double>() );
    _max_coord = _max_coord.cwiseMax( in_Particles.x[i].cast<double>() );
  }
  
  const Eigen::Vector3d coord_center = ( _max_coord + _min_coord ) * 0.5;
  const Eigen::Vector3d _width = ( _max_coord - _min_coord ).array() + 4.0 * inclusion_gap;
  
  const Eigen::Vector3i res = ( _width / inclusion_gap ).unaryExpr( []( double s ){ return ceil(s); } ).cast<int>();
  
  const Eigen::Vector3d half_width = res.cast<double>() * inclusion_gap * 0.5;
  
  const Eigen::Vector3d min_coord = coord_center - half_width;
  // const Eigen::Vector3d max_coord = coord_center + half_width;
 
  UniformGrid3D* ugrd = new UniformGrid3D( min_coord, res, inclusion_gap );
  
  for( int i=0; i<in_Particles.num_points; i++ )
  {
    Eigen::Vector3d _min = in_Particles.x[i].cast<double>().array() - inclusion_gap;
    Eigen::Vector3d _max = in_Particles.x[i].cast<double>().array() + inclusion_gap;
    ugrd->registerData( _min, _max, i );
  }
  
  return ugrd;
}

void selectInclusions( const Particles& in_Particles, const double in_gap, Inclusions& out_Inclusions )
{
  std::set<int> active_set;
  for( int i=0; i<in_Particles.num_points; i++ )
    active_set.insert( i );
   
  std::set<int> inclusion_list;
  clearInclusionData( out_Inclusions );
  
  UniformGrid3D* ugrd = setupUniformGrid( in_Particles, in_gap );
  
  int chosen = std::min( in_Particles.num_points - 1, std::max( 0, int( in_Particles.num_points * dist(mt) ) ) );
  
  while( 1 )
  {
    auto q = active_set.find( chosen );
    if( q == active_set.end() )
    {
      std::cout << "This is a bug" << std::endl;
      exit(-1);
    }
    
    active_set.erase( q );
    
    inclusion_list.insert( chosen );
        
    Eigen::Vector3i gidx;
    ugrd->getGridID( in_Particles.x[chosen].cast<double>(), gidx );
    
    int nData; const int* IDs;
    ugrd->getIDs( gidx, &nData, &IDs );
    
    for( int k=0; k<nData; k++ )
    {
      auto p = active_set.find( IDs[k] );
      if( p != active_set.end() )
      {
        const double dist = ( in_Particles.x[chosen] - in_Particles.x[IDs[k]] ).norm();
        if( dist < in_gap )
          active_set.erase( p );
      }
    }
        
    if( active_set.empty() )
      break;
    
    chosen = *active_set.begin();
  }
  
  for( auto p: inclusion_list )
  {
    addInclusion( out_Inclusions, in_Particles, p );
  }
  
  std::cout << "Selected " << out_Inclusions.num_inclusions << " inclusions out of " << in_Particles.num_points << " material points" << std::endl;
  
  delete ugrd;
}

bool fileExists( const char* in_FileName )
{
  FILE* f = fopen( in_FileName, "r" );
  if( f != NULL )
  {
    fclose(f);
    return true;
  }
  return false;
}

void usage()
{
  std::cout << "InclusionTracker <material point template filename> <output inclusion object template filename> <distance>" << std::endl;
  std::cout << "  Example: InclusionTracker config_%06d.dat inclusion_%06d.obj 0.2" << std::endl;
}

int main( int argc, const char * argv[] )
{
  if( argc != 4 )
  {
    usage();
    exit(0);
  }
  
  char template_material_point_filename[256]; //"config_%06d.dat";
  strcpy( template_material_point_filename, argv[1] );
  char template_inclusion_filename[256]; //"inclusion_%06d.obj";
  strcpy( template_inclusion_filename, argv[2] );
  
  const double gap = atof( argv[3] );
  
  char buf_filename[256];
  int count = 0;
  sprintf( buf_filename, template_material_point_filename, count );
  if( !fileExists( buf_filename ) )
  {
    std::cout << "Material point file " << buf_filename << " does not exist." << std::endl;
    exit(-1);
  }
  
  std::cout << "Step1: selecting inclusions" << std::endl;
  Particles particles;
  loadData( buf_filename, particles );
  Inclusions inclusions;
  selectInclusions( particles, gap, inclusions );
  
  std::cout << "Step2: tracking inclusions" << std::endl;
  std::vector<IndependentTriangle> markers;
  while(1)
  {
    generateInclusionMarkers( inclusions, markers, gap );
    sprintf( buf_filename, template_inclusion_filename, count );
    saveMakers( buf_filename, markers );
    
    count++;
    sprintf( buf_filename, template_material_point_filename, count );
    if( !fileExists( buf_filename ) )
    {
      std::cout << "Processed " << count << " files in total." << std::endl;
      break;
    }
    
    loadData( buf_filename, particles );
    updateInclusions( inclusions, particles );
  }
  
  return 0;
}
