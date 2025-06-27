#ifndef UniformGrid_h
#define UniformGrid_h

#include <Eigen/Core>
#include <Eigen/Dense>

class UniformGrid3D
{
	UniformGrid3D();
public:
  UniformGrid3D( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3i& in_res, const double in_cell_width);
	~UniformGrid3D();
  
  void reallocation( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3i& in_res, const double in_cell_width );
	
	void clearData();
  void registerData( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3d& in_max_coords, int id );
	void registerPointData( const Eigen::Vector3d& coords, int id );
	void getGridID( const Eigen::Vector3d& coords, Eigen::Vector3i& grid_idx ) const;
	void getGridIDRange( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3d& in_max_coords, Eigen::Vector3i& min_grid_idx, Eigen::Vector3i& max_grid_idx ) const;
	void getIDs( const Eigen::Vector3i& grid_idx, int* out_nData, const int** out_IDs ) const;
	
  Eigen::Vector3i getFirstNonEmptyCell( int* out_nData, const int** out_IDs ) const;
  Eigen::Vector3i getNextNonEmptyCell( const Eigen::Vector3i& in_prev_cell, int* out_nData, const int** out_IDs ) const;
  Eigen::Vector3i getFirstCellWithMultipleElements( int* out_nData, const int** out_IDs ) const;
  Eigen::Vector3i getNextCellWithMultipleElements( const Eigen::Vector3i& in_prev_cell, int* out_nData, const int** out_IDs ) const;
  bool isAtEnd( const Eigen::Vector3i& cell ) const;
  
protected:
  
  int getFlatIdx( const Eigen::Vector3i& in_grid_idx ) const;
  int numCells() const;
  
	Eigen::Vector3i m_Res;
	Eigen::Vector3d m_MinCoords;
  double m_CellWidth;
	
  int m_nAllocCells;
	int* m_nData;
	int* m_nAllocPerCell;
	int** m_IDs;
};

#endif
