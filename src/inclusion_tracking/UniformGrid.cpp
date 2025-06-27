#include "UniformGrid.h"

UniformGrid3D::UniformGrid3D( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3i& in_res, const double in_cell_width )
: m_nAllocCells(0), m_nData( nullptr ), m_nAllocPerCell( nullptr ), m_IDs( nullptr )
{
  reallocation( in_min_coords, in_res, in_cell_width );
}

UniformGrid3D::~UniformGrid3D()
{
  for( int i=0; i<m_nAllocCells; i++ ) free( m_IDs[i] );
	free( m_IDs );
	free( m_nData );
	free( m_nAllocPerCell );
}

void UniformGrid3D::reallocation( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3i& in_res, const double in_cell_width )
{
  m_Res = in_res;
  m_CellWidth = in_cell_width;
  m_MinCoords = in_min_coords;
  
  const int newNCells = numCells();
  if( m_nAllocCells >= newNCells )
  {
    clearData();
    return;
  }
  
  m_nData = ( int* )realloc( m_nData, sizeof(int) * newNCells );
  for( int i=0; i<newNCells; i++ ) m_nData[i] = 0;
  
  m_nAllocPerCell = ( int* )realloc( m_nAllocPerCell, sizeof(int) * newNCells );
  for( int i=m_nAllocCells; i<newNCells; i++ ) m_nAllocPerCell[i] = 0;
  
  m_IDs = ( int** )realloc( m_IDs, sizeof(int*) * newNCells );
  for( int i=m_nAllocCells; i<newNCells; i++ ) m_IDs[i] = nullptr;
  
  m_nAllocCells = newNCells;
}
	
void UniformGrid3D::clearData()
{
  const int nCells = numCells();
	for( int i=0; i<nCells; i++ ) m_nData[i] = 0;
}

void UniformGrid3D::registerData( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3d& in_max_coords, int id )
{
	Eigen::Vector3i idm; getGridID( in_min_coords, idm );
	Eigen::Vector3i idM; getGridID( in_max_coords, idM );
	
  for( unsigned k=idm.z(); k<=idM.z(); k++ )
  {
    for( unsigned j=idm.y(); j<=idM.y(); j++ )
    {
      for( unsigned i=idm.x(); i<=idM.x(); i++ )
      {
        const unsigned flat_idx = getFlatIdx( Eigen::Vector3i{ i, j, k } );
        if( m_nAllocPerCell[flat_idx] <= m_nData[flat_idx] )
        {
          int newSize;
          if( m_nAllocPerCell[flat_idx] < 10 ) newSize = 16;
          else if( m_nAllocPerCell[flat_idx] < 2000 ) newSize = m_nAllocPerCell[flat_idx] * 2;
          else newSize = m_nAllocPerCell[flat_idx] + 1024;
			
          m_nAllocPerCell[flat_idx] = newSize;
          m_IDs[flat_idx] = ( int* )realloc( m_IDs[flat_idx], sizeof(int)*m_nAllocPerCell[flat_idx] );
        }
		
        m_IDs[flat_idx][m_nData[flat_idx]] = id;
        m_nData[flat_idx]++;
      }
		}
	}
}

void UniformGrid3D::registerPointData( const Eigen::Vector3d& coords, int id )
{
	Eigen::Vector3i idm; getGridID( coords, idm );
	
	const unsigned flat_idx = getFlatIdx( idm );
	if(m_nAllocPerCell[flat_idx] <= m_nData[flat_idx])
	{
		int newSize;
		if( m_nAllocPerCell[flat_idx] < 10 ) newSize = 16;
		else if( m_nAllocPerCell[flat_idx] < 2000 ) newSize = m_nAllocPerCell[flat_idx] * 2;
		else newSize = m_nAllocPerCell[flat_idx] + 1024;
			
		m_nAllocPerCell[flat_idx] = newSize;
		m_IDs[flat_idx] = ( int* )realloc( m_IDs[flat_idx], sizeof(int)*m_nAllocPerCell[flat_idx] );
	}
		
	m_IDs[flat_idx][m_nData[flat_idx]] = id;
	m_nData[flat_idx]++;
}

void UniformGrid3D::getGridID( const Eigen::Vector3d& coords, Eigen::Vector3i& grid_idx ) const
{
  const Eigen::Vector3d _idx = ( coords - m_MinCoords ) / m_CellWidth;
	grid_idx(0) = std::max<int>( 0, std::min<int>( m_Res.x()-1, int(std::floor(_idx(0))) ) );
	grid_idx(1) = std::max<int>( 0, std::min<int>( m_Res.y()-1, int(std::floor(_idx(1))) ) );
  grid_idx(2) = std::max<int>( 0, std::min<int>( m_Res.z()-1, int(std::floor(_idx(2))) ) );
}

void UniformGrid3D::getGridIDRange( const Eigen::Vector3d& in_min_coords, const Eigen::Vector3d& in_max_coords, Eigen::Vector3i& min_grid_idx, Eigen::Vector3i& max_grid_idx ) const
{
	getGridID( in_min_coords, min_grid_idx );
	getGridID( in_max_coords, max_grid_idx );
}

void UniformGrid3D::getIDs( const Eigen::Vector3i& grid_idx, int* out_nData, const int** out_IDs ) const
{
	if((grid_idx(0) >= m_Res.x()) || (grid_idx(1) >= m_Res.y()) || (grid_idx(2) >= m_Res.z()))
	{
		*out_nData = 0;
		*out_IDs = NULL;
	}
	else
	{
		const unsigned flat_idx = getFlatIdx( grid_idx );
		*out_nData = m_nData[flat_idx];
		*out_IDs = m_IDs[flat_idx];
	}
}

Eigen::Vector3i UniformGrid3D::getFirstNonEmptyCell( int* out_nData, const int** out_IDs ) const
{
  for( int k=0; k<m_Res.z(); k++ )
  {
    for( int j=0; j<m_Res.y(); j++ )
    {
      for( int i=0; i<m_Res.x(); i++ )
      {
        const unsigned flat_idx = getFlatIdx( Eigen::Vector3i{ i, j, k } );
        if( m_nData[flat_idx] > 0 )
        {
          *out_nData = m_nData[flat_idx];
          *out_IDs = m_IDs[flat_idx];
          return Eigen::Vector3i { i, j, k };
        }
      }
    }
  }
  
  *out_nData = 0;
  *out_IDs = NULL;
  return Eigen::Vector3i { m_Res.x(), m_Res.y(), m_Res.z() };
}

Eigen::Vector3i UniformGrid3D::getNextNonEmptyCell( const Eigen::Vector3i& in_prev_cell, int* out_nData, const int** out_IDs ) const
{
  int x_start = in_prev_cell(0) + 1;
  int y_start = in_prev_cell(1);
  for( int k=in_prev_cell(2); k<m_Res.z(); k++ )
  {
    for( int j=y_start; j<m_Res.y(); j++ )
    {
      for( int i=x_start; i<m_Res.x(); i++ )
      {
        const unsigned flat_idx = getFlatIdx( Eigen::Vector3i{ i, j, k } );
        if( m_nData[flat_idx] > 0 )
        {
          *out_nData = m_nData[flat_idx];
          *out_IDs = m_IDs[flat_idx];
          return Eigen::Vector3i { i, j, k };
        }
      }
      x_start = 0;
    }
    y_start = 0;
  }
  
  *out_nData = 0;
  *out_IDs = NULL;
  return Eigen::Vector3i { m_Res.x(), m_Res.y(), m_Res.z() };
}

Eigen::Vector3i UniformGrid3D::getFirstCellWithMultipleElements( int* out_nData, const int** out_IDs ) const
{
  for( int k=0; k<m_Res.z(); k++ )
  {
    for( int j=0; j<m_Res.y(); j++ )
    {
      for( int i=0; i<m_Res.x(); i++ )
      {
        const unsigned flat_idx = getFlatIdx( Eigen::Vector3i{ i, j, k } );
        if( m_nData[flat_idx] > 1 )
        {
          *out_nData = m_nData[flat_idx];
          *out_IDs = m_IDs[flat_idx];
          return Eigen::Vector3i { i, j, k };
        }
      }
    }
  }
  
  *out_nData = 0;
  *out_IDs = NULL;
  return Eigen::Vector3i { m_Res.x(), m_Res.y(), m_Res.z() };
}

Eigen::Vector3i UniformGrid3D::getNextCellWithMultipleElements( const Eigen::Vector3i& in_prev_cell, int* out_nData, const int** out_IDs ) const
{
  int x_start = in_prev_cell(0) + 1;
  int y_start = in_prev_cell(1);
  for( int k=in_prev_cell(2); k<m_Res.z(); k++ )
  {
    for( int j=y_start; j<m_Res.y(); j++ )
    {
      for( int i=x_start; i<m_Res.x(); i++ )
      {
        const unsigned flat_idx = getFlatIdx( Eigen::Vector3i{ i, j, k } );
        if( m_nData[flat_idx] > 1 )
        {
          *out_nData = m_nData[flat_idx];
          *out_IDs = m_IDs[flat_idx];
          return Eigen::Vector3i { i, j, k };
        }
      }
      x_start = 0;
    }
    y_start = 0;
  }
  
  *out_nData = 0;
  *out_IDs = NULL;
  return Eigen::Vector3i { m_Res.x(), m_Res.y(), m_Res.z() };
}

bool UniformGrid3D::isAtEnd( const Eigen::Vector3i& cell ) const
{
  return cell(2) >= m_Res.z();
}

int UniformGrid3D::getFlatIdx( const Eigen::Vector3i& in_grid_idx ) const
{
  return in_grid_idx.z() * m_Res.x() * m_Res.y() + in_grid_idx.y() * m_Res.x() + in_grid_idx.x();
}

int UniformGrid3D::numCells() const
{
  return m_Res.x() * m_Res.y() * m_Res.z();
}
