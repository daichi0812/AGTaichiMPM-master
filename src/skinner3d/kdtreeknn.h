#ifndef __KD_TREE_KNN_H__
#define __KD_TREE_KNN_H__

#include <vector>
#include <algorithm>

#include "priorityqueue.h"
#include "simplepriorityarray.h"

struct SMinMaxBV
{
	double e_min[3];
	double e_max[3];
};

struct SPrim
{
	unsigned long id;
	double coord;
};

struct SKDTreeNode
{
	double pos[3];
	SKDTreeNode* firstChildPtr;
	unsigned long id;
	unsigned long dim;
};

struct SQueueElem
{
	double key;
	double nearest_pos[3];
	const SKDTreeNode* ptr;
};

template <typename T>
struct SPriorityArrayElem
{
	double key;
	T* ptr;
};

struct P_sort_functor_cmp
{
	inline bool operator()(const SPrim& in_lhs, const SPrim& in_rhs)
	{
		if(in_lhs.coord < in_rhs.coord)
			return true;
		else if((in_lhs.coord == in_rhs.coord) && (in_lhs.id < in_rhs.id))
			return true;
		else
			return false;
	}
};

struct SInternalKDTreeNode
{
	float pos[3];
	unsigned long id;
	SInternalKDTreeNode* child[2];
	unsigned long dim;
	std::vector<SPrim>* list[3];
};

inline CPriorityQueue<SQueueElem>* genKdTreeNodeQueue(unsigned int size = 1024 * 16)
{
	SQueueElem min_q; min_q.key = -1.0;
	return new CPriorityQueue<SQueueElem>(size, min_q);
}

template <typename T, int K>
class CKDTreeKNN
{
public:
	CKDTreeKNN()
	{
		m_MaxAvailableNodes = 10;
		m_nNodes = 0;
		m_Balanced = false;
		m_Nodes = (T*)malloc(sizeof(T)*m_MaxAvailableNodes);
		m_KDTreeNodes = NULL;
		m_InternalKDTreeNode = NULL;
	}
	
	~CKDTreeKNN()
	{
		if(m_Nodes != NULL)
			free(m_Nodes);
		
		if(m_KDTreeNodes != NULL)
			free(m_KDTreeNodes);
	}
	
	void storeNode(const T& in_Node)
	{
		if(m_nNodes >= m_MaxAvailableNodes)
		{
			m_MaxAvailableNodes = std::min(2*m_MaxAvailableNodes, m_MaxAvailableNodes+1024*1024);
			m_Nodes = (T*)realloc(m_Nodes, sizeof(T)*m_MaxAvailableNodes);
		}
		m_Nodes[m_nNodes++] = in_Node;
		m_Balanced = false;
	}
	
	void query(const double pos[3], const double max_dist2, CPriorityQueue<SQueueElem>* io_NodeQueue, CSimplePriorityArray<SPriorityArrayElem<T>, K>* io_ElemArray) const
	{
		const double eps = 0.0000000001;
		io_ElemArray->clear();
		
		if(!m_Balanced)
		{
			std::cout << "forgot to balance(). returning null..." << std::endl;
			return;
			//balance();
		}
		
		io_NodeQueue->clear();
		
		SQueueElem the_Root;
		the_Root.ptr = &m_KDTreeNodes[1];
		the_Root.nearest_pos[0] = std::min(m_BV.e_max[0], std::max(m_BV.e_min[0], pos[0]));
		the_Root.nearest_pos[1] = std::min(m_BV.e_max[1], std::max(m_BV.e_min[1], pos[1]));
		the_Root.nearest_pos[2] = std::min(m_BV.e_max[2], std::max(m_BV.e_min[2], pos[2]));
		
		the_Root.key = 1.0/eps;
		
		io_NodeQueue->insert(the_Root);
		double _max_dist2 = max_dist2;
		double min_key = 1.0/(max_dist2+eps);
		
		//int the_FoundID = -1;
		
		while(1)
		{
			SQueueElem v = io_NodeQueue->get();
			if(v.key < 0.0)
				break;
			
			if(v.key < min_key)
				continue;
			
			const SKDTreeNode* ptr = v.ptr;
			//			if(ptr->dim == 3)
			//				continue;
			
			const float dist2 = (ptr->pos[0]-pos[0])*(ptr->pos[0]-pos[0])+(ptr->pos[1]-pos[1])*(ptr->pos[1]-pos[1])+(ptr->pos[2]-pos[2])*(ptr->pos[2]-pos[2]);

			if(dist2 < _max_dist2)
			{
				SPriorityArrayElem<T> elem;
				elem.key = dist2;
				elem.ptr = &m_Nodes[ptr->id];
				
				io_ElemArray->insert(elem);
				
				if(io_ElemArray->full())
				{
					_max_dist2 = io_ElemArray->getLargestElem()->key;
					min_key = 1.0/(_max_dist2+eps);
				}
			}

			const SKDTreeNode* ptr_0 = ptr->firstChildPtr;
			const SKDTreeNode* ptr_1 = ptr->firstChildPtr + 1;
			
			if(ptr_0->dim != 3)
			{
				SQueueElem node_0;
				node_0.nearest_pos[0] = v.nearest_pos[0];
				node_0.nearest_pos[1] = v.nearest_pos[1];
				node_0.nearest_pos[2] = v.nearest_pos[2];
				//if(pos[ptr->dim] > ptr->pos[ptr->dim])
				//	node_0.nearest_pos[ptr->dim] = ptr->pos[ptr->dim];
				
				node_0.nearest_pos[ptr->dim] = std::min(node_0.nearest_pos[ptr->dim], ptr->pos[ptr->dim]);
				
				node_0.ptr = ptr_0;
				const float _dist2 = (node_0.nearest_pos[0]-pos[0])*(node_0.nearest_pos[0]-pos[0])
					+ (node_0.nearest_pos[1]-pos[1])*(node_0.nearest_pos[1]-pos[1])
					+ (node_0.nearest_pos[2]-pos[2])*(node_0.nearest_pos[2]-pos[2]);
				node_0.key = 1.0/(_dist2+eps);
				if(node_0.key >= min_key)
					io_NodeQueue->insert(node_0);
			}
			
			if(ptr_1->dim != 3)
			{
				SQueueElem node_1;
				node_1.nearest_pos[0] = v.nearest_pos[0];
				node_1.nearest_pos[1] = v.nearest_pos[1];
				node_1.nearest_pos[2] = v.nearest_pos[2];
				//if(pos[ptr->dim] < ptr->pos[ptr->dim])
				//	node_1.nearest_pos[ptr->dim] = ptr->pos[ptr->dim];
				
				node_1.nearest_pos[ptr->dim] = std::max(node_1.nearest_pos[ptr->dim], ptr->pos[ptr->dim]);
				
				node_1.ptr = ptr_1;
				const float _dist2 = (node_1.nearest_pos[0]-pos[0])*(node_1.nearest_pos[0]-pos[0])
					+ (node_1.nearest_pos[1]-pos[1])*(node_1.nearest_pos[1]-pos[1])
					+ (node_1.nearest_pos[2]-pos[2])*(node_1.nearest_pos[2]-pos[2]);
				node_1.key = 1.0/(_dist2+eps);
				if(node_1.key >= min_key)
					io_NodeQueue->insert(node_1);
			}
		}
	}
	
	void balance()
	{
		double minX = HUGE_VAL; double maxX = -HUGE_VAL;
		double minY = HUGE_VAL; double maxY = -HUGE_VAL;
		double minZ = HUGE_VAL; double maxZ = -HUGE_VAL;
		
		m_InternalKDTreeNode = new SInternalKDTreeNode();
		m_nTreeNodes = 2;
		
		for(int i=0; i<3; i++) m_InternalKDTreeNode->list[i] = new std::vector<SPrim>();
		
		m_InternalKDTreeNode->child[0] = NULL;
		m_InternalKDTreeNode->child[1] = NULL;
		for(unsigned int i=0; i<m_nNodes; i++)
		{
			SPrim _x = {i, m_Nodes[i].pos[0]};
			m_InternalKDTreeNode->list[0]->push_back(_x);
			SPrim _y = {i, m_Nodes[i].pos[1]};
			m_InternalKDTreeNode->list[1]->push_back(_y);
			SPrim _z = {i, m_Nodes[i].pos[2]};
			m_InternalKDTreeNode->list[2]->push_back(_z);
			
			minX = std::min(minX, m_Nodes[i].pos[0]); maxX = std::max(maxX, m_Nodes[i].pos[0]);
			minY = std::min(minY, m_Nodes[i].pos[1]); maxY = std::max(maxY, m_Nodes[i].pos[1]);
			minZ = std::min(minZ, m_Nodes[i].pos[2]); maxZ = std::max(maxZ, m_Nodes[i].pos[2]);
		}
		
		std::sort(m_InternalKDTreeNode->list[0]->begin(), m_InternalKDTreeNode->list[0]->end(), P_sort_functor_cmp());
		std::sort(m_InternalKDTreeNode->list[1]->begin(), m_InternalKDTreeNode->list[1]->end(), P_sort_functor_cmp());
		std::sort(m_InternalKDTreeNode->list[2]->begin(), m_InternalKDTreeNode->list[2]->end(), P_sort_functor_cmp());
		
		m_BV.e_min[0] = minX; m_BV.e_min[1] = minY; m_BV.e_min[2] = minZ;
		m_BV.e_max[0] = maxX; m_BV.e_max[1] = maxY; m_BV.e_max[2] = maxZ;
		
		std::cout << "nodes sorted." << std::endl;
		
		subdiv(m_InternalKDTreeNode, 0, m_BV);
		std::cout << "tree subdivided." << std::endl;
		
		reconstructNodes();
		std::cout << "tree reconstructed." << std::endl;
		
		m_InternalKDTreeNode = NULL;
		m_Balanced = true;
	}
	
	/*
	T* getResult()
	{
		return (m_ResultID < 0) ? NULL : &m_Nodes[m_ResultID];
	}
	
	double getResultDist2()
	{
		return m_ResultDist2;
	}
	//*/
	
	//for debug
	SKDTreeNode* getKDTreeNode()
	{
		return &m_KDTreeNodes[1];
	}
	
	SMinMaxBV getBV()
	{
		return m_BV;
	}
	
protected:
	T* m_Nodes;
	//int m_ResultID;
	//double m_ResultDist2;
	SKDTreeNode* m_KDTreeNodes;
	T* m_KDTreeNodePtr;
	bool m_Balanced;
	unsigned long m_MaxAvailableNodes;
	unsigned long m_nNodes;
	SMinMaxBV m_BV;
	
private:
	void subdiv(SInternalKDTreeNode* in_Node, const int in_Depth, const SMinMaxBV& in_bv)
	{
		m_nTreeNodes += 2;
		
		if(in_Node->list[0]->size() == 1)
		{
			const unsigned long _id = in_Node->list[0]->at(0).id;
			in_Node->id = _id;
			in_Node->pos[0] = m_Nodes[_id].pos[0];
			in_Node->pos[1] = m_Nodes[_id].pos[1];
			in_Node->pos[2] = m_Nodes[_id].pos[2];
			return;
		}
		
		//subdiv the longest coord
		const float diff_x = in_bv.e_max[0] - in_bv.e_min[0];
		const float diff_y = in_bv.e_max[1] - in_bv.e_min[1];
		const float diff_z = in_bv.e_max[2] - in_bv.e_min[2];
		
		const unsigned int axis = ((diff_x >= diff_y) && (diff_x >= diff_z)) ? 0 : (((diff_y >= diff_z) && (diff_y >= diff_x)) ? 1 : 2);
		in_Node->dim = axis;
		
		//split at median
		const unsigned long the_split_pos = in_Node->list[axis]->size()/2;
		const unsigned long the_split_id = in_Node->list[axis]->at(the_split_pos).id;
		const float the_split_coord = in_Node->list[axis]->at(the_split_pos).coord;
		
		SMinMaxBV _bv_left = in_bv;
		_bv_left.e_max[axis] = the_split_coord;
		SMinMaxBV _bv_right = in_bv;
		_bv_right.e_min[axis] = the_split_coord;
		
		in_Node->id = the_split_id;
		in_Node->pos[0] = m_Nodes[the_split_id].pos[0];
		in_Node->pos[1] = m_Nodes[the_split_id].pos[1];
		in_Node->pos[2] = m_Nodes[the_split_id].pos[2];
		
		in_Node->child[0] = new SInternalKDTreeNode();
		in_Node->child[0]->child[0] = NULL;
		in_Node->child[0]->child[1] = NULL;
		for(int i=0; i<3; i++) in_Node->child[0]->list[i] = new std::vector<SPrim>();
		
		in_Node->child[1] = new SInternalKDTreeNode();
		in_Node->child[1]->child[0] = NULL;
		in_Node->child[1]->child[1] = NULL;
		for(int i=0; i<3; i++) in_Node->child[1]->list[i] = new std::vector<SPrim>();
		
		for(int j=0; j<3; j++)
		{
			std::vector<SPrim>::iterator q = in_Node->list[j]->begin();
			for(; q!=in_Node->list[j]->end(); q++)
			{
				const unsigned long _id = q->id;
				if(_id == the_split_id)
					;
				else if(m_Nodes[_id].pos[axis] <= the_split_coord)
				{
					SPrim the_Prim = {_id, q->coord};
					in_Node->child[0]->list[j]->push_back(the_Prim);
				}
				else
				{
					SPrim the_Prim = {_id, q->coord};
					in_Node->child[1]->list[j]->push_back(the_Prim);
				}
			}
			in_Node->list[j]->clear();
			delete in_Node->list[j];
			in_Node->list[j] = NULL;
		}
		
		bool b_make_left_node = (in_Node->child[0]->list[0]->size() >= 1);
		bool b_make_right_node = (in_Node->child[1]->list[0]->size() >= 1);
		
		if(!b_make_left_node)
		{
			delete in_Node->child[0];
			in_Node->child[0] = NULL;
		}
		
		if(!b_make_right_node)
		{
			delete in_Node->child[1];
			in_Node->child[1] = NULL;
		}
		
		if(b_make_left_node)
			subdiv(in_Node->child[0], in_Depth+1, _bv_left);
		
		if(b_make_right_node)
			subdiv(in_Node->child[1], in_Depth+1, _bv_right);
	}
	
	void reconstructNodes()
	{
		m_KDTreeNodes = (SKDTreeNode*)realloc(m_KDTreeNodes, sizeof(SKDTreeNode)*m_nTreeNodes);
		m_tmpPos = 2;
		_reconstructNode(m_InternalKDTreeNode, 1);
	}
	
	void _reconstructNode(SInternalKDTreeNode* in_Target, unsigned int in_Offset)
	{
		if(in_Target != NULL)
		{
			m_tmpPos += 2;
			
			m_KDTreeNodes[in_Offset].dim = in_Target->dim;
			m_KDTreeNodes[in_Offset].pos[0] = in_Target->pos[0];
			m_KDTreeNodes[in_Offset].pos[1] = in_Target->pos[1];
			m_KDTreeNodes[in_Offset].pos[2] = in_Target->pos[2];
			m_KDTreeNodes[in_Offset].id = in_Target->id;
			m_KDTreeNodes[in_Offset].firstChildPtr = &m_KDTreeNodes[m_tmpPos-2];
			
			for(int i=0; i<3; i++)
			{
				if(in_Target->list[i] != NULL)
				{
					in_Target->list[i]->clear();
					delete in_Target->list[i];
					in_Target->list[i] = NULL;
				}
			}
			
			const unsigned int next_tgt = m_tmpPos-2;
			
			_reconstructNode(in_Target->child[0], next_tgt);
			_reconstructNode(in_Target->child[1], next_tgt+1);
			delete in_Target;
		}
		else
		{
			m_KDTreeNodes[in_Offset].dim = 3;
			m_KDTreeNodes[in_Offset].pos[0] = 0.0;
			m_KDTreeNodes[in_Offset].pos[1] = 0.0;
			m_KDTreeNodes[in_Offset].pos[2] = 0.0;
			m_KDTreeNodes[in_Offset].id = -1;
			m_KDTreeNodes[in_Offset].firstChildPtr = NULL;
		}
	}
	
	SInternalKDTreeNode* m_InternalKDTreeNode;
	unsigned long m_nTreeNodes;
	unsigned long m_tmpPos;
	
	//CPriorityQueue<SQueueElem> *m_NodeQueue;
	//CPriorityArray<SArrayElem<float>, K> *m_LeafArray;
};

#endif
