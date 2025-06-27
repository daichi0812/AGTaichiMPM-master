#ifndef __KD_TREE_1NN_H__
#define __KD_TREE_1NN_H__

#include <vector>
#include <algorithm>
using namespace std;

#include "priorityqueue.h"

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
	vector<SPrim> list[3];
};

template <typename T>
class CKDTree1NN
{
public:
	static const int MAX_NQ_ELEMS = 1024 * 16;
	
	CKDTree1NN()
	{
		m_MaxAvailableNodes = 10;
		m_nNodes = 0;
		m_Balanced = false;
		m_Nodes = (T*)malloc(sizeof(T)*m_MaxAvailableNodes);
		m_KDTreeNodes = NULL;
		m_InternalKDTreeNode = NULL;
		
		SQueueElem min_q; min_q.key = -1.0;
		m_NodeQueue = new CPriorityQueue<SQueueElem>(MAX_NQ_ELEMS, min_q);
	}
	
	~CKDTree1NN()
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
			m_MaxAvailableNodes = min(2*m_MaxAvailableNodes, m_MaxAvailableNodes+1024*1024);
			m_Nodes = (T*)realloc(m_Nodes, sizeof(T)*m_MaxAvailableNodes);
		}
		m_Nodes[m_nNodes++] = in_Node;
		m_Balanced = false;
	}
	
	void query(const double pos[3], const double max_dist2)
	{
		const double eps = 0.0000000001;
		if(!m_Balanced)
			balance();
		
		m_NodeQueue->clear();
		
		SQueueElem the_Root;
		the_Root.ptr = &m_KDTreeNodes[1];
		the_Root.nearest_pos[0] = pos[0];
		the_Root.nearest_pos[1] = pos[1];
		the_Root.nearest_pos[2] = pos[2];
		the_Root.key = 1.0/eps;
		
		m_NodeQueue->insert(the_Root);
		double _max_dist2 = max_dist2;
		double min_key = 1.0/(max_dist2+eps);
		
		int the_FoundID = -1;
		
		while(1)
		{
			SQueueElem v = m_NodeQueue->get();
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
				_max_dist2 = dist2;
				the_FoundID = ptr->id;
				min_key = 1.0/(_max_dist2+eps);
			}

			const SKDTreeNode* ptr_0 = ptr->firstChildPtr;
			const SKDTreeNode* ptr_1 = ptr->firstChildPtr + 1;
			
			if(ptr_0->dim != 3)
			{
				SQueueElem node_0;
				node_0.nearest_pos[0] = v.nearest_pos[0];
				node_0.nearest_pos[1] = v.nearest_pos[1];
				node_0.nearest_pos[2] = v.nearest_pos[2];
				if(pos[ptr->dim] > ptr->pos[ptr->dim])
					node_0.nearest_pos[ptr->dim] = ptr->pos[ptr->dim];
				node_0.ptr = ptr_0;
				const float _dist2 = (node_0.nearest_pos[0]-pos[0])*(node_0.nearest_pos[0]-pos[0])
					+ (node_0.nearest_pos[1]-pos[1])*(node_0.nearest_pos[1]-pos[1])
					+ (node_0.nearest_pos[2]-pos[2])*(node_0.nearest_pos[2]-pos[2]);
				node_0.key = 1.0/(_dist2+eps);
				if(node_0.key >= min_key)
					m_NodeQueue->insert(node_0);
			}
			
			if(ptr_1->dim != 3)
			{
				SQueueElem node_1;
				node_1.nearest_pos[0] = v.nearest_pos[0];
				node_1.nearest_pos[1] = v.nearest_pos[1];
				node_1.nearest_pos[2] = v.nearest_pos[2];
				if(pos[ptr->dim] < ptr->pos[ptr->dim])
					node_1.nearest_pos[ptr->dim] = ptr->pos[ptr->dim];
				node_1.ptr = ptr_1;
				const float _dist2 = (node_1.nearest_pos[0]-pos[0])*(node_1.nearest_pos[0]-pos[0])
					+ (node_1.nearest_pos[1]-pos[1])*(node_1.nearest_pos[1]-pos[1])
					+ (node_1.nearest_pos[2]-pos[2])*(node_1.nearest_pos[2]-pos[2]);
				node_1.key = 1.0/(_dist2+eps);
				if(node_1.key >= min_key)
					m_NodeQueue->insert(node_1);
			}
		}
		
		//printf("count: %d\n", count);
		
		m_ResultDist2 = _max_dist2;
		m_ResultID = the_FoundID;
	}
	
	void balance()
	{
		double minX = HUGE_VAL; double maxX = -HUGE_VAL;
		double minY = HUGE_VAL; double maxY = -HUGE_VAL;
		double minZ = HUGE_VAL; double maxZ = -HUGE_VAL;
		
		m_InternalKDTreeNode = new SInternalKDTreeNode();
		m_nTreeNodes = 2;
		
		m_InternalKDTreeNode->child[0] = NULL;
		m_InternalKDTreeNode->child[1] = NULL;
		for(unsigned int i=0; i<m_nNodes; i++)
		{
			SPrim _x = {i, m_Nodes[i].pos[0]};
			m_InternalKDTreeNode->list[0].push_back(_x);
			SPrim _y = {i, m_Nodes[i].pos[1]};
			m_InternalKDTreeNode->list[1].push_back(_y);
			SPrim _z = {i, m_Nodes[i].pos[2]};
			m_InternalKDTreeNode->list[2].push_back(_z);
			
			minX = min(minX, m_Nodes[i].pos[0]); maxX = max(maxX, m_Nodes[i].pos[0]);
			minY = min(minY, m_Nodes[i].pos[1]); maxY = max(maxY, m_Nodes[i].pos[1]);
			minZ = min(minZ, m_Nodes[i].pos[2]); maxZ = max(maxZ, m_Nodes[i].pos[2]);
		}
		
		sort(m_InternalKDTreeNode->list[0].begin(), m_InternalKDTreeNode->list[0].end(), P_sort_functor_cmp());
		sort(m_InternalKDTreeNode->list[1].begin(), m_InternalKDTreeNode->list[1].end(), P_sort_functor_cmp());
		sort(m_InternalKDTreeNode->list[2].begin(), m_InternalKDTreeNode->list[2].end(), P_sort_functor_cmp());
		
		m_BV.e_min[0] = minX; m_BV.e_min[1] = minY; m_BV.e_min[2] = minZ;
		m_BV.e_max[0] = maxX; m_BV.e_max[1] = maxY; m_BV.e_max[2] = maxZ;
		
		subdiv(m_InternalKDTreeNode, 0, m_BV);
		reconstructNodes();
		m_InternalKDTreeNode = NULL;
		m_Balanced = true;
	}
	
	T* getResult()
	{
		return (m_ResultID < 0) ? NULL : &m_Nodes[m_ResultID];
	}
	
	double getResultDist2()
	{
		return m_ResultDist2;
	}
	
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
	int m_ResultID;
	double m_ResultDist2;
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
		
		if(in_Node->list[0].size() == 1)
		{
			const unsigned long _id = in_Node->list[0][0].id;
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
		const unsigned long the_split_pos = in_Node->list[axis].size()/2;
		const unsigned long the_split_id = in_Node->list[axis][the_split_pos].id;
		const float the_split_coord = in_Node->list[axis][the_split_pos].coord;
		
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
		
		in_Node->child[1] = new SInternalKDTreeNode();
		in_Node->child[1]->child[0] = NULL;
		in_Node->child[1]->child[1] = NULL;
		
		for(int j=0; j<3; j++)
		{
			for(unsigned int i=0; i<in_Node->list[j].size(); i++)
			{
				const unsigned long _id = in_Node->list[j][i].id;
				if(_id == the_split_id)
					;
				else if(m_Nodes[_id].pos[axis] <= the_split_coord)
				{
					SPrim the_Prim = {_id, in_Node->list[j][i].coord};
					in_Node->child[0]->list[j].push_back(the_Prim);
				}
				else
				{
					SPrim the_Prim = {_id, in_Node->list[j][i].coord};
					in_Node->child[1]->list[j].push_back(the_Prim);
				}
			}
			in_Node->list[j].clear();
		}
		
		bool b_make_left_node = (in_Node->child[0]->list[0].size() >= 1);
		bool b_make_right_node = (in_Node->child[1]->list[0].size() >= 1);
		
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
			
			in_Target->list[0].clear();
			in_Target->list[1].clear();
			in_Target->list[2].clear();
			
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
	
	CPriorityQueue<SQueueElem> *m_NodeQueue;
	//CPriorityArray<SArrayElem<float>, K> *m_LeafArray;
};

#endif
