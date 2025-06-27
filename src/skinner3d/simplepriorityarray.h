#ifndef __SIMPLE_PRIORITY_ARRAY_H__
#define __SIMPLE_PRIORITY_ARRAY_H__

template <typename T, int K>
class CSimplePriorityArray
{
public:
	CSimplePriorityArray()
	{
		clear();
	}
	
	void clear()
	{
		activeElems = 0;
	}
	
	void insert(const T& elem)
	{
		int insert_pos = 0;
		while(insert_pos < activeElems)
		{
			if(elem.key < m_Elems[insert_pos].key) break;
			else insert_pos++;
		}
		
		if(insert_pos >= K) return;
		
		activeElems = min(activeElems+1, K);
		
		for(int i=activeElems-1; i>=insert_pos+1; i--)
			m_Elems[i] = m_Elems[i-1];
		
		m_Elems[insert_pos] = elem;
	}
	
	T* getLargestElem()
	{
		if(activeElems <= 0) return NULL;
		else return &m_Elems[activeElems-1];
	}
	
	bool full()
	{
		return activeElems == K;
	}
	
	int getActiveElemCount()
	{
		return activeElems;
	}
	
	T* getElemArrayPtr()
	{
		return m_Elems;
	}

protected:
	T m_Elems[K];
	int activeElems;
};

#endif
