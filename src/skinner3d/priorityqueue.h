#ifndef __PRIORITY_QUEUE_H__
#define __PRIORITY_QUEUE_H__

#include <iostream>
#include <algorithm>
using namespace std;

template <typename T>
class CPriorityQueue
{
	CPriorityQueue();
public:
	CPriorityQueue(const unsigned long in_max_size, const T& in_minimal)
	{
		size = 0;
		max_size = in_max_size;
		buffer = (T*)malloc(sizeof(T)*max_size);
		min = in_minimal;
	}
	
	void clear()
	{
		size = 0;
	}
	
	void insert(const T& x)
	{
		size++;
		buffer[size] = x;
		upHeap(size);
	}
	
	T get()
	{
		if(size < 1)
			return min;
		
		T v = buffer[1];
		buffer[1] = buffer[size];
		size--;
		downHeap(1);
		return v;
	}
	
private:
	void upHeap(int cursor)
	{
		int parent;
		const T v = buffer[cursor];//
		while(1)
		{
			parent = cursor >> 1;
			if(parent < 1)
				break;
			if(buffer[parent].key < v.key)//if(buffer[parent].key < buffer[cursor].key)
				buffer[cursor] = buffer[parent];//swap(buffer[parent], buffer[cursor]);
			else
				break;
			cursor = parent;
		}
		buffer[cursor] = v;//
	}
	
	void downHeap(int cursor)
	{
		int child;
		const T v = buffer[cursor];//
		while(1)
		{
			if(cursor > size>>1)
				break;
			child = cursor << 1; //// left child
			if(child < size && buffer[child].key < buffer[child+1].key)
				child++;
			if(v.key < buffer[child].key)//if(buffer[cursor].key < buffer[child].key)
				buffer[cursor] = buffer[child];//swap(buffer[cursor], buffer[child]);
			else
				break;
			cursor = child;
		}
		buffer[cursor] = v;//
	}
	
	int size;
	unsigned long max_size;
	T* buffer;
	T min;
};

#endif
