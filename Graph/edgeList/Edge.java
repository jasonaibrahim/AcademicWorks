package edgeList;

class Edge{
    int weight;
    Object o1, o2;
    Edge prev, next;

    public Edge(int w, Object obj1, Object obj2, Edge p, Edge n){
	weight = w;
	o1 = obj1;
	o2 = obj2;
	prev = p;
	next = n;
    }
}
