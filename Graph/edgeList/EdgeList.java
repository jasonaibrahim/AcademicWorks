package edgeList;

public class EdgeList{

	private Edge sentinel;
	private int length;

	/**Constructs an empty Edge list.
	 */
	public EdgeList(){
		sentinel = new Edge(0, null, null, null, null);
		sentinel.prev = sentinel.next = sentinel;
		length = 0;
	}

	/**The size of the list.
	 * @return The number of edges stored.
	 */
	public int Size(){
		return length;
	}


	/**Inserts the Edge at the beginning of the list.
	 * @param weight The weight of the edge.
	 * @param obj1 The first object.
	 * @param obj2 The second object.
	 */
	public void Insert(int weight, Object obj1, Object obj2){
		Edge it = new Edge(weight, obj1, obj2, sentinel, sentinel.next);
		sentinel.next.prev = it;
		sentinel.next = it;
		length++;
	}

	/**Removes the first Edge from the list.
	 * Doesn't do anything if the list is empty.
	 */
	public void RemoveFirst(){
		if(sentinel.next == sentinel){ return; }
		sentinel.next = sentinel.next.next;
		sentinel.next.prev = sentinel;
		length--;
	}

	/**Gets the first Edge's weight.
	 * If empty, returns zero.
	 * @return The first Edge's weight.
	 */
	public int FirstWeight(){
		return sentinel.next.weight;
	}

	/**Gets the first Edge's first object.
	 * If empty, returns null.
	 * @return The first Edge's first object.
	 */
	public Object FirstObject1(){
		return sentinel.next.o1;
	}

	/**Gets the first Edge's second object.
	 * If empty, returns null.
	 * @return The first Edge's second object.
	 */
	public Object FirstObject2(){
		return sentinel.next.o2;
	}

	/**Sorts the list from smallest weight (first element) to largest.
	 * Uses merge sort.
	 */
	public void Sort(){
		if(length < 2){ return; }
		EdgeList first = new EdgeList();
		EdgeList last = new EdgeList();
		SplitInto(first, last);
		first.Sort();
		last.Sort();
		Merge(first, last);
	}

	/**Splits this list into two lists, giving them this list's Edges.
	 * So this becomes empty at the end of the function call.
	 * The last half will take the middle Edge if it exists.
	 * @param first The EdgeList to recieve the first half of the Edges.
	 * @param last The EdgeList to recieve the last half of the Edges.
	 */
	protected void SplitInto(EdgeList first, EdgeList last){
		int i = 0, mid = (length%2 == 0)?(length/2):(length/2+1);
		Edge next;
		while(sentinel.next != sentinel){
			next = sentinel.next.next;
			if(length > mid){
				first.Take(this);
			}else{
				last.Take(this);
			}
			i++;
		}
	}

	/**Collects all the edges from first and last, sorting them from smallest
	 * (first) to largest.
	 * This will leave the argument EdgeLists empty.
	 * @param first The first EdgeList to be merged (must already be sorted).
	 * @param last The other EdgeList to be merged (must already be sorted).
	 */
	protected void Merge(EdgeList first, EdgeList last){
		Edge next;
		while(first.Size() != 0 || last.Size() != 0){
			if(first.Size() == 0){
				Take(last);
			}else if(last.Size() == 0){
				Take(first);
			}else{
				if(first.FirstWeight() < last.FirstWeight()){
					Take(first);
				}else{
					Take(last);
				}
			}
		}
	}

	/**Inserts the first Edge of a list at the end of this list. Removing it
	 * from the EdgeList it was a part of before.
	 * @param edge The EdgeList to take the first Edge from.
	 */
	protected void Take(EdgeList other){
		//Extract it from the other list.
		Edge edge = other.ExtractFirst();
		//Insert it into this list.
		edge.prev = sentinel.prev;
		edge.next = sentinel;
		sentinel.prev.next = edge;
		sentinel.prev = edge;
		length++;
	}

	/**Extracts and returns the first Edge in an EdgeList.
	 * @return The first Edge of an EdgeList, or null if empty.
	 */
	protected Edge ExtractFirst(){
		if(sentinel.next == sentinel){ return null; }
		Edge ret = sentinel.next;
		sentinel.next.next.prev = sentinel;
		sentinel.next = sentinel.next.next;
		length--;
		return(ret);
	}

	/**This is strictly for debugging purposes... */
	public static void main(String[] args){
		EdgeList el = new EdgeList();
		for(int i = 0;i < 15;i++){
			el.Insert((int)(Math.floor(Math.random()*50)), new Integer(i), new Integer(-i));
		}
		System.out.println(el);
		el.Sort();
		System.out.println(el);
	}

	/**Converts this EdgeList to a string.
	 * @return The string representation.
	 */
	public String toString(){
		String ret = "[";
		Edge cur = sentinel.next;
		while(cur != sentinel){
			ret += " "+cur.weight+":("+cur.o1+","+cur.o2+")";
			cur = cur.next;
		}
		cur = sentinel.prev;
		ret += " ]";
		return ret;
	}
}
