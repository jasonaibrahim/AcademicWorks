/* Kruskal.java */

import dict.HashTable_DisjointSets;
import graph.*;
import set.*;
import edgeList.*;

/**
 * The Kruskal class contains the method minSpanTree(), which implements
 * Kruskal's algorithm for computing a minimum spanning tree of a graph.
 */

public class Kruskal {

  /**
   * minSpanTree() returns a WUGraph that represents the minimum spanning tree
   * of the WUGraph g.  The original WUGraph g is NOT changed.
   */
  public static WUGraph minSpanTree(WUGraph g){
  	WUGraph t = new WUGraph();
  	Object[] vertices = g.getVertices();
        DisjointSets ds = new DisjointSets(vertices.length);
        HashTable_DisjointSets htds = new HashTable_DisjointSets();

  	for (Object vertex: vertices) {
  		t.addVertex(vertex);
                htds.insert(vertex);
  	}

  	EdgeList allEdges = new EdgeList();
  	for (Object vertex : vertices){
  		Neighbors neighbor = g.getNeighbors(vertex);
                if(neighbor != null){
                    for (int i = 0; i< neighbor.neighborList.length; i++){
                            Object y = neighbor.neighborList[i];
                            int weight = neighbor.weightList[i];
                            allEdges.Insert(weight, vertex, y);
                    }
                }
  	}
        allEdges.Sort();
        while(ds.size() > 1 && allEdges.Size() != 0){
            int ind1 = ds.find(htds.find(allEdges.FirstObject1()));
            int ind2 = ds.find(htds.find(allEdges.FirstObject2()));
            if(ind1 != ind2){
                t.addEdge(allEdges.FirstObject1(),allEdges.FirstObject2(), allEdges.FirstWeight());
                ds.union(ind1, ind2);
            }
            allEdges.RemoveFirst();
        }
        return t;
  }
}


