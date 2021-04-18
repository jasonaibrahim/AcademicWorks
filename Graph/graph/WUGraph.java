/* WUGraph.java */

package graph;

import java.util.Hashtable;
import list.*;
import dict.*;

/**
 * The WUGraph class represents a weighted, undirected graph.  Self-edges are
 * permitted.
 */

public class WUGraph {

    private HashTableChained vertexHash;
    private HashTableChained edgeHash;
    public DList vertexList;
    public DList edgeList;
    protected int vertices;
    protected int edges;

  /**
   * WUGraph() constructs a graph having no vertices or edges.
   *
   * Running time:  O(1).
   */
  public WUGraph(){
      vertexHash = new HashTableChained(250);
      vertexList = new DList();
      edgeHash = new HashTableChained(250);
      edgeList = new DList();
      vertices = 0;
      edges = 0;
}

  /**
   * vertexCount() returns the number of vertices in the graph.
   *
   * Running time:  O(1).
   */
  public int vertexCount(){ return vertices; }

  /**
   * edgeCount() returns the number of edges in the graph.
   *
   * Running time:  O(1).
   */
  public int edgeCount() { return edges; }

  /**
   * getVertices() returns an array containing all the objects that serve
   * as vertices of the graph.  The array's length is exactly equal to the
   * number of vertices.  If the graph has no vertices, the array has length
   * zero.
   *
   * (NOTE:  Do not return any internal data structure you use to represent
   * vertices!  Return only the same objects that were provided by the
   * calling application in calls to addVertex().)
   *
   * Running time:  O(|V|).
   */
  public Object[] getVertices(){
      Object[] temp = new Object[vertices];
      if(vertices == 0) { return temp; }

      DNode node = vertexList.getFirst();
      int i=0;
      while(node != vertexList.tail){
          temp[i] = ((Vertex)node.getElement()).o;
          i++;
          node = node.getNext();
      }
      return temp;
  }

  /**
   * addVertex() adds a vertex (with no incident edges) to the graph.  The
   * vertex's "name" is the object provided as the parameter "vertex".
   * If this object is already a vertex of the graph, the graph is unchanged.
   *
   * Running time:  O(1).
   */
  public void addVertex(Object vertex){
          if(vertexHash.find(vertex) == null){
              DNode temp = new DNode(new Vertex(vertex));
              ((Vertex)temp.getElement()).setParent(temp);
              vertexList.insertEnd(temp);
              vertexHash.insert(vertex,temp);
              vertices++;
          }
  }

  /**
   * removeVertex() removes a vertex from the graph.  All edges incident on the
   * deleted vertex are removed as well.  If the parameter "vertex" does not
   * represent a vertex of the graph, the graph is unchanged.
   *
   * Running time:  O(d), where d is the degree of "vertex".
   */
  public void removeVertex(Object vertex){
      Entry e = vertexHash.find(vertex);
      if(e == null){ return; }

      Vertex v = ((Vertex)((DNode)e.value()).getElement());
      Edge edge;
      while(v.collection.head.getNext() != v.collection.tail){
          edge = ((Edge)v.collection.head.getNext().getElement());
          removeEdge(edge.v1.o, edge.v2.o);
      }
      vertexList.remove(((DNode)e.value()));
      vertexHash.remove(vertex);
      vertices--;
  }

  /**
   * isVertex() returns true if the parameter "vertex" represents a vertex of
   * the graph.
   *
   * Running time:  O(1).
   */
  public boolean isVertex(Object vertex){
      if(vertexHash.find(vertex) != null){
          return true;
      }
      else { return false; }
  }

  /**
   * degree() returns the degree of a vertex.  Self-edges add only one to the
   * degree of a vertex.  If the parameter "vertex" doesn't represent a vertex
   * of the graph, zero is returned.
   *
   * Running time:  O(1).
   */
  public int degree(Object vertex){
      Entry v = vertexHash.find(vertex);
      if(v == null){
          return 0;
      }
      else{
          return ((Vertex)((DNode)v.value()).getElement()).collection.size();
      }
  }

  /**
   * getNeighbors() returns a new Neighbors object referencing two arrays.  The
   * Neighbors.neighborList array contains each object that is connected to the
   * input object by an edge.  The Neighbors.weightList array contains the
   * weights of the corresponding edges.  The length of both arrays is equal to
   * the number of edges incident on the input vertex.  If the vertex has
   * degree zero, or if the parameter "vertex" does not represent a vertex of
   * the graph, null is returned (instead of a Neighbors object).
   *
   * The returned Neighbors object, and the two arrays, are both newly created.
   * No previously existing Neighbors object or array is changed.
   *
   * (NOTE:  In the neighborList array, do not return any internal data
   * structure you use to represent vertices!  Return only the same objects
   * that were provided by the calling application in calls to addVertex().)
   *
   * Running time:  O(d), where d is the degree of "vertex".
   */
  public Neighbors getNeighbors(Object vertex){
      Entry e = vertexHash.find(vertex);
      if( e == null) { return null; }

      Vertex v = ((Vertex)((DNode)e.value()).getElement());
      if(v.collection.size() == 0){ return null; }

      DNode node = v.collection.getFirst();
      Neighbors neigh = new Neighbors();
      neigh.neighborList = new Object[v.collection.size()];
      neigh.weightList = new int[v.collection.size()];
      int i=0;
      while(node != v.collection.tail){
          if(((Edge)node.getElement()).v1 == v){
              neigh.neighborList[i] = ((Edge)node.getElement()).v2.o;
          }
          else {
              neigh.neighborList[i] = ((Edge)node.getElement()).v1.o;
          }
          neigh.weightList[i] = ((Edge)node.getElement()).weight;
          i++;
          node = node.getNext();
      }
      return neigh;
  }

  /**
   * addEdge() adds an edge (u, v) to the graph.  If either of the parameters
   * u and v does not represent a vertex of the graph, the graph is unchanged.
   * The edge is assigned a weight of "weight".  If the edge is already
   * contained in the graph, the weight is updated to reflect the new value.
   * Self-edges (where u == v) are allowed.
   *
   * Running time:  O(1).
   */
  public void addEdge(Object u, Object v, int weight){
          VertexPair uv = new VertexPair(u,v);
          Entry v1 = vertexHash.find(u);
          Entry v2 = vertexHash.find(v);
          if(v1 == null || v2 == null) { return; }

          Entry e = edgeHash.find(uv);
          if(e == null){
              Edge edge = new Edge();
              DNode goes_in_graph_eList = new DNode(edge);
              DNode goes_in_v1_eList = new DNode(edge);
              DNode goes_in_v2_eList = new DNode(edge);

              edge.v1 = ((Vertex)((DNode)v1.value()).getElement());
              edge.v2 = ((Vertex)((DNode)v2.value()).getElement());
              edge.collection1 = goes_in_v1_eList;
              edge.collection2 = goes_in_v2_eList;
              edge.parentNode = goes_in_graph_eList;
              edge.weight = weight;

              if(u != v){
                  ((Vertex)((DNode)v1.value()).getElement()).collection.insertEnd(goes_in_v1_eList);
                  ((Vertex)((DNode)v2.value()).getElement()).collection.insertEnd(goes_in_v2_eList);
              }
              else{
                  ((Vertex)((DNode)v1.value()).getElement()).collection.insertEnd(goes_in_v1_eList);
              }
              edgeList.insertEnd(goes_in_graph_eList);
              edgeHash.insert(uv, goes_in_graph_eList);
              edges++;
          }else {
              ((Edge)((DNode)e.value()).getElement()).weight = weight;
          }
  }

  /**
   * removeEdge() removes an edge (u, v) from the graph.  If either of the
   * parameters u and v does not represent a vertex of the graph, the graph
   * is unchanged.  If (u, v) is not an edge of the graph, the graph is
   * unchanged.
   *
   * Running time:  O(1).
   */
  public void removeEdge(Object u, Object v){
      VertexPair uv = new VertexPair(u,v);
      Entry v1 = vertexHash.find(u);
      Entry v2 = vertexHash.find(v);
      Entry e = edgeHash.find(uv);
      if(v1 == null || v2 == null || e == null){
          return;
      }
      Edge edge = ((Edge)((DNode)e.value()).getElement());
      if(u != v){
          edge.v1.collection.remove(edge.collection1);
          edge.v2.collection.remove(edge.collection2);
      }
      else {
          edge.v1.collection.remove(edge.collection1);
      }
      edgeList.remove(((DNode)e.value()));
      edgeHash.remove(uv);
      edge.collection1 = null;
      edge.collection2 = null;
      edge.v1 = null;
      edge.v2 = null;
      edge.weight = 0;
      edges--;
  }

  /**
   * isEdge() returns true if (u, v) is an edge of the graph.  Returns false
   * if (u, v) is not an edge (including the case where either of the
   * parameters u and v does not represent a vertex of the graph).
   *
   * Running time:  O(1).
   */
  public boolean isEdge(Object u, Object v){
      Entry v1 = vertexHash.find(u);
      Entry v2 = vertexHash.find(v);
      Entry e = edgeHash.find(new VertexPair(u,v));
      if(v1 == null || v2 == null || e == null) {
          return false;
      }
      return true;
  }

  /**
   * weight() returns the weight of (u, v).  Returns zero if (u, v) is not
   * an edge (including the case where either of the parameters u and v does
   * not represent a vertex of the graph).
   *
   * (NOTE:  A well-behaved application should try to avoid calling this
   * method for an edge that is not in the graph, and should certainly not
   * treat the result as if it actually represents an edge with weight zero.
   * However, some sort of default response is necessary for missing edges,
   * so we return zero.  An exception would be more appropriate, but
   * also more annoying.)
   *
   * Running time:  O(1).
   */
  public int weight(Object u, Object v){
      Entry e = edgeHash.find(new VertexPair(u,v));
      if(e == null) {
          return 0;
      }
      else return ((Edge)((DNode)e.value()).getElement()).weight;
  }
}
