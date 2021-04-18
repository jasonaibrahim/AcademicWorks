
package graph;

import list.*;

/**
 *
 * @author jasontosh
 */
public class Vertex {
    protected Object o;
    protected DList collection;
    protected DNode parentNode;

    public Vertex(){
        this.o = null;
        collection = new DList();
    }

    public Vertex(Object o){
        this.o = o;
        this.collection = new DList();
    }

    public void setParent(DNode p){
        this.parentNode = p;
    }

}
