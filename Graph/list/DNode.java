
package list;

/**
 *
 * @author jasontosh
 */
public class DNode {
    private Object element;
    private DNode next,prev;

    public DNode(Object e, DNode p, DNode n){
        this.element = e;
        this.next = n;
        this.prev = p;
    }

    public DNode(Object e){
        this.element = e;
        this.next = null;
        this.prev = null;
    }

    public DNode(){
        element = null;
        next = null;
        prev = null;
    }

    public Object getElement() { return this.element; }

    public DNode getPrev() { return this.prev; }

    public DNode getNext() { return this.next; }

    public void setElement(Object newElement) { this.element = newElement; }

    public void setPrev(DNode newPrev) { this.prev = newPrev; }

    public void setNext(DNode newNext) { this.next = newNext; }

}
