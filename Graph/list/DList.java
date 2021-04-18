
package list;

/**
 *
 * @author jasontosh
 */
public class DList {
    public DNode head;
    public DNode tail;
    int size;

    public DList(){
        size = 0;
        head = new DNode((Object)Integer.MAX_VALUE,null,null);
        tail = new DNode((Object)Integer.MIN_VALUE,head,null);
        head.setNext(tail);
    }

    public void decrementSize() { size--; }

    public int size() { return size; }

    public boolean isHead(DNode v) { return v == head; }

    public boolean isTail(DNode v) { return v == tail; }

    public boolean isEmpty() { return (size == 0); }

    public DNode getFirst(){
        if(isEmpty()) { System.out.println("You are trying to access the first "
                + "element of an empty list.");
                return null;}

        return head.getNext();
    }

    public DNode getLast(){
        if(isEmpty()) { System.out.println("You are tryingto access the last "
                + "element of an empty list.");
                return null;
        }
        return tail.getPrev();
    }

    public DNode getPrev(DNode v){
        if(v == head){
            System.out.println("The node you passed in to getPrev is the head"
                    + " of the list");
            return null;
        }
        return v.getPrev();
    }

    public DNode getNext(DNode v){
        if(v == tail) {
            System.out.println("The node you passed into getNext is the tail"
                    + " of the list.");
            return null;
        }
        return v.getNext();
    }

    public void insertFront(DNode n){
        insertAfter(head,n);
    }

    public void insertBefore(DNode v, DNode z){
        if(isHead(v)) { System.out.println("Can't insert before head"); return; }
        DNode u = getPrev(v);
        z.setPrev(u);
        z.setNext(v);
        v.setPrev(z);
        u.setNext(z);
        size++;
    }

    public void insertAfter(DNode v, DNode z){
        if(isTail(v)) { System.out.println("Can't insert after tail"); return; }
        DNode w = v.getNext();
        z.setPrev(v);
        z.setNext(w);
        w.setPrev(z);
        v.setNext(z);
        size++;
    }

    public void insertEnd(DNode n){
        insertBefore(tail,n);
    }

    public void removeLast(){
        if (size == 0) { return; }
        else{
            DNode v = tail.getPrev();
            DNode u = v.getPrev();
            tail.setPrev(u);
            u.setNext(tail);
            v.setPrev(null);
            v.setNext(null);
            size--;
        }
    }

    public void remove(DNode v){
        DNode u = v.getPrev();
        DNode w = v.getNext();
        if(u != null)
        w.setPrev(u);
        u.setNext(w);
        v.setPrev(null);
        v.setNext(null);
        v.setElement(null);
        size--;
    }

    public void removeAll(){
        head.setNext(tail);
        tail.setPrev(head);
        size = 0;
    }

    public DNode find(DNode node){
        DNode r1 = getFirst();
        while(r1 != tail){
            if(r1 == node){
                break;
            }
            else r1 = r1.getNext();
        }

        if(r1 == tail){
            System.out.println("Could not find node " + node +" in list");
            return null;
        }
        else return r1;
    }

    public boolean hasPrev(DNode v) { return v != head; }

    public boolean hasNext(DNode v) { return v != tail; }

    public String toString() {
        String s = "[";
        DNode v = head.getNext();
        while(v != tail){
            s+=v.getElement();
            v = v.getNext();
            if( v != tail)
                s += ",";
        }
        s += "]";
        return s;
    }
}
