

package list;

/* SList.java
 The purpose of SList and SListNode is only for the use of HashTableChained
 The graph will use the DNode and DList classes.*/

public class SList {

  public SListNode head;
  public int size;

  /**
   *  SList() constructs an empty list.
   **/

  public SList() {
    size = 0;
    head = null;
  }

  /**
   *  isEmpty() indicates whether the list is empty.
   *  @return true if the list is empty, false otherwise.
   **/

  public boolean isEmpty() {
    return size == 0;
  }

  /**
   *  length() returns the length of this list.
   *  @return the length of this list.
   **/

  public int length() {
    return size;
  }

  /**
   *  insertFront() inserts item "obj" at the beginning of this list.
   *  @param obj the item to be inserted.
   **/

  public void insertFront(Object entry) {
    head = new SListNode(entry, head);
    size++;
  }

  public SListNode getFirst(){
      if(head == null){
          System.out.println("The head of this list is null.");
      }
      return head;
  }

  /**
   *  insertEnd() inserts item "obj" at the end of this list.
   *  @param obj the item to be inserted.
   **/

  public void insertEnd(Object entry) {
    if (head == null) {
      head = new SListNode(entry,null);
    } else {
      SListNode node = head;
      while (node.next != null) {
        node = node.next;
      }
      node.next = new SListNode(entry,null);
    }
    size++;
  }

  public void remove(SListNode s){
      if (s.equals(this.head)) this.head = null;
      else{
          SListNode r1 = this.head;
          while(r1.next != s){
              r1 = r1.next;
          }
          r1.next = s.next;
      }

  }


  public String toString() {
    int i;
    Object entry;
    String result = "[  ";

    SListNode cur = head;

    while (cur != null) {
      entry = cur.entry;
      result = result + entry.toString();
      cur = cur.next;
    }
    result = result + "]";
    return result;
  }



}

