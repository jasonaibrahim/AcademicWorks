

package list;

/* SList.java */

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

  public void insertFront(Object m) {
    head = new SListNode(m, head);
    size++;
  }

  /**
   *  insertEnd() inserts item "obj" at the end of this list.
   *  @param obj the item to be inserted.
   **/

  public void insertEnd(Object m) {
    if (head == null) {
      head = new SListNode(m,null);
    } else {
      SListNode node = head;
      while (node.next != null) {
        node = node.next;
      }
      node.next = new SListNode(m,null);
    }
    size++;
  }

  /**
   * Returns the first SListNode.
   * @return The first SListNode.
   */
  public SListNode first(){
    return head;
  }

  public String toString() {
    int i;
    Object m;
    String result = "[  ";

    SListNode cur = head;

    while (cur != null) {
      m = cur.m;
      result = result + m.toString();
      cur = cur.next;
    }
    result = result + "]";
    return result;
  }



}

