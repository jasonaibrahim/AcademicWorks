

public class DList {

  DListNode head;
  private int size;

  public DList() {
    size = 0;
    head = null;
  }

  /**
   *  isEmpty() indicates whether the list is empty.
   *  @return true if the list is empty, false otherwise.
   **/

  public boolean isEmpty() {
    return (head == null);
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

  public void insertFront(int species, int step, int runLength) {
      
      head = new DListNode(null,species,step,runLength,head);
      size++;
  }

  /**
   *  insertEnd() inserts item "obj" at the end of this list.
   *  @param obj the item to be inserted.
   **/

  public void insertEnd(int species, int step, int runLength) {
      
    if (head == null) {
        head = new DListNode(species,step,runLength);
    } 
    
    else {
        DListNode node = head;
        while(node.next!=null){
            node = node.next;
        }
        node.next = new DListNode(node,species,step,runLength,null);
    }
    
    size++;
  }


  public void squish() {

      if(head==null)
          return;
      else{
          DListNode r1=head;
          DListNode r2=r1.next;
          
          while(r2!=null){
              if(r1.species == r2.species && r1.starveTime == r2.starveTime){
                  r1.next=r2.next;
                  r1.runLength++;
                  r2=r2.next;
                  
                  if(r1.next != null)
                  r1.next.prev = r1;
                  
              }
              
              else{
                  r1=r1.next;
                  r2=r2.next;
              }
          }
          
      }
  }
          
public void unsquish() {
    
    if (head == null)
        return;
    else {
    
    DListNode r1 = this.head;
    DList temp = new DList();
    int R_L;
    
    while(r1 !=null) {
        
        R_L = r1.runLength;
        
        for (int i=0; i<R_L; i++) {
            temp.insertEnd(r1.species, r1.starveTime, 1);
        }
    
        r1 = r1.next;
        
    }
    
    this.head = temp.head;
    
    }
          
    
}
  
}



