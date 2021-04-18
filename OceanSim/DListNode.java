

class DListNode {
  int species;
  int runLength;
  int starveTime;
  DListNode next;
  DListNode prev;


  DListNode(DListNode prev, int species, int step, int runLength, DListNode next) {
      this.prev = prev;
      this.species = species;
      this.starveTime = step;
      this.runLength = runLength;
      this.next = next;
  }


  DListNode(int species, int step, int runLength) {
    this(null,species,step,runLength, null);
  }
  
  
}
