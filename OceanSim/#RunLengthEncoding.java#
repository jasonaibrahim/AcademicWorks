/* RunLengthEncoding.java */


/**
 *  The RunLengthEncoding class defines an object that run-length encodes an
 *  Ocean object.  Descriptions of the methods you must implement appear below.
 *  They include constructors of the form
 *
 *      public RunLengthEncoding(int i, int j, int starveTime);
 *      public RunLengthEncoding(int i, int j, int starveTime,
 *                               int[] runTypes, int[] runLengths) {
 *      public RunLengthEncoding(Ocean ocean) {
 *
 *  that create a run-length encoding of an Ocean having width i and height j,
 *  in which sharks starve after starveTime timesteps.
 *
 *  The first constructor creates a run-length encoding of an Ocean in which
 *  every cell is empty.  The second constructor creates a run-length encoding
 *  for which the runs are provided as parameters.  The third constructor
 *  converts an Ocean object into a run-length encoding of that object.
 *
 *  See the README file accompanying this project for additional details.
 */

public class RunLengthEncoding {

  /**
   *  Define any variables associated with a RunLengthEncoding object here.
   *  These variables MUST be private.
   */
    private DList run_length_encoding = new DList();
    private DListNode run_pointer;
    private static int run_count = 0;
    private int oceanWidth;
    private int oceanHeight;
    private int starveTime;
    private int temp;

  /**
   *  The following methods are required for Part II.
   */

  /**
   *  RunLengthEncoding() (with three parameters) is a constructor that creates
   *  a run-length encoding of an empty ocean having width i and height j,
   *  in which sharks starve after starveTime timesteps.
   *  @param i is the width of the ocean.
   *  @param j is the height of the ocean.
   *  @param starveTime is the number of timesteps sharks survive without food.
   */

  public RunLengthEncoding(int i, int j, int starveTime) { 
      oceanWidth = i;
      oceanHeight = j;
      this.starveTime = starveTime;
      run_length_encoding.insertFront(Ocean.EMPTY,this.starveTime,i*j); 
      run_pointer = run_length_encoding.head;
  }

  /**
   *  RunLengthEncoding() (with five parameters) is a constructor that creates
   *  a run-length encoding of an ocean having width i and height j, in which
   *  sharks starve after starveTime timesteps.  The runs of the run-length
   *  encoding are taken from two input arrays.  Run i has length runLengths[i]
   *  and species runTypes[i].
   *  @param i is the width of the ocean.
   *  @param j is the height of the ocean.
   *  @param starveTime is the number of timesteps sharks survive without food.
   *  @param runTypes is an array that represents the species represented by
   *         each run.  Each element of runTypes is Ocean.EMPTY, Ocean.FISH,
   *         or Ocean.SHARK.  Any run of sharks is treated as a run of newborn
   *         sharks (which are equivalent to sharks that have just eaten).
   *  @param runLengths is an array that represents the length of each run.
   *         The sum of all elements of the runLengths array should be i * j.
   */

  public RunLengthEncoding(int i, int j, int starveTime,
          int[] runTypes, int[] runLengths) {
      
      oceanWidth = i;
      oceanHeight = j;
      this.starveTime = starveTime;
      run_length_encoding.head = null;
      
      for(int k=0; k < runTypes.length; k++) {
        run_length_encoding.insertEnd(runTypes[k], this.starveTime, runLengths[k]);
      }
      
      run_pointer = run_length_encoding.head;
    
  }

  /**
   *  restartRuns() and nextRun() are two methods that work together to return
   *  all the runs in the run-length encoding, one by one.  Each time
   *  nextRun() is invoked, it returns a different run (represented as a
   *  TypeAndSize object), until every run has been returned.  The first time
   *  nextRun() is invoked, it returns the first run in the encoding, which
   *  contains cell (0, 0).  After every run has been returned, nextRun()
   *  returns null, which lets the calling program know that there are no more
   *  runs in the encoding.
   *
   *  The restartRuns() method resets the enumeration, so that nextRun() will
   *  once again enumerate all the runs as if nextRun() were being invoked for
   *  the first time.
   *
   *  (Note:  Don't worry about what might happen if nextRun() is interleaved
   *  with addFish() or addShark(); it won't happen.)
   */

  /**
   *  restartRuns() resets the enumeration as described above, so that
   *  nextRun() will enumerate all the runs from the beginning.
   */

  public void restartRuns() {
      run_pointer = run_length_encoding.head;
      run_count = 0;
  }

  /**
   *  nextRun() returns the next run in the enumeration, as described above.
   *  If the runs have been exhausted, it returns null.  The return value is
   *  a TypeAndSize object, which is nothing more than a way to return two
   *  integers at once.
   *  @return the next run in the enumeration, represented by a TypeAndSize
   *          object.
   */

  public TypeAndSize nextRun() {
      
      if(run_pointer != null){
        int temp1 = run_pointer.species;
        int temp2 = run_pointer.runLength;
        run_pointer = run_pointer.next;
        return new TypeAndSize(temp1, temp2);
      }
      else
          return null;
  }

  /**
   *  toOcean() converts a run-length encoding of an ocean into an Ocean
   *  object.  You will need to implement the three-parameter addShark method
   *  in the Ocean class for this method's use.
   *  @return the Ocean represented by a run-length encoding.
   */

  public Ocean toOcean() {
      
      Ocean newO = new Ocean(this.oceanWidth,this.oceanHeight,this.starveTime);
      run_pointer = run_length_encoding.head;
      int count=0;
      
      for (int y=0; y<oceanHeight; y++){
          for (int x =0; x<oceanWidth; x++){
              
              if(count == run_pointer.runLength){
                  run_pointer = run_pointer.next;
                  count=0;
                  
                  if(run_pointer.species == Ocean.SHARK){
                      newO.addShark(x, y,this.run_pointer.starveTime);
                      count++;
                  }
                  else if(run_pointer.species == Ocean.FISH){
                      newO.addFish(x,y);
                      count++;
                  }
                  else if (run_pointer.species == Ocean.EMPTY){
                      newO.addEmpty(x,y);
                      count++;
                  }
              }
              else {
                  
                  if(run_pointer.species == Ocean.SHARK){
                      newO.addShark(x, y,this.run_pointer.starveTime);
                      count++;
                  }
                  else if(run_pointer.species == Ocean.FISH){
                      newO.addFish(x,y);
                      count++;
                  }
                  else if (run_pointer.species == Ocean.EMPTY){
                      newO.addEmpty(x,y);
                      count++;
                  }
              }
          }
      }
      
    return newO;
  }

  /**
   *  The following method is required for Part III.
   */

  /**
   *  RunLengthEncoding() (with one parameter) is a constructor that creates
   *  a run-length encoding of an input Ocean.  You will need to implement
   *  the sharkFeeding method in the Ocean class for this constructor's use.
   *  @param sea is the ocean to encode.
   */

  public RunLengthEncoding(Ocean sea) {

      oceanWidth = sea.width();
      oceanHeight = sea.height();
      this.starveTime = sea.starveTime();
      run_length_encoding = new DList();
      
      for(int j=0; j < oceanHeight; j++) {
          for (int i=0; i < oceanWidth; i++) {
              
              if(sea.cellContents(i,j) == Ocean.SHARK)
                  run_length_encoding.insertEnd(Ocean.SHARK,sea.sharkFeeding(i,j),1);
              else if (sea.cellContents(i,j) == Ocean.FISH)
                  run_length_encoding.insertEnd(Ocean.FISH,-99,1);
              else if (sea.cellContents(i,j) == Ocean.EMPTY)
                  run_length_encoding.insertEnd(Ocean.EMPTY,-99,1);
              
          }
      }
      
      run_length_encoding.squish();
      run_pointer = run_length_encoding.head;
      check();
  }

  /**
   *  The following methods are required for Part IV.
   */

  /**
   *  addFish() places a fish in cell (x, y) if the cell is empty.  If the
   *  cell is already occupied, leave the cell as it is.  The final run-length
   *  encoding should be compressed as much as possible; there should not be
   *  two consecutive runs of sharks with the same degree of hunger.
   *  @param x is the x-coordinate of the cell to place a fish in.
   *  @param y is the y-coordinate of the cell to place a fish in.
   */

  public void addFish(int x, int y) {
      
      int index = (y*this.oceanHeight) + x;
      DListNode r1 = this.run_length_encoding.head;
      DListNode r2;
      DList temp = new DList();
      int count = 0;
      boolean added = false;
      
      while(r1 != null){
          
          
          if(r1.species != Ocean.EMPTY){
              temp.insertEnd(r1.species, r1.starveTime, r1.runLength);
              count = count + r1.runLength;
              r1 = r1.next;
              
          } else
              
              if (added == false) {
                  
                  count = count + r1.runLength;
                  
                  if(((count > index) & (count < index + r1.next.runLength)) || count == index) {
                      
                      if(r1.runLength > 1){
                          
                      r1.runLength = r1.runLength-1;
                      temp.insertEnd(r1.species, r1.starveTime, r1.runLength);
                      temp.insertEnd(r1.next.species, r1.starveTime, 1);
                      added = true;
                      r1 = r1.next;
                      }
                      
                      else{   
                          
                      temp.insertEnd(Ocean.FISH, -99, 1);
                      added = true;
                      r1 = r1.next;
                      }
                  }
                  
                  else {
                      temp.insertEnd(r1.species,r1.starveTime,r1.runLength);
                      r1 = r1.next;
                  }
              }
                 
      }
      
      
      r1 = temp.head;
      r2 = r1.next;
          
      while(r2 != null) {
          if(r1.species == Ocean.FISH && r2.species == Ocean.FISH) {
              r1.runLength = r1.runLength + r2.runLength;
              r1.next=r2.next;
              r2=r2.next;
          }
          else {
              r1 = r1.next;
              r2 = r2.next;
          }
      }
          
      run_length_encoding.head = temp.head;
      check();
  }

  /**
   *  addShark() (with two parameters) places a newborn shark in cell (x, y) if
   *  the cell is empty.  A "newborn" shark is equivalent to a shark that has
   *  just eaten.  If the cell is already occupied, leave the cell as it is.
   *  The final run-length encoding should be compressed as much as possible;
   *  there should not be two consecutive runs of sharks with the same degree
   *  of hunger.
   *  @param x is the x-coordinate of the cell to place a shark in.
   *  @param y is the y-coordinate of the cell to place a shark in.
   */

  public void addShark(int x, int y) {
      
      int index = (y*this.oceanHeight) + x;
      run_length_encoding.unsquish();

      DListNode r1 = this.run_length_encoding.head;
      
      for (int i=0; i<index; i++) {
          r1=r1.next;
      }
      
      DListNode newFish = new DListNode(r1.prev,Ocean.SHARK,this.starveTime,1,r1.next);
      r1.prev.next = newFish;
      r1.next.prev = newFish;
      
      run_length_encoding.squish();
 
      check();
  }

  /**
   *  check() walks through the run-length encoding and prints an error message
   *  if two consecutive runs have the same contents, or if the sum of all run
   *  lengths does not equal the number of cells in the ocean.
   */

  public void check() {
      DListNode r1 = run_length_encoding.head;
      DListNode r2 = r1.next;
      int count=0;
      
      while(r1.next != null){
          
          count = count + r1.runLength;
          
          if(r1.species == r2.species && r1.starveTime == r2.starveTime){
              System.out.println("ERROR: Two consecutive runs.");
              System.out.println("Run 1 Species: " + r1.species);
              System.out.println("Run 1 Starvetime: " + r1.starveTime);
              System.out.println("Run 2 Species: " + r2.species);
              System.out.println("Run 2 Starvetime: " + r2.starveTime);
          }
            
          r1=r1.next;
          r2=r2.next;
      }
      
      count = count + r1.runLength;
      if(count != this.oceanHeight*this.oceanWidth)
          System.out.println("ERROR: The sum of the run lengths is "
                  + "not equal to the size of the ocean,");
  }

}
