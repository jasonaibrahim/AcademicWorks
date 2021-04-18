/* Ocean.java */

/**
 *  The Ocean class defines an object that models an ocean full of sharks and
 *  fish.  Descriptions of the methods you must implement appear below.  They
 *  include a constructor of the form
 *
 *      public Ocean(int i, int j, int starveTime);
 *
 *  that creates an empty ocean having width i and height j, in which sharks
 *  starve after starveTime timesteps.
 *
 *  See the README file accompanying this project for additional details.
 */

public class Ocean { 

  /**
   *  Do not rename these constants.  WARNING:  if you change the numbers, you
   *  will need to recompile Test4.java.  Failure to do so will give you a very
   *  hard-to-find bug.
   */

  public final static int EMPTY = 0;
  public final static int SHARK = 1;
  public final static int FISH = 2;

  /**
   *  Define any variables associated with an Ocean object here.  These
   *  variables MUST be private.
   */

  private int[][] theOcean;
  private static int[][] sharkState;
  private static int shark_initializer = 0;
  private int starveTime;
  


  /**
   *  The following methods are required for Part I.
   */

  /**
   *  Ocean() is a constructor that creates an empty ocean having width i and
   *  height j, in which sharks starve after starveTime timesteps.
   *  @param i is the width of the ocean.
   *  @param j is the height of the ocean.
   *  @param starveTime is the number of timesteps sharks survive without food.
   */

  public Ocean(int i, int j, int starveTime) {
      
      theOcean = new int[j][i];
      initialize_sharkState(i,j);
      shark_initializer++;
      this.starveTime=starveTime;
  }
  
  private void initialize_sharkState(int i, int j) {
      
      if(shark_initializer == 0){
          sharkState = new int[j][i];
          
          for(int h = 0; h < j; h++){
            for (int k = 0; k < i; k++){
                sharkState[h][k] = -99;
            }
          }
          
      }
          
  }
  
  public int[][] copyGrid() {
 
      int[][] array = new int[this.theOcean.length][this.theOcean[0].length];
      
      for(int j = 0; j < this.theOcean.length; j++){
          for (int i = 0; i < this.theOcean[0].length; i++){
              array[j][i] = this.theOcean[j][i];
          }
      }
  
      return array;
  }
  
  public boolean Rule1(int[][] grid, int i, int j) {
      
      if(
                 grid[j][wrapx(i-1)]==FISH 
              || grid[j][wrapx(i+1)]==FISH 
              || grid[wrapy(j-1)][i]==FISH
              || grid[wrapy(j+1)][i]==FISH 
              || grid[wrapy(j-1)][wrapx(i-1)]==FISH 
              || grid[wrapy(j-1)][wrapx(i+1)]==FISH 
              || grid[wrapy(j+1)][wrapx(i-1)]==FISH
              || grid[wrapy(j+1)][wrapx(i+1)]==FISH ) {

          return true;
      } else
          return false;
  }
  
  public boolean Rule2(int[][] grid, int i, int j) {
      
      if(        grid[wrapy(j)][wrapx(i-1)]   != FISH 
              && grid[wrapy(j)][wrapx(i+1)]   != FISH 
              && grid[wrapy(j-1)][wrapx(i)]   != FISH
              && grid[wrapy(j+1)][wrapx(i)]   != FISH 
              && grid[wrapy(j-1)][wrapx(i-1)] != FISH 
              && grid[wrapy(j-1)][wrapx(i+1)] != FISH 
              && grid[wrapy(j+1)][wrapx(i-1)] != FISH
              && grid[wrapy(j+1)][wrapx(i+1)] != FISH ) {

          return true;
      } else
          return false;
  }
  
    public boolean Rule3(int[][] grid, int i, int j) {
      
      if(        (grid[wrapy(j)][wrapx(i-1)]==FISH   || grid[wrapy(j)][wrapx(i-1)]==EMPTY)
              && (grid[wrapy(j)][wrapx(i+1)]==FISH   || grid[wrapy(j)][wrapx(i+1)]==EMPTY)
              && (grid[wrapy(j-1)][wrapx(i)]==FISH   || grid[wrapy(j-1)][wrapx(i)]==EMPTY)
              && (grid[wrapy(j+1)][wrapx(i)]==FISH   || grid[wrapy(j+1)][wrapx(i)]==EMPTY)
              && (grid[wrapy(j-1)][wrapx(i-1)]==FISH || grid[wrapy(j-1)][wrapx(i-1)]==EMPTY)
              && (grid[wrapy(j-1)][wrapx(i+1)]==FISH || grid[wrapy(j-1)][wrapx(i+1)]==EMPTY)
              && (grid[wrapy(j+1)][wrapx(i-1)]==FISH || grid[wrapy(j+1)][wrapx(i-1)]==EMPTY)
              && (grid[wrapy(j+1)][wrapx(i+1)]==FISH || grid[wrapy(j+1)][wrapx(i+1)]==EMPTY)){

          return true;
      } else
          return false;
  }
    
    public boolean Rule4(int[][] grid, int i, int j) {
      
      if(        grid[wrapy(j)][wrapx(i-1)]==SHARK
              || grid[wrapy(j)][wrapx(i+1)]==SHARK
              || grid[wrapy(j-1)][wrapx(i)]==SHARK
              || grid[wrapy(j+1)][wrapx(i)]==SHARK
              || grid[wrapy(j-1)][wrapx(i-1)]==SHARK
              || grid[wrapy(j-1)][wrapx(i+1)]==SHARK 
              || grid[wrapy(j+1)][wrapx(i-1)]==SHARK 
              || grid[wrapy(j+1)][wrapx(i+1)]==SHARK){

          return true;
      } else
          return false;
  }
    
    public boolean Rule5(int[][] grid, int i, int j) {
        int counter=0;
      
        if(grid[wrapy(j)][wrapx(i-1)]==SHARK)
            counter++;
        if(grid[wrapy(j)][wrapx(i+1)]==SHARK)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i)]==SHARK)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i)]==SHARK)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==SHARK)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==SHARK)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==SHARK)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==SHARK)
            counter++;
        
        if(counter > 1)
            return true;
        else
            return false;
  }
    
    public boolean Rule6(int[][] grid, int i, int j) {
        int counter=0;
      
        if(grid[wrapy(j)][wrapx(i-1)]==FISH)
            counter++;
        if(grid[wrapy(j)][wrapx(i+1)]==FISH)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i)]==FISH)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i)]==FISH)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==FISH)
            counter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==FISH)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==FISH)
            counter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==FISH)
            counter++;
        
        if(counter < 2)
            return true;
        else
            return false;
  }
    
    public boolean Rule7(int[][] grid, int i, int j) {
        int sharkCounter=0;
        int fishCounter=0;
      
        if(grid[wrapy(j)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        
        if(grid[wrapy(j)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j)][wrapx(i+1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==FISH)
            fishCounter++;
        
        if(fishCounter > 1 && sharkCounter < 2)
            return true;
        else
            return false;
  }
    
    public boolean Rule8(int[][] grid, int i, int j) {
        int sharkCounter=0;
        int fishCounter=0;
      
        if(grid[wrapy(j)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==SHARK)
            sharkCounter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==SHARK)
            sharkCounter++;
        
        if(grid[wrapy(j)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j)][wrapx(i+1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j-1)][wrapx(i+1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i-1)]==FISH)
            fishCounter++;
        if(grid[wrapy(j+1)][wrapx(i+1)]==FISH)
            fishCounter++;
        
        if(fishCounter > 1 && sharkCounter > 1)
            return true;
        else
            return false;
  }
              
          


  /**
   *  width() returns the width of an Ocean object.
   *  @return the width of the ocean.
   */

  public int width() {
    return this.theOcean[0].length;
  }

  /**
   *  height() returns the height of an Ocean object.
   *  @return the height of the ocean.
   */

  public int height() {
    return this.theOcean.length;
  }

  /**
   *  starveTime() returns the number of timesteps sharks survive without food.
   *  @return the number of timesteps sharks survive without food.
   */

  public int starveTime() {
    return starveTime;
  }

  /**
   *  addFish() places a fish in cell (x, y) if the cell is empty.  If the
   *  cell is already occupied, leave the cell as it is.
   *  @param x is the x-coordinate of the cell to place a fish in.
   *  @param y is the y-coordinate of the cell to place a fish in.
   */
  
  public int wrapx(int x) {
      if (x < 0)
          return (this.theOcean[0].length -1);
       else
          return (x % this.theOcean[0].length);
  }
  
  public int wrapy(int y) {
      if (y < 0)
          return (this.theOcean.length -1);
       else
          return (y % this.theOcean.length);
  }


  public void addFish(int x, int y) {
      x = wrapx(x);
      y = wrapy(y);
      
      if (this.theOcean[y][x]==EMPTY){
          this.theOcean[y][x]=FISH;
          sharkState[y][x] = -99;
      }
      else
          return;
  }
  
  public void addEmpty(int x, int y) {
      x = wrapx(x);
      y = wrapy(y);
      if (this.theOcean[y][x]==EMPTY)        
          sharkState[y][x] = -99;
  }

  /**
   *  addShark() (with two parameters) places a newborn shark in cell (x, y) if
   *  the cell is empty.  A "newborn" shark is equivalent to a shark that has
   *  just eaten.  If the cell is already occupied, leave the cell as it is.
   *  @param x is the x-coordinate of the cell to place a shark in.
   *  @param y is the y-coordinate of the cell to place a shark in.
   */

  public void addShark(int x, int y) {
      x = wrapx(x);
      y = wrapy(y);
      
      if(this.theOcean[y][x]==EMPTY){
         this.theOcean[y][x]=SHARK;
         sharkState[y][x] = starveTime;
      }
      
      else
          return;
  }

  /**
   *  cellContents() returns EMPTY if cell (x, y) is empty, FISH if it contains
   *  a fish, and SHARK if it contains a shark.
   *  @param x is the x-coordinate of the cell whose contents are queried.
   *  @param y is the y-coordinate of the cell whose contents are queried.
   */

  public int cellContents(int x, int y) {
      x = wrapx(x);
      y = wrapy(y);
      
      if(this.theOcean[y][x]==EMPTY){
         sharkState[y][x] = -99;
         return EMPTY;
      }
      
      else if(this.theOcean[y][x]==FISH){
         sharkState[y][x] = -99;
         return FISH;
      }
      
      else
         return SHARK;
  }

  /**
   *  timeStep() performs a simulation timestep as described in README.
   *  @return an ocean representing the elapse of one timestep.
   */

  public Ocean timeStep() {
      int x = width();
      int y = height();
      Ocean newOcean = new Ocean(x,y,starveTime);
      
      for(int j = 0; j < y; j++){
          for (int i = 0; i < x; i++){
              newOcean.theOcean[j][i] = this.theOcean[j][i];
          }
      }
      
      for(int j = 0; j < y; j++){
          for (int i = 0; i < x; i++){
              
              
          // RULES FOR SHARKS
              
              if(this.theOcean[j][i]==SHARK){
                  
              // RULE 1
                  
                if(Rule1(this.theOcean,i,j)){
                    newOcean.theOcean[j][i] = SHARK;
                    sharkState[j][i] = starveTime;   
                }
                
              // RULE 2
                
                    if(Rule2(this.theOcean,i,j)){
                    sharkState[j][i] = sharkState[j][i] - 1;
                    newOcean.theOcean[j][i] = SHARK;
                  
                    
                    }
                
                }
              
              
          // RULES FOR FISH
              
              if(this.theOcean[j][i]==FISH){
                  
              // RULE 3
                  
                  if(Rule3(this.theOcean,i,j)) {
                      newOcean.theOcean[j][i] = FISH;
    
                  } 
                  
              // RULE 4
                  
                  if(Rule4(this.theOcean,i,j)) {
                      newOcean.theOcean[j][i] = EMPTY;
                      sharkState[j][i] = -99;
                  }
                  
              // RULE 5
                  
                  if(Rule5(this.theOcean,i,j)) {
                      newOcean.theOcean[j][i]=SHARK;
                      sharkState[j][i]=starveTime;    
                  }
              }
              
          // RULES FOR EMPTY CELLS
              
              if(this.theOcean[j][i]==EMPTY){
                  
              // Rule 6    
                  
                  if(Rule6(this.theOcean,i,j)){
                      sharkState[j][i] = -99;
                }  
                  
              // Rule 7
                  
                  if(Rule7(this.theOcean,i,j)) {
                    newOcean.theOcean[j][i]=FISH;
                    sharkState[j][i] = -99;
                }  
                  
              // Rule 8    
                  
                  if(Rule8(this.theOcean,i,j)) {
                    newOcean.theOcean[j][i]=SHARK;
                    sharkState[j][i]=starveTime;
                }
                  
              }
              
        }
      }
      
      // This runs through all the cells of shark state and checks to see
      // which sharks have expired hunger. if their hunger is expired, we replace
      // the cell with an empty and reset the hunger cell with -99, which corresponds
      // to empty
      
      // SHARK SWEEPER
      
      for(int j = 0; j < y; j++){
          for (int i = 0; i < x; i++){
             
               if ((sharkState[j][i]) == -1){
                  newOcean.theOcean[j][i]=EMPTY;
                  sharkState[j][i] = -99;
              }
          }
      }
      
      
      return newOcean;
  }

  /**
   *  The following method is required for Part II.
   */

  /**
   *  addShark() (with three parameters) places a shark in cell (x, y) if the
   *  cell is empty.  The shark's hunger is represented by the third parameter.
   *  If the cell is already occupied, leave the cell as it is.  You will need
   *  this method to help convert run-length encodings to Oceans.
   *  @param x is the x-coordinate of the cell to place a shark in.
   *  @param y is the y-coordinate of the cell to place a shark in.
   *  @param feeding is an integer that indicates the shark's hunger.  You may
   *         encode it any way you want; for instance, "feeding" may be the
   *         last timestep the shark was fed, or the amount of time that has
   *         passed since the shark was last fed, or the amount of time left
   *         before the shark will starve.  It's up to you, but be consistent.
   */

  public void addShark(int x, int y, int feeding) {
      x = wrapx(x);
      y = wrapy(y);
      
      if(this.theOcean[y][x]==EMPTY){
         this.theOcean[y][x]=SHARK;
         sharkState[y][x] = feeding;
      }
      
      else
         return;
  }

  /**
   *  The following method is required for Part III.
   */

  /**
   *  sharkFeeding() returns an integer that indicates the hunger of the shark
   *  in cell (x, y), using the same "feeding" representation as the parameter
   *  to addShark() described above.  If cell (x, y) does not contain a shark,
   *  then its return value is undefined--that is, anything you want.
   *  Normally, this method should not be called if cell (x, y) does not
   *  contain a shark.  You will need this method to help convert Oceans to
   *  run-length encodings.
   *  @param x is the x-coordinate of the cell whose contents are queried.
   *  @param y is the y-coordinate of the cell whose contents are queried.
   */

  public int sharkFeeding(int x, int y) {
      x=wrapx(x);
      y=wrapy(y);
      return sharkState[y][x];
  }

}
