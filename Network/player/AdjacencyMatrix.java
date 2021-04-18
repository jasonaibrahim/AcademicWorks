/**
 * A Network Adjacency Matrix class.
 * Specifically can tell if a valid network exists.
 * @author Aaron Beal
 **/

package player;

public class AdjacencyMatrix{
    private static final int NONE = 0;
    public static final int UP_LEFT    = 1;
    public static final int UP         = 2;
    public static final int UP_RIGHT   = 3;
    public static final int RIGHT      = 4;
    public static final int DOWN_RIGHT = 5;
    public static final int DOWN       = 6;
    public static final int DOWN_LEFT  = 7;
    public static final int LEFT       = 8;
    private static final int DIRECTON_COUNT = 9;

    public static final int[] DIRECTIONS = {UP_LEFT, UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT};
    public static final int DIRECTION_LENGTH = 8;
    public static final int[][] DIRECTION_MOVE = {{ 0, 0},//NONE
                                                  {-1,-1},//UP_LEFT
                                                  { 0,-1},//UP
                                                  { 1,-1},//UP_RIGHT
                                                  { 1, 0},//RIGHT
                                                  { 1, 1},//DOWN_RIGHT
                                                  { 0, 1},//DOWN
                                                  {-1, 1},//DOWN_LEFT
                                                  {-1, 0}};//LEFT

    protected int[][] matrix;

    /**
     * The initalizer. Creates an empty matrix (all disconnected).
     * @param nodes The number of nodes the matrix will handle.
     **/
    public AdjacencyMatrix(int nodes){
        matrix = new int[nodes][nodes];
    }

    /**
     * Adds a connection between two nodes.
     * @param aNode The first node.
     * @param bNode The second node.
     * @param direction The direction from the first node to the second.
     **/
    public void addConnection(int aNode, int bNode, int direction){
        matrix[aNode][bNode] = direction;
        matrix[bNode][aNode] = reverse(direction);
    }

    /**
     * Returns the opposite direction.
     * @param direction The direction to reverse.
     * @return The opposite direction of that given.
     **/
    protected int reverse(int direction){
        return((direction <= 4)?(direction+4):(direction-4));
    }

    /**
     * Returns is a winning network exists.
     * The "staring" and "ending" goals are arbitrary... They could be swapped
     * without consequence.
     * @param start1 A node index in the "starting" goal. Should be a valid
     *               index (why check otherwise?).
     * @param start2 Another node index in the "starting" goal. May be invalid.
     * @param end1 A node index in the "ending" goal. Should be a valid index
     *             (why check otherwise?).
     * @param end2 Another node index in the "ending" goal. May be invalid.
     * @return If a winning network can be formed...
     **/
    public boolean networkExists(int start1, int start2, int end1, int end2){
        boolean[] used = new boolean[matrix.length];
        for(int i = 0;i < matrix.length;i++){ used[i] = false; }
        //Check for the first.
        if(start1 >= 0 && start1 < matrix.length){
            if(checkNetwork(start1, NONE, 1, used, end1, end2, start2)){
                return(true);
            }
        }
        //Check for the second.
        if(start2 >= 0 && start2 < matrix.length){
            if(checkNetwork(start2, NONE, 1, used, end1, end2, start1)){
                return(true);
            }
        }
        return(false);
    }

    /**
     * Returns if a winning network has been found.
     * @param node The current node.
     * @param direction The direction taken from the previous node to this one.
     * @param depth The number of nodes that are alreay part of this "potential"
     *              network.
     * @param used An array of booleans representing the indices that have been
     *             used so far. Shouldn't be permanently changed by this
     *             function.
     * @param end1 A node index in the ending goal. Should be a valid index
     *             (why check otherwise?).
     * @param end2 Another node index in the ending goal. May be invalid.
     * @param invalid A point that should be avoided (the other start). May be
     *                invalid.
     * @return If a winning network has been found.
     **/
    public boolean checkNetwork(int node, int direction, int depth,
                                boolean[] used, int end1, int end2, int invalid){
        //Avoid the invalid.
        if(node == invalid){ return(false); }
        //Can only touch end nodes when at the end of a network.
        if(node == end1 || node == end2){
            //System.out.print("Worked: [ "+node);
            return(depth >= 6);
        }
        //This node is in use until it's connections have been exhausted.
        used[node] = true;
        for(int other = 0;other < matrix.length;other++){
            //If it's connected, hasn't been used/checked, and causes a
            //turn, then check it.
            if(matrix[node][other] != NONE && !used[other]
            && matrix[node][other] != direction){/*
                for(int s = 0;s < depth-1;s++){ System.out.print("  "); }
                System.out.println("("+depth+") Checking from "+node+" to "+other+" ... ");*/
                if(checkNetwork(other, matrix[node][other], depth+1, used,
                                end1, end2, invalid)){
                    //System.out.println(" "+node);
                    //if(depth == 0){ System.out.print(" ]\n"); }
                    return(true);
                }
            }
        }
        used[node] = false;
        return(false);
    }

    /**
     * Prints out the matrix in a human readable way.
     * For debugging fun!
     **/
    public String toString(){
        String ret = "  ";
        for(int x = 0;x < matrix.length;x++){
            ret += "  "+x;
        }
        ret += "\n";
        for(int y = 0;y < matrix.length;y++){
            ret += y+" [";
            for(int x = 0;x < matrix.length;x++){
                switch(matrix[x][y]){
                case UP_LEFT:
                    ret += " UL";
                    break;
                case UP:
                    ret += " U-";
                    break;
                case UP_RIGHT:
                    ret += " UR";
                    break;
                case RIGHT:
                    ret += " -R";
                    break;
                case DOWN_RIGHT:
                    ret += " DR";
                    break;
                case DOWN:
                    ret += " D-";
                    break;
                case DOWN_LEFT:
                    ret += " DL";
                    break;
                case LEFT:
                    ret += " -L";
                    break;
                default://NONE!
                    ret += " --";
                    break;
                }
            }
            ret += " ]\n";
        }
        return(ret);
    }
}
