/* MachinePlayer.java */

package player;

import list.*;

/**
 *  An implementation of an automatic Network player.  Keeps track of moves
 *  made by both players.  Can select a move for itself.
 */
public class MachinePlayer extends Player {
    public static final int BLACK = 0;
    public static final int WHITE = 1;
    public static final int EMPTY = -1;
    private final int color;
    private GameBoard board; 
    private int searchDepth;

    // Creates a machine player with the given color.  Color is either 0 (black)
    // or 1 (white).  (White has the first move.)
    public MachinePlayer(int color) {
      this.color = color;
      this.board = new GameBoard();
      this.searchDepth = 3;
    }

    // Creates a machine player with the given color and search depth.  Color is
    // either 0 (black) or 1 (white).  (White has the first move.)
    public MachinePlayer(int color, int searchDepth) {
      this.color = color;
      this.board = new GameBoard();
      this.searchDepth = searchDepth;
    }

    // Returns a new move by "this" player.  Internally records the move (updates
    // the internal game board) as a move by "this" player.
    public Move chooseMove() {
        int opponentColor = (color == BLACK)?(WHITE):(BLACK);
        Move ret = null;
        float best = -1, curScore;
        SList moves = board.getLegalMoves(color);
        SListNode cur = moves.first();
        GameBoard temp, bestBoard = null;
        while(cur != null){
            try{
                temp = board.doMove((Move)cur.m, color);
                curScore = scoreMove(temp, opponentColor, -1, 1, 1);
				//System.out.println(((Move)cur.m)+": "+curScore);
				if(curScore == 1){
                    best = curScore;
                    bestBoard = temp;
                    ret = (Move)cur.m;
					break;
				}
                if(curScore > best){
                    best = curScore;
                    bestBoard = temp;
                    ret = (Move)cur.m;
                }
            }catch(InvalidMoveException e){
                System.out.println("An invalid move was returned by getLegalMoves()!?!");
            }
            cur = cur.next();
        }
		//System.out.println(board);
		try{
        	board = board.doMove(ret, color);
		}catch(InvalidMoveException e){
            System.out.println("An invalid move was returned by getLegalMoves()!?!");
        }
        return ret;
    }

    private float scoreMove(GameBoard gb, int side, float alpha, float beta, int depth){
        int oppColor = (side == BLACK)?(WHITE):(BLACK);
		int winner = gb.checkForWinner(oppColor);
		if(winner == color){ return 1.0f - ((float)depth)/3000; }
		int realOpponentColor = (color == BLACK)?(WHITE):(BLACK);
		if(winner == realOpponentColor){ return -1.0f + ((float)depth)/3000; }
        if(depth >= searchDepth){
			float score = Analysis.scoreBoard(gb, oppColor);
			//System.out.print(" "+score);
			return(score);
		}
        float bestScore = (side == color)?(alpha):(beta);
        float curScore;
        SList moves = gb.getLegalMoves(side);
        SListNode cur = moves.first();
        GameBoard temp = null;
        while(cur != null){
            try{
                temp = gb.doMove((Move)cur.m, side);
                curScore = scoreMove(temp, oppColor, alpha, beta, depth+1);/*
				if(curScore != 0){
					for(int i = 0;i < depth;i++){ System.out.print("  "); }
					System.out.println(((Move)cur.m)+": "+curScore);
				}*/
                if(side == color){
					if(curScore > bestScore){
	                    alpha = bestScore = curScore;
					}
                }else{
					if(curScore < bestScore){
	                    beta = bestScore = curScore;
					}
                }
                if(alpha >= beta){
                    return bestScore;
                }
            }catch(InvalidMoveException e){
                System.out.println("An invalid move was returned by getLegalMoves()!?!");
            }
            cur = cur.next();
        }
        return bestScore;
    }

    // If the Move m is legal, records the move as a move by the opponent
    // (updates the internal game board) and returns true.  If the move is
    // illegal, returns false without modifying the internal state of "this"
    // player.  This method allows your opponents to inform you of their moves.
    public boolean opponentMove(Move m) {
        try{
            this.board = board.doMove(m, (color == BLACK)?(WHITE):(BLACK));
        }catch(InvalidMoveException e1){
            return false;
        }
        return true;
    }

    // If the Move m is legal, records the move as a move by "this" player
    // (updates the internal game board) and returns true.  If the move is
    // illegal, returns false without modifying the internal state of "this"
    // player.  This method is used to help set up "Network problems" for your
    // player to solve.
    public boolean forceMove(Move m) {
        try{
            this.board = board.doMove(m,color);
        }catch(InvalidMoveException e1){
            return false;
        }
        return true;
    }

}
