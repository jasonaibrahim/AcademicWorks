/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package player;

/**
 *
 * @author jasontosh
 */
public class InvalidMoveException extends Exception {

    /**
     * Creates a new instance of <code>InvalidMoveException</code> without detail message.
     */
    public InvalidMoveException() {
    }

    /**
     * Constructs an instance of <code>InvalidMoveException</code> with the specified detail message.
     * @param msg the detail message.
     */
    public InvalidMoveException(String msg) {
        super(msg);
    }
}
