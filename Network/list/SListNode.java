/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package list;

public class SListNode{
  public Object m;
  protected SListNode next;

  SListNode(Object m, SListNode next) {
    this.m = m;
    this.next = next;
  }
  
  public SListNode next(){
    return next;
  }
}

