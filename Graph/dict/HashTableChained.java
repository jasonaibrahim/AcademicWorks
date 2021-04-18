/* HashTableChained.java */

package dict;
import list.*;
import java.util.Random;
/**
 *  HashTableChained implements a Dictionary as a hash table with chaining.
 *  All objects used as keys must have a valid hashCode() method, which is
 *  used to determine which bucket of the hash table an entry is stored in.
 *  Each object's hashCode() is presumed to return an int between
 *  Integer.MIN_VALUE and Integer.MAX_VALUE.  The HashTableChained class
 *  implements only the compression function, which maps the hash code to
 *  a bucket in the table's range.
 *
 *  DO NOT CHANGE ANY PROTOTYPES IN THIS FILE.
 **/

public class HashTableChained implements Dictionary {

  /**
   *  Place any data fields here.
   **/
    Object[] table;
    Random generator = new Random();
    int a;
    int b;
    int p;
    int N;
    int size;



  /**
   *  Construct a new empty hash table intended to hold roughly sizeEstimate
   *  entries.  (The precise number of buckets is up to you, but we recommend
   *  you use a prime number, and shoot for a load factor between 0.5 and 1.)
   **/

  public HashTableChained(int sizeEstimate) {
    int[] primes = generatePrimes(2*sizeEstimate);
    this.table = new Object[primes[primes.length-1]];
    for(int i=0; i< table.length; i++){
        this.table[i] = new SList();
    }
    primes = generatePrimes(13*primes.length);
    p = primes[primes.length-1];
    a = generator.nextInt(p);
    b= generator.nextInt(p);
    N = this.table.length;
  }

  /**
   *  Construct a new empty hash table with a default size.  Say, a prime in
   *  the neighborhood of 100.
   **/

  public HashTableChained() {
    this.table = new Object[103];
    for(int i=0; i< table.length; i++){
        this.table[i] = new SList();
    }
    int[] primes = generatePrimes(17*103);
    p = primes[primes.length-1];
    a = generator.nextInt(p);
    b= generator.nextInt(p);
    N = this.table.length;
  }

  /**
   *  Converts a hash code in the range Integer.MIN_VALUE...Integer.MAX_VALUE
   *  to a value in the range 0...(size of hash table) - 1.
   *
   *  This function should have package protection (so we can test it), and
   *  should be used by insert, find, and remove.
   **/

  int compFunction(int code) {
      return Math.abs((Math.abs((a*code + b)) % p)) % N;
  }

  /**
   *  Returns the number of entries stored in the dictionary.  Entries with
   *  the same key (or even the same key and value) each still count as
   *  a separate entry.
   *  @return number of entries in the dictionary.
   **/

  public int size() {
      return this.size;
  }

  /**
   *  Tests if the dictionary is empty.
   *
   *  @return true if the dictionary has no entries; false otherwise.
   **/

  public boolean isEmpty() {
      return size == 0;
  }

  /**
   *  Create a new Entry object referencing the input key and associated value,
   *  and insert the entry into the dictionary.  Return a reference to the new
   *  entry.  Multiple entries with the same key (or even the same key and
   *  value) can coexist in the dictionary.
   *
   *  This method should run in O(1) time if the number of collisions is small.
   *
   *  @param key the key by which the entry can be retrieved.
   *  @param value an arbitrary object.
   *  @return an entry containing the key and value.
   **/

  public Entry insert(Object key, Object value) {
        int k = key.hashCode();
        k = compFunction(k);

        Entry e = new Entry(key,value);
        ((SList)this.table[k]).insertFront(e);

        this.size++;

        //Check load factor and resize if necessary
        if(size >= 0.75*(double)this.table.length){
            resizeTable();
        }
        return e;
  }

  /**
   *  Search for an entry with the specified key.  If such an entry is found,
   *  return it; otherwise return null.  If several entries have the specified
   *  key, choose one arbitrarily and return it.
   *
   *  This method should run in O(1) time if the number of collisions is small.
   *
   *  @param key the search key.
   *  @return an entry containing the key and an associated value, or null if
   *          no entry contains the specified key.
   **/

  public Entry find(Object key) {
      try{
          int k = key.hashCode();
          k = compFunction(k);

          SListNode r1 = ((SList)this.table[k]).head;
          if(r1 == null) { return null; }
          else{
            while(r1 != null){
                if(((Entry)r1.entry).key.equals(key)) break;
                else r1 = r1.next;
            }
            return (Entry)r1.entry;
          }
      }catch(NullPointerException e) {return null;}



  }

  /**
   *  Remove an entry with the specified key.  If such an entry is found,
   *  remove it from the table and return it; otherwise return null.
   *  If several entries have the specified key, choose one arbitrarily, then
   *  remove and return it.
   *
   *  This method should run in O(1) time if the number of collisions is small.
   *
   *  @param key the search key.
   *  @return an entry containing the key and an associated value, or null if
   *          no entry contains the specified key.
   */

  public Entry remove(Object key) {

          int k = key.hashCode();
          k = compFunction(k);

          SListNode r1 = ((SList)this.table[k]).head;
          if(r1 == null) { return null; }
          else{
              while(r1 != null){
                  if(((Entry)r1.entry).key.equals(key)){
                      ((SList)this.table[k]).remove(r1);
                      size--;
                      break;
                  }
                  else r1 = r1.next;
              }

              if(r1 != null){
                Entry e = (Entry)r1.entry;
                return e;
              }
              else return null;
          }
  }


  /**
   *  Remove all entries from the dictionary.
   */
  public void makeEmpty() {
      for(int i = 0; i < this.table.length; i++){
        ((SList)this.table[i]).head = null;
    }
      this.size = 0;
  }

  public void resizeTable(){
      HashTableChained newTable = new HashTableChained(this.table.length*2);

      for(int i=0; i<this.table.length; i++){
          if(((SList)this.table[i]).size == 1){
              SListNode s = ((SList)this.table[i]).getFirst();
              Entry e = (Entry)s.entry;
              newTable.insert(e.key,e.value);
          }
          else if(((SList)this.table[i]).size > 1){
              SListNode s = ((SList)this.table[i]).getFirst();
              while(s != null){
                  Entry e = (Entry)s.entry;
                  newTable.insert(e.key,e.value);
                  s = s.next;
              }
          }
      }
      this.N = newTable.N;
      this.a = newTable.a;
      this.b = newTable.b;
      this.generator = newTable.generator;
      this.p = newTable.p;
      this.size = newTable.size;
      this.table = newTable.table;
  }

  private static int [] generatePrimes(int max) {
    boolean[] isComposite = new boolean[max + 1];
    for (int i = 2; i * i <= max; i++) {
        if (!isComposite [i]) {
            for (int j = i; i * j <= max; j++) {
                isComposite [i*j] = true;
            }
        }
    }
    int numPrimes = 0;
    for (int i = 2; i <= max; i++) {
        if (!isComposite [i]) numPrimes++;
    }
    int [] primes = new int [numPrimes];
    int index = 0;
    for (int i = 2; i <= max; i++) {
        if (!isComposite [i]) primes [index++] = i;
    }

    return primes;
  }

}
