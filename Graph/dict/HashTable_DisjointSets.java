/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package dict;
import java.util.Random;
import list.*;

/**
 *
 * @author jasontosh
 */
public class HashTable_DisjointSets{

    Object[] table;
    int size;
    Random generator = new Random();
    int a;
    int b;
    int p;
    int N;
    //This field maps a unique integer to any vertex object for use in DisjointSets.java
    int value;

    public HashTable_DisjointSets(int sizeEstimate){
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
        value=0;
    }

    public HashTable_DisjointSets(){
        this.table = new Object[103];
        for(int i=0; i< table.length; i++){
            this.table[i] = new SList();
        }
        int[] primes = generatePrimes(17*103);
        p = primes[primes.length-1];
        a = generator.nextInt(p);
        b= generator.nextInt(p);
        N = this.table.length;
        value = 0;
    }

    private int compFunction(int code) {
      return Math.abs((Math.abs((a*code + b)) % p)) % N;
    }

    public int size(){ return this.size; }

    public boolean isEmpty(){ return this.size == 0; }

    public void insert(Object key){
        int k = key.hashCode();
        k = compFunction(k);

        Entry e = new Entry(key,this.value);
        this.value++;
        ((SList)this.table[k]).insertFront(e);
        this.size++;

        //Check load factor and resize if necessary
        if(size >= 0.75*(double)this.table.length){
            resizeTable();
        }
    }

    /*
     * Returns the unique integer associated with the given key
     */
    public int find(Object key){

        int k = key.hashCode();
        k = compFunction(k);

        SListNode r1 = ((SList)this.table[k]).head;
        while(r1 != null){
            if(((Entry)r1.entry).key.equals(key)){
                return ((Integer)((Entry)r1.entry).value());
            }
            else r1 = r1.next;
        }
        //Could not find vertex so return -99
        System.out.println("Given key is not a vertex of the graph.");
        return -99;
    }

    private void resizeTable(){
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

    public void makeEmpty(){
        for(int i=0; i<this.table.length; i++){
            this.table[i] = null;
        }
        size = 0;
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
