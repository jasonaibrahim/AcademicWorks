#strLen returns the number of characters between a given starting address (in this case 0) and a zero byte terminating character
    lui $r0 0x00                    #initialize $r0 to 0
    lui $r1 0x00                    #initialize $r1 to 0 (probably redundant)
    lui $r2 0x00                    #initialize $r2 to 0 (probably redundant)

loop:
    lw $r1 0($r0)               #load word into register
    zb $r2 $r1                  #check to see if the word has a zero byte. put result in $r2
    lui $r1 0x00                #initialize $r1 back to zero
    bne $r2 $r1 exit            #if $r2 has a zero byte, leave loop
    addi $r0 $r0 1              #increment pointer
    j loop

exit:
    lw $r1 0($r0)               #load last word to check final condition
    add $r0 $r0 $r0             #add $r0 to itself to account for 2 chars per word so far.
    lui $r2 0xFF                #initialize $r2 to [11111111; 00000000]
    and $r1 $r1 $r2             #isolate the upper portion of $r1
    lui $r2 0x00                #initialize $r2 to [00000000;00000000]
    bc $r1 $r1                  #$r1 = # of set bits. either 0 or not 0.
    beq $r1 $r2 dont_add        #if $r1 = 0 then dont increment count by one.
    addi $r0 $r0 1              #increment $r0 by one

dont_add:
    disp $r0 0                  #display the final count.
    jr $r3                      #return


