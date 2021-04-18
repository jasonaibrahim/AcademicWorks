main:   lui $r2 0xFF
        jal fib
        j return
        
fib:
        lw $r0 0($r3)
        jr $r0

return:
        ori $r2 $r2, 0xFF