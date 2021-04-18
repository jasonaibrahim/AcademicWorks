#Octal
OCTAL:
lui $r0 0x00				#initialize $r0 to zero
lui $r1 0x00				#initialize $r1 to zero
lui $r2 0x00				#initialize $r2 to zero

lw $r2 0($r2)				#load the first word in memory into #r2. $r2 = [0123 4567; 89ab cdef]
lui $r0 0x01				#make $r0 look like [0000 0001; 0000 0000]
ori $r0 $r0 0xFF			#make $r0 look like [0000 0001; 1111 1111]
and $r2 $r2 $r0	       		#isolate lower 9 bits. $r2 = [0000 0007;89ab cdef]

andi $r0 $r0 0x02			#$r0 = [0000 0000; 0000 0010]
sllv $r2 $r2 $r0			#r2 = [0000 0789; abcd ef00]

lui $r0 0x00				#initialize $r0 to zero
ori $r0 $r0 0xFF			#$r0 = [0000 0000; 1111 1111]
and $r0 $r0 $r2				#$r0 = [0000 0000; abcd ef00]

ori $r1 $r1 0x01			#$r1 = [0000 0000; 0000 0001]
srlv $r0 $r0 $r1			#$r0 = [0000 0000; 0abc def0]

lui $r1 0xFF				#$r1 = [1111 1111; 0000 0000]
and $r2 $r2 $r1				#$r2 = [0000 0789; 0000 0000]
or $r2 $r2 $r0				#$r2 = [0000 0789; 0abc def0]
ori $r1 $r1 0xF0			#$r1 = [1111 1111; 1111 0000]
and $r2 $r2 $r1				#$r2 = [0000 0789; 0abc 0000]

andi $r0 $r0 0x0F			#$r0 = [0000 0000; 0000 def0]
lui $r1 0x00				#$r1 = [0000 0000; garbage  ]
andi $r1 $r1 0x00			#$r1 = [0000 0000; 0000 0000]
ori $r1 $r1 0x01			#$r1 = [0000 0000; 0000 0001]
srlv $r0 $r0 $r1			#$r0 = [0000 0000; 0000 0def]

or $r2 $r2 $r0				#$r2 = [0000 0789; 0abc 0def]

disp $r2 0
jr $r3
