andi $r0, $r0, 0x0000

disp $r0, 0   # After this instruction, DISP[0] should show 0000



ori $r0, $r0, 0x1   # DISP[0] should still show 0000

add $r0, $r0, $r0   # DISP[0] should stlll show 0000



disp $r0, 0   # DISP[0] should now show 0002