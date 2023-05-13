;C64 executable code based on HermIRES's 'AFLI Exe' and Grahams FLI displayer on Codebase64
;NTSC/PAL/Drean compatible
bmpdata=$0B00
bitmap=$4000
vidram=$6000
srcptr=$fc
tgtptr=$fe
tab18   = $0e00
tab11   = $0f00
!to "afliview.prg",cbm
!sl "aflilabels.txt"

		*= $0801
!word ss,2022
!byte $9e
!text "2064"	;Sys 2064

ss
!word 0

*= $0810		;INITIALIZATION AT PROGRAM START - set IRQ handlers, screen, and main VIC registers
begin	sei
		jsr udetect
		lda #$35
		sta $01
		jsr Init
		;jsr wrtitle
		jmp *

irq0:	pha
		lda $d019
		sta $d019
		inc $d012
		lda #<irq1
		sta $fffe      ; set up 2nd IRQ to get a stable IRQ
		cli

			; Following here: A bunch of NOPs which allow the 2nd IRQ
			; to be triggered with either 0 or 1 clock cycle delay
			; resulting in an "almost" stable IRQ.

		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
irq1:
ntsc1:  lda #$ea		; modified to NOP NOP on NTSC
		lda #$80
		sta $d018		; setup first color RAM address early
		lda #$38
		sta $d011		; setup first DMA access early
		pla
		pla
		pla
		lda $d019
		sta $d019
		lda #$6b
		sta $d015
		lda #$2d
		sta $d012
		lda #<irq0		;2
		sta $fffe		;4 switch IRQ back to first stabilizer IRQ
		lda $d012		;4
		cmp $d012		;4 stabilize last jittering cycle
		beq delay		;2/3 if equal, 2 cycles delay. else 3 cycles delay
delay:
        stx savex+1		;4

		ldx #$0b		;2
wait:	dex				;2
		bne wait		;2/3
		ldy $ea			;3 cycles;lda #$ea		; modified to NOP NOP on NTSC


		ldx #$38
        ldy #25	;nop
		;nop
		bne l1
        nop
l0:
		ldx #$38	;2	shift badlines up, because no badline/DMA possible after line $f8
		lda #$80	;2
		sta $d018	;4	change video-ram (colour)
		stx $d011	;4	trigger badline
		eor $d02f	;4	wait for 6 cycles (can be a useful command too...)
nt2		lda #$ea	;2-4

l1:		lda #$90	;2	etc...
		inx			;2
		sta $d018	;4
		stx $d011	;4
		eor $d02f	;4	wait for 6 cycles (can be a useful command too...)
nt3		lda #$ea	;2-4
		lda #$a0	;2
		inx			;2	(14)
		sta $d018
		stx $d011
		eor $d02f	;4	;wait for 6 cycles (can be a useful command too...)
nt4		lda #$ea
		lda #$b0
		inx
		sta $d018
		stx $d011
		eor $d02f	;4	;wait for 6 cycles (can be a useful command too...)
nt5		lda #$ea
		lda #$c0
		inx
		sta $d018
		stx $d011
		eor $d02f	;4	;wait for 6 cycles (can be a useful command too...)
nt6		lda #$ea
		lda #$d0
		inx
		sta $d018
		stx $d011
		eor $d02f	;4	;wait for 6 cycles (can be a useful command too...)
nt7		lda #$ea
		lda #$e0
		inx			;2
		sta $d018	;4
		stx $d011	;4
nt8		lda #$ea	;2
		nop			;2
		nop			;2
		lda #$f0	;2
		inx			;2
		sta $d018
		stx $d011
		dey
ntsc4:	bne l0         ; branches to l0-1 on NTSC for 2 extra cycles per rasterline

        lda #$70
        sta $d011      ; open upper/lower border
		lda #$00
		sta $d015

;       lda $d016
;       eor #$01       ; IFLI: 1 hires pixel shift every 2nd frame
;       sta $d016
;       lda $dd00
;       eor #$02       ; IFLI: flip between banks $4000 and $C000 every frame
;       sta $dd00

savex:  ldx #$00
        pla
nmi:    rti


;--------------------------------------------------------
Init	lda bmpdata+$3FE8 ;border-colour
		sta $d020
		sta $d021
		sta $d026
		ldy #0			;copy bitmap & colour data to its place
		lda #<(bmpdata+$4000)
		ldx #>(bmpdata+$4000)
		sta srcptr+0
		stx srcptr+1
		lda #<(bitmap+$4000)
		ldx #>(bitmap+$4000)
		sta tgtptr+0
		stx tgtptr+1
		ldy #0
-		lda (srcptr),y
		sta (tgtptr),y
		iny
		bne -
		dec srcptr+1
		dec tgtptr+1
		lda tgtptr+1
		cmp #>(bitmap)
		bpl -
		
		lda #$08		; Multicolor off
		sta $d016
		;lda #$08
		sta $d018
		lda #$96       ; VIC bank $4000-$7FFF
		sta $dd00		
		lda #$ff
		sta $3fff
		sta $7fff		;disable idle-graphics

        lda #$7f
        sta $dc0d      ; no timer IRQs
        lda $dc0d      ; clear timer IRQ flags

        lda #$2b
        sta $d011
        lda #$2d
        sta $d012

        lda #<nmi
        sta $fffa
        lda #>nmi
        sta $fffb      ; dummy NMI to avoid crashing due to RESTORE
        lda #<irq0
        sta $fffe
        lda #>irq0
        sta $ffff
		jsr inittables
        lda #$01
        sta $d01a      ; enable raster IRQs
        lda $d019
        dec $d019      ; clear raster IRQ flag
        cli
        rts

inittables:
	;Position sprites
	ldy #$0F
-	lda VICdata,y
	sta $d000,y
	dey
	bpl -
	lda #$6b
	sta $d015
	sta $d01c
	sta $d017
	lda #$ff
	sta $d01b
	ldy #7
-	lda #($1f40/$40)		;most pointers should be set by AFLI exporter
	sta vidram+$400*7+$3f8,y
	lda #($1f80/$40)
	sta vidram+$400*7+$3f8+6 ;lowest (6th) sprite is shorter
	dey
	bpl -

	rts
		
; Detect C64 model
udetect:
	;sei
	lda #$FE
	and $DC0E
	sta $DC0E
	lda #$38
	sta $DC04
	lda #$4F
	sta $DC05
	lda #<MIRQ
	sta $0314
	lda #>MIRQ
	sta $0315
;Wait for raster line zero
.z1	lda $D012
	bne .z1
	lda $D011
	and #$80
	bne .z1
	sta Flg		;Clear test flag
	inc $DC0E	;Start timer
	cli
.f1	ldy Flg
	beq .f1		;Wait for test flag
	lda Ras
.s1	cmp #$0B	;PAL-B/G?
	beq .n0
	;no, adapt for NTSC/Drean
	lda #$ea
	sta ntsc1
	sta nt2
	sta nt3
	sta nt4
	sta nt5
	sta nt6
	sta nt7
	sta nt8
	dec ntsc4+1
	inc wait-1
	
;
.n0	sei
	rts

MIRQ
	lda $DC0D
	cmp #$81
	bne .p1
	ldx Flg
	bne .p1
	inc Flg
	lda $D012
	sta Ras
	;inc V_BORDER
.p1	jmp $ea81

Flg	!byte 00
Ras	!byte 00	;PALG 11 NTSC 50 NTSCold 56 PALN 1
;				S0		S1		$2		S3		S4		S5		S6		S7
VICdata !byte $18,$84,$18,$30,$00,$00,$18,$5A,$00,$00,$18,$ae,$18,$d8,$00,$00

ENDOFCODE
!if ENDOFCODE >= bmpdata {
	!error "ERROR - TOO MUCH CODE BEFORE BITMAP-DATA! CHANGE BMPDATA ADDRESS!"
}
;====================================================================================================
!align $FF, $00	;Align to next page