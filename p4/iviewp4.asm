;Plus/4 executable for HiRes pictures
bmpdata=$1100
bitmap=$2000
color=$1c00
luma=$1800
srcptr=$fc
tgtptr=$fe
!to "p4view.prg",cbm
!sl "p4vlabels.txt"

		*= $1001
!word ss,2022
!byte $9e
!text "4109"	;Sys 4109

ss
!word 0

*= $100d		;INITIALIZATION AT PROGRAM START - set IRQ handlers, screen, and main VIC registers
begin	sei
        ;sta $ff3f   ; TED reads screen from RAM
		jsr Init
		;jsr wrtitle
		jmp *


nmi:    rti


;--------------------------------------------------------
Init	lda #00        ;lda bmpdata+$23E8 ;border-colour
		sta $ff19
		ldy #0			;copy bitmap & colour data to its place
		lda #<(bmpdata+$2700)
		ldx #>(bmpdata+$2700)
		sta srcptr+0
		stx srcptr+1
		lda #<(luma+$2700)
		ldx #>(luma+$2700)
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
		cmp #>(luma)
		bpl -
		
        lda $1be8   	; Gfx mode
		ora $ff07
		sta $ff07
		lda #$3b		; Hires
		sta $ff06

		lda $1be9
		sta $ff15
		lda $1bea
		sta $ff16

		lda #$08
		sta $ff12       ;Bitmap $2000
        lda #$18
        sta $ff14       ;Attributes $0800
        rts


ENDOFCODE
!if ENDOFCODE >= bmpdata {
	!error "ERROR - TOO MUCH CODE BEFORE BITMAP-DATA! CHANGE BMPDATA ADDRESS!"
}
;====================================================================================================
!align $FF, $00	;Align to next page