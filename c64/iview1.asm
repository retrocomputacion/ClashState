bmpdata=$0900
bitmap=$4000
vidram=$6000
srcptr=$fc
tgtptr=$fe
!to "64view.prg",cbm
!sl "64vlabels.txt"

		*= $0801
!word ss,2022
!byte $9e
!text "2064"	;Sys 2064

ss
!word 0

*= $0810		;INITIALIZATION AT PROGRAM START - set IRQ handlers, screen, and main VIC registers
begin	sei
		lda #$35
		sta $01
		jsr Init
		;jsr wrtitle
		jmp *


nmi:    rti


;--------------------------------------------------------
Init	lda bmpdata+$23E8 ;border-colour
		sta $d020
		sta $d021
		ldy #0			;copy bitmap & colour data to its place
		lda #<(bmpdata+$2300)
		ldx #>(bmpdata+$2300)
		sta srcptr+0
		stx srcptr+1
		lda #<(bitmap+$2300)
		ldx #>(bitmap+$2300)
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
		
        lda bmpdata+$23E9   ; Gfx mode
		;lda #$08		; Multicolor off
		sta $d016
        cmp #$18
        bne +
        ldx #$00        ; Copy ColorRAM
-       lda bmpdata+$2400,x
        sta $D800,x
        lda bmpdata+$2500,x
        sta $D900,x
        lda bmpdata+$2600,x
        sta $DA00,x
        lda bmpdata+$2700,x
        sta $DB00,x
        inx
        bne -


+		lda #$80        ;Screen $6000
		sta $d018       ;Bitmap $4000
		lda #$96       ; VIC bank $4000-$7FFF
		sta $dd00		

;        lda #$7f
;        sta $dc0d      ; no timer IRQs
;        lda $dc0d      ; clear timer IRQ flags

        lda #$3b
        sta $d011
        ; lda #$2d
        ; sta $d012

        lda #<nmi
        sta $fffa
        lda #>nmi
        sta $fffb      ; dummy NMI to avoid crashing due to RESTORE
        ; lda #<irq0
        ; sta $fffe
        ; lda #>irq0
        ; sta $ffff
        ; lda #$01
        ; sta $d01a      ; enable raster IRQs
        ; lda $d019
        ; dec $d019      ; clear raster IRQ flag
        ; cli
        rts


ENDOFCODE
!if ENDOFCODE >= bmpdata {
	!error "ERROR - TOO MUCH CODE BEFORE BITMAP-DATA! CHANGE BMPDATA ADDRESS!"
}
;====================================================================================================
!align $FF, $00	;Align to next page