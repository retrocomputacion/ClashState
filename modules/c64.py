#########################################3
# C64 Routines
#
import numpy as np
import os

import modules.common as CC
import modules.palette as Palette

#Palette structure
Palette_Colodore = [{'color':'Black','RGBA':[0x00,0x00,0x00,0xff],'enabled':True,'index':0},
    {'color':'White','RGBA':[0xff,0xff,0xff,0xff],'enabled':True,'index':1},{'color':'Red','RGBA':[0x96,0x28,0x2e,0xff],'enabled':True,'index':2},
    {'color':'Cyan','RGBA':[0x5b,0xd6,0xce,0xff],'enabled':True,'index':3},{'color':'Purple','RGBA':[0x9f,0x2d,0xad,0xff],'enabled':True,'index':4},
    {'color':'Green','RGBA':[0x41,0xb9,0x36,0xff],'enabled':True,'index':5},{'color':'Blue','RGBA':[0x27,0x24,0xc4,0xff],'enabled':True,'index':6},
    {'color':'Yellow','RGBA':[0xef,0xf3,0x47,0xff],'enabled':True,'index':7},{'color':'Orange','RGBA':[0x9f,0x48,0x15,0xff],'enabled':True,'index':8},
    {'color':'Brown','RGBA':[0x5e,0x35,0x00,0xff],'enabled':True,'index':9},{'color':'Pink','RGBA':[0xda,0x5f,0x66,0xff],'enabled':True,'index':10},
    {'color':'Dark Grey','RGBA':[0x47,0x47,0x47,0xff],'enabled':True,'index':11},{'color':'Medium Grey','RGBA':[0x78,0x78,0x78,0xff],'enabled':True,'index':12},
    {'color':'Light Green','RGBA':[0x91,0xff,0x84,0xff],'enabled':True,'index':13},{'color':'Light Blue','RGBA':[0x68,0x64,0xff,0xff],'enabled':True,'index':14},
    {'color':'Light Grey','RGBA':[0xae,0xae,0xae,0xff],'enabled':True,'index':15}]

Palette_PeptoNTSC = [{'color':'Black','RGBA':[0x00,0x00,0x00,0xff],'enabled':True,'index':0},
    {'color':'White','RGBA':[0xff,0xff,0xff,0xff],'enabled':True,'index':1},{'color':'Red','RGBA':[0x7C,0x35,0x2B,0xff],'enabled':True,'index':2},
    {'color':'Cyan','RGBA':[0x5A,0xA6,0xB1,0xff],'enabled':True,'index':3},{'color':'Purple','RGBA':[0x69,0x41,0x85,0xff],'enabled':True,'index':4},
    {'color':'Green','RGBA':[0x5D,0x86,0x43,0xff],'enabled':True,'index':5},{'color':'Blue','RGBA':[0x21,0x2E,0x78,0xff],'enabled':True,'index':6},
    {'color':'Yellow','RGBA':[0xCF,0xBE,0x6F,0xff],'enabled':True,'index':7},{'color':'Orange','RGBA':[0x89,0x4A,0x26,0xff],'enabled':True,'index':8},
    {'color':'Brown','RGBA':[0x5B,0x33,0x00,0xff],'enabled':True,'index':9},{'color':'Pink','RGBA':[0xAF,0x64,0x59,0xff],'enabled':True,'index':10},
    {'color':'Dark Grey','RGBA':[0x43,0x43,0x43,0xff],'enabled':True,'index':11},{'color':'Medium Grey','RGBA':[0x6b,0x6b,0x6b,0xff],'enabled':True,'index':12},
    {'color':'Light Green','RGBA':[0xA0,0xCB,0x84,0xff],'enabled':True,'index':13},{'color':'Light Blue','RGBA':[0x56,0x65,0xB3,0xff],'enabled':True,'index':14},
    {'color':'Light Grey','RGBA':[0x95,0x95,0x95,0xff],'enabled':True,'index':15}]

C64Palettes = [['Colodore',Palette_Colodore],['Pepto (NTSC,Sony)',Palette_PeptoNTSC]]

#HiRes
def c64_get2closest(colors,p_in,p_out,fixed):
    cd = [[197000 for j in range(len(p_in))] for i in range(len(colors))]
    closest = []
    _indexes = [1,1]
    xmin = -1
    for x in range(0,len(colors)):
        #yr = [b for b in range(len(p_in)) if b not in _indexes] #avoid repeated indexes
        for y in range(0,len(p_in)):
            if y != xmin:
                # rd = colors[x][1][0] - p_in[y][0][0]
                # gd = colors[x][1][1] - p_in[y][0][1]
                # bd = colors[x][1][2] - p_in[y][0][2]
                cd[x][y] = CC.Redmean(colors[x][1],p_in[y][0])  #(rd * rd + gd * gd + bd * bd)
        xmin=cd[x].index(min(cd[x]))
        cc = p_in[xmin][1]
        m = p_in[xmin][0] #p_out[cc]
        closest.append(CC.RGB24(m).tolist())
        _indexes[x] = cc
    if len(closest) == 1:
        closest.append(CC.RGB24(p_in[0][0]).tolist())
        _indexes[1]= 0
    tix = sorted(_indexes)  #Sort by color index
    if tix != _indexes:
        closest.reverse()
        _indexes = tix
    return(_indexes,Palette.Palette(closest))

#Multicolor
def c64_get4closest(colors, p_in, p_out, bgcolor):
    cd = [[0 for j in range(len(p_in))] for i in range(len(colors))]
    brgb = CC.RGB24(next(x[0].tolist() for x in p_in if x[1]==bgcolor[0]))
    closest = [brgb,brgb,brgb,brgb]
    _indexes = [bgcolor[0],bgcolor[0],bgcolor[0],bgcolor[0]]
    #Attr
    indexes = 0#0x33
    cram = 2
    #Find least used color
    if len(colors) >= 4:
        #c_counts = [colors[i][0] for i in range(len(colors))]
        bi = colors.index(min(colors))
    else:
        bi = 5
    xx = 1
    #npin = np.asarray([c[0] for c in p_in])
    #ncolors = np.asarray([c[1] for c in colors])
    for x in range(0,len(colors)):
        if x == bi:
            continue
        for y in range(0,len(p_in)):
            # rd = colors[x][1][0] - p_in[y][0][0]
            # gd = colors[x][1][1] - p_in[y][0][1]
            # bd = colors[x][1][2] - p_in[y][0][2]
            cd[x][y] = CC.Redmean(colors[x][1],p_in[y][0])  #(rd * rd + gd * gd + bd * bd)    #This is the fastest distance method
            # cd[x][y] = sum([(a - b)**2 for a, b in zip(colors[x][1], p_in[y][0])])
            # cd[x][y] = np.linalg.norm(ncolors[x]-npin[y])
        xmin=cd[x].index(min(cd[x]))
        cc = p_in[xmin][1]
        m = p_in[xmin][0] #p_out[cc]
        closest[xx] = CC.RGB24(m).tolist()  #m[2]+m[1]*256+m[0]*65536
        _indexes[xx] = cc
        xx += 1

    return(_indexes,Palette.Palette(closest))

# def packmulticolor(cell):
#     cell_a = np.asarray(cell)
#     out = b''
#     for y in range(8):
#         tbyte = 0
#         for x in range(4):
#             tbyte += int(cell_a[y,x])<<((3-x)*2)
#         out+= tbyte.to_bytes(1,'big')
#     return(out)

def bmpackhi(column,row,cell,buffers):
    if len(buffers)<4:
        offset = (column*8)+(row*320)
        buffers[0][offset:offset+8]=list(np.packbits(np.asarray(cell,dtype='bool')))
    else:
        offset = ((column+3)*8)+(row//8)*320+(row&7)
        buffers[0][offset]=list(np.packbits(np.asarray(cell,dtype='bool')))[0]

def bmpackmulti(column,row,cell,buffers):
    cell_a = np.asarray(cell)
    offset = (column*8)+(row*320)
    for y in range(8):
        tbyte = 0
        for x in range(4):
            tbyte += int(cell_a[y,x])<<((3-x)*2)
        buffers[0][offset+y] = tbyte
        #out+= tbyte.to_bytes(1,'big')

def attrpack(column,row,attr,buffers):
    if len(buffers) < 4:
        offset = column+(row*40)    #Normal
    else:
        offset = column+3+((row//8)*40) #(A)FLI
    if len(attr) == 2:
        if len(buffers) == 2:
            buffers[1][offset]=attr[0]+(attr[1]*16) #HIRES
        else:
            buffers[1+(row % 8)][offset]=attr[0]+(attr[1]*16) #AFLI
    else:
        buffers[1][offset]=attr[2]+(attr[1]*16)
        buffers[2][offset]=attr[3]


# Returns a list of lists
def get_buffers(mode:int):
    if mode == 3:
        x = 8
    else:
        x = 1 
    buffers=[]
    buffers.append([0]*8000)    # Bitmap
    for i in range(x):
        buffers.append([0xf0]*1000)    # Screen RAM(s)
    if mode == 2:
        buffers.append([0]*1000)    # Color RAM
    return buffers

def buildfile(buffers,bg,baseMode):
    if baseMode == 1:   #Save Koala
        t_data = b'\x00\x60' #Load address
        #Bitmap
        t_data += bytes(buffers[0])#bitmap
        #Screen
        t_data += bytes(buffers[1])#screen
        #ColorRAM
        t_data += bytes(buffers[2])#cram
        #Background
        bg_color = int(bg[0]) if bg[0]>0 else 0 #background color
        t_data += bg_color.to_bytes(1,'big')
    elif baseMode == 0:   #Save ArtStudio
        t_data = b'\x00\x20' #Load address
        #Bitmap
        t_data += bytes(buffers[0])#bitmap
        #Screen
        t_data += bytes(buffers[1])#screen
        #Border
        t_data += b'\x00'
        #Padding
        t_data += b"M 'STU"
    elif baseMode == 2:   #Save Hires Manager
        t_data=b'\x00\x40'
        for i in range(1,9):
            t_data += bytes(buffers[i]) #screens
            t_data += bytes(24)
        t_data += bytes(buffers[0])
        t_data += bytes(191)
    elif baseMode == 3:   #AFLI PRG
        with open(os.path.abspath(os.path.join(CC.bundle_dir, "c64/afliview.prg")),'rb') as vf:
            t_data = vf.read()
            t_data += bytes(buffers[0])
            for i in range(21):
                t_data+=b'\xff\xff\xff' #FLI-bug underlay-sprite rows all MC color2
            t_data += b'\x00'
            for i in range(16):
                t_data+=b'\xff\xff\xff' #Last sprite is shorter
            t_data += bytes(80)
            #t_data += bytes(192)
            for i in range(1,8):
                t_data += bytes(buffers[i]) #screens
                t_data += bytes(16)
                t_data += bytes([125]*6)
                t_data += bytes([126]*2)    #sprite pointers
            t_data += bytes(buffers[8])
            t_data += b'\x00'   #border color
    elif baseMode >= 4:     #Hires PRG
        with open(os.path.abspath(os.path.join(CC.bundle_dir, "c64/64view.prg")),'rb') as vf:
            t_data = vf.read()
            t_data += bytes(buffers[0]) #Bitmap
            t_data += bytes(192)
            t_data += bytes(buffers[1]) #Screen
            bg_color = int(bg[0]) if bg[0]>0 else 0
            t_data += bg_color.to_bytes(1,'big')   #background color
            if baseMode == 5:
                t_data += b'\x18'   #Multicolor mode
                t_data += bytes(22)
                t_data += bytes(buffers[2]) #ColorRAM
            else:
                t_data += b'\x08'   #Hires mode
    return(t_data)
#############################

#####################################################################################################################
# Graphic modes structure
# name: Name displayed in the combobox
# bpp: bits per pixel
# attr: attribute size in pixels
# global_colors: a boolean tuple of 2^bpp elements, True if the color for that index is global for the whole screen
# palettes: a list of name/palette pairs

# This field is for the planned rewrite of the conversion routine(s), unused right now.
# attributes: list of attributes:
#               dim: area this/these attributes cover
#               get_attr: function call to get closest color(s) for an attribute cell
#               bm_pack:  function call to pack the bitmap from 8bpp into the native format order (optional)
#               attr_pack: function call to pack the individual cell color(s) into attribute byte(s) (optional)
#               Need more fields to set GUI options -> name and get best color    

# in_size: input image dimensions, converted image will also be displayed with these dimensions
# out_size: native image dimensions
# get_attr: function call to get closest colors for an attribute cell
# bm_pack:  function call to pack the bitmap from 8bpp into the native format order
# attr_pack: function call to pack the individual cell colors into attribute byte(s)
# get_buffers: function call returns the native bitmap and attribute buffers
# save_output: a list of lists in the format ['name','extension',save_function]

GFX_MODES=[{'name':'C64 HiRes','bpp':1,'attr':(8,8),'global_colors':(False,False),'palettes':C64Palettes,
            'global_names':[],
            'attributes':[{'dim':(8,8),'get_attr':c64_get2closest,'bm_pack':bmpackhi,'attr_pack':attrpack}],
            'in_size':(320,200),'out_size':(320,200),'get_attr':c64_get2closest,'bm_pack':bmpackhi,'attr_pack':attrpack,
            'get_buffers':lambda: get_buffers(1),'save_output':[['Art Studio','.art',lambda buf,c: buildfile(buf,c,0)],
            ['C64 Program','.prg', lambda buf,c: buildfile(buf,c,4)]]},
            {'name':'C64 Multicolor','bpp':2,'attr':(4,8),'global_colors':(True,False,False,False),'palettes':C64Palettes,
             'global_names':['Background color'],
            'attributes':[{'dim':(160,200),'get_attr':None,'bm_pack':None,'attr_pack':None},
                        {'dim':(4,8),'get_attr':c64_get4closest,'bm_pack':bmpackmulti,'attr_pack':attrpack}],
            'in_size':(320,200),'out_size':(160,200),'get_attr':c64_get4closest,'bm_pack':bmpackmulti,'attr_pack':attrpack,
            'get_buffers':lambda: get_buffers(2),'save_output':[['Koala Paint','.koa',lambda buf,c:buildfile(buf,c,1)],
            ['C64 Program','.prg', lambda buf,c: buildfile(buf,c,5)]]},
            {'name':'C64 AFLI','bpp':1,'attr':(8,1),'global_colors':(False,False),'palettes':C64Palettes,
            'global_names':[],
            'attributes':[{'dim':(8,1),'get_attr':c64_get2closest,'bm_pack':bmpackhi,'attr_pack':attrpack}],
            'in_size':(296,200),'out_size':(296,200),'get_attr':c64_get2closest,'bm_pack': bmpackhi,'attr_pack':attrpack,
            'get_buffers':lambda: get_buffers(3),'save_output':[['AFLI Editor','.afl',lambda buf,c:buildfile(buf,c,2)],
            ['C64 Program','.prg',lambda buf,c: buildfile(buf,c,3)]]},
            {'name':'C64 Unrestricted','bpp':4,'attr':(0,0),'global_colors':tuple([False])*16,'palettes':C64Palettes,
            'global_names':[],
            'attributes':[],
            'in_size':(320,200),'out_size':(320,200),'get_attr':None,'bm_pack':None,'attr_pack':None,'get_buffers':None,'save_output':None}]

