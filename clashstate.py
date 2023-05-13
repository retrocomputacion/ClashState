#from ctypes.wintypes import RGB
from colorsys import rgb_to_yiq
import cv2
from cv2 import threshold
import numpy as np
import math
from PIL import Image, ImageEnhance, ImageStat, ImageOps
import dearpygui.dearpygui as dpg
import hitherdither
import datetime as dt
from os.path import exists as file_exists
import os
import sys
import copy
from regex import D

import modules.common as CC
import modules.c64 as c64
import modules.plus4 as p4
import modules.msx as msx
import modules.zxspectrum as zx
import modules.palette as Palette
import modules.dither as DT

# Exec directory
CC.bundle_dir = ''

# Version number
version = 0.8

# Main loop flag
Quit = False

# Progress global

Progress = [0]

# Drag/Zoom data
drag_data = {'wheel':0,'drag':[0,0],'zoom':1, 'pos':[0,0], 'old_drag':[0,0], 'old_mouse':[0,0], 'on_drag':False, 'Release':False}

#Texture buffers
og_tex = None
cv_tex = None
prev_tex = np.zeros((240,200,3), dtype=np.float32)  #array.array('f', prev_data)

#Image buffers
input_img = None    #Unchanged input image
og_img = None       #Resized/cropped image
pre_img = None      #Preprocessed image
cv_img = None       #Converted image

#Converted image, native data
cv_data = None

#Dither method names
ditherlist = ['None','Bayer 2x2', 'Bayer 4x4', 'Bayer 4x4 (Odd)', 'Bayer 4x4 (Even)', 'Bayer 4x4 (Spotty)', 'Bayer 8x8', 'Yliluoma (slow)', 'Cluster dot', 'Floyd-Steinberg']


#Gfx modes
GFX_MODES = []
#Current Gfx mode index
gfx_ix = 0

#Work Palette
Work_Palette : list
#Preview Palette
View_Palette : list
#Current palettes indexes
Palettes=[0,0]

PaletteRGB = [[0x00,0x00,0x00],[0xff,0xff,0xff],[0x81,0x33,0x38],[0x75,0xce,0xc8],[0x8e,0x3c,0x97],[0x56,0xac,0x4d],[0x2e,0x2c,0x9b],[0xed,0xf1,0x71],
    [0x8e,0x50,0x29],[0x55,0x33,0x00],[0xc4,0x6c,0x71],[0x4a,0x4a,0x4a],[0x7b,0x7b,0x7b],[0xa9,0xff,0x9f],[0x70,0x6d,0xeb],[0xb2,0xb2,0xb2]]


bgcolor = [-1]

####################################
# Get GFX modes from loaded modules
# (We're not dynamically loading modules yet)
def build_modes():
    for m in c64.GFX_MODES:
        GFX_MODES.append(m)
    for m in p4.GFX_MODES:
        GFX_MODES.append(m)
    for m in msx.GFX_MODES:
        GFX_MODES.append(m)
    for m in zx.GFX_MODES:
        GFX_MODES.append(m)


################################
# Image crop and resize
def frameResize(i_image, gfxmode):
    i_ratio = i_image.size[0] / i_image.size[1]
    in_size = GFX_MODES[gfxmode]['in_size']
    dst_ratio = in_size[0]/in_size[1]   #320/200
    box=(0,0,0,0)
    zoom = 1

    if dst_ratio >= i_ratio:
        zoom = (in_size[0]*i_image.size[1]/i_image.size[0])/i_image.size[1]
        i_image = i_image.resize((in_size[0],in_size[0]*i_image.size[1]//i_image.size[0]), Image.LANCZOS)
        box = (0,(i_image.size[1]-in_size[1])/2,i_image.size[0],(i_image.size[1]+in_size[1])/2)
        i_image = i_image.crop(box)
    elif dst_ratio < i_ratio:
        zoom = (in_size[1]*i_image.size[0]/i_image.size[1])/i_image.size[0]
        i_image = i_image.resize((in_size[1]*i_image.size[0]//i_image.size[1],in_size[1]),Image.LANCZOS)
        box = ((i_image.size[0]-in_size[0])/2,0,(i_image.size[0]+in_size[0])/2,i_image.size[1])
        i_image = i_image.crop(box)

    pos = [-box[0],-box[1]]
    return(i_image, zoom, pos)



######################################
#   Preset brightness/contrast/etc
#
def imagePreset(o_img:Image):
    stat = ImageStat.Stat(o_img)

    r,g,b = stat.rms
    perbright = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

    if perbright < 25:
        perbright = 25
    elif perbright > 128:
        perbright = 128

    brightness= 1.9*(64/perbright)

    #Saturation
    tmpImg = o_img.convert('HSV')
    stat = ImageStat.Stat(tmpImg)
    h,s,v = stat.rms
    c = tmpImg.getextrema()[2]

    tmpImg.close()

    if s < 96:
        s = 96
    elif s > 128:
        s = 128

    contrast = 1+((c[1]-c[0])/255)
    color = 3-(2*(s/128))

    return brightness, contrast, color


#########################################
#   Adjust image brightness/contrast/etc
#
def imageProcess(o_img:Image, bright:float, contrast:float, saturation:float, hue:int, sharp:float):

    hue = int(((hue-180)/360)*255)
    enhPic = ImageEnhance.Brightness(o_img)
    tPic = enhPic.enhance(bright)
    enhPic = ImageEnhance.Contrast(tPic)
    tPic = enhPic.enhance(contrast)
    enhPic = ImageEnhance.Color(tPic)
    tPic = enhPic.enhance(saturation)
    enhPic = ImageEnhance.Sharpness(tPic)
    tPic = enhPic.enhance(sharp)
    H,S,V = tPic.convert('HSV').split()
    th = np.asarray(H,dtype=np.uint16)
    temp = np.mod(th+hue,255).astype(np.uint8)
    o_img = Image.merge('HSV',(Image.fromarray(temp),S,V)).convert('RGB')

    return o_img

#########################################################################
#   Global colors like the background color in C64 Multicolor mode is   #
#   a kludge right now, a correct implementation, which would allow for #
#   the local 8x8 ColorRAM attribute on C64 FLI mode too, needs a near  #
#   complete rewrite of the conversion routine(s).                      #
#   The commented function below was the start of that rewrite, maybe   #
#   in the future I'll get the inspiration to actually complete it.     #
#########################################################################
#   Convert an attribute cell
#   Recursive
#
#   in_img: quantized image
#   attr: dictionary list from GFX_MODES
#   index: attr list index
#
# def Cell_convert(in_img, attr, index):
#     if type(attr) == list:
#         for x in range(0,in_img.size[0],attr[0]['dim'][0]): # 0 to width step cell width
#             for y in range(0,in_img.size[1],attr[0]['dim'][1]): # 0 to height step cell height
#                 ...
#         ...
#     ...

######################################
#   Convert image
#
def Image_convert(Source:Image, in_pal:list, out_pal:list, gfxmode:int=1, lumaD:int=0, fullD:int=6, Ythr:int=3, Fthr:int=4 , bg_color=[-1]):

    global Progress

    Matchmodes = {'Euclidean': Palette.colordelta.EUCLIDEAN,'CCIR 601': Palette.colordelta.CCIR,'LAb DeltaE CIEDE2000': Palette.colordelta.LAB}

    Progress[0] = 0
    #dpg.set_value('progress',0)

    Mode = GFX_MODES[gfxmode]

    pixelcount = Mode['out_size'][0]*Mode['out_size'][1]

    # Callbacks
    bm_pack = Mode['bm_pack']
    get_buffers = Mode['get_buffers']
    get_attr = Mode['get_attr']
    attr_pack = Mode['attr_pack']

    lmode = dpg.get_value('luma_mode')

    # Generate palette(s)
    rgb_in = []     # contains the [[r,g,b],index values] of all the enabled colors
    rgb_out = [0]*len(out_pal)    # contains the r,g,b values of all the colors 
    rgb_y = []      # Luminance palette as r,g,b
    hd_in = []      # 24bit rgb values of enabled colors
    hd_out = [0]*len(out_pal)     # 24bit rgb values of all colors
    y_in = []       # '24bit' Luminance palette
    for c in in_pal: # iterate colors
        if c['enabled']:
            rgb = CC.RGB24(c['RGBA'])
            rgb_in.append([np.array(c['RGBA'][:3]),c['index']])   # ignore alpha for now
            hd_in.append(rgb)
            if  lmode == 'Over input palette':
                rgb_y.append([CC.Luma(c['RGBA']),CC.Luma(c['RGBA']),CC.Luma(c['RGBA'])])
                y_in.append(CC.RGB24([CC.Luma(c['RGBA']),CC.Luma(c['RGBA']),CC.Luma(c['RGBA'])]))

    if lmode == 'Black & White':
        rgb_y = [[0,0,0],[255,255,255]]
        y_in = [0x000000,0xffffff]

    for c in out_pal:
        rgb = CC.RGB24(c['RGBA'])
        rgb_out[c['index']]=np.array(c['RGBA'][:3])   # ignore alpha for now
        hd_out[c['index']]=rgb

    # PIL Display palette
    i_to = np.array([x[1] for x in rgb_in])
    tPal = [element for sublist in rgb_out for element in sublist]
    plen = len(tPal)//3
    tPal.extend(tPal[:3]*(256-plen))


    in_PaletteH = Palette.Palette(hd_in)   #hitherdither.palette.Palette(hd_in)   #Palette to dither/quantize against
    out_PaletteH = Palette.Palette(hd_out)  #hitherdither.palette.Palette(hd_out) #Palette to display
    y_PaletteH = Palette.Palette(y_in)  #hitherdither.palette.Palette(y_in)     #Luminance palette to dither/quantize against

    # Set color compare method
    cmatch = dpg.get_value("ccir")
    in_PaletteH.colordelta = Matchmodes[cmatch]
    out_PaletteH.colordelta = Matchmodes[cmatch]
    y_PaletteH.colordelta = Matchmodes[cmatch]

    c_count = len(rgb_in)   # Number of colors to quantize to

    o_img = Source

    if Mode['in_size']!=Mode['out_size']:
        o_img = o_img.resize(Mode['out_size'],Image.LANCZOS)
    width = Mode['out_size'][0]
    height = Mode['out_size'][1]
    k = 2<<(Mode['bpp']-1)
    x_step = Mode['attr'][0]
    y_step = Mode['attr'][1]

    if Mode['attr']!=(0,0):
        cell_prog = 0.45/((width/x_step)*(height/y_step))    # Percentage of progress per cell
    progress = 0.5
    
    start = dt.datetime.now()

    # Dither thresholds
    Ythr =[2<<Ythr]*3   #int(perbright/4)
    Fthr =[2<<Fthr]*3   #int(perbright/4)

    # Luminance dither
    if lumaD > 0:  
        Ybr_img = o_img.convert('YCbCr')
        Y,cb,cr = Ybr_img.split()
        if lumaD < 7:
            Y = DT.custom_dithering(Image.merge('RGB',(Y,Y,Y)), y_PaletteH, Ythr,type=lumaD-1)
        elif lumaD == 7:
            Y = DT.yliluomas_1_ordered_dithering(Image.merge('RGB',(Y,Y,Y)), y_PaletteH,Progress,order=8) #Slow, must use order = 8
        elif lumaD == 8:
            Y = hitherdither.ordered.cluster.cluster_dot_dithering(Image.merge('RGB',(Y,Y,Y)), y_PaletteH,order=8, thresholds=Ythr) #Fast
        else:
            #create tmp PIL image with desired palette
            fsPal = [element for sublist in rgb_y for element in sublist]
            plen = len(fsPal)//3
            fsPal.extend(fsPal[:3]*(256-plen))
            tmpI = Image.new('P',(1,1))
            tmpI.putpalette(fsPal)
            Y = Image.merge('RGB',(Y,Y,Y)).quantize(colors=len(rgb_y), palette=tmpI)
        Y = Y.convert('RGB')
        t,t,Y = Y.split()
        Ybr_img = Image.merge('YCbCr',(Y,cb,cr))
        o_img = Ybr_img.convert('RGB')
        #dpg.set_value('progress',0.25) #Reset progress bar
        Progress[0] = 0.25
    # Full dither
    if fullD > 0:
        if fullD < 7:
            o_img = DT.custom_dithering(o_img, in_PaletteH, Fthr, type = fullD-1) # Fastest custom matrix
        elif fullD == 7:
            o_img = DT.yliluomas_1_ordered_dithering(o_img, in_PaletteH, Progress,order=8) #Slow, must use order = 8
        elif fullD == 8:
            o_img = hitherdither.ordered.cluster.cluster_dot_dithering(o_img, in_PaletteH,order=8, thresholds=Fthr) #Fast
        else:
            fsPal = [element for sublist in rgb_in for element in sublist[0]]
            plen = len(fsPal)//3
            fsPal.extend(fsPal[:3]*(256-plen))
            #create tmp PIL image with desired palette
            tmpI = Image.new('P',(1,1))
            tmpI.putpalette(fsPal)
            o_img = o_img.quantize(colors=len(rgb_in), palette=tmpI)
    else:
        o_img = in_PaletteH.create_PIL_png_from_rgb_array(o_img)  # Fastest, no dither

    #dpg.set_value('progress',0.5) #Reset progress bar
    Progress[0] = 0.5

    d_colors = o_img.getcolors()
    d_counts = [d_colors[i][0] for i in range(len(d_colors))]

    #Prevent iterating for more than one best global color, replace with estimated
    for x in range(len(bg_color)):
        if (bg_color[x] == -2) and (bg_color.count(-2)>1):
            bg_color[x] = -1

    bestbg = [False]*k
    if Mode['global_colors'].count(True) > 0:
        if -1 in bg_color:  #Estimate best global colors
            n_img = np.asarray(o_img)
            ccount = [np.array([],np.int16)] * len(bg_color)
            for j in range(0,height,y_step):        # Step thru attribute cells
                for i in range(0,width,x_step):
                    z = np.reshape(n_img[j:j+y_step,i:i+x_step],(-1))   #get bitmap cell
                    if len(np.unique(z)) >= k:
                        ucount = np.bincount(z)
                        for l,t in enumerate(bg_color):
                            if t == -1:
                                ccount[l] = np.append(ccount[l],np.argmax(ucount))
                                ucount[np.argmax(ucount)] = -1  #Remove the color 
            for j in range(len(bg_color)):
                if bg_color[j] == -1:
                    if len(ccount[j]) > 0:
                        bg_color[j] = np.argmax(np.bincount(ccount[j]))
                    else:
                        bg_color[j] = 0
            #bg_color = rgb_in[d_colors[np.abs(d_counts-np.percentile(d_counts,55)).argmin()][1]][1] #d_counts.index(max(d_counts))
        bestbg = [True if x == -2 else False for x in bg_color]

    if Mode['attr']!=(0,0):
        o2_img = o_img.convert('RGB')
        n_img = np.asarray(o2_img)

        if True in bestbg:    #gfxmode == 2 and bestbg:
            cells3 = [[] for j in range(c_count)]     #[[] for j in range(16)]
            buffers = [get_buffers() for j in range(c_count)]
        else:
            cells3 = []     #[[] for j in range(16)]
            buffers = get_buffers()
        # cells3 = []
        # buffers = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #k = 4

        pal:Palette.Palette    #hitherdither.palette.Palette

        for j in range(0,height,y_step):        # Step thru attribute cells
            for i in range(0,width,x_step):
                z = np.reshape(n_img[j:j+y_step,i:i+x_step],(-1,3))   #get bitmap cell
                z = np.float32(z)
                ret,label,center=cv2.kmeans(z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) #kmeans to k colors
                center = np.uint8(center)
                res = center[label.flatten()]
                pc = Image.fromarray(res.reshape((y_step,x_step,3)))
                cc = pc.getcolors() # Kmeans palette [count,[r,g,b]]
                if True in bestbg:
                    for bg in range(0,c_count):  #Iterate through background colors
                        bg_color[bestbg.index(True)] = bg
                        attr,pal = get_attr(cc,rgb_in,rgb_out,bg_color)        #get closest colors to the kmeans palette + background color
                        pcc = pal.create_PIL_png_from_rgb_array(pc)
                        bm_pack(i//x_step,j//y_step,pcc,buffers[bg])
                        attr_pack(i//x_step,j//y_step,attr,buffers[bg])
                        #i_to = np.array(attr)   #Translate to display palette
                        #t_img=np.asarray(pcc)
                        #mask = i_to[t_img-t_img.min()]
                        #pcc= Image.fromarray(np.uint8(mask))    #now it should be indexed to the display palette
                        #pcc.putpalette(tPal)
                        cells3[bg].append(np.asarray(pcc.convert('RGB'))) #bitmap cells
                else:
                    attr,pal = get_attr(cc,rgb_in,rgb_out,bg_color)     #get closest colors to the kmeans palette + background color
                    pcc = pal.create_PIL_png_from_rgb_array(pc)     #converted from rgb to dither palette colors
                    #pcc = custom_dithering(pc, pal, Fthr, type = 5)
                    bm_pack(i//x_step,j//y_step,pcc,buffers)
                    attr_pack(i//x_step,j//y_step,attr,buffers)
                    #i_to = np.array(attr)   #Translate to display palette
                    #t_img=np.asarray(pcc)
                    #mask = i_to[t_img-t_img.min()]
                    #pcc= Image.fromarray(np.uint8(mask))    #now it should be indexed to the display palette
                    #pcc.putpalette(tPal)
                    cells3.append(np.asarray(pcc.convert('RGB'))) #bitmap cells
                progress += cell_prog
                #dpg.set_value('progress',progress)
                Progress[0] = progress
        columns=width//x_step
        if True in bestbg:
            bix = bestbg.index(True)
            en_img = [np.zeros([height,width,3],dtype='uint8') for j in range(c_count)]
            e_img = [None for j in range(c_count)]
            err = []
            for bg in range(0,c_count):
                for i,c in enumerate(cells3[bg]):
                    sr = int(i/columns)*y_step
                    er = sr+y_step
                    sc = (i*x_step)%width
                    ec = sc+x_step
                    en_img[bg][sr:er,sc:ec] = c
                err.append(1-(cv2.norm( n_img, en_img[bg], cv2.NORM_L2 ))/pixelcount)
                e_img[bg] = Image.fromarray(en_img[bg]).resize(Mode['in_size'],Image.NEAREST) 
            bg_color[bix] = err.index(max(err)) #Best bg color (index to...)
            n_img = np.asarray(in_PaletteH.create_PIL_png_from_rgb_array(np.asarray(e_img[bg_color[bix]])))
            mask = i_to[n_img]
            o_img= Image.fromarray(np.uint8(mask))
            #o_img = in_PaletteH.create_PIL_png_from_rgb_array(mask)
            o_img.putpalette(tPal)
            e_img = o_img.convert('RGB')
            buffers=buffers[bg_color[0]]
        else:
            en_img = np.zeros([height,width,3],dtype='uint8')   #[np.zeros([200,160,3],dtype='uint8') for j in range(16)]
            #Build final bitmap image
            for i,c in enumerate(cells3):
                sr = int(i/columns)*y_step
                er = sr+y_step
                sc = (i*x_step)%width
                ec = sc+x_step
                en_img[sr:er,sc:ec] = c
            e_img = Image.fromarray(en_img)
            if Mode['in_size']!=Mode['out_size']:
                e_img = e_img.resize(Mode['in_size'],Image.NEAREST)
            n_img = np.asarray(in_PaletteH.create_PIL_png_from_rgb_array(np.asarray(e_img)))
            mask = i_to[n_img]
            #rmap = i_to[mask.argmax(1)]
            o_img= Image.fromarray(np.uint8(mask))
            #o_img = in_PaletteH.create_PIL_png_from_rgb_array(mask)
            o_img.putpalette(tPal)
            e_img = o_img.convert('RGB')
    else:   #Unrestricted
        n_img = np.asarray(o_img)
        mask = i_to[n_img]
        #rmap = i_to[mask.argmax(1)]
        o_img= Image.fromarray(np.uint8(mask))
        #o_img = in_PaletteH.create_PIL_png_from_rgb_array(mask)
        o_img.putpalette(tPal)
        e_img = o_img.convert('RGB')
        buffers =[]
    #if bestbg:
    #    return(e_img[bg_color],buffers[bg_color],rgb_in[bg_color][1])
    #else:
    Progress[0] = 1
    # if (bg_color[0]<0) or (Mode['global_colors'].count(True) == 0):
    #     bg_color[0]=0
    # else:
    #     bg_color[0]=next(i for i,x in enumerate(rgb_in) if x[1]==bg_color[0])
    return(e_img,buffers,bg_color)
#############################


def clear_preview():
    global prev_tex
    prev_tex.fill(0)

########## Callbacks
#############################

# Quit program
def quit_callback():
    global Quit
    dpg.configure_item("quit_id", show=False)
    Quit = True

def print_me(sender, app_data, user_data):
    print(f"Menu Item: {sender} - {app_data} - {user_data}")
    return

# Show about modal window
def show_about():
    cfg = dpg.get_item_configuration('MainW')
    dpg.configure_item("aboutw_id", pos=((cfg['width']/2)-180, (cfg['height']/2)-120))
    dpg.configure_item("aboutw_id", show = True)

# Show file->open dialog
def show_dialog():
    dpg.show_item("click_handler")
    dpg.show_item("open_dialog")

# Open file from File dialog
def open_file(sender, app_data):
    global og_img, input_img, og_tex, pre_img
    dpg.hide_item("click_handler")  # Stop mouse click reporting
    try:
        Progress[0] = 0
        input_img = Image.open(app_data['file_path_name']).convert('RGB') # Open file
        id = next(i for i, item in enumerate(GFX_MODES) if item["name"] == dpg.get_value("gfx_mode"))
        og_img, drag_data['zoom'], drag_data['pos'] = frameResize(input_img,id) #Crop/Resize
        drag_data['old_drag'] = [0,0]
        if dpg.get_value('autocolor') == True:
            b, c, s = imagePreset(og_img)
            dpg.set_value('im_b',b)
            dpg.set_value('im_s',s)
            dpg.set_value('im_c',c)
            sharp = 1.5
            dpg.set_value('im_sharp',sharp)
            h = 180
            dpg.set_value('im_h', h)
        else:
            b = dpg.get_value('im_b')
            s = dpg.get_value('im_s')
            c = dpg.get_value('im_c')
            h = dpg.get_value('im_h')
            sharp = dpg.get_value('im_sharp')
        pre_img = imageProcess(og_img,b, c, s, h, sharp) # Processed image
        tmp = np.asarray(pre_img, dtype=np.float32)/255
        og_tex[:] = tmp[:] # copy input image to Dear PyQUI buffer
        cv_tex.fill(0)  # clear converted preview image
        cv_img = None   # <-/
        dpg.set_value('filter_save','-save')    # hide save menues
        #Start drag/zoom handlers
        dpg.show_item('global_drag_handler')
        dpg.bind_item_handler_registry("og_button", "drag_handler")
    except Exception as e:
        print(e)
        dpg.configure_item('error_id', show= True)
        dpg.configure_item('errortext_id', default_value="Could not open file")

# Save file
def save_file(sender, app_data, user_data):
    dpg.configure_item("ow_id", show=False)
    if user_data[1][2] == None:
        cv_img.save(user_data[0],'PNG')
    else:
        f_data = user_data[1][2](cv_data[0],cv_data[1])
        cfile = open(user_data[0],"wb")
        cfile.write(f_data)
        cfile.close
    return

# Check if file exist and prompt for overwrite if so
def check_file(sender,app_data, user_data):
    if file_exists(app_data['file_path_name']):
        dpg.configure_item('owb_id', user_data=[app_data['file_path_name'],user_data])
        dpg.configure_item('ow_id', show=True)
    else:
        save_file(None,None,[app_data['file_path_name'],user_data])

def show_save(sender, app_data):
    if "png" in sender:
        filetype=['PNG','.png',None]
    else:
        filetype=GFX_MODES[int(sender[7:9])]['save_output'][int(sender[10:12])]
    # Save file dialog
    with dpg.file_dialog(label="Save "+filetype[0]+" file", directory_selector=False, show=True, callback=check_file, min_size=(500,400), user_data=filetype):
        dpg.add_file_extension("", color=(150, 255, 150, 255))
        dpg.add_file_extension(extension=filetype[1], color=(255, 255, 64, 255))


# Process mouse clicks for file dialog image preview
def fileclick(sender, app_data):
    data = dpg.get_file_dialog_info("open_dialog")
    u_data = dpg.get_item_user_data("open_dialog")
    if data['file_path_name']!=u_data:
        dpg.set_item_user_data("open_dialog", data['file_path_name'])
        try:    #Try loading the image preview
            tmp = Image.open(data['file_path_name']).convert('RGB')
            pv_img= ImageOps.pad(tmp,(200,240))
            tmp = np.asarray(pv_img, dtype=np.float32)/255
            prev_tex[:] = tmp[:]    #copy input image to Dear PyQUI buffer
        except:
            clear_preview()

# Color process image
def enhance_callback():
    global pre_img, og_tex
    if og_img != None:
        b = dpg.get_value('im_b')
        s = dpg.get_value('im_s')
        c = dpg.get_value('im_c')
        h = dpg.get_value('im_h')
        sharp = dpg.get_value('im_sharp')
        pre_img = imageProcess(og_img,b, c, s, h, sharp) #Processed image
        tmp = np.asarray(pre_img, dtype=np.float32)/255
        og_tex[:] = tmp[:] #copy input image to Dear PyQUI buffer
        Progress[0] = 0

# Reset Color process sliders
def resetsliders():
    dpg.set_value('im_b',1)
    dpg.set_value('im_s',1)
    dpg.set_value('im_c',1)
    dpg.set_value('im_sharp',1)
    dpg.set_value('im_h', 180)
    enhance_callback()

# Show or hide threshold widgets
def change_dither(sender,app_data):
    notrh = ['None','Yliluoma (slow)', 'Floyd-Steinberg']
    if sender == 'ldither':
        target = 'yth'
    else:
        target = 'fth'
    dpg.configure_item(target,show=(app_data not in notrh))
    if (app_data == notrh[1]) and (len(Work_Palette) > 16):
        dpg.configure_item('error_id', show= True)
        dpg.configure_item('errortext_id', default_value="Yliluoma is extremely slow for more than 16 colors!")

# Set background color
def set_bgcolor(sender):
    global bgcolor
    bgcolor[int(sender[0])] = int(sender[7:])
    dpg.set_value('bg_color'+sender[0], dpg.get_value(sender))
    dpg.configure_item('bgselect_id', show=False)
    dpg.delete_item('bgselect_id')

# Set graphic mode
def set_mode(sender, appdata):
    global bgcolor,gfx_ix
    global Work_Palette,View_Palette,Palettes
    global og_tex,cv_tex,og_img,pre_img,drag_data,cv_img

    if sender == 'gfx_mode':
        new_id = next(i for i, item in enumerate(GFX_MODES) if item["name"] == dpg.get_value("gfx_mode"))
        if new_id != gfx_ix:
            Progress[0] = 0
            # Remove dither palette checkboxes and themes
            dpg.delete_item('d_palette',children_only=True)
            for i in range(len(Work_Palette)):
                dpg.remove_alias('pt'+str(i).zfill(3))
            # Remove display palette color buttons
            dpg.delete_item('v_palette',children_only=True)
            # Remove background color select buttons
            for k,i in enumerate(GFX_MODES[new_id]['global_names']):
                if i!=None:
                    dpg.delete_item('bgselect_id')
            dpg.delete_item('bg_group',children_only=True)

            # Generate new themes and checkboxes
            # Add new color buttons
            pals = [x[0] for x in GFX_MODES[new_id]['palettes']]
            Work_Palette=copy.deepcopy(GFX_MODES[new_id]['palettes'][0][1])
            View_Palette=copy.deepcopy(GFX_MODES[new_id]['palettes'][0][1])
            Palettes=[0,0]
            palette_widgets(new_id)
            dpg.configure_item('dpalette',items=pals,default_value=pals[0])
            dpg.configure_item('vpalette',items=pals,default_value=pals[0])
            bgcolor = [-1]*len(GFX_MODES[new_id]['global_names'])
            for k,i in enumerate(GFX_MODES[new_id]['global_names']):
                if i!=None:
                    dpg.set_value('bg_color'+str(k),Work_Palette[bgcolor[k] if bgcolor[k]>=0 else 0]['RGBA'])
            # Change Image preview buffer/textures if new size
            if GFX_MODES[gfx_ix]['in_size']!=GFX_MODES[new_id]['in_size']:
                #Delete textures
                dpg.remove_alias('ogimg_id')
                dpg.remove_alias('cvimg_id')
                in_size=GFX_MODES[new_id]['in_size']
                og_tex=np.zeros((in_size[1],in_size[0],3), dtype=np.float32)
                cv_tex=np.zeros((in_size[1],in_size[0],3), dtype=np.float32)
                with dpg.texture_registry():
                    dpg.add_raw_texture(width=in_size[0], height=in_size[1], default_value= og_tex, format=dpg.mvFormat_Float_rgb, tag="ogimg_id")
                    dpg.add_raw_texture(width=in_size[0], height=in_size[1], default_value= cv_tex, format=dpg.mvFormat_Float_rgb, tag="cvimg_id")
                dpg.configure_item('og_button',width=in_size[0],height=in_size[1],texture_tag="ogimg_id")
                dpg.configure_item('cv_button',width=in_size[0],height=in_size[1],texture_tag="cvimg_id")
                #Reset preview images
                if input_img != None:
                    og_img, drag_data['zoom'], drag_data['pos'] = frameResize(input_img,new_id) #Crop/Resize
                    drag_data['old_drag'] = [0,0]
                    b = dpg.get_value('im_b')
                    s = dpg.get_value('im_s')
                    c = dpg.get_value('im_c')
                    h = dpg.get_value('im_h')
                    sharp = dpg.get_value('im_sharp')
                    pre_img = imageProcess(og_img,b, c, s, h, sharp) # Processed image
                    tmp = np.asarray(pre_img, dtype=np.float32)/255
                    og_tex[:] = tmp[:] # copy input image to Dear PyQUI buffer
                    cv_tex.fill(0)  # clear converted preview image
                    cv_img = None   # <-/
                dpg.set_value('filter_save','-save')    # Buffers got destroyed, hide save menues
            gfx_ix = new_id
    else:
        wx = int(sender[-1])
        dpg.configure_item('bg_group2_'+str(wx), show= (appdata=='Manual selection'))
        if appdata=='Manual selection':
            bgcolor[wx] = next(i for i, item in enumerate(Work_Palette) if item['RGBA'] == dpg.get_value("bg_color"+str(wx)))#PaletteRGB.index(dpg.get_value('bg_color')[:3])
        elif appdata=='Estimate':
            bgcolor[wx] = -1
        else:
            bgcolor[wx] = -2

# Convert image
def convert():
    global bgcolor, cv_img, cv_data

    def pre_mono():
        # is assumed that the Work_Palette has only 2 enabled colors when calling this
        tmp=[[0,[]],[0,[]]] #position, [r,g,b,a]
        count=0
        for j,c in enumerate(Work_Palette):
            if count>1:
                continue
            if c['enabled']:
                tmp[count]=[j,c['RGBA']]
                count+=1
        result=copy.deepcopy(Work_Palette)
        Y0=CC.Luma(tmp[0][1])
        Y1=CC.Luma(tmp[1][1])
        result[tmp[0][0]]['RGBA']=([0,0,0,255]if Y0<=Y1 else [255,255,255,255])
        result[tmp[1][0]]['RGBA']=([0,0,0,255]if Y1<Y0 else [255,255,255,255])
        return result

    if pre_img != None:
        for wx, n in enumerate(GFX_MODES[gfx_ix]['global_names']):
            if n != None:
                set_mode('bg_mode'+str(wx),dpg.get_value('bg_mode'+str(wx)))
        if dpg.get_value('mono1'):
            pal_in=pre_mono()   # modify palette for b&w process
        else:
            pal_in=Work_Palette
        dpg.configure_item('busy_id', show=True, pos=(370,120)) #Busy modal window kept creeping 1px right each time
        #id = next(i for i, item in enumerate(GFX_MODES) if item["name"] == dpg.get_value("gfx_mode"))
        cv_img, data, bgcolor = Image_convert(pre_img, pal_in, View_Palette,gfx_ix,
                        ditherlist.index(dpg.get_value('ldither')), ditherlist.index(dpg.get_value('fdither')),
                        dpg.get_value('yth'), dpg.get_value('fth'), bgcolor)
        cv_data = [data, bgcolor]
        dpg.configure_item('busy_id', show=False)
        tmp = np.asarray(cv_img, dtype=np.float32)/255
        cv_tex[:] = tmp[:] #copy input image to DearPyGUI buffer
        for i,n in enumerate(GFX_MODES[gfx_ix]['global_names']):
            if n != None:
                dpg.set_value('bg_color'+str(i), Work_Palette[bgcolor[i]]['RGBA'])
        dpg.set_value('filter_save','png,save_id'+str(gfx_ix).zfill(2))

# Input handlers for source image drag/zoom

def global_handler(sender, app_data):
    global drag_data
#    global og_img
    change = False
 
    if dpg.is_item_hovered('og_button'):
        htype=dpg.get_item_info(sender)["type"]
        if htype=="mvAppItemType::mvMouseWheelHandler":
            drag_data['wheel'] = app_data


def drag_handler(sender, app_data):
    global drag_data
    change = False
    global og_img

    htype=dpg.get_item_info(sender)["type"]
    if htype=="mvAppItemType::mvHoverHandler":
        if drag_data['wheel'] != 0:
            drag_data['zoom'] += drag_data['wheel']/100
            drag_data['wheel'] = 0
            drag_data['zoom'] = max(0.05,min(drag_data['zoom'], 5)) # Clamp zoom (5%-500%)
            change = True
    elif htype=="mvAppItemType::mvFocusHandler":
        if dpg.is_item_active('og_button'):
            if not drag_data['on_drag']:
                drag_data['on_drag']= True
                drag_data['old_mouse']= dpg.get_mouse_pos()
            drag_data['drag'][0]= dpg.get_mouse_pos()[0] - drag_data['old_mouse'][0]
            drag_data['drag'][1]= dpg.get_mouse_pos()[1] - drag_data['old_mouse'][1]
        else:
            drag_data['on_drag']= False
            drag_data['drag']=[0,0]
            drag_data['old_drag']=[0,0]
            
        if drag_data['drag']!=drag_data['old_drag']:
            drag_data['pos'][0] += drag_data['drag'][0]-drag_data['old_drag'][0]
            drag_data['pos'][1] += drag_data['drag'][1]-drag_data['old_drag'][1]
            drag_data['old_drag']= drag_data['drag'].copy()
            change = True
    elif htype=="mvAppItemType::mvClickedHandler":
        og_img, drag_data['zoom'], drag_data['pos'] = frameResize(input_img,gfx_ix) #Reset zoom/drag
        enhance_callback()
        change = False
    if change:
        in_size= GFX_MODES[gfx_ix]['in_size']
        nwidth = int(input_img.size[0]*drag_data['zoom'])
        nheight = int(input_img.size[1]*drag_data['zoom'])
        og_img = input_img.resize((nwidth,nheight),Image.LANCZOS)
        og_img = og_img.crop((-drag_data['pos'][0],-drag_data['pos'][1],-drag_data['pos'][0]+in_size[0],-drag_data['pos'][1]+in_size[1]))
        enhance_callback()

# Palette enable colors
def palette_handler(sender, app_data):
    global Work_Palette
    ncols = sum(1 for c in Work_Palette if c['enabled']==True)
    if not app_data and ncols == 2:
        dpg.set_value(sender,True)
        return
    Work_Palette[int(sender[-3:])]['enabled']=app_data
    ncols = sum(1 for c in Work_Palette if c['enabled']==True)
    dpg.configure_item('mono1',show=ncols==2)
    dpg.set_value('mono1',dpg.get_value('mono1') and ncols==2)

# Palette selection
def palette_select_handler(sender, app_data):
    global Work_Palette,View_Palette
    if sender=='dpalette':
        #Dither palette
        pal = next(i for i,item in enumerate(GFX_MODES[gfx_ix]['palettes']) if item[0] == app_data)
        if Palettes[0]!=pal:
            Palettes[0]=pal
            Work_Palette=copy.deepcopy(GFX_MODES[gfx_ix]['palettes'][pal][1])
            #Update color checkboxes
            for i in range(len(Work_Palette)):
                dpg.remove_alias('pt'+str(i).zfill(3))
                with dpg.theme(tag='pt'+str(i).zfill(3)):
                    with dpg.theme_component(dpg.mvAll):
                        rgb = Work_Palette[i]['RGBA']
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,rgb,category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,rgb,category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,rgb,category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_CheckMark,[255,255,255] if CC.Luma(rgb)<128 else [0,0,0],category=dpg.mvThemeCat_Core)
                dpg.bind_item_theme('palette'+str(i).zfill(3),'pt'+str(i).zfill(3))
                Work_Palette[i]['enabled']=dpg.get_value('palette'+str(i).zfill(3))
                for k,j in enumerate(GFX_MODES[gfx_ix]['global_names']):
                    if j != None:
                        dpg.set_value(str(k)+'bcolor'+str(i),Work_Palette[i]['RGBA'])
            for k,j in enumerate(GFX_MODES[gfx_ix]['global_names']):
                if j != None:
                    dpg.set_value('bg_color'+str(k),Work_Palette[bgcolor[0]]['RGBA'])
    else:
        #Display palette
        pal = next(i for i,item in enumerate(GFX_MODES[gfx_ix]['palettes']) if item[0] == app_data)
        if Palettes[1]!=pal:
            Palettes[1]=pal
            View_Palette=copy.deepcopy(GFX_MODES[gfx_ix]['palettes'][pal][1])
            #Update color buttons
            for i in range(len(View_Palette)):
                dpg.set_value('vcolor'+str(i),View_Palette[i]['RGBA'])
            #TODO: Update Converted image (replace palette on I pil image)


########## GUI Elements
#############################

def color_selector(sender, app_data):
    color_count = len(Work_Palette)
    ix = int(sender[8:])
    name = GFX_MODES[gfx_ix]['global_names'][ix]
    with dpg.window(tag="bgselect_id", show=True, no_close=True, modal=True,pos=(200,200)):
        dpg.add_text("Select "+name)
        with dpg.group(tag='bg_select'):
                for j in range(0,color_count,8):
                    with dpg.group(horizontal=True):
                        for k in range(j,j+8 if j+8<color_count else color_count):
                            dpg.add_color_button(Work_Palette[k]['RGBA'],tag=str(ix)+'bcolor'+str(k), callback=set_bgcolor)


def palette_widgets(gmode):
    color_count = len(Work_Palette)

    with dpg.group(parent='bg_group', show=(len(GFX_MODES[gmode]['global_names'])>0)):
        for i,name in enumerate(GFX_MODES[gmode]['global_names']):
            if name != None:
                dpg.add_text(name)
                dpg.add_separator()
                dpg.add_combo(['Estimate','Find best (slow)','Manual selection'], default_value='Estimate', tag='bg_mode'+str(i), callback=set_mode, width=200)
                with dpg.group(horizontal=True, tag='bg_group2_'+str(i), show=False):
                    b = dpg.add_color_button(Work_Palette[0]['RGBA'],label=name, tag='bg_color'+str(i), callback=color_selector) #
                    dpg.add_text(name)

    for i,r in enumerate(Work_Palette):
        with dpg.theme(tag='pt'+str(i).zfill(3)):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,r['RGBA'],category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,r['RGBA'],category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,r['RGBA'],category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,[255,255,255] if CC.Luma(r['RGBA'])<128 else [0,0,0],category=dpg.mvThemeCat_Core)
    for j in range(0,color_count,16):
        with dpg.group(horizontal=True,parent='d_palette'):
            for i in range(j,j+16 if j+16<color_count else color_count):
                b = dpg.add_checkbox(label='',default_value=Work_Palette[i]['enabled'],enabled=Work_Palette[i]['enabled'],
                    tag='palette'+str(i).zfill(3),callback=palette_handler)
                if Work_Palette[i]['color']!='':
                    with dpg.tooltip(b):
                        dpg.add_text(Work_Palette[i]['color'])
                dpg.bind_item_theme('palette'+str(i).zfill(3),'pt'+str(i).zfill(3))


    color_count = len(View_Palette)
    for j in range(0,color_count,16):
        with dpg.group(horizontal=True,parent='v_palette'):
            for i in range(j,j+16 if j+16<color_count else color_count):
                b=dpg.add_color_button(View_Palette[i]['RGBA'], tag='vcolor'+str(i))
                if View_Palette[i]['color']!='':
                    with dpg.tooltip(b):
                        dpg.add_text(View_Palette[i]['color'])

def Init_GUI():
    global og_tex, cv_tex

    dpg.create_context()
    dpg.create_viewport(title='ClashState '+str(version), width=800,height=600, resizable=False)
    dpg.setup_dearpygui()

    # GUI Font
    with dpg.font_registry():
        font1 = dpg.add_font(os.path.abspath(os.path.join(CC.bundle_dir, "assets/Chicago1st3.ttf")), 15)
    dpg.bind_font(font1)

    # Global input handlers
    with dpg.handler_registry(show=False, tag="global_drag_handler"):
        m_wheel = dpg.add_mouse_wheel_handler(callback= global_handler)



    # Error prompt
    with dpg.window(label="Error", modal=True, show=False, id="error_id", pos=(200,200)):
        dpg.add_text("Error", tag='errortext_id')
        dpg.add_button(label="OK", width=75, callback=lambda:dpg.configure_item('error_id',show=False))

    # Quit dialog
    with dpg.window(label="Quit Program", modal=True, show=False, id="quit_id", no_title_bar=True, pos=(200,200), no_resize=True):
        dpg.add_text("Exit program?")
        dpg.add_separator()
        with dpg.group(horizontal=True, pos=(0,60)):
            dpg.add_button(label="OK", width=75, callback=quit_callback)
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("quit_id", show=False))

    # Overwrite dialog
    with dpg.window(label="File Exists", modal=True, show=False, id="ow_id", no_title_bar=True, pos=(200,200), no_resize=True):
        dpg.add_text("Overwrite file?")
        dpg.add_separator()
        with dpg.group(horizontal=True, pos=(0,60)):
            dpg.add_button(label="OK", width=75, tag="owb_id", callback=save_file)
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("ow_id", show=False))

    # About splash
    width, height, channels, data = dpg.load_image(os.path.abspath(os.path.join(CC.bundle_dir, 'assets/splash.gif'))) # 0: width, 1: height, 2: channels, 3: data

    in_size=GFX_MODES[0]['in_size']
    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag="splash_id")
        dpg.add_raw_texture(width=in_size[0], height=in_size[1], default_value= og_tex, format=dpg.mvFormat_Float_rgb, tag="ogimg_id")
        dpg.add_raw_texture(width=in_size[0], height=in_size[1], default_value= cv_tex, format=dpg.mvFormat_Float_rgb, tag="cvimg_id")
        dpg.add_raw_texture(width=200, height=240, default_value= prev_tex, format=dpg.mvFormat_Float_rgb, tag="preview_id")

    # Open file dialog
    with dpg.file_dialog(label="Open file", directory_selector=False, show=False, callback=open_file, tag="open_dialog", min_size=(700,400)):
        dpg.add_file_extension("", color=(255, 155, 50, 255))
        dpg.add_file_extension("Image files{.gif,.GIF,.jpg,.JPG,.jpeg,.JPEG,.png,.PNG}", color=(0, 255, 64, 255))
        with dpg.child_window(width= 240, height=290):
            dpg.add_text("Preview")
            dpg.add_image(texture_tag="preview_id", width=200, height=240)
        with dpg.handler_registry(show=False, tag='click_handler'):
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=fileclick)

    #About modal
    with dpg.window(label="About", modal=True, show=False, tag="aboutw_id", no_title_bar=True,pos=(200,200),no_resize=True):
        dpg.add_image_button(texture_tag="splash_id", frame_padding=0, callback=lambda: dpg.configure_item("aboutw_id", show = False))
        dpg.add_separator()
        dpg.add_text("ClashState v"+str(version),color=(128,240,0))
        dpg.add_separator()
        dpg.add_text("©2022-2023 by Durandal/Retrocomputacion")

    #Busy modal
    with dpg.window(modal=True, show=False, tag='busy_id', no_title_bar=True, no_move=True, no_resize=True, no_background=True, pos=(370,120)):
        dpg.add_loading_indicator()

    # Main window
    with dpg.window(tag="MainW"):
        # Menues
        with dpg.menu_bar():
            with dpg.menu(label="File", tag="file_menu"):
                dpg.add_menu_item(label='Open', callback=show_dialog)
                with dpg.filter_set(tag='filter_save'):
                    dpg.add_menu_item(label="Save PNG", show=True, callback=show_save, tag="save_png", filter_key="save_png")
                    for ix, i in enumerate(GFX_MODES):  # Add all the output file format menu items
                        if i['save_output']!=None:
                            for iy, f in enumerate(i['save_output']):
                                f_tag="save_id"+str(ix).zfill(2)+"_"+str(iy).zfill(2)
                                dpg.add_menu_item(label="Save as "+f[0],show=True,callback=show_save,tag=f_tag,filter_key=f_tag)
                dpg.set_value('filter_save','-save')
                dpg.add_menu_item(label="Quit", callback= lambda: dpg.configure_item("quit_id", show = True), tag="quitm_id")
            with dpg.menu(label="Options"):
                with dpg.menu(label="Color matching"):
                    dpg.add_radio_button(('Euclidean','CCIR 601','LAb DeltaE CIEDE2000'), default_value='Euclidean', tag='ccir')
                with dpg.menu(label="Luminance dithering"):
                    dpg.add_radio_button(('Over input palette','Black & White'), default_value='Over input palette',tag='luma_mode')
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About...", callback= show_about)

        # Image previews
        with dpg.group(horizontal=True):
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_text("Original image")
                    t = dpg.add_text("(?)", color=[0,255,0])
                    with dpg.tooltip(t):
                        dpg.add_text("Move: Left click & drag\nZoom: Mouse wheel\nReset: Right click")
                dpg.add_image_button(texture_tag="ogimg_id", tag="og_button", frame_padding=0, indent=30)
            with dpg.group(indent=400):
                dpg.add_text("Converted image")
                dpg.add_image_button(texture_tag="cvimg_id", tag="cv_button", frame_padding=0, indent=30)

        # Image drag/zoom item handler
        with dpg.item_handler_registry(tag="drag_handler", show=True):
            dpg.add_item_hover_handler(callback=drag_handler)
            dpg.add_item_focus_handler(callback=drag_handler)
            dpg.add_item_clicked_handler(dpg.mvMouseButton_Right, callback=drag_handler)

        # Settings
        with dpg.child_window(height=245, border=False):
            with dpg.group(horizontal=True):
                # Color adjustments
                with dpg.tab_bar():
                    with dpg.tab(label="Color Adjustments"):
                        #dpg.add_separator()
                        dpg.add_slider_float(label="Contrast", tag='im_c',width=200, min_value= 0, max_value= 2, default_value= 1, callback=enhance_callback)
                        dpg.add_slider_float(label="Brightness", tag='im_b', width=200, min_value= 0, max_value= 2, default_value= 1, callback=enhance_callback)
                        dpg.add_slider_int(label="Hue", tag='im_h', width=200, min_value= 0, max_value= 360, default_value= 180, callback=enhance_callback)
                        dpg.add_slider_float(label="Saturation", tag='im_s', width=200, min_value= 0, max_value= 2, default_value= 1, callback=enhance_callback)
                        dpg.add_slider_float(label="Sharpness", tag='im_sharp', width=200, min_value= 0, max_value= 2, default_value= 1, callback=enhance_callback)
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="Auto-set on file open", tag='autocolor', default_value=True)
                            dpg.add_button(label='Reset', indent= 240, callback=resetsliders)
                    # Palette selector/editor
                    with dpg.tab(label="Palette"):
                        pals = [x[0] for x in GFX_MODES[0]['palettes']]
                        # Dither Palette
                        dpg.add_combo(pals,label='Dither palette',default_value=pals[0],width=260,tag='dpalette',callback=palette_select_handler)
                        # Create themes
                        with dpg.theme(tag='pal_theme'):
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,2,2,category=dpg.mvThemeCat_Core)
                                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,5,0,category=dpg.mvThemeCat_Core)
                        dpg.add_group(tag='d_palette')
                        dpg.bind_item_theme('d_palette','pal_theme')
                        color_count=len(Work_Palette)
                        dpg.add_checkbox(label='Process as B&W',tag='mono1', default_value=False,show=(color_count==2))
                        # Display Palette
                        with dpg.theme(tag='combo'):
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,10,10,category=dpg.mvThemeCat_Core)
                        dpg.add_combo(pals,label='Display palette',default_value=pals[0],width=260,tag='vpalette',callback=palette_select_handler)
                        dpg.add_group(tag='v_palette')
                        dpg.bind_item_theme('v_palette','pal_theme')
                        # Add color checkboxes as buttons

                # Dithering methods, etc
                with dpg.group(indent = 400):
                    with dpg.tab_bar():
                        with dpg.tab(label="Dithering"):
                            #dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_combo( ditherlist, label='Luminance',default_value='None',tag='ldither', width=160, callback=change_dither)
                                dpg.add_drag_int(tag='yth',default_value=4, min_value=1, max_value=5, label='Threshold', width=50, clamped=True, speed=0.2, indent=250, show= False)
                            with dpg.group(horizontal=True):
                                dpg.add_combo( ditherlist, label='Final',default_value='Bayer 8x8',tag='fdither', width=160, callback=change_dither)
                                dpg.add_drag_int(tag='fth',default_value=4, min_value=1, max_value=5, label='Threshold', width=50, clamped=True, speed=0.2, indent=250)
                            #Graphic mode
                            dpg.add_text("Graphic Mode")
                            dpg.add_separator()

                            #Generate GFX modes list
                            gfxmodes=[]
                            for item in GFX_MODES:
                                gfxmodes.append(item['name'])

                            dpg.add_combo(gfxmodes,default_value=gfxmodes[0], tag='gfx_mode', callback=set_mode, width=200)
                            #Background color
                            dpg.add_group(tag='bg_group', show=True)
                           
        dpg.add_separator()
        dpg.add_button(label='Convert!', width=200, height= 50, indent=300, callback=convert)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_progress_bar(tag='progress',default_value=0,width=-20,indent=20)

    palette_widgets(0)

    dpg.set_viewport_small_icon("assets/icon.gif")  #App icon
    dpg.show_viewport()
    dpg.set_primary_window("MainW",True)

#################################

########Main
#################################
if __name__ == '__main__':

    build_modes()

    Work_Palette = copy.deepcopy(GFX_MODES[0]['palettes'][0][1])  # Copy default palette from default gfx mode 
    View_Palette = copy.deepcopy(GFX_MODES[0]['palettes'][0][1])  # Copy default palette from default gfx mode 

    # Initialize texture buffers
    in_size=GFX_MODES[0]['in_size']
    og_tex=np.zeros((in_size[1],in_size[0],3), dtype=np.float32)
    cv_tex=np.zeros((in_size[1],in_size[0],3), dtype=np.float32)

    CC.bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__))) #When compiling with Nuitka dont use this

    Init_GUI()
    _prog = Progress[0]

    while dpg.is_dearpygui_running() and not Quit:
        # insert here any code you would like to run in the render loop
        # you can manually stop by using stop_dearpygui()
        if Progress[0] != _prog:
            dpg.set_value('progress',Progress[0])
            _prog = Progress[0]
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
