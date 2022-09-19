###############################################################
#                  Methane labeller GUI                            
#             Author: Tai-Long He (taihe@uw.edu)              
###############################################################
from tkinter import END
import tkinter as tk
from backend.ee_retrieval import satellite_database, get_plume
from backend.utils import zstandard
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.widgets import LassoSelector
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import zscore
from tkinter import messagebox
import os
import warnings
warnings.filterwarnings("ignore")

plt.rc('font', size=6) #controls default text size

LARGE_FONT = ("Courier", 12)
NORM_FONT = ("Courier", 10)
SMALL_FONT = ("Courier", 0)
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SATE_OPTIONS = ['Landsat 8', 'Landsat 7', 'Landsat 5', 'Landsat 4', 'Sentinel-2']

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[-43:, :] = np.array([256/256, 0/256, 0/256, 1])
newcmp1 = ListedColormap(newcolors)
newcmp1.set_over(color='cyan')
newcmp1.set_bad(color='k')

viridis = cm.get_cmap('coolwarm', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[-43:, :] = np.array([200/256, 0/256, 0/256, 1])
newcmp2 = ListedColormap(newcolors)
newcmp2.set_over(color='black')



global leftbound
leftbound = 0.85
global fig, ax1, ax2, ax3, ax4, reset # ax5, ax6
global cbar_ax2, cbar_ax4, cbar_ax1, cbar_ax7 # cbar_ax3
global klicker, masked
global mask_img
mask_img = None
fig = Figure(figsize=(6.4, 4), dpi=200)
ax1 = fig.add_subplot(2, 4,  1)
ax2 = fig.add_subplot(2, 4,  2)
ax3 = fig.add_subplot(2, 4,  5)
ax4 = fig.add_subplot(2, 4,  6)
# ax5 = fig.add_subplot(3, 4,  9, projection=ccrs.PlateCarree())
# ax6 = fig.add_subplot(3, 4, 10, projection=ccrs.PlateCarree())
ax7 = fig.add_subplot(1, 2, 2)

cbar_ax2 = fig.add_axes([0, 0, 0.1, 0.1])
# cbar_ax3 = fig.add_axes([0, 0, 0.1, 0.1])
cbar_ax4 = fig.add_axes([0, 0, 0.1, 0.1])
cbar_ax1 = fig.add_axes([0, 0, 0.1, 0.1])
# cbar_ax6 = fig.add_axes([0, 0, 0.1, 0.1])
cbar_ax7 = fig.add_axes([0, 0, 0.1, 0.1])

fig.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.87, hspace=0.30)
reset = False
masked = False

global img_idx, current_sate, center_lat, center_lon
global img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, imgmask, u10m, v10m
img_id_list, date_list, img_date_list = [], [], []
imgchannels, imgxch4, imglons, imglats, imgmask, u10m, v10m = None, None, None, None, None, None, None
img_idx = 0
global brightness_scaler
brightness_scaler = {
    'Landsat 8': 0.4,
    'Landsat 7': 0.4,
    'Landsat 5': 0.4,
    'Landsat 4': 0.4,
    'Sentinel-2': 0.3,
}

class Application(tk.Tk):
    """
    The class for the GUI, configuration, buttons, menus, etc..
    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Methane Labeller ver. 1.0')
        self.geometry('1520x800')
        self.satenow = tk.StringVar(self)
        self.latnow = tk.DoubleVar(self)
        self.lonnow = tk.DoubleVar(self)
        self.yynow = tk.IntVar(self)
        self.mmnow = tk.IntVar(self)
        self.dxdy = tk.DoubleVar(self)
        
        # Disablize the resizability of the GUI.
        self.resizable(0, 0)        

        self.imgindex = 0
        
        # create widget
        self.create_wigets()

    def create_wigets(self):
        # text field for printings
        self.print_text = tk.Text(self, width=29, height=29, state='disabled')
        self.print_text.place(relx=leftbound, rely=0.5)
        
        self.menubar = tk.Menu(self, background='#ff8000', foreground='black', activebackground='white', activeforeground='black')  
        self.file = tk.Menu(self.menubar, tearoff=1, background='#ffcc99', foreground='black')  
        self.file.add_command(label="Exit", command=self.quit)  
        self.menubar.add_cascade(label="File", menu=self.file)  
    
        # Create Entry widgets to accept User Input
        self.sate_option = tk.OptionMenu(self, self.satenow, *SATE_OPTIONS, command=self.option_changed)
        self.sate_option.place(relx=0.92, rely=0.05)
        self.sate_option.config(width=10)
        self.sate_label = tk.Label(self, foreground='red', text='Satellite: ')
        self.sate_label.place(relx=leftbound, rely=0.05)
        
        self.lon_entry = tk.Entry(self, width = 12)
        self.lon_entry.place(relx=0.92, rely=0.1)
        self.lon_label = tk.Label(self, foreground='red', text='Longitude: ')
        self.lon_label.place(relx=leftbound, rely=0.1)
        
        self.lat_entry = tk.Entry(self, width = 12)
        self.lat_entry.place(relx=0.92, rely=0.15)
        self.lat_label = tk.Label(self, foreground='red', text='Latitude: ')
        self.lat_label.place(relx=leftbound, rely=0.15)
        
        self.yy_entry = tk.Entry(self, width = 12)
        self.yy_entry.place(relx=0.92, rely=0.20)
        self.yy_label = tk.Label(self, foreground='red', text='Year: ')
        self.yy_label.place(relx=leftbound, rely=0.20)
        
        self.mm_entry = tk.Entry(self, width = 12)
        self.mm_entry.place(relx=0.92, rely=0.25)
        self.mm_label = tk.Label(self, foreground='red', text='Month: ')
        self.mm_label.place(relx=leftbound, rely=0.25)
        
        self.dxdy_entry = tk.Entry(self, width = 12)
        self.dxdy_entry.place(relx=0.92, rely=0.3)
        self.dxdy_entry.insert(0, "1.5")
        self.dxdy_label = tk.Label(self, foreground='red', text='dX/dY (km): ')
        self.dxdy_label.place(relx=leftbound, rely=0.3)

        # Buttons
        self.search_button = tk.Button(self, text='Search', command=lambda:self.do_search())
        self.search_button.place(relx=leftbound, rely=0.35)
        
        self.left_button = tk.Button(self, text='<<', command=lambda:move_img(-1))
        self.left_button.place(relx=0.92, rely=0.35)
        self.right_button = tk.Button(self, text='>>', command=lambda:move_img(1))
        self.right_button.place(relx=0.95, rely=0.35)
        self.save_button = tk.Button(self, text='Save', command=lambda:save_output())
        self.save_button.place(relx=0.95, rely=0.40)
        self.fill_button = tk.Button(self, text='Clear mask', command=lambda:clear_mask())
        self.fill_button.place(relx=leftbound, rely=0.40)
        
        # action status label
        self.action_label = tk.Label(self, foreground='red')
        self.action_label.place(relx=leftbound, rely=0.45)
        
        # Figure frame
        self.figure_frame = tk.Frame(self, width = 60, height=50)
        self.figure_frame.place(relx=0.0, rely=0.0)
        self.canvas = FigureCanvasTkAgg(fig, self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # toolbar = NavigationToolbar2TkAgg( canvas, self )
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.config(menu=self.menubar)
        
    def option_changed(self, *args):
        self.action_label['text'] = f'You selected: {self.satenow.get()}'
        
    def post_print(self, msg):
        self.print_text.configure(state='normal')
        self.print_text.insert(END, msg + '\n')
        self.print_text.configure(state='disabled')
        self.print_text.see("end")
        
    def check_config(self):
        try:
            latnow = float(self.lat_entry.get())
            lonnow = float(self.lon_entry.get())
            satenow = self.satenow.get()
            yynow = int(self.yy_entry.get())
            mmnow = int(self.mm_entry.get())

            if latnow > 85 or latnow < -85 or lonnow < -180 or lonnow > 360 or mmnow not in MONTHS or satenow not in satellite_database.keys():
                print(satellite_database.keys())
                self.post_print('Something wrong with the information entered: \n')
                self.post_print('>  ==> Satellite: %s' % satenow)
                self.post_print('>  ==> Longitude: %.2f' % lonnow)
                self.post_print('>  ==> Latitude: %.2f' % latnow)
                self.post_print('>  ==> Year: %04d' % yynow)
                self.post_print('>  ==> Month: %02d' % mmnow)
                return False
            else:
                return True
        except Exception as e:
            self.post_print('Something wrong with the information entered: \n' + str(e))
            return False
    
    def do_search(self):
        global reset, img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, center_lat, center_lon, current_sate, imgmask, img_idx, v10m, u10m
        
        success = self.check_config()
        if success is True: # 
            self.latnow = float(self.lat_entry.get())
            self.lonnow = float(self.lon_entry.get())
            self.yynow = int(self.yy_entry.get())
            self.mmnow = int(self.mm_entry.get())
            self.dxdy = float(self.dxdy_entry.get())

            self.post_print('>  ~~~~~~~~~~~~~~~~~~~~~')
            self.post_print('> Current configuration: ')
            self.post_print('>  ==> Satellite: %s' % self.satenow.get())
            self.post_print('>  ==> Longitude: %.3f' % self.lonnow)
            self.post_print('>  ==> Latitude : %.3f' % self.latnow)
            self.post_print('>  ==> Year:      %04d' % self.yynow)
            self.post_print('>  ==> Month:     %02d' % self.mmnow)
            self.post_print('>  ==> dXdY (km): %.1f' % self.dxdy)
            self.post_print('>  ~~~~~~~~~~~~~~~~~~~~~')
            current_sate = self.satenow.get()
            center_lon, center_lat = self.lonnow, self.latnow
            start_time = datetime(self.yynow, self.mmnow, 1, 0, 0)  #
            end_time = start_time + timedelta(days=40)
            end_time = datetime(end_time.year, end_time.month, 1, 0, 0)
            self.post_print('> Searching for: ' + start_time.strftime('%Y-%m-%d --> ') + end_time.strftime('%Y-%m-%d'))
            img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, u10m, v10m = get_plume(self, self.lonnow, self.latnow, start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'), dX=self.dxdy, dY=self.dxdy, do_retrieval=False, satellite=current_sate)  # 5.905613686710645, 31.65857047520231  # 31.6585, 5.9053  # self.satenow.get()
            img_idx = 0

            # start_time = datetime(2020, 3, 1, 0, 0)
            # end_time = datetime(2020, 4, 1, 0, 0)
            # current_sate = 'Landsat 8'
            # center_lon = 5.905613686710645
            # center_lat = 31.65857047520231
            # img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats = get_plume(self, 5.905613686710645, 31.65857047520231, \
            #                             start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'), dX=0.01, dY=0.01, do_retrieval=False, satellite='Landsat 8') 

            imgmask = np.zeros((imgchannels.shape[0], imgchannels.shape[1], imgchannels.shape[2]))
            
            reset = True
    
def animate(i):
    global reset, img_id_list, date_list, img_date_list, imgchannels, imgxch4, imglons, imglats, center_lon, center_lat, u10m, v10m
    global img_idx, masked, mask_img
    
    if reset and imgchannels is not None:
        app.action_label['text'] = 'Updating images...'
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax7.clear()
        

        rgb = np.stack([imgchannels[img_idx, :, :, 1], imgchannels[img_idx, :, :, 2], imgchannels[img_idx, :, :, 3]], axis=-1)/brightness_scaler[current_sate]
        ndvi = (imgchannels[img_idx, :, :, 4] - imgchannels[img_idx, :, :, 1])/(imgchannels[img_idx, :, :, 4] + imgchannels[img_idx, :, :, 1])
        ndmi = (imgchannels[img_idx, :, :, 5] - imgchannels[img_idx, :, :, 6])/(imgchannels[img_idx, :, :, 5] + imgchannels[img_idx, :, :, 6])
        datestr = img_date_list[img_idx].strftime('%Y-%m-%d %H:%M:%S')
        xx = np.arange(-0.5, imgchannels[img_idx, :, :, 0].shape[1]+0.5, 1)
        yy = np.arange(-0.5, imgchannels[img_idx, :, :, 0].shape[0]+0.5, 1)

        fig.suptitle('%s: [Lon=%.2f, Lat=%.2f], %s ( %d / %d )'%(current_sate, center_lon, center_lat, datestr, img_idx+1, imgchannels.shape[0]), y=0.98, fontsize=10)
        print(imgchannels.shape, rgb.shape, ndmi.shape)
        print(imglons[img_idx, 0, 0], imglons[img_idx, -1, -1])
        print(imglats[img_idx, 0, 0], imglats[img_idx, -1, -1])

        coolwarm = cm.get_cmap('coolwarm', 600)
        newcolors = coolwarm(np.linspace(0, 1, 600))

        im = ax1.pcolormesh(xx, yy, zstandard(ndvi), cmap=newcmp2, vmin=-3, vmax=3)
        ax1.set_xlim((-0.5, np.max(xx)))
        ax1.set_ylim((-0.5, np.max(yy)))
        ax1.set_xticks( [0, np.max(xx)-0.5])
        ax1.set_yticks( [0, np.max(yy)-0.5])
        ax1.invert_yaxis()
        ax1.xaxis.tick_top()
        ax1.set_yticklabels(ax1.get_yticks().astype(int), rotation = 90)
        posn = ax1.get_position()
        # ax1.set_extent([imglons[img_idx, 0, 0], imglons[img_idx, -1, -1], imglats[img_idx, 0, 0], imglats[img_idx, -1, -1]], crs=ccrs.PlateCarree())
        cbar_ax1.set_position([posn.x0, posn.y0 - 0.025,
                          posn.width, 0.01])
        plt.colorbar(im, cax=cbar_ax1, orientation='horizontal', extend='both')
        ax1.set_title('NDVI', y=1)
        

        im = ax2.pcolormesh(xx, yy, zstandard(ndmi), cmap=newcmp2, vmin=-3, vmax=3)
        ax2.set_xlim((-0.5, np.max(xx)))
        ax2.set_ylim((-0.5, np.max(yy)))
        ax2.set_xticks( [0, np.max(xx)-0.5])
        ax2.set_yticks( [0, np.max(yy)-0.5])
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()
        ax2.set_yticklabels(ax2.get_yticks().astype(int), rotation = 90)
        posn = ax2.get_position()
        cbar_ax2.set_position([posn.x0, posn.y0 - 0.025,
                          posn.width, 0.01])
        plt.colorbar(im, cax=cbar_ax2, orientation='horizontal', extend='both')
        ax2.set_title('NDMI', y=1)


        im = ax3.imshow(rgb)
                #transform=ccrs.PlateCarree(), extent=[imglons[img_idx, 0, 0], imglats[img_idx, 0, 0], imglons[img_idx, -1, -1], imglats[img_idx, -1, -1]])
        ax3.set_title('RGB True Color')
        ax3.set_xticks( [0, np.max(xx)-0.5])
        ax3.set_yticks( [0, np.max(yy)-0.5])
        
        im = ax4.pcolormesh(xx, yy, imgchannels[img_idx, :, :, 7], cmap=cm.binary_r, vmin=0, vmax=80)
        ax4.set_xlim((-0.5, np.max(xx)))
        ax4.set_ylim((-0.5, np.max(yy)))
        ax4.set_xticks( [0, np.max(xx)-0.5])
        ax4.set_yticks( [0, np.max(yy)-0.5])
        ax4.invert_yaxis()
        ax4.xaxis.tick_top()
        ax4.set_yticklabels(ax4.get_yticks().astype(int), rotation = 90)
        posn = ax4.get_position()
        cbar_ax4.set_position([posn.x0, posn.y0 - 0.02,
                          posn.width, 0.01])
        plt.colorbar(im, cax=cbar_ax4, orientation='horizontal', extend='max')
        ax4.set_title('Cloud Score', y=1)
        
        # im = ax3.pcolormesh(imglons[img_idx], imglats[img_idx], zscore(imgchannels[img_idx, :, :, 0]), cmap=cm.viridis, transform=ccrs.PlateCarree(), vmin=-2, vmax=2)
        # posn = ax3.get_position()
        # cbar_ax3.set_position([posn.x0 + posn.width + 0.01, posn.y0,
        #                   0.02, posn.height])
        # plt.colorbar(im, cax=cbar_ax3, orientation='vertical')
        # ax3.set_title('dR Z-score')
        # mask_img = ax4.pcolormesh(imglons[img_idx], imglats[img_idx], imgmask[img_idx], cmap=cm.binary, vmin=0, vmax=1, transform=ccrs.PlateCarree())
        # posn = ax4.get_position()
        # cbar_ax4.set_position([posn.x0 + posn.width + 0.01, posn.y0,
        #                   0.02, posn.height])
        # plt.colorbar(mask_img, cax=cbar_ax4, orientation='vertical')
        # ax4.set_title('Result of masking')
        
        masktemp = imgmask[img_idx].copy()
        masktemp[masktemp > 0] = np.nan
        im = ax7.pcolormesh(xx, yy, zstandard(imgchannels[img_idx, :, :, 0])+masktemp, cmap=newcmp1, vmin=-3, vmax=3)
        ax7.set_xlim((-0.5, np.max(xx)))
        ax7.set_ylim((-0.5, np.max(yy)))
        ax7.set_xticks([0, np.max(xx)-0.5])
        ax7.set_yticks([0, np.max(yy)-0.5])
        ax7.invert_yaxis()
        ax7.set_yticklabels(ax7.get_yticks().astype(int), rotation=90)
        posn = ax7.get_position()
        cbar_ax7.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                          0.02, posn.height])
        plt.colorbar(im, cax=cbar_ax7, orientation='vertical', extend='both')
        
        axins = ax7.inset_axes([1.01, 1.01, 0.18, 0.17])
        axins.set_xlim((0.95, 1.05))
        axins.set_ylim((0.95, 1.05))
        axins.set_xticks([])
        axins.set_yticks([])
        try:
            axins.barbs(1, 1, u10m[img_idx], v10m[img_idx], length=3, pivot='middle')
            windsp = np.sqrt(u10m[img_idx]**2 + v10m[img_idx]**2)
            axins.quiver(1, 1, u10m[img_idx]/windsp, v10m[img_idx]/windsp, scale=15, scale_units='xy', pivot='middle', width=0.02)
        except:
            axins.text(0.96, 1, 'Wind NA')
        ax7.set_title('dR Z-score: LMB/RMB drag to mask/demask', y=1)
        # ax7.text(0.5, 1.1, 'dR Z-score: LMB/RMB drag to mask/demask', transform=ax7.transAxes)

        app.action_label['text'] = 'Image update done: %s'%(current_sate)
        reset = False
        

def find_2d_ind(arr, value):
    diff = np.abs(arr - value)
    return np.unravel_index(diff.argmin(), diff.shape)


def clear_mask():
    global img_idx, imgmask, reset
    imgmask[img_idx, :, :] = 0
    reset = True
    
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.quit()
    
def move_img(change):
    global reset, img_idx, imgchannels
    
    img_idx = int((img_idx + int(change) )%len(imgchannels))
    reset = True
      

# Mouse event + Lasso selector control
#------------------------------------------------
def updateArray(array, indices, direction=1):
    lin = np.arange(array.size)
    newArray = array.flatten()
    if direction == 1:
        newArray[lin[indices]] = 1
    else:
        newArray[lin[indices]] = 0
    return newArray.reshape(array.shape)

def onselect(verts):
    global imgmask, img_idx, mask_img, reset

    xv, yv = np.meshgrid(np.arange(imgmask[img_idx].shape[1]), np.arange(imgmask[img_idx].shape[0]))
    pix = np.vstack( (xv.flatten(), yv.flatten()) ).T

    p = path.Path(verts)
    ind = p.contains_points(pix, radius=0.3)
    imgmask[img_idx] = updateArray(imgmask[img_idx], ind, direction=1)
    


def onselect_delete(verts):
    global imgmask, img_idx, mask_img, reset

    xv, yv = np.meshgrid(np.arange(imgmask[img_idx].shape[1]), np.arange(imgmask[img_idx].shape[0]))
    pix = np.vstack( (xv.flatten(), yv.flatten()) ).T

    p = path.Path(verts)
    ind = p.contains_points(pix, radius=0.6)
    imgmask[img_idx] = updateArray(imgmask[img_idx], ind, direction=-1)
    # mask_img.set_data(imgmask[img_idx])
    # fig.canvas.draw_idle()
    

def onrelease(event):
    global reset
    print('Released')
    reset = True
    
#------------------------------------------------

def get_filename():
    return '%s_%.3fEx%.3fN_%s_imgMonID%d'%(satellite_database[current_sate]['Shortname'], center_lon, center_lat, img_date_list[img_idx].strftime('%Y%m%d_%H%M%S'), img_idx)

def save_output():
    global app, fig, img_idx
    global img_id_list, img_date_list, imgchannels, imglons, imglats, imgmask, u10m, v10m

    outdir = 'labelled_plumes/'
    subdir = "%.3f_%.3f/"%(center_lon, center_lat)
    if not os.path.exists(outdir + subdir + get_filename()):
        os.makedirs(outdir + subdir + get_filename())

    outname = outdir + subdir + get_filename() + '/' + get_filename()
    app.post_print('> Saving results to --> '+ outname)
    print('> Saving results to --> '+ outname)
    np.savez(outname + '.npz', channels=imgchannels[img_idx], 
                               img_id_list=img_id_list[img_idx],
                               img_date_list=img_date_list[img_idx],
                               longrid=imglons[img_idx], 
                               latgrid=imglats[img_idx], 
                               mask=imgmask[img_idx],
                               u10m=u10m[img_idx],
                               v10m=v10m[img_idx])
    fig.savefig(outname + '.png', dpi=300)
    app.post_print('> Finished saving ( %d / %d ).'%(img_idx+1, len(img_date_list)))
    print('> Finished saving ( %d / %d ).'%(img_idx+1, len(img_date_list)))

fig.canvas.mpl_connect('button_release_event', onrelease)
    
        
# Initialize the GUI
global app
app = Application()


ani = animation.FuncAnimation(fig, animate, interval=1000)

lasso_add = LassoSelector(ax7, onselect, lineprops={'color': 'blue', 'linewidth': 0.5}, button=1)
lasso_delete = LassoSelector(ax7, onselect_delete, lineprops={'color': 'white', 'linewidth': 0.5}, button=3)

app.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI.
app.mainloop()
