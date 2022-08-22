import numpy as np
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import proplot as pplt
import matplotlib.pyplot as plt

def plotter(channels, date_obj, title, mask=None):
    # dR, rchannel, gchannel, bchannel, nirchannel, swir1channel, swir2channel, cloudscore
    rgb = np.stack([channels[:, :, 1], channels[:, :, 2], channels[:, :, 3]], axis=-1)
    ndvi = (channels[:, :, 4] - channels[:, :, 1])/(channels[:, :, 4] + channels[:, :, 1])
    ndmi = (channels[:, :, 5] - channels[:, :, 6])/(channels[:, :, 5] + channels[:, :, 6])
    date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    
    gs = pplt.GridSpec(nrows=2, ncols=3)
    fig = pplt.figure(refwidth=2)

    imgidx = 0
    ax = fig.subplot(gs[imgidx])
    im = ax.imshow(channels[:, :, 0], cmap=cm.bwr)
    ax.colorbar(im)
    ax.format(title=date_str + ' dR')
    imgidx += 1
    
    ax = fig.subplot(gs[imgidx])
    im = ax.imshow(rgb)
    ax.format(title=date_str + ' RGB')
    imgidx += 1
    
    ax = fig.subplot(gs[imgidx])
    im = ax.imshow(channels[:, :, 7], cmap=cm.binary_r, vmin=0, vmax=80)
    ax.colorbar(im)
    ax.format(title=date_str + ' Cloud score')
    imgidx += 1

    ax = fig.subplot(gs[imgidx])
    im = ax.imshow(ndvi, cmap=cm.bwr)
    ax.colorbar(im)
    ax.format(title=date_str + ' NDVI')
    imgidx += 1

    ax = fig.subplot(gs[imgidx])
    im = ax.imshow(ndmi, cmap=cm.bwr)
    ax.colorbar(im)
    ax.format(title=date_str + ' NDMI')
    imgidx += 1
    
    if mask is not None:
        ax = fig.subplot(gs[imgidx])
        im = ax.imshow(mask, cmap=cm.jet)
        ax.colorbar(im)
        ax.format(title=date_str + ' GMM-detected plume mask')
        imgidx += 1
        
    fig.format(suptitle=title)
    # plt.savefig('Algeria-%s.png'%(date_str))