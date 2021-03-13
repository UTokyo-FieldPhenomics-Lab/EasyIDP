import os
import re
import json
import time
import pyproj
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
from easyric.external import shapefile
from shapely.geometry import Polygon, MultiPolygon

class TiffSpliter:
    """
    The coordinates of this program as follows, starts at lower left
    
    ##########################
    #              x         #
    # North O------------->  #
    #       |  ________      #
    #       | |        |     #
    #       | |  img   |     #
    #       | |________|     #
    #       v                #
    ##########################
    
    x: horizontal,  width, longitude
    y:   vertical, height,  latitude
    
    the default order of this package is (x, y) order
    
    ###############################################################
    but in numpy matrix and tiff tile system, they use (y, x) order
    ###############################################################
    """
    
    def __init__(self, tif_path, grid_w, grid_h, extend=False, grid_buffer=0):
        """
        Parameters
        ----------
        tif_path: str
            the path string of target splitting geotiff file
        grid_w: int
            the width (pixel) of grid
        grid_h: int
            the height (pixel) of grid
        extend: bool
            if the last grid doesn't fit (grid_w, grid_h), extend the grid to the given size
                e.g. grid_w = 1000, grid_h=800, however, the last grid only 400x300
                    True:  the last grid with display on a 1000x800 image
                    False: the last grid will save to 400x300 image
        """
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_buffer = grid_buffer
        self.tif_path = tif_path
        self.extend = extend
        self.save_folder = '.'
        
        # open tif files
        with tf.TiffFile(tif_path) as tif:
            page = tif.pages[0]
        
            self.img_h, self.img_w = page.imagelength, page.imagewidth
            self.img_depth = page.imagedepth
            if self.img_depth > 1:
                raise ValueError('Current version only support images with one depth')
            self.img_band_num = page.samplesperpixel
            self.nodata = page.nodata
            self.dtype = page.dtype

            # 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 329982.13726166, 5080412.59175851, 0)
            self.geo_xmin = page.tags[33922].value[3]
            self.geo_ymax = page.tags[33922].value[4]

            # 33550 model_pixel_scale (3d) (0.0600000000000001, 0.06000000000000508, 0.0)
            self.scale_x = page.tags[33550].value[0]
            self.scale_y = page.tags[33550].value[1]
            
            # extract geotiff projection coordinate
            if 'PCSCitationGeoKey' in page.geotiff_tags.keys():
                self.proj = pyproj.CRS.from_string(page.geotiff_tags['PCSCitationGeoKey'])
            elif 'GTCitationGeoKey' in page.geotiff_tags.keys():
                self.proj = pyproj.CRS.from_string(page.geotiff_tags['GTCitationGeoKey'])
            elif 'ProjectedCSTypeGeoKey' in page.geotiff_tags.keys():
                self.proj = pyproj.CRS.from_epsg(page.geotiff_tags['ProjectedCSTypeGeoKey'].value)
            else:
                raise ValueError(f'Could not read related geotiff projection info\n{page.geotiff_tags}')
        
        self.make_grid()
        
    def make_grid(self):
        """
        Split the images into several rectangle grids by given grid width and height

        Parameters
        ----------
        self.img_w, self.img_h, self.grid_h, self.grid_w
            e.g. image width=30, height=31, split to grids width=7, height=8
        
        Returns
        -------
        self.wgrid_st: a numpy array of each grid horizontal (width) start pixel
            e.g. [ 0,  7, 14, 21, 28]
        self.hgrid_st: a numpy array of each grid vertical (height) start pixel
            e.g. [ 0,  8, 16, 24]
        self.wgrid_len: a numpy array of each grid horizontal (width) length
            e.g. [ 7,  7,  7,  7,  2]
        self.hgrid_len: a numpy array of each grid vertical (height) length
            e.g. [ 8,  8,  8,  7]
        """
        self.wgrid_st = np.arange(start=0, stop=self.img_w, step=self.grid_w)
        self.hgrid_st = np.arange(start=0, stop=self.img_h, step=self.grid_h)

        self.wgrid_len = np.ones(len(self.wgrid_st), dtype=np.uint) * self.grid_w
        self.hgrid_len = np.ones(len(self.hgrid_st), dtype=np.uint) * self.grid_h

        # the last grid is not full grid, change the length
        if self.img_w % self.grid_w != 0:
            self.wgrid_len[-1] = self.img_w - self.wgrid_st[-1] - 1
        if self.img_h % self.grid_h != 0:
            self.hgrid_len[-1] = self.img_h - self.hgrid_st[-1] - 1
        
    
    def search_grid(self, pixel_w, pixel_h):
        """
        Return the index of grid, for self.hgrid_st, wgrid_st, hgrid_len, wgrid_len use
        
        Parameters
        ----------
        pixel_h: int
            the pixel height (vertical)
        pixel_w: int
            the pixel width (horizontal)
        
        Returns
        -------
        grid_id: tuple (w_id, h_id)
            e.g.
                  0       1       2       3
            0 | (0,0) | (1,0) | (2,0) | (3,0) |
            1 | (0,1) |       |       |       |
            2 | (0,2) |       |       |       |
        """
        h_order = np.floor(pixel_h / self.grid_h).astype(int)
        w_order = np.floor(pixel_w / self.grid_w).astype(int)
        
        return (w_order, h_order)
    

    def id2name(self, w_id, h_id, format='tif'):
        digit_w = len(str(len(self.wgrid_st)))
        digit_h = len(str(len(self.hgrid_st)))
        # >>> '{0:04}'.format(1)
        # 0001
        # >>> f"{{0:0{3}}}"
        # '{0:03}'
        w_id = f"{{0:0{digit_w}}}".format(w_id)
        h_id = f"{{0:0{digit_h}}}".format(h_id)
        return f'grid_x{w_id}_y{h_id}.{format}'
    
    @staticmethod
    def name2id(grid_name, format='tif'):
        _, w_id, h_id, _ = re.split(f'grid_x|_y|.{format}', grid_name)
        return int(w_id), int(h_id)
    
    def pixel2geo(self, points_hv):
        """
        convert geotiff pixel coordinate (horizontal, vertical) 
        to 
        geology coordinate (horizontal_longtitue, vertical_latitude)
        
        Parameters
        ----------
        points_hv: numpy nx2 array
            [horizontal, vertical] pixel points -> (width, height)

        Returns
        -------
        geo_hv: The ndarray geo coordiantes of these points (horizontal, vertical)
        """
        pix_ph = points_hv[:, 0]
        pix_pv = points_hv[:, 1]

        geo_px = self.geo_xmin + pix_ph * self.scale_x
        geo_py = self.geo_ymax - pix_pv * self.scale_y

        # geo_px = [x1, x2, x3, x4]
        # geo_py = [y1, y2, y3, y4]
        # merge to 
        # [[x1, y1], 
        #  [x2, y2], 
        #  [x3, y3],
        #  [x4, y4]]
        geo_hv = np.concatenate([geo_px[:, None], geo_py[:, None]], axis=1)
        
        return geo_hv
    
    def geo2pixel(self, geo_hv):
        """
        convert geo coordinate (horizontal, vertical)
        to 
        geotiff pixel coordinate (horizontal, vertical) 
        
        Parameters
        ----------
        geo_hv: numpy nx2 array
            [horizontal, vertical] geo points

        Returns
        -------
        pixel_hv: The ndarray pixel coordiantes of these points (horizontal, vertical)
        """

        geo_ph = geo_hv[:, 0]
        geo_pv = geo_hv[:, 1]

        pixel_h = (geo_ph - self.geo_xmin) // self.scale_x
        pixel_v = (self.geo_ymax - geo_pv) // self.scale_y

        # pixel_h = [x1, x2, x3, x4]
        # pixel_v = [y1, y2, y3, y4]
        # merge to 
        # [[x1, y1], 
        #  [x2, y2], 
        #  [x3, y3],
        #  [x4, y4]]
        pixel_hv = np.concatenate([pixel_h[:, None], pixel_v[:, None]], axis=1)

        return pixel_hv.astype(int)
    
    def get_crop(self, page, i0, j0, h, w):
        """
        Extract a crop from a TIFF image file directory (IFD).

        Modified from: 
        https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60

        Only the tiles englobing the crop area are loaded and not the whole page.
        This is usefull for large Whole slide images that can't fit int RAM.
        
        Parameters
        ----------
        page : TiffPage
            TIFF image file directory (IFD) from which the crop must be extracted.
        i0, j0: int
            Coordinates of the top left corner of the desired crop.
            i0 = h_st, j0 = w_st
        h: int
            Desired crop height.
        w: int
            Desired crop width.
            
        Returns
        -------
        out : ndarray of shape (h, w, sampleperpixel)
            Extracted crop.""
        """
        if page.is_tiled:
            out = self._get_tiled_crop(page, i0, j0, h, w)
        else:
            out = self._get_untiled_crop(page, i0, j0, h, w)
            
        return out

    @staticmethod
    def _get_tiled_crop(page, i0, j0, h, w):
        """
        The submodule of self.get_crop() for those tiled geotiff
        """
        if not page.is_tiled:
            raise ValueError("Input page must be tiled")

        im_width = page.imagewidth
        im_height = page.imagelength

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")
            
        i1, j1 = i0 + h, j0 + w
        if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
            raise ValueError(f"Requested crop area is out of image bounds.{i0}_{i1}_{im_height}, {j0}_{j1}_{im_width}")

        tile_width, tile_height = page.tilewidth, page.tilelength

        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        # older version: (img_depth, h, w, dim)
        out = np.empty((page.imagedepth,
                        (tile_i1 - tile_i0) * tile_height,
                        (tile_j1 - tile_j0) * tile_width,
                        page.samplesperpixel), dtype=page.dtype)

        fh = page.parent.filehandle

        for i in range(tile_i0, tile_i1):
            for j in range(tile_j0, tile_j1):
                index = int(i * tile_per_line + j)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                fh.seek(offset)
                data = fh.read(bytecount)
                tile, indices, shape = page.decode(data, index)

                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        # old version: out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]
        return out[0, im_i0: im_i0 + h, im_j0: im_j0 + w, :]
    
    @staticmethod
    def _get_untiled_crop(page, i0, j0, h, w):
        """
        The submodule of self.get_crop(), for those untiled geotiff
        """
        if page.is_tiled:
            raise ValueError("Input page must not be tiled")

        im_width = page.imagewidth
        im_height = page.imagelength

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        i1, j1 = i0 + h, j0 + w
        if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
            raise ValueError(f"Requested crop area is out of image bounds.{i0}_{i1}_{im_height}, {j0}_{j1}_{im_width}")
        
        out = np.empty((page.imagedepth, h, w, page.samplesperpixel), dtype=page.dtype)
        fh = page.parent.filehandle

        for index in range(i0, i1):
            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)

            tile, indices, shape = page.decode(data, index)

            out[:,index-i0,:,:] = tile[:,:,j0:j1,:]

        return out[0,:,:,:]
    
    def _make_empty_container(self, h, w, layer_num=None):
        """
        Produce a empty image, suit the requirement for nodata
        """
        # possible dsm with only one band
        if self.img_band_num == 1:
            # old version: np.ones((self.img_depth, h, w, 1))
            empty_template = np.ones((h, w, 1)) * self.nodata
            
        # possible RGB band
        elif self.img_band_num == 3 and self.dtype==np.uint8:
            if layer_num == 4:
                # old version: np.ones((self.img_depth, h, w, 1))
                empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
                empty_template[:,:,3] = empty_template[:,:,3] * 0
            else:
                # old version: np.ones((self.img_depth, h, w, 1))
                empty_template = np.ones((h, w, 3)).astype(np.uint8) * 255
            
        # possible RGBA band, empty defined by alpha = 0
        elif self.img_band_num == 4 and self.dtype==np.uint8:
            # old version: np.ones((h, w, 1))
            empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
            empty_template[:,:,3] = empty_template[:,:,3] * 0
        else:
            raise ValueError('Current version only support DSM, RGB and RGBA images')
            
        return empty_template
    
    def is_empty_image(self, img_array):
        """
        Judge if current img_array is empty grids, e.g. dsm=-10000, rgb=[255,255,255], [0,0,0],
                                                        and RGBA with full empty alpha layer
        
        Parameters
        ----------
        img_array: np.ndarray
            the outputs of self.get_crop()
        
        Returns
        -------
        bool: True is empty image.
        
        """
        is_empty = False
        h, w, n = img_array.shape
        
        empty_template = self._make_empty_container(h, w)
        
        if self.img_band_num == 1:
            if np.array_equal(img_array, empty_template):
                is_empty = True
        if self.img_band_num == 3:
            # for the case that use (255, 255, 255) white as background
            if np.array_equal(img_array, empty_template):
                is_empty = True
            # in case some use (0, 0, 0) black as background
            if np.array_equal(img_array, np.zeros((h, w, n))):
                is_empty = True
        # for those RGBA with alpha layers, assume alpha=0 as empty
        if self.img_band_num == 4:
            if np.array_equal(img_array[:,:,3], empty_template[:,:,3]):
                is_empty = True

        return is_empty
        
    def save_one_grid(self, save_path, w_id, h_id, page=None, extend=False, skip_empty=True):
        """
        Clip one grid to 4 layer geotiff file
        
        Parameters
        ----------
        save_path: str
            the path/image_name.tiff
        w_id: int
            the id of self.wgrid_*
        h_id: int
            the id of self.hgrid_*
        page : TiffPage
            TIFF image file directory (IFD) from tifffile.TiffFile.pages[0]
        extend: bool
            True: when cropped image is smaller than grid size, fill with empty data
        skip_empty: bool
            True: when cropped part is empty, ignore save processing
            
        Returns
        -------
        bool, True: succefully write, False: skipeed
        """
        # in common case, need call this function in a for loop, so the geotiff file keeps
        #     open in order to avoid too frequently open and close
        # however, this function should also be called individually, so need open geotiff
        #     at the begining of this function, and close at the end of this function
        # that is why specify the page_given value
        if page is None:
            page_given = False
        else:
            page_given = True
        
        if not page_given:
            tif = tf.TiffFile(self.tif_path)
            page = tif.pages[0]
        
        w_st = self.wgrid_st[w_id]
        h_st = self.hgrid_st[h_id]
        w = self.wgrid_len[w_id] + self.grid_buffer
        h = self.hgrid_len[h_id] + self.grid_buffer

        if w_st + w > self.img_w:
            w = self.img_w - w_st - 1
        if h_st + h > self.img_h:
            h = self.img_h - h_st - 1

        img_clip = self.get_crop(page, i0=h_st, j0=w_st, h=h, w=w)
        _, _, n = img_clip.shape
        
        if skip_empty and self.is_empty_image(img_clip):
            return False
        
        if w < (self.grid_w + self.grid_buffer) or h < (self.grid_h + self.grid_buffer):
            need_extend = True
        else:
            need_extend = False
        
        if extend and need_extend:
            if self.img_band_num == 3 or self.img_band_num == 4:
                img_new = self._make_empty_container(self.grid_h + self.grid_buffer, 
                                                     self.grid_w + self.grid_buffer, 
                                                     4)
                img_new[0:h, 0:w, 3] = 255
                img_new[0:h, 0:w, 0:n] = img_clip
            else:
                img_new = self._make_empty_container(self.grid_h + self.grid_buffer, 
                                                     self.grid_w + self.grid_buffer)
                img_new[0:h, 0:w,:] = img_clip
        else:
            img_new = img_clip

        format = save_path.split('.')[-1]
            
        if format == 'tif':
            # prepare geotiff tags
            ## pixel2geo need points_hv
            geo_corner = self.pixel2geo(np.asarray([[w_st, h_st]]))
            geo_x = geo_corner[0, 0]
            geo_y = geo_corner[0, 1]

            container = []
            for k in page.tags.keys():
                if k < 30000:
                    continue

                t = page.tags[k]
                if t.dtype[0] == '1':
                    dtype = t.dtype[-1]
                else:
                    dtype = t.dtype

                if k == 33922:
                    value = (0, 0, 0, geo_x, geo_y, 0)
                else:
                    value = t.value

                container.append((t.code, dtype, t.count, value, True))

            # write to file
            with tf.TiffWriter(save_path) as wtif:
                wtif.save(data=img_new, software='sigmameow', 
                        photometric=page.photometric, 
                        planarconfig=page.planarconfig, 
                        compress=page.compression, 
                        resolution=page.tags[33550].value[0:2], extratags=container)

        elif format == "png":
            plt.imsave(save_path, img_new)

        elif format == "jpg":
            # jpg does not support transparent layer
            plt.imsave(save_path, img_new[:,:,0:3])
        else:
            raise TypeError("Only 'tif', 'png', 'jpg' are supported")
            
        if not page_given:
            tif.close()
            
        return True
    
    def save_all_grids(self, save_folder, extend=False, skip_empty=True, format='tif'):
        """
        Save all grid to geotiff tile
        
        Parameters
        ----------
        save_folder: str
            the folder path that save those grid geotiff files
        ingore_empty: bool
            determine whether ignore those image with nodata (or white background)
        format: str
            the format of saved file, option: "geotiff", "jpg", "png"
        
        Returns
        -------
        ignored_grid_list: list, the empty grids are ignored. items are "x_y" grid id string
        """
        # update the save folder
        self.save_folder = save_folder
        # count time
        st = time.time()
        
        ignored_grid_list = []
        
        tif = tf.TiffFile(self.tif_path)
        page = tif.pages[0]
        
        # progress bar
        w_len = len(self.wgrid_len)
        h_len = len(self.hgrid_len)
        total = w_len * h_len
        pre_stage = 0   # percent
        current = 0
        for w_id, w_st in enumerate(self.wgrid_st):
            for h_id, h_st in enumerate(self.hgrid_st):
                img_name = self.id2name(w_id=w_id, h_id=h_id, format=format)  # e.g.: 'grid_x{w_id}_y{h_id}.tif'
                tiff_path = os.path.join(save_folder, img_name)
                status = self.save_one_grid(tiff_path, page=page, w_id=w_id, h_id=h_id, extend=extend, skip_empty=skip_empty)
                
                if not status:
                    ignored_grid_list.append(img_name)
                    
                # progress bar
                current += 1
                percent = np.floor(current / total * 100)
                if percent > pre_stage:
                    time_used = round(time.time() - st)
                    time_left = round(time_used / percent * (100 - percent))
                    print(f"{img_name} | {percent} % done | {time_used} s passed, {time_left} s left    ", end='\r')
                    pre_stage = np.copy(percent)

        tif.close()
        print(f"\nCost {round(time.time() - st)} s in total")
        
        return ignored_grid_list
    
    @staticmethod
    def get_shp_fields(shp_path, encoding='utf-8'):
        """
        Read shp field data to pandas.DataFrame, for further json metadata usage
        
        Parameters
        ----------
        shp_path: str, the file path of *.shp
        encoding: str, default is 'utf-8', however, or some chinese characters, 'gbk' is required
        
        Returns
        -------
        field_df: pandas.DataFrame, all shp field records
        
        """
        shp = shapefile.Reader(shp_path, encoding=encoding)
        keys = {}
        for i, l in enumerate(shp.fields):
            if isinstance(l, list):
                keys[l[0]] = i

        field_df = pd.DataFrame(columns=keys.keys())
        
        for i, k in enumerate(shp.records()):
            field_df.loc[i] = list(k)
        
        return field_df
    
    @staticmethod
    def read_proj(prj_path):
        """
        read *.prj file to pyproj object
        
        Parameters
        ----------
        prj_path: str, the file path of shp *.prj
        
        Returns
        -------
        proj: the pyproj object
        """
        with open(prj_path, 'r') as f:
            wkt_string = f.readline()

        proj = pyproj.CRS.from_wkt(wkt_string)
        
        if proj.name == 'WGS 84':
            proj = pyproj.CRS.from_epsg(4326)

        return proj

    def read_shp(self, shp_path, name_field=None, shp_proj=None, encoding='utf-8'):
        """
        read shp file to python numpy object, and also provide the geo coordinate transfrom based on pyproj package
        
        Parameters
        ----------
        shp_path: str, the file path of *.shp
        name_field: str or int, the id or name of shp file fields as output dictionary keys
        shp_proj: pyproj object, default will read automatically from prj file, or given by
                  >>> read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)
        encoding: str, default is 'utf-8', however, or some chinese characters, 'gbk' is required
        
        Returns
        -------
        shp_dict: dict, the dictionary with read numpy polygon coordinates
                  {'id1': np.array([[x1,y1],[x2,y2],...]),
                   'id2': np.array([[x1,y1],[x2,y2],...]),...}
        """
        shp = shapefile.Reader(shp_path, encoding=encoding)
        geotiff_proj = self.proj
        
        # read shp file fields
        shp_fields = {}
        for i, l in enumerate(shp.fields):
            if isinstance(l, list):
                # the fields 0 -> delection flags, and is a tuple type, ignore this tag
                # [('DeletionFlag', 'C', 1, 0),
                #  ['ID', 'C', 36, 0],
                #  ['MASSIFID', 'C', 19, 0],
                #  ['CROPTYPE', 'C', 36, 0],
                #  ['CROPDATE', 'D', 8, 0],
                #  ['CROPAREA', 'N', 13, 5],
                #  ['ATTID', 'C', 36, 0]]
                shp_fields[l[0]] = i - 1
        print(f'Shp fields: {shp_fields}')
        shp_dict = {}

        # try to find current projection
        if shp_proj is None:
            prj_path = shp_path[:-4] + '.prj'
            if os.path.exists(prj_path):
                shp_proj = self.read_proj(shp_path[:-4] + '.prj')
                print(f'[io][shp][proj] find ESRI projection file {prj_path}, and successfully obtain projection '
                      f'{shp_proj.coordinate_system}')
            else:
                print(f'[io][shp][proj] could not find ESRI projection file {prj_path}, could not operate auto-convention')

        for i, shape in enumerate(shp.shapes()):
            # find the fields id
            if name_field is None:
                field_id = -1
            elif isinstance(name_field, int):
                field_id = name_field
            elif isinstance(name_field, str):
                field_id = shp_fields[name_field]
            else:
                raise KeyError(f'Can not find key {name_field} in {shp_fields}')
            
            plot_name = shp.records()[i][field_id]
            if isinstance(plot_name, str):
                plot_name = plot_name.replace(r'/', '_')
                plot_name = plot_name.replace(r'\\', '_')
            else:
                plot_name = str(plot_name)

            coord_np = np.asarray(shape.points)

            # Old version: the pyshp package load seems get (lon, lat), however, the pyproj use (lat, lon), so need to revert
            # latest version:
            # when shp unit is (degrees) lat, lon, the order is reversed
            # however, if using epsg as unit (meter), the order doesn't need to be changed
            if coord_np.max() <= 180.0:
                coord_np = np.flip(coord_np, axis=1)

            if geotiff_proj is not None and shp_proj is not None and shp_proj.name != geotiff_proj.name:
                transformer = pyproj.Transformer.from_proj(shp_proj, geotiff_proj)
                transformed = transformer.transform(coord_np[:, 0], coord_np[:, 1])
                coord_np = np.asarray(transformed).T

                if True in np.isinf(coord_np):
                    raise ValueError(f'Fail to convert points from "{shp_proj.name}"(shp projection) to '
                                     f'"{geotiff_proj.name}"(dsm projection), '
                                     f'this may caused by the uncertainty of .prj file strings, please check the coordinate '
                                     f'manually via QGIS Layer Infomation, get the EPGS code, and specify the function argument'
                                     f'read_shp2d(..., given_proj=pyproj.CRS.from_epsg(xxxx))')

            shp_dict[plot_name] = coord_np

        return shp_dict
    
    
    def get_grid_polygon(self, h_id, w_id):
        """
        Make a grid to shapely.geometry.Polygon object for intersection use
        
        Parameters
        ----------
        h_id: the id of self.hgrid_*
        w_id: the id of self.wgrid_*
        
        Returns
        -------
        img_bound: shapely.geometry.Polygon object of specified grid
        
        """
        h0 = self.hgrid_st[h_id]
        w0 = self.wgrid_st[w_id]
        h  = h0 + self.hgrid_len[h_id]
        w  = w0 + self.wgrid_len[w_id]
        
        # .butter(0) to avoid later self-intersection error
        # > https://github.com/gboeing/osmnx/issues/278
        img_bound = Polygon([[w0,h0],[w,h0],[w,h],[w0,h],[w0,h0]]).buffer(0) 
        
        return img_bound
    
    
    def crop_polygon_on_grids(self, polygon_ndarray):
        """
        Clip one polygon by grids
        
        Parameters
        ----------
        polygon_ndarray: np.ndarray
           the polygon described by pixel coordinates
           in the format like [[x_pix1, y_pix1],  # horizontal, vertical order
                               [x_pix2, y_pix2], ...]
        
        Returns
        -------
        out_dict: dict, 
            contains the relationship between grid_name and clipped polygon
            e.g. out_dict = {'grid_x1_y1.tif': [poly1, poly2], 
                             'grid_x2_y1.tif': [poly1], ...}
                    where poly1 is the same structrue like previous polygon_ndarray
        
        """
        # convert geo coordinate to pixel coordinate
        polygon_pixel = self.geo2pixel(polygon_ndarray)
        given_polygon = Polygon(polygon_pixel).buffer(0) 
        
        # find the left-upper and right-lower corner of given polygon
        w_min, h_min = polygon_pixel.min(axis=0).tolist()
        w_max, h_max = polygon_pixel.max(axis=0).tolist()

        # search the left-upper and right-lower covered girds
        w_min_id, h_min_id = self.search_grid(pixel_w=w_min, pixel_h=h_min)
        w_max_id, h_max_id = self.search_grid(pixel_w=w_max, pixel_h=h_max)
        
        out_dict = {}
        for w_id in range(w_min_id, w_max_id+1):   # + 1 here is because range(0,3) only produce 0,1,2, not including 3
            for h_id in range(h_min_id, h_max_id+1):
                grid_polygon = self.get_grid_polygon(h_id=h_id, w_id=w_id)
                offset = np.asarray([self.wgrid_st[w_id], self.hgrid_st[h_id]])
                if grid_polygon.intersects(given_polygon):
                    img_name = self.id2name(w_id=w_id, h_id=h_id)   # f'grid_x{w_id}_y{h_id}.tif'
                    inter_polygon = grid_polygon.intersection(given_polygon)
                    # judge clippped to one polygon or more polygons
                    if isinstance(inter_polygon, Polygon):
                        inter_x, inter_y = inter_polygon.exterior.coords.xy
                        #print(f'Inter_x={inter_x}\nInter_y={inter_y}')
                        off_x = inter_x - self.wgrid_st[w_id]
                        off_y = inter_y - self.hgrid_st[h_id]
                        #print(f"Shifted_x={off_x}\nShifted_y={off_y}")
                        out_dict[img_name] = [np.concatenate([off_x[:, None], 
                                                               off_y[:, None]], axis=1)]
                    elif isinstance(inter_polygon, MultiPolygon):   # multipolygon
                        poly_list = []
                        for poly in inter_polygon:
                            inter_x, inter_y = poly.exterior.coords.xy
                            off_x = inter_x - self.wgrid_st[w_id]
                            off_y = inter_y - self.hgrid_st[h_id]
                            poly_list.append(np.concatenate([off_x[:, None], 
                                                             off_y[:, None]], axis=1))
                        out_dict[img_name] = poly_list
                    else:  # points others etc.
                        pass
                else:
                    continue
                
        return out_dict
    
    def exam_crop_polygon_result(self, out_dict, tiff_folder, save_name=False, show=False, dpi=300, size_rate=1):
        """
        Exam the accuracy of previous out_dict = self.crop_polygon_on_grids()
        not so necessary and current only support square girds
        
        Parameters
        ----------
        out_dict: dict
            the result of self.crop_polygon_on_grids()
        tiff_folder: str
            the folder path that contains the clipped gird tiff files
        save_name: str or False
            str: the imgpath and name of output file, e.g. 'D:/xxxx/xxxx.jpg'
            False: doesn't save file
        show: bool
            True: show in jupyter-notebook
            False：doesn't show figures
        """
        wid_list = []
        hid_list = []
        for name in out_dict.keys():
            wid, hid = self.name2id(name)
            wid_list.append(wid)
            hid_list.append(hid)

        w_min, w_max = min(wid_list), max(wid_list)
        h_min, h_max = min(hid_list), max(hid_list)

        w_num = w_max - w_min + 1
        h_num = h_max - h_min + 1

        fig, ax = plt.subplots(h_num, w_num, figsize=(w_num * size_rate, h_num * size_rate), sharex=True, sharey=True, dpi=dpi)

        for w in range(w_min, w_max+1):
            for h in range(h_min, h_max+1):
                ax_temp = ax[h-h_min, w-w_min]
                
                name = self.id2name(w_id=w, h_id=h)   # f"grid_x{w}_y{h}.tif"
                img_name = os.path.join(tiff_folder, name)
                
                with tf.TiffFile(img_name) as tif:
                    img_nd = tif.asarray()
                    
                ax_temp.imshow(img_nd)
                
                if name in out_dict.keys():
                    ax_temp.plot(*out_dict[name][0].T, 'r')

                    ax_temp.set_xlim(0, self.grid_w)
                    ax_temp.set_ylim(0, self.grid_h)
                    ax_temp.invert_yaxis()

                ax_temp.axes.xaxis.set_ticks([])
                ax_temp.axes.yaxis.set_ticks([])
                
        plt.subplots_adjust(left=0.01, top= 0.99, right = 0.99, bottom = 0.01, wspace=0, hspace=0)
        
        if save_name:
            plt.savefig(save_name)
        
        if show:
            plt.show()
    
        plt.clf()
        plt.close(fig)
        del fig, ax
    
    def shp_dict_clip_grids(self, shp_dict, save_exam=False):          
        """
        Clip all polygons by grids, and integrate them to a pandas.DataFrame table
        
        Parameters
        ----------
        shp_dict: dict
            the polygon read result of self.read_shp()
        save_exam: bool
            whether save exam images of clip_grids results by self.exam_crop_polygon_result()
            it will auto load the self.save_folder path defined in previous self.save_all_grids()
        
        Returns
        -------
        grid_df: pandas.DataFrame object
            a dataframe to record the result of clip, like the following table
            +----------+------------------+---------------------+
            | dict_key |     grid_name    |    polygon_list     |
            |----------|------------------|---------------------|
            | 'key1'   | 'grid_x1_y1.tif' | [poly1, poly2, ...] |
            | 'key1'   | 'grid_x2_y1.tif' | [poly1, poly2, ...] |
            | 'key1'   | 'grid_x1_y2.tif' | [poly1, poly2, ...] |
            |   ...    |      ...         |         ...         |
            | 'key2'   | 'grid_x1_y1.tif' | [poly1, poly2, ...] |
            | 'key2'   | 'grid_x3_y2.tif' | [poly1, poly2, ...] |
            +----------+------------------+---------------------+
        """
        grid_df = pd.DataFrame(columns=['dict_key', 'grid_name', 'polygon_list'])
        for shp_key, shp_poly in shp_dict.items():
            out_dict = self.crop_polygon_on_grids(shp_poly)
            
            if save_exam:
                exam_img_name = os.path.join(self.save_folder, f'{shp_key}.png')
                self.exam_crop_polygon_result(out_dict, self.save_folder, exam_img_name)
                
            for grid_name, grid_poly in out_dict.items():
                line = len(grid_df)
                grid_df.loc[line] = [shp_key, grid_name, grid_poly]
                
        return grid_df
    
    def dataframe_add_shp_tags(self, grid_df, field_df, align_field, tag_field):
        """
        Add label to polygon result of self.shp_dict_clip_grids(), 
           the label data comes from field_df = self.get_shp_fields():
        
           ===== grid_df = self.shp_dict_clip_grids() =====    === field_df ===
        +----------+------------------+---------------------+     +-------+
        | dict_key |     grid_name    |    polygon_list     |     |  ???  |
        |----------|------------------|---------------------|     |-------|
        | 'key1'   | 'grid_x1_y1.tif' | [poly1, poly2, ...] |     | field |
        | 'key1'   | 'grid_x2_y1.tif' | [poly1, poly2, ...] |  +  | field |
        | 'key1'   | 'grid_x1_y2.tif' | [poly1, poly2, ...] |     | field |
        |   ...    |      ...         |         ...         |     |  ...  | 
        | 'key2'   | 'grid_x1_y1.tif' | [poly1, poly2, ...] |     | crops |
        | 'key2'   | 'grid_x3_y2.tif' | [poly1, poly2, ...] |     | crops |
        +----------+------------------+---------------------+     +-------+
        
        Parameters
        ----------
        grid_df: pandas.DataFrame
            The shp file clipped by grids result, the output of self.shp_dict_clip_grids()
            with columns of [dict_key, grid_name, polygon_list]
        field_df: pandas.DataFrame
            the shp file field data, the output of self.get_shp_fields()
        align_field: str
            the column key of shp_field, for example:
                Shp fields: {'ID': 0, 'MASSIFID': 1, 'CROPTYPE': 2, 'CROPDATE': 3, ...}
            which is the same data of "dict_key",
                which produced by self.read_shp(..., name_field= "ID" | 0)
        tag_field: str
            the column key of shp_field, for exampe, previous CROPTYPE key
            
        Returns
        -------
        grid_tagged: pandas.DataFrame object
            The 4 column dataframe shows in this function introduction, sorted by "grid_name"
            +------------------+----------+---------------------+-------+
            |     grid_name    | dict_key |    polygon_list     |  tag  |
            |------------------|----------|---------------------|-------|
            | 'grid_x1_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
            | 'grid_x1_y1.tif' | 'key2'   | [poly1, poly2, ...] | crops |
            | 'grid_x1_y2.tif' | 'key1'   | [poly1, poly2, ...] | field |
            | 'grid_x2_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
            | 'grid_x3_y2.tif' | 'key2'   | [poly1, poly2, ...] | crops |
            |      ...         |   ...    |         ...         |  ...  | 
            +------------------+----------+---------------------+-------+
        """
        if align_field not in field_df.columns.values:
            raise KeyError(f'Can not find align fields of "{align_field}" in {field_df.columns.values.tolist()}')
        if tag_field not in field_df.columns.values:
            raise KeyError(f'Can not find tag fields of "{tag_field}" in {field_df.columns.values.tolist()}')
        
        merged = pd.merge(grid_df, field_df, left_on='dict_key', right_on=align_field)
        merged_select = merged[['grid_name','dict_key', 'polygon_list', tag_field]].sort_values(by='grid_name')
        
        grid_tagged = merged_select.rename(columns={tag_field: "tag"})
        
        return grid_tagged
    
    
    @staticmethod
    def dict2json(clip_dict, json_path, indent=0):
        """
        Convert dict to the same structure json file
        
        Parameters
        ----------
        clip_dict: dict
            the dict object want to save as json file
        json_path: str
            the path including json file name to save the json file
            e.g. "D:/xxx/xxxx/save.json"
        """
        if isinstance(json_path, str) and json_path[-5:] == '.json':
            with open(json_path, 'w', encoding='utf-8') as result_file:
                json.dump(clip_dict, result_file, ensure_ascii=False, cls=MyEncoder, indent=indent)

                # print(f'Save Json file -> {os.path.abspath(json_path)}')
        
    def save_jsons(self, grid_tagged, json_folder, minimize=True):
        """
        Save the tagged shp polygon clip result to json file, for deeplearing use
            The single json file has the following structure:
                {
                  "version": "4.5.6",  # the Labelme.exe version, optional
                  "flags": {},
                  "imagePath": "xxxx.tiff",
                  "imageHeight": 1000,
                  "imageWidth": 1000,
                  "imageData": null,
                  "shapes": [{ }, { }, { }]
                }             |    |    |
            for each {} items in shapes:
                {
                  "label": "field",
                  "group_id": null,
                  "shape_type": "polygon",
                  "flags": {},
                  "points": [[x1, y1], [x2, y2], [x3, y3]]  # with or without the first point
                }
        
        Parameters
        ----------
        grid_tagged: pandas.DataFrame
            the output of self.dataframe_add_shp_tags()
        json_folder: str
            the folder or path to save those json files
        minimize: bool
            True:  {"name":"lucy","sex":"boy"}
            False：
                   {
                      "name":"lucy",
                      "sex":"boy"
                   } 
        """
        total_dict = {}
        for i in range(len(grid_tagged)):
            img_name = grid_tagged.loc[i]['grid_name']
            poly = grid_tagged.loc[i]['polygon_list']
            tag = grid_tagged.loc[i]['tag']
            
            if img_name not in total_dict.keys():
                single_dict = {"version": "4.5.6", 
                               "flags": {},
                               "imagePath": img_name,
                               "imageHeight": 1000,
                               "imageWidth": 1000,
                               "imageData": None,
                               "shapes": []}
            else:
                single_dict = total_dict[img_name]
                
            for item in poly:
                single_item = {"label": tag,
                               "group_id": None,   # json null = python None
                               "shape_type": "polygon",
                               "flags": {},
                               "points": item.tolist()}
                single_dict['shapes'].append(single_item)
                
            total_dict[img_name] = single_dict
            
        # after iter all items
        for k, d in total_dict.items():
            json_name = k.replace('.tif', '.json')
            if minimize:
                self.dict2json(d, os.path.join(json_folder, json_name))
            else:
                self.dict2json(d, os.path.join(json_folder, json_name), indent=2)
                
                
class MyEncoder(json.JSONEncoder):
    """
    The original json package doesn't compatible to numpy object, add this compatible encoder to it.
    usage: json.dump(..., cls=MyEncoder)
    
    references: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)