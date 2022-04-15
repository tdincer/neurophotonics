import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Square:
    def __init__(self, h, w, name="", is_emitter=None):
        self.height = h
        self.width = w
        self.name = name
        self.is_emitter = is_emitter

        if np.all([h, w]):
            self.make_vertices()

    def make_vertices(self):
        self.tl = np.array([-self.width / 2, 0, self.height / 2])  # top left
        self.bl = np.array([-self.width / 2, 0, -self.height / 2])  # bottom left
        self.br = np.array([self.width / 2, 0, -self.height / 2])  # bottom right
        self.tr = np.array([self.width / 2, 0, self.height / 2])  # top right

        self.centroid = np.array([0.0, 0.0, 0.0])
        self.n = np.array([0.0, 1.0, 0.0])  # normal vector of the surface

    def rotate(self, seq="x", angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.centroid += r

    def vertices_positions(self, precision=13):
        xs = np.round([self.tl[0], self.bl[0], self.br[0], self.tr[0]], precision)
        ys = np.round([self.tl[1], self.bl[1], self.br[1], self.tr[1]], precision)
        zs = np.round([self.tl[2], self.bl[2], self.br[2], self.tr[2]], precision)
        return xs, ys, zs

    def plot_plane(self, precision=13, ax=None, show=False):
        if not ax:
            plt.figure("Box", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        xs, ys, zs = self.vertices_positions(precision=precision)
        # ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=0.25, facecolor="turquoise")
        srf.set_edgecolor("black")
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
        # ax.set_adjustable("datalim")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        delta = self.height / 2
        ax.set_xlim(self.centroid[0] - delta, self.centroid[0] + delta)
        ax.set_ylim(self.centroid[1] - delta, self.centroid[1] + delta)
        ax.set_zlim(self.centroid[2] - delta, self.centroid[2] + delta)

        if show:
            plt.show()


class Shank:
    def __init__(
        self, shank_dimensions,
        e_box_dimensions=None, e_separation_dimensions=None,
        d_box_dimensions=None, d_separation_dimensions=None
    ):

        if len(shank_dimensions) == 3:
            self.height = shank_dimensions[0]  # height of shank
            self.width = shank_dimensions[1]  # width of shank
            self.tip = shank_dimensions[2]  # length between flat bottom and tip
            self.make_vertices()

        if np.all(e_box_dimensions) and len(e_separation_dimensions) == 2:
            self.e_box_height = e_box_dimensions[0]
            self.e_box_width = e_box_dimensions[1]
            self.e_sep_height = e_separation_dimensions[0]
            self.e_sep_width = e_separation_dimensions[1]
            self.init_e_boxes()
        else:
            self.e_pixels = None

        if np.all(d_box_dimensions) and len(d_separation_dimensions) == 2:
            self.d_box_height = d_box_dimensions[0]
            self.d_box_width = d_box_dimensions[1]
            self.d_sep_height = d_separation_dimensions[0]
            self.d_sep_width = d_separation_dimensions[1]
            self.init_d_boxes()
        else:
            self.d_pixels = None

        if self.d_pixels:
            self.boxes = self.d_pixels
        elif self.e_pixels:
            self.boxes = self.e_pixels
        else:
            self.boxes = None

        # This code will set boxes equal to all availible pixels
        # if self.e_pixels:
        #     if self.d_pixels:
        #         self.boxes = self.e_pixels + self.d_pixels
        #     else:
        #         self.boxes = self.e_pixels
        # elif self.d_pixels:
        #     self.boxes = self.d_pixels
        # else:
        #     self.boxes = None

    def make_vertices(self):
        self.tl = np.array([-self.width / 2, 0, self.height / 2])  # top left
        self.bl = np.array([-self.width / 2, 0, -self.height / 2])  # bottom left
        self.br = np.array([self.width / 2, 0, -self.height / 2])  # bottom right
        self.tr = np.array([self.width / 2, 0, self.height / 2])  # top right
        self.tip = np.array([0, 0, -self.tip / 2])  # tip

        self.centroid = np.array([0, 0, 0])
        self.n = np.array([0, 1, 0])  # normal vector of the surface

    def rotate(self, seq="x", angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.tip = r.apply(self.tip)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

        if self.boxes:
            [box.rotate(seq, angles) for box in self.boxes]

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.tip += r
        self.centroid += r

        if self.boxes:
            [box.translate(r) for box in self.boxes]

    def vertices_positions(self, precision=13):
        xs = np.round(
            [self.br[0], self.tr[0], self.tl[0], self.bl[0], self.tip[0]], precision
        )
        ys = np.round(
            [self.br[1], self.tr[1], self.tl[1], self.bl[1], self.tip[1]], precision
        )
        zs = np.round(
            [self.br[2], self.tr[2], self.tl[2], self.bl[2], self.tip[2]], precision
        )
        return xs, ys, zs

    def plot_plane(self, precision=13, ax=None, show=False):
        if not ax:
            plt.figure("Shank", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        xs, ys, zs = self.vertices_positions(precision=precision)
        # ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=0.5, facecolor="gray")
        srf.set_edgecolor("black")
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
        ax.set_adjustable("datalim")

        if "boxes" in self.__dict__.keys():
            for box in self.boxes:
                xs, ys, zs = box.vertices_positions(precision=precision)
                # ax.scatter(xs, ys, zs)

                # 1. create vertices from points
                verts = [list(zip(xs, ys, zs))]
                # 2. create 3d polygons and specify parameters
                srf = Poly3DCollection(verts, alpha=0.5, facecolor="turquoise")
                # 3. add polygon to the figure (current axes)
                plt.gca().add_collection3d(srf)
                ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
                ax.set_adjustable("datalim")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        delta = self.height / 2
        ax.set_xlim(self.centroid[0] - delta, self.centroid[0] + delta)
        ax.set_ylim(self.centroid[1] - delta, self.centroid[1] + delta)
        ax.set_zlim(self.centroid[2] - delta, self.centroid[2] + delta)

        if show:
            plt.show()

    def plot_2d(self, show=False):
        fig, ax = plt.subplots(ncols=1, nrows=1)

        xs, _, zs = self.vertices_positions(precision=13)
        verts = list(zip(xs, zs))
        poly = plt.Polygon(verts, ec="k", fc="gray")
        ax.add_patch(poly)

        ax.scatter(xs, zs, s=0)

        if self.e_pixels:
            for box in self.e_pixels:
                xs, _, zs = box.vertices_positions(precision=13)
                verts = list(zip(xs, zs))
                poly = plt.Polygon(verts, ec="k", fc="blue")
                ax.add_patch(poly)
                ax.scatter(xs, zs, s=0)
        if self.d_pixels:
            for box in self.d_pixels:
                xs, _, zs = box.vertices_positions(precision=13)
                verts = list(zip(xs, zs))
                poly = plt.Polygon(verts, ec="k", fc="green")
                ax.add_patch(poly)
                ax.scatter(xs, zs, s=0)

        if show:
            plt.show()


    def count(self, shank_length, box_length, box_sep):
        return np.ceil((shank_length + box_sep) / (box_length + box_sep)) -1


    def margin(self, shank_length, box_length, sep_length, count):
        return (shank_length - (count * box_length + (count - 1) * sep_length)) / 2


    def e_centroids(self):
        # find the greatest number of pixels that can fit on the width of the shank
        row_count = self.count(self.width, self.e_box_width, self.e_sep_width) 
        # find the margin for each side of the row
        row_margin = self.margin(self.width, self.e_box_width, self.e_sep_width, row_count)

        # find the greatest number of pixels that can fit on the length of the shank
        column_count = self.count(self.height, self.e_box_height, self.e_sep_height) 
        # find the margin for each side of the column
        column_margin = self.margin(self.height, self.e_box_height, self.e_sep_height, column_count)


        # calculate the x & z positions of boxes
        xs = np.arange(
            self.bl[0] + row_margin + self.e_box_width * 3 / 2,
            self.width/2 - row_margin,
            self.e_box_width + self.e_sep_width,
        )
        zs = np.arange(
            self.bl[2] + column_margin + self.e_box_height,
            self.height/2 - column_margin,
            self.e_box_height + self.e_sep_height,
        )

        return [[x, 0, z] for x in xs for z in zs]


    def d_centroids(self):
        # find the greatest number of pixels that can fit on the width of the shank
        row_count = self.count(self.width, self.e_box_width, self.e_sep_width) 
        # find the margin for each side of the row
        row_margin = self.margin(self.width, self.e_box_width, self.e_sep_width, row_count)

        # find the greatest number of pixels that can fit on the length of the shank
        column_count = self.count(self.height, self.e_box_height, self.e_sep_height) 
        # find the margin for each side of the column
        column_margin = self.margin(self.height, self.e_box_height, self.e_sep_height, column_count)

        # calculate the x & z positions of boxes
        xs = np.arange(
            self.bl[0] + row_margin + self.e_box_width / 2 - 2.5,
            self.width/2 - row_margin,
            self.d_box_width + self.d_sep_width,
        )
        zs = np.arange(
            self.bl[2] + column_margin + self.e_box_height / 2 - 2.5,
            self.height/2 - column_margin,
            self.d_box_height + self.d_sep_height,
        )

        candidates = [[x, 0, z] for x in xs for z in zs]
        e_pixels = self.e_centroids()
        e_xs = set([centroid[0] for centroid in e_pixels])
        e_zs = set([centroid[2] for centroid in e_pixels])
        x_overlaps = [x for x in xs for ex in e_xs if(abs(x-ex) <= self.e_box_width/2)]
        z_overlaps = [z for z in zs for ez in e_zs if(abs(z-ez) <= self.e_box_height/2)]
        return [candidate for candidate in candidates
            if not ( (candidate[0] in x_overlaps) and (candidate[2] in z_overlaps) )
        ]
        

    def init_e_boxes(self):
        coords = self.e_centroids()
        self.e_pixels = [Square(self.e_box_height, self.e_box_width, True) for coor in coords]
        [i[0].translate(i[1]) for i in zip(self.e_pixels, coords)]


    def init_d_boxes(self):
        coords = self.d_centroids()
        self.d_pixels = [Square(self.d_box_height, self.d_box_width, False) for coor in coords]
        [i[0].translate(i[1]) for i in zip(self.d_pixels, coords)]


class ShankGroup:
    def __init__(
        self,
        n_shanks=3,
        shank_dimensions=[1200, 100, 1300],
        box_dimensions=None,
        separation_dimensions=None,
        gname="",
        snames=None,
    ):
        self.n_shanks = n_shanks
        self.shank = [
            Shank(shank_dimensions, box_dimensions, separation_dimensions)
            for i in range(self.n_shanks)
        ]
        self.gname = gname

        if isinstance(snames, list):
            snames = snames
        else:
            snames = [snames]

        if len(snames) == self.n_shanks:
            for sname in snames:
                self.shank.name = sname

    def plot_shanks(self, ax=None, show=True):
        if not ax:
            plt.figure("ShankGroup", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        for shank in self.shank:
            shank.plot_plane(ax=ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_adjustable("datalim")

        if not np.diff(
            [self.shank[i].centroid for i in range(self.n_shanks)], 1, 0
        ).sum():
            sh0 = self.shank[0]
            delta = sh0.h / 2
            ax.set_xlim(sh0.centroid[0] - delta, sh0.centroid[0] + delta)
            ax.set_ylim(sh0.centroid[1] - delta, sh0.centroid[1] + delta)
            ax.set_zlim(sh0.centroid[2] - delta, sh0.centroid[2] + delta)
        else:
            big_list = [shank.w for shank in self.shank] + [
                shank.h for shank in self.shank
            ]
            delta = np.max(big_list)
            centroid = np.asarray([shank.centroid for shank in self.shank]).mean(0)
            ax.set_xlim(centroid[0] - delta, centroid[0] + delta)
            ax.set_ylim(centroid[1] - delta, centroid[1] + delta)
            ax.set_zlim(centroid[2] - delta, centroid[2] + delta)

        if show:
            plt.show()

    def to_df(self):
        df = pd.DataFrame(columns=["BoxType", "center", "normal", "top", "h", "w", "t"])

        df["BoxType"] = [box.name for shank in self.shank for box in shank.boxes]
        df["center"] = [
            ('"' + str(tuple(box.centroid)) + '"')
            for shank in self.shank
            for box in shank.boxes
        ]
        df["normal"] = [
            ('"' + str(tuple(box.n)) + '"')
            for shank in self.shank
            for box in shank.boxes
        ]
        df["h"] = [box.h for shank in self.shank for box in shank.boxes]
        df["w"] = [box.w for shank in self.shank for box in shank.boxes]
        df["t"] = 0

        return df
