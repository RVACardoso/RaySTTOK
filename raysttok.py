import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import degrees, radians, asin, cos, sin, floor, sqrt
from scipy.special import jv
from scipy.stats import multivariate_normal
import random
import sys
sys.path.append('./resources')

from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.material import Lambert, UniformSurfaceEmitter, AbsorbingSurface, UniformVolumeEmitter
from raysect.optical.library import RoughNickel
from raysect.optical.library.spectra.colours import red, green, blue, yellow, purple, orange, cyan, red_orange
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, Pixel, PowerPipeline0D, TargettedPixel
from raysect.primitive import Box, Sphere, Cylinder
from raysect.primitive.mesh import import_stl
from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.material import InhomogeneousVolumeEmitter, HomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator



class PlasmaProf(InhomogeneousVolumeEmitter):

    def __init__(self, shape='gaussian', integ_step=0.01, emissiv_gain=1.0, mean=None, cov=None):

        super().__init__(integrator=NumericalIntegrator(step=integ_step, min_samples=1))
        self.divertor_pos_seq = np.load('resources/divertorpos.npy')
        self.x_nl = np.array([[2.4048, 5.5201, 8.6537, 11.7915, 14.9309, 18.07106, 21.2116],
                              [3.8317, 7.0156, 10.1735, 13.3237, 16.4706, 19.6159, 22.7601],
                              [5.1356, 8.4172, 11.6198, 14.796, 17.9598, 21.1170, 24.2701],
                              [6.3802, 9.761, 13.0152, 16.2235, 19.4094, 22.5827, 25.7482],
                              [7.5883, 11.0647, 14.3725, 17.616, 20.8269, 24.0190, 27.1991],
                              [8.7715, 12.3386, 15.7002, 18.9801, 22.2178, 25.4303, 28.6266],
                              [9.9361, 13.5893, 17.0038, 20.3208, 23.5861, 26.8202, 30.0337]])

        self.emissiv_gain = emissiv_gain
        self.x_centroid = random.uniform(-0.06, 0.06)
        self.y_centroid = random.uniform(-0.06, 0.06)

        if shape == 'gaussian':
            self.plasma_shape = self.plasma_gaussian
            if mean == None or cov == None:
                cov_value = random.uniform(0.0005, 0.005)
                self.gaussian_func = multivariate_normal([self.x_centroid, self.y_centroid], [[cov_value, 0.0], [0.0, cov_value]])
            else:
                self.gaussian_func = multivariate_normal(mean, cov)

        elif shape == 'bessel':
            self.plasma_shape = self.plasma_bessel
            self.n_count = 5
            self.l_count = 5
            self.k1 = np.random.rand(self.n_count * self.l_count) * 2.0 - 1.0
            self.k2 = np.random.rand(self.n_count * self.l_count) * 2.0 - 1.0

        self.plot_profile()

    def radial_bessel(self, n, l, r):
        r_new = np.copy(r)
        r_scl = r_new * self.x_nl[n, l] * 10
        radial_value = jv(n, r_scl)

        return radial_value

    def plasma_bessel(self, x_point, y_point):

        radius = np.sqrt((x_point - self.x_centroid) ** 2 + (y_point - self.y_centroid) ** 2)
        theta = np.arctan2(y_point - self.y_centroid, x_point - self.x_centroid)

        if np.sqrt(x_point*x_point+y_point*y_point) > 0.085:
            return 0
        else:
            tomo = 0
            i = 0
            for n in range(self.n_count):
                # print(n)
                for l in range(self.l_count):
                    radial_value = self.radial_bessel(n, l, radius)
                    tomo += radial_value * (self.k1[i] * np.cos(n * theta) + self.k2[i] * np.sin(n * theta))
                    i += 1
            if tomo < 0:
                return 0
            else:
                return tomo

    def plasma_gaussian(self, x_point, y_point):

        radius = np.sqrt((x_point - self.x_centroid) ** 2 + (y_point - self.y_centroid) ** 2)
        if np.sqrt(x_point*x_point+y_point*y_point) > 0.085:
            return 0
        else:
            tomo = self.gaussian_func.pdf([x_point, y_point])
            if tomo < 0:
                return 0
            else:
                return tomo

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):
        spectrum.samples += self.emissiv_gain * self.plasma_shape(-point.x, point.y)
        return spectrum

    def plot_profile(self):
        tomogram = []
        for x in np.linspace(start=-0.1, stop=0.1, num=60):
            for y in np.linspace(start=-0.1, stop=0.1, num=60):
                    tomogram.append(self.emissiv_gain * 1e-3 * self.plasma_shape(y, -x))
        tomogram = np.array(tomogram).reshape((60, 60))
        plt.figure()
        plt.imshow(tomogram, cmap=matplotlib.cm.get_cmap("plasma"))
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Emissivity ($kWm^{-3}$)', rotation=90)
        plt.xticks(np.linspace(0, 59, num=3),
                   np.round(np.arange(start=-0.1, stop=0.1 + 0.1, step=0.1), 2))
        plt.yticks(np.linspace(0, 59-1, num=5),
                   np.flipud(np.round(np.arange(start=-0.1, stop=0.1 + 0.02, step=0.05), 2)))
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")


class PixelMaterial(InhomogeneousVolumeEmitter):

    def __init__(self, pixel_side, pixel_id, print=True):

        self.INTEG_STEP = 0.0005
        super().__init__(integrator=NumericalIntegrator(step=self.INTEG_STEP, min_samples=1))

        self.pixel_id = pixel_id
        self.pixel_side = pixel_side

        self.pixel_size = 0.2/self.pixel_side
        self.pos_x = -(self.pixel_id % self.pixel_side)*self.pixel_size + 0.1
        self.pos_y = -floor(self.pixel_id/self.pixel_side)*self.pixel_size + 0.1

        if print:
            self.plot_pixel()

    def plot_pixel(self):
        out = np.zeros(self.pixel_side*self.pixel_side)
        out[self.pixel_id] = 1.0
        out = np.array(out).reshape((self.pixel_side, self.pixel_side))

        plt.figure()
        plt.imshow(out.reshape((self.pixel_side, self.pixel_side)), cmap=matplotlib.cm.get_cmap("plasma"))
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Emissivity ($kWm^{-3}$)', rotation=90)
        plt.xticks(np.linspace(0, self.pixel_side-1, num=3),
                   np.round(np.arange(start=-0.1, stop=0.1 + 0.1, step=0.1), 2))
        plt.yticks(np.linspace(0, self.pixel_side-1, num=5),
                   np.flipud(np.round(np.arange(start=-0.1, stop=0.1 + 0.02, step=0.05), 2)))
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")

        return np.array(out)

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

        if self.pos_x > point.x > self.pos_x-self.pixel_size and self.pos_y > point.y > self.pos_y - self.pixel_size:
            spectrum.samples += 1.0
        else:
            spectrum.samples += 0.0
        return spectrum


class Vacuum(HomogeneousVolumeEmitter):

    def emission_function(self, direction, spectrum, world, ray, primitive, to_local, to_world):
        spectrum.samples += 0.0
        return spectrum


class RaySTTOK:

    def __init__(self, reflections=True, pixel_samples=1000):
        self.reflections = reflections
        self.pixel_samples = pixel_samples

        self.world = World()

        self.vessel_width = 0.060
        self.vessel_in_rad = 0.100
        self.vessel_out_rad = 0.101

        self.lid_height = 0.0016
        self.cam_in_radius = 0.0184
        self.tube_height = 0.0401

        self.lid_top = 0.00155
        self.lid_outer = 0.00155

        self.x_shift_top = -0.005
        self.y_shift_top = 0.09685
        self.x_shift_outer = -0.1091

        self.top_twist = 0.000 + 0.0003
        self.top_px_first_x = self.x_shift_top + 0.0001 + 0.000375 + 7 * 0.00095 + 0.0
        self.top_px_first_y = self.y_shift_top + 0.009 + self.top_twist - 0.0003
        self.top_px_z = self.vessel_width / 2

        self.out_twist = 0.0007 - 0.0003
        self.out_px_first_y = 0.0001 + 0.000375 + 7 * 0.00095 - 0.00015 + 0.0001
        self.out_px_first_x = self.x_shift_outer - 0.009 + self.out_twist + 0.00035 - 0.0005
        self.out_px_z = self.vessel_width / 2

        self.vessel = None
        self.camera_top, self.camera_outer = None, None
        self.top_pinhole, self.out_pinhole = None, None
        self.top_data = []
        self.out_data = []

        self.add_isttok()

    def check_scene(self, max_iter=200):

        self.vessel.material = Lambert(blue)
        self.camera_outer.material = Lambert(yellow)
        self.camera_top.material = Lambert(yellow)
        self.source.material = Lambert(green)
        self.top_pinhole.material = Lambert(green)
        self.out_pinhole.material = Lambert(green)


        # cube walls
        bottom = Box(lower=Point3D(-0.99, -1.02, -0.99), upper=Point3D(0.99, -1.01, 0.99), parent=self.world,
                      material=Lambert(red))
        # top = Box(lower=Point3D(-0.99, 1.01, -0.99), upper=Point3D(0.99, 1.02, 0.99), parent=self.world,
        #           material=Lambert(red))
        left = Box(lower=Point3D(1.01, -0.99, -0.99), upper=Point3D(1.02, 0.99, 0.99), parent=self.world,
                    material=Lambert(yellow))
        # right = Box(lower=Point3D(-1.02, -0.99, -0.99), upper=Point3D(-1.01, 0.99, 0.99), parent=self.world,
        #             material=Lambert(purple))
        back = Box(lower=Point3D(-0.99, -0.99, 1.01), upper=Point3D(0.99, 0.99, 1.02), parent=self.world,
                   material=Lambert(orange))

        # various wall light sources
        light_front = Box(lower=Point3D(-1.5, -1.5, -10.1), upper=Point3D(1.5, 1.5, -10), parent=self.world,
                          material=UniformSurfaceEmitter(d65_white, 1.0))
        light_top = Box(lower=Point3D(-0.99, 1.01, -0.99), upper=Point3D(0.99, 1.02, 0.99), parent=self.world,
                        material=UniformSurfaceEmitter(d65_white, 1.0), transform=translate(0, 1.0, 0))

        light_bottom = Box(lower=Point3D(-0.99, -3.02, -0.99), upper=Point3D(0.99, -3.01, 0.99), parent=self.world,
                        material=UniformSurfaceEmitter(d65_white, 1.0), transform=translate(0, 1.0, 0))

        light_right = Box(lower=Point3D(-1.92, -0.99, -0.99), upper=Point3D(-1.91, 0.99, 0.99), parent=self.world,
                          material=UniformSurfaceEmitter(d65_white, 1.0))

        light_left = Box(lower=Point3D(1.91, -0.99, -0.99), upper=Point3D(1.92, 0.99, 0.99), parent=self.world,
                          material=UniformSurfaceEmitter(d65_white, 1.0))

        # Process the ray-traced spectra with the RGB pipeline.
        rgb = RGBPipeline2D()

        # camera
        pix = 1000
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(-0.01, 0.0, -0.25) * rotate(0, 0, 0))
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(0.0, 0.0, 0.4) * rotate(180, 0, 0))
        # top view
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(0.0, self.vessel_out_rad+0.15, self.vessel_width/2)*rotate(0, -90, 0))
        # prof
        camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(-0.13, 0.13, -0.2) * rotate(-25, -25.0, 0))

        # camera top side
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(self.x_shift_top, self.top_px_first_y+0.0004, self.top_px_z-self.cam_in_radius+0.005)*rotate(0, 0, 0))
        # camera top down-up
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(self.x_shift_top, self.top_px_first_y-0.01, self.vessel_width/2)*rotate(0, 90, 0))
        # camera top up-down
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(self.x_shift_top-0.004, self.top_px_first_y+self.lid_top+self.tube_height-0.01, self.vessel_width/2)*rotate(0, -90, 0))

        # camera out side
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(-self.vessel_out_rad-0.015, 0.000, self.vessel_width/2-self.cam_in_radius/2+0.0001))
        # camera out down-up
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(self.out_px_first_x+0.005+0.005, 0.0, self.vessel_width/2)*rotate(90, 0, 0))
        # camera out up-down
        # camera = PinholeCamera((pix, pix), pipelines=[rgb], transform=translate(-self.vessel_out_rad-self.tube_height-0.01, 0.0, self.vessel_width/2-0.005)*rotate(-90, 0, 0))

        # camera - pixel sampling settings
        camera.fov = 60  # 45
        camera.pixel_samples = 10

        # camera - ray sampling settings
        camera.spectral_rays = 1
        camera.spectral_bins = 25
        camera.parent = self.world

        plt.ion()
        p = 1
        while not camera.render_complete:
            print("Rendering pass {}...".format(p))
            camera.observe()
            print()
            p += 1
            if p > max_iter:
                break

        plt.ioff()
        rgb.display()

    def add_isttok(self):
        self.top_px = []
        self.out_px = []
        self.top_power = []
        self.out_power = []

        nickel_roughness = 0.23
        min_wl = 50
        max_wl = 51

        # add vessel and cameras
        if self.reflections:
            self.vessel = import_stl("../isttok_3d/vessel5_stl.stl", scaling=1, mode='binary', parent=self.world,
                                 material=RoughNickel(nickel_roughness),
                                 transform=translate(0, 0, 0) * rotate(0, 0, 0))
        else:
            self.vessel = import_stl("../isttok_3d/vessel5_stl.stl", scaling=1, mode='binary', parent=self.world,
                                    material=AbsorbingSurface(),
                                    transform=translate(0, 0, 0) * rotate(0, 0, 0))

        self.camera_top = import_stl("../isttok_3d/camera_top3_stl.stl", scaling=1, mode='binary', parent=self.world,
                                     material=AbsorbingSurface(),
                                     transform=translate(self.x_shift_top,
                                                         self.y_shift_top + self.tube_height + self.lid_top,
                                                         self.vessel_width / 2.0) * rotate(0, -90, 0))

        self.camera_outer = import_stl("../isttok_3d/camera_outer4_newpin_stl.stl", scaling=1, mode='binary', parent=self.world,
                                       material=AbsorbingSurface(),
                                       transform=translate(self.x_shift_outer - self.tube_height - self.lid_outer,
                                                           0.0,
                                                           self.vessel_width / 2.0) * rotate(-90, 0, 0))

        pinhole_sphere_radius = 0.0005
        self.top_pinhole = Sphere(radius=pinhole_sphere_radius, parent=self.world,
                                  transform=translate(self.x_shift_top, self.y_shift_top, self.vessel_width/2),
                                  material=Vacuum())
        pinhole_sphere_radius = 0.00035
        self.out_pinhole = Sphere(radius=pinhole_sphere_radius, parent=self.world,
                                  transform=translate(self.x_shift_outer, 0.0, self.vessel_width / 2),
                                  material=Vacuum())

        for i in range(16):
            self.top_power.append(PowerPipeline0D(accumulate=False))
            self.out_power.append(PowerPipeline0D(accumulate=False))

            top_px_x = self.top_px_first_x - i * 0.00095
            top_px_y = self.top_px_first_y - i * 2 * (self.top_twist / 15)
            top_angle = degrees(asin(2 * self.top_twist / 0.01425))

            out_px_y = self.out_px_first_y - i * 0.00095
            out_px_x = self.out_px_first_x - i * 2 * (self.out_twist / 15)
            out_angle = -degrees(asin(2 * self.out_twist / 0.01425))

            self.top_px.append(TargettedPixel(targets=[self.top_pinhole], targetted_path_prob=1.0, pipelines=[self.top_power[i]],
                                     x_width=0.00075, y_width=0.00405,
                                     min_wavelength=min_wl, max_wavelength=max_wl,
                                     spectral_bins=1, pixel_samples=self.pixel_samples, parent=self.world, quiet=True,
                                     ray_importance_sampling=True, ray_important_path_weight=0.05, ray_max_depth=50,
                                     transform=translate(top_px_x, top_px_y, self.top_px_z) *
                                               rotate(0, 0, top_angle) * rotate(0, -90, 0)))

            self.out_px.append(TargettedPixel(targets=[self.out_pinhole], targetted_path_prob=1.0, pipelines=[self.out_power[i]],
                                     x_width=0.00075, y_width=0.00405,
                                     min_wavelength=min_wl, max_wavelength=max_wl,
                                     spectral_bins=1, pixel_samples=self.pixel_samples, parent=self.world, quiet=True,
                                     ray_importance_sampling=True, ray_important_path_weight=0.05, ray_max_depth=50,
                                     transform=translate(out_px_x, out_px_y, self.out_px_z) *
                                               rotate(0, 0, out_angle) * rotate(-90, 0, 90)))

    def place_source(self):
        raise NotImplementedError

    @staticmethod
    def plot_lamp_tomogram(radius, angle):
        lamp_rad = 0.0041 / 2.0
        radius = radius
        angle = angle * np.pi / 180.0
        lamp_x = radius * cos(angle)
        lamp_y = radius * sin(angle)
        pixel_nr = 60
        pixel_size = 0.2 / pixel_nr
        center_pixel_x = floor((lamp_x + 0.1) / pixel_size)
        center_pixel_y = floor((0.1 - lamp_y) / pixel_size)

        pixel_div = 4
        pixel_x_list, pixel_y_list = [], []
        point_count_list = []
        for pixel_x in [center_pixel_x - 2, center_pixel_x - 1, center_pixel_x, center_pixel_x + 1, center_pixel_x + 2]:
            for pixel_y in [center_pixel_y - 2, center_pixel_y - 1, center_pixel_y, center_pixel_y + 1,
                            center_pixel_y + 2]:

                point_count = 0
                points_x = np.linspace(start=pixel_x * pixel_size, stop=(pixel_x + 1) * pixel_size, num=pixel_div) - 0.1
                points_y = 0.1 - np.linspace(start=pixel_y * pixel_size, stop=(pixel_y + 1) * pixel_size, num=pixel_div)

                for x_values in points_x:
                    for y_values in points_y:
                        if (x_values - lamp_x) ** 2 + (y_values - lamp_y) ** 2 <= lamp_rad ** 2:
                            point_count += 1

                pixel_x_list.append(pixel_x)
                pixel_y_list.append(pixel_y)
                point_count_list.append(point_count)

        nonzero_idx = np.nonzero(point_count_list)[0]
        pixel_x_list = np.array(pixel_x_list)[nonzero_idx]
        pixel_y_list = np.array(pixel_y_list)[nonzero_idx]

        point_count_list = np.array(point_count_list)[nonzero_idx]
        point_count_list = point_count_list / np.max(point_count_list)

        tomo = np.zeros(pixel_nr * pixel_nr)
        for x_px, y_px, count in zip(pixel_x_list, pixel_y_list, point_count_list):
            idx = y_px * 60 + x_px
            tomo[idx] = count

        plt.figure()
        plt.imshow(tomo.reshape((pixel_nr, pixel_nr)), cmap=matplotlib.cm.get_cmap("plasma"))
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Emissivity ($kWm^{-3}$)', rotation=90)
        plt.xticks(np.linspace(0, 59, num=3),
                   np.round(np.arange(start=-0.1, stop=0.1 + 0.1, step=0.1), 2))
        plt.yticks(np.linspace(0, 59, num=5),
                   np.flipud(np.round(np.arange(start=-0.1, stop=0.1 + 0.02, step=0.05), 2)))
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")

    def place_lamp(self, radius, angle):
        self.lamp_radius = radius
        self.lamp_angle = angle
        pos_x = -self.lamp_radius * cos(radians(self.lamp_angle))
        pos_y = self.lamp_radius * sin(radians(self.lamp_angle))
        self.lamp_source = Cylinder(radius=0.00205, height=0.15, parent=self.world,
                               transform=translate(pos_x, pos_y, -0.0454),
                               material=UniformVolumeEmitter(ConstantSF(1.0), scale=1.0),
                               name="light cylinder")

        self.plot_lamp_tomogram(radius=radius, angle=angle)

    def place_plasma(self, shape='gaussian', emissiv_gain=1.0, mean=None, cov=None):
        fourier_plasma = PlasmaProf(shape=shape, emissiv_gain=emissiv_gain, mean=mean, cov=cov)
        # print(type(fourier_plasma))
        self.plasma_source = Cylinder(radius=self.vessel_in_rad - 0.0015, height=self.vessel_width,
                               parent=self.world, transform=translate(0, 0, 0) * rotate(0, 0, 0), name='PlasmaSource',
                               material=fourier_plasma)

    def place_single_pixel(self, pixel_id, pixel_side=60):
        pixel_material = PixelMaterial(pixel_side=pixel_side, pixel_id=pixel_id)
        self.pixel_source = Cylinder(radius=self.vessel_in_rad - 0.0015, height=self.vessel_width,
                                      parent=self.world, transform=translate(0, 0, 0) * rotate(0, 0, 0),
                                      name='PlasmaSource',
                                      material=pixel_material)

    def simulate_rays(self, pixels=range(16)):
        top_data = []
        out_data = []
        for i in pixels:
            self.top_px[i].observe()
            top_data.append(self.top_power[i].value.mean)
            self.out_px[i].observe()
            out_data.append(self.out_power[i].value.mean)

        self.top_data = np.array(top_data)
        self.out_data = np.array(out_data)

        return self.top_data, self.out_data

    def plot_detectors(self):

        font = {'family': 'normal',
                'size': 32}
        matplotlib.rc('font', **font)

        plt.figure()
        cam_nr = np.arange(16)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

        plt.subplot(121)
        plt.bar(cam_nr, self.top_data)
        plt.xticks(cam_nr, cam_nr)
        plt.ylabel("Power (W)")
        plt.xlabel("Detector number")
        plt.title("Power collected on\ntop camera")
        plt.tick_params(axis='x', which='major', labelsize=17)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.subplot(122)
        plt.bar(cam_nr, self.out_data)
        plt.xticks(cam_nr, cam_nr)
        plt.ylabel("Power (W)")
        plt.xlabel("Detector number")
        plt.title("Power collected on\n  outer camera")
        plt.tick_params(axis='x', which='major', labelsize=17)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    def print_detectors(self):
        print("Top detectors: " + str(self.top_data))
        print("Outer detectors: " + str(self.out_data))

    def get_proj_matrix(self, pixel_side=60, out_file="proj_matrix1"):

        geo_matrix = np.empty((32, 0))
        for pixel in range(pixel_side*pixel_side):
            print('Progress: ' + str(round(100*pixel/(pixel_side*pixel_side), 1)) + " %")

            self.world = World()
            self.add_isttok()

            self.pixel_source = Cylinder(radius=self.vessel_in_rad - 0.0015, height=self.vessel_width,
                                         parent=self.world, transform=translate(0, 0, 0) * rotate(0, 0, 0),
                                         name='PlasmaSource')
            one_pixel = PixelMaterial(pixel_side=pixel_side, pixel_id=pixel, print=False)
            self.pixel_source.material = one_pixel
            self.simulate_rays()

            column = np.hstack((self.top_data, self.out_data)).reshape((32, 1))
            geo_matrix = np.hstack((geo_matrix, column))

        np.save(out_file, geo_matrix)

    def plot_line_matrix(self, line, mat_file):
        plt.figure()
        proj_mat = np.load(mat_file)
        pixel_side = round(sqrt(proj_mat.shape[1]))
        plt.imshow(proj_mat[line].reshape((pixel_side, pixel_side)), cmap=matplotlib.cm.get_cmap("plasma"))
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Emissivity ($kWm^{-3}$)', rotation=90)
        plt.xticks(np.linspace(0, pixel_side-1, num=3),
                   np.round(np.arange(start=-0.1, stop=0.1 + 0.1, step=0.1), 2))
        plt.yticks(np.linspace(0, pixel_side-1, num=5),
                   np.flipud(np.round(np.arange(start=-0.1, stop=0.1 + 0.02, step=0.05), 2)))
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")

    def show_plots(self):
        plt.show()

