import h5py
import numpy as np
import struct
import random
import sys
from sys import byteorder
import xmlParser
from xmlParser import *
import taichi as ti
import ctypes
import math
# ti.init(arch=ti.cuda, default_fp=ti.f64)
real = ti.f32
realnp = np.float32
ti.init(default_fp=real, arch=ti.gpu)

@ti.data_oriented
class AGTaichiMPM:
    def __init__(self, xmlData):
        # material parameters
        self.hb_n = xmlData.integratorData.herschel_bulkley_power
        self.hb_eta = xmlData.integratorData.eta
        self.hb_sigmaY = xmlData.integratorData.yield_stress
        self.kappa = xmlData.integratorData.bulk_modulus
        self.mu = xmlData.integratorData.shear_modulus

        # flip-pic alpha
        self.alpha = xmlData.integratorData.flip_pic_alpha

        # temporal/spatial resolution
        self.dt = xmlData.integratorData.dt
        self.dx = xmlData.gridData.cell_width
        self.invdx = 1.0 / self.dx

        # near earth gravity
        self.g = ti.Vector(xmlData.nearEarthGravityData.g)

        # iteration count
        self.iteration = ti.field(dtype=int, shape=())
        self.iteration[None] = 0

        # auto save
        self.auto_save_fps = xmlData.autoSaveData.fps
        self.auto_save_filename = xmlData.autoSaveData.filename

        # max time
        self.max_time = xmlData.integratorData.max_time

        # configuring grid by using the specified grid center and cell width as is
        # min and max will be recomputed because the specified grid size may not agree with the specified cell width

        # compute grid center and tentative grid width
        grid_center = (xmlData.gridData.max + xmlData.gridData.min) * 0.5
        grid_width = xmlData.gridData.max - xmlData.gridData.min
        self.cell_count = np.ceil(grid_width / self.dx).astype(int)

        # recompute grid width, min and max
        grid_width = self.cell_count.astype(realnp) * self.dx
        self.grid_min = ti.Vector(grid_center - 0.5 * grid_width)
        self.grid_max = ti.Vector(grid_center + 0.5 * grid_width)

        # allocating fields for grid mass and velocity (momentum)
        self.grid_m = ti.field(dtype=float, shape=self.cell_count)
        self.grid_v = ti.Vector.field(3, dtype=float, shape=self.cell_count)
        self.grid_a = ti.Vector.field(3, dtype=float, shape=self.cell_count)
        # for debug
        self.grid_pos = ti.Vector.field(3, dtype=float, shape=np.prod(self.cell_count))
        self.grid_color = ti.field(ti.i32, shape=np.prod(self.cell_count))

        self.particle_ndcount = np.array([1,1])
        self.particle_init_min = ti.Vector(np.array([0.0, 0.0, 0.0]))
        self.particle_init_cell_samples_per_dim = 2
        self.need_particle_position_initialization = False

        # particles
        if xmlData.cuboidData is not None:
            cuboid_width = xmlData.cuboidData.max - xmlData.cuboidData.min
            self.particle_ndcount = np.ceil(cuboid_width * xmlData.cuboidData.cell_samples_per_dim / self.dx).astype(int)
            self.particle_count = np.prod(self.particle_ndcount)
            self.particle_init_min = ti.Vector(xmlData.cuboidData.min)
            self.particle_init_cell_samples_per_dim = xmlData.cuboidData.cell_samples_per_dim
            self.particle_init_vel = ti.Vector(xmlData.cuboidData.vel)

            self.particle_hl = 0.5 * self.dx / xmlData.cuboidData.cell_samples_per_dim
            self.particle_volume = (self.dx / xmlData.cuboidData.cell_samples_per_dim)**3
            self.particle_mass = xmlData.cuboidData.density * self.particle_volume
            self.particle_x = ti.Vector.field(3, dtype=float, shape=self.particle_count)

            self.need_particle_position_initialization = True

        elif xmlData.pointFileData is not None:
            f = open(xmlData.pointFileData.file_name, 'rb')
            # int32_t
            self.particle_count = struct.unpack('i', f.read(4))[0]
            self.particle_x = ti.Vector.field(3, dtype=real, shape=self.particle_count)
            # double
            self.particle_volume = struct.unpack('d', f.read(8))[0]
            # double
            self.particle_hl = struct.unpack('d', f.read(8))[0] * 0.5

            self.particle_volume = self.particle_hl * self.particle_hl * self.particle_hl * 8.0

            self.particle_init_vel = ti.Vector(xmlData.pointFileData.velocity)
            self.particle_mass = xmlData.pointFileData.density * self.particle_volume

            print('particle_count: ', self.particle_count)
            print('particle_init_vel: ', self.particle_init_vel)
            print('particle_mass: ', self.particle_mass)
            print('particle_volume: ', self.particle_volume)
            print('particle_hl: ', self.particle_hl)

            data = ( np.fromfile(f, dtype=np.double, count=self.particle_count*3, sep='').reshape(self.particle_count, 3) + xmlData.pointFileData.start_point )
            self.particle_x.from_numpy(data.astype(realnp))

            f.close()

        self.particle_v = ti.Vector.field(3, dtype=float, shape=self.particle_count)
        self.particle_be = ti.Matrix.field(3, 3, dtype=float, shape=self.particle_count)
        self.particle_D = ti.Matrix.field(3, 3, dtype=float, shape=self.particle_count)
        self.particle_B = ti.Matrix.field(3, 3, dtype=float, shape=self.particle_count)
        # for instancing inclusions
        self.particle_rot = ti.Matrix.field(3, 3, dtype=float, shape=self.particle_count)
        # for debug
        self.particle_color_f = ti.field(real, shape=self.particle_count)
        self.particle_color = ti.field(ti.i32, shape=self.particle_count)

        # static box list
        self.num_boxes = len(xmlData.staticBoxList)
        self.static_box_min = ti.Vector.field(3, dtype=real, shape=self.num_boxes)
        self.static_box_max = ti.Vector.field(3, dtype=real, shape=self.num_boxes)
        self.static_box_type = ti.field(ti.i32, shape=self.num_boxes)

        for i in range(self.num_boxes):
            self.static_box_max[i] = xmlData.staticBoxList[i].max
            self.static_box_min[i] = xmlData.staticBoxList[i].min
            if xmlData.staticBoxList[i].isSticky:
                self.static_box_type[i] = 1
            else:
                self.static_box_type[i] = 0

        # dynamic rigid objects
        self.num_dynamic_rigid_objects = len(xmlData.dynamicRigidObjectList)
        self.dro_grid_start = []
        self.dro_cell_delta = []
        self.dro_grid_dimension = []
        self.dro_sdf_vals = []
        self.dro_grid_pivot = []
        self.dro_motion_params = []
        self.dro_isSticky = []
        self.dro_motion_function_type = []
        self.dro_sdf_normals = []
        self.dro_center = []
        self.dro_initial_center = []
        self.dro_translation = []
        self.dro_velocity = []
        self.dro_time = []
        self.dro_quat = []

        for i in range(self.num_dynamic_rigid_objects):
            f = open(xmlData.dynamicRigidObjectList[i].file_name, 'rb')
            _dro_grid_start = ti.field(dtype=real, shape=3)
            grid_start = np.fromfile(f, dtype=np.double, count=3, sep='').astype(realnp) + xmlData.dynamicRigidObjectList[i].start_point.astype(realnp)
            _dro_grid_start.from_numpy(grid_start)
            self.dro_grid_start.append(_dro_grid_start)

            print('grid_start: ', _dro_grid_start)

            cell_delta = np.fromfile(f, dtype=np.double, count=3, sep='').astype(realnp)
            _dro_cell_delta = ti.field(dtype=real, shape=3)
            _dro_cell_delta.from_numpy(cell_delta)
            self.dro_cell_delta.append(_dro_cell_delta)

            print('cell_delta: ', _dro_cell_delta)

            _dro_grid_dimension = ti.field(dtype=ti.i32, shape=3)
            _data_grid_dimension = np.fromfile(f, dtype='int32', count=3, sep='')
            _dro_grid_dimension.from_numpy(_data_grid_dimension)
            self.dro_grid_dimension.append(_dro_grid_dimension)
            _grid_dimension = np.array([_data_grid_dimension[2], _data_grid_dimension[1], _data_grid_dimension[0]])

            print('grid_dimension: ', _dro_grid_dimension)

            data = np.fromfile(f, dtype=np.double, count=np.prod(_grid_dimension), sep='')
            _dro_sdf_vals = ti.field(dtype=real, shape=_grid_dimension)
            _dro_sdf_vals.from_numpy(data.reshape(_grid_dimension).astype(realnp))
            self.dro_sdf_vals.append(_dro_sdf_vals)

            _dro_grid_pivot = ti.field(dtype=real, shape=3)
            _dro_grid_pivot.from_numpy(np.array(xmlData.dynamicRigidObjectList[i].params[6:9]).astype(realnp) + xmlData.dynamicRigidObjectList[i].start_point.astype(realnp))
            self.dro_grid_pivot.append(_dro_grid_pivot)

            _dro_motion_params = ti.field(dtype=real, shape=12)
            _dro_motion_params.from_numpy(xmlData.dynamicRigidObjectList[i].params.astype(realnp))
            self.dro_motion_params.append(_dro_motion_params)

            _dro_isSticky = ti.field(dtype=ti.i32, shape=1)
            _dro_isSticky.from_numpy(np.array([int(xmlData.dynamicRigidObjectList[i].isSticky)]).astype(np.int32))
            self.dro_isSticky.append(_dro_isSticky)

            _dro_motion_function_type = ti.field(dtype=ti.i32, shape=1)
            data_dro_0_motion_function_type = np.array([xmlData.dynamicRigidObjectList[i].motion_function_type]).astype(np.int32)
            print(data_dro_0_motion_function_type)
            _dro_motion_function_type.from_numpy(data_dro_0_motion_function_type)
            self.dro_motion_function_type.append(_dro_motion_function_type)

            _dro_sdf_normals = ti.Vector.field(3, dtype=real, shape=_grid_dimension)
            self.dro_sdf_normals.append(_dro_sdf_normals)

            initial_center = grid_start + cell_delta + _data_grid_dimension.astype(realnp) * 0.5
            _dro_center = ti.Vector.field(3, dtype=real, shape=1)
            _dro_center.from_numpy(initial_center.reshape(1,3))
            self.dro_center.append(_dro_center)

            _dro_initial_center = ti.Vector.field(3, dtype=real, shape=1)
            _dro_initial_center.from_numpy(initial_center.reshape(1,3))
            self.dro_initial_center.append(_dro_initial_center)

            _dro_translation = ti.Vector.field(3, dtype=real, shape=1)
            _dro_translation.from_numpy(np.zeros(3).astype(realnp).reshape(1,3))
            self.dro_translation.append(_dro_translation)

            _dro_velocity = ti.Vector.field(3, dtype=real, shape=1)
            _dro_velocity.from_numpy(np.zeros(3).astype(realnp).reshape(1,3))
            self.dro_velocity.append(_dro_velocity)

            _dro_time = ti.field(dtype=real, shape=1)
            _dro_time.from_numpy(np.zeros(1).astype(realnp))
            self.dro_time.append(_dro_time)

            _dro_quat = ti.field(dtype=real, shape=4)
            _dro_quat.from_numpy(np.array([1.0, 0.0, 0.0, 0.0]).astype(realnp))
            self.dro_quat.append(_dro_quat)
            f.close()


        #self.dynamic_rigid_objects = []
        #for i in range(self.num_dynamic_rigid_objects):
        #    _dynamic_rigid_objects = dynamicRigidObject(xmlData.dynamicRigidObjectList[i])
        #    _dynamic_rigid_objects.computeNormals()
        #    self.dynamic_rigid_objects.append(_dynamic_rigid_objects)

    @ti.func
    def linearInterpolation(self, v0, v1, alpha):
        return v0 * (1.0 - alpha) + v1  * alpha

    @ti.func
    def trilinearInterpolation(self, v000, v100, v010, v110, v001, v101, v011, v111, bc):
        v00 = self.linearInterpolation(v000, v100, bc[0])
        v10 = self.linearInterpolation(v010, v110, bc[0])
        v01 = self.linearInterpolation(v001, v101, bc[0])
        v11 = self.linearInterpolation(v011, v111, bc[0])
        v0 = self.linearInterpolation(v00, v10, bc[1])
        v1 = self.linearInterpolation(v01, v11, bc[1])
        return self.linearInterpolation(v0, v1, bc[2])

    @ti.func
    def droComputeNormals(self):
        for r in ti.static(range(self.num_dynamic_rigid_objects)):
            for k, j, i in self.dro_sdf_normals[r]:
                im = i-1 if i > 0 else i
                ip = i+1 if i < self.dro_grid_dimension[r][0] - 1 else i
                jm = j-1 if j > 0 else j
                jp = j+1 if j < self.dro_grid_dimension[r][1] - 1 else j
                km = k-1 if k > 0 else k
                kp = k+1 if k < self.dro_grid_dimension[r][2] - 1 else k

                dx = (ip-im) * self.dro_cell_delta[r][0]
                dy = (jp-jm) * self.dro_cell_delta[r][1]
                dz = (kp-km) * self.dro_cell_delta[r][2]

                vim = self.dro_sdf_vals[r][k, j, im]
                vip = self.dro_sdf_vals[r][k, j, ip]
                vjm = self.dro_sdf_vals[r][k, jm, i]
                vjp = self.dro_sdf_vals[r][k, jp, i]
                vkm = self.dro_sdf_vals[r][km, j, i]
                vkp = self.dro_sdf_vals[r][kp, j, i]

                nx = (vip - vim) / dx
                ny = (vjp - vim) / dy
                nz = (vkp - vkm) / dz

                inv_length = 1.0 / ti.sqrt(nx*nx+ny*ny+nz*nz)
                self.dro_sdf_normals[r][k, j, i] = ti.Vector([nx, ny, nz]) * inv_length

    @ti.kernel
    def initialize(self):
        # clear grid values
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(real, 3)
            self.grid_a[I] = ti.Vector.zero(real, 3)

        # compute grid point locations (for debug)
        for i in range(self.cell_count[0]*self.cell_count[1]):
            gi = i % self.cell_count[0]
            gj = (i // self.cell_count[0]) % self.cell_count[1]
            gk = i // (self.cell_count[0] * self.cell_count[1])
            I = ti.Vector([gi, gj, gk])
            self.grid_pos[i] = self.grid_min + I.cast(real) * self.dx

        # initialize particles
        for i in range(self.particle_count):
            self.particle_v[i] = self.particle_init_vel
            self.particle_be[i] = ti.Matrix.identity(float, 3)
            self.particle_B[i] = ti.Matrix.zero(float, 3, 3)
            self.particle_D[i] = ti.Matrix.identity(float, 3)
            self.particle_rot[i] = ti.Matrix.identity(float, 3)

        if self.need_particle_position_initialization:
            for i in range(self.particle_count):
                pi = i % self.particle_ndcount[0]
                pj = (i // self.particle_ndcount[0]) % self.particle_ndcount[1]
                pk = i // (self.particle_ndcount[0] * self.particle_ndcount[1])
                r = ti.Vector([ti.random(), ti.random(), ti.random()])

                _I = ti.Vector([pi, pj, pk]).cast(float) + r

                self.particle_x[i] = self.particle_init_min + (self.dx / self.particle_init_cell_samples_per_dim) * _I

        min_C = ti.Vector([100.0, 100.0, 100.0])
        max_C = ti.Vector([-100.0, -100.0, -100.0])
        for i in range(self.particle_count):
            min_C = ti.min(min_C, self.particle_x[i])
            max_C = ti.max(max_C, self.particle_x[i])

        print('min_C: ', min_C)
        print('max_C: ', max_C)

        self.droComputeNormals()

    @ti.func
    def check_particle_consistency(self):
        for i in range(self.particle_count):
            if self.particle_v[i].norm() > 1000.0:
                print('too large norm for particle velocity; i: ', i, ', v: ', self.particle_v[i])
            if self.particle_be[i].determinant() <= 0.0:
                print('non positive determinant for particle be; i: ', i, ', ', self.particle_be[i])
            if self.particle_be[i].determinant() > 3.0:
                print('too large determinant for particle be; i: ', i, ', ', self.particle_be[i])
            if ti.abs(self.particle_B[i].determinant()) > 3000.0:
                print('too large or too small determinant for particle B; i: ', i, ', ', self.particle_B[i])
            if ti.abs(self.particle_rot[i].determinant()) > 3000.0:
                print('too large or too small determinant for particle rot; i: ', i, ', ', self.particle_rot[i])
            if self.particle_x[i].norm() > 1000.0:
                print('too large norm for particle position; i: ', i, ', x: ', self.particle_x[i])

    @ti.func
    def check_grid_consistency(self):
        for I in ti.grouped(self.grid_m):
            if (self.grid_v[I].norm() > 0.0) and (self.grid_m[I] == 0.0):
                print('non zero velocity at a zero-mass grid point: ', I, '; m: ', self.grid_m[I], ', v:', self.grid_v[I])
            if (self.grid_v[I].norm() > 1000.0):
                print('too large norm for grif velocity; I: ', I, ', v: ', self.grid_v[I])

    ## linear basis functions
    #@staticmethod
    #@ti.func
    #def linearN(x):
    #    absx = ti.abs(x)
    #    ret = 1.0 - absx if absx < 1.0 else 0.0
    #    return ret

    #@staticmethod
    #@ti.func
    #def lineardN(x):
    #    ret = 0.0
    #    if ti.abs(x) <= 1.0:
    #        ret = -1.0 if x >= 0.0 else 1.0
    #    return ret

    #@staticmethod
    #@ti.func
    #def linearStencil():
    #    return ti.ndrange(2, 2, 2)

    #@ti.func
    #def linearBase(self, particle_pos):
    #    return ((particle_pos - self.grid_min) * self.invdx).cast(int)

    #@ti.func
    #def linearWeightAndGrad(self, particle_pos, grid_pos):
    #    delta_over_h = (particle_pos - grid_pos) * self.invdx
    #    wx = self.linearN(delta_over_h[0])
    #    wy = self.linearN(delta_over_h[1])
    #    wz = self.linearN(delta_over_h[2])
    #    weight = wx * wy * wz
    #    weight_grad = ti.Vector([wy * wz * self.lineardN(delta_over_h[0]), wx * wz * self.lineardN(delta_over_h[1]), wx * wy * self.lineardN(delta_over_h[2])]) * self.invdx
    #    return weight, weight_grad

    # uGIMP basis functions
    @staticmethod
    @ti.func
    def linearIntegral(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = ((w + hl - diff) ** 2) / (2.0 * w)
        elif diff >= hl:
            ret = 2.0 * hl * (1.0 - diff / w)
        else:
            ret = 2.0 * hl - (hl * hl + diff * diff) / w
        return ret

    @staticmethod
    @ti.func
    def linearIntegralGrad(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        sgn = 1.0 if xp - xi >= 0.0 else -1.0
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = -sgn * (w + hl - diff) / w
        elif diff >= hl:
            ret = -sgn * 2.0 * hl / w
        else:
            ret = 2.0 * (xi - xp) / w
        return ret

    @staticmethod
    @ti.func
    def uGIMPStencil():
        return ti.ndrange(3, 3, 3)

    @ti.func
    def uGIMPBase(self, particle_pos):
        return ((particle_pos - self.particle_hl - self.grid_min) * self.invdx).cast(int)

    @ti.func
    def uGIMPWeightAndGrad(self, particle_pos, grid_pos):
        wx = self.linearIntegral(particle_pos[0], self.particle_hl, grid_pos[0], self.dx)
        wy = self.linearIntegral(particle_pos[1], self.particle_hl, grid_pos[1], self.dx)
        wz = self.linearIntegral(particle_pos[2], self.particle_hl, grid_pos[2], self.dx)
        weight = wx * wy * wz / self.particle_volume
        weight_grad = ti.Vector([wy * wz * self.linearIntegralGrad(particle_pos[0], self.particle_hl, grid_pos[0], self.dx), wx * wz * self.linearIntegralGrad(particle_pos[1], self.particle_hl, grid_pos[1], self.dx), wx * wy * self.linearIntegralGrad(particle_pos[2], self.particle_hl, grid_pos[2], self.dx)]) / self.particle_volume
        return weight, weight_grad

    @ti.func
    def hb_eval(self, s, s_tr, inv_eta, inv_n, sigma_Y, __mu__):
        inside = ti.max(0.0, (s - sigma_Y) * inv_eta)
        inside_pwr = 0.0 if inside == 0.0 else ti.pow(inside, inv_n)

        return s - s_tr + 2.0 * __mu__ * self.dt * inside_pwr


    @ti.kernel
    def dbg_dynamic_rigid_step(self):
        self.iteration[None] += 1
        t = self.iteration[None] * self.dt
        self.droSetTime(t)

        #pos = ti.Vector([0.4, 0.2, 0.2])
        #v = ti.Vector([1.0, 0.0, 0.0])
        #print('v (adv): ', v)
        #v = self.droResponce(pos, v, 0.0)
        #print('v (post): ', v)
        for p in self.particle_x:
            # advect
            self.particle_x[p] += self.dt * self.particle_v[p]
            # dynamic rigid objects
            # flag, self.particle_v[p], self.particle_C[p] = self.droResponce(self.particle_x[p], self.particle_v[p], self.particle_C[p], 0.0)
            # if not flag:
            #     self.particle_v[p] = ti.Vector.zero(real, 3)
            #     self.particle_C[p] = ti.Matrix.zero(real, 3, 3)



    @ti.kernel
    def step(self):
        self.iteration[None] += 1

        t = self.iteration[None] * self.dt
        self.droSetTime(t)

        # clear grid data
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(float, 3)
            self.grid_a[I] = ti.Vector.zero(float, 3)

        #compute paricle D
        for p in self.particle_x:
            base = self.uGIMPBase(self.particle_x[p])
            stencil = self.uGIMPStencil()

            self.particle_D[p] = ti.Matrix.zero(real, 3, 3)

            for i, j, k in ti.static(stencil):
                offset = ti.Vector([i, j, k])
                # grid point position
                gp = self.grid_min + (base + offset).cast(float) * self.dx

                # compute weight and weight grad
                weight, weight_grad = self.uGIMPWeightAndGrad(self.particle_x[p], gp)
                self.particle_D[p] += weight * (gp - self.particle_x[p]).outer_product(gp - self.particle_x[p])

            #if self.particle_D[p].determinant() > 1000.0:
            #    print('p: ', p, ', D / (dx^2): ', self.particle_D[p] / (self.dx * self.dx))


        # particle status update and p2g
        for p in self.particle_x:
            # base = self.linearBase(self.particle_x[p])
            # stencil = self.linearStencil()
            base = self.uGIMPBase(self.particle_x[p])
            stencil = self.uGIMPStencil()

            # compute particle stress
            J = ti.sqrt(self.particle_be[p].determinant())
            be_bar = self.particle_be[p] * pow(J, -2.0/3.0)
            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(float, 3) / 3.0
            tau = self.kappa * 0.5 * (J+1.0) * (J-1.0) * ti.Matrix.identity(float, 3) + self.mu * dev_be_bar

            # p2g
            for i, j, k in ti.static(stencil):
                offset = ti.Vector([i, j, k])
                # grid point position
                gp = self.grid_min + (base + offset).cast(float) * self.dx

                # compute weight and weight grad
                # weight, weight_grad = self.linearWeightAndGrad(self.particle_x[p], gp)
                weight, weight_grad = self.uGIMPWeightAndGrad(self.particle_x[p], gp)

                # internal force
                f_internal = - self.particle_volume * tau @ weight_grad

                # accumulate grid velocity, acceleration and mass
                #self.grid_v[base + offset] += weight * self.particle_mass * ( self.particle_v[p] + self.particle_C[p] @ ( gp - self.particle_x[p] ) )
                self.grid_v[base + offset] += weight * self.particle_mass * ( self.particle_v[p] + self.particle_B[p] @ self.particle_D[p].inverse() @ ( gp - self.particle_x[p] ) )
                self.grid_a[base + offset] += f_internal
                self.grid_m[base + offset] += weight * self.particle_mass

        #print('After particle status update and p2g (', self.iteration[None], '): ')
        #self.check_particle_consistency()
        #self.check_grid_consistency()

        # grid update
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                old_momentum = self.grid_v[i, j, k]
                new_momentum = old_momentum + self.dt * ( self.grid_a[i, j, k] + self.grid_m[i, j, k] * self.g )

                # boundary conditions
                gp = self.grid_min + self.dx * ti.Vector([i, j, k])
                for s in ti.static(range(self.num_boxes)):
                    if self.static_box_min[s][0] <= gp[0] <= self.static_box_max[s][0]:
                        if self.static_box_min[s][1] <= gp[1] <= self.static_box_max[s][1]:
                            if self.static_box_min[s][2] <= gp[2] <= self.static_box_max[s][2]:
                                new_momentum = ti.Vector.zero(float, 3)

                                # vnormal = self.static_plane_n[s].dot(self.grid_v[i, j, k])
                                # if vnormal <= 0.0:
                                #     self.grid_v[i, j, k] -= vnormal * self.static_plane_n[s]

                # dynamic rigid objects
                flag, projected_v = self.droResponce(gp, new_momentum / self.grid_m[i, j, k], 0.0)
                self.grid_v[i, j, k] = projected_v
                self.grid_a[i, j, k] = ( projected_v * self.grid_m[i, j, k] - old_momentum ) / ( self.grid_m[i, j, k] * self.dt )
                #self.grid_v[i, j, k] = new_momentum / self.grid_m[i, j, k]
                #self.grid_a[i, j, k] = ( new_momentum - old_momentum ) / ( self.grid_m[i, j, k] * self.dt )


        #print('After particle grid update (', self.iteration[None], '): ')
        #self.check_particle_consistency()
        #self.check_grid_consistency()

        # g2p and update deformation status
        for p in self.particle_x:
            # base = self.linearBase(self.particle_x[p])
            # stencil = self.linearStencil()
            base = self.uGIMPBase(self.particle_x[p])
            stencil = self.uGIMPStencil()

            v_pic = ti.Vector.zero(float, 3)
            grid_a = ti.Vector.zero(float, 3)
            vel_grad = ti.Matrix.zero(float, 3, 3)

            self.particle_B[p] = ti.Matrix.zero(real, 3, 3)

            # compute velocity gradient and particle velocity
            for i, j, k in ti.static(stencil):
                offset = ti.Vector([i, j, k])
                # grid point position
                gp = self.grid_min + (base + offset).cast(float) * self.dx

                # compute weight and weight grad
                # weight, weight_grad = self.linearWeightAndGrad(self.particle_x[p], gp)
                weight, weight_grad = self.uGIMPWeightAndGrad(self.particle_x[p], gp)

                vel_grad += self.grid_v[base + offset].outer_product(weight_grad)
                v_pic += weight * self.grid_v[base + offset]
                grid_a += weight * self.grid_a[base + offset]
                self.particle_B[p] += weight * self.grid_v[base + offset].outer_product(gp - self.particle_x[p])

            self.particle_v[p] = v_pic

            # elastic prediction
            be = self.particle_be[p] + self.dt * (vel_grad @ self.particle_be[p] + self.particle_be[p] @ vel_grad.transpose())

            #plastic correction
            J = ti.sqrt(be.determinant())
            be_bar = be * pow(J, -2.0/3.0)
            Ie_bar = be_bar.trace() / 3.0
            __mu__ = Ie_bar * self.mu
            s_trial = self.mu * (be_bar - ti.Matrix.identity(float, 3) * Ie_bar)
            s_trial_len = s_trial.norm()

            sigma_Y = self.hb_sigmaY * ti.sqrt(2.0 / 3.0)
            phi_trial = s_trial_len - sigma_Y
            #self.particle_color_f[p] = 0.0
            #self.particle_color_f[p] = s_trial_len * 100.0
            if phi_trial > 0.0:
                new_s = 0.0

                ## [Fei+ 2019, A Multi-Scale Model for Coupling Strands with Shear-Dependent Liquid (37)]
                if self.hb_eta == 0.0:
                    new_s = sigma_Y
                elif self.hb_n == 1.0:
                    new_s = sigma_Y + phi_trial * ti.exp(-2.0 * __mu__ * self.dt / self.hb_eta)
                else:
                    new_s = sigma_Y + ti.pow( ti.pow(phi_trial, (self.hb_n - 1.0) / self.hb_n) - 2.0 * __mu__ * self.dt * (1.0 - 1.0 / self.hb_n) * ti.pow(self.hb_eta, -1.0 / self.hb_n), self.hb_n / (self.hb_n - 1.0))

                # [Continuum Foam]
                #if self.hb_eta == 0.0 or self.hb_n == 1.0:
                #    c = self.hb_eta / ( 2.0 * __mu__ * self.dt )
                #    new_s = ( sigma_Y + c * s_trial_len ) / ( 1.0 + c )
                #else:
                #    Ms = s_trial_len
                #    ms = sigma_Y
                #    eps = 1.0e-6
                #    abs_eps = 1.0e-8
                #    eps_s = 1.0e-6
                #    inv_eta = 1.0 / self.hb_eta
                #    inv_n = 1.0 / self.hb_n

                #    while True:
                #        s = (Ms + ms) * 0.5
                #        v = self.hb_eval( s, s_trial_len, inv_eta, inv_n, sigma_Y, __mu__ )
                #        if ti.abs( v ) < eps * s_trial_len or ti.abs( v ) < abs_eps or ti.abs( Ms - ms ) < eps_s:
                #            new_s = s
                #            break
                #        if v > 0.0:
                #            Ms = s
                #        else:
                #            ms = s

                n = ti.Matrix.zero(float, 3, 3)
                if s_trial_len > 0.0:
                    n = s_trial * (1.0 / s_trial_len)
                s = new_s * n
                new_be_bar = s / self.mu + ti.Matrix.identity(float, 3) * Ie_bar
                be = new_be_bar * pow(J, 2.0/3.0) * ti.pow(new_be_bar.determinant(), -1.0/3.0)
            self.particle_be[p] = be

            # for inclusions
            incremental_deformation_gradient = ti.Matrix.identity(float, 3) + vel_grad * self.dt
            frot, fsym = ti.polar_decompose(be)
            self.particle_rot[p] = frot @ self.particle_rot[p]

            # advect
            self.particle_x[p] += self.dt * v_pic

            # boundary conditions
            for s in ti.static(range(self.num_boxes)):
                if self.static_box_min[s][0] <= self.particle_x[p][0] <= self.static_box_max[s][0]:
                    if self.static_box_min[s][1] <= self.particle_x[p][1] <= self.static_box_max[s][1]:
                        if self.static_box_min[s][2] <= self.particle_x[p][2] <= self.static_box_max[s][2]:
                            self.particle_v[p] = ti.Vector.zero(float, 3)
                            #self.particle_C[p] = ti.Matrix.zero(float, 3, 3)
                            #self.particle_B[p] = ti.Matrix.zero(float, 3, 3)

            # dynamic rigid objects
            #dummy_B = ti.Matrix.zero(float, 3, 3)
            #flag, self.particle_v[p], self.particle_B[p] = self.droResponce(self.particle_x[p], self.particle_v[p], self.particle_B[p], 0.0)
            flag, self.particle_v[p] = self.droResponce(self.particle_x[p], self.particle_v[p], 0.0)

        #print('After g2p and update deformation status (', self.iteration[None], '): ')
        #self.check_particle_consistency()
        #self.check_grid_consistency()

        # for debug
        #for p in self.particle_x:
        #    _c = ti.min(255, ti.max(0, int(256 * self.particle_color_f[p] * 10.0)))
        #    c = 256 * 256 * _c + 256 * _c + _c
        #    self.particle_color[p] = c

        # for debug
        #for i in range(self.cell_count[0]*self.cell_count[1]):
        #    gi = i % self.cell_count[0]
        #    gj = i // self.cell_count[0]
        #    I = ti.Vector([gi, gj])
        #    _c = ti.min(255, ti.max(0, int(256 * self.grid_m[I] * 10000.0)))
        #    c = 256 * 256 * _c + 256 * _c + _c
        #    self.grid_color[i] = c

    @ti.func
    def builtInSinFunction(self, t):
        amplitude = self.motion_params[0]
        frequency = self.motion_params[1]
        phase = self.motion_params[2]
        orientation = ti.Vector([self.motion_params[3], self.motion_params[4], self.motion_params[5]])

        v = amplitude * ti.sin(2.0 * math.pi * frequency * (t + phase))
        dvdt = 2.0 * math.pi * frequency * amplitude * ti.cos(2.0 * math.pi * frequency * (t + phase))
        displacement = orientation * v
        velocity = orientation * dvdt
        return displacement, velocity

    @ti.func
    def builtInStaticFunction(self, t):
        displacement = ti.Vector.zero(real, 3)
        velocity = ti.Vector.zero(real, 3)
        return displacement, velocity

    @ti.func
    def builtInSeroriDipFunction(self, t):
        startT = 0.0 #in_Params.val[4]
        T1 = 3.0/4.0
        T2 = 2.0/3.0 + T1
        T3 = 1.5/4.0 + T2
        endT = 2.0/1.0 + T3 #in_Params.val[5]
        theta_1 = -30.0 #in_Params.val[5]
        theta_2 = 0.0 #in_Params.val[5]
        theta_3 = 60.0 #in_Params.val[5]
        theta_4 = 90.0 #in_Params.val[5]
        length1 = 3.0
        length2 = 2.0
        length3 = 1.5
        length4 = 2.0
        orientation1 = ti.Vector( [ti.cos( math.pi * theta_1 / 180.0 ), ti.sin( math.pi * theta_1 / 180.0 ), 0.0] )
        orientation2 = ti.Vector( [ti.cos( math.pi * theta_2 / 180.0 ), ti.sin( math.pi * theta_2 / 180.0 ), 0.0] )
        orientation3 = ti.Vector( [ti.cos( math.pi * theta_3 / 180.0 ), ti.sin( math.pi * theta_3 / 180.0 ), 0.0] )
        orientation4 = ti.Vector( [ti.cos( math.pi * theta_4 / 180.0 ), ti.sin( math.pi * theta_4 / 180.0 ), 0.0] )

        v = 0.0
        x = 0.0
        vel = 0.0

        displacement = ti.Vector.zero(real, 3)
        velocity = ti.Vector.zero(real, 3)

        if t <= startT:
            v = 0.0
            x = 0.0
            displacement = ti.Vector.zero(real, 3)
            velocity = v * orientation1
        elif t <= T1:
            vel = length1 / (T1 - startT)
            v = vel
            x = vel * (t - startT)
            displacement = x * orientation1
            velocity = v * orientation1
        elif t <= T2:
            vel = length2 / (T2 - T1)
            v = vel
            x = vel * (t - T1)
            displacement = x * orientation2 + length1 * orientation1
            velocity = v * orientation2
        elif t <= T3:
            vel = length3 / (T3 - T2)
            v = vel
            x = vel * (t - T2)
            displacement =  x * orientation3 + length2 * orientation2 + length1 * orientation1
            velocity = v * orientation3
        elif t <= endT:
            vel = length4 / (endT - T3)
            v = vel
            x = vel * (t - T3)
            displacement =  x * orientation4 + length3 * orientation3 + length2 * orientation2 + length1 * orientation1
            velocity = v * orientation4
        else:
            v = 0.0
            displacement = length4 * orientation4 + length3 * orientation3 + length2 * orientation2 + length1 * orientation1
            velocity = v * orientation3

        return displacement, velocity

    @ti.func
    def droSetTime(self, t):
        for r in ti.static(range(self.num_dynamic_rigid_objects)):
            displacement = ti.Vector.zero(real, 3)
            velocity = ti.Vector.zero(real, 3)

            if self.dro_motion_function_type[r][0] == MF_STATIC:
                displacement, velocity = self.builtInStaticFunction(t)
                #print('type = ', self.dro_motion_function_type[r][0])
            elif self.dro_motion_function_type[r][0] == MF_SERORI:
                displacement, velocity = self.builtInSeroriDipFunction(t)
            else:
                print('unsupported built_in_motion_function, type = ', self.dro_motion_function_type[r][0])

            #print('displacement ', displacement)
            #print('velocity ', velocity)

            #print(displacement)
            self.dro_center[r][0] = self.dro_initial_center[r][0] + displacement
            self.dro_translation[r][0] = displacement
            self.dro_velocity[r][0] = velocity
            self.dro_time[r][0] = t

    @ti.func
    def droResponce(self, x, v, boundary_offset):
        flag = 0
        ret_vel = ti.Vector.zero(real, 3)
        for r in ti.static(range(self.num_dynamic_rigid_objects)):
            #small_value = ti.Vector([0.000001, 0.000001, 0.000001])
            #x0 = x - ( self.dro_translation[r][0] + small_value )
            x0 = x - self.dro_translation[r][0]

            #droComputeDistAndNormalInitialFrame(self, idx, x0, boundary_offset):
            cell_idx_x = int(ti.floor( ( x0[0] - self.dro_grid_start[r][0] ) / self.dro_cell_delta[r][0] ) )
            cell_idx_y = int(ti.floor( ( x0[1] - self.dro_grid_start[r][1] ) / self.dro_cell_delta[r][1] ) )
            cell_idx_z = int(ti.floor( ( x0[2] - self.dro_grid_start[r][2] ) / self.dro_cell_delta[r][2] ) )

            outside = 1 if (cell_idx_x < 0) or (cell_idx_x > self.dro_grid_dimension[r][0] - 2) or (cell_idx_y < 0) or (cell_idx_y > self.dro_grid_dimension[r][1] - 2) or (cell_idx_z < 0) or (cell_idx_z > self.dro_grid_dimension[r][2] - 2) else 0

            cell_idx_x = ti.max(0, ti.min(self.dro_grid_dimension[r][0]-2, cell_idx_x))
            cell_idx_y = ti.max(0, ti.min(self.dro_grid_dimension[r][1]-2, cell_idx_y))
            cell_idx_z = ti.max(0, ti.min(self.dro_grid_dimension[r][2]-2, cell_idx_z))

            bc_x = ( x0[0] - ( self.dro_grid_start[r][0] + float(cell_idx_x) * self.dro_cell_delta[r][0] ) ) / self.dro_cell_delta[r][0]
            bc_y = ( x0[1] - ( self.dro_grid_start[r][1] + float(cell_idx_y) * self.dro_cell_delta[r][1] ) ) / self.dro_cell_delta[r][1]
            bc_z = ( x0[2] - ( self.dro_grid_start[r][2] + float(cell_idx_z) * self.dro_cell_delta[r][2] ) ) / self.dro_cell_delta[r][2]

            bc = ti.Vector([bc_x, bc_y, bc_z])
            val = self.trilinearInterpolation( self.dro_sdf_vals[r][cell_idx_z, cell_idx_y, cell_idx_x], self.dro_sdf_vals[r][cell_idx_z, cell_idx_y, cell_idx_x+1], self.dro_sdf_vals[r][cell_idx_z, cell_idx_y+1, cell_idx_x], self.dro_sdf_vals[r][cell_idx_z, cell_idx_y+1, cell_idx_x+1], self.dro_sdf_vals[r][cell_idx_z+1, cell_idx_y, cell_idx_x], self.dro_sdf_vals[r][cell_idx_z+1, cell_idx_y, cell_idx_x+1], self.dro_sdf_vals[r][cell_idx_z+1, cell_idx_y+1, cell_idx_x], self.dro_sdf_vals[r][cell_idx_z+1, cell_idx_y+1, cell_idx_x+1], bc ) - boundary_offset
            n = self.trilinearInterpolation( self.dro_sdf_normals[r][cell_idx_z, cell_idx_y, cell_idx_x], self.dro_sdf_normals[r][cell_idx_z, cell_idx_y, cell_idx_x+1], self.dro_sdf_normals[r][cell_idx_z, cell_idx_y+1, cell_idx_x], self.dro_sdf_normals[r][cell_idx_z, cell_idx_y+1, cell_idx_x+1], self.dro_sdf_normals[r][cell_idx_z+1, cell_idx_y, cell_idx_x], self.dro_sdf_normals[r][cell_idx_z+1, cell_idx_y, cell_idx_x+1], self.dro_sdf_normals[r][cell_idx_z+1, cell_idx_y+1, cell_idx_x], self.dro_sdf_normals[r][cell_idx_z+1, cell_idx_y+1, cell_idx_x+1], bc )

            with_in_object = 1 if (val < 0.0) and (outside == 0) else 0

            n = n.normalized()

            if flag == 0:
                flag = 1 if with_in_object != 0 else 0
                ret_vel = self.dro_velocity[r][0] if with_in_object != 0 else v

        return flag, ret_vel

    #@ti.func
    #def droComputeDistAndNormalInitialFrame(self, idx, x0, boundary_offset):
    #    cell_idx_x = int(ti.floor( ( x0[0] - self.dro_grid_start[idx][0] ) / self.dro_cell_delta[idx][0] ) )
    #    cell_idx_y = int(ti.floor( ( x0[1] - self.dro_grid_start[idx][1] ) / self.dro_cell_delta[idx][1] ) )
    #    cell_idx_z = int(ti.floor( ( x0[2] - self.dro_grid_start[idx][2] ) / self.dro_cell_delta[idx][2] ) )

    #    cell_idx_x = ti.max(0, ti.min(self.dro_grid_dimension[idx][0]-2, cell_idx_x))
    #    cell_idx_y = ti.max(0, ti.min(self.dro_grid_dimension[idx][1]-2, cell_idx_y))
    #    cell_idx_z = ti.max(0, ti.min(self.dro_grid_dimension[idx][2]-2, cell_idx_z))

    #    bc_x = ( x0[0] - ( self.dro_grid_start[idx][0] + float(cell_idx_x) * self.dro_cell_delta[idx][0] ) ) / self.dro_cell_delta[idx][0]
    #    bc_y = ( x0[1] - ( self.dro_grid_start[idx][1] + float(cell_idx_y) * self.dro_cell_delta[idx][1] ) ) / self.dro_cell_delta[idx][1]
    #    bc_z = ( x0[2] - ( self.dro_grid_start[idx][2] + float(cell_idx_z) * self.dro_cell_delta[idx][2] ) ) / self.dro_cell_delta[idx][2]

    #    bc = ti.Vector([bc_x, bc_y, bc_z])
    #    v = trilinearInterpolation( self.dro_sdf_vals[idx][cell_idx_z, cell_idx_y, cell_idx_x], self.dro_sdf_vals[idx][cell_idx_z, cell_idx_y, cell_idx_x+1], self.dro_sdf_vals[idx][cell_idx_z, cell_idx_y+1, cell_idx_x], self.dro_sdf_vals[idx][cell_idx_z, cell_idx_y+1, cell_idx_x+1], self.dro_sdf_vals[idx][cell_idx_z+1, cell_idx_y, cell_idx_x], self.dro_sdf_vals[idx][cell_idx_z+1, cell_idx_y, cell_idx_x+1], self.dro_sdf_vals[idx][cell_idx_z+1, cell_idx_y+1, cell_idx_x], self.dro_sdf_vals[idx][cell_idx_z+1, cell_idx_y+1, cell_idx_x+1], bc ) - boundary_offset
    #    n = trilinearInterpolation( self.dro_sdf_normals[idx][cell_idx_z, cell_idx_y, cell_idx_x], self.dro_sdf_normals[idx][cell_idx_z, cell_idx_y, cell_idx_x+1], self.dro_sdf_normals[idx][cell_idx_z, cell_idx_y+1, cell_idx_x], self.dro_sdf_normals[idx][cell_idx_z, cell_idx_y+1, cell_idx_x+1], self.dro_sdf_normals[idx][cell_idx_z+1, cell_idx_y, cell_idx_x], self.dro_sdf_normals[idx][cell_idx_z+1, cell_idx_y, cell_idx_x+1], self.dro_sdf_normals[idx][cell_idx_z+1, cell_idx_y+1, cell_idx_x], self.dro_sdf_normals[idx][cell_idx_z+1, cell_idx_y+1, cell_idx_x+1], bc )

    #    n = n.normalized()

    #    return v, n

    def saveState(self, file_name):
        f = open(file_name, 'wb')
        f.write(ctypes.c_int32(self.particle_count))
        self.particle_x.to_numpy()[0:self.particle_count].astype(np.float32).flatten().tofile(f)
        hls = np.ones(self.particle_count, np.float32) * self.particle_hl
        hls.flatten().tofile(f)
        self.particle_v.to_numpy()[0:self.particle_count].astype(np.float32).flatten().tofile(f)
        id = np.ones(self.particle_count, ctypes.c_int32)
        id.flatten().tofile(f)
        self.particle_rot.to_numpy()[0:self.particle_count].astype(np.float32).flatten().tofile(f)
        # rot will be saved in row-major order

    def saveHDF5(self, file_name):
        print('[AGTaichiMPM3D] saving state to ' + file_name)
        with h5py.File(file_name, 'w') as h5:
            h5.create_group('mpm3d')
            h5['mpm3d'].create_dataset('timestep', data=self.dt)
            h5['mpm3d'].create_dataset('iteration', data=self.iteration[None])
            h5['mpm3d'].create_dataset('time', data=float(self.iteration[None])*self.dt)
            h5['mpm3d'].create_dataset('dx', data=self.dx)
            h5['mpm3d'].create_dataset('hl', data=self.particle_hl)
            h5['mpm3d'].create_dataset('m', data=self.particle_mass)
            h5['mpm3d'].create_dataset('Vp', data=self.particle_volume)
            h5['mpm3d'].create_dataset('q', data=self.particle_x.to_numpy())
            h5['mpm3d'].create_dataset('v', data=self.particle_v.to_numpy())
            h5['mpm3d'].create_dataset('be', data=self.particle_be.to_numpy())

if len(sys.argv) <= 1:
    print("usage: python AGTaichiMPM.py <xml file>")
    exit(-1)

xmlData = xmlParser.MPMXMLData(sys.argv[1])
xmlData.show()

agTaichiMPM = AGTaichiMPM(xmlData)
agTaichiMPM.initialize()

gui = ti.GUI("AGTaichiMPM3D")

def T(a):
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0]*0.1+0.1, a[:, 1]*0.1, a[:, 2]*0.1
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5

prev_auto_save_time = -100000.0
save_count = -1

while True:
    if gui.get_event():
        if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
            agTaichiMPM.initialize()

    # for i in range(100):
        # agTaichiMPM.dbg_dynamic_rigid_step()
    # the usual simulation steps:
    for i in range(100):
        agTaichiMPM.step()
        #agTaichiMPM.dbg_dynamic_rigid_step()
        time = agTaichiMPM.iteration[None] * agTaichiMPM.dt

        if time >= prev_auto_save_time + 1.0 / agTaichiMPM.auto_save_fps:
            save_count += 1
            file_name = agTaichiMPM.auto_save_filename % save_count
            agTaichiMPM.saveState(file_name)
            print('[AGTaichiMPM3D] auto save %s' % file_name)
            prev_auto_save_time = time

        if time >= agTaichiMPM.max_time:
            agTaichiMPM.saveState('result.dat')
            print('[AGTaichiMPM3D] simulation done.')
            exit(0)

    gui.circles(T(agTaichiMPM.particle_x.to_numpy()), radius=2, color=0xFFFFFF)
    #gui.circles(agTaichiMPM.particle_x.to_numpy(), radius=2, color=agTaichiMPM.particle_color.to_numpy())
    #gui.circles(agTaichiMPM.grid_pos.to_numpy(), radius=2, color=agTaichiMPM.grid_color.to_numpy())
    gui.show()
