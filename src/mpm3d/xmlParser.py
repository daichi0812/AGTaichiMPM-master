# https://docs.python.org/ja/3/library/xml.etree.elementtree.html

import numpy as np
import xml.etree.ElementTree as ET

MF_PRESCRIBED_STRAIGHT_MOTION_FUNCTION = 1
MF_ROTATION = 2
MF_ROTATIONMOVE = 3
MF_STATIC = 4
MF_MIXSPOON = 5
MF_MIXSPOONLEAN = 6
MF_MIXPLATE = 7
MF_SERORI = 8
MF_MIXSPOONLEANMOVE = 9
MF_MIXPLATEMOVE = 10
MF_ROTCUBE1 = 11
MF_ROTCUBE2 = 12
MF_BLENDER_HANE = 13
MF_BLENDER_BOWL = 14
MF_SPATURA = 15
MF_SPATURANOMIX = 16

class MPMXMLIntegratorData:
    def __init__(self):
        self.dt = 0.0
        self.bulk_modulus = 1.0
        self.shear_modulus = 1.0
        self.herschel_bulkley_power = 1.0
        self.eta = 0.0
        self.yield_stress = 0.0
        self.flip_pic_alpha = 0.95
        self.max_time = 0.0

    def setFromXMLNode(self, node):
        self.dt = float(node.attrib['dt'])
        self.bulk_modulus = float(node.attrib['bulk_modulus'])
        self.shear_modulus = float(node.attrib['shear_modulus'])
        self.herschel_bulkley_power = float(node.attrib['herschel_bulkley_power'])
        self.eta = float(node.attrib['eta'])
        self.yield_stress = float(node.attrib['yield_stress'])
        self.flip_pic_alpha = float(node.attrib['flip_pic_alpha'])
        self.max_time = float(node.attrib['max_time'])

    def show(self):
        print('*** Integrator ***')
        print('  dt: ' + str(self.dt))
        print('  bulk_modulus: ' + str(self.bulk_modulus))
        print('  shear_modulus: ' + str(self.shear_modulus))
        print('  herschel_bulkley_power: ' + str(self.herschel_bulkley_power))
        print('  eta: ' + str(self.eta))
        print('  yield_stress: ' + str(self.yield_stress))
        print('  flip_pic_alpha: ' + str(self.flip_pic_alpha))
        print('  max_time: ' + str(self.max_time))

class MPMXMLAutoSaveData:
    def __init__(self):
        self.fps = 50.0
        self.filename = "template_%010d.dat"

    def setFromXMLNode(self, node):
        self.fps = float(node.attrib['fps'])
        self.filename = node.attrib['filename']

    def show(self):
        print('*** Auto Save ***')
        print('  fps: ' + str(self.fps))
        print('  filename: ' + self.filename)

class MPMXMLGridData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.cell_width = 1.0

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.cell_width = float(node.attrib['cell_width'])

    def show(self):
        print('*** Grid ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  cell_width: ' + str(self.cell_width))

class MPMXMLPointFileData:
    def __init__(self, dim):
        self.file_name = ""
        self.start_point = np.zeros(dim)
        self.density = 0.0
        self.velocity = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.file_name = node.attrib['filename']
        self.start_point = np.array([float(e) for e in node.attrib['start_point'].split(' ')])
        self.density = float(node.attrib['density'])
        self.velocity = np.array([float(e) for e in node.attrib['velocity'].split(' ')])

    def show(self):
        print('*** Point File ***')
        print('  file_name: ' + self.file_name)
        print('  start_point: ' + str(self.start_point))
        print('  density: ' + str(self.density))
        print('  velocity: ' + str(self.velocity))

class MPMXMLCuboidData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.density = 1.0
        self.cell_samples_per_dim = 2
        self.vel = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.density = float(node.attrib['density'])
        self.cell_samples_per_dim = int(node.attrib['cell_samples_per_dim'])
        self.vel = np.array([float(e) for e in node.attrib['vel'].split(' ')])

    def show(self):
        print('*** Cuboid ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  density: ' + str(self.density))
        print('  cell_samples_per_dim: ' + str(self.cell_samples_per_dim))
        print('  vel: ' + str(self.vel))

# class MPMXMLStaticPlaneData:
#     def __init__(self, dim):
#         self.x = np.zeros(dim)
#         self.n = np.zeros(dim)
#         self.isSticky = False
#
#     def setFromXMLNode(self, node):
#         self.x = np.array([float(e) for e in node.attrib['x'].split(' ')])
#         self.n = np.array([float(e) for e in node.attrib['n'].split(' ')])
#         if node.attrib['boundary_behavior'] == 'sticking':
#             self.isSticky = True
#         else:
#             self.isSticky = False
#
#     def show(self):
#         print('*** Static plane ***')
#         print('  x: ' + str(self.x))
#         print('  n: ' + str(self.n))
#         print('  isSticky: ' + str(self.isSticky))

class MPMXMLStaticBoxData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.isSticky = False

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        if node.attrib['boundary_behavior'] == 'sticking':
            self.isSticky = True
        else:
            self.isSticky = False

    def show(self):
        print('*** Static box ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  isSticky: ' + str(self.isSticky))

class MPMXMLNearEarthGravityData:
    def __init__(self, dim):
        self.g = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.g = np.array([float(e) for e in node.attrib['f'].split(' ')])

    def show(self):
        print('*** Near earth gravity ***')
        print('  g: ' + str(self.g))

class MPMXMLDynamicRigidObjectData:
    def __init__(self, dim):
        self.file_name = ""
        self.velocity = np.zeros(dim)
        self.motion_function_type = MF_STATIC
        self.isSticky = False
        self.start_point = np.zeros(dim)
        self.params = np.zeros(12)

    def setFromXMLNode(self, node):
        self.file_name = node.attrib['filename']
        self.velocity = np.array([float(e) for e in node.attrib['velocity'].split(' ')])

        motion_function_str = node.attrib['motion_function']
        if motion_function_str == 'straight':
            self.motion_function_type = MF_PRESCRIBED_STRAIGHT_MOTION_FUNCTION
        elif motion_function_str == 'rotation':
            self.motion_function_type = MF_ROTATION
        elif motion_function_str == 'rotationmove':
            self.motion_function_type = MF_ROTATIONMOVE
        elif motion_function_str == 'static':
            self.motion_function_type = MF_STATIC
        elif motion_function_str == 'mixpoon':
            self.motion_function_type = MF_MIXSPOON
        elif motion_function_str == 'mixpoonlean':
            self.motion_function_type = MF_MIXSPOONLEAN
        elif motion_function_str == 'mixplate':
            self.motion_function_type = MF_MIXPLATE
        elif motion_function_str == 'dip':
            self.motion_function_type = MF_SERORI
        elif motion_function_str == 'mixpoonleanmove':
            self.motion_function_type = MF_MIXSPOONLEANMOVE
        elif motion_function_str == 'mixplatemove':
            self.motion_function_type = MF_MIXPLATEMOVE
        elif motion_function_str == 'rotcube1':
            self.motion_function_type = MF_ROTCUBE1
        elif motion_function_str == 'rotcube2':
            self.motion_function_type = MF_ROTCUBE2
        elif motion_function_str == 'blenderhane':
            self.motion_function_type = MF_BLENDER_HANE
        elif motion_function_str == 'blenderbowl':
            self.motion_function_type = MF_BLENDER_BOWL
        elif motion_function_str == 'spatura':
            self.motion_function_type = MF_SPATURA
        elif motion_function_str == 'spaturanomix':
            self.motion_function_type = MF_SPATURANOMIX
        else:
            print('unknown motion function string: ' + motion_function_str)
            exit(-1)

        if node.attrib['boundary_behavior'] == 'sticking':
            self.isSticky = True
        else:
            self.isSticky = False
        self.start_point = np.array([float(e) for e in node.attrib['start_point'].split(' ')])

        if 'param0' in node.attrib:
            self.params[0] = float(node.attrib['param0'])
        if 'param1' in node.attrib:
            self.params[1] = float(node.attrib['param1'])
        if 'param2' in node.attrib:
            self.params[2] = float(node.attrib['param2'])
        if 'param3' in node.attrib:
            self.params[3] = float(node.attrib['param3'])
        if 'param4' in node.attrib:
            self.params[4] = float(node.attrib['param4'])
        if 'param5' in node.attrib:
            self.params[5] = float(node.attrib['param5'])
        if 'param6' in node.attrib:
            self.params[6] = float(node.attrib['param6'])
        if 'param7' in node.attrib:
            self.params[7] = float(node.attrib['param7'])
        if 'param8' in node.attrib:
            self.params[8] = float(node.attrib['param8'])
        if 'param9' in node.attrib:
            self.params[9] = float(node.attrib['param9'])
        if 'param10' in node.attrib:
            self.params[10] = float(node.attrib['param10'])
        if 'param11' in node.attrib:
            self.params[11] = float(node.attrib['param11'])

    def show(self):
        print('*** Dynamic Rigid Object ***')
        print('  file_name: ' + self.file_name)
        print('  velocity: ' + str(self.velocity))
        print('  motion_function_type: ' + str(self.motion_function_type))
        print('  isSticky: ' + str(self.isSticky))
        print('  start_point: ' + str(self.start_point))
        print('  params: ' + str(self.params))

class MPMXMLData:
    def __init__(self, file_name):
        print('[AGTaichiMPM3D] Parsing xml file: ' + str(file_name))
        tree = ET.parse(file_name)
        root = tree.getroot()
        if root.tag != 'AGTaichiMPM3D':
            print('[AGTaichiMPM3D] Could not find root note AGTaichiMPM3D. Exiting...')
            exit(-1)

        integrator = root.find('integrator')
        self.integratorData = MPMXMLIntegratorData()
        self.integratorData.setFromXMLNode(integrator)

        autoSave = root.find('auto_save')
        self.autoSaveData = MPMXMLAutoSaveData()
        self.autoSaveData.setFromXMLNode(autoSave)

        grid = root.find('grid')
        self.gridData = MPMXMLGridData(3)
        self.gridData.setFromXMLNode(grid)

        cuboid = root.find('cuboid')
        if cuboid is None:
            self.cuboidData = None
        else:
            self.cuboidData = MPMXMLCuboidData(3)
            self.cuboidData.setFromXMLNode(cuboid)

        point_file = root.find('point_file')
        if point_file is None:
            self.pointFileData = None
        else:
            self.pointFileData = MPMXMLPointFileData(3)
            self.pointFileData.setFromXMLNode(point_file)

#         self.staticPlaneList = []
#         for static_plane in root.findall('static_plane'):
#             staticPlaneData = MPMXMLStaticPlaneData(3)
#             staticPlaneData.setFromXMLNode(static_plane)
#             self.staticPlaneList.append(staticPlaneData)

        self.staticBoxList = []
        for static_box in root.findall('static_box'):
            staticBoxData = MPMXMLStaticBoxData(3)
            staticBoxData.setFromXMLNode(static_box)
            self.staticBoxList.append(staticBoxData)

        self.dynamicRigidObjectList = []
        for dynamic_rigid_object in root.findall('dynamic_rigid_object'):
            dynamicRigidObjectData = MPMXMLDynamicRigidObjectData(3)
            dynamicRigidObjectData.setFromXMLNode(dynamic_rigid_object)
            self.dynamicRigidObjectList.append(dynamicRigidObjectData)

        nearEarthGravity = root.find('near_earth_gravity')
        self.nearEarthGravityData = MPMXMLNearEarthGravityData(3)
        self.nearEarthGravityData.setFromXMLNode(nearEarthGravity)

    def show(self):
        print('[AGTaichiMPM3D] XML Data:')
        self.integratorData.show()
        self.autoSaveData.show()
        self.gridData.show()

        count = 0

        if self.cuboidData is not None:
            self.cuboidData.show()
            count += 1

        if self.pointFileData is not None:
            self.pointFileData.show()
            count += 1

        if count == 0:
            print("Did not find any specification for particle initialization. Use either cuboid or point_file.")
            exit(-1)

        if count != 1:
            print("Unsupported particle initialization. Should use only one of cuboid or point_file, not a combination of them.")
            exit(-1)

        self.nearEarthGravityData.show()

        for sp in self.staticBoxList:
            sp.show()

        for dro in self.dynamicRigidObjectList:
            dro.show()
