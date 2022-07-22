from PymoNNto import *
#from Old.Input_Behaviours.Images.Helper import *
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def rotatearoundpoint(p, rot, middle):
    aa = p[0] - middle[0]
    bb = p[1] - middle[1]
    cc = np.sqrt(aa * aa + bb * bb)
    a = 0
    if cc != 0:
        a = np.rad2deg(np.arcsin(aa / cc))
    if bb < 0:
        a = 180 - a
    a = np.deg2rad(a + rot)
    aa = np.sin(a) * cc
    bb = np.cos(a) * cc
    return (middle[0] + aa, middle[1] + bb)

def picture_to_array(image, max=1):
    pil_image_gray = image.convert('L')
    result = np.array(pil_image_gray).astype(np.float64)
    return result/np.maximum(max, np.max(image))

def get_Input_Mask(neurons, patterns):
    Input_Mask = neurons.get_neuron_vec()

    for pattern in patterns:
        Input_Mask += pattern.flatten()

    return Input_Mask > 0

def getLinePicture(deg, center_x, center_y, length, width, height):
    im = Image.new('L', (width, height), (0))
    draw = ImageDraw.Draw(im)
    rot_point = rotatearoundpoint((length, 0), deg, (0, 0))
    #x, y = pol2cart(length, np.deg2rad(deg))
    #rot_point=(x,y)
    draw.line((center_x - np.floor(rot_point[0]), center_y - np.floor(rot_point[1]), center_x + np.floor(rot_point[0]), center_y + np.floor(rot_point[1])), fill=255)
    return picture_to_array(im)

class Line_Patterns(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Input')

        self.strength=self.get_init_attr('strength', 1.0)

        center_x = self.get_init_attr('center_x', 1)
        center_y = self.get_init_attr('center_y', 1)
        degree = self.get_init_attr('degree', 1)
        line_length = self.get_init_attr('line_length', 1)

        self.random_order = self.get_init_attr('random_order', True)

        pattern_count=0
        if type(center_x) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(center_x))
        if type(center_y) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(center_y))
        if type(degree) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(degree))
        if pattern_count == 0: pattern_count=1

        if not type(center_x) in [list, np.ndarray]: center_x=np.ones(pattern_count)*center_x
        if not type(center_y) in [list, np.ndarray]: center_y=np.ones(pattern_count)*center_y
        if not type(degree) in [list, np.ndarray]: degree=np.ones(pattern_count)*degree

        self.patterns = []
        self.labels = []
        for i in range(pattern_count):
            self.patterns.append(np.array([getLinePicture(degree[i], center_x[i], center_y[i], line_length/2, neurons.width, neurons.height)]))
            self.labels.append('deg{}cy{}cx{}'.format(degree[i], center_x[i], center_y[i]))

        neurons.Input_Mask = get_Input_Mask(neurons, self.patterns)

        self.pattern_count = pattern_count
        self.pattern_indx=0

        self.pattern_indices = np.array(np.arange(self.pattern_count))


    def new_iteration(self, neurons):
        if self.random_order:
            if len(self.pattern_indices)==0:
                self.pattern_indices = np.array(np.arange(self.pattern_count))
            self.pattern_indx = np.random.choice(self.pattern_indices)
        else:
            self.pattern_indx += 1
            if self.pattern_indx>=len(self.patterns):
                self.pattern_indx=0

        pattern = self.patterns[self.pattern_indx][0]
        neurons.activity += pattern.flatten()*self.strength
