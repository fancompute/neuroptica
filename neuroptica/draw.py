import SchemDraw.elements as e
import numpy as np

wg_width = 5.0
wg_height = -2.0
wg1_y = 0
wg2_y = wg_height
wg_x1 = 0
wg_x2 = wg_width
box_width = 1.25
box_height = 0.75


def box(x, y, w, h):
    return [[x - w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2], [x + w / 2, y - h / 2]]


mzi_curve_x = np.linspace(wg_x1, wg_x2, 100)
mzi_curve_y = np.concatenate([np.zeros(15), .45 * (np.cos(np.linspace(0, 2 * np.pi, 20)) - 1), np.zeros(15)] * 2)
mzi_curve_1 = np.vstack((mzi_curve_x, mzi_curve_y + wg1_y)).T
mzi_curve_2 = np.vstack((mzi_curve_x, -1 * mzi_curve_y + wg2_y)).T

MZI_DRAWING = {
    'name': 'MZI',
    'paths': [mzi_curve_1, mzi_curve_2],
    'shapes': [{'shape': 'poly',
                'xy': box(wg_width / 2, wg1_y, box_width, box_height),
                'closed': True,
                'fill': True,
                'fillcolor': 'white'},
               {'shape': 'poly',
                'xy': box(0, wg1_y, box_width, box_height),
                'closed': True,
                'fill': True,
                'fillcolor': 'white'}],
    'anchors': {
        'in1': [wg_x1, wg1_y],
        'in2': [wg_x1, wg2_y],
        'out1': [wg_x2, wg1_y],
        'out2': [wg_x2, wg2_y],
        'theta': [wg_width / 2, wg1_y],
        'phi': [0, wg1_y],
    },
    'extend': False,
}

MZI_DRAWING_INV = {
    'name': 'MZI_inv',
    'paths': [mzi_curve_1, mzi_curve_2],
    'shapes': [{'shape': 'poly',
                'xy': box(wg_width / 2, wg1_y, box_width, box_height),
                'closed': True,
                'fill': True,
                'fillcolor': 'white'},
               {'shape': 'poly',
                'xy': box(wg_width, wg1_y, box_width, box_height),
                'closed': True,
                'fill': True,
                'fillcolor': 'white'}],
    'anchors': {
        'in1': [wg_x1, wg1_y],
        'in2': [wg_x1, wg2_y],
        'out1': [wg_x2, wg1_y],
        'out2': [wg_x2, wg2_y],
        'theta': [wg_width / 2, wg1_y],
        'phi': [wg_width, wg1_y],
    },
    'extend': False,
}


def addMZI(d, V_theta, V_phi, xy, color='black', label='', inverted=False):
    if inverted:
        MZI_DRAWING_INV['labels'] = [
            {'label': '{:1.2f}V'.format(float(V_theta)), 'pos': MZI_DRAWING_INV['anchors']['theta']},
            {'label': '{:1.2f}V'.format(float(V_phi)), 'pos': MZI_DRAWING_INV['anchors']['phi']}]
        return d.add(MZI_DRAWING_INV, xy=xy, color=color, label=label)
    else:
        MZI_DRAWING['labels'] = [{'label': '{:1.2f}V'.format(float(V_theta)), 'pos': MZI_DRAWING['anchors']['theta']},
                                 {'label': '{:1.2f}V'.format(float(V_phi)), 'pos': MZI_DRAWING['anchors']['phi']}]
        return d.add(MZI_DRAWING, xy=xy, color=color, label=label)


PHASE_SHIFTER = {
    'name': 'phaseShifter',
    'paths': [[[0, 0], [3, 0]]],
    'shapes': [{'shape': 'poly',
                'xy': box(1.5, 0, box_width, box_height),
                'closed': True,
                'fill': True,
                'fillcolor': 'white'}],
    'anchors': {
        'in': [0, 0],
        'out': [3, 0],
        'phi': [1.5, 0],
    },
    'extend': False,
}


def addPhaseShifter(d, V_phi, xy, color='black', label=''):
    PHASE_SHIFTER['labels'] = [{'label': '{:1.2f}V'.format(float(V_phi)), 'pos': PHASE_SHIFTER['anchors']['phi']}]
    return d.add(PHASE_SHIFTER, xy=xy, color=color, label=label)


# Opamp
_oa_back = 1.75
_oa_xlen = _oa_back * np.sqrt(3) / 2
_oa_lblx = _oa_xlen / 8
_oa_pluslen = .2
GAIN = {
    'name': 'GAIN',
    'paths': [[[0, 0], [1, 0], [1, _oa_back / 2], [1 + _oa_xlen, 0], [1, -_oa_back / 2], [1, 0],
               [np.nan, np.nan], [1 + _oa_xlen, 0], [2 + _oa_xlen, 0]]],
    'anchors': {'center': [1 + _oa_xlen / 2, 0],
                'in': [0, 0],
                'out': [2 + _oa_xlen, 0]},
    'extend': False,
}


def addGain(d, gain, xy, color='black', label=''):
    GAIN['labels'] = [{'label': '{:3.1f}$\\times$'.format(float(gain)), 'pos': GAIN['anchors']['center']}]
    return d.add(GAIN, xy=xy, color=color, label=label)


def drawBox(d, outputs, label=''):
    ''' Draw a dotted box around the SMU element '''
    xmin = min([output[0][0] for output in outputs]) + .5
    xmax = max([output[-1][0] for output in outputs]) - .5
    ymin = min([output[-1][1] for output in outputs]) - 1
    ymax = max([output[-1][1] for output in outputs]) + 1
    right = d.add(e.LINE, xy=[xmax, ymax], d='down', toy=ymin, ls=':')
    bot = d.add(e.LINE, d='left', tox=xmin, ls=':')
    left = d.add(e.LINE, d='up', toy=ymax, ls=':')
    top = d.add(e.LINE, d='right', tox=xmax, ls=':')
    top.add_label(label, loc='center', align=('center', 'bottom'), size=36)


relu_x = np.linspace(-.3, .3, num=25)
relu_y = np.maximum(relu_x, 0)
RELU = {
    'name': 'RELU',
    'base': e.SOURCE,
    'paths': [np.vstack((relu_x + .5, relu_y)).T]
}

RELU2 = {
    'name': 'RELU',
    'paths': [[[0, 0], [2, 0]]],
    'shapes': [{'shape': 'circle',
                'center': [1, 0],
                'radius': 0.75,
                'fill': True,
                'fillcolor': 'white'}],
    'anchors': {'center': [1, 0],
                'in': [0, 0],
                'out': [2, 0]},
    'extend': False,
}


def addActivation(d, activationType, xy, color='black', label=''):
    RELU2['labels'] = [{'label': activationType, 'pos': RELU2['anchors']['center']}]
    return d.add(RELU2, xy=xy, color=color, label=label)
