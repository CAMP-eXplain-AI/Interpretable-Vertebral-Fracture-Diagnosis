import streamlit as st
import numpy as np
import PIL.Image
from st_clickable_images import clickable_images
import os
from monai.transforms import CenterSpatialCrop, ScaleIntensityRange, Orientation
import base64
from io import BytesIO
import torch
from glob import glob
from model import VerseFxClassifier
from netdissect import nethook, imgviz
import tempfile
import nibabel as nib
import pathlib

import warnings
warnings.filterwarnings("ignore")

# inlined Network Dissection results
unit_levels = torch.tensor([1.8715330362319946, 1.5618106126785278, 1.2870054244995117, 2.801919937133789, 1.1172661781311035, 2.2070984840393066, 2.3209457397460938, 2.022796392440796, 2.0127036571502686, 2.782788038253784, 1.013718843460083, 2.491750955581665, 1.5298184156417847, 1.7949274778366089, 2.1840720176696777, 2.73867130279541, 1.9927071332931519, 1.4070216417312622, 1.8516860008239746, 1.4621922969818115, 1.7988444566726685, 2.0956199169158936, 2.890246629714966, 0.9635668992996216, 1.8309086561203003, 1.8866947889328003, 1.8208155632019043, 1.3282618522644043, 2.787090301513672, 1.6975336074829102, 2.388171434402466, 3.1032965183258057, 1.996658444404602, 1.8226428031921387, 2.557448148727417, 1.8223134279251099, 1.2595659494400024, 1.8109630346298218, 2.6617250442504883, 1.9107582569122314, 2.254500389099121, 1.218552827835083, 3.087602376937866, 3.3800148963928223, 3.153672218322754, 2.919377326965332, 2.0350027084350586, 3.0219407081604004, 2.4654042720794678, 1.4958505630493164, 2.2895171642303467, 1.3284631967544556, 3.229510545730591, 1.9460035562515259, 1.855022668838501, 3.15183424949646, 2.582113742828369, 1.8321630954742432, 2.7707386016845703, 2.824443817138672, 2.662318468093872, 2.466081380844116, 1.0707639455795288, 1.856846570968628, 1.9820237159729004, 2.5840156078338623, 1.603718638420105, 2.741654396057129, 1.7408792972564697, 1.5616865158081055, 2.621121406555176, 2.187910318374634, 2.029402494430542, 2.3087165355682373, 2.3417551517486572, 2.4370405673980713, 2.363990545272827, 1.7908833026885986, 2.29636287689209, 2.5254483222961426, 3.2696034908294678, 1.4013628959655762, 1.645676851272583, 2.7126476764678955, 2.717543125152588, 1.0994248390197754, 1.9232852458953857, 1.985698938369751, 2.004666328430176, 2.385585069656372, 2.5118658542633057, 3.444154977798462, 2.0752625465393066, 2.9441027641296387, 1.6907892227172852, 2.695660352706909, 3.08571457862854, 1.8869487047195435, 1.5935581922531128, 2.224071502685547, 2.877380609512329, 3.0157597064971924, 2.1446480751037598, 2.4394376277923584, 3.298722267150879, 2.208728313446045, 1.9590588808059692, 1.789717197418213, 2.6814987659454346, 2.2261674404144287, 3.002722978591919, 3.0650651454925537, 1.9212583303451538, 1.6315948963165283, 1.6328997611999512, 2.4739739894866943, 0.9252153635025024, 3.089088201522827, 2.7511496543884277, 1.997342586517334, 2.5561487674713135, 1.6858017444610596, 2.7134108543395996, 2.513460159301758, 1.8604570627212524, 2.7962076663970947, 1.111690878868103, 2.1877119541168213, 2.1126585006713867, 2.9239501953125, 1.4319941997528076, 3.041599988937378, 2.2168679237365723, 1.792368769645691, 2.1387674808502197, 1.3679250478744507, 1.347702145576477, 3.0506792068481445, 1.5423274040222168, 1.8090440034866333, 1.869529366493225, 2.8993425369262695, 1.5416679382324219, 3.003296375274658, 3.1893210411071777, 2.3816075325012207, 2.281187057495117, 2.7733864784240723, 1.3033744096755981, 1.4627212285995483, 1.942519187927246, 1.4943166971206665, 2.48635196685791, 1.9112900495529175, 2.908750534057617, 1.9310427904129028, 1.8946770429611206, 1.2220033407211304, 2.0171048641204834, 1.197824478149414, 2.093484878540039, 2.240743398666382, 1.4367271661758423, 1.5200153589248657, 2.6623482704162598, 2.34277606010437, 2.378328323364258, 3.4981000423431396, 1.8303442001342773, 2.1322264671325684, 1.8304965496063232, 2.0963211059570312, 1.932998776435852, 0.9879118800163269, 1.989233136177063, 2.0391933917999268, 3.078193187713623, 2.9010426998138428, 1.451486587524414, 1.4458937644958496, 3.3362858295440674, 0.8172016143798828, 2.8464856147766113, 2.3619463443756104, 2.0269312858581543, 1.87027108669281, 2.5867714881896973, 1.0947588682174683, 2.485373020172119, 1.4596120119094849, 2.9054574966430664, 2.267271041870117, 1.9901957511901855, 1.708791971206665, 1.5335347652435303, 3.0039384365081787, 1.581254482269287, 1.6688708066940308, 2.138035535812378, 1.8489503860473633, 1.463232398033142, 2.745103597640991, 1.7890992164611816, 3.209639310836792, 2.186699628829956, 1.384399175643921, 2.347090482711792, 2.911564350128174, 2.7910614013671875, 3.0139355659484863, 2.8508076667785645, 3.1651434898376465, 2.020735263824463, 1.3879002332687378, 1.347353458404541, 1.3600330352783203, 1.563052773475647, 2.427166223526001, 2.3583383560180664, 2.0502967834472656, 1.0467418432235718, 1.5168964862823486, 2.550285816192627, 2.2569706439971924, 1.280961275100708, 2.153566360473633, 0.8621286749839783, 1.5903816223144531, 1.6175390481948853, 1.2808561325073242, 2.129512310028076, 1.923080563545227, 2.4000368118286133, 2.7758114337921143, 2.756497859954834, 2.8936665058135986, 1.9632121324539185, 1.4698351621627808, 2.9193220138549805, 2.2707347869873047, 2.1808905601501465, 2.915626049041748, 2.199504852294922, 2.225417375564575, 1.8788528442382812, 1.6902912855148315, 2.703303098678589, 1.6111797094345093, 1.4749184846878052, 2.7335896492004395, 1.1770113706588745, 1.5911366939544678, 2.5799360275268555, 2.450134515762329, 1.584707498550415, 2.0303263664245605, 1.5416966676712036, 1.6474940776824951, 3.166107654571533, 1.8914194107055664, 2.731400489807129, 3.456698179244995, 3.1407928466796875, 2.657524585723877, 1.8312366008758545, 1.3835384845733643, 1.3457938432693481, 1.1902421712875366, 1.739147663116455, 2.8404054641723633, 1.5782982110977173, 1.4647060632705688, 1.3077998161315918, 1.8057410717010498, 1.1732816696166992, 1.4494800567626953, 2.1183741092681885, 3.5306854248046875, 2.348907470703125, 1.5650557279586792, 1.6930912733078003, 2.298933267593384, 1.1758023500442505, 1.6107817888259888, 1.3251513242721558, 2.080108404159546, 1.862548589706421, 3.099520206451416, 2.8438494205474854, 1.6832661628723145, 2.074307680130005, 2.0457262992858887, 2.8403425216674805, 3.117814540863037, 2.058823823928833, 2.234037160873413, 1.2487999200820923, 1.7322322130203247, 2.6813132762908936, 2.924269199371338, 1.7503197193145752, 3.2688212394714355, 1.8045146465301514, 3.1042702198028564, 2.327272891998291, 2.7761642932891846, 2.3101589679718018, 2.8489952087402344, 2.132847547531128, 1.554833173751831, 1.3879495859146118, 1.8847209215164185, 1.728200912475586, 1.6019946336746216, 3.04852294921875, 3.0847041606903076, 2.528338670730591, 2.277801275253296, 3.1020517349243164, 2.7520859241485596, 3.03950834274292, 1.8526620864868164, 2.6675875186920166, 2.201525926589966, 1.3852479457855225, 1.744421362876892, 2.172621488571167, 2.681896924972534, 2.4530863761901855, 2.0969560146331787, 1.3115235567092896, 2.049104928970337, 1.7683310508728027, 1.7026116847991943, 2.3060457706451416, 3.208275318145752, 2.6523375511169434, 1.7658361196517944, 1.9047954082489014, 2.9763565063476562, 1.834631323814392, 3.142353057861328, 1.9534238576889038, 1.7625831365585327, 2.1041769981384277, 1.945776104927063, 2.970412015914917, 1.8245426416397095, 1.4031907320022583, 1.3985518217086792, 2.8565142154693604, 1.8306998014450073, 2.6509435176849365, 1.452415108680725, 2.7498743534088135, 2.0770175457000732, 1.8407188653945923, 1.5940998792648315, 2.4943857192993164, 3.0113513469696045, 3.450936794281006, 1.2603273391723633, 1.5098024606704712, 1.647451400756836, 2.344951868057251, 2.499359369277954, 1.9027211666107178, 1.6656138896942139, 1.5507005453109741, 2.177579641342163, 1.4274533987045288, 2.7495903968811035, 1.4635711908340454, 2.0104260444641113, 2.4939937591552734, 2.069014072418213, 1.3013184070587158, 3.4216034412384033, 1.9525243043899536, 2.196475028991699, 2.7452564239501953, 2.1965861320495605, 2.8216114044189453, 2.2089548110961914, 2.936760902404785, 1.3354514837265015, 1.3799076080322266, 2.2054338455200195, 1.3158196210861206, 1.084631085395813, 2.4761247634887695, 1.4672796726226807, 1.7008095979690552, 1.5144485235214233, 1.7634273767471313, 2.5879948139190674, 2.024614095687866, 1.7365692853927612, 1.5214873552322388, 1.1093666553497314, 1.7518495321273804, 2.188833713531494, 3.439579963684082, 2.6817214488983154, 1.636168122291565, 2.1104257106781006, 3.0666251182556152, 3.1396965980529785, 1.7993018627166748, 1.897646427154541, 1.2042944431304932, 2.8433687686920166, 2.068439483642578, 2.4039862155914307, 1.3701140880584717, 1.262689471244812, 1.827138066291809, 2.1528568267822266, 3.259542465209961, 1.7049492597579956, 1.9919352531433105, 2.1563854217529297, 2.035381317138672, 3.0388429164886475, 1.8345075845718384, 2.22445011138916, 1.5946440696716309, 2.3479206562042236, 1.281639575958252, 1.4048471450805664, 1.0306495428085327, 1.05494225025177, 1.9470269680023193, 1.6934491395950317, 2.1934640407562256, 2.6225905418395996, 1.974666714668274, 3.4361391067504883, 1.148988127708435, 2.7689907550811768, 2.478999614715576, 2.292860984802246, 1.380311131477356, 1.8914124965667725, 1.251215934753418, 1.3892083168029785, 3.1640305519104004, 2.3226025104522705, 2.3283731937408447, 3.2135708332061768, 1.2665305137634277, 2.8611419200897217, 2.735239267349243, 1.348517894744873, 1.2256826162338257, 2.5687448978424072, 1.9984424114227295, 2.913726568222046, 1.79617440700531, 3.3642163276672363, 1.405514121055603, 1.7745602130889893, 2.080112934112549, 2.5899147987365723, 1.9730525016784668, 1.6167746782302856, 1.2985221147537231, 1.6463950872421265, 1.2983338832855225, 3.4439616203308105, 1.8814938068389893, 1.1827762126922607, 3.0138072967529297, 2.0302090644836426, 3.2060086727142334, 1.7749220132827759, 1.6361336708068848, 2.207552194595337, 3.1703994274139404, 2.6205763816833496, 2.2056334018707275, 1.3571845293045044, 2.4915218353271484, 1.3841928243637085, 1.9503673315048218, 1.6178065538406372, 3.2435460090637207, 1.1473424434661865, 2.2226922512054443, 1.9872846603393555, 2.009683132171631, 3.1938722133636475, 3.248166799545288, 2.4461867809295654, 1.8230010271072388, 2.1673691272735596, 2.776118278503418, 2.054086685180664, 1.6877385377883911, 2.3526558876037598, 2.648297071456909, 1.3525688648223877, 2.819364309310913, 2.9533910751342773, 1.636002540588379, 1.5173200368881226, 2.315584421157837, 1.5832545757293701, 3121535301208496, 1.679909348487854, 2.9136874675750732, 2.4349215030670166])
corr_rank = {299: 98, 194: 401, 281: 441, 1: 419, 289: 227, 65: 268, 23: 101, 453: 418, 321: 362, 259: 431, 257: 446, 477: 17, 92: 497, 17: 234, 314: 60, 331: 354, 315: 123, 318: 192, 445: 233, 238: 240, 311: 489, 265: 7, 126: 22, 431: 254, 223: 10, 179: 14, 362: 230, 448: 78, 478: 199, 418: 197, 139: 249, 403: 111, 262: 206, 316: 282, 150: 34, 142: 12, 444: 288, 261: 147, 180: 205, 413: 455, 322: 11, 5: 400, 474: 198, 167: 204, 226: 257, 230: 304, 225: 406, 482: 222, 373: 77, 352: 164, 2: 299, 329: 136, 152: 74, 271: 420, 386: 369, 377: 196, 165: 13, 229: 106, 457: 170, 192: 466, 99: 440, 214: 316, 461: 511, 19: 430, 82: 337, 505: 344, 199: 457, 416: 329, 484: 390, 104: 83, 496: 210, 465: 301, 462: 484, 361: 91, 231: 317, 100: 32, 467: 477, 63: 310, 421: 320, 273: 387, 368: 307, 449: 220, 385: 328, 8: 336, 432: 168, 217: 182, 255: 436, 366: 461, 151: 487, 67: 159, 14: 503, 36: 27, 131: 131, 112: 300, 37: 470, 29: 391, 163: 449, 510: 212, 21: 303, 202: 402, 409: 277, 158: 315, 16: 464, 44: 173, 197: 479, 410: 35, 495: 366, 341: 163, 425: 124, 77: 296, 38: 67, 175: 498, 181: 252, 248: 396, 15: 65, 301: 149, 18: 172, 81: 372, 154: 237, 306: 504, 123: 176, 105: 408, 185: 456, 145: 338, 398: 99, 40: 331, 451: 184, 108: 57, 94: 425, 213: 6, 286: 30, 203: 120, 39: 404, 64: 463, 10: 126, 348: 241, 28: 295, 278: 207, 216: 216, 224: 263, 330: 421, 303: 52, 206: 395, 417: 166, 354: 291, 228: 264, 446: 154, 509: 346, 440: 385, 85: 201, 363: 313, 121: 393, 423: 232, 277: 162, 86: 188, 483: 411, 364: 505, 374: 287, 176: 251, 78: 415, 351: 374, 227: 414, 51: 39, 250: 368, 488: 363, 288: 253, 434: 501, 68: 323, 276: 469, 469: 115, 130: 248, 168: 333, 397: 428, 365: 25, 128: 214, 3: 185, 26: 447, 307: 183, 419: 417, 222: 133, 493: 157, 382: 4, 319: 193, 60: 40, 327: 416, 433: 424, 430: 62, 345: 460, 486: 155, 35: 18, 45: 267, 407: 112, 507: 375, 141: 23, 260: 96, 338: 492, 387: 454, 189: 85, 182: 58, 282: 191, 350: 153, 323: 427, 143: 179, 472: 26, 302: 361, 0: 378, 244: 379, 426: 355, 215: 334, 140: 281, 253: 318, 489: 494, 267: 305, 111: 33, 346: 152, 390: 139, 210: 105, 212: 273, 188: 413, 239: 258, 439: 208, 391: 308, 378: 422, 312: 382, 97: 224, 491: 63, 162: 359, 389: 265, 272: 97, 443: 386, 75: 118, 245: 297, 263: 148, 284: 107, 137: 178, 415: 399, 209: 161, 201: 42, 335: 90, 135: 73, 310: 432, 122: 29, 172: 405, 328: 478, 173: 215, 308: 459, 480: 491, 412: 388, 494: 36, 476: 246, 479: 9, 9: 64, 344: 383, 119: 332, 55: 174, 103: 202, 395: 506, 56: 28, 73: 512, 353: 326, 120: 94, 339: 218, 12: 266, 193: 16, 295: 458, 106: 495, 287: 217, 304: 228, 124: 499, 148: 250, 422: 483, 264: 46, 113: 3, 166: 103, 369: 327, 156: 2, 498: 321, 334: 151, 169: 442, 506: 48, 4: 352, 43: 158, 249: 135, 31: 127, 233: 134, 427: 356, 375: 510, 107: 409, 183: 144, 320: 89, 138: 465, 211: 189, 41: 358, 292: 51, 79: 235, 456: 389, 116: 130, 280: 306, 343: 342, 473: 438, 357: 351, 511: 93, 450: 209, 279: 451, 144: 236, 293: 108, 187: 256, 11: 171, 127: 5, 471: 302, 313: 319, 13: 330, 468: 380, 291: 340, 243: 480, 475: 261, 487: 294, 160: 41, 254: 481, 429: 8, 59: 213, 326: 150, 57: 142, 125: 247, 359: 298, 87: 70, 258: 95, 70: 69, 383: 486, 347: 325, 497: 493, 69: 223, 129: 271, 492: 156, 384: 219, 91: 160, 360: 137, 74: 117, 285: 398, 340: 473, 240: 121, 424: 467, 266: 397, 508: 140, 178: 50, 400: 474, 388: 259, 235: 59, 420: 433, 376: 243, 294: 452, 232: 353, 402: 231, 49: 371, 317: 341, 408: 423, 372: 276, 48: 102, 102: 269, 256: 472, 428: 167, 171: 439, 275: 349, 435: 245, 84: 175, 134: 203, 71: 116, 161: 410, 324: 194, 342: 496, 207: 43, 242: 360, 25: 284, 218: 239, 499: 37, 195: 145, 83: 187, 186: 238, 283: 286, 247: 75, 490: 407, 464: 345, 436: 384, 164: 488, 219: 365, 118: 49, 251: 445, 503: 211, 394: 448, 190: 226, 153: 272, 62: 79, 170: 290, 96: 82, 80: 143, 305: 53, 157: 412, 191: 364, 332: 350, 7: 507, 101: 72, 399: 335, 437: 169, 333: 502, 447: 475, 401: 462, 290: 482, 252: 24, 370: 109, 438: 55, 452: 125, 241: 312, 297: 429, 184: 15, 93: 81, 337: 476, 463: 221, 349: 229, 208: 44, 61: 289, 136: 177, 355: 242, 20: 500, 356: 76, 381: 47, 33: 370, 296: 275, 274: 426, 117: 84, 442: 490, 458: 260, 34: 186, 236: 86, 481: 468, 309: 373, 269: 88, 455: 113, 500: 255, 66: 471, 58: 434, 174: 119, 396: 110, 268: 87, 504: 322, 155: 435, 298: 122, 200: 181, 6: 485, 72: 129, 76: 508, 109: 392, 204: 339, 46: 347, 237: 274, 196: 367, 454: 293, 110: 21, 90: 19, 502: 38, 325: 453, 52: 128, 95: 71, 159: 225, 441: 278, 459: 92, 42: 56, 405: 403, 485: 200, 54: 444, 205: 285, 32: 66, 30: 357, 336: 61, 234: 190, 379: 244, 393: 309, 146: 394, 404: 292, 221: 348, 147: 180, 115: 270, 53: 314, 24: 443, 371: 165, 246: 146, 411: 31, 88: 1, 133: 324, 380: 138, 22: 280, 98: 20, 460: 114, 114: 262, 27: 141, 466: 343, 501: 279, 358: 195, 470: 381, 50: 376, 132: 311, 270: 437, 220: 45, 47: 68, 89: 80, 149: 132, 367: 283, 406: 100, 392: 450, 177: 104, 198: 509, 414: 54, 300: 377}

# inlined and adapted pytorch_grad_cam/activations_and_gradients.py
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output[0]
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

# inlined and adapted pytorch_grad_cam/grad_cam.py
class DetectorGradCAM:
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model,
            target_layer, reshape_transform)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self, input_tensor, target_category, activations, grads, k=5):
        a = torch.tensor(activations)
        return torch.topk((a * (a > unit_levels.view(unit_levels.shape[0], 1, 1, 1).repeat(1, 8, 8, 8)))[0].sum(dim=(1,2,3)), k=k).indices

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, input_tensor, target_category, activations, grads, eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, k=5):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        return self.get_cam_weights(input_tensor, target_category, activations, grads, k=k).tolist()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# hide header bar for print
hide_streamlit_style = """
<style>
header {visibility:hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

received_input = None

scale = ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True)
crop = CenterSpatialCrop(roi_size=(64,64,64))

preprocess = lambda arr: scale(crop(arr[None, ...].clip(-1000, 1000)))
to_image = lambda v: PIL.Image.fromarray((255*v[0,:,:,v.shape[-1]//2]).astype('uint8')).convert('RGB')

def to_base64(image: PIL.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_slice(path: str):
    return to_base64(to_image(preprocess(np.load(path))))

def bundle_builder(path: str, local=True, desc=""):
    if local: path = os.path.join(".", path)

    vertebra = np.float32(preprocess(np.load(path)))
    slice = to_base64(to_image(vertebra))

    return (slice, desc, vertebra)

examples = [
        bundle_builder("examples/l5.npy", desc="L5 - no fracture"),
]

with st.empty():
    with st.container():
        upload = st.file_uploader("Upload vertebra to classify (nii, nii.gz, npy)")

        if upload is not None:
            suffix = ''.join(pathlib.Path(upload.name).suffixes)
            with tempfile.NamedTemporaryFile(suffix=suffix) as fp:
                fp.write(upload.getvalue())
                fp.seek(0)

                if 'nii' in suffix:
                    try:
                        nii = nib.load(fp.name)
                    except:
                        raise Exception("Unable to load uploaded NIfTI file. Please ensure that it has the correct file extensions.")

                    nifti_data = nii.get_fdata()
                    data = Orientation(axcodes='IPL')(nifti_data[None, ...], affine=nii.affine)[0][0]
                elif 'npy' in suffix:
                    try:
                        data = np.load(fp)
                    except:
                        raise Exception("Unable to load provided NumPy file.")
                else:
                    raise Exception("Invalid input data format. Please provide a NIfTI or NumPy array file.")

                assert len(data.shape) == 3, "Invalid number of dimensions. Expects three-dimensional input."
                assert all([a >= 64 for a in data.shape]), "Invalid shape. Shape must not be smaller than 64x64x64."

                fp.close()

                vertebra = np.float32(preprocess(data))
                slice = to_base64(to_image(vertebra))

                received_input = (slice, upload.name, vertebra)

        with st.container():
            st.caption("Or pick one of these examples:")

            clicked = clickable_images(
                [ex[0] for ex in examples],
                titles=[ex[1] for ex in examples],
                div_style={"display": "flex", "justify-content": "left", "flex-wrap": "wrap"},
                img_style={"margin": "0 5px 5px 0", "height": "135px"},
            )

            if clicked > -1:
                received_input = examples[clicked]

    if received_input is not None:
        with st.container():
            col1, col2 = st.columns([1,3])
            with col1:
                st.image(received_input[0], width=140)
            with col2:
                top_container = st.container()
                top_container.write("**Concept Visualization**")
                top_container.write(f"Input: {received_input[1]}")
            with st.spinner('Running inference'):
                saved_checkpoint = "moonlit-flower-278.ckpt"

                # TODO inline config
                checkpoint = torch.load(saved_checkpoint, map_location="cpu")

                checkpoint['hyper_parameters']['dataset_path'] = '.'
                checkpoint['hyper_parameters']['batch_size'] = 1

                module = VerseFxClassifier.load_from_checkpoint(saved_checkpoint, hparams=checkpoint['hyper_parameters'], map_location="cpu")
                model = module.backbone

                model.eval()

                sample = torch.tensor(received_input[2][None, ...])

                cam = DetectorGradCAM(model, model.down_tr512, use_cuda=False)

                detectors = cam.forward(input_tensor=sample, target_category=0, k=5)
                ranks = [corr_rank[unit] for unit in detectors]

                model = nethook.InstrumentedModel(model)
                model.retain_layer("down_tr512")

                pred = (torch.sigmoid(model(sample)) > 0.5).long().item()

                acts = model.retained_layer("down_tr512")[0]

                ld_res = acts.shape[-1]
                img_slices = torch.linspace(int(64/ld_res/2), 64-int(64/ld_res/2), ld_res, dtype=torch.long)

                iv = imgviz.ImageVisualizer(224, image_size=64, source="zc", percent_level=0.99)

            top_container.write(f"Prediction: {'fracture' if pred==1 else 'no fracture'}")

            image_margin = """
            <style>
            img{margin-right:5px}*/
            </style>
            """
            st.markdown(image_margin, unsafe_allow_html=True)

            for i, detector in enumerate(detectors):
                def paper_typo_fix(d):
                    # in the paper, unit 424 is mistakenly referred to as unit 22.
                    # to ensure consistency, we simply swap the label of both
                    if d != 424 and d!= 22: return str(d)
                    if d == 424: return "22"
                    else: return "424"

                st.markdown(f"Detector unit #{paper_typo_fix(detector)} (relevance rank {i+1}, positive correlation rank {ranks[i]})")
                concepts = glob(f"concepts/{detector}_*.png")
                if len(concepts) == 0:
                    st.caption("No statistically significant activations, unable to show general concept")
                else:
                    st.caption("General concept")
                    sorted_concepts = sorted(concepts, key=lambda x: int(x.replace('.png', '').split('/')[-1].split('_')[1]))
                    st.image([to_base64(PIL.Image.open(c)) for c in sorted_concepts], width=75)
                activations = [to_base64(PIL.Image.fromarray(iv.pytorch_masked_image(
                                (sample[0, ..., img_slices[slice]]).repeat(3, 1, 1),
                                acts[..., slice],
                                detector,
                                level=unit_levels[detector]).permute(1,2,0).cpu().numpy())) for slice in range(0, ld_res)]
                st.caption("Image-specific activation")
                st.image(activations, width=75)

                st.markdown('<div style="margin-top:20px;border-top: 1px solid rgba(49, 51, 63, 0.2);margin-bottom:40px"></div>', unsafe_allow_html=True)

            def on_click(*args, **kwargs):
                # force reload of the page to reset internal state
                st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

            st.button("Reset", on_click=on_click)