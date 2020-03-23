#!/usr/bin/env python3

#
# ...   imports
#

# misc
import numpy as np
import torch.nn

# mspec
from mspec.demunic.networks.common import SINetworkBase
from mspec.demunic.networks.resnets import ResNetFixup
from mspec.demunic.models.utils import instantiate_network


class DirectApproximation(SINetworkBase):
    def __init__(self):
        super().__init__()

    def _create_model(self):
        T = self.config['principal_axis']
        T_inv = T.transpose()
        cmf = self.config['cmf']

        A_np = T_inv @ np.linalg.inv(cmf @ T_inv)
        self.A = torch.nn.Parameter(torch.ones(self.n_output, self.n_input))
        self.A.data = torch.FloatTensor(A_np.T)
        self.A.requires_grad = False

        if torch.cuda.is_available():
            self.A.data = self.A.data.cuda()

    def forward(self, x):
        rgb = x.permute(0, 2, 3, 1)

        # apply linear mapping
        spec_pred = torch.matmul(rgb, self.A)
        return spec_pred.permute((0,3,1,2))

    def _define_config(self):
        # for backward compatibility
        cmf_def = "[[0.41817228817366264, 1.8542544891172272, 4.474841967859706, 6.886033661972714, 8.397145158219766, 8.111969500858363, 6.614556589748532, 4.2806546763138895, 1.7617124499074843, 0.35388741028610976, 0.08350447425499596, 0.8198362494663061, 2.5766688519527046, 5.175067248699752, 8.244797634696893, 11.59403605257534, 15.432222052412284, 19.22736473441391, 22.192583231249138, 24.47626429342193, 24.595962792943638, 22.549709284666733, 18.738110794321354, 14.168341570351016, 9.443861488680778, 5.871769410998117, 3.3386034140818204, 1.7782032706094146, 0.8939267759161398, 0.43636996133677963, 0.20956822049349733], [0.0438328465601226, 0.19147894855339928, 0.4677850832007497, 0.8457788732584823, 1.357519265572107, 1.9562518069658388, 2.8035395938207337, 4.04979288289219, 5.545563623196434, 7.416266589739603, 10.0764156552752, 13.268402092763594, 16.658340499450865, 19.13938808158624, 21.037054678047905, 21.688139960743243, 21.81014327892271, 20.896310215634742, 19.002120680938724, 17.000535861141547, 14.396807043123964, 11.545660131920778, 8.704835064449572, 6.199513654893653, 3.9325349886318737, 2.353752132211972, 1.3182437754394087, 0.6954211004542621, 0.34781770496078923, 0.16945318370838058, 0.08130070194094503], [1.8821318382719243, 8.520293885316699, 21.28163130785347, 33.99399573912922, 43.048966151913206, 43.651172014068386, 38.193024913873344, 28.83148094989771, 16.89601022225915, 9.086787539367332, 4.781365744645063, 2.451800640236755, 1.3284635060166823, 0.6663434123723664, 0.29926480271926975, 0.08726733205940683, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]"
        pa_def = "[[0.13833635158117522, 0.1383363515811754, 0.1383363515811752, 0.1394392364765867, 0.15386032444163164, 0.16920097988339922, 0.17850789101943382, 0.18198752919508265, 0.18273551963159532, 0.1828330379624523, 0.18332553273638297, 0.18368874907789778, 0.18518898320661542, 0.18915896066604967, 0.19319393614654937, 0.1952557159786154, 0.19514173670270732, 0.19409461105702083, 0.19362755736771248, 0.19329255511972984, 0.19373543442562172, 0.1938645303805135, 0.1916884101455254, 0.18910598346623098, 0.18749622181616124, 0.18605768664711694, 0.18618975985483638, 0.18579529852516916, 0.1794225416188501, 0.17078711242539063, 0.16618076493536757], [-0.22026611321389852, -0.22026611321389916, -0.22026611321389866, -0.220700657505417, -0.22817618645697002, -0.2360002771774404, -0.23459757875860873, -0.22405650844140068, -0.20834676582863926, -0.18918867256796537, -0.16598646390464142, -0.137892518644446, -0.10562739765412868, -0.07103461633269251, -0.0358929943280188, -0.0023067480768921523, 0.029349056603937725, 0.05994078373435737, 0.0887942047834511, 0.11500675693812237, 0.13778363191342072, 0.15620214677843264, 0.17063126342695664, 0.1825816333980164, 0.1923355710323581, 0.19955858654933492, 0.2063132935817465, 0.21257464706402188, 0.21657397639128487, 0.22373012456745356, 0.2378017139976782], [0.16729648885565912, 0.16729648885565984, 0.16729648885565918, 0.16625718577128157, 0.1548435731287623, 0.14069006206978413, 0.12029797549130514, 0.09369453998666166, 0.06015403286277676, 0.016923087834812896, -0.039784454244374164, -0.10845225706561866, -0.18368259689241592, -0.2536324759166103, -0.30188745763917774, -0.32060053630324653, -0.3117759562464418, -0.27849763357496193, -0.2259797512223055, -0.161041576025447, -0.09350583349386334, -0.029212814839789443, 0.027736803575569423, 0.07635391744557246, 0.12096005746024309, 0.16265773900541464, 0.20043810804667486, 0.22671015836134958, 0.2258304585603252, 0.19601638120975676, 0.14862241696121561]]"
        self.add_config_option('cmf', co_type='nparray', co_default=cmf_def)
        self.add_config_option('principal_axis', co_type='nparray', co_default=pa_def)


class SSREstResidual(SINetworkBase):
    def __init__(self):
        super().__init__()

    def _create_model(self):
        self.network_info = self.config['network']

        self.spec_est = torch.nn.Conv2d(self.n_input, self.n_output, kernel_size=3, padding=1, bias=False)
        if self.config['spec_est_ini'] is not None:
            se_np = self.config['spec_est_ini']
            self.spec_est.weight.data = torch.FloatTensor(se_np)

        # residual estimation
        network = instantiate_network(self.network_info['name'], custom_nets=globals())
        network.configure(self.network_info)
        self.spec_residual_network = network

    def _define_config(self):
        network_def = ResNetFixup().get_default_config()

        self.add_config_option('spec_est_ini', co_type='nparray', co_default=None)
        self.add_config_option('network', co_type='str', co_default=network_def)

    def forward(self, x):
        spec_ini = self.spec_est(x)
        spec_res = self.spec_residual_network(x)

        return spec_ini + spec_res


class SSRDirectResidual(SINetworkBase):
    def __init__(self):
        super().__init__()

    def _create_model(self):
        self.de_info = self.config['directE']
        self.network_info = self.config['network']

        network = instantiate_network(self.de_info['name'], custom_nets=globals())
        network.configure(self.de_info)
        self.de_network = network

        network = instantiate_network(self.network_info['name'], custom_nets=globals())
        network.configure(self.network_info)
        self.spec_residual_network = network

    def _define_config(self):
        directE_def = DirectApproximation().get_default_config()
        network_def = ResNetFixup().get_default_config()

        self.add_config_option('directE', co_type='str', co_default=directE_def)
        self.add_config_option('network', co_type='str', co_default=network_def)

    def forward(self, x):
        spec_ini = self.de_network(x)
        spec_res = self.spec_residual_network(x)

        return spec_ini + spec_res