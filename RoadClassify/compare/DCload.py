import scipy.io as scio

def dcload():
    # DC = 'WVUSUB'  # 1408
    # DC = 'WVUINTER'  # 1640
    # DC = 'ChinaCity'
    # DC = 'UDDS'
    DC = 'WVUCITY'

    data_path = '../Data_Standard Driving Cycles/Standard_'+DC+'.mat'
    data = scio.loadmat(data_path)
    car_spd_one = data['speed_vector']
    return car_spd_one[0]*2.237  # m/s->mph

# a = dcload()
# print(len(a))