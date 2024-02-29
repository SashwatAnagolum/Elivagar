from braket.aws import AwsDevice


def get_braket_device(device_name):
    device_names_list = ['ionq_device', 'oqc_lucy', 'rigetti_aspen_m_3']
    device_arns_list = ['arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
                  'arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy',
                  'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3']
    
    return AwsDevice(device_arns_list[device_names_list.index(device_name)])