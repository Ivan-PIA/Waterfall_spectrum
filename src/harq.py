import numpy as np



def CRC_RX(data):
    G = [1,0,1,0,0,1,1,1,0,1,0,0,0,1,0,1,1]
    for i in range(0,len(data)-16):
        if(data[i] == 1):
            for j in range(len(G)):
                data[i+j] = data[i+j] ^ G[j]
    crc = data[len(data)-16:]

    return np.asarray(crc)


def ACK(crc,number_slot):
    error = 0
    for i in range(len(crc)):
        if crc[i] != 0:
            error += 1

    if error == 0:
        return -1
    else:
        return number_slot          

    