from PIL import Image
from context import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time

gain_rx = 20
gain_tx = 0
animamat = 1
animamat2 = 0

choise_matrix = 5
count_zero = 0

def gif_to_matrices(gif_path):

    img = Image.open(gif_path)
    frames = []  
    try:
        while True:    
            grayscale_frame = img.convert('L')
            width, height = grayscale_frame.size
            frame_matrix = []
            for y in range(height):
                row = []
                for x in range(width):
                    
                    pixel = grayscale_frame.getpixel((x, y))
                    row.append(pixel)
                frame_matrix.append(row)
            frames.append(frame_matrix)
            img.seek(img.tell() + 1)
    except EOFError:  
        pass
    return frames

def form_ofdm(data_mat):
    
    zeros = np.zeros(count_zero)
    data_ofdm = np.zeros(0)

    for j in range(len(data_mat)):    # номер матрицы
        for i in range(len(data_mat[0])):    #  номер строки
            data_and_zero = np.concatenate([zeros, data_mat[j][i], zeros])
            data_ifft = np.fft.ifft(data_and_zero)
            data_ofdm = np.concatenate([data_ofdm, data_ifft])

    return data_ofdm

def add_pss_in_signal(ofdm_tx, N_fft):
    pss = zadoff_chu(PSS=True)

    # Частотное представление (внутри OFDM)
    pss_freq = np.zeros(N_fft, dtype=complex)
    half = N_fft // 2

    # левая половина
    pss_freq[half-31:half] = pss[:31]

    # правая половина
    pss_freq[half+1:half+32] = pss[31:]

    # во временную область
    pss_time = np.fft.ifft(pss_freq)*2**10

    # сцепляем
    return np.concatenate([pss_time, ofdm_tx])

if choise_matrix == 1:
    gif_path = 'path'  
elif choise_matrix == 2:
    gif_path = 'path' 
elif choise_matrix == 3:
    gif_path = 'path' 
elif choise_matrix == 4:
    gif_path = 'path'
elif choise_matrix == 5:
    gif_path = 'path'

start = time.time()
matrices = gif_to_matrices(gif_path)
end =  time.time()
print("Convert gif to matrix: ", end - start)

colm = len(matrices[0][0])
rows = len(matrices[0])

count_frame = len(matrices)

print("colm : ", colm)
print("rows : ", rows)
print("count_frame : ", count_frame)


start = time.time()
if choise_matrix == 1:
    # ofdm = complex_cpp_file("path")
    # ofdm = form_ofdm(matrices)
    # np.savetxt("path",ofdm)
    ofdm = np.loadtxt("path", dtype=complex)

elif choise_matrix == 2:
    # ofdm = complex_cpp_file("path")
    # ofdm = form_ofdm(matrices)
    # np.savetxt("path",ofdm)
    ofdm = np.loadtxt("path", dtype=complex)

elif choise_matrix == 3:
    # ofdm = complex_cpp_file("path")
    # ofdm = form_ofdm(matrices)
    # np.savetxt("path",ofdm)
    ofdm = np.loadtxt("path", dtype=complex)

else:
    ofdm = form_ofdm(matrices)

end =  time.time()
print("OFDM convert: ", end - start)

# ofdm = add_pss_in_signal(ofdm, 128)

ofdm = ofdm * 2**7

# ofdm1 = corr_pss_time(ofdm, 128)
# data_ofdm = ofdm1[:rows*colm]
# data_ofdm = data_ofdm.reshape(rows,colm)
# print(len(data_ofdm))
# data = np.fft.fft(data_ofdm, axis=1)

# plt.figure()
# plt.imshow(abs(data), cmap='jet',interpolation='nearest', aspect='auto')
# plt.show()

# Animation
if animamat:
    sdr  = standart_settings("ip:192.168.2.1", 2e6, rows*colm)
    sdr2 = standart_settings("ip:192.168.2.1", 2e6, rows*colm)

    tx_signal(sdr,1900e6,0,ofdm)
    rx = rx_signal(sdr2,1900e6,gain_rx,1)
    # rx = corr_pss_time(rx, 128)
    rx = rx[:rows*colm]
    
    rx = rx.reshape(rows,colm)

    rx = np.fft.fft(rx)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(abs(rx), cmap='jet',interpolation='nearest', aspect='auto')

    def update(frame):
        rx = rx_signal(sdr2,1900e6,gain_rx,1)
        # rx = corr_pss_time(rx, 128)
        # rx = calculate_correlation(120, rx, 1e6/120) 
        rx = rx[:rows*colm]
        rx = rx.reshape(rows,colm)
        rx = np.fft.fft(rx)

        heatmap.set_array(abs(rx))
        return [heatmap]
    
    
    ani = animation.FuncAnimation(fig, update, frames=100, interval=200)

plt.show()

