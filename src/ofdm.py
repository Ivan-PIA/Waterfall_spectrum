import numpy as np
import matplotlib.pyplot as plt
from src.base import *
from src.modular import *
from src.demodular import *
from src.plots import *
from src.harq import *


def pss_time(fft_len):

    pss = zadoff_chu(PSS = True)
        
    zeros = fft_len // 2 - 31
    pss_ifft = np.insert(pss, 32, 0)
    pss_ifft = np.insert(pss_ifft, 0, np.zeros(zeros))
    pss_ifft = np.append(pss_ifft, np.zeros(zeros-1))
        
    pss_ifft = np.fft.fftshift(pss_ifft)
    pss_ifft = np.fft.ifft(pss_ifft)

    return pss_ifft

def calculate_correlation(fft_len, matrix_name, m):
    """
    Calculates correlation between pss and matrix_name with filtering and delay.

    Args:
    pss: The reference signal.
    matrix_name: The signal to compare with pss.
    m: Decimation factor.

    Returns:
    A tuple containing the correlation and carrier frequency offset (CFO).
    """
    pss = pss_time(fft_len)
    L = len(pss)
    #print("pss",L)
    # Flipped and complex conjugated reference signal
    corr_coef = np.flip(np.conjugate(pss))

    # Filter reference signal sections
    partA = np.convolve(corr_coef[:L // 2], matrix_name, mode='full')
    xDelayed = np.concatenate((np.zeros(L // 2), matrix_name[:-L // 2]))
    partB = np.convolve(corr_coef[L // 2:], xDelayed, mode='full')

    # Calculate correlation and phase difference
    correlation = np.abs(partA + partB)
    phaseDiff = partA * np.conj(partB)

    # Find maximum correlation and corresponding phase difference
    istart = np.argmax(correlation)
    phaseDiff_max = phaseDiff[istart]

    # Calculate CFO
    CFO = np.angle(phaseDiff_max) / (np.pi * 1 / m)
    t = np.arange(0,len(matrix_name))
    t = t / 1920000

    print("CFO : ", CFO)
    data_offset = matrix_name * np.exp(-1j * 2 * np.pi * np.conjugate(CFO) * t)

    return data_offset


def corr_pss_time(rx, N_fft):
    
    pss = zadoff_chu(PSS=True)

    pss_freq = np.zeros(N_fft, dtype=complex)
    half = N_fft // 2

    pss_freq[half-31:half] = pss[:31]

    pss_freq[half+1:half+32] = pss[31:]

    pss_ifft = np.fft.ifft(pss_freq)

    o = np.abs(np.convolve(np.flip(np.conjugate(pss_ifft)), rx, mode = "full"))
    if 0:
        plt.figure()
        plt.title("Correlation PSS")
        plt.plot(abs(o))
        plt.show()
    indexes_max  =  o / np.max(o)  
    indexes_max = [i for i in range(len(indexes_max)) if indexes_max[i] > 0.80]    

    maxi = indexes_max[0]
        
    rx = rx[maxi:]
    print("maxi = ",maxi)
    return rx

def activ_carriers(N_fft, GB_len, pilot_carriers, pilots = False):
        """
        N_fft - subcarrier

        GB - guard_band_len

        PC - pilot_carriers
        
        Возвращает массив поднесущих на которых имеются данные
        """
        fft_len = N_fft
        GB = GB_len // 2
        PilCar = pilot_carriers

        if pilots:
            activ = np.array([
                    i
                    for i in range(0, fft_len)
                    if (i in range(GB, fft_len - GB - 1))
                    and (i != fft_len/2)
                ])
        else:
            activ = np.array([
                    i
                    for i in range(0, fft_len)
                    if (i in range(GB, fft_len - GB - 1))
                    and (i not in PilCar)
                    and (i != fft_len/2)
                ])
        
        #activ = activ + (self.N_fft / 2)
        
        return activ

def indexs_of_CP_after_PSS(rx, cp, fft_len):
    """
    Возвращает массив начала символов (вместе с CP) (чтобы только символ был нужно index + 16)
    """
    corr = [] # Массив корреляции 

    for i in range(len(rx) - fft_len): # узнать почему - ффт
        o = norm_corr((rx[:cp]), rx[fft_len:fft_len+cp])

        corr.append(abs(o))
        rx = np.roll(rx, -1)
    #print("00000000000000000000000000000000000000000000000000000")
    #print(corr)
    corr = np.array(corr) / np.max(corr) # Нормирование
    max_len_cycle = len(corr)
    # if corr[0] > 0.97:
    #     max_len_cycle = len(corr)
    # else:
    #     max_len_cycle = len(corr)-(fft_len+cp)

    ind = np.argmax(corr[0 : (fft_len+cp)// 2 ])
    arr_index = [] # Массив индексов максимальных значений corr
    arr_index.append(ind)
    for i in range((fft_len+cp) // 2, max_len_cycle, (fft_len+cp)):
        #print(i, i+(fft_len+cp))
        max = np.max(corr[i : i+(fft_len+cp)])
        if max > 0.90: 
            ind = i + np.argmax(corr[i : i+(fft_len+cp)])
            if ind < (len(corr)):
                arr_index.append(ind)
    
    ### DEBUG
    
    # print(corr)
    #plt.figure()
    #plt.plot(abs(corr))
    #plt.show()
    return arr_index

def indiv_symbols(ofdm, N_fft, CP_len):
    cp = CP_len
    all_sym = N_fft + cp
    
    
    index = indexs_of_CP_after_PSS(ofdm, cp, N_fft)
    index = index[:5]
    #print(index)
    symbols = []
    for ind in index:
        symbols.append(ofdm[ind+cp : ind+all_sym])
        
    return np.asarray(symbols)

def generate_pilot_carriers(N_fft, GB_len, N_pil):
    """
    Generates indices representing pilot subcarriers.

    Args:
        N_pilot (int): Number of pilot subcarriers.

    Returns:
        np.ndarray: Array of pilot subcarrier indices within the usable bandwidth.
    """
    usable_bandwidth = N_fft - GB_len
    pilot_spacing = int(usable_bandwidth / (N_pil - 1))  # Spacing between pilots
    #ic(usable_bandwidth,pilot_spacing)
    # Можно менять значение от 0 до 1
    #                          ↓
    pilot_carriers = np.arange(0 + GB_len//2, N_fft - GB_len//2, pilot_spacing)
    #pilot_carriers = np.linspace(0 + self.GB_len//2, self.N_fft - self.GB_len//2+1, N_pil)

    for i in range(len(pilot_carriers)):
        if pilot_carriers[i] == 32:
            pilot_carriers[i] += 1
        
    # Handle potential rounding errors or edge cases
    if len(pilot_carriers) < N_pil:
        pilot_carriers = np.concatenate((pilot_carriers, [N_fft // 2 + 1]))  # Add center carrier if needed
    elif len(pilot_carriers) > N_pil:
        pilot_carriers = pilot_carriers[:N_pil]  # Truncate if there are too many
    
    pilot_carriers[-1] = N_fft - GB_len//2 - 2 # Последний пилот на последней доступной поднесущей
    
    return pilot_carriers


def OFDM_MOD(N_fft, GB_len, N_pil, QAM , CP, ampl = 2**14): 
    """
        В разработке!
    """

    def add_pss(fft_len, symbols, amplitude): 
        """
        Добавление PSS 
        
        Работает правильно
        """
        #len_subcarr = len(self.activ_carriers(True))
        
        pss = zadoff_chu(PSS=True) * amplitude
        arr = np.zeros(fft_len, dtype=complex)

        # Массив с защитными поднесущими и 0 в центре
        arr[fft_len//2 - 31 : fft_len//2] = pss[:31]
        arr[fft_len//2 + 1: fft_len//2 + 32] = pss[31:]
        
        symbols = np.insert(symbols, 0, arr, axis=0)
        
        for i in range(6, symbols.shape[0], 6):
            symbols = np.insert(symbols, i, arr, axis=0)

        return symbols
    
    pilot = complex(0.7,0.7) 

    len_data = N_fft-N_pil-GB_len

    pilot_carrier = generate_pilot_carriers(N_fft, GB_len, N_pil)

    data_carrier = activ_carriers(N_fft, GB_len, pilot_carrier)

    count_ofdm_symbol = len(QAM) // len(data_carrier) + 1 
    ofdm = np.zeros((count_ofdm_symbol, N_fft),dtype=np.complex128)
    pss = add_pss(N_fft,CP)*2
    
    ofdm_ifft_cp = np.zeros(0)
    for j in range(count_ofdm_symbol):
            if j % 6 != 0:
                if len_data == len(QAM[j * len_data :(j+1)*len_data]):
                    ofdm[j][pilot_carrier] = pilot
                    ofdm[j][data_carrier] = QAM[j * len_data :(j+1)*len_data]
                    ifft_ofdm = np.fft.ifft((np.fft.fftshift(ofdm[j])),N_fft)
                    ofdm_ifft_cp = np.concatenate([ofdm_ifft_cp, ifft_ofdm[-CP:], ifft_ofdm])
                else:
                    data_carrier1 = data_carrier[:len(QAM[j * len_data :(j+1)*len_data])]
                    ofdm[j][pilot_carrier] = pilot
                    ofdm[j][data_carrier1] = QAM[j * len_data :(j+1)*len_data]
                    ifft_ofdm = np.fft.ifft((np.fft.fftshift(ofdm[j])),N_fft)
                    ofdm_ifft_cp = np.concatenate([ofdm_ifft_cp, ifft_ofdm[-CP:], ifft_ofdm])
            else:
                ofdm_ifft_cp = np.concatenate([ofdm_ifft_cp, pss])
    
    
    #ofdm_ifft_cp_pss = np.concatenate([pss, ofdm_ifft_cp]) 
    ofdm_ifft_cp *= ampl
    return ofdm_ifft_cp


def fft(num_carrier,GB_len, ofdm_symbols, pilot_carrier, ravel = True, GB = False, pilots = True):
    fft = []
    len_c = np.shape(ofdm_symbols)[0]
    for i in range(len_c):
        if len_c == 1:
            zn = np.fft.fftshift(np.fft.fft(ofdm_symbols))
        else:
            zn = np.fft.fftshift(np.fft.fft(ofdm_symbols[i]))
            
        if (GB is False) and (pilots is False):
            zn = zn[activ_carriers(num_carrier, GB_len, pilot_carrier, pilots = False)]
        elif (GB is True):
            pass
        else:
            zn = zn[activ_carriers(num_carrier, GB_len, pilot_carrier, pilots = True)]
            
        fft.append(zn)
            
    if ravel:
        ret = np.ravel(fft)
        return ret
    else:
        return fft
    

def modulation(N_fft, CP_len, GB_len, QAM_sym, N_pilot, amplitude_all=2**14, amplitude_data=1, amplitude_pss=3, amplitude_pilots=3, ravel=True):

    """
    OFDM модуляция.

    Returns:
        np.ndarray: Массив OFDM-сигналов.
    """
    # Разделение массива symbols на матрицу(по n в строке)
    def reshape_symbols(symbols, activ):
        len_arr = len(activ)
        try:
            if (len(symbols) % len_arr) != 0:
                symbols1 = np.array_split(symbols[: -(len(symbols) % len_arr)], len(symbols) / len_arr)
                symbols2 = np.array((symbols[-(len(symbols) % len_arr) :]))
                
                ran_bits = randomDataGenerator((len_arr - len(symbols2))*2)
                zero_qpsk = list(QPSK(ran_bits, amplitude=1)) # 1+1j 
                
                zeros_last = np.array(zero_qpsk)
                symbols2 = np.concatenate((symbols2, zeros_last))
                symbols1.append(symbols2)
                symbols = symbols1
            else:
                symbols = np.array_split(symbols, len(symbols) / len_arr)
        except ValueError:
            zero = np.zeros(len_arr - len(symbols))
            symbols = np.concatenate((symbols, zero))
        
        return symbols

    def generate_pilot_symbol(n_pil):
        pilot = list(QPSK([0,0], amplitude=1)) 
        pilot_symbols = pilot * n_pil
        return np.array(pilot_symbols)


    def distrib_subcarriers(symbols, activ, fft_len, amplitude):
        len_symbols = np.shape(symbols)
        # Создание матрицы, в строчке по n символов QPSK
        if len(len_symbols) > 1: 
            arr_symols = np.zeros((len_symbols[0], fft_len), dtype=complex)
        else: # если данных только 1 OFDM символ
            arr_symols = np.zeros((1, fft_len), dtype=complex)
        
        # Распределение строк символов по OFDM символам(с GB и пилотами)
        pilot_carriers = generate_pilot_carriers(N_fft, GB_len, N_pilot)
        pilot_symbols = generate_pilot_symbol(N_pilot)
        for i, symbol in enumerate(arr_symols):
            index_pilot = 0
            index_sym = 0
            for j in range(len(symbol)):
                if j in pilot_carriers:
                    arr_symols[i][j] = pilot_symbols[index_pilot] * amplitude
                    index_pilot += 1
                elif (j in activ) and (index_sym < len_symbols[-1]):
                    if len(len_symbols) > 1:
                        arr_symols[i][j] = symbols[i][index_sym]
                    else:
                        arr_symols[i][j] = symbols[index_sym]
                    index_sym += 1
        
        return arr_symols

    def split_into_slots(symbols, chunk_size):
        chunks = []
        sym = list(symbols)
        # Разбивает `symbols` на фрагменты по `chunk_size`
        for i in range(0, len(sym), chunk_size):
            chunks.append(sym[i:i + chunk_size])
        return chunks

    def to_binary_fixed_length(number, length=8):
        binary_array = []
        for i in range(length):
            bit = number & (1 << (length - i - 1))
            binary_array.append(bit >> (length - i - 1))
        return binary_array

    def add_pss(fft_len, symbols, amplitude): 
        """
        Добавление PSS 
        
        Работает правильно
        """
        #len_subcarr = len(self.activ_carriers(True))
        
        pss = zadoff_chu(PSS=True) * amplitude
        arr = np.zeros(fft_len, dtype=complex)

        # Массив с защитными поднесущими и 0 в центре
        arr[fft_len//2 - 31 : fft_len//2] = pss[:31]
        arr[fft_len//2 + 1: fft_len//2 + 32] = pss[31:]
        
        symbols = np.insert(symbols, 0, arr, axis=0)

        len_sym = len(symbols)
        pss = 6
        while pss < len_sym:
            symbols = np.insert(symbols, pss, arr, axis=0)
            len_sym+=1
            pss+=6
            

        return symbols

    def add_CRC(slot_pre_post):

        data = dem_qpsk(slot_pre_post) # Демодуляция по QPSK
        #print(data)
        data = list(data)
        G = [1,0,1,0,0,1,1,1,0,1,0,0,0,1,0,1,1] # полином для вычисления crc

        data_crc = data + 16 * [0]
        for i in range(0,len(data_crc)-16):
            if(data_crc[i] == 1):
                for j in range(len(G)):
                    data_crc[i+j] = data_crc[i+j] ^ G[j]
        crc = data_crc[len(data_crc)-16:]

        return np.array(crc)

    fft_len = N_fft
    _cyclic_prefix_len = CP_len
    _guard_band_len = GB_len
    symbols = QAM_sym
    pilot_carrier = generate_pilot_carriers(N_fft, GB_len, N_pilot)
    activ = activ_carriers(N_fft, GB_len, pilot_carrier, pilots = False)

    len_prefix_max_slots = int(np.log2(1024))

    # Нормирование амплитуд
    am_max = np.max([amplitude_data, amplitude_pilots, amplitude_pss])
    amplitude_data = amplitude_data / am_max
    amplitude_pilots = amplitude_pilots / am_max
    amplitude_pss = amplitude_pss / am_max
    #print("pp",len(activ))

    symbols = split_into_slots(symbols, (-len_prefix_max_slots +(len(activ))*5)  -13)
    # Делим массив символов на матрицу (в строке элеметнов = доступных поднесущих)
    slots = []
    for slot, i in zip(symbols, range(len(symbols))):
        # Заполнение префикса
        slot_number = QPSK(to_binary_fixed_length(i+1, len_prefix_max_slots), amplitude=1)
        total_slots = QPSK(to_binary_fixed_length(len(symbols), len_prefix_max_slots), amplitude=1)
        useful_bits = QPSK(to_binary_fixed_length(len(slot)+23, 10), amplitude=1)

        slot_pre_post  = np.concatenate((slot_number, total_slots, useful_bits, slot))
        # CRC
        #print(slot_pre_post)
        crc = QPSK(add_CRC(slot_pre_post), amplitude=1) 
        
        slot_pre_post  = np.concatenate((slot_pre_post, crc))
        
        
        slot_pre_post = reshape_symbols(slot_pre_post, activ) 

        ran_bits = randomDataGenerator(len(activ)*2)
        zero_qpsk = list(QPSK(ran_bits, amplitude=1)) # 1+1j 
        
        # Добавление недостающих OFDM символов для заполнения слота
        empty_symbol = []
        for em in range(0, 5 - np.shape(slot_pre_post)[0]):
            empty_symbol.append(zero_qpsk)
        if len(empty_symbol) > 0:
            slot_pre_post = np.concatenate((slot_pre_post, empty_symbol))
        
        #ic(np.shape(slot_pre_post))
        slots.append(slot_pre_post)
    
    slots = np.concatenate(slots, axis=0)
    
    #print("len slots",len(slot))

    slots = slots * amplitude_data
    arr_symols = distrib_subcarriers(slots, activ, fft_len, amplitude_pilots)
    arr_symols = add_pss(fft_len, arr_symols, amplitude_pss)
    
    arr_symols = np.fft.fftshift(arr_symols, axes=1)
    #print("count pss",len(arr_symols), len(arr_symols[0]))
    # IFFT
    ifft = np.zeros((np.shape(arr_symols)[0], fft_len), dtype=complex)
    for i in range(len(arr_symols)):
        ifft[i] = np.fft.ifft(arr_symols[i])
    
    # Добавление циклического префикса
    fft_cp = np.zeros((np.shape(arr_symols)[0], (fft_len + _cyclic_prefix_len)), dtype=complex)
    for i in range(np.shape(arr_symols)[0]):
        fft_cp[i] = np.concatenate((ifft[i][-_cyclic_prefix_len:], ifft[i]))
    
    fft_cp = fft_cp * amplitude_all

    if ravel:
        return np.ravel(fft_cp)
    return fft_cp


def del_pss_in_frame(frame, Nfft, cp):
    frame = frame.reshape(len(frame)//(Nfft+cp), (Nfft+cp))
    #indices_to_remove = [5, 11, 17, 23, 29,35]
    #frame = [row for idx, row in enumerate(frame) if idx not in indices_to_remove]
    new_matrix = np.delete(frame, np.arange(5, len(frame), 6), axis=0)

    return np.asarray(new_matrix).flatten()


def del_pilot(fft_rx_inter,num_carrier,GB_len, data_not_pilot):
    
    #print("----",ofdm_mod.pilot_carriers)

    #print("not_pilot1 ",data_not_pilot)
    for i in range(len(data_not_pilot)):
        if data_not_pilot[i] > num_carrier//2:
            data_not_pilot[i] -=1

    data_not_pilot = data_not_pilot - GB_len//2 
    
    #print("not_pilot2 ",data_not_pilot)  

    ofdm2 = fft_rx_inter.reshape(len(fft_rx_inter)//(num_carrier-GB_len-1),num_carrier-GB_len-1)

    data = np.zeros(0)
    for i in range(len(ofdm2)):
            #print(i)
            qpsk = ofdm2[i][data_not_pilot]
            data = np.concatenate([data, qpsk])

    return data

def converted_bits(rx_bit):
    rx_bit = list(rx_bit)
    binary_str = ''.join(map(str, rx_bit))
    decimal_value = int(binary_str, 2)
   
    return decimal_value


def interpolatin_pilot(pilot_carrier, rx_sync,GB_len):
    rx = np.asarray(rx_sync)
    #Hls = rx[0][index_pilot]
       
    rx_pilot = np.array([np.take(row, pilot_carrier) for row in rx])
    #print("pilot int1",pilot_carrier)
    count_ofdm = len(rx_sync)
    num_carrier = len(rx_sync[0])
    
    pilot = complex(1,1) 
    Hls = rx_pilot / pilot                                                    # частотная характеристика канала на пилотах

    Hls1 = Hls.flatten()

    if 0:                                                                    
        plt.figure(7)
        plt.title("Частотная характеристика канала на пилотах")
        plt.stem(abs(Hls1), "r",label='pilot - ampl')
        plt.stem(np.angle(Hls1),label='pilot - phase')
        plt.legend(loc='upper right')

   # pilot_carrier = pilot_carrier-GB_len//2                                   # индексы пилотов без защитных нулей
   # for i in range(len(pilot_carrier)):
        #if pilot_carrier[i] >  num_carrier//2:
           # pilot_carrier[i] -= 1
    #print("pppp",pilot_carrier)
    #print(Hls)

    all_inter = np.zeros(0)

    ### Интерполяция ###
    for i in  range(count_ofdm):                                               # цикл по количеству ofdm символов
        x_interp = np.linspace(pilot_carrier[0], pilot_carrier[-1], num_carrier)#np.linspace(0, num_carrier, num_carrier)

        #print("pilot val = ", Hls[i])
        interpol = np.interp(x_interp, pilot_carrier, Hls[i])
        #print("interpol = ", interpol)

        
        all_inter = np.concatenate([all_inter, interpol])


    #print(all_inter)
    #interpol = y_interp
    #print("len inter",len(interpol))
    #interpol = interpol.flatten()
    if 0:
        plt.figure(8)
        plt.title('Интерполяция')
        plt.stem(abs(all_inter))
    return all_inter

def interpol(fft_rx1, num_carrier, GB_len, pilot_carrier):
    

    for i in range(len(pilot_carrier)):
        if pilot_carrier[i] > num_carrier//2:
            pilot_carrier[i] -= 1

    pilot_carrier -= GB_len//2  
   
    ofdm1 = fft_rx1.reshape(len(fft_rx1)//(num_carrier-GB_len-1),(num_carrier-GB_len-1))

    fft_rx_inter = interpolatin_pilot(pilot_carrier, ofdm1, GB_len)



    data_pilot = np.zeros(0)
    data_carrier1 = activ_carriers(num_carrier, GB_len, pilot_carrier, pilots = True)
    #print("00000",data_carrier1)
    for i in range(len(data_carrier1)):
        if data_carrier1[i] > num_carrier//2:
            data_carrier1[i] -=1
    data_carrier1 -= GB_len//2

    #print("1111",data_carrier1 , len(data_carrier1))

    for i in range(len(ofdm1)):
            #print(i)
            qpsk = ofdm1[i][data_carrier1]
            data_pilot = np.concatenate([data_pilot, qpsk])

    return  data_pilot / fft_rx_inter



def get_inform_slot(rx_sig, Nfft, N_pilot, GB_len,mode , cp):
    """
        Получение информации из слотов

        Параметры
        ---------
            `rx_sig`  - два слота  (для корректной работы символьной синхронизации)
            `Nfft`    - кол-во поднесущих
            `N_pilot` - кол-во пилотов
            `GB_len`  - защитный интервал
            `mode`    - режим работы 
                mode = 1 :  возвращает номер слота и битовую информацию в этом слоте (используется для формирования исходной битовой последовательности)
                mode = 2 :  возвращает кол-во слотов (сколько слотов занимает сообщение)
            `cp`      - циклический префикс 
            
    """
    slot_ofdm = rx_sig


    pilot_carrier = generate_pilot_carriers(Nfft, GB_len, N_pilot)
    data_not_pilot = activ_carriers(Nfft, GB_len, pilot_carrier, pilots = False)
    rx_synс = indiv_symbols(slot_ofdm, Nfft, cp)

    pilot_carrier1 = pilot_carrier.copy()
    data_not_pilot1 = data_not_pilot.copy()

    fft_rx = fft(Nfft, GB_len, rx_synс, pilot_carrier1)

    interpolate = interpol(fft_rx, Nfft, GB_len,pilot_carrier1)

    qpsk = del_pilot(interpolate, Nfft, GB_len, data_not_pilot1)
    #plot_QAM(qpsk)
    #print("EVM = ",EVM_qpsk(qpsk), " dB")
    bits_with_prefix = DeQPSK(qpsk)
    
    number_slot = bits_with_prefix[:8]
    count_slot = bits_with_prefix[8:16]
    good_inf = bits_with_prefix[16:24]
    print(count_slot)
    
    number_slot = converted_bits(number_slot)
    count_slot = converted_bits(count_slot)
    good_inf = converted_bits(good_inf)


    if mode == 1:
        return int(number_slot), bits_with_prefix
    if mode == 2:
        return int(count_slot)

def decode_slots(rx_sig, Nfft, cp, GB_len, N_slots,N_pilot):
    """
        Получение информации из слотов

        Параметры
        ---------
            `rx_sig`  - все слоты без pss (для корректной работы символьной синхронизации)
            `Nfft`    - кол-во поднесущих
            `cp`      - циклический префикс 
            `GB_len`  - защитный интервал
            `N_slots` - кол-во слотов
            `N_pilot` - кол-во пилотов
            
        Возвращает: 
            `slots`   - биты содержащие сообщение
    """        
    num_slot_slot = {}
    j = 0
    for i in range(N_slots):
        num_slot, bits_with_prefix = get_inform_slot(rx_sig[(800) * i :((800)*(i+1))+800], Nfft, N_pilot, GB_len, 1,cp)

        num_slot_slot[num_slot] = bits_with_prefix

    sorted_keys = sorted(num_slot_slot.keys())

    sorted_slots = np.zeros(0)
    for key in sorted_keys:
        sorted_slots = np.concatenate([sorted_slots, num_slot_slot[key]])
    sorted_slots = sorted_slots.astype(int)

    slot_matrix = sorted_slots.reshape(len(sorted_slots)//660, 660)
    len_slot = 660
    slot_matrix = slot_matrix[:, 24:(len_slot-16)]
    slots = slot_matrix.flatten()
    return slots


broken_slot = []
lenslot = 0

def get_inform_slot_bit10(rx_sig, Nfft, N_pilot, GB_len,mode , cp):
    """
        Получение информации из слотов

        Параметры
        ---------
            `rx_sig`  - два слота  (для корректной работы символьной синхронизации)
            `Nfft`    - кол-во поднесущих
            `N_pilot` - кол-во пилотов
            `GB_len`  - защитный интервал
            `mode`    - режим работы 
                mode = 1 :  возвращает номер слота и битовую информацию в этом слоте (используется для формирования исходной битовой последовательности)
                mode = 2 :  возвращает кол-во слотов (сколько слотов занимает сообщение)
            `cp`      - циклический префикс 
            
    """
    slot_ofdm = rx_sig


    pilot_carrier = generate_pilot_carriers(Nfft, GB_len, N_pilot)
    data_not_pilot = activ_carriers(Nfft, GB_len, pilot_carrier, pilots = False)
    rx_synс = indiv_symbols(slot_ofdm, Nfft, cp)

    pilot_carrier1 = pilot_carrier.copy()
    data_not_pilot1 = data_not_pilot.copy()

    fft_rx = fft(Nfft, GB_len, rx_synс, pilot_carrier1)
    # plot_QAM(fft_rx)
    interpolate = interpol(fft_rx, Nfft, GB_len,pilot_carrier1)

    qpsk = del_pilot(interpolate, Nfft, GB_len, data_not_pilot1)
    # plot_QAM(qpsk)
    #print("EVM = ",EVM_qpsk(qpsk), " dB")
    #print(qpsk[-8:])
    bits_with_prefix = DeQPSK(qpsk)

    number_slot = bits_with_prefix[:10]
    count_slot = bits_with_prefix[10:20]
    good_inf = bits_with_prefix[20:30]
    #crc = bits_with_prefix[-16:]
    #print(good_inf)
    #print(number_slot)
    #print(count_slot)
    
    number_slot = converted_bits(number_slot)
    count_slot = converted_bits(count_slot)
    good_inf = converted_bits(good_inf)

    # if (number_slot == 15):
    #     evm = "EVM = " + str(EVM_qpsk(qpsk)) + " dB"
    #     plot_QAM(qpsk, evm)

    if mode == 1:
        global lenslot
        lenslot += (good_inf*2) - (10+10+10+16)
        bit_for_crc = bits_with_prefix.copy()
        bit_for_crc = bit_for_crc[:good_inf*2]
        chek_crc =  CRC_RX(bit_for_crc)

        broken_slot.append(ACK(chek_crc, number_slot))
        return int(number_slot), bits_with_prefix
    if mode == 2:
        return int(count_slot)


def decode_slots_bit10(rx_sig, Nfft, cp, GB_len, N_slots,N_pilot):
    """
        Получение информации из слотов

        Параметры
        ---------
            `rx_sig`  - все слоты без pss (для корректной работы символьной синхронизации)
            `Nfft`    - кол-во поднесущих
            `cp`      - циклический префикс 
            `GB_len`  - защитный интервал
            `N_slots` - кол-во слотов
            `N_pilot` - кол-во пилотов
            
        Возвращает: 
            `slots`   - биты содержащие сообщение
    """        
    num_slot_slot = {}
    j = 0
    for i in range(N_slots):
        num_slot, bits_with_prefix = get_inform_slot_bit10(rx_sig[(800) * i :((800)*(i+1))+800], Nfft, N_pilot, GB_len, 1,cp)

        num_slot_slot[num_slot] = bits_with_prefix

    sorted_keys = sorted(num_slot_slot.keys())

    sorted_slots = np.zeros(0)
    for key in sorted_keys:
        sorted_slots = np.concatenate([sorted_slots, num_slot_slot[key]])
    sorted_slots = sorted_slots.astype(int)
    len_slot = 660
    slot_matrix = sorted_slots.reshape(len(sorted_slots)//len_slot, len_slot)
    
    slot_matrix = slot_matrix[:, 30:(len_slot-16)]
    #print("len slot",len(slot_matrix[0]))
    slots = slot_matrix.flatten()

    slots = slots[:lenslot]
    
    ### придумать как лучше вынести в функцию
    result = []  
    #print(broken_slot, len(broken_slot))
    for element in broken_slot:
       
        if element > 0 and element not in result:
            result.append(element)
    ###

    return slots, result