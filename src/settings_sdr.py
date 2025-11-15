import adi
import numpy as np


def standart_settings(Pluto_IP = "192.168.2.1", sample_rate = 1e6, buffer_size = 1e3, gain_mode = 'manual'):
    """
    Базовые настройки sdr
    
    Параметры
    ----------
        `ip` : "192.168.3.1" / "192.168.2.1"
        
        `buffer_size` = 1e3 [samples]
            до 16_770_000
        
        `sample_rate` = 1e6 [samples]
            от 521_000 до 61_440_000
        
        `mode` : str, optional
            slow_attack, fast_attack, manual
            
    Возвращает
    ----------  
        `sdr`: настроенный класс "sdr"
    """    
    sdr = adi.Pluto(Pluto_IP)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    sdr.gain_control_mode_chan0 = gain_mode

    return sdr


def rx_signal(sdr, rx_lo, gain_rx, cycle):

    """
        Параметры:

        `sdr` : настроенный класс sdr

        `rx_lo` : несущая частота (на какой частоте принимаем)
        от 325 [МГц] до 3.8 [ГГц] | 
        hacked 70 [МГц] до 6 [ГГц]

        `rx_gain`: чувствительность приёма [dBm]
        рекомендуемое значение от 0 до 74,5 [дБ]

        `cycle` : кол-во принимаемых буферов

        Возвращает
    
        `data`: принятые данные

    """

    sdr.rx_lo = int(rx_lo)
    sdr.rx_hardwaregain_chan0 = gain_rx
    data = np.zeros(0)
    for i in range(cycle):
        rx = sdr.rx()
        data = np.concatenate([data,rx])
        
    sdr.tx_destroy_buffer()

    return data
    
    
    

def tx_signal(sdr, tx_lo, gain_tx, data, tx_cycle: bool = True):
    """
        Параметры:

        `sdr` : настроенный класс sdr

        `tx_lo` : несущая частота (на какой частоте отправляем)
        от 325 [МГц] до 3.8 [ГГц] | 
        hacked 70 [МГц] до 6 [ГГц]

        `tx_gain`: сила передачи [dBm]
        рекомендуемое значение от -90 до 0 [дБ]

        `data` : данные на передачу

        `tx_cycle`: циклическое отправление
    """

    sdr.tx_lo = int(tx_lo)
    sdr.tx_hardwaregain_chan0 = gain_tx
    
    sdr.tx_cyclic_buffer = tx_cycle
    sdr.tx(data)
    #sdr.tx_destroy_buffer()