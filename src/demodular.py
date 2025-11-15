import numpy as np


def DeQPSK(sample):
    """
        Демодуляция qpsk символов по четвертям

        Параметры
        ---------
            `samle`: array
                Символы qpsk

        Возвращает
        ---------
            `bit` : numpy array
            
    """
    bit = []
    for i in range(len(sample)):
        if sample.real[i] > 0 and sample.imag[i] > 0:
            bit.append(0)
            bit.append(0)

        if sample.real[i] < 0 and sample.imag[i] > 0:
            bit.append(1)
            bit.append(0)

        if sample.real[i] > 0 and sample.imag[i] < 0:
            bit.append(0)
            bit.append(1)

        if sample.real[i] < 0 and sample.imag[i] < 0:
            bit.append(1)
            bit.append(1)
        
    return np.asarray(bit)


def dem_qpsk(symbols):
    """
    Демодуляция qpsk символов

    Параметры
    ---------
        `symbols`: array
            Символы qpsk

    Возвращает
    ---------
        `decoded_bits_array` : numpy array
            
    """
        
    def demodulate_qpsk_symbol(symbol):
        # Определяем QPSK символы
        qpsk_symbols = [1+1j, 1-1j, -1+1j, -1-1j]

        # Проверяем, находится ли символ в пределах 0.5 от каждого QPSK символа
        for i, qpsk_symbol in enumerate(qpsk_symbols):
            if abs(symbol - qpsk_symbol) <= 0.5:
                # Возвращаем соответствующий бит
                return np.array([i//2, i%2])

        # Если символ не соответствует ни одному из QPSK символов, возвращаем [0, 0]
        return np.array([0, 0])

    maxi = max(max(symbols.real), max(symbols.imag))
    symbols = symbols / maxi
    symbols = symbols * 1.3334
    
    # from .plots import cool_scatter 
    # cool_scatter(symbols)
    
    decoded_bits_array = np.array([demodulate_qpsk_symbol(sym) for sym in symbols])
    return decoded_bits_array.flatten()
