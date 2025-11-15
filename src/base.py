import random
import numpy as np

def randomDataGenerator(size):
	data = [random.randint(0, 1) for i in range(size)]
	return data

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return np.asarray(list(map(int,bits.zfill(8 * ((len(bits) + 7) // 8)))))


def bits_array_to_text(bits_array):
    bits_string = ''.join([str(bit) for bit in bits_array])
    bits_string = bits_string.replace(" ", "")
    n = int(bits_string, 2)
    text = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode('latin1')
    return text

def norm_corr(x,y):
    #x_normalized = (cp1 - np.mean(cp1)) / np.std(cp1)
    #y_normalized = (cp2 - np.mean(cp2)) / np.std(cp2)

    c_real = np.vdot(x.real, y.real) / (np.linalg.norm(x.real) * np.linalg.norm(y.real))
    c_imag = np.vdot(x.imag, y.imag) / (np.linalg.norm(x.imag) * np.linalg.norm(y.imag))
    
    return c_real+1j*c_imag


def zadoff_chu(N=1, u=29, PSS=False):
    """
    Zadoff-Chu sequence
        N - length
        
        u - root index 25 29 34
        
        PSS [optional] - Primary synchronization signal
            N - 63 
            Len - 62
    """
    if PSS:
        N = 63
        n = np.arange(0, 31)
        ex1 = np.exp(-1j * np.pi * u * n * (n + 1) / N)
        n = np.arange(31, 62)
        ex2 = np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / N)
        return np.concatenate([ex1, ex2])
    else:  
        n = np.arange(0, N)
        return np.exp(-1j * np.pi * u * n * (n + 1) / N)
    

def converted_bits_to_file(rx_bit,path_final_file):
    rx_bit = list(rx_bit)
    binary_string = ''.join(map(str, rx_bit))
    converted_array = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]

    #print(converted_array)

    with open(path_final_file, "wb") as file:
        for binary_data in converted_array:
            file.write(bytes([binary_data]))    

def converted_bits(rx_bit):
    rx_bit = list(rx_bit)
    binary_str = ''.join(map(str, rx_bit))
    decimal_value = int(binary_str, 2)
   
    return decimal_value

def converted_file_to_bits(path_file):
    binary_data_array = []
    with open(path_file, "rb") as file:
        byte = file.read(1)
        while byte:
            binary_data_array.append(int.from_bytes(byte, byteorder="big"))
            byte = file.read(1)

    binary_list = [bin(num)[2:].zfill(8) for num in binary_data_array]  
    binary_string = ' '.join(binary_list)  
    binary_numbers = binary_string.split(' ')
    binary_array = [int(bit) for binary_number in binary_numbers for bit in binary_number]
    
    return np.asarray(binary_array)


def EVM_qpsk(qpsk):
    """
        `qpsk` - символы qpsk
        не демодулирует только считает ошибку
    """
    Co = np.array([1+1j, 1-1j, -1+1j, -1-1j])

    evm_sum = 0

    for i in range(len(qpsk)):
        t = 0
        temp = []
        for j in range(len(Co)):
            
            
            co2 = Co[j].real**2 + Co[j].imag**2

            evn2 =  (Co[j].real - qpsk[i].real)**2 + (Co[j].imag - qpsk[i].imag)**2

            evm = np.sqrt(evn2/co2)
            temp.append(evm)
        t = min(temp)
            
        evm_sum += t

    evm_db = np.abs(20 * np.log10(evm_sum/len(qpsk)))

    return evm_db