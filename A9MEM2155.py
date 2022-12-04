# Parametry miernika A9MEM2155
# BaudRate = 9600, Parity = None, Stopbits = 1, Device ID=1

import time
import sqlite3
from pymodbus.client.sync import ModbusSerialClient as ModbusClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder as Decode


def data_logger(today_date, current, voltage, power_w, power_va, power_var, power_factor, freq, energy_w, energy_varh):
    try:
        conn = sqlite3.connect('measurement.db')
        curs = conn.cursor()
        print("Connected to SQLite")
        sqlite_insert_with_param = """INSERT INTO reading
                          (date, current, voltage, active_power, apparent_power, reactive_power, power_factor, 
                          frequency, total_active_energy, 
                          total_reactive_energy) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
        data_tuple = (
            today_date, current, voltage, power_w, power_va, power_var, power_factor, freq, energy_w, energy_varh)
        curs.execute(sqlite_insert_with_param, data_tuple)
        conn.commit()
        print("Variables inserted successfully")
        curs.close()
    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if conn:
            conn.close()
            print("The SQLite connection is closed")



def main():
    # A9MEM2155 meter defined as client
    client = ModbusClient(method='rtu', port='/dev/ttyUSB0', timeout=1, baudrate=9600, parity='N', stopbits=1,
                          bytesize=8)
    client.connect()
    
    # Current Values
    a = client.read_holding_registers(2999, 2, unit=1)
    
    a_d = Decode.fromRegisters(a.registers, Endian.Big)
    a_d = {'float': a_d.decode_32bit_float(), }
    
    #####################################################
    # Voltage Values
    
    vln = client.read_holding_registers(3027, 2, unit=1)
    
    vln_d = Decode.fromRegisters(vln.registers, Endian.Big)
    vln_d = {'float': vln_d.decode_32bit_float(), }
    
    ######################################################
    # Power Values
    
    w = client.read_holding_registers(3053, 2, unit=1)
    
    va = client.read_holding_registers(3075, 2, unit=1)
    
    var = client.read_holding_registers(3067, 2, unit=1)
    
    w_d = Decode.fromRegisters(w.registers, Endian.Big)
    w_d = {'float': w_d.decode_32bit_float(), }
    
    va_d = Decode.fromRegisters(va.registers, Endian.Big)
    va_d = {'float': va_d.decode_32bit_float(), }
    
    var_d = Decode.fromRegisters(var.registers, Endian.Big)
    var_d = {'float': var_d.decode_32bit_float(), }
    
    ######################################################
    # Power Factor Values
    pf = client.read_holding_registers(3083, 2, unit=1)
    
    pf_d = Decode.fromRegisters(pf.registers, Endian.Big)
    pf_d = {'float': pf_d.decode_32bit_float(), }
    
    ######################################################
    # Frequency Values
    f = client.read_holding_registers(3109, 2, unit=1)
    f_d = Decode.fromRegisters(f.registers, Endian.Big)
    f_d = {'float': f_d.decode_32bit_float(), }
    ######################################################
    # Energy Values
    varh = client.read_holding_registers(45103, 2, unit=1)
    wh = client.read_holding_registers(45099, 2, unit=1)
    varh_d = Decode.fromRegisters(varh.registers, Endian.Big)
    varh_d = {'float': varh_d.decode_32bit_float(), }
    wh_d = Decode.fromRegisters(wh.registers, Endian.Big)
    wh_d = {'float': wh_d.decode_32bit_float(), }
    ######################################################

    timestamp = time.strftime('%H:%M %d-%m-%Y')

#Current
    
    for i, value in a_d.items():
        a = value
    
#Voltage
    
    for i, value in vln_d.items():
        vln = value
    
#Power factor
    
    for i, value in pf_d.items():
        pf = value

#Frequency
    
    for i, value in f_d.items():
        f = value

#Power parameters
    
    for i, value in w_d.items():
        w = value
    
    for i, value in va_d.items():
        va = value
        
    for i, value in var_d.items():
        var = value

#Energy parameters
    
    for i, value in varh_d.items():
        varh = value
    
    for i, value in wh_d.items():
        wh = value

    data_logger(timestamp, a, vln, w, va, var, pf, f, wh, varh)

    client.close()


if __name__ == "__main__":
    main()
