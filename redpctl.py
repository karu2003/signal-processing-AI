import redpitaya_scpi as scpi
import numpy as np
import time

class RedCtl:
    """RedPitaya ctrl class
    parameters:
    """
    def __init__(self, ip='192.168.0.15', trig = 0.0 ):

        self.data = []
        self.ip = ip
        self.rp_s = scpi.scpi(self.ip)
        self.trig_lev = trig

        self.rp_s.tx_txt('ACQ:RST')

        self.rp_s.tx_txt('ACQ:DATA:FORMAT ASCII')
        self.rp_s.tx_txt('ACQ:DATA:UNITS VOLTS')
        self.rp_s.tx_txt('ACQ:DEC 1')
        self.rp_s.tx_txt('ACQ:TRIG:DLY 0')
        self.rp_s.tx_txt('ACQ:TRIG:LEV ' + str(self.trig_lev))

    def read(self, quantity = 800, counter = 50):

        self.data.clear()

        self.rp_s.tx_txt('ACQ:START')
        time.sleep(0.05)
        self.rp_s.tx_txt('ACQ:TRIG CH1_PE')

        for i in range(counter):

            while 1:
                self.rp_s.tx_txt('ACQ:TRIG:STAT?')
                if self.rp_s.rx_txt() == 'TD':
                    break

            while 1:
                self.rp_s.tx_txt('ACQ:TRIG:FILL?')
                if self.rp_s.rx_txt() == '1':
                    break

            self.rp_s.tx_txt('ACQ:SOUR1:DATA:OLD:N? '+ str(quantity))

            buff_string = self.rp_s.rx_txt()
            buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
            buff = list(map(float, buff_string))
            arr = np.array(buff)
            self.data.append(arr)

            # data = np.append(data,np.array(buff))

        return self.data

    def set_gen(self, wave_form = "square", freq = 500000, ampl = 0.5):
        # wave_form "sine" "square"
        self.rp_s.tx_txt('GEN:RST')
        self.rp_s.sour_set(1, wave_form, ampl, freq)
        self.rp_s.tx_txt('OUTPUT1:STATE ON')
        self.rp_s.tx_txt('SOUR1:TRIG:INT')
        # self.rp_s.close()
 