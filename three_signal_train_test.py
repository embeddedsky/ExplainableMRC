import numpy as np
from matplotlib import pyplot as plt
import math
from decimal import Decimal
T=0.5
POINT=300

def u_chazhi(t,t_point,u_point):
    u=0
    for i in range(0,POINT-1):
        if t>=t_point[i] and t<=t_point[i+1]:
            u=u_point[0][i]+(t-t_point[i])*(u_point[0][i+1]-u_point[0][i])/(t_point[i+1]-t_point[i])
            return u

    print("a")

def u1_cal(t,shift,ratio,y1,y0,period):
    if float((t-shift)%period)<=period*ratio:
        return y1
    else:
        return y0
def NARMA_Diverges(t_point,u_point):
    t2 = np.linspace(start=0, stop=T, num=1800)

    y=[0.1,0.1]
    for i in range(2, 1800):
        u = u_chazhi(t2[i],t_point,u_point)
        t=0.4 * y[i - 1] + 0.4 * y[i - 1] * y[i - 2] + 0.6 * pow(u, 3) + 0.1
        if t > 1:
            return True
        y.append(t)
    return False
def create_signal():
    t_point = np.linspace(start=0, stop=T, num=POINT)
    u_point = np.random.uniform(low=0.0, high=0.5, size=(1, POINT))

    while (NARMA_Diverges(t_point,u_point)):
        u_point=np.random.uniform(low=0.0, high=0.5, size=(1, POINT))

    t2 = np.linspace(start=0, stop=T, num=1800)

    y=[0.1,0.1]
    for i in range(2,1800):
        u=u_chazhi(t2[i],t_point,u_point)
        y.append( 0.4 * y[i - 1] + 0.4 * y[i - 1] * y[i - 2] + 0.6 * pow(u, 3) + 0.1)

    plt.figure(1)
    plt.plot(t2,y)
    plt.show()
    # 训练文件
    # 1
    #5ms 负脉宽2.5ms
    period=5e-3
    ratio=0.5

    u1=[]
    v1=[]


    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u1.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u1.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u1.append(u_chazhi(float(float(period*int(t2[i]/period))+period*ratio),t_point,u_point))
        v1.append(2.5 * u1[i] + 2.5)

    with open('Narma/Train/signal1.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v1[i])+'\n')
    f.close()

    plt.figure(2)
    plt.plot(t2[0:900],v1)
    plt.show()

    # 2
    #10ms 脉宽2ms
    period=10e-3
    ratio=0.8

    u2=[]
    v2=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u2.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u2.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u2.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v2.append(2.5 * u2[i] + 2.5)

    with open('Narma/Train/signal2.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v2[i])+'\n')
    f.close()

    plt.figure(3)
    plt.plot(t2[0:900],v2)
    plt.show()
    # 3
    #20ms 脉宽4ms
    period=20e-3
    ratio=0.8

    u3=[]
    v3=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u3.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u3.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u3.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v3.append(2.5 * u3[i] + 2.5)

    with open('Narma/Train/signal3.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v3[i])+'\n')
    f.close()

    plt.figure(4)
    plt.plot(t2[0:900],v3)
    plt.show()

    # 4
    #1ms 脉宽0.8ms
    period=1e-3
    shift=period/4.0
    ratio=0.2

    u4=[]
    v4=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u4.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u4.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u4.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v4.append(2.5 * u4[i] + 2.5)

    with open('Narma/Train/signal4.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v4[i])+'\n')
    f.close()

    plt.figure(5)
    plt.plot(t2[0:900],v4)
    plt.show()

    # 5
    #4ms 脉宽0.8ms
    period=4e-3
    ratio=0.8

    u5=[]
    v5=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u5.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u5.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u5.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v5.append(2.5 * u5[i] + 2.5)

    with open('Narma/Train/signal5.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v5[i])+'\n')
    f.close()

    plt.figure(6)
    plt.plot(t2[0:900],v5)
    plt.show()

    # 6
    #2ms 脉宽1ms
    period=2e-3
    ratio=0.5

    u6=[]
    v6=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u6.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u6.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u6.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v6.append(2.5 * u6[i] + 2.5)

    with open('Narma/Train/signal6.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v6[i])+'\n')
    f.close()

    plt.figure(7)
    plt.plot(t2[0:900],v6)
    plt.show()

    # 7
    #3ms 脉宽0.3ms
    period=3e-3
    ratio=0.9

    u7=[]
    v7=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u7.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u7.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u7.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v7.append(2.5 * u7[i] + 2.5)

    with open('Narma/Train/signal7.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v7[i])+'\n')
    f.close()

    plt.figure(8)
    plt.plot(t2[0:900],v7)
    plt.show()

    # 8
    #6ms 脉宽3ms
    period=6e-3
    ratio=0.5

    u8=[]
    v8=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u8.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u8.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u8.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v8.append(2.5 * u8[i] + 2.5)

    with open('Narma/Train/signal8.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v8[i])+'\n')
    f.close()

    plt.figure(9)
    plt.plot(t2[0:900],v8)
    plt.show()

    # 9
    #8ms 脉宽6.4ms
    period=8e-3
    ratio=0.2

    u9=[]
    v9=[]

    for i in range(0,900):
        if float(t2[i] % period) <= period * ratio:
            u9.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u9.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u9.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v9.append(2.5 * u9[i] + 2.5)

    with open('Narma/Train/signal9.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(v9[i])+'\n')
    f.close()

    plt.figure(10)
    plt.plot(t2[0:900],v9)
    plt.show()


    with open('Narma/Train/output.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(0,900):
            f.writelines(str(t2[i])+' '+str(y[i])+'\n')
    f.close()

    np.save("Narma/Train/ref",np.array(y[0:900]))

    # 训练文件
    # 1
    #5ms 脉宽2.5ms
    period=5e-3
    ratio=0.5

    u1=[]
    v1=[]


    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u1.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period)) == 0:
            u1.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u1.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v1.append(2.5 * u1[i-900] + 2.5)

    with open('Narma/Test/signal1.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i])+' '+str(v1[i-900])+'\n')
    f.close()

    plt.figure(11)
    plt.plot(t2[900:1800],v1)
    plt.show()

    # 2
    #10ms 脉宽2ms
    period=10e-3
    ratio=0.8

    u2=[]
    v2=[]

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u2.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u2.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u2.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v2.append(2.5 * u2[i-900] + 2.5)

    with open('Narma/Test/signal2.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i])+' '+str(v2[i-900])+'\n')
    f.close()

    plt.figure(12)
    plt.plot(t2[900:1800],v2)
    plt.show()
    # 3
    #20ms 脉宽4ms
    period=20e-3
    ratio=0.8

    u3=[]
    v3=[]

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u3.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u3.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u3.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v3.append(2.5 * u3[i-900] + 2.5)

    with open('Narma/Test/signal3.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i])+' '+str(v3[i-900])+'\n')
    f.close()

    plt.figure(13)
    plt.plot(t2[900:1800],v3)
    plt.show()

    # 4
    # 1ms 脉宽0.8ms
    period = 1e-3
    ratio = 0.2

    u4 = []
    v4 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u4.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u4.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u4.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v4.append(2.5 * u4[i-900] + 2.5)

    with open('Narma/Test/signal4.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v4[i-900]) + '\n')
    f.close()

    plt.figure(14)
    plt.plot(t2[900:1800], v4)
    plt.show()

    # 5
    # 4ms 脉宽0.8ms
    period = 4e-3
    ratio = 0.8

    u5 = []
    v5 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u5.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u5.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u5.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v5.append(2.5 * u5[i-900] + 2.5)

    with open('Narma/Test/signal5.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v5[i-900]) + '\n')
    f.close()

    plt.figure(15)
    plt.plot(t2[900:1800], v5)
    plt.show()

    # 6
    # 2ms 脉宽1ms
    period = 2e-3
    ratio = 0.5

    u6 = []
    v6 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u6.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u6.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u6.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v6.append(2.5 * u6[i-900] + 2.5)

    with open('Narma/Test/signal6.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v6[i-900]) + '\n')
    f.close()

    plt.figure(16)
    plt.plot(t2[900:1800], v6)
    plt.show()

    # 7
    # 3ms 脉宽0.3ms
    period = 3e-3
    ratio = 0.9

    u7 = []
    v7 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u7.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u7.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u7.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v7.append(2.5 * u7[i-900] + 2.5)

    with open('Narma/Test/signal7.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v7[i-900]) + '\n')
    f.close()

    plt.figure(17)
    plt.plot(t2[900:1800], v7)
    plt.show()

    # 8
    # 6ms 脉宽3ms
    period = 6e-3
    ratio = 0.5

    u8 = []
    v8 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u8.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u8.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u8.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v8.append(2.5 * u8[i-900] + 2.5)

    with open('Narma/Test/signal8.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v8[i-900]) + '\n')
    f.close()

    plt.figure(18)
    plt.plot(t2[900:1800], v8)
    plt.show()

    # 9
    # 8ms 脉宽6.4ms
    period = 8e-3
    ratio = 0.2

    u9 = []
    v9 = []

    for i in range(900,1800):
        if float(t2[i] % period) <= period * ratio:
            u9.append(0.0)
        elif Decimal(str(t2[i])) % Decimal(str(period))==0:
            u9.append(u_chazhi(float(float(period * math.floor(t2[i] / period-1)) + period * ratio), t_point, u_point))
        else:
            u9.append(u_chazhi(float(float(period * math.floor(t2[i] / period)) + period * ratio), t_point, u_point))
        v9.append(2.5 * u9[i-900] + 2.5)

    with open('Narma/Test/signal9.m', 'w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i]) + ' ' + str(v9[i-900]) + '\n')
    f.close()

    plt.figure(19)
    plt.plot(t2[900:1800], v9)
    plt.show()
    with open('Narma/Test/output.m','w+') as f:
        f.writelines('# name: x\n')
        f.writelines('# type: matrix\n')
        f.writelines('# rows: 900\n')
        f.writelines('# columns: 2\n')
        for i in range(900,1800):
            f.writelines(str(t2[i])+' '+str(y[i])+'\n')
    f.close()

    np.save("Narma/Test/ref",np.array(y[900:1800]))

create_signal()