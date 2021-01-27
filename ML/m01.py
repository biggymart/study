import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # 0.1 단위로 100개 생성
y = np.sin(x) # numpy provides sin function

plt.plot(x, y)
plt.show()
