import re
import pandas as pd
import numpy as np
import seaborn as sb
import zapimoveis_scraper
from matplotlib import pyplot as plt

zap_imoveis = zapimoveis_scraper.search()
new_list_zap = {}

prices = []
areas = []
for zap_imovel in zap_imoveis:
    prices.append(int(re.sub('[^0-9]', '', zap_imovel.price)))
    areas.append(int(str(zap_imovel.total_area_m2).replace('m2', '')))

new_list_zap['price'] = prices
new_list_zap['area_m2'] = areas

apes = pd.DataFrame(new_list_zap)
type(apes)

plt.plot(apes.area_m2, apes.price, 'o')
plt.show()

sb.pairplot(apes)

x = np.array(apes.area_m2)
y = np.array(apes.price)

pl = np.polyfit(x, y, 1)

ypredito = pl[0] * x + pl[1]
yresiduo = y - ypredito 
SQresiduo = sum(pow(yresiduo, 2))
SQtotal = len(y) * np.var(y)
R2 = 1 - SQresiduo/SQtotal

print(pl)
print(R2)

plt.plot(x, y, 'o')
plt.plot(x, np.polyval(pl, x), 'g--')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

